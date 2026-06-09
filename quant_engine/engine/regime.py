"""
quant_engine/engine/regime.py
══════════════════════════════
Market regime classifier based on OHLCV data.

Regime types:
  trending_up   — ADX > 20, EMA(20) slope positive, price making HH/HL
  trending_down — ADX > 20, EMA(20) slope negative, price making LL/LH
  ranging       — ADX < 20, price oscillating within a horizontal band
  volatile      — sharp ATR expansion (> 2× recent average), regime unclear

Used by:
  - app/planner/pipeline.py (de-ranks incompatible signals before plan is built)
  - app/planner/param_resolver.py (scales SL/TP baseline by regime volatility)
  - persona_responder (explains market context to the user)

All inputs are plain pandas DataFrames. No external dependencies beyond numpy/pandas.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ADX_TREND_THRESHOLD  = 20.0   # ADX above this → trending
_ATR_VOLATILE_MULT    = 2.0    # current ATR > N × recent average → volatile
_LOOKBACK_BARS        = 50     # candles to use for regime detection
_EMA_PERIOD           = 20     # EMA used for slope and trend direction
_HH_HL_LOOKBACK       = 10     # bars to check for higher-high / higher-low


def _adx(df: pd.DataFrame, period: int = 14) -> float | None:
    """Return the most recent ADX value. None if insufficient data."""
    if len(df) < period * 2:
        return None
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    dm_plus  = (high - prev_high).clip(lower=0.0)
    dm_minus = (prev_low - low).clip(lower=0.0)
    # Where +DM and -DM are equal the directional move is ambiguous → zero both
    equal_mask = dm_plus == dm_minus
    dm_plus[equal_mask]  = 0.0
    dm_minus[equal_mask] = 0.0
    # Where +DM <= -DM, zero +DM (and vice versa)
    dm_plus[dm_plus  <= dm_minus] = 0.0
    dm_minus[dm_minus <= dm_plus] = 0.0

    alpha = 1.0 / period
    atr_s  = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    dip_s  = dm_plus.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    dim_s  = dm_minus.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    safe_atr = atr_s.replace(0, np.nan)
    di_plus  = (dip_s / safe_atr * 100).fillna(0)
    di_minus = (dim_s / safe_atr * 100).fillna(0)

    di_sum  = (di_plus + di_minus).replace(0, np.nan)
    dx      = ((di_plus - di_minus).abs() / di_sum * 100).fillna(0)
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    last = adx_val.dropna()
    return float(last.iloc[-1]) if len(last) else None


def _ema_slope(close: pd.Series, period: int = 20) -> float | None:
    """
    Return the slope of EMA(period) as a % change over the last `period//2` bars.
    Positive → uptrend, negative → downtrend, None if insufficient data.
    """
    if len(close) < period + 1:
        return None
    ema_vals = close.ewm(span=period, adjust=False, min_periods=period).mean().dropna()
    if len(ema_vals) < 2:
        return None
    lookback = max(2, period // 2)
    old_val  = float(ema_vals.iloc[max(-lookback - 1, -len(ema_vals))])
    new_val  = float(ema_vals.iloc[-1])
    if old_val <= 0:
        return None
    return (new_val - old_val) / old_val * 100.0


def _is_hh_hl(close: pd.Series, lookback: int = 10) -> bool:
    """True if the last `lookback` bars show higher-highs and higher-lows pattern."""
    if len(close) < lookback * 2:
        return False
    segment = close.iloc[-lookback * 2:].values
    mid     = len(segment) // 2
    first_half  = segment[:mid]
    second_half = segment[mid:]
    return bool(
        second_half.max() > first_half.max()
        and second_half.min() > first_half.min()
    )


def _is_ll_lh(close: pd.Series, lookback: int = 10) -> bool:
    """True if the last `lookback` bars show lower-lows and lower-highs pattern."""
    if len(close) < lookback * 2:
        return False
    segment = close.iloc[-lookback * 2:].values
    mid     = len(segment) // 2
    first_half  = segment[:mid]
    second_half = segment[mid:]
    return bool(
        second_half.min() < first_half.min()
        and second_half.max() < first_half.max()
    )


def _atr_expansion(df: pd.DataFrame, period: int = 14) -> float | None:
    """
    Return current ATR as a multiple of its own recent average.
    Values > 2 indicate a volatility spike.
    """
    if len(df) < period * 2 + 1:
        return None
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(period).mean().dropna()
    if len(atr_series) < 2:
        return None
    current_atr = float(atr_series.iloc[-1])
    baseline    = float(atr_series.iloc[-period - 1:-1].mean())
    if baseline <= 0:
        return None
    return current_atr / baseline


def classify_regime_series(
    ohlcv: pd.DataFrame,
    lookback: int = _LOOKBACK_BARS,
) -> pd.Series:
    """Per-bar regime label, computed CAUSALLY.

    Bar i is classified using ONLY its trailing `lookback` window (bars ≤ i) via
    the exact same `classify_regime` logic — so the per-bar label never leaks
    future data and always matches the snapshot classifier on that window. This
    is the single source of truth for both the Phase-2 regime entry gate and the
    Phase-3 per-regime performance breakdown.

    Returns a Series of regime-type strings ("trending_up" | "trending_down" |
    "ranging" | "volatile") aligned to ohlcv.index. Early bars (insufficient
    history) fall back to "ranging", matching classify_regime's default.
    """
    n = len(ohlcv)
    labels: list[str] = []
    for i in range(n):
        start = max(0, i - lookback + 1)
        window = ohlcv.iloc[start : i + 1]
        labels.append(classify_regime(window, lookback)["type"])
    return pd.Series(labels, index=ohlcv.index, name="REGIME")


def classify_regime(
    ohlcv: pd.DataFrame,
    lookback: int = _LOOKBACK_BARS,
) -> dict:
    """
    Classify the market regime from OHLCV data.

    Parameters
    ----------
    ohlcv    : DataFrame with columns [open, high, low, close, volume]
    lookback : number of recent bars to use for classification

    Returns
    -------
    dict with keys:
        type               — "trending_up" | "trending_down" | "ranging" | "volatile"
        adx                — float or None
        volatility_pct     — 30-bar rolling return std (%), None if insufficient
        trend_strength     — 0.0–1.0 (ADX / 50, capped)
        ema_slope_pct      — EMA(20) slope as % change, None if insufficient
        regime_suitable_for — list of signal family tags the regime favours
    """
    result: dict = {
        "type": "ranging",
        "adx": None,
        "volatility_pct": None,
        "trend_strength": 0.0,
        "ema_slope_pct": None,
        "regime_suitable_for": [],
    }

    if not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < 20:
        logger.debug("regime|insufficient_data|bars=%s", len(ohlcv) if isinstance(ohlcv, pd.DataFrame) else 0)
        return result

    # Use the most recent `lookback` bars for efficiency
    df = ohlcv.tail(lookback).copy()
    close = df["close"].astype(float)

    # ── Volatility expansion check (trumps all — volatile regime first) ────────
    atr_mult = _atr_expansion(df)
    if atr_mult is not None and atr_mult >= _ATR_VOLATILE_MULT:
        vol_pct = float(close.pct_change().rolling(min(30, len(close))).std().iloc[-1]) * 100.0
        result.update({
            "type": "volatile",
            "adx": _adx(df),
            "volatility_pct": round(vol_pct, 4) if not np.isnan(vol_pct) else None,
            "trend_strength": 0.0,
            "ema_slope_pct": _ema_slope(close),
            "regime_suitable_for": ["orb", "vwap_reversal", "volume_breakout", "opening_drive"],
        })
        logger.info("regime|type=volatile|atr_expansion=%.2f", atr_mult)
        return result

    # ── ADX + slope for trend vs ranging classification ────────────────────────
    adx_val    = _adx(df)
    slope      = _ema_slope(close)
    hh_hl      = _is_hh_hl(close, _HH_HL_LOOKBACK)
    ll_lh      = _is_ll_lh(close, _HH_HL_LOOKBACK)
    vol_pct    = float(close.pct_change().rolling(min(30, len(close))).std().iloc[-1]) * 100.0
    trend_str  = round(min(1.0, (adx_val or 0.0) / 50.0), 3)

    result["adx"]           = round(adx_val, 2) if adx_val is not None else None
    result["volatility_pct"] = round(vol_pct, 4) if not np.isnan(vol_pct) else None
    result["trend_strength"] = trend_str
    result["ema_slope_pct"]  = round(slope, 4) if slope is not None else None

    is_trending = (adx_val or 0.0) > _ADX_TREND_THRESHOLD

    if is_trending and (slope or 0.0) > 0 and hh_hl:
        regime_type = "trending_up"
        suitable    = ["momentum", "breakout", "ema_pullback", "relative_strength", "supertrend", "trend"]
    elif is_trending and (slope or 0.0) < 0 and ll_lh:
        regime_type = "trending_down"
        suitable    = ["breakdown", "momentum_bearish", "ema_pullback_bearish", "supertrend"]
    elif is_trending:
        # ADX trending but mixed structural evidence — use slope for direction
        if (slope or 0.0) >= 0:
            regime_type = "trending_up"
            suitable    = ["momentum", "breakout", "ema_pullback", "trend"]
        else:
            regime_type = "trending_down"
            suitable    = ["breakdown", "momentum_bearish", "trend"]
    else:
        regime_type = "ranging"
        suitable    = ["vwap_reversion", "mean_reversion", "range_breakout", "rsi_reversal"]

    result["type"]               = regime_type
    result["regime_suitable_for"] = suitable

    logger.info(
        "regime|type=%s|adx=%.1f|slope=%.2f%%|hh_hl=%s|ll_lh=%s|trend_strength=%.2f",
        regime_type,
        adx_val or 0.0,
        slope or 0.0,
        hh_hl,
        ll_lh,
        trend_str,
    )
    return result
