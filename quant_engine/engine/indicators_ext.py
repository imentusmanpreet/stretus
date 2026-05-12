"""
engine/indicators_ext.py
─────────────────────────
Phase-2 indicator pack. Implements the ~20 indicators the catalog marks as
engine_status="implemented" that aren't in the original core file. Each
function takes a DataFrame (with lowercase OHLCV columns) and returns
either a Series or a tuple of Series so the orchestrator can register
each output as a separate precomputed column.

These are NOT separately exposed in conditions.py; the orchestrator
(`add_extended_indicators`) writes them to the DataFrame under suffixed
column names that match the parser's lookup convention (e.g. STOCH_K_14,
SUPERTREND_10_3 — multi-param indicators use `_` between values, with
floats rendered as `<int>p<frac>` to remain identifier-safe).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .indicators import atr, ema, sma  # reuse core indicator math


# ── helpers ────────────────────────────────────────────────────────────────────


def _hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    return df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)


def _typical_price(df: pd.DataFrame) -> pd.Series:
    h, l, c = _hlc(df)
    return (h + l + c) / 3.0


def _suffix_for_float(v: float) -> str:
    """Identifier-safe rendering for float params (e.g. 3.0 → '3', 2.5 → '2p5')."""
    if v == int(v):
        return str(int(v))
    return str(v).replace(".", "p").replace("-", "neg")


def column_suffix(*params) -> str:
    parts = []
    for p in params:
        if isinstance(p, float):
            parts.append(_suffix_for_float(p))
        else:
            parts.append(str(int(p)))
    return "_".join(parts)


# ── Trend ──────────────────────────────────────────────────────────────────────


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX, +DI, -DI. Returns (adx, di_plus, di_minus)."""
    h, l, c = _hlc(df)
    up_move   = h.diff()
    down_move = -l.diff()
    plus_dm   = pd.Series(np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0), index=df.index)
    minus_dm  = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr_series = atr(df, 1) * 1  # TR via existing helper at period 1 ≈ raw TR
    # The above relies on ATR(1) producing the unsmoothed TR. To be safe we
    # reconstruct TR explicitly:
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)

    alpha = 1.0 / n
    tr_smooth      = tr.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    plus_dm_smooth  = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean()

    di_plus  = 100.0 * (plus_dm_smooth  / tr_smooth.replace(0, np.nan))
    di_minus = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_series = dx.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    return adx_series, di_plus, di_minus


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction). Direction is +1 (bullish) or -1
    (bearish). Implementation: classic ATR-band flip."""
    h, l, c = _hlc(df)
    a = atr(df, period)
    median = (h + l) / 2.0
    upper_band_basic = median + multiplier * a
    lower_band_basic = median - multiplier * a

    n = len(df)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    st    = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(n):
        ub_basic = upper_band_basic.iloc[i]
        lb_basic = lower_band_basic.iloc[i]
        if np.isnan(ub_basic) or np.isnan(lb_basic):
            # Waiting for ATR warm-up. Leave SAR / bands as NaN.
            continue
        # First valid bar (or prev was NaN) — seed the running bands.
        if i == 0 or np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            upper[i] = ub_basic
            lower[i] = lb_basic
            close_i = c.iloc[i]
            st[i] = lower[i] if close_i > upper[i] else upper[i]
            direction[i] = 1 if st[i] == lower[i] else -1
            continue
        prev_close = c.iloc[i - 1]
        upper[i] = (
            ub_basic
            if (ub_basic < upper[i - 1] or prev_close > upper[i - 1])
            else upper[i - 1]
        )
        lower[i] = (
            lb_basic
            if (lb_basic > lower[i - 1] or prev_close < lower[i - 1])
            else lower[i - 1]
        )
        close_i = c.iloc[i]
        if st[i - 1] == upper[i - 1]:
            st[i] = lower[i] if close_i > upper[i] else upper[i]
        else:
            st[i] = upper[i] if close_i < lower[i] else lower[i]
        direction[i] = 1 if st[i] == lower[i] else -1

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


def parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Welles Wilder Parabolic SAR. Returns the SAR series."""
    h, l, c = _hlc(df)
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    # Initial: assume uptrend, EP = high, SAR = low
    bull = True
    af = af_step
    ep = float(h.iloc[0])
    sar[0] = float(l.iloc[0])
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar_i = prev_sar + af * (ep - prev_sar)
            sar_i = min(sar_i, float(l.iloc[i - 1]), float(l.iloc[max(0, i - 2)]))
            if float(l.iloc[i]) < sar_i:
                bull = False
                sar_i = ep
                ep = float(l.iloc[i])
                af = af_step
            else:
                if float(h.iloc[i]) > ep:
                    ep = float(h.iloc[i])
                    af = min(af + af_step, af_max)
        else:
            sar_i = prev_sar + af * (ep - prev_sar)
            sar_i = max(sar_i, float(h.iloc[i - 1]), float(h.iloc[max(0, i - 2)]))
            if float(h.iloc[i]) > sar_i:
                bull = True
                sar_i = ep
                ep = float(h.iloc[i])
                af = af_step
            else:
                if float(l.iloc[i]) < ep:
                    ep = float(l.iloc[i])
                    af = min(af + af_step, af_max)
        sar[i] = sar_i
    return pd.Series(sar, index=df.index)


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    h, l, _ = _hlc(df)
    upper = h.rolling(n, min_periods=n).max()
    lower = l.rolling(n, min_periods=n).min()
    mid   = (upper + lower) / 2.0
    return upper, lower, mid


def keltner(df: pd.DataFrame, n: int = 20, atr_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = ema(df["close"].astype(float), n)
    rng   = atr(df, n)
    upper = mid + atr_mult * rng
    lower = mid - atr_mult * rng
    return upper, lower, mid


def atr_bands(df: pd.DataFrame, n: int = 14, atr_mult: float = 2.0) -> tuple[pd.Series, pd.Series]:
    a = atr(df, n)
    upper = df["close"].astype(float) + atr_mult * a
    lower = df["close"].astype(float) - atr_mult * a
    return upper, lower


def roc(series: pd.Series, n: int = 10) -> pd.Series:
    return (series / series.shift(n) - 1.0) * 100.0


def disparity(series: pd.Series, n: int = 14) -> pd.Series:
    m = sma(series, n)
    return (series - m) / m.replace(0, np.nan) * 100.0


def trix(series: pd.Series, n: int = 15) -> pd.Series:
    a = ema(series, n)
    b = ema(a, n)
    c = ema(b, n)
    return c.pct_change() * 100.0


# ── Bollinger derivatives ─────────────────────────────────────────────────────


def bollinger_bandwidth(series: pd.Series, n: int = 20) -> pd.Series:
    mid = sma(series, n)
    std = series.rolling(n, min_periods=n).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    return (upper - lower) / mid.replace(0, np.nan)


def bollinger_pct_b(series: pd.Series, n: int = 20) -> pd.Series:
    mid = sma(series, n)
    std = series.rolling(n, min_periods=n).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    rng = (upper - lower).replace(0, np.nan)
    return (series - lower) / rng


# ── Momentum ───────────────────────────────────────────────────────────────────


def stochastic_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = _hlc(df)
    lowest  = l.rolling(n, min_periods=n).min()
    highest = h.rolling(n, min_periods=n).max()
    rng = (highest - lowest).replace(0, np.nan)
    return 100.0 * (c - lowest) / rng


def stochastic_d(k_series: pd.Series, d_period: int = 3) -> pd.Series:
    return sma(k_series, d_period)


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = _hlc(df)
    highest = h.rolling(n, min_periods=n).max()
    lowest  = l.rolling(n, min_periods=n).min()
    rng = (highest - lowest).replace(0, np.nan)
    return -100.0 * (highest - c) / rng


def momentum(series: pd.Series, n: int = 10) -> pd.Series:
    return series - series.shift(n)


def cmo(series: pd.Series, n: int = 9) -> pd.Series:
    diff = series.diff()
    up = diff.clip(lower=0).rolling(n, min_periods=n).sum()
    dn = (-diff.clip(upper=0)).rolling(n, min_periods=n).sum()
    return 100.0 * (up - dn) / (up + dn).replace(0, np.nan)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = _typical_price(df)
    mean_tp = tp.rolling(n, min_periods=n).mean()
    mean_dev = (tp - mean_tp).abs().rolling(n, min_periods=n).mean()
    return (tp - mean_tp) / (0.015 * mean_dev.replace(0, np.nan))


# ── Volatility ─────────────────────────────────────────────────────────────────


def stdev(series: pd.Series, n: int = 20) -> pd.Series:
    return series.rolling(n, min_periods=n).std()


def historical_volatility(series: pd.Series, n: int = 20) -> pd.Series:
    """Annualised stdev of log returns. 252 trading-day convention."""
    logret = np.log(series / series.shift(1))
    return logret.rolling(n, min_periods=n).std() * math.sqrt(252)


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, _ = _hlc(df)
    tr_sum = pd.concat(
        [h - l,
         (h - df["close"].shift(1)).abs(),
         (l - df["close"].shift(1)).abs()],
        axis=1,
    ).max(axis=1).rolling(n, min_periods=n).sum()
    rng = h.rolling(n, min_periods=n).max() - l.rolling(n, min_periods=n).min()
    safe = rng.replace(0, np.nan)
    return 100.0 * np.log10(tr_sum / safe) / math.log10(n)


# ── Volume ─────────────────────────────────────────────────────────────────────


def obv(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    v = df["volume"].astype(float).fillna(0.0)
    direction = np.sign(c.diff()).fillna(0.0)
    return (direction * v).cumsum()


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    h, l, c = _hlc(df)
    v = df["volume"].astype(float).fillna(0.0)
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = (mfm.fillna(0.0)) * v
    return mfv.cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    h, l, c = _hlc(df)
    v = df["volume"].astype(float).fillna(0.0)
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm.fillna(0.0) * v
    return mfv.rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = _hlc(df)
    v = df["volume"].astype(float).fillna(0.0)
    tp = (h + l + c) / 3.0
    raw_mf = tp * v
    up   = raw_mf.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    down = raw_mf.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    mfr = up / down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + mfr))


def volume_sma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["volume"].astype(float).rolling(n, min_periods=n).mean()


def vroc(df: pd.DataFrame, n: int = 12) -> pd.Series:
    v = df["volume"].astype(float)
    return (v / v.shift(n) - 1.0) * 100.0


# ── Oscillators ────────────────────────────────────────────────────────────────


def aroon(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    h, l, _ = _hlc(df)
    up   = h.rolling(n + 1, min_periods=n + 1).apply(lambda x: 100.0 * (n - x[::-1].argmax()) / n, raw=True)
    down = l.rolling(n + 1, min_periods=n + 1).apply(lambda x: 100.0 * (n - x[::-1].argmin()) / n, raw=True)
    osc  = up - down
    return up, down, osc


# ── Support / resistance ───────────────────────────────────────────────────────


def hhv(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["high"].astype(float).rolling(n, min_periods=n).max()


def llv(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["low"].astype(float).rolling(n, min_periods=n).min()


def daily_pivots(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Classic daily floor pivots computed from PREVIOUS day's HLC. Intraday
    bars inherit the day's pivot levels; daily bars do too (shift by 1)."""
    is_daily = (
        isinstance(df.index, pd.DatetimeIndex)
        and len(df) >= 2
        and df.index.to_series().diff().dropna().median() >= pd.Timedelta(hours=6)
    )
    h, l, c = _hlc(df)
    if is_daily:
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        prev_c = c.shift(1)
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            day_key = df.index.normalize()
            session_high  = h.groupby(day_key).transform("max")
            session_low   = l.groupby(day_key).transform("min")
            session_close = c.groupby(day_key).transform("last")
            # Shift one session back. Use a tiny lookup of last-bar-per-day.
            session_first_idx = df.groupby(day_key).head(1).index
            session_pivot_map = pd.DataFrame(
                {"H": session_high.loc[session_first_idx].values,
                 "L": session_low.loc[session_first_idx].values,
                 "C": session_close.loc[session_first_idx].values},
                index=session_first_idx.normalize(),
            ).shift(1)
            # Re-broadcast per bar
            prev_h = pd.Series(session_pivot_map.loc[day_key, "H"].values, index=df.index)
            prev_l = pd.Series(session_pivot_map.loc[day_key, "L"].values, index=df.index)
            prev_c = pd.Series(session_pivot_map.loc[day_key, "C"].values, index=df.index)
        else:
            prev_h = h.shift(1)
            prev_l = l.shift(1)
            prev_c = c.shift(1)

    pivot = (prev_h + prev_l + prev_c) / 3.0
    r1 = 2 * pivot - prev_l
    s1 = 2 * pivot - prev_h
    r2 = pivot + (prev_h - prev_l)
    s2 = pivot - (prev_h - prev_l)
    return pivot, r1, r2, s1, s2


# ── Composite ─────────────────────────────────────────────────────────────────


def median_price(df: pd.DataFrame) -> pd.Series:
    h, l, _ = _hlc(df)
    return (h + l) / 2.0


def typical_price(df: pd.DataFrame) -> pd.Series:
    return _typical_price(df)


def weighted_close(df: pd.DataFrame) -> pd.Series:
    h, l, c = _hlc(df)
    return (h + l + 2.0 * c) / 4.0


def high_minus_low(df: pd.DataFrame) -> pd.Series:
    h, l, _ = _hlc(df)
    return h - l


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = _hlc(df)
    prev_c = c.shift(1)
    return pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)


# ── Orchestrator extension ─────────────────────────────────────────────────────


def add_extended_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    """Companion to indicators.add_all_indicators(). Adds columns for every
    extended indicator the caller requested. Unknown names are ignored
    silently (the catalog already gates what gets through)."""
    df = df.copy()
    close = df["close"].astype(float)
    if "volume" not in df.columns:
        df["volume"] = 0.0

    for indicator, periods in indicator_config.items():
        ind = indicator.upper()

        # Periodic single-param indicators ────────────────────────────────
        if ind == "ADX":
            for n in (periods or [14]):
                a, dp, dm = adx(df, int(n))
                df[f"ADX_{int(n)}"] = a
                df[f"DI_PLUS_{int(n)}"]  = dp
                df[f"DI_MINUS_{int(n)}"] = dm
        elif ind == "DMI":
            for n in (periods or [14]):
                _, dp, dm = adx(df, int(n))
                df[f"DI_PLUS_{int(n)}"]  = dp
                df[f"DI_MINUS_{int(n)}"] = dm
        elif ind in ("STOCH_K", "STOCH"):
            for n in (periods or [14]):
                k = stochastic_k(df, int(n))
                df[f"STOCH_K_{int(n)}"] = k
                df[f"STOCH_D_{int(n)}"] = stochastic_d(k, 3)
        elif ind == "STOCH_D":
            for n in (periods or [3]):
                k = stochastic_k(df, 14)
                df[f"STOCH_D_{int(n)}"] = stochastic_d(k, int(n))
        elif ind == "WILLR":
            for n in (periods or [14]):
                df[f"WILLR_{int(n)}"] = williams_r(df, int(n))
        elif ind == "ROC":
            for n in (periods or [10]):
                df[f"ROC_{int(n)}"] = roc(close, int(n))
        elif ind == "MOMENTUM":
            for n in (periods or [10]):
                df[f"MOMENTUM_{int(n)}"] = momentum(close, int(n))
        elif ind == "CMO":
            for n in (periods or [9]):
                df[f"CMO_{int(n)}"] = cmo(close, int(n))
        elif ind == "CCI":
            for n in (periods or [20]):
                df[f"CCI_{int(n)}"] = cci(df, int(n))
        elif ind == "STDEV":
            for n in (periods or [20]):
                df[f"STDEV_{int(n)}"] = stdev(close, int(n))
        elif ind == "HV":
            for n in (periods or [20]):
                df[f"HV_{int(n)}"] = historical_volatility(close, int(n))
        elif ind == "CHOPPINESS":
            for n in (periods or [14]):
                df[f"CHOPPINESS_{int(n)}"] = choppiness(df, int(n))
        elif ind == "DISPARITY":
            for n in (periods or [14]):
                df[f"DISPARITY_{int(n)}"] = disparity(close, int(n))
        elif ind == "TRIX":
            for n in (periods or [15]):
                df[f"TRIX_{int(n)}"] = trix(close, int(n))
        elif ind == "BB_WIDTH":
            for n in (periods or [20]):
                df[f"BB_WIDTH_{int(n)}"] = bollinger_bandwidth(close, int(n))
        elif ind == "BB_PCT_B":
            for n in (periods or [20]):
                df[f"BB_PCT_B_{int(n)}"] = bollinger_pct_b(close, int(n))
        elif ind == "VOLUME_SMA":
            for n in (periods or [20]):
                df[f"VOLUME_SMA_{int(n)}"] = volume_sma(df, int(n))
        elif ind == "VROC":
            for n in (periods or [12]):
                df[f"VROC_{int(n)}"] = vroc(df, int(n))
        elif ind == "MFI":
            for n in (periods or [14]):
                df[f"MFI_{int(n)}"] = mfi(df, int(n))
        elif ind == "CMF":
            for n in (periods or [20]):
                df[f"CMF_{int(n)}"] = cmf(df, int(n))
        elif ind == "AROON_UP":
            for n in (periods or [14]):
                up, down, osc = aroon(df, int(n))
                df[f"AROON_UP_{int(n)}"]   = up
                df[f"AROON_DOWN_{int(n)}"] = down
                df[f"AROON_OSC_{int(n)}"]  = osc
        elif ind == "AROON_DOWN":
            for n in (periods or [14]):
                _, down, _ = aroon(df, int(n))
                df[f"AROON_DOWN_{int(n)}"] = down
        elif ind == "AROON_OSC":
            for n in (periods or [14]):
                _, _, osc = aroon(df, int(n))
                df[f"AROON_OSC_{int(n)}"]  = osc
        elif ind == "HHV":
            for n in (periods or [20]):
                df[f"HHV_{int(n)}"] = hhv(df, int(n))
        elif ind == "LLV":
            for n in (periods or [20]):
                df[f"LLV_{int(n)}"] = llv(df, int(n))
        elif ind == "DONCHIAN":
            for n in (periods or [20]):
                upper, lower, mid = donchian(df, int(n))
                df[f"DON_UPPER_{int(n)}"] = upper
                df[f"DON_LOWER_{int(n)}"] = lower
                df[f"DON_MID_{int(n)}"]   = mid

        # Multi-param indicators ─────────────────────────────────────────
        elif ind == "SUPERTREND":
            for param_set in _ensure_param_sets(periods, default=(10, 3.0)):
                period, multiplier = int(param_set[0]), float(param_set[1])
                st_line, st_dir = supertrend(df, period, multiplier)
                suffix = column_suffix(period, multiplier)
                df[f"SUPERTREND_{suffix}"]     = st_line
                df[f"SUPERTREND_DIR_{suffix}"] = st_dir
        elif ind == "KELTNER":
            for param_set in _ensure_param_sets(periods, default=(20, 2.0)):
                period, mult = int(param_set[0]), float(param_set[1])
                u, l, m = keltner(df, period, mult)
                suffix = column_suffix(period, mult)
                df[f"KC_UPPER_{suffix}"] = u
                df[f"KC_LOWER_{suffix}"] = l
                df[f"KC_MID_{suffix}"]   = m
        elif ind == "ATR_UPPER":
            for param_set in _ensure_param_sets(periods, default=(14, 2.0)):
                period, mult = int(param_set[0]), float(param_set[1])
                u, _ = atr_bands(df, period, mult)
                df[f"ATR_UPPER_{column_suffix(period, mult)}"] = u
        elif ind == "ATR_LOWER":
            for param_set in _ensure_param_sets(periods, default=(14, 2.0)):
                period, mult = int(param_set[0]), float(param_set[1])
                _, l = atr_bands(df, period, mult)
                df[f"ATR_LOWER_{column_suffix(period, mult)}"] = l
        elif ind == "PSAR":
            for param_set in _ensure_param_sets(periods, default=(0.02, 0.2)):
                step, max_af = float(param_set[0]), float(param_set[1])
                df[f"PSAR_{column_suffix(step, max_af)}"] = parabolic_sar(df, step, max_af)

        # Scalar (no-period) indicators ──────────────────────────────────
        elif ind == "OBV":
            df["OBV"] = obv(df)
        elif ind == "ACCDIST":
            df["ACCDIST"] = accumulation_distribution(df)
        elif ind == "PIVOT":
            p, r1, r2, s1, s2 = daily_pivots(df)
            df["PIVOT"] = p
            df["R1"] = r1; df["R2"] = r2
            df["S1"] = s1; df["S2"] = s2
        elif ind == "MEDIAN_PRICE":
            df["MEDIAN_PRICE"] = median_price(df)
        elif ind == "TYPICAL_PRICE":
            df["TYPICAL_PRICE"] = typical_price(df)
        elif ind == "WEIGHTED_CLOSE":
            df["WEIGHTED_CLOSE"] = weighted_close(df)
        elif ind == "HIGH_LOW":
            df["HIGH_LOW"] = high_minus_low(df)
        elif ind == "TRUE_RANGE":
            df["TRUE_RANGE"] = true_range(df)

    return df


def _ensure_param_sets(periods, default: tuple) -> list[tuple]:
    """Normalise the indicator_config value into a list of param tuples.

    Accepts:
      - None / []                 → [default]
      - [n1, n2]                  → [(n1, default[1:]...), (n2, ...)] (single int per set)
      - [(n, mult), (n, mult)]    → as-is
    """
    if not periods:
        return [default]
    out: list[tuple] = []
    for entry in periods:
        if isinstance(entry, (list, tuple)):
            if len(entry) < len(default):
                entry = tuple(entry) + default[len(entry):]
            out.append(tuple(entry))
        else:
            out.append((entry,) + default[1:])
    return out
