"""
app/core/signal_param_estimator.py
────────────────────────────────────
Statistical Signal Parameter Estimation from OHLCV Data.

WHY THIS EXISTS
───────────────
Every signal has a default parameter on its SignalCard (app/kb/signals/) — RSI(14),
SMA(20,50), volume multiplier 1.5, etc.  These are universal defaults
invented decades ago for US daily charts.  They are NOT tuned for
HDFCBANK on a 1-minute chart or ADANIENT on a 5-minute chart.

The right approach is NOT to brute-force search for the best parameters
(that is called curve fitting / overfitting and will fail on live data).

The right approach is STATISTICAL ESTIMATION:
  - Ask the price data "how volatile are you?" → pick a faster or slower RSI
  - Ask the volume data "what volume level is unusual for you?" → set the spike threshold
  - Ask the price data "how much does a typical bar move?" → set MA periods

This describes the NATURE of the data, not what happened to work in the past.
It generalises to future data instead of overfitting to historical data.

SIMPLE ANALOGY
──────────────
A doctor doesn't give everyone the same dose of medicine.
They measure your weight and kidney function and calculate YOUR dose.

We measure the stock's volatility and volume behaviour and calculate ITS
optimal signal parameters.

HOW IT INTEGRATES
─────────────────
Resolution order inside app/planner/param_resolver.resolve_params:
  1. Performance cache  — if ≥10 backtests recorded for (signal, stock, tf)
  2. Statistical estimate — this module, using OHLCV data
  3. SignalCard defaults  — params_by_timeframe[tf] from app/kb/signals/<name>.yaml

Called by the planner via param_resolver.resolve_params() and also by
app.planner.strategy_assembler.enrich_plan_with_ohlcv() after the plan dict
is materialised (re-renders entry/exit condition strings).

MINIMUM DATA REQUIREMENT
────────────────────────
All estimators need at least 60 OHLCV bars to produce a reliable estimate.
Below this threshold every function returns the SignalCard defaults unchanged.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum bars required before we trust the statistical estimate.
# Below this we fall back to the SignalCard defaults — better a known-good default
# than a noisy estimate from too little data.
_MIN_BARS = 60


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI — same formula used by the quant engine."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _safe_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else fallback
    except (TypeError, ValueError):
        return fallback


def ohlcv_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of OHLCV dicts (from fetch_ohlcv_records) to a DataFrame.
    Handles both lowercase and mixed-case column names.
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp").sort_index()
    return df


# ── RSI parameter estimation ──────────────────────────────────────────────────

def estimate_rsi_params(
    df: pd.DataFrame,
    yaml_params: dict,
    objective: str = "intraday",
) -> dict:
    """
    Estimate RSI period and threshold from price volatility.

    SIMPLE EXPLANATION
    ──────────────────
    RSI measures how fast price reverses. Imagine two runners:
    - A sprinter (volatile stock): needs a shorter track to show their pace
    - A marathon runner (calm stock): needs a longer track

    If HDFCBANK 1m moves 0.3% per bar on average and SBI moves 0.1%,
    HDFCBANK needs a faster RSI (shorter period) or it's already reversed
    by the time RSI registers the signal.

    HOW WE MEASURE IT
    ─────────────────
    1. Compute the stock's recent volatility (std of last 20 returns)
    2. Compute the longer-term volatility (std of last 60 returns)
    3. If recent > long-term → stock is volatile now → shorter RSI
       If recent < long-term → stock is calm now → longer RSI

    Formula: new_period = default_period / sqrt(vol_ratio)
    Clamped to [7, 21] — never too extreme.

    For thresholds (oversold/overbought):
    Instead of always using 30/70, compute the historical RSI distribution
    for THIS stock and use the natural 15th/85th percentile.
    Example: HDFCBANK might rarely dip below RSI=25 (use 25, not 30).
    """
    close = df.get("close", pd.Series(dtype=float)).astype(float)
    if len(close) < _MIN_BARS:
        return dict(yaml_params)

    default_period = int(yaml_params.get("window", 14))
    returns = close.pct_change().dropna()
    if len(returns) < _MIN_BARS:
        return dict(yaml_params)

    recent_vol = _safe_float(returns.tail(20).std(), 0.0)
    long_vol   = _safe_float(returns.tail(60).std(), 0.0)

    if long_vol <= 0.0:
        return dict(yaml_params)

    # vol_ratio > 1 → more volatile recently → shorter period (faster RSI)
    # vol_ratio < 1 → calmer recently → longer period (smoother RSI)
    vol_ratio = max(0.4, min(2.5, recent_vol / long_vol))
    estimated_period = max(7, min(21, int(round(default_period / (vol_ratio ** 0.5)))))

    result = dict(yaml_params)
    result["window"] = estimated_period

    # Re-estimate threshold if this is an oversold/overbought signal
    if "threshold" in yaml_params:
        default_threshold = float(yaml_params["threshold"])

        # crossover signal (threshold=50): keep at 50 — it's a direction signal
        if 45.0 <= default_threshold <= 55.0:
            result["threshold"] = default_threshold
        else:
            rsi_series = _compute_rsi(close, estimated_period).dropna()
            if len(rsi_series) >= 30:
                if default_threshold <= 40.0:
                    # Oversold: use 15th percentile of RSI history
                    p15 = _safe_float(rsi_series.quantile(0.15), default_threshold)
                    rounded = round(p15 / 2.5) * 2.5  # round to nearest 2.5
                    result["threshold"] = max(20.0, min(40.0, rounded))
                else:
                    # Overbought: use 85th percentile of RSI history
                    p85 = _safe_float(rsi_series.quantile(0.85), default_threshold)
                    rounded = round(p85 / 2.5) * 2.5
                    result["threshold"] = max(60.0, min(82.0, rounded))

    logger.debug(
        "signal_param_estimator|rsi|vol_ratio=%.2f|period:%d→%d|threshold:%s→%s",
        vol_ratio,
        default_period, result["window"],
        yaml_params.get("threshold", "n/a"), result.get("threshold", "n/a"),
    )
    return result


# ── SMA / EMA period estimation ───────────────────────────────────────────────

def estimate_ma_params(
    df: pd.DataFrame,
    yaml_params: dict,
    objective: str = "intraday",
) -> dict:
    """
    Estimate SMA/EMA crossover or price-vs-MA periods from ATR.

    SIMPLE EXPLANATION
    ──────────────────
    A moving average should "smooth out noise but still catch moves."
    The right period depends on how noisy each candle is.

    ATR (Average True Range) tells us how much price typically moves per bar.
    If ATR is large relative to price, each bar is noisy.
    Noisy bars → you need shorter periods (react faster, fewer false lags).
    Calm bars  → longer periods work better (filter the small wiggles).

    Reference: for a typical NSE intraday stock on 15m, ATR ≈ 0.20% per bar.
    If the stock we're analysing has ATR = 0.40% (2× noisier), we shrink
    the MA periods by ~25% (the square root of 0.5 ≈ 0.7).

    Formula: new_period = default × (reference_atr / stock_atr)^0.35
    Clamped to reasonable ranges.

    This signal works for: sma_cross_up/down, ema_cross_up/down,
    price_above_ema, price_below_ema, is_above_sma.
    """
    close  = df.get("close", pd.Series(dtype=float)).astype(float)
    high   = df.get("high",  pd.Series(dtype=float)).astype(float)
    low    = df.get("low",   pd.Series(dtype=float)).astype(float)

    if len(close) < _MIN_BARS:
        return dict(yaml_params)

    # True Range: max of (H-L), |H-prevC|, |L-prevC|
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_pct = _safe_float(
        (tr / close.replace(0, np.nan)).rolling(20).mean().dropna().iloc[-1],
        0.0,
    )
    if atr_pct <= 0.0:
        return dict(yaml_params)

    # Reference ATR% per bar for typical NSE stocks (calibrated empirically)
    # Intraday: 15m ≈ 0.20%, 5m ≈ 0.12%, 1m ≈ 0.05%
    # Positional: daily ≈ 1.5%
    reference_atr = 0.002 if objective == "intraday" else 0.015

    # vol_factor: if stock is noisier than reference → shorter periods
    vol_factor = (reference_atr / atr_pct) ** 0.35
    vol_factor = max(0.50, min(1.80, vol_factor))

    result = dict(yaml_params)

    if "window_fast" in yaml_params and "window_slow" in yaml_params:
        default_fast = int(yaml_params["window_fast"])
        default_slow = int(yaml_params["window_slow"])
        new_fast = max(5,  min(30,  int(round(default_fast * vol_factor))))
        new_slow = max(new_fast + 10, min(200, int(round(default_slow * vol_factor))))
        result["window_fast"] = new_fast
        result["window_slow"] = new_slow
    elif "window" in yaml_params:
        default_w = int(yaml_params["window"])
        result["window"] = max(5, min(200, int(round(default_w * vol_factor))))

    logger.debug(
        "signal_param_estimator|ma|atr_pct=%.4f|vol_factor=%.2f|params:%s→%s",
        atr_pct, vol_factor, yaml_params, result,
    )
    return result


# ── Volume spike / dry-up estimation ─────────────────────────────────────────

def estimate_volume_params(
    df: pd.DataFrame,
    yaml_params: dict,
) -> dict:
    """
    Estimate volume spike multiplier from historical volume percentiles.

    SIMPLE EXPLANATION
    ──────────────────
    "Volume spike" means volume is unusually high.
    But what counts as "unusual" is different for every stock.

    HDFCBANK trades ₹3,000 crore per day.
    A small-cap trades ₹50 crore per day.
    Using the same multiplier (1.5×) for both is wrong.

    The right approach: look at the actual volume history of THIS stock.
    Compute the 80th percentile of (today's volume / 20-day avg volume).
    That IS the natural spike threshold — the level where volume is
    genuinely unusual for this specific stock.

    Example:
    - HDFCBANK: 80th percentile = 1.8× → use multiplier 1.8
    - ADANIENT: 80th percentile = 2.4× → use multiplier 2.4
    """
    volume = df.get("volume", pd.Series(dtype=float)).astype(float)
    if len(volume) < _MIN_BARS:
        return dict(yaml_params)

    default_window = int(yaml_params.get("window", 20))
    default_mult   = float(yaml_params.get("multiplier", 1.5))

    rolling_mean = volume.rolling(default_window).mean()
    vol_ratios = (volume / rolling_mean.replace(0, np.nan)).dropna()

    if len(vol_ratios) < 20:
        return dict(yaml_params)

    # 80th percentile = the level where volume is genuinely elevated
    p80 = _safe_float(vol_ratios.quantile(0.80), default_mult)

    result = dict(yaml_params)
    result["multiplier"] = round(max(1.2, min(4.0, p80)), 1)

    logger.debug(
        "signal_param_estimator|volume|p80_ratio=%.2f|multiplier:%s→%s",
        p80, default_mult, result["multiplier"],
    )
    return result


# ── Bollinger Band estimation ─────────────────────────────────────────────────

def estimate_bb_params(
    df: pd.DataFrame,
    yaml_params: dict,
    objective: str = "intraday",
) -> dict:
    """
    Estimate Bollinger Band std-dev width from realized volatility.

    SIMPLE EXPLANATION
    ──────────────────
    Bollinger Bands are supposed to contain ~95% of price moves within
    the band (assuming a normal distribution at 2 standard deviations).

    But this only holds if the stock's volatility is "average."
    - High-volatility stocks (ADANIENT): price escapes 2-std bands too often
      → false breakout signals → use wider bands (2.5 std devs)
    - Low-volatility stocks (HDFCBANK): price rarely touches 2-std bands
      → missed signals → use narrower bands (1.5 std devs)

    We compute the annualized realized volatility of THIS stock and
    choose the appropriate band width automatically.

    Volatility buckets:
      < 25% annual  → 1.5 std devs (narrow, more signals)
      25–50% annual → 2.0 std devs (standard)
      50–80% annual → 2.5 std devs (wider, fewer false breaks)
      > 80% annual  → 3.0 std devs (very wide, only real extremes)
    """
    close = df.get("close", pd.Series(dtype=float)).astype(float)
    if len(close) < _MIN_BARS:
        return dict(yaml_params)

    returns = close.pct_change().dropna()
    if len(returns) < 30:
        return dict(yaml_params)

    # Annualise: intraday 1m = 375 bars/session × 250 sessions; daily = 250
    bars_per_year = 375 * 250 if objective == "intraday" else 250
    realized_vol = _safe_float(returns.tail(60).std(), 0.0) * (bars_per_year ** 0.5)

    if realized_vol <= 0.0:
        return dict(yaml_params)

    if realized_vol < 0.25:
        new_dev = 1.5
    elif realized_vol < 0.50:
        new_dev = 2.0
    elif realized_vol < 0.80:
        new_dev = 2.5
    else:
        new_dev = 3.0

    result = dict(yaml_params)
    result["window_dev"] = new_dev

    logger.debug(
        "signal_param_estimator|bb|realized_vol=%.2f|window_dev:%s→%s",
        realized_vol, yaml_params.get("window_dev", 2), new_dev,
    )
    return result


# ── Gap threshold estimation ──────────────────────────────────────────────────

def estimate_gap_params(
    df: pd.DataFrame,
    yaml_params: dict,
) -> dict:
    """
    Estimate gap threshold from historical overnight gap distribution.

    SIMPLE EXPLANATION
    ──────────────────
    A "gap up" means the market opens significantly higher than yesterday's close.
    But what counts as "significant" differs by stock.

    HDFC Bank might gap by 0.2% on most days — that's just normal open variation.
    Adani Enterprises might gap by 0.5% on normal days.

    If we use 0.5% as the threshold for HDFC Bank, we'll get signals on
    almost every day. If we use 0.5% for Adani, we might miss real gaps.

    Solution: look at the actual gap history.
    The 75th percentile of historical gaps = the "real gap" threshold.
    Small gaps below this are just normal open noise. Larger ones are real.
    """
    close      = df.get("close", pd.Series(dtype=float)).astype(float)
    open_price = df.get("open",  pd.Series(dtype=float)).astype(float)

    if len(close) < _MIN_BARS:
        return dict(yaml_params)

    # Overnight gap as % of previous close
    gaps_pct = ((open_price - close.shift(1)) / close.shift(1)).abs() * 100.0
    gaps_pct = gaps_pct.dropna()

    if len(gaps_pct) < 20:
        return dict(yaml_params)

    # 75th percentile: above this level a gap is genuinely unusual
    p75 = _safe_float(gaps_pct.quantile(0.75), float(yaml_params.get("gap_pct", 0.5)))

    result = dict(yaml_params)
    result["gap_pct"] = round(max(0.15, min(2.0, p75)), 2)

    logger.debug(
        "signal_param_estimator|gap|p75_gap_pct=%.3f|gap_pct:%s→%s",
        p75, yaml_params.get("gap_pct", 0.5), result["gap_pct"],
    )
    return result


# ── Main dispatcher ───────────────────────────────────────────────────────────

def estimate_params_for_signal(
    signal_name: str,
    df: pd.DataFrame,
    objective: str,
    timeframe: str,
    yaml_params: dict,
) -> dict:
    """
    Route to the right estimator based on signal name.
    Always returns a valid params dict (falls back to yaml_params on any error).

    Signals that have no free parameters (VWAP, inside_bar) are returned unchanged.
    MACD params are intentionally kept at defaults (12/26/9 is universally accepted).
    """
    if not isinstance(df, pd.DataFrame) or len(df) < _MIN_BARS:
        return dict(yaml_params)

    name = signal_name.lower().strip()

    try:
        # ── RSI family ────────────────────────────────────────────────────────
        if name in {
            "rsi_oversold", "rsi_overbought",
            "rsi_cross_up", "rsi_cross_down",
        }:
            return estimate_rsi_params(df, yaml_params, objective)

        # ── Moving average family ─────────────────────────────────────────────
        if name in {
            "sma_cross_up", "sma_cross_down",
            "ema_cross_up", "ema_cross_down",
            "price_above_ema", "price_below_ema",
            "is_above_sma",
        }:
            return estimate_ma_params(df, yaml_params, objective)

        # ── Volume family ─────────────────────────────────────────────────────
        if name in {
            "volume_spike", "volume_dry_up", "high_delivery_volume",
        }:
            return estimate_volume_params(df, yaml_params)

        # ── Bollinger Band family ─────────────────────────────────────────────
        if name in {
            "price_above_bb_upper", "price_below_bb_lower",
            "bb_pct_b_high", "bb_pct_b_low",
        }:
            return estimate_bb_params(df, yaml_params, objective)

        # ── Gap family ────────────────────────────────────────────────────────
        if name in {"gap_up_open", "gap_down_open"}:
            return estimate_gap_params(df, yaml_params)

        # ── No estimation needed (params are fixed by definition) ─────────────
        # vwap_bullish / vwap_bearish      — no params
        # inside_bar                        — no params
        # macd_bullish_cross / _bearish     — 12/26/9 is universally accepted
        # opening_range_breakout            — session-timing, not price-derived
        return dict(yaml_params)

    except Exception as exc:
        logger.warning(
            "signal_param_estimator|error|signal=%s|error=%s — using yaml defaults",
            signal_name, exc,
        )
        return dict(yaml_params)


# ── Plan-level enrichment (the main public API) ───────────────────────────────

def enrich_plan_params(
    plan: dict,
    df: pd.DataFrame,
    objective: str,
    timeframe: str,
) -> dict:
    """
    Update ALL signal parameters in a plan using OHLCV statistical estimation.

    Takes the signal plan dict from plan_strategy_signals() and returns a NEW
    plan dict with statistically estimated params and re-rendered formula conditions.

    This is the function you call from chat_service.py after fetching OHLCV data.

    Example:
        ohlcv_records = await fetch_ohlcv_records(market_data_request)
        df = ohlcv_to_df(ohlcv_records)
        builder.signal_plan = enrich_plan_params(
            builder.signal_plan, df, builder.objective, builder.timeframe
        )

    What it does:
        1. For each entry signal → estimate best params from OHLCV
        2. For each exit signal  → estimate best params from OHLCV
        3. Re-render the entry_condition and exit_condition formula strings
           using the new params
        4. Log what changed (for debugging)

    If OHLCV has fewer than 60 bars, returns the original plan unchanged.
    """
    if not isinstance(df, pd.DataFrame) or len(df) < _MIN_BARS or not plan:
        return plan

    # Lazy import to avoid circular dependency at module level
    from app.planner.formulas import render_formula as _signal_formula  # noqa: PLC0415

    new_plan = copy.deepcopy(plan)
    changes: list[str] = []

    all_signals = new_plan.get("entry", []) + new_plan.get("exit", [])
    for signal in all_signals:
        name = str(signal.get("name") or "")
        if not name:
            continue

        old_params = dict(signal.get("params") or {})
        new_params = estimate_params_for_signal(name, df, objective, timeframe, old_params)
        signal["params"] = new_params

        # Track what changed for logging
        diffs = {k: (old_params.get(k), new_params[k]) for k in new_params if new_params[k] != old_params.get(k)}
        if diffs:
            changes.append(f"{name}: {diffs}")

    # Re-render formula strings with new params
    entry_parts = [
        f
        for s in new_plan.get("entry", [])
        for f in [_signal_formula(s.get("name", ""), s.get("params", {}))]
        if f is not None
    ]
    if entry_parts:
        new_plan["entry_condition"] = " AND ".join(entry_parts)

    exit_parts = [
        f
        for s in new_plan.get("exit", [])
        for f in [_signal_formula(s.get("name", ""), s.get("params", {}))]
        if f is not None
    ]
    if exit_parts:
        new_plan["exit_condition"] = exit_parts[0]

    if changes:
        logger.info(
            "signal_param_estimator|plan_enriched|bars=%d|objective=%s|timeframe=%s|changes=%s"
            "|entry_condition=%r|exit_condition=%r",
            len(df), objective, timeframe, changes,
            new_plan.get("entry_condition"), new_plan.get("exit_condition"),
        )
    else:
        logger.debug(
            "signal_param_estimator|plan_enriched|bars=%d|no_param_changes",
            len(df),
        )

    return new_plan
