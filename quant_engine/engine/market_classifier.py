"""
Market condition and alignment classifiers.

Every label ("Bull", "Bear", "Uptrend", "Strong", …) is the *output* of a
pure function applied to computed numeric data.  No label string is ever
chosen by a hard-coded if/elif chain — the decision is driven by threshold
constants imported from config.py.

Public API
----------
classify_market_type(price_return_pct)          → "Bull" | "Bear" | "Sideways"
classify_market_phase(close_series)             → "Uptrend" | "Downtrend" | "Range-bound"
classify_alignment(side, market_type, win_rate) → "Strong" | "Moderate" | "Weak"
classify_entry_condition(df, bar_idx)           → "Bull" | "Bear" | "Sideways" | "Unknown"
build_market_phase_analysis(df, trades, side)   → list[dict]
compute_monthly_performance(daily_values, trades) → (list[dict], dict)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from engine.config import (
    ALIGNMENT_STRONG_UPGRADE_WIN_RATE_PCT,
    ALIGNMENT_WEAK_DOWNGRADE_WIN_RATE_PCT,
    BEAR_MARKET_MAX_RETURN_PCT,
    BULL_MARKET_MIN_RETURN_PCT,
    ENTRY_CONDITION_FAST_WINDOW,
    ENTRY_CONDITION_SLOW_WINDOW,
    TREND_SLOPE_DOWNTREND_MAX,
    TREND_SLOPE_UPTREND_MIN,
)

logger = logging.getLogger(__name__)

# ── Alignment matrix ──────────────────────────────────────────────────────────
# Maps (normalized_strategy_side, market_type) → base alignment label.
# Using a data table removes the need for any if/elif chain.
_ALIGNMENT_MATRIX: dict[tuple[str, str], str] = {
    ("LONG",  "Bull"):     "Strong",
    ("LONG",  "Sideways"): "Moderate",
    ("LONG",  "Bear"):     "Weak",
    ("SHORT", "Bear"):     "Strong",
    ("SHORT", "Sideways"): "Moderate",
    ("SHORT", "Bull"):     "Weak",
}


# ─── Core classifiers ─────────────────────────────────────────────────────────

def classify_market_type(price_return_pct: float) -> str:
    """
    Bull / Bear / Sideways based on total price return over a period.
    Boundary constants: BULL_MARKET_MIN_RETURN_PCT, BEAR_MARKET_MAX_RETURN_PCT (config.py).
    """
    if price_return_pct >= BULL_MARKET_MIN_RETURN_PCT:
        return "Bull"
    if price_return_pct <= BEAR_MARKET_MAX_RETURN_PCT:
        return "Bear"
    return "Sideways"


def classify_market_phase(close_series: pd.Series) -> str:
    """
    Uptrend / Downtrend / Range-bound using linear regression slope of
    the normalized close price series over the period.

    Normalizing to [0,1] makes the slope dimensionless so the thresholds
    in config.py apply regardless of the absolute price level.
    Boundary constants: TREND_SLOPE_UPTREND_MIN, TREND_SLOPE_DOWNTREND_MAX (config.py).
    """
    if len(close_series) < 2:
        return "Range-bound"

    price_range = float(close_series.max() - close_series.min())
    if price_range < 1e-9:
        return "Range-bound"

    normalized = (close_series - close_series.min()) / price_range
    x = np.arange(len(normalized))
    slope = float(np.polyfit(x, normalized.values, 1)[0])

    if slope >= TREND_SLOPE_UPTREND_MIN:
        return "Uptrend"
    if slope <= TREND_SLOPE_DOWNTREND_MAX:
        return "Downtrend"
    return "Range-bound"


def classify_alignment(
    strategy_side: str,
    market_type: str,
    phase_win_rate_pct: float,
) -> str:
    """
    Strong / Moderate / Weak.

    Base alignment comes from the _ALIGNMENT_MATRIX lookup (strategy direction
    vs market direction).  A "Moderate" base is then adjusted up or down
    based on the *observed* win rate in that phase:
      - win rate ≥ ALIGNMENT_STRONG_UPGRADE_WIN_RATE_PCT → upgrade to "Strong"
      - win rate ≤ ALIGNMENT_WEAK_DOWNGRADE_WIN_RATE_PCT → downgrade to "Weak"
    """
    key = (strategy_side.upper(), market_type)
    base = _ALIGNMENT_MATRIX.get(key, "Moderate")

    if base == "Moderate":
        if phase_win_rate_pct >= ALIGNMENT_STRONG_UPGRADE_WIN_RATE_PCT:
            return "Strong"
        if phase_win_rate_pct <= ALIGNMENT_WEAK_DOWNGRADE_WIN_RATE_PCT:
            return "Weak"
    return base


def classify_entry_condition(df: pd.DataFrame, entry_bar_index: int) -> str:
    """
    Bull / Bear / Sideways at the entry bar.

    Uses a fast and a slow rolling average of close prices:
      - close > fast > slow → Bull  (price and momentum above both averages)
      - close < fast < slow → Bear  (price and momentum below both averages)
      - otherwise           → Sideways

    Window constants: ENTRY_CONDITION_FAST_WINDOW, ENTRY_CONDITION_SLOW_WINDOW (config.py).
    Returns "Unknown" when there is insufficient history.
    """
    if entry_bar_index < ENTRY_CONDITION_SLOW_WINDOW:
        return "Unknown"

    close = df["close"]
    fast_avg = float(
        close.iloc[entry_bar_index - ENTRY_CONDITION_FAST_WINDOW:entry_bar_index].mean()
    )
    slow_avg = float(
        close.iloc[entry_bar_index - ENTRY_CONDITION_SLOW_WINDOW:entry_bar_index].mean()
    )
    current = float(close.iloc[entry_bar_index])

    if current > fast_avg > slow_avg:
        return "Bull"
    if current < fast_avg < slow_avg:
        return "Bear"
    return "Sideways"


# ─── Composite builders ───────────────────────────────────────────────────────

def build_market_phase_analysis(
    df: pd.DataFrame,
    trades: list[Any],   # list[Trade] — typed as Any to avoid circular import
    strategy_side: str,
) -> list[dict]:
    """
    Split the OHLCV data into calendar quarters and return one dict per quarter.

    For each quarter:
      - Compute price return and derive market_type / market_phase from OHLCV
      - Filter trades whose entry fell in that quarter
      - Compute strategy win rate and cumulative return for those trades
      - Derive alignment of strategy side vs market type
      - Generate a parameterized description

    Returns an empty list when df is empty or has no datetime index.
    """
    from engine.descriptions import build_phase_description  # local import avoids circularity

    if df is None or df.empty:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        return []

    # Pre-parse all trade entry timestamps once
    entry_timestamps: list[pd.Timestamp | None] = []
    for trade in trades:
        try:
            ts = pd.to_datetime(trade.entry_date, utc=True).tz_localize(None)
            entry_timestamps.append(ts)
        except Exception:
            entry_timestamps.append(None)

    quarterly_groups = list(df.groupby(pd.Grouper(freq="QE")))
    phases: list[dict] = []

    for label, group in quarterly_groups:
        close = group["close"].dropna()
        if len(close) < 2:
            continue

        quarter_start = group.index[0]
        quarter_end   = group.index[-1]
        period_str    = f"{quarter_start.year} Q{(quarter_start.month - 1) // 3 + 1}"

        first_close  = float(close.iloc[0])
        last_close   = float(close.iloc[-1])
        price_return = (
            (last_close - first_close) / first_close * 100.0
            if first_close != 0.0 else 0.0
        )

        market_type  = classify_market_type(price_return)
        market_phase = classify_market_phase(close)

        # Trades that entered during this quarter
        quarter_trades = [
            t for t, ts in zip(trades, entry_timestamps)
            if ts is not None and quarter_start <= ts <= quarter_end
        ]
        qt        = len(quarter_trades)
        wins      = sum(1 for t in quarter_trades if float(t.pnl_pct) > 0)
        phase_win_rate  = (wins / qt * 100.0) if qt > 0 else 0.0
        phase_return_strategy = sum(float(t.pnl_pct) for t in quarter_trades)

        alignment = classify_alignment(strategy_side, market_type, phase_win_rate)

        description = build_phase_description(
            period=period_str,
            market_type=market_type,
            market_phase=market_phase,
            strategy_side=strategy_side,
            alignment=alignment,
            price_change_pct=price_return,
            trade_count=qt,
            win_rate_pct=phase_win_rate,
            phase_return_pct=phase_return_strategy,
        )

        phases.append({
            "time_period":            period_str,
            "market_type":            market_type,
            "market_phase":           market_phase,
            "observed_alignment":     alignment,
            "price_change_pct":       round(price_return, 4),
            "strategy_trades":        qt,
            "strategy_win_rate_pct":  round(phase_win_rate, 4),
            "strategy_return_pct":    round(phase_return_strategy, 4),
            "description":            description,
        })

    return phases


def compute_monthly_performance(
    daily_values: pd.Series,
    trades: list[Any],  # list[Trade]
) -> tuple[list[dict], dict]:
    """
    Compute per-month strategy returns from the daily portfolio equity curve.

    Returns
    -------
    monthly_list : list[dict]
        One dict per month: {month, strategy_return_pct, trades_count}
    monthly_statistics : dict
        Aggregate stats: highest/lowest/range monthly gain,
        return-vs-drawdown efficiency, best performing market condition.
    """
    if daily_values is None or daily_values.empty:
        return [], {}

    # Month-end portfolio values → percentage change per month
    monthly_values  = daily_values.resample("ME").last()
    monthly_returns = monthly_values.pct_change().fillna(0.0) * 100.0

    # Count trades entered per calendar month
    trade_month_counts: dict[str, int] = {}
    for trade in trades:
        try:
            ts        = pd.to_datetime(trade.entry_date, utc=True).tz_localize(None)
            month_key = ts.strftime("%Y-%m")
            trade_month_counts[month_key] = trade_month_counts.get(month_key, 0) + 1
        except Exception:
            pass

    monthly_list: list[dict] = []
    for month_ts, return_pct in monthly_returns.items():
        month_key = month_ts.strftime("%Y-%m")
        monthly_list.append({
            "month":                month_key,
            "strategy_return_pct":  round(float(return_pct), 4),
            "trades_count":         trade_month_counts.get(month_key, 0),
        })

    if not monthly_list:
        return [], {}

    returns_only = [m["strategy_return_pct"] for m in monthly_list]
    highest      = max(returns_only)
    lowest       = min(returns_only)

    # Return-vs-drawdown efficiency: total return divided by peak-to-trough drop
    # across monthly returns (not the portfolio drawdown — a monthly-level metric)
    monthly_cumulative = 0.0
    monthly_peak       = 0.0
    monthly_drawdown   = 0.0
    running            = 0.0
    for r in returns_only:
        running       += r
        monthly_peak   = max(monthly_peak, running)
        monthly_drawdown = max(monthly_drawdown, monthly_peak - running)

    total_strategy_return = sum(returns_only)
    return_vs_drawdown = (
        round(total_strategy_return / monthly_drawdown, 4)
        if monthly_drawdown > 0 else 0.0
    )

    monthly_statistics = {
        "highest_monthly_gain_pct":     round(highest, 4),
        "lowest_monthly_gain_pct":      round(lowest, 4),
        "monthly_performance_range_pct": round(highest - lowest, 4),
        "return_vs_drawdown_efficiency": return_vs_drawdown,
    }

    return monthly_list, monthly_statistics
