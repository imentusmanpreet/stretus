"""
Trade simulator — supports both intraday and positional strategies.

Key differences by objective:
  intraday:
    - warm_up_candles: first N bars skipped (indicators not yet ready)
    - max_holding_candles enforced per session (e.g. 25 for 15m = NSE full day)
    - daily_loss_cap_pct: halt new entries once portfolio drops >= cap% in a calendar day
    - max_trades_per_day: limit number of round-trips per calendar day (0 = unlimited)
    - STT: 0% on buy, 0.025% on sell (intraday equity, NSE rules)

  positional:
    - warm_up_candles: first N bars skipped
    - max_holding_candles based on multi-day window (e.g. 20 daily bars ~= 1 month)
    - no per-day trade or loss constraints (intended to hold across sessions)
    - STT: 0.1% on both buy and sell (delivery equity, NSE rules)

Indian market cost model:
  Total entry cost = slippage + commission + STT_buy
  Total exit cost  = slippage + commission + STT_sell
  Both expressed as a percentage of the trade price.

Condition diagnostics:
    Each candle evaluation is tracked and returned so callers can understand
    exactly why/where entries and exits occurred (or didn't).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import logging
from typing import Any

import numpy as np
import pandas as pd

from engine.conditions import CompiledCondition, build_arrays_from_df, evaluate_condition
from engine.htf import HtfContext, all_htf_gates_pass
from engine.kb_signals import KB_REGISTRY_AVAILABLE, evaluate_kb_signal_rules

logger = logging.getLogger(__name__)


# ─── Trade dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Trade:
    entry_date: str
    exit_date: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_abs: float          # fractional return, e.g. 0.05 = 5% gain (used for compounding)
    pnl_inr: float          # absolute rupee P&L per unit (exit_price - entry_price)
    exit_reason: str
    holding_candles: int
    # Maximum Adverse Excursion: worst unrealised loss during the trade (always ≤ 0 for long)
    mae_pct: float = 0.0
    # Maximum Favorable Excursion: best unrealised gain during the trade (always ≥ 0 for long)
    mfe_pct: float = 0.0


# ─── Candle diagnostic ────────────────────────────────────────────────────────

@dataclass
class CandleDiagnostic:
    index: int
    timestamp: str
    warm_up_skip: bool = False
    in_trade: bool = False
    entry_evaluated: bool = False
    entry_signal: bool = False
    entry_blocked_daily_cap: bool = False
    entry_blocked_max_trades: bool = False
    entry_blocked_htf: bool = False           # Phase 5: an HTF entry-gate failed
    entry_blocked_time_exit: bool = False     # Phase 8b: past the time_exit cutoff
    # Phase 10 gates
    entry_blocked_entry_window: bool = False  # outside the entry_window
    entry_blocked_consecutive_losses: bool = False
    entry_blocked_cooldown: bool = False
    entry_blocked_spread: bool = False
    entry_blocked_gap: bool = False
    entry_blocked_confirmation: bool = False  # signal not yet held N consecutive bars
    entry_blocked_rsi_band: bool = False
    entry_blocked_volume_ratio: bool = False
    exit_evaluated: bool = False
    exit_signal: bool = False
    stop_hit: bool = False
    tp_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index":                           self.index,
            "timestamp":                       self.timestamp,
            "warm_up_skip":                    self.warm_up_skip,
            "in_trade":                        self.in_trade,
            "entry_evaluated":                 self.entry_evaluated,
            "entry_signal":                    self.entry_signal,
            "entry_blocked_daily_cap":         self.entry_blocked_daily_cap,
            "entry_blocked_max_trades":        self.entry_blocked_max_trades,
            "entry_blocked_htf":               self.entry_blocked_htf,
            "entry_blocked_time_exit":         self.entry_blocked_time_exit,
            "entry_blocked_entry_window":      self.entry_blocked_entry_window,
            "entry_blocked_consecutive_losses":self.entry_blocked_consecutive_losses,
            "entry_blocked_cooldown":          self.entry_blocked_cooldown,
            "entry_blocked_spread":            self.entry_blocked_spread,
            "entry_blocked_gap":               self.entry_blocked_gap,
            "entry_blocked_confirmation":      self.entry_blocked_confirmation,
            "entry_blocked_rsi_band":          self.entry_blocked_rsi_band,
            "entry_blocked_volume_ratio":      self.entry_blocked_volume_ratio,
            "exit_evaluated":                  self.exit_evaluated,
            "exit_signal":                     self.exit_signal,
            "stop_hit":                        self.stop_hit,
            "tp_hit":                          self.tp_hit,
        }


# ─── Cost helpers ──────────────────────────────────────────────────────────────
#
# Indian equity costs:
#   slippage_bps  = market impact (bid-ask spread + execution imperfection)
#   commission_bps = broker fee in basis points  (OR use flat ₹20 model separately)
#   stt_pct       = Securities Transaction Tax as a plain percentage of price
#
# Entry: price increases by (slippage + commission) bps PLUS stt_pct on buy side
# Exit:  price decreases by (slippage + commission) bps PLUS stt_pct on sell side

def _apply_entry_costs(
    raw_price: float,
    slippage_bps: float,
    commission_bps: float,
    stt_pct: float = 0.0,
) -> float:
    """
    Effective entry price after all buy-side costs.
    stt_pct = 0.0 for intraday (no STT on buy), 0.1 for delivery.
    """
    bps_cost = (slippage_bps + commission_bps) / 10_000.0
    return raw_price * (1.0 + bps_cost + stt_pct / 100.0)


def _apply_exit_costs(
    raw_price: float,
    slippage_bps: float,
    commission_bps: float,
    stt_pct: float = 0.025,
) -> float:
    """
    Effective exit price after all sell-side costs.
    stt_pct = 0.025 for intraday sell, 0.1 for delivery.
    """
    bps_cost = (slippage_bps + commission_bps) / 10_000.0
    return raw_price * (1.0 - bps_cost - stt_pct / 100.0)


def _trade_variables(
    entry_price: float,
    current_close: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict[str, float]:
    profit_pct = ((current_close - entry_price) / entry_price) * 100.0
    loss_pct   = -profit_pct
    return {
        "PROFIT":             profit_pct,
        "LOSS":               loss_pct,
        "TAKE_PROFIT_TARGET": take_profit_pct,
        "STOP_LOSS_TARGET":   stop_loss_pct,
    }


# ─── Phase 3: structural & trailing stop helpers ──────────────────────────────

def _session_open_low(
    low_arr: np.ndarray,
    day_ordinals: np.ndarray,
    entry_index: int,
    opening_bars: int,
) -> float:
    """Lowest low across the first `opening_bars` bars of entry_index's session.

    Returns the low from the last available bar in that opening range — the SL
    is anchored at this level. Returns the entry-bar low itself as a sane
    fallback when the entry bar is inside the opening range (rare for ORB
    strategies but possible for non-intraday backtests with weird sessions).
    """
    if opening_bars <= 0 or entry_index < 0 or entry_index >= len(low_arr):
        return float(low_arr[max(entry_index, 0)])
    session_ord = int(day_ordinals[entry_index])
    # Find the first bar of this session (the opening bar).
    session_start = entry_index
    while session_start > 0 and int(day_ordinals[session_start - 1]) == session_ord:
        session_start -= 1
    end = min(session_start + opening_bars, entry_index + 1, len(low_arr))
    return float(np.min(low_arr[session_start:end]))


def _session_open_high(
    high_arr: np.ndarray,
    day_ordinals: np.ndarray,
    entry_index: int,
    opening_bars: int,
) -> float:
    """Highest high across the first `opening_bars` bars of entry_index's session."""
    if opening_bars <= 0 or entry_index < 0 or entry_index >= len(high_arr):
        return float(high_arr[max(entry_index, 0)])
    session_ord = int(day_ordinals[entry_index])
    session_start = entry_index
    while session_start > 0 and int(day_ordinals[session_start - 1]) == session_ord:
        session_start -= 1
    end = min(session_start + opening_bars, entry_index + 1, len(high_arr))
    return float(np.max(high_arr[session_start:end]))


def _compute_initial_stop_long(
    spec: dict | None,
    *,
    fallback_pct: float,
    entry_price: float,
    entry_index: int,
    arrays: dict[str, np.ndarray],
    low_arr: np.ndarray,
    day_ordinals: np.ndarray,
) -> float:
    """Return the initial stop price for a LONG entry.
    NaN-safe; falls back to the legacy percent-based SL if the spec can't be
    resolved (e.g. ATR column not yet warmed up)."""
    if spec is None:
        return entry_price * (1.0 - fallback_pct / 100.0)

    sl_type = spec.get("type")

    if sl_type == "percent":
        return entry_price * (1.0 - float(spec["pct"]) / 100.0)

    if sl_type == "structural":
        anchor = spec.get("anchor")
        padding_pct = float(spec.get("padding_pct", 0.0))
        anchor_price: float
        if anchor == "opening_range_low":
            anchor_price = _session_open_low(
                low_arr, day_ordinals, entry_index, int(spec.get("opening_bars", 3)),
            )
        elif anchor == "opening_range_high":
            # Reserved for shorts; for a long entry an upside anchor is nonsensical
            # so we degrade to the % fallback instead of producing a stop above entry.
            return entry_price * (1.0 - fallback_pct / 100.0)
        elif anchor == "prev_n_bar_low":
            window = int(spec.get("window", 5))
            start = max(0, entry_index - window)
            anchor_price = float(np.min(low_arr[start:entry_index])) if entry_index > start else float(low_arr[entry_index])
        elif anchor == "prev_n_bar_high":
            return entry_price * (1.0 - fallback_pct / 100.0)
        else:
            return entry_price * (1.0 - fallback_pct / 100.0)
        # padding pulls the SL slightly *below* the anchor so a wick to the level
        # doesn't trigger an exit (e.g. anchor=99.5, padding=0.1% → SL=99.4)
        return float(anchor_price) * (1.0 - padding_pct / 100.0)

    if sl_type == "atr":
        window = int(spec["window"])
        atr_col = arrays.get(f"ATR_{window}")
        if atr_col is None or entry_index >= len(atr_col):
            return entry_price * (1.0 - fallback_pct / 100.0)
        atr_value = float(atr_col[entry_index])
        if atr_value != atr_value or atr_value <= 0:        # NaN check
            return entry_price * (1.0 - fallback_pct / 100.0)
        return entry_price - float(spec["multiplier"]) * atr_value

    return entry_price * (1.0 - fallback_pct / 100.0)


def _compute_trailing_floor_long(
    spec: dict | None,
    *,
    entry_price: float,
    current_index: int,
    arrays: dict[str, np.ndarray],
    highest_high_since_entry: float,
    current_close: float,
) -> float:
    """Return today's trailing-floor candidate for a LONG trade.
    Returns -inf when trailing isn't applicable yet so it can't override the
    initial SL (the caller takes max(initial_stop, this_value)).
    """
    if spec is None:
        return float("-inf")

    activate = float(spec.get("activate_after_pct", 0.0))
    if activate > 0.0:
        gain_pct = ((current_close - entry_price) / entry_price) * 100.0
        if gain_pct < activate:
            return float("-inf")

    ts_type = spec.get("type")

    if ts_type == "percent":
        return highest_high_since_entry * (1.0 - float(spec["distance_pct"]) / 100.0)

    if ts_type in {"atr", "chandelier"}:
        window = int(spec["window"])
        atr_col = arrays.get(f"ATR_{window}")
        if atr_col is None or current_index >= len(atr_col):
            return float("-inf")
        atr_value = float(atr_col[current_index])
        if atr_value != atr_value or atr_value <= 0:
            return float("-inf")
        return highest_high_since_entry - float(spec["multiplier"]) * atr_value

    if ts_type == "ema":
        window = int(spec["window"])
        ema_col = arrays.get(f"EMA_{window}")
        if ema_col is None or current_index >= len(ema_col):
            return float("-inf")
        ema_value = float(ema_col[current_index])
        if ema_value != ema_value:
            return float("-inf")
        return ema_value

    return float("-inf")


# ─── Session-day helpers ───────────────────────────────────────────────────────

def _bar_session_date(df: pd.DataFrame, i: int) -> date | None:
    """Return the calendar date of bar i, or None if unknown."""
    try:
        ts = df.index[i]
        if hasattr(ts, "date"):
            return ts.date()
    except Exception:
        pass
    return None


def _is_session_last_bar(df: pd.DataFrame, i: int) -> bool:
    """
    True when bar i is the final bar in its trading session.

    For intraday strategies we treat any session boundary as a hard square-off
    point so positions cannot spill into the next trading day.
    """
    if i >= len(df) - 1:
        return True
    current_date = _bar_session_date(df, i)
    next_date = _bar_session_date(df, i + 1)
    return current_date is not None and next_date is not None and current_date != next_date


# ─── Main simulation function ──────────────────────────────────────────────────

def simulate_trades(
    *,
    df: pd.DataFrame,
    symbol: str,
    entry_condition: str,
    exit_condition: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    slippage_bps: float,
    commission_bps: float,
    warm_up_candles: int = 0,
    max_holding_candles: int | None = None,
    objective: str = "positional",
    daily_loss_cap_pct: float = 0.0,
    max_trades_per_day: int = 0,
    # Indian market STT (Securities Transaction Tax)
    # Intraday equity: 0% on buy, 0.025% on sell
    # Delivery equity: 0.1% on buy AND sell
    stt_intraday_sell_pct: float = 0.025,
    stt_delivery_pct: float = 0.1,
    entry_evaluation_mode: str = "formula",
    exit_evaluation_mode: str = "formula",
    entry_signal_rules: list[dict[str, Any]] | None = None,
    exit_signal_rules: list[dict[str, Any]] | None = None,
    compiled_entry: CompiledCondition | None = None,
    compiled_exit: CompiledCondition | None = None,
    # Phase 3 — when set, the simulator computes the SL price from the spec
    # instead of treating it as a flat percent of entry. trailing_stop_spec
    # ratchets the SL upward as the trade goes favorably. Both are optional
    # and additive: omitting them yields the legacy behavior exactly.
    stop_loss_spec: dict[str, Any] | None = None,
    trailing_stop_spec: dict[str, Any] | None = None,
    # Phase 5 — higher-timeframe entry gates. When non-empty, every gate's
    # condition must pass on its most recently *closed* HTF bar before an
    # LTF entry signal is allowed to fill. Empty list = no HTF gating.
    htf_contexts: list[HtfContext] | None = None,
    # Phase 8b — wall-clock intraday cutoff. When set, the simulator force-
    # exits any open trade once a bar's UTC time-of-day reaches the cutoff,
    # and blocks new entries past that time. Loader pre-computes
    # `utc_minutes_of_day` so this hot-path stays a plain int compare.
    time_exit_spec: dict[str, Any] | None = None,
    # Phase 10 — entry window. Blocks new entries outside the UTC window
    # (does NOT force-exit open trades, unlike time_exit_spec).
    entry_window_start_utc: int | None = None,
    entry_window_end_utc: int | None = None,
    # Phase 10 — risk circuit breakers
    max_consecutive_losses: int = 0,      # 0 = disabled
    cooldown_bars_after_loss: int = 0,    # 0 = disabled
    cooldown_bars_after_profit: int = 0,  # 0 = disabled
    # Phase 10 — entry gate controls
    max_spread_bps: float = 0.0,          # 0 = disabled
    gap_filter: str = "none",             # "none" | "ignore_gap_up" | "ignore_gap_down" | "ignore_both"
    gap_threshold_pct: float = 0.5,
    entry_confirmation_bars: int = 1,     # signal must hold N consecutive bars
    rsi_entry_band_min: float | None = None,
    rsi_entry_band_max: float | None = None,
    volume_ratio_threshold: float | None = None,  # volume >= N × 20-bar avg to enter
) -> tuple[list[Trade], list[dict]]:
    """
    Simulate trades on the supplied OHLCV DataFrame.

    Returns:
        (trades, diagnostics)
        trades      — list of Trade objects
        diagnostics — list of per-candle diagnostic dicts
    """
    if stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be greater than zero.")
    if take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be greater than zero.")

    is_intraday = str(objective).lower() == "intraday"

    # Determine which STT rates apply for this strategy type.
    # Intraday equity:  no STT on buy, 0.025% on sell.
    # Delivery equity:  0.1% on both buy and sell.
    stt_entry_pct = 0.0                   if is_intraday else stt_delivery_pct
    stt_exit_pct  = stt_intraday_sell_pct if is_intraday else stt_delivery_pct

    logger.info(
        "Starting trade simulation | objective=%s candles=%s warm_up=%s stop_loss_pct=%.4f "
        "take_profit_pct=%.4f max_holding_candles=%s daily_loss_cap_pct=%.2f "
        "max_trades_per_day=%s slippage_bps=%.2f commission_bps=%.2f "
        "stt_entry_pct=%.4f stt_exit_pct=%.4f entry_mode=%s exit_mode=%s",
        objective, len(df), warm_up_candles,
        stop_loss_pct, take_profit_pct, max_holding_candles,
        daily_loss_cap_pct, max_trades_per_day,
        slippage_bps, commission_bps,
        stt_entry_pct, stt_exit_pct,
        entry_evaluation_mode,
        exit_evaluation_mode,
    )

    # ── Fix 4: extract numpy arrays once, drop pandas inside the bar loop ─────
    # Profiling showed `df.iloc[i]` was 69% of simulator time. Pre-extracting
    # to numpy turns each per-bar lookup into an O(1) memory read.
    n_rows = len(df)
    arrays = build_arrays_from_df(df)
    open_arr  = arrays.get("open",  np.zeros(n_rows))
    high_arr  = arrays.get("high",  np.zeros(n_rows))
    low_arr   = arrays.get("low",   np.zeros(n_rows))
    close_arr = arrays.get("close", np.zeros(n_rows))

    # Cache row timestamps as ISO strings + ordinal day numbers (for session
    # rollover detection). `df.index[i].date()` was another hot pandas call.
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps_iso = df.index.astype(str).to_numpy()
        day_ordinals = df.index.normalize().asi8 // (10**9 * 86400)  # int days since epoch
    else:
        timestamps_iso = np.array([str(x) for x in df.index])
        day_ordinals = np.zeros(n_rows, dtype=np.int64)

    # Boolean mask: True if bar i is the last bar of its session.
    # Vectorised once instead of computing per bar.
    if n_rows > 0:
        session_last_mask = np.empty(n_rows, dtype=bool)
        session_last_mask[:-1] = day_ordinals[:-1] != day_ordinals[1:]
        session_last_mask[-1] = True
    else:
        session_last_mask = np.zeros(0, dtype=bool)

    # Phase 8b — pre-compute each bar's UTC minutes-of-day for the time_exit
    # cutoff. Computed unconditionally so we don't branch in the hot loop;
    # the per-bar check is gated by `time_exit_cutoff_minutes is not None`.
    if isinstance(df.index, pd.DatetimeIndex) and n_rows > 0:
        bar_utc_minutes = (df.index.hour * 60 + df.index.minute).to_numpy(dtype=np.int32)
    else:
        bar_utc_minutes = np.zeros(n_rows, dtype=np.int32)
    time_exit_cutoff_minutes: int | None = (
        int(time_exit_spec["utc_minutes_of_day"])
        if isinstance(time_exit_spec, dict) and "utc_minutes_of_day" in time_exit_spec
        else None
    )

    # Choose evaluator path based on whether the compiled conditions are
    # fast-path-safe (no AVG/MAX/MIN/two-arg-SMA references).
    use_fast_entry = compiled_entry is not None and compiled_entry.fast_path_safe
    use_fast_exit  = compiled_exit  is not None and compiled_exit.fast_path_safe

    trades: list[Trade] = []
    diagnostics: list[CandleDiagnostic] = []

    in_trade    = False
    entry_price = 0.0
    entry_index = -1
    entry_date  = ""

    # Per-trade MAE/MFE tracking (reset on each new entry)
    _trade_min_low:  float = float("inf")   # lowest low seen during holding period
    _trade_max_high: float = float("-inf")  # highest high seen during holding period

    # Per-trade SL state (Phase 3). _initial_stop_price is set on entry from
    # stop_loss_spec; _trailing_floor is the running max of trailing_stop_spec
    # candidates so the floor never moves backward (ratchet). When both specs
    # are absent these stay at sentinel values and the legacy percent SL is
    # used unchanged.
    _initial_stop_price: float = 0.0
    _trailing_floor:     float = float("-inf")

    # Per-day state (for intraday circuit-breakers).
    # Day key is an int ordinal (days since epoch) — way faster than `date` objects.
    _session_cumulative_pnl_pct: float = 0.0
    _session_trades_today: int = 0
    _current_session_ord: int = -1
    _portfolio_balance_factor: float = 1.0

    # Phase 10 — cross-session state for risk circuit breakers
    _consecutive_losses: int = 0        # running count of consecutive losing trades
    _loss_cooldown_until: int = -1      # bar index before which new entries are blocked (after loss)
    _profit_cooldown_until: int = -1    # bar index before which new entries are blocked (after profit)
    # Phase 10 — per-session gap state: if gap_filter blocks a session, mark it so
    # the block applies for the whole session (reset on new session).
    _session_gap_blocked: bool = False
    _session_prev_close: float = 0.0    # close of the last bar of the previous session
    # Phase 10 — confirmation counter: consecutive bars the entry signal has been True
    _signal_consecutive_bars: int = 0

    # Phase 10 — pre-compute volume arrays once to avoid pandas in the hot loop.
    if "volume" in df.columns:
        _vol_arr: np.ndarray | None = df["volume"].astype(float).to_numpy()
        if volume_ratio_threshold is not None:
            _vol_avg_arr: np.ndarray | None = (
                df["volume"].astype(float).rolling(20, min_periods=5).mean().to_numpy()
            )
        else:
            _vol_avg_arr = None
    else:
        _vol_arr = None
        _vol_avg_arr = None

    # Phase 10 — pre-compute RSI_14 array if the RSI band gate is active.
    # Check df for the column added by add_all_indicators.
    _rsi_arr: np.ndarray | None = None
    if rsi_entry_band_min is not None or rsi_entry_band_max is not None:
        rsi_col = next(
            (c for c in df.columns if c.upper().startswith("RSI_")),
            None,
        )
        if rsi_col:
            _rsi_arr = df[rsi_col].to_numpy(dtype=float)

    def _reset_session_state(session_ord: int) -> None:
        nonlocal _session_cumulative_pnl_pct, _session_trades_today, _current_session_ord
        nonlocal _session_gap_blocked
        _session_cumulative_pnl_pct = 0.0
        _session_trades_today       = 0
        _current_session_ord        = session_ord
        _session_gap_blocked        = False  # re-evaluate gap at start of each new session

    def _daily_cap_breached() -> bool:
        """True if today's cumulative realized losses exceed the daily cap."""
        if daily_loss_cap_pct <= 0 or not is_intraday:
            return False
        return _session_cumulative_pnl_pct <= -daily_loss_cap_pct

    def _max_trades_breached() -> bool:
        if max_trades_per_day <= 0 or not is_intraday:
            return False
        return _session_trades_today >= max_trades_per_day

    for i in range(n_rows):
        bar_day_ord = int(day_ordinals[i])
        is_session_last_bar = is_intraday and bool(session_last_mask[i])
        diag = CandleDiagnostic(
            index=i,
            timestamp=str(timestamps_iso[i]),
            in_trade=in_trade,
        )

        # ── Warm-up skip ──────────────────────────────────────────────────────
        # The first `warm_up_candles` bars have NaN indicator values because
        # indicators like SMA(50) need 50 bars to produce their first value.
        # Evaluating entry conditions on these bars would silently fail (NaN
        # comparisons return False), so we skip them explicitly and note it.
        if i < warm_up_candles:
            diag.warm_up_skip = True
            diagnostics.append(diag)
            continue

        # Session rollover (resets intraday circuit-breaker counters at start of new day)
        if is_intraday and bar_day_ord != _current_session_ord:
            # Phase 10 — capture prev-session close for gap filter before resetting
            if _current_session_ord >= 0:
                _session_prev_close = float(close_arr[i - 1]) if i > 0 else 0.0
            _reset_session_state(bar_day_ord)

            # Phase 10 — gap filter: compare today's open to previous session's close
            if gap_filter != "none" and _session_prev_close > 0 and i < n_rows:
                today_open = float(open_arr[i])
                gap_pct = (today_open - _session_prev_close) / _session_prev_close * 100.0
                gap_up   = gap_pct >  gap_threshold_pct
                gap_down = gap_pct < -gap_threshold_pct
                if gap_filter == "ignore_gap_up"   and gap_up:
                    _session_gap_blocked = True
                elif gap_filter == "ignore_gap_down" and gap_down:
                    _session_gap_blocked = True
                elif gap_filter == "ignore_both"    and (gap_up or gap_down):
                    _session_gap_blocked = True

        if not in_trade:
            # Cannot enter a trade on the very last candle (need next bar's open to fill)
            if i >= n_rows - 1:
                diagnostics.append(diag)
                break

            # Evaluate entry condition at this candle
            diag.entry_evaluated = True
            if entry_evaluation_mode == "registry":
                if not KB_REGISTRY_AVAILABLE:
                    raise RuntimeError(
                        "Strategy requires KB registry evaluation but stretus_kb could not be loaded."
                    )
                entry_signal = evaluate_kb_signal_rules(entry_signal_rules or [], df, i)
            elif use_fast_entry:
                entry_signal = compiled_entry.evaluate_arrays(arrays, n_rows, i)
            elif compiled_entry is not None:
                entry_signal = compiled_entry.evaluate(df, i)
            else:
                entry_signal = evaluate_condition(entry_condition, df, i)
            diag.entry_signal = entry_signal

            # Phase 10 — confirmation counter: track consecutive bars with entry_signal True
            if entry_signal:
                _signal_consecutive_bars += 1
            else:
                _signal_consecutive_bars = 0

            # Signal fired but hasn't held long enough yet — flag and skip gate checks
            if entry_signal and _signal_consecutive_bars < entry_confirmation_bars:
                diag.entry_blocked_confirmation = True
                diagnostics.append(diag)
                continue

            if entry_signal and _signal_consecutive_bars >= entry_confirmation_bars:
                cap_blocked    = _daily_cap_breached()
                trades_blocked = _max_trades_breached()
                # Phase 5: every HTF gate must pass on its most recently
                # closed bar. This is how we enforce "daily trend bullish AND
                # 1h trend bullish AND 15m entry trigger" without leaking
                # future HTF data into the LTF decision.
                htf_blocked = bool(htf_contexts) and not all_htf_gates_pass(htf_contexts, i)
                # Phase 8b: block new entries past the time_exit cutoff. No
                # point filling a trade that would immediately be force-closed.
                time_exit_blocked = (
                    time_exit_cutoff_minutes is not None
                    and int(bar_utc_minutes[i]) >= time_exit_cutoff_minutes
                )
                # Phase 10: entry window gate (block outside defined trading window)
                entry_window_blocked = False
                if entry_window_start_utc is not None or entry_window_end_utc is not None:
                    bar_min = int(bar_utc_minutes[i])
                    if entry_window_start_utc is not None and bar_min < entry_window_start_utc:
                        entry_window_blocked = True
                    if entry_window_end_utc is not None and bar_min > entry_window_end_utc:
                        entry_window_blocked = True

                # Phase 10: consecutive-loss circuit breaker
                consec_loss_blocked = (
                    max_consecutive_losses > 0
                    and _consecutive_losses >= max_consecutive_losses
                )
                # Phase 10: cooldown gates
                cooldown_blocked = (
                    i < _loss_cooldown_until or i < _profit_cooldown_until
                )
                # Phase 10: spread gate — estimate spread as (high - low) / close * 10000
                spread_blocked = False
                if max_spread_bps > 0:
                    bar_close = float(close_arr[i])
                    if bar_close > 0:
                        estimated_spread_bps = (float(high_arr[i]) - float(low_arr[i])) / bar_close * 10_000.0
                        spread_blocked = estimated_spread_bps > max_spread_bps

                # Phase 10: gap filter gate
                gap_blocked = _session_gap_blocked

                # Phase 10: RSI band gate
                rsi_band_blocked = False
                if _rsi_arr is not None:
                    rsi_val = float(_rsi_arr[i]) if not np.isnan(_rsi_arr[i]) else None
                    if rsi_val is not None:
                        if rsi_entry_band_min is not None and rsi_val < rsi_entry_band_min:
                            rsi_band_blocked = True
                        if rsi_entry_band_max is not None and rsi_val > rsi_entry_band_max:
                            rsi_band_blocked = True

                # Phase 10: volume ratio gate
                vol_ratio_blocked = False
                if _vol_arr is not None and _vol_avg_arr is not None and volume_ratio_threshold is not None:
                    avg_vol = float(_vol_avg_arr[i])
                    if avg_vol > 0:
                        current_vol = float(_vol_arr[i])
                        if current_vol < volume_ratio_threshold * avg_vol:
                            vol_ratio_blocked = True

                diag.entry_blocked_daily_cap          = cap_blocked
                diag.entry_blocked_max_trades         = trades_blocked
                diag.entry_blocked_htf                = htf_blocked
                diag.entry_blocked_time_exit          = time_exit_blocked
                diag.entry_blocked_entry_window       = entry_window_blocked
                diag.entry_blocked_consecutive_losses = consec_loss_blocked
                diag.entry_blocked_cooldown           = cooldown_blocked
                diag.entry_blocked_spread             = spread_blocked
                diag.entry_blocked_gap                = gap_blocked
                diag.entry_blocked_rsi_band           = rsi_band_blocked
                diag.entry_blocked_volume_ratio       = vol_ratio_blocked

                # Intraday entries must be fillable within the same session.
                if (not cap_blocked and not trades_blocked and not htf_blocked
                        and not time_exit_blocked and not entry_window_blocked
                        and not consec_loss_blocked and not cooldown_blocked
                        and not spread_blocked and not gap_blocked
                        and not rsi_band_blocked and not vol_ratio_blocked
                        and not is_session_last_bar):
                    # Fill at next bar's open price (realistic execution)
                    next_open = float(open_arr[i + 1])
                    entry_price = _apply_entry_costs(
                        next_open, slippage_bps, commission_bps, stt_entry_pct
                    )
                    entry_index = i + 1
                    entry_date  = str(timestamps_iso[i + 1])
                    in_trade    = True
                    diag.in_trade = True
                    # Reset MAE/MFE accumulators for this new trade
                    _trade_min_low  = next_open
                    _trade_max_high = next_open
                    # Phase 3: lock in the initial stop and reset trailing.
                    # Computed once here at entry so we don't re-evaluate
                    # opening_range / ATR-at-entry on every bar.
                    _initial_stop_price = _compute_initial_stop_long(
                        stop_loss_spec,
                        fallback_pct=stop_loss_pct,
                        entry_price=entry_price,
                        entry_index=entry_index,
                        arrays=arrays,
                        low_arr=low_arr,
                        day_ordinals=day_ordinals,
                    )
                    _trailing_floor = float("-inf")
                    logger.debug(
                        "Entered trade | symbol=%s signal_index=%s entry_index=%s entry_price=%.4f initial_stop=%.4f",
                        symbol, i, entry_index, entry_price, _initial_stop_price,
                    )

            diagnostics.append(diag)
            continue

        if i < entry_index:
            diagnostics.append(diag)
            continue

        # ── In-trade: check exits ─────────────────────────────────────────────
        high_price  = float(high_arr[i])
        low_price   = float(low_arr[i])
        close_price = float(close_arr[i])
        holding_candles = i - entry_index

        # Update running MAE/MFE accumulators every bar
        if low_price  < _trade_min_low:  _trade_min_low  = low_price
        if high_price > _trade_max_high: _trade_max_high = high_price

        # Phase 3 — effective stop is the higher of the initial SL (locked at
        # entry) and the trailing floor (running max since entry, only when
        # a trailing spec is configured). Ratchet means the floor never moves
        # backwards even if subsequent candles are lower.
        candidate_floor = _compute_trailing_floor_long(
            trailing_stop_spec,
            entry_price=entry_price,
            current_index=i,
            arrays=arrays,
            highest_high_since_entry=_trade_max_high,
            current_close=close_price,
        )
        if candidate_floor > _trailing_floor:
            _trailing_floor = candidate_floor

        # Effective stop is the higher of the initial SL and the trailing floor.
        # Trailing-floor candidates use intra-bar highs, so the floor can
        # legitimately sit between the current bar's low and the bar's high —
        # in that case the stop fires *this* bar at the floor price (worst-case
        # fill, mirrors the static-SL code path below).
        stop_price = max(_initial_stop_price, _trailing_floor)
        take_profit_price = entry_price * (1 + take_profit_pct / 100.0)

        stop_hit        = low_price  <= stop_price
        take_profit_hit = high_price >= take_profit_price
        # A trailing-stop exit and a static-SL exit are functionally the same
        # mechanic, but the user wants to know which one fired in the report.
        is_trailing_exit = stop_hit and _trailing_floor > _initial_stop_price

        diag.stop_hit = stop_hit
        diag.tp_hit   = take_profit_hit

        exit_price:  float | None = None
        exit_reason: str   | None = None
        exit_date = timestamps_iso[i]

        # ── Worst-case rule: if both stop AND TP touched on the same bar ──────
        # We assume the stop was hit first (conservative for the trader).
        if stop_hit and take_profit_hit:
            exit_price  = _apply_exit_costs(stop_price, slippage_bps, commission_bps, stt_exit_pct)
            exit_reason = "TRAILING_STOP_AND_TAKE_PROFIT_SAME_BAR" if is_trailing_exit else "STOP_LOSS_AND_TAKE_PROFIT_SAME_BAR"

        elif stop_hit:
            exit_price  = _apply_exit_costs(stop_price, slippage_bps, commission_bps, stt_exit_pct)
            exit_reason = "TRAILING_STOP" if is_trailing_exit else "STOP_LOSS"

        elif take_profit_hit:
            exit_price  = _apply_exit_costs(take_profit_price, slippage_bps, commission_bps, stt_exit_pct)
            exit_reason = "TAKE_PROFIT"

        else:
            # Check signal-based exit (formula with PROFIT/LOSS vars, or KB registry)
            exit_signal = False
            if exit_evaluation_mode == "registry":
                if exit_signal_rules:
                    if not KB_REGISTRY_AVAILABLE:
                        raise RuntimeError(
                            "Strategy requires KB registry evaluation but stretus_kb could not be loaded."
                        )
                    diag.exit_evaluated = True
                    exit_signal = evaluate_kb_signal_rules(exit_signal_rules, df, i)
                    diag.exit_signal = exit_signal
            elif exit_condition or compiled_exit is not None:
                variables = _trade_variables(entry_price, close_price, stop_loss_pct, take_profit_pct)
                diag.exit_evaluated = True
                if use_fast_exit:
                    exit_signal = compiled_exit.evaluate_arrays(arrays, n_rows, i, variables=variables)
                elif compiled_exit is not None:
                    exit_signal = compiled_exit.evaluate(df, i, variables=variables)
                else:
                    exit_signal = evaluate_condition(exit_condition, df, i, variables=variables)
                diag.exit_signal = exit_signal

            if exit_signal:
                if is_session_last_bar or i + 1 >= n_rows:
                    exit_price = _apply_exit_costs(
                        close_price, slippage_bps, commission_bps, stt_exit_pct
                    )
                    exit_date = timestamps_iso[i]
                else:
                    exit_price = _apply_exit_costs(
                        float(open_arr[i + 1]), slippage_bps, commission_bps, stt_exit_pct
                    )
                    exit_date = timestamps_iso[i + 1]
                exit_reason = "EXIT_SIGNAL"

            # Check max holding candles (time-based stop)
            if exit_price is None and max_holding_candles is not None and holding_candles >= max_holding_candles:
                if is_session_last_bar or i + 1 >= n_rows:
                    exit_price = _apply_exit_costs(
                        close_price, slippage_bps, commission_bps, stt_exit_pct
                    )
                    exit_date = timestamps_iso[i]
                else:
                    exit_price = _apply_exit_costs(
                        float(open_arr[i + 1]), slippage_bps, commission_bps, stt_exit_pct
                    )
                    exit_date = timestamps_iso[i + 1]
                exit_reason = "MAX_HOLDING"

            # Phase 8b — wall-clock cutoff. The FIRST bar whose timestamp
            # crosses the cutoff is the exit bar. We use this bar's close
            # (rather than next bar's open) so the cutoff is honored
            # strictly even when the bar at the cutoff is the last bar of
            # the data window. Conservative same-as-SESSION_END convention.
            if (
                exit_price is None
                and time_exit_cutoff_minutes is not None
                and int(bar_utc_minutes[i]) >= time_exit_cutoff_minutes
            ):
                exit_price = _apply_exit_costs(
                    close_price, slippage_bps, commission_bps, stt_exit_pct
                )
                exit_date = timestamps_iso[i]
                exit_reason = "TIME_EXIT"

            if exit_price is None and is_session_last_bar:
                exit_price = _apply_exit_costs(
                    close_price, slippage_bps, commission_bps, stt_exit_pct
                )
                exit_date = timestamps_iso[i]
                exit_reason = "SESSION_END"

        # ── Book the trade ───────────────────────────────────────────────────
        if exit_price is not None and exit_reason is not None:
            pnl_inr = exit_price - entry_price          # absolute per-unit P&L in ₹
            pnl_abs = pnl_inr / entry_price             # fractional return (for compounding)
            pnl_pct = pnl_abs * 100.0

            # MAE: worst unrealised loss = (min_low - entry) / entry * 100  (≤ 0 for long)
            # MFE: best unrealised gain  = (max_high - entry) / entry * 100 (≥ 0 for long)
            mae_pct = ((_trade_min_low  - entry_price) / entry_price * 100.0) if entry_price else 0.0
            mfe_pct = ((_trade_max_high - entry_price) / entry_price * 100.0) if entry_price else 0.0

            trades.append(Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                symbol=symbol,
                side="LONG",
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                pnl_pct=round(pnl_pct, 4),
                pnl_abs=round(pnl_abs, 6),
                pnl_inr=round(pnl_inr, 4),
                exit_reason=exit_reason,
                holding_candles=max(holding_candles, 0),
                mae_pct=round(mae_pct, 4),
                mfe_pct=round(mfe_pct, 4),
            ))

            logger.debug(
                "Exited trade | symbol=%s exit_reason=%s exit_price=%.4f pnl_pct=%.4f holding_candles=%s",
                symbol, exit_reason, exit_price, pnl_pct, holding_candles,
            )

            if is_intraday:
                _session_cumulative_pnl_pct += pnl_pct
                _session_trades_today       += 1

            _portfolio_balance_factor *= (1.0 + pnl_abs)

            # Phase 10 — update consecutive loss counter + cooldown timers
            trade_was_loss = pnl_pct < 0
            if trade_was_loss:
                _consecutive_losses += 1
                if cooldown_bars_after_loss > 0:
                    _loss_cooldown_until = i + cooldown_bars_after_loss
            else:
                _consecutive_losses = 0
                if cooldown_bars_after_profit > 0:
                    _profit_cooldown_until = i + cooldown_bars_after_profit

            # Reset confirmation counter on trade exit so next entry needs fresh confirmation
            _signal_consecutive_bars = 0

            in_trade    = False
            entry_price = 0.0
            entry_index = -1
            entry_date  = ""

        diagnostics.append(diag)

    # ── Force-close open trade at end of data ─────────────────────────────────
    if in_trade and entry_index >= 0:
        last_close = _apply_exit_costs(
            float(close_arr[-1]), slippage_bps, commission_bps, stt_exit_pct
        )
        pnl_inr = last_close - entry_price
        pnl_abs = pnl_inr / entry_price
        pnl_pct = pnl_abs * 100.0
        mae_pct = ((_trade_min_low  - entry_price) / entry_price * 100.0) if entry_price else 0.0
        mfe_pct = ((_trade_max_high - entry_price) / entry_price * 100.0) if entry_price else 0.0
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=str(timestamps_iso[-1]),
            symbol=symbol,
            side="LONG",
            entry_price=round(entry_price, 4),
            exit_price=round(last_close, 4),
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(pnl_abs, 6),
            pnl_inr=round(pnl_inr, 4),
            exit_reason="END_OF_DATA",
            holding_candles=max(n_rows - 1 - entry_index, 0),
            mae_pct=round(mae_pct, 4),
            mfe_pct=round(mfe_pct, 4),
        ))
        logger.debug("Force-closed open trade at end of data | symbol=%s pnl_pct=%.4f", symbol, pnl_pct)

    diag_summary = _summarise_diagnostics(diagnostics)
    logger.info(
        "Trade simulation complete | trades=%s warm_up_skips=%s entry_signals=%s "
        "daily_cap_blocks=%s max_trades_blocks=%s",
        len(trades),
        diag_summary["warm_up_skips"],
        diag_summary["entry_signals"],
        diag_summary["daily_cap_blocks"],
        diag_summary["max_trades_blocks"],
    )

    diag_dicts = [d.to_dict() for d in diagnostics]
    return trades, diag_dicts


def _summarise_diagnostics(diagnostics: list[CandleDiagnostic]) -> dict[str, int]:
    return {
        "total_candles":         len(diagnostics),
        "warm_up_skips":         sum(1 for d in diagnostics if d.warm_up_skip),
        "entry_signals":         sum(1 for d in diagnostics if d.entry_signal),
        "entry_blocked_daily_cap":   sum(1 for d in diagnostics if d.entry_blocked_daily_cap),
        "entry_blocked_max_trades":  sum(1 for d in diagnostics if d.entry_blocked_max_trades),
        "entry_blocked_htf":     sum(1 for d in diagnostics if d.entry_blocked_htf),
        "entry_blocked_time_exit": sum(1 for d in diagnostics if d.entry_blocked_time_exit),
        "daily_cap_blocks":      sum(1 for d in diagnostics if d.entry_blocked_daily_cap),
        "max_trades_blocks":     sum(1 for d in diagnostics if d.entry_blocked_max_trades),
        "htf_blocks":            sum(1 for d in diagnostics if d.entry_blocked_htf),
        "time_exit_blocks":      sum(1 for d in diagnostics if d.entry_blocked_time_exit),
        "exit_signals":          sum(1 for d in diagnostics if d.exit_signal),
        "stop_hits":             sum(1 for d in diagnostics if d.stop_hit),
        "tp_hits":               sum(1 for d in diagnostics if d.tp_hit),
    }
