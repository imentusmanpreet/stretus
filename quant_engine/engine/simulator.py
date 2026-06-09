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

from collections import Counter
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


# Number of bars to pause new entries after the consecutive-loss streak hits
# `max_consecutive_losses`. Intentionally hardcoded for now (one source of truth).
# The bar-count basis makes the pause scale naturally with each strategy's
# timeframe:
#     5m  × 3 = 15-minute cooling-off
#     15m × 3 = 45-minute cooling-off
#     1h  × 3 = 3-hour cooling-off
#     1d  × 3 = 3 trading days
# When the pause expires the streak counter resumes from zero, so the strategy
# resumes with a clean slate rather than being permanently locked out.
_CONSEC_LOSS_PAUSE_BARS = 3


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
    exit_reason: str        # short machine code, e.g. STOP_LOSS / TAKE_PROFIT / EXIT_SIGNAL
    holding_candles: int
    # Maximum Adverse Excursion: worst unrealised loss during the trade (always ≤ 0 for long)
    mae_pct: float = 0.0
    # Maximum Favorable Excursion: best unrealised gain during the trade (always ≥ 0 for long)
    mfe_pct: float = 0.0
    # Structured, human-readable rationale captured at the moment of entry/exit.
    # entry_reason explains WHY the trade opened (signal bar, matched indicators,
    # fill price, costs, initial stop). exit_reason_detail explains WHY it closed
    # (trigger bar, trigger price, fill, P&L) and embeds the short `exit_reason`
    # code under its "code" key. Both are None only for Trades built outside the
    # simulator (e.g. unit-test fixtures).
    entry_reason: dict[str, Any] | None = None
    exit_reason_detail: dict[str, Any] | None = None


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
    entry_blocked_volatility: bool = False    # Phase 2: outside the volatility band
    entry_blocked_regime: bool = False        # Phase 2: regime not in allow-list
    entry_blocked_relative_strength: bool = False  # Phase 2: underperforming reference
    entry_blocked_event: bool = False         # Phase 2: configured event/blackout date
    entry_blocked_lunch_lull: bool = False    # Phase 2: inside the midday lull window
    # All gates cleared but the entry was rejected because it couldn't be filled
    # within the same session (intraday objective on a bar that ends its session,
    # e.g. an intraday strategy on a 1d timeframe → every bar triggers this).
    entry_blocked_session_last: bool = False
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
            "entry_blocked_volatility":        self.entry_blocked_volatility,
            "entry_blocked_regime":            self.entry_blocked_regime,
            "entry_blocked_relative_strength": self.entry_blocked_relative_strength,
            "entry_blocked_event":             self.entry_blocked_event,
            "entry_blocked_lunch_lull":        self.entry_blocked_lunch_lull,
            "entry_blocked_session_last":      self.entry_blocked_session_last,
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
    side: str = "LONG",
) -> dict[str, float]:
    raw = ((current_close - entry_price) / entry_price) * 100.0
    # A short profits when price falls, so its P&L is the negation of the long
    # mark-to-market — flip before exposing PROFIT/LOSS to exit formulas.
    profit_pct = -raw if str(side).upper() == "SHORT" else raw
    loss_pct   = -profit_pct
    return {
        "PROFIT":             profit_pct,
        "LOSS":               loss_pct,
        "TAKE_PROFIT_TARGET": take_profit_pct,
        "STOP_LOSS_TARGET":   stop_loss_pct,
    }


# ─── Trade-reason capture helpers ───────────────────────────────────────────────
#
# These build the structured `entry_reason` / `exit_reason_detail` payloads that
# accompany every Trade so the report can explain *exactly* why each trade opened
# and closed: which bar, at what time, what price, which indicator values matched,
# and what costs applied. They run once per trade (not per bar), so the captured
# detail is essentially free relative to the bar loop.

def _arr_val(
    arrays: dict[str, np.ndarray], key: str, i: int, ndigits: int = 6
) -> float | None:
    """Value of array `key` at bar `i`, rounded; None if missing/out-of-range/NaN."""
    arr = arrays.get(key)
    if arr is None or i < 0 or i >= len(arr):
        return None
    value = float(arr[i])
    if value != value:  # NaN
        return None
    return round(value, ndigits)


def _snapshot_bar(
    arrays: dict[str, np.ndarray], i: int, timestamps_iso: np.ndarray
) -> dict[str, Any]:
    """OHLCV + timestamp at bar `i` — the raw market data the decision saw."""
    snap: dict[str, Any] = {
        "index":     int(i),
        "timestamp": str(timestamps_iso[i]) if 0 <= i < len(timestamps_iso) else None,
        "open":      _arr_val(arrays, "open",  i, 4),
        "high":      _arr_val(arrays, "high",  i, 4),
        "low":       _arr_val(arrays, "low",   i, 4),
        "close":     _arr_val(arrays, "close", i, 4),
    }
    volume = _arr_val(arrays, "volume", i, 2)
    if volume is not None:
        snap["volume"] = volume
    return snap


def _snapshot_indicators(
    arrays: dict[str, np.ndarray],
    i: int,
    compiled: CompiledCondition | None,
) -> dict[str, float | None]:
    """Value at bar `i` of every indicator / scalar / pattern column the condition
    references — i.e. the exact numbers that made the formula evaluate True.

    Empty when the condition isn't compiled (registry path or a raw string passed
    straight to the simulator), since we can't enumerate referenced columns then.
    """
    if compiled is None:
        return {}
    out: dict[str, float | None] = {}
    for ref in compiled.indicator_refs:          # e.g. RSI_14, EMA_20, BB_UPPER_20
        out[ref.column] = _arr_val(arrays, ref.column, i)
    for name in compiled.scalar_refs:            # MACD, MACD_SIGNAL, VWAP
        out[name] = _arr_val(arrays, name, i)
    for name in compiled.pattern_refs:           # IS_SWING_HIGH, IS_BOS_BULLISH, ...
        out[name] = _arr_val(arrays, name, i)
    return out


def _fmt_indicator_values(indicators: dict[str, float | None]) -> str:
    """Compact 'RSI_14=28.3, EMA_20=99.5' rendering for narrative summaries."""
    return ", ".join(f"{k}={v}" for k, v in indicators.items() if v is not None)


def _fmt_breakdown(counts: dict[str, int]) -> str:
    """'STOP_LOSS=61 TAKE_PROFIT=58' — only non-zero entries; 'none' when all zero."""
    rendered = " ".join(f"{k}={v}" for k, v in counts.items() if v)
    return rendered or "none"


def _condition_text(
    evaluation_mode: str,
    compiled: CompiledCondition | None,
    raw_condition: str,
) -> str:
    """Human-readable description of the active entry/exit condition."""
    if evaluation_mode == "registry":
        return "KB registry rules"
    if compiled is not None:
        return compiled.raw
    return raw_condition or ""


def _exit_trigger_clause(exit_reason: str, trigger: dict[str, Any], side: str = "LONG") -> str:
    """Plain-language clause explaining what fired the exit, for the summary."""
    is_short = str(side).upper() == "SHORT"
    # A short is stopped when price RISES (bar high breaches a stop above) and
    # takes profit when price FALLS (bar low reaches a target below) — the
    # mirror of a long. Pick the bar extreme that actually triggered.
    stop_extreme = ("bar high", trigger.get("bar_high")) if is_short else ("bar low", trigger.get("bar_low"))
    tp_extreme   = ("bar low",  trigger.get("bar_low"))  if is_short else ("bar high", trigger.get("bar_high"))
    ttype = trigger.get("type")
    if ttype in {"stop_loss", "trailing_stop"}:
        label = "trailing stop" if ttype == "trailing_stop" else "stop loss"
        return (
            f"{stop_extreme[0]} {stop_extreme[1]} breached the {label} at "
            f"{trigger.get('stop_price')}"
        )
    if ttype == "stop_and_take_profit_same_bar":
        return (
            f"both the stop ({trigger.get('stop_price')}) and the target "
            f"({trigger.get('take_profit_price')}) were touched on the same bar; "
            "the stop is assumed to fill first (pessimistic)"
        )
    if ttype == "take_profit":
        return (
            f"{tp_extreme[0]} {tp_extreme[1]} reached the take-profit target at "
            f"{trigger.get('take_profit_price')}"
        )
    if ttype == "exit_signal":
        return f"exit signal [{trigger.get('condition')}] became true"
    if ttype == "max_holding":
        return (
            f"the trade reached the max-holding limit of "
            f"{trigger.get('max_holding_candles')} candle(s)"
        )
    if ttype == "time_exit":
        return (
            f"the bar's time ({trigger.get('bar_utc_minutes')} min UTC) reached the "
            f"configured cutoff ({trigger.get('cutoff_utc_minutes')} min UTC)"
        )
    if ttype == "session_end":
        return "the intraday session ended (square-off on the last bar of the day)"
    if ttype == "end_of_data":
        return "the backtest data window ended while the position was still open"
    return exit_reason


def _build_exit_reason_detail(
    *,
    exit_reason: str,
    exit_trigger: dict[str, Any] | None,
    exit_bar: dict[str, Any],
    fill_timestamp: str,
    raw_exit_price: float,
    effective_exit_price: float,
    slippage_bps: float,
    commission_bps: float,
    stt_exit_pct: float,
    holding_candles: int,
    pnl_pct: float,
    pnl_inr: float,
    exit_indicators: dict[str, float | None],
    side: str = "LONG",
) -> dict[str, Any]:
    """Assemble the structured exit rationale (narrative + machine-readable detail)."""
    trigger = exit_trigger or {"type": exit_reason.lower()}
    clause = _exit_trigger_clause(exit_reason, trigger, side)
    fill_basis = trigger.get("fill_basis", "current_close")
    summary = (
        f"Exited {str(side).upper()} after holding {holding_candles} candle(s). "
        f"Reason: {exit_reason} — {clause} on bar #{exit_bar.get('index')} "
        f"({exit_bar.get('timestamp')}). "
        f"Filled at {effective_exit_price:.4f} on {fill_timestamp} ({fill_basis}), "
        f"from raw price {raw_exit_price:.4f} minus costs "
        f"(slippage {slippage_bps:.1f}bps, commission {commission_bps:.1f}bps, "
        f"STT {stt_exit_pct:.3f}%). "
        f"Net P&L {pnl_pct:+.2f}% ({pnl_inr:+.4f} per unit)."
    )
    return {
        "summary":  summary,
        "code":     exit_reason,
        "trigger":  trigger,
        "exit_bar": exit_bar,
        "fill": {
            "timestamp":       fill_timestamp,
            "raw_price":       round(raw_exit_price, 6),
            "effective_price": round(effective_exit_price, 6),
            "slippage_bps":    float(slippage_bps),
            "commission_bps":  float(commission_bps),
            "stt_pct":         float(stt_exit_pct),
        },
        "holding_candles": int(holding_candles),
        "pnl_pct":         round(pnl_pct, 4),
        "pnl_inr":         round(pnl_inr, 4),
        "indicators":      exit_indicators,
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


def _compute_initial_stop_short(
    spec: dict | None,
    *,
    fallback_pct: float,
    entry_price: float,
    entry_index: int,
    arrays: dict[str, np.ndarray],
    high_arr: np.ndarray,
    day_ordinals: np.ndarray,
) -> float:
    """Return the initial stop price for a SHORT entry — the mirror of
    _compute_initial_stop_long. The stop sits ABOVE the entry (a short is
    stopped out when price RISES). NaN-safe; falls back to the legacy
    percent-based SL above entry if the spec can't be resolved."""
    if spec is None:
        return entry_price * (1.0 + fallback_pct / 100.0)

    sl_type = spec.get("type")

    if sl_type == "percent":
        return entry_price * (1.0 + float(spec["pct"]) / 100.0)

    if sl_type == "structural":
        anchor = spec.get("anchor")
        padding_pct = float(spec.get("padding_pct", 0.0))
        anchor_price: float
        if anchor == "opening_range_high":
            anchor_price = _session_open_high(
                high_arr, day_ordinals, entry_index, int(spec.get("opening_bars", 3)),
            )
        elif anchor == "opening_range_low":
            # Downside anchor is nonsensical for a short stop (must sit above
            # entry) — degrade to the % fallback above entry.
            return entry_price * (1.0 + fallback_pct / 100.0)
        elif anchor == "prev_n_bar_high":
            window = int(spec.get("window", 5))
            start = max(0, entry_index - window)
            anchor_price = float(np.max(high_arr[start:entry_index])) if entry_index > start else float(high_arr[entry_index])
        elif anchor == "prev_n_bar_low":
            return entry_price * (1.0 + fallback_pct / 100.0)
        else:
            return entry_price * (1.0 + fallback_pct / 100.0)
        # padding pushes the SL slightly *above* the anchor so a wick to the
        # level doesn't trigger an exit (mirror of the long padding).
        return float(anchor_price) * (1.0 + padding_pct / 100.0)

    if sl_type == "atr":
        window = int(spec["window"])
        atr_col = arrays.get(f"ATR_{window}")
        if atr_col is None or entry_index >= len(atr_col):
            return entry_price * (1.0 + fallback_pct / 100.0)
        atr_value = float(atr_col[entry_index])
        if atr_value != atr_value or atr_value <= 0:        # NaN check
            return entry_price * (1.0 + fallback_pct / 100.0)
        return entry_price + float(spec["multiplier"]) * atr_value

    return entry_price * (1.0 + fallback_pct / 100.0)


def _compute_trailing_ceiling_short(
    spec: dict | None,
    *,
    entry_price: float,
    current_index: int,
    arrays: dict[str, np.ndarray],
    lowest_low_since_entry: float,
    current_close: float,
) -> float:
    """Return today's trailing-ceiling candidate for a SHORT trade — the mirror
    of _compute_trailing_floor_long. Returns +inf when trailing isn't applicable
    yet so it can't override the initial SL (the caller takes min(initial, this)).
    """
    if spec is None:
        return float("inf")

    activate = float(spec.get("activate_after_pct", 0.0))
    if activate > 0.0:
        gain_pct = ((entry_price - current_close) / entry_price) * 100.0
        if gain_pct < activate:
            return float("inf")

    ts_type = spec.get("type")

    if ts_type == "percent":
        return lowest_low_since_entry * (1.0 + float(spec["distance_pct"]) / 100.0)

    if ts_type in {"atr", "chandelier"}:
        window = int(spec["window"])
        atr_col = arrays.get(f"ATR_{window}")
        if atr_col is None or current_index >= len(atr_col):
            return float("inf")
        atr_value = float(atr_col[current_index])
        if atr_value != atr_value or atr_value <= 0:
            return float("inf")
        return lowest_low_since_entry + float(spec["multiplier"]) * atr_value

    if ts_type == "ema":
        window = int(spec["window"])
        ema_col = arrays.get(f"EMA_{window}")
        if ema_col is None or current_index >= len(ema_col):
            return float("inf")
        ema_value = float(ema_col[current_index])
        if ema_value != ema_value:
            return float("inf")
        return ema_value

    return float("inf")


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


# ─── 1-minute execution helper (Phase 11) ──────────────────────────────────────

def build_minute_arrays(df_1m: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract the numpy arrays the simulator's 1-minute walk needs.

    Returns {"open","high","low","close"} float arrays plus a "timestamp"
    array of ISO strings, mirroring how the strategy-bar arrays are prepared
    inside simulate_trades. Pairs with engine.resample.build_subbar_slices.
    """
    n = len(df_1m)
    arrays = build_arrays_from_df(df_1m)
    out: dict[str, np.ndarray] = {
        "open":  arrays.get("open",  np.zeros(n)),
        "high":  arrays.get("high",  np.zeros(n)),
        "low":   arrays.get("low",   np.zeros(n)),
        "close": arrays.get("close", np.zeros(n)),
    }
    if isinstance(df_1m.index, pd.DatetimeIndex):
        out["timestamp"] = df_1m.index.astype(str).to_numpy()
    else:
        out["timestamp"] = np.array([str(x) for x in df_1m.index])
    return out


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
    # Trade direction for THIS pass. "LONG" (default) preserves the legacy
    # behaviour exactly. "SHORT" mirrors every direction-dependent mechanic:
    # the stop sits above entry, the target below, P&L = entry - exit, MAE/MFE
    # flip, and entry/exit costs swap buy/sell side. The runner runs one pass
    # per active direction (a "both" strategy = a LONG pass + a SHORT pass).
    side: str = "LONG",
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
    # Phase 2 — volatility-band gate. Reads ATR_{w}/NATR_{w}; blocks entries when
    # volatility is below vol_filter_min (dead market) or above vol_filter_max
    # (chaotic). Causal: ATR/NATR at bar i use only bars ≤ i.
    vol_filter_metric: str | None = None,          # "atr" | "natr" | None (disabled)
    vol_filter_window: int = 14,
    vol_filter_min: float | None = None,
    vol_filter_max: float | None = None,
    # Phase 2 — regime gate. Only enter when the causally-detected regime is in
    # this allow-list. Empty/None = disabled. Causal: each bar's regime uses only
    # bars ≤ i (classify_regime_series).
    regime_filter_allowed: tuple[str, ...] | None = None,
    # Phase 2 — relative-strength gate. Block unless the symbol outperforms its
    # reference over rs_filter_window bars (RS ratio > rs_filter_min_ratio).
    # Needs REF_close (reference_symbol merged in). None = disabled.
    rs_filter_window: int | None = None,
    rs_filter_min_ratio: float = 1.0,
    # Phase 2 — event filter. Skip new entries whose bar date (UTC) is in this
    # set of "YYYY-MM-DD" strings. Empty/None = disabled.
    event_skip_dates: tuple[str, ...] | None = None,
    # Phase 2 — lunch-lull skip. Block new entries whose UTC minute-of-day is in
    # [start, end] (inclusive). Both None = disabled.
    lunch_lull_start_utc: int | None = None,
    lunch_lull_end_utc: int | None = None,
    # Phase 11 — 1-minute execution. Signals are still evaluated on `df` (the
    # strategy-timeframe frame), but when these are supplied the price-path
    # exits (stop-loss / take-profit / trailing) are resolved by walking the
    # underlying 1-minute bars *in order* within each strategy bar, so we know
    # which of the stop/target was actually touched first and at what price —
    # instead of guessing from a single coarse candle. All three must be
    # supplied together; when any is None the simulator uses the legacy
    # strategy-bar resolution and behaves identically to before.
    #   minute_arrays  — {"open","high","low","close"} numpy arrays + "timestamp"
    #                    ISO strings for the 1-minute series (see
    #                    build_minute_arrays()).
    #   subbar_starts  — per strategy bar, start index into the minute arrays
    #   subbar_ends    — per strategy bar, end index (half-open) into the arrays
    minute_arrays: dict[str, np.ndarray] | None = None,
    subbar_starts: np.ndarray | None = None,
    subbar_ends: np.ndarray | None = None,
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

    # ── Direction setup ───────────────────────────────────────────────────────
    # For a short, the opening trade is a SELL and the close is a BUY, so the
    # buy/sell cost legs swap relative to a long. _entry_fill/_exit_fill name the
    # cost actually paid at each end of THIS trade regardless of direction, and
    # eff_entry_stt/eff_exit_stt are the STT rates that applied (for the report).
    is_short = str(side).upper() == "SHORT"
    if is_short:
        _entry_fill = lambda p: _apply_exit_costs(p, slippage_bps, commission_bps, stt_exit_pct)
        _exit_fill  = lambda p: _apply_entry_costs(p, slippage_bps, commission_bps, stt_entry_pct)
        eff_entry_stt, eff_exit_stt = stt_exit_pct, stt_entry_pct
    else:
        _entry_fill = lambda p: _apply_entry_costs(p, slippage_bps, commission_bps, stt_entry_pct)
        _exit_fill  = lambda p: _apply_exit_costs(p, slippage_bps, commission_bps, stt_exit_pct)
        eff_entry_stt, eff_exit_stt = stt_entry_pct, stt_exit_pct

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

    # ── Phase 11: 1-minute execution arrays ───────────────────────────────────
    # When the runner supplies the underlying 1-minute series and the per-
    # strategy-bar sub-bar slices, price-path exits are resolved by walking
    # those minutes in order (see the in-trade block below). All three pieces
    # must be present and lengths must line up, otherwise we fall back to the
    # legacy strategy-bar resolution — this keeps the change strictly additive.
    intrabar_execution = (
        minute_arrays is not None
        and subbar_starts is not None
        and subbar_ends is not None
        and len(subbar_starts) == n_rows
        and len(subbar_ends) == n_rows
    )
    if intrabar_execution:
        m_open  = minute_arrays.get("open")
        m_high  = minute_arrays.get("high")
        m_low   = minute_arrays.get("low")
        m_close = minute_arrays.get("close")
        m_ts    = minute_arrays.get("timestamp")
        if any(a is None for a in (m_high, m_low, m_close)):
            raise ValueError(
                "simulate_trades: minute_arrays must contain 'high', 'low' and "
                "'close' for 1-minute execution."
            )
        logger.info(
            "🕐 1-minute execution enabled | minute_bars=%s strategy_bars=%s",
            len(m_close), n_rows,
        )
    else:
        m_open = m_high = m_low = m_close = m_ts = None

    # Choose evaluator path based on whether the compiled conditions are
    # fast-path-safe (no AVG/MAX/MIN/two-arg-SMA references).
    use_fast_entry = compiled_entry is not None and compiled_entry.fast_path_safe
    use_fast_exit  = compiled_exit  is not None and compiled_exit.fast_path_safe

    # Human-readable condition strings, resolved once for reuse in every trade's
    # entry/exit rationale.
    entry_condition_text = _condition_text(entry_evaluation_mode, compiled_entry, entry_condition)
    exit_condition_text  = _condition_text(exit_evaluation_mode, compiled_exit, exit_condition)

    # The set of entry gates that are actually enforced for this run. Because a
    # trade only fills once every active gate passes, listing them in the entry
    # rationale tells the reader which filters the bar cleared.
    active_entry_gates: list[str] = []
    if daily_loss_cap_pct > 0 and is_intraday:
        active_entry_gates.append("daily_loss_cap")
    if max_trades_per_day > 0 and is_intraday:
        active_entry_gates.append("max_trades_per_day")
    if htf_contexts:
        active_entry_gates.append("htf_trend")
    if time_exit_cutoff_minutes is not None:
        active_entry_gates.append("time_exit_cutoff")
    if entry_window_start_utc is not None or entry_window_end_utc is not None:
        active_entry_gates.append("entry_window")
    if max_consecutive_losses > 0:
        active_entry_gates.append("consecutive_loss_pause")
    if cooldown_bars_after_loss > 0 or cooldown_bars_after_profit > 0:
        active_entry_gates.append("cooldown")
    if max_spread_bps > 0:
        active_entry_gates.append("spread")
    if gap_filter != "none":
        active_entry_gates.append("gap_filter")
    if rsi_entry_band_min is not None or rsi_entry_band_max is not None:
        active_entry_gates.append("rsi_band")
    if volume_ratio_threshold is not None:
        active_entry_gates.append("volume_ratio")
    if entry_confirmation_bars > 1:
        active_entry_gates.append("entry_confirmation")

    trades: list[Trade] = []
    diagnostics: list[CandleDiagnostic] = []
    # Running tally of exit reasons for the completion summary log.
    exit_reason_counts: Counter[str] = Counter()

    in_trade    = False
    entry_price = 0.0
    entry_index = -1
    entry_date  = ""

    # Structured entry rationale for the currently-open trade. Captured at fill
    # time and attached to the Trade when it eventually closes (so the order of
    # operations — open now, close many bars later — is preserved).
    entry_reason_payload: dict[str, Any] | None = None

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
    # When the loss streak hits `max_consecutive_losses`, this is set to the
    # bar index at which trading is allowed to resume. The counter itself is
    # reset to 0 at the same moment so the next streak measures from a clean
    # slate after the cooling-off period ends.
    _consec_loss_pause_until: int = -1
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

    # Phase 2 — pre-compute the volatility-band array (ATR_{w} or NATR_{w}). The
    # runner injects this column into the indicator config so it's present; if it
    # somehow isn't, the gate stays disabled (no-raise contract).
    _vol_filter_arr: np.ndarray | None = None
    if vol_filter_metric in ("atr", "natr") and (
        vol_filter_min is not None or vol_filter_max is not None
    ):
        _vf_col = f"{vol_filter_metric.upper()}_{int(vol_filter_window)}"
        if _vf_col in df.columns:
            _vol_filter_arr = df[_vf_col].to_numpy(dtype=float)

    # Phase 2 — regime gate: per-bar causal regime label (computed once).
    _regime_arr: np.ndarray | None = None
    _regime_allowed: set[str] = set()
    if regime_filter_allowed:
        _regime_allowed = {str(r).lower() for r in regime_filter_allowed}
        from engine.regime import classify_regime_series
        _regime_arr = classify_regime_series(df).to_numpy()

    # Phase 2 — relative-strength gate: precompute RS ratio vs reference, once.
    # RS = (close / close[i-w]) / (REF_close / REF_close[i-w]). > min_ratio means
    # the symbol outperformed its benchmark over the window. NaN where reference
    # data or history is missing → gate can't evaluate → does not block.
    _rs_arr: np.ndarray | None = None
    if rs_filter_window and "REF_close" in df.columns:
        w = int(rs_filter_window)
        close_s = df["close"].astype(float)
        ref_s = df["REF_close"].astype(float)
        sym_ret = close_s / close_s.shift(w)
        ref_ret = ref_s / ref_s.shift(w)
        _rs_arr = (sym_ret / ref_ret.replace(0.0, np.nan)).to_numpy(dtype=float)

    # Phase 2 — event filter: per-bar UTC date strings + the skip set.
    _bar_dates: np.ndarray | None = None
    _event_skip: set[str] = set()
    if event_skip_dates and isinstance(df.index, pd.DatetimeIndex):
        _event_skip = {str(d)[:10] for d in event_skip_dates}
        _bar_dates = df.index.strftime("%Y-%m-%d").to_numpy()

    def _reset_session_state(session_ord: int) -> None:
        nonlocal _session_cumulative_pnl_pct, _session_trades_today, _current_session_ord
        nonlocal _session_gap_blocked
        nonlocal _consecutive_losses, _loss_cooldown_until, _profit_cooldown_until
        nonlocal _signal_consecutive_bars, _consec_loss_pause_until
        _session_cumulative_pnl_pct = 0.0
        _session_trades_today       = 0
        _current_session_ord        = session_ord
        _session_gap_blocked        = False  # re-evaluate gap at start of each new session
        # Intraday safety net on top of the bar-pause model: clear all
        # circuit-breaker state at the day boundary so a near-threshold
        # streak (e.g. 2 losses with threshold=3) doesn't lurk into the
        # next session, and any in-flight pause / cooldown bar-indices
        # from late on the previous day can't bleed into today's bars.
        # The pause itself is short (_CONSEC_LOSS_PAUSE_BARS) so it
        # almost always expires inside its own session anyway — this is
        # belt-and-braces. Positional strategies don't hit this branch,
        # so for them the streak and pause persist across the whole run
        # exactly as the bar-pause model intends.
        if is_intraday:
            _consecutive_losses      = 0
            _loss_cooldown_until     = -1
            _profit_cooldown_until   = -1
            _signal_consecutive_bars = 0
            _consec_loss_pause_until = -1

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

                # Phase 10: consecutive-loss circuit breaker — bar-pause model.
                # Trigger and pause length are set when a losing trade pushes
                # _consecutive_losses to the threshold (see trade-booking
                # block below). The gate here only checks whether we are
                # still inside the pause window.
                consec_loss_blocked = (
                    max_consecutive_losses > 0
                    and i < _consec_loss_pause_until
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

                # Phase 2: volatility-band gate — block dead/chaotic conditions.
                volatility_blocked = False
                if _vol_filter_arr is not None:
                    vol_val = float(_vol_filter_arr[i])
                    if vol_val == vol_val:  # not NaN (warm-up safe)
                        if vol_filter_min is not None and vol_val < vol_filter_min:
                            volatility_blocked = True
                        if vol_filter_max is not None and vol_val > vol_filter_max:
                            volatility_blocked = True

                # Phase 2: regime gate — block when detected regime not allowed.
                regime_blocked = False
                if _regime_arr is not None:
                    regime_blocked = str(_regime_arr[i]) not in _regime_allowed

                # Phase 2: relative-strength gate — block when underperforming ref.
                rs_blocked = False
                if _rs_arr is not None:
                    rs_val = float(_rs_arr[i])
                    if rs_val == rs_val:  # not NaN (else can't evaluate → don't block)
                        rs_blocked = rs_val < rs_filter_min_ratio

                # Phase 2: event filter — block new entries on blackout dates.
                event_blocked = False
                if _bar_dates is not None:
                    event_blocked = str(_bar_dates[i]) in _event_skip

                # Phase 2: lunch-lull — block entries inside the midday window.
                lunch_lull_blocked = False
                if lunch_lull_start_utc is not None and lunch_lull_end_utc is not None:
                    _bm = int(bar_utc_minutes[i])
                    lunch_lull_blocked = lunch_lull_start_utc <= _bm <= lunch_lull_end_utc

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
                diag.entry_blocked_volatility         = volatility_blocked
                diag.entry_blocked_regime             = regime_blocked
                diag.entry_blocked_relative_strength  = rs_blocked
                diag.entry_blocked_event              = event_blocked
                diag.entry_blocked_lunch_lull         = lunch_lull_blocked

                # Every configurable gate cleared? (Excludes the structural
                # same-session constraint below, which isn't a user gate.)
                gates_clear = (
                    not cap_blocked and not trades_blocked and not htf_blocked
                    and not time_exit_blocked and not entry_window_blocked
                    and not consec_loss_blocked and not cooldown_blocked
                    and not spread_blocked and not gap_blocked
                    and not rsi_band_blocked and not vol_ratio_blocked
                    and not volatility_blocked and not regime_blocked
                    and not rs_blocked and not event_blocked
                    and not lunch_lull_blocked
                )

                # Intraday entries must be fillable within the same session. When
                # an intraday strategy runs on bars whose timeframe spans a whole
                # session (e.g. a 1d timeframe), EVERY bar is its session's last
                # bar, so every entry is rejected here. Record it explicitly so
                # the no-trade diagnostic can name this cause instead of guessing.
                if gates_clear and is_session_last_bar:
                    diag.entry_blocked_session_last = True

                if gates_clear and not is_session_last_bar:
                    # Fill at next bar's open price (realistic execution)
                    next_open = float(open_arr[i + 1])
                    entry_price = _entry_fill(next_open)
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
                    if is_short:
                        _initial_stop_price = _compute_initial_stop_short(
                            stop_loss_spec,
                            fallback_pct=stop_loss_pct,
                            entry_price=entry_price,
                            entry_index=entry_index,
                            arrays=arrays,
                            high_arr=high_arr,
                            day_ordinals=day_ordinals,
                        )
                        # SHORT: the trailing stop is a ceiling that ratchets
                        # DOWN, so it starts at +inf (no constraint yet).
                        _trailing_floor = float("inf")
                    else:
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

                    # ── Capture the structured entry rationale ────────────────
                    signal_indicators = _snapshot_indicators(arrays, i, compiled_entry)
                    signal_close = float(close_arr[i])
                    ind_str = _fmt_indicator_values(signal_indicators)
                    entry_summary = (
                        f"Entered {side.upper()} at {entry_price:.4f} on {timestamps_iso[entry_index]} "
                        f"(bar #{entry_index}). Entry signal [{entry_condition_text}] became "
                        f"true on bar #{i} ({timestamps_iso[i]}) where close={signal_close:.4f}"
                        f"{(', ' + ind_str) if ind_str else ''}. "
                        f"Signal held {_signal_consecutive_bars}/{entry_confirmation_bars} "
                        f"confirmation bar(s). Filled at next bar open {next_open:.4f} plus costs "
                        f"(slippage {slippage_bps:.1f}bps, commission {commission_bps:.1f}bps, "
                        f"STT {eff_entry_stt:.3f}%) = {entry_price:.4f}. "
                        f"Initial stop at {_initial_stop_price:.4f}."
                    )
                    entry_reason_payload = {
                        "summary":          entry_summary,
                        "evaluation_mode":  entry_evaluation_mode,
                        "condition":        entry_condition_text,
                        "signal_bar":       _snapshot_bar(arrays, i, timestamps_iso),
                        "entry_bar": {
                            "index":     int(entry_index),
                            "timestamp": str(timestamps_iso[entry_index]),
                        },
                        "indicators":                  signal_indicators,
                        "confirmation_bars_required":  int(entry_confirmation_bars),
                        "confirmation_bars_observed":  int(_signal_consecutive_bars),
                        "fill": {
                            "basis":           "next_bar_open",
                            "raw_price":       round(next_open, 6),
                            "effective_price": round(entry_price, 6),
                            "slippage_bps":    float(slippage_bps),
                            "commission_bps":  float(commission_bps),
                            "stt_pct":         float(eff_entry_stt),
                        },
                        "initial_stop_price": round(_initial_stop_price, 6),
                        "gates_passed":       list(active_entry_gates),
                    }
                    if entry_evaluation_mode == "registry":
                        entry_reason_payload["entry_signal_rules"] = entry_signal_rules or []

                    logger.debug("Entry rationale | %s", entry_summary)

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

        # SHORT takes profit when price FALLS, so its target sits below entry.
        take_profit_price = (
            entry_price * (1 - take_profit_pct / 100.0) if is_short
            else entry_price * (1 + take_profit_pct / 100.0)
        )

        # Timestamp the price-path exit fills at. Defaults to the strategy bar;
        # in 1-minute execution it is refined to the exact minute the stop or
        # target was touched.
        price_exit_ts = timestamps_iso[i]

        if intrabar_execution and int(subbar_ends[i]) > int(subbar_starts[i]):
            # ── 1-minute execution: walk this strategy bar's minutes in order ──
            # The FIRST minute that touches the stop or the target determines the
            # exit and the price it fills at. Only when a *single* minute touches
            # BOTH do we fall back to the pessimistic "stop filled first" rule —
            # so the same-bar ambiguity is confined to one minute instead of the
            # whole (possibly day-long) strategy bar. MAE/MFE and the trailing
            # floor advance minute by minute for true intrabar resolution; the
            # ATR/EMA reference for trailing stays on the strategy timeframe
            # (current_index=i) since that is the timeframe the strategy trades.
            stop_hit = take_profit_hit = False
            is_trailing_exit = False
            stop_price = min(_initial_stop_price, _trailing_floor) if is_short else max(_initial_stop_price, _trailing_floor)
            for m in range(int(subbar_starts[i]), int(subbar_ends[i])):
                m_h = float(m_high[m]); m_l = float(m_low[m]); m_c = float(m_close[m])
                if m_l < _trade_min_low:  _trade_min_low  = m_l
                if m_h > _trade_max_high: _trade_max_high = m_h

                if is_short:
                    # SHORT: trailing is a ceiling that ratchets DOWN with the
                    # lowest low; effective stop = min(initial, ceiling); the
                    # stop fires when a high pierces it, the target when a low
                    # reaches the (below-entry) take-profit.
                    candidate_floor = _compute_trailing_ceiling_short(
                        trailing_stop_spec,
                        entry_price=entry_price,
                        current_index=i,
                        arrays=arrays,
                        lowest_low_since_entry=_trade_min_low,
                        current_close=m_c,
                    )
                    if candidate_floor < _trailing_floor:
                        _trailing_floor = candidate_floor
                    stop_price = min(_initial_stop_price, _trailing_floor)
                    m_stop_hit = m_h >= stop_price
                    m_tp_hit   = m_l <= take_profit_price
                    trailing_active = _trailing_floor < _initial_stop_price
                else:
                    candidate_floor = _compute_trailing_floor_long(
                        trailing_stop_spec,
                        entry_price=entry_price,
                        current_index=i,
                        arrays=arrays,
                        highest_high_since_entry=_trade_max_high,
                        current_close=m_c,
                    )
                    if candidate_floor > _trailing_floor:
                        _trailing_floor = candidate_floor
                    stop_price = max(_initial_stop_price, _trailing_floor)
                    m_stop_hit = m_l <= stop_price
                    m_tp_hit   = m_h >= take_profit_price
                    trailing_active = _trailing_floor > _initial_stop_price

                if m_stop_hit or m_tp_hit:
                    stop_hit         = m_stop_hit
                    take_profit_hit  = m_tp_hit
                    is_trailing_exit = m_stop_hit and trailing_active
                    # Trigger payloads and the exit timestamp reflect the firing
                    # minute, not the coarse strategy bar.
                    low_price  = m_l
                    high_price = m_h
                    if m_ts is not None:
                        price_exit_ts = str(m_ts[m])
                    break
        else:
            # ── Legacy strategy-bar resolution ────────────────────────────────
            # Used when 1-minute execution is off, and as the safe fallback when
            # a strategy bar has no underlying minute data (empty sub-bar slice).
            if low_price  < _trade_min_low:  _trade_min_low  = low_price
            if high_price > _trade_max_high: _trade_max_high = high_price

            # Phase 3 — effective stop is the higher of the initial SL (locked at
            # entry) and the trailing floor (running max since entry, only when
            # a trailing spec is configured). Ratchet means the floor never moves
            # backwards even if subsequent candles are lower. SHORT mirrors this:
            # a ceiling that ratchets DOWN with the lowest low, fired by a high.
            if is_short:
                candidate_floor = _compute_trailing_ceiling_short(
                    trailing_stop_spec,
                    entry_price=entry_price,
                    current_index=i,
                    arrays=arrays,
                    lowest_low_since_entry=_trade_min_low,
                    current_close=close_price,
                )
                if candidate_floor < _trailing_floor:
                    _trailing_floor = candidate_floor
                stop_price = min(_initial_stop_price, _trailing_floor)
                stop_hit        = high_price >= stop_price
                take_profit_hit = low_price  <= take_profit_price
                is_trailing_exit = stop_hit and _trailing_floor < _initial_stop_price
            else:
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

                # Effective stop is the higher of the initial SL and the trailing
                # floor. Trailing-floor candidates use intra-bar highs, so the floor
                # can legitimately sit between the current bar's low and the bar's
                # high — in that case the stop fires *this* bar at the floor price
                # (worst-case fill, mirrors the static-SL code path below).
                stop_price = max(_initial_stop_price, _trailing_floor)
                stop_hit        = low_price  <= stop_price
                take_profit_hit = high_price >= take_profit_price
                # A trailing-stop exit and a static-SL exit are functionally the same
                # mechanic, but the user wants to know which one fired in the report.
                is_trailing_exit = stop_hit and _trailing_floor > _initial_stop_price

        diag.stop_hit = stop_hit
        diag.tp_hit   = take_profit_hit

        exit_price:  float | None = None
        exit_reason: str   | None = None
        exit_date = price_exit_ts
        # Raw (pre-cost) price the exit filled at, plus a structured description
        # of what triggered it. Both feed the exit rationale built when booking.
        raw_exit_price: float | None = None
        exit_trigger: dict[str, Any] | None = None

        # ── Worst-case rule: if both stop AND TP touched on the same bar ──────
        # We assume the stop was hit first (conservative for the trader).
        if stop_hit and take_profit_hit:
            raw_exit_price = stop_price
            exit_price  = _exit_fill(stop_price)
            exit_reason = "TRAILING_STOP_AND_TAKE_PROFIT_SAME_BAR" if is_trailing_exit else "STOP_LOSS_AND_TAKE_PROFIT_SAME_BAR"
            exit_trigger = {
                "type":              "stop_and_take_profit_same_bar",
                "stop_price":        round(stop_price, 6),
                "take_profit_price": round(take_profit_price, 6),
                "bar_low":           round(low_price, 4),
                "bar_high":          round(high_price, 4),
                "is_trailing_stop":  bool(is_trailing_exit),
                "resolution":        "pessimistic — stop assumed filled before target",
            }

        elif stop_hit:
            raw_exit_price = stop_price
            exit_price  = _exit_fill(stop_price)
            exit_reason = "TRAILING_STOP" if is_trailing_exit else "STOP_LOSS"
            exit_trigger = {
                "type":               "trailing_stop" if is_trailing_exit else "stop_loss",
                "stop_price":         round(stop_price, 6),
                "bar_low":            round(low_price, 4),
                "bar_high":           round(high_price, 4),
                "is_trailing_stop":   bool(is_trailing_exit),
                "initial_stop_price": round(_initial_stop_price, 6),
                "trailing_floor":     round(_trailing_floor, 6) if _trailing_floor not in (float("-inf"), float("inf")) else None,
            }

        elif take_profit_hit:
            raw_exit_price = take_profit_price
            exit_price  = _exit_fill(take_profit_price)
            exit_reason = "TAKE_PROFIT"
            exit_trigger = {
                "type":              "take_profit",
                "take_profit_price": round(take_profit_price, 6),
                "bar_high":          round(high_price, 4),
                "bar_low":           round(low_price, 4),
                "take_profit_pct":   float(take_profit_pct),
            }

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
                variables = _trade_variables(entry_price, close_price, stop_loss_pct, take_profit_pct, side)
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
                    raw_exit_price = close_price
                    fill_basis = "current_close"
                    exit_price = _exit_fill(close_price)
                    exit_date = timestamps_iso[i]
                else:
                    raw_exit_price = float(open_arr[i + 1])
                    fill_basis = "next_bar_open"
                    exit_price = _exit_fill(raw_exit_price)
                    exit_date = timestamps_iso[i + 1]
                exit_reason = "EXIT_SIGNAL"
                exit_trigger = {
                    "type":       "exit_signal",
                    "condition":  exit_condition_text,
                    "fill_basis": fill_basis,
                }

            # Check max holding candles (time-based stop)
            if exit_price is None and max_holding_candles is not None and holding_candles >= max_holding_candles:
                if is_session_last_bar or i + 1 >= n_rows:
                    raw_exit_price = close_price
                    fill_basis = "current_close"
                    exit_price = _exit_fill(close_price)
                    exit_date = timestamps_iso[i]
                else:
                    raw_exit_price = float(open_arr[i + 1])
                    fill_basis = "next_bar_open"
                    exit_price = _exit_fill(raw_exit_price)
                    exit_date = timestamps_iso[i + 1]
                exit_reason = "MAX_HOLDING"
                exit_trigger = {
                    "type":                "max_holding",
                    "max_holding_candles": int(max_holding_candles),
                    "holding_candles":     int(holding_candles),
                    "fill_basis":          fill_basis,
                }

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
                raw_exit_price = close_price
                exit_price = _exit_fill(close_price)
                exit_date = timestamps_iso[i]
                exit_reason = "TIME_EXIT"
                exit_trigger = {
                    "type":                "time_exit",
                    "cutoff_utc_minutes":  int(time_exit_cutoff_minutes),
                    "bar_utc_minutes":     int(bar_utc_minutes[i]),
                    "fill_basis":          "current_close",
                }

            if exit_price is None and is_session_last_bar:
                raw_exit_price = close_price
                exit_price = _exit_fill(close_price)
                exit_date = timestamps_iso[i]
                exit_reason = "SESSION_END"
                exit_trigger = {
                    "type":       "session_end",
                    "fill_basis": "current_close",
                    "note":       "intraday square-off at final bar of the session",
                }

        # ── Book the trade ───────────────────────────────────────────────────
        if exit_price is not None and exit_reason is not None:
            # SHORT profits when the cover price is BELOW the entry, so its P&L
            # is the mirror of a long: entry - exit instead of exit - entry.
            pnl_inr = (entry_price - exit_price) if is_short else (exit_price - entry_price)
            pnl_abs = pnl_inr / entry_price             # fractional return (for compounding)
            pnl_pct = pnl_abs * 100.0

            # MAE/MFE — for a LONG, adverse excursion is the lowest low and
            # favourable the highest high. For a SHORT both flip: the highest
            # high is the worst case and the lowest low the best.
            if is_short:
                mae_pct = ((entry_price - _trade_max_high) / entry_price * 100.0) if entry_price else 0.0
                mfe_pct = ((entry_price - _trade_min_low)  / entry_price * 100.0) if entry_price else 0.0
            else:
                mae_pct = ((_trade_min_low  - entry_price) / entry_price * 100.0) if entry_price else 0.0
                mfe_pct = ((_trade_max_high - entry_price) / entry_price * 100.0) if entry_price else 0.0

            exit_reason_detail = _build_exit_reason_detail(
                exit_reason=exit_reason,
                exit_trigger=exit_trigger,
                exit_bar=_snapshot_bar(arrays, i, timestamps_iso),
                fill_timestamp=str(exit_date),
                raw_exit_price=raw_exit_price if raw_exit_price is not None else exit_price,
                effective_exit_price=exit_price,
                slippage_bps=slippage_bps,
                commission_bps=commission_bps,
                stt_exit_pct=eff_exit_stt,
                holding_candles=max(holding_candles, 0),
                pnl_pct=pnl_pct,
                pnl_inr=pnl_inr,
                exit_indicators=_snapshot_indicators(arrays, i, compiled_exit),
                side=side,
            )

            trades.append(Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                symbol=symbol,
                side=side.upper(),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                pnl_pct=round(pnl_pct, 4),
                pnl_abs=round(pnl_abs, 6),
                pnl_inr=round(pnl_inr, 4),
                exit_reason=exit_reason,
                holding_candles=max(holding_candles, 0),
                mae_pct=round(mae_pct, 4),
                mfe_pct=round(mfe_pct, 4),
                entry_reason=entry_reason_payload,
                exit_reason_detail=exit_reason_detail,
            ))
            exit_reason_counts[exit_reason] += 1

            logger.debug("Exit rationale | %s", exit_reason_detail["summary"])

            if is_intraday:
                _session_cumulative_pnl_pct += pnl_pct
                _session_trades_today       += 1

            _portfolio_balance_factor *= (1.0 + pnl_abs)

            # Phase 10 — update consecutive loss counter + cooldown timers
            trade_was_loss = pnl_pct < 0
            if trade_was_loss:
                _consecutive_losses += 1
                # Streak hit the threshold → start a bar-count pause and
                # reset the counter so the next streak begins fresh after
                # the cooling-off window. The pause length is hardcoded
                # in bars (see _CONSEC_LOSS_PAUSE_BARS), which scales with
                # the strategy's timeframe automatically.
                if (
                    max_consecutive_losses > 0
                    and _consecutive_losses >= max_consecutive_losses
                ):
                    _consec_loss_pause_until = i + _CONSEC_LOSS_PAUSE_BARS
                    _consecutive_losses = 0
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
        raw_last_close = float(close_arr[-1])
        last_close = _exit_fill(raw_last_close)
        pnl_inr = (entry_price - last_close) if is_short else (last_close - entry_price)
        pnl_abs = pnl_inr / entry_price
        pnl_pct = pnl_abs * 100.0
        if is_short:
            mae_pct = ((entry_price - _trade_max_high) / entry_price * 100.0) if entry_price else 0.0
            mfe_pct = ((entry_price - _trade_min_low)  / entry_price * 100.0) if entry_price else 0.0
        else:
            mae_pct = ((_trade_min_low  - entry_price) / entry_price * 100.0) if entry_price else 0.0
            mfe_pct = ((_trade_max_high - entry_price) / entry_price * 100.0) if entry_price else 0.0
        eod_holding_candles = max(n_rows - 1 - entry_index, 0)
        eod_exit_detail = _build_exit_reason_detail(
            exit_reason="END_OF_DATA",
            exit_trigger={"type": "end_of_data", "fill_basis": "last_bar_close"},
            exit_bar=_snapshot_bar(arrays, n_rows - 1, timestamps_iso),
            fill_timestamp=str(timestamps_iso[-1]),
            raw_exit_price=raw_last_close,
            effective_exit_price=last_close,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            stt_exit_pct=eff_exit_stt,
            holding_candles=eod_holding_candles,
            pnl_pct=pnl_pct,
            pnl_inr=pnl_inr,
            exit_indicators=_snapshot_indicators(arrays, n_rows - 1, compiled_exit),
            side=side,
        )
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=str(timestamps_iso[-1]),
            symbol=symbol,
            side=side.upper(),
            entry_price=round(entry_price, 4),
            exit_price=round(last_close, 4),
            pnl_pct=round(pnl_pct, 4),
            pnl_abs=round(pnl_abs, 6),
            pnl_inr=round(pnl_inr, 4),
            exit_reason="END_OF_DATA",
            holding_candles=eod_holding_candles,
            mae_pct=round(mae_pct, 4),
            mfe_pct=round(mfe_pct, 4),
            entry_reason=entry_reason_payload,
            exit_reason_detail=eod_exit_detail,
        ))
        exit_reason_counts["END_OF_DATA"] += 1
        logger.debug("Exit rationale | %s", eod_exit_detail["summary"])

    diag_summary = _summarise_diagnostics(diagnostics)
    # Aggregate entry-block breakdown so the default (INFO) log explains *why*
    # signals didn't become trades — one line, no per-bar flooding.
    blocked_breakdown = {
        "daily_cap":           diag_summary["entry_blocked_daily_cap"],
        "max_trades":          diag_summary["entry_blocked_max_trades"],
        "htf_trend":           diag_summary["entry_blocked_htf"],
        "time_exit":           diag_summary["entry_blocked_time_exit"],
        "entry_window":        diag_summary["entry_blocked_entry_window"],
        "consecutive_losses":  diag_summary["entry_blocked_consecutive_losses"],
        "cooldown":            diag_summary["entry_blocked_cooldown"],
        "spread":              diag_summary["entry_blocked_spread"],
        "gap":                 diag_summary["entry_blocked_gap"],
        "confirmation":        diag_summary["entry_blocked_confirmation"],
        "rsi_band":            diag_summary["entry_blocked_rsi_band"],
        "volume_ratio":        diag_summary["entry_blocked_volume_ratio"],
    }
    logger.info(
        "Trade simulation complete | trades=%d | exits[%s] | warm_up_skips=%d "
        "entry_signals=%d | blocked[%s]",
        len(trades),
        _fmt_breakdown(dict(exit_reason_counts)),
        diag_summary["warm_up_skips"],
        diag_summary["entry_signals"],
        _fmt_breakdown(blocked_breakdown),
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
        "entry_blocked_entry_window":      sum(1 for d in diagnostics if d.entry_blocked_entry_window),
        "entry_blocked_consecutive_losses": sum(1 for d in diagnostics if d.entry_blocked_consecutive_losses),
        "entry_blocked_cooldown":  sum(1 for d in diagnostics if d.entry_blocked_cooldown),
        "entry_blocked_spread":    sum(1 for d in diagnostics if d.entry_blocked_spread),
        "entry_blocked_gap":       sum(1 for d in diagnostics if d.entry_blocked_gap),
        "entry_blocked_confirmation": sum(1 for d in diagnostics if d.entry_blocked_confirmation),
        "entry_blocked_rsi_band":  sum(1 for d in diagnostics if d.entry_blocked_rsi_band),
        "entry_blocked_volume_ratio": sum(1 for d in diagnostics if d.entry_blocked_volume_ratio),
        "daily_cap_blocks":      sum(1 for d in diagnostics if d.entry_blocked_daily_cap),
        "max_trades_blocks":     sum(1 for d in diagnostics if d.entry_blocked_max_trades),
        "htf_blocks":            sum(1 for d in diagnostics if d.entry_blocked_htf),
        "time_exit_blocks":      sum(1 for d in diagnostics if d.entry_blocked_time_exit),
        "exit_signals":          sum(1 for d in diagnostics if d.exit_signal),
        "stop_hits":             sum(1 for d in diagnostics if d.stop_hit),
        "tp_hits":               sum(1 for d in diagnostics if d.tp_hit),
    }
