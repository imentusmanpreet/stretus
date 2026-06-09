"""
app/services/execution/entry_gates.py
───────────────────────────────────────
Phase 10 entry-gate evaluator for the live execution path.

Backtest parity
───────────────
The quant-engine simulator (`quant_engine/engine/simulator.py`) applies a set
of entry gates inside its candle-by-candle hot loop. Live signal generation
(`StrategyEvaluator`) historically evaluated only the trigger/filter rules and
ignored those gates — so a strategy could behave differently in backtest vs
live. This module closes that gap.

It re-implements the same 9 gates the simulator enforces, but for a single
decision point (the latest closed candle):

  1. direction              — long_only / short_only / both
  2. entry_window           — equity: IST trading window. crypto: 24/7 UTC.
  3. max_consecutive_losses — circuit breaker after N losing trades
  4. cooldown               — wait N bars after a loss / profit
  5. max_spread_bps         — reject when the estimated spread is too wide
  6. gap_filter             — skip sessions that opened with a large gap
  7. entry_confirmation_bars— signal must hold True for N consecutive bars
  8. rsi_entry_band         — RSI must sit inside [min, max]
  9. volume_ratio_threshold — current volume >= N x 20-bar average

Multi-asset notes
─────────────────
• For ``equity_cash`` the entry_window time strings are interpreted as IST
  clock times (no behaviour change — Indian retail brokers use IST).
• For ``crypto_spot`` the entry_window times are interpreted as **UTC** (the
  market is 24/7 and Binance kline timestamps are UTC). When both window
  bounds are absent the gate becomes a no-op for crypto.
• Gap filter is naturally meaningful only for asset classes with discrete
  sessions; for crypto we skip it (continuous market = no overnight gap).

Most gates are stateless and derived purely from the candle history. Only the
consecutive-loss and cooldown gates need carried state, which is supplied via
`ExecutionStatePayload` (consecutive_losses, last_trade_was_loss,
bars_since_last_trade).

Design notes
────────────
• The evaluator never raises on missing data — if a gate cannot be computed
  (too few candles, missing column) it logs and *passes*, matching the
  simulator's defensive behaviour. A gate should block only on a real signal.
• Gates are checked in cheap-to-expensive order; the first block short-circuits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.schemas.execution import AssetClass

logger = logging.getLogger(__name__)

# IST is UTC+5:30. Candle DataFrames from MarketDataService carry a UTC-aware
# DatetimeIndex, so IST clock times must be shifted by this many minutes.
_IST_OFFSET_MINUTES = 330


@dataclass
class GateResult:
    """Outcome of running every entry gate against the latest candle."""

    passed: bool = True
    blocked_by: Optional[str] = None
    messages: List[str] = field(default_factory=list)


# ── Column / value helpers ─────────────────────────────────────────────────────

def _series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """
    Return an OHLCV column regardless of capitalisation.

    MarketDataService emits capitalised columns (Open/High/Low/Close/Volume);
    other producers use lowercase. We accept either.
    """
    for cand in (name, name.lower(), name.capitalize(), name.upper()):
        if cand in df.columns:
            return df[cand]
    return None


def _hhmm_to_utc_minutes(value: str, *, ist: bool) -> Optional[int]:
    """
    Convert an 'HH:MM' clock string to minutes-of-day in UTC.

    ``ist=True``  : interpret input as IST (equity_cash path).
    ``ist=False`` : interpret input as UTC directly (crypto_spot path).
    """
    try:
        hh, mm = value.strip().split(":")
        minutes = int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        logger.warning("entry_gates: could not parse time '%s' — gate skipped.", value)
        return None
    if ist:
        return (minutes - _IST_OFFSET_MINUTES) % 1440
    return minutes % 1440


def _ist_hhmm_to_utc_minutes(value: str) -> Optional[int]:
    """Back-compat: equity-path helper retained so external imports still work."""
    return _hhmm_to_utc_minutes(value, ist=True)


def _bar_utc_minutes(ts: Any) -> Optional[int]:
    """Minutes-of-day (UTC) for a candle timestamp."""
    try:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        return int(ts.hour) * 60 + int(ts.minute)
    except Exception:  # noqa: BLE001 — defensive: never block on a bad timestamp
        return None


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — same definition the quant engine uses for RSI_14."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


# ── Main entry point ────────────────────────────────────────────────────────────

def evaluate_entry_gates(
    *,
    df: pd.DataFrame,
    gates: Any,
    exec_state: Any,
    side: str,
    rule_engine: Any,
    entry_block: Dict[str, Any],
    asset_class: AssetClass = AssetClass.equity_cash,
) -> GateResult:
    """
    Run all Phase 10 entry gates against the latest candle of `df`.

    Parameters
    ----------
    df           : OHLCV candles, oldest→newest, UTC-aware DatetimeIndex.
                   The last row is the candle the entry signal fired on.
    gates        : GatesConfig (pydantic) — the strategy's gate constraints.
    exec_state   : ExecutionStatePayload — supplies consecutive_losses,
                   last_trade_was_loss, bars_since_last_trade.
    side         : "BUY" or "SELL" — the side this live entry would take.
    rule_engine  : RuleEngine — reused to re-evaluate the signal on prior bars
                   for the entry_confirmation_bars gate.
    entry_block  : {"trigger": {...}, "filters": [...]} — passed to rule_engine.

    Returns a GateResult; `passed=False` with `blocked_by` set on the first
    gate that rejects the entry.
    """
    result = GateResult()

    if df is None or len(df) == 0:
        result.messages.append("  ⚠️  Entry gates skipped — no candle data.")
        return result

    last_ts = df.index[-1]

    def _block(gate: str, msg: str) -> GateResult:
        result.passed = False
        result.blocked_by = gate
        result.messages.append(f"  ⛔ GATE [{gate}] BLOCKED — {msg}")
        return result

    # ── Gate 1: direction ──────────────────────────────────────────────────────
    direction = getattr(gates, "direction", "both") or "both"
    if direction == "short_only" and side == "BUY":
        return _block(
            "direction",
            "strategy is short_only but live evaluation only generates long entries.",
        )
    if direction == "long_only" and side == "SELL":
        return _block(
            "direction",
            "strategy is long_only but a short entry was requested.",
        )
    result.messages.append(f"  ✅ GATE [direction] → {direction} allows {side}.")

    # ── Gate 2: entry_window ───────────────────────────────────────────────────
    # equity_cash : window times are IST clock times (Indian retail broker norm).
    #               When both bounds are absent we fall back to the NSE cash
    #               session 09:15–15:30 IST so equity strategies don't trade
    #               outside market hours by accident.
    # crypto_spot : window times are interpreted as UTC (Binance is 24/7 UTC).
    #               When both bounds are absent the gate is a no-op — crypto
    #               strategies trade around the clock by default.
    win_start = getattr(gates, "entry_window_start", None)
    win_end = getattr(gates, "entry_window_end", None)
    if not (win_start or win_end) and asset_class == AssetClass.equity_cash:
        win_start, win_end = "09:15", "15:30"
    if win_start or win_end:
        bar_min = _bar_utc_minutes(last_ts)
        if bar_min is None:
            result.messages.append("  ⚠️  GATE [entry_window] skipped — unreadable bar timestamp.")
        else:
            use_ist = asset_class == AssetClass.equity_cash
            tz_label = "IST" if use_ist else "UTC"
            start_min = _hhmm_to_utc_minutes(win_start, ist=use_ist) if win_start else None
            end_min   = _hhmm_to_utc_minutes(win_end,   ist=use_ist) if win_end   else None
            if start_min is not None and bar_min < start_min:
                return _block(
                    "entry_window",
                    f"bar before window start ({win_start} {tz_label}).",
                )
            if end_min is not None and bar_min > end_min:
                return _block(
                    "entry_window",
                    f"bar after window end ({win_end} {tz_label}) — no fresh entries.",
                )
            result.messages.append(
                f"  ✅ GATE [entry_window] → inside {win_start or '--'}–{win_end or '--'} {tz_label}."
            )
    elif asset_class == AssetClass.crypto_spot:
        result.messages.append("  ✅ GATE [entry_window] → crypto 24/7 — no window constraint.")

    # ── Gate 3: max_consecutive_losses ─────────────────────────────────────────
    max_consec = int(getattr(gates, "max_consecutive_losses", 0) or 0)
    if max_consec > 0:
        consec = int(getattr(exec_state, "consecutive_losses", 0) or 0)
        if consec >= max_consec:
            return _block(
                "max_consecutive_losses",
                f"{consec} consecutive losses >= limit {max_consec} — circuit breaker tripped.",
            )
        result.messages.append(
            f"  ✅ GATE [max_consecutive_losses] → {consec}/{max_consec} losses."
        )

    # ── Gate 4: cooldown after loss / profit ───────────────────────────────────
    cd_loss = int(getattr(gates, "cooldown_bars_after_loss", 0) or 0)
    cd_profit = int(getattr(gates, "cooldown_bars_after_profit", 0) or 0)
    if cd_loss > 0 or cd_profit > 0:
        last_was_loss = getattr(exec_state, "last_trade_was_loss", None)
        bars_since = int(getattr(exec_state, "bars_since_last_trade", 0) or 0)
        if last_was_loss is True and cd_loss > 0 and bars_since < cd_loss:
            return _block(
                "cooldown",
                f"{bars_since} bars since last loss < cooldown {cd_loss}.",
            )
        if last_was_loss is False and cd_profit > 0 and bars_since < cd_profit:
            return _block(
                "cooldown",
                f"{bars_since} bars since last profit < cooldown {cd_profit}.",
            )
        result.messages.append("  ✅ GATE [cooldown] → no active cooldown.")

    # ── Gate 5: max_spread_bps ─────────────────────────────────────────────────
    max_spread = float(getattr(gates, "max_spread_bps", 0.0) or 0.0)
    if max_spread > 0:
        high = _series(df, "high")
        low = _series(df, "low")
        close = _series(df, "close")
        if high is not None and low is not None and close is not None:
            c = float(close.iloc[-1])
            if c > 0:
                spread_bps = (float(high.iloc[-1]) - float(low.iloc[-1])) / c * 10_000.0
                if spread_bps > max_spread:
                    return _block(
                        "max_spread_bps",
                        f"estimated spread {spread_bps:.1f}bps > limit {max_spread:.1f}bps.",
                    )
                result.messages.append(
                    f"  ✅ GATE [max_spread_bps] → {spread_bps:.1f}bps <= {max_spread:.1f}bps."
                )
        else:
            result.messages.append("  ⚠️  GATE [max_spread_bps] skipped — missing OHLC columns.")

    # ── Gate 6: gap_filter ─────────────────────────────────────────────────────
    # Gaps are a session-boundary phenomenon. Crypto trades 24/7, so the gap
    # filter is intentionally a no-op there.
    gap_filter = getattr(gates, "gap_filter", "none") or "none"
    if gap_filter != "none" and asset_class == AssetClass.crypto_spot:
        result.messages.append(
            "  ✅ GATE [gap_filter] → crypto 24/7 — gap filter not applicable."
        )
    elif gap_filter != "none":
        open_s = _series(df, "open")
        close_s = _series(df, "close")
        if open_s is not None and close_s is not None and isinstance(df.index, pd.DatetimeIndex):
            day = df.index.normalize()
            last_day = day[-1]
            today_mask = day == last_day
            prev_mask = day < last_day
            if prev_mask.any() and today_mask.any():
                today_open = float(open_s[today_mask].iloc[0])
                prev_close = float(close_s[prev_mask].iloc[-1])
                if prev_close > 0:
                    gap_pct = (today_open - prev_close) / prev_close * 100.0
                    threshold = float(getattr(gates, "gap_threshold_pct", 0.5) or 0.5)
                    gap_up = gap_pct > threshold
                    gap_down = gap_pct < -threshold
                    blocked = (
                        (gap_filter == "ignore_gap_up" and gap_up)
                        or (gap_filter == "ignore_gap_down" and gap_down)
                        or (gap_filter == "ignore_both" and (gap_up or gap_down))
                    )
                    if blocked:
                        return _block(
                            "gap_filter",
                            f"session opened {gap_pct:+.2f}% gap (threshold ±{threshold}%).",
                        )
                    result.messages.append(
                        f"  ✅ GATE [gap_filter] → session gap {gap_pct:+.2f}% within ±{threshold}%."
                    )
            else:
                result.messages.append(
                    "  ⚠️  GATE [gap_filter] skipped — no prior session in candle history."
                )
        else:
            result.messages.append("  ⚠️  GATE [gap_filter] skipped — missing data/index.")

    # ── Gate 7: entry_confirmation_bars ────────────────────────────────────────
    confirm_bars = int(getattr(gates, "entry_confirmation_bars", 1) or 1)
    if confirm_bars > 1:
        # The latest bar already fired (caller confirmed entry_signal). Re-check
        # the prior (confirm_bars - 1) closed bars: the signal must hold on each.
        held = True
        for k in range(1, confirm_bars):
            sub = df.iloc[: len(df) - k]
            if len(sub) == 0:
                held = False
                break
            ok, _ = rule_engine.evaluate_entry(sub, entry_block)
            if not ok:
                held = False
                break
        if not held:
            return _block(
                "entry_confirmation_bars",
                f"signal did not hold for {confirm_bars} consecutive closed bars.",
            )
        result.messages.append(
            f"  ✅ GATE [entry_confirmation_bars] → signal held {confirm_bars} bars."
        )

    # ── Gate 8: rsi_entry_band ─────────────────────────────────────────────────
    rsi_min = getattr(gates, "rsi_entry_band_min", None)
    rsi_max = getattr(gates, "rsi_entry_band_max", None)
    if rsi_min is not None or rsi_max is not None:
        close_s = _series(df, "close")
        if close_s is not None and len(close_s) >= 15:
            rsi_val = float(_wilder_rsi(close_s).iloc[-1])
            if rsi_val != rsi_val:  # NaN guard
                result.messages.append("  ⚠️  GATE [rsi_entry_band] skipped — RSI not yet warm.")
            else:
                if rsi_min is not None and rsi_val < float(rsi_min):
                    return _block(
                        "rsi_entry_band",
                        f"RSI {rsi_val:.1f} < band min {rsi_min}.",
                    )
                if rsi_max is not None and rsi_val > float(rsi_max):
                    return _block(
                        "rsi_entry_band",
                        f"RSI {rsi_val:.1f} > band max {rsi_max}.",
                    )
                result.messages.append(
                    f"  ✅ GATE [rsi_entry_band] → RSI {rsi_val:.1f} inside band."
                )
        else:
            result.messages.append("  ⚠️  GATE [rsi_entry_band] skipped — too few candles.")

    # ── Gate 9: volume_ratio_threshold ─────────────────────────────────────────
    vol_thr = getattr(gates, "volume_ratio_threshold", None)
    if vol_thr is not None and float(vol_thr) > 0:
        vol_s = _series(df, "volume")
        if vol_s is not None and len(vol_s) >= 5:
            avg_vol = float(vol_s.astype(float).rolling(20, min_periods=5).mean().iloc[-1])
            current_vol = float(vol_s.iloc[-1])
            if avg_vol > 0:
                ratio = current_vol / avg_vol
                if ratio < float(vol_thr):
                    return _block(
                        "volume_ratio_threshold",
                        f"volume {ratio:.2f}x average < required {vol_thr}x.",
                    )
                result.messages.append(
                    f"  ✅ GATE [volume_ratio_threshold] → volume {ratio:.2f}x >= {vol_thr}x."
                )
            else:
                result.messages.append(
                    "  ⚠️  GATE [volume_ratio_threshold] skipped — zero average volume."
                )
        else:
            result.messages.append("  ⚠️  GATE [volume_ratio_threshold] skipped — no volume data.")

    # ── Gate 10: volatility band (ATR/NATR) ────────────────────────────────────
    # Parity with the backtest simulator's vol_filter_* gate: same TA-Lib study,
    # same band semantics. Blocks when volatility is too low (dead market) or too
    # high (chaotic). No-raise: skip if data/talib unavailable or not yet warm.
    vf_metric = getattr(gates, "vol_filter_metric", None)
    vf_min = getattr(gates, "vol_filter_min", None)
    vf_max = getattr(gates, "vol_filter_max", None)
    if vf_metric in ("atr", "natr") and (vf_min is not None or vf_max is not None):
        window = int(getattr(gates, "vol_filter_window", 14) or 14)
        h, l, c = _series(df, "high"), _series(df, "low"), _series(df, "close")
        if h is not None and l is not None and c is not None and len(c) >= window + 1:
            try:
                import numpy as _np
                import talib as _talib
                fn = _talib.NATR if vf_metric == "natr" else _talib.ATR
                vol_series = fn(
                    _np.ascontiguousarray(h.to_numpy(dtype=float)),
                    _np.ascontiguousarray(l.to_numpy(dtype=float)),
                    _np.ascontiguousarray(c.to_numpy(dtype=float)),
                    timeperiod=window,
                )
                vol_val = float(vol_series[-1])
            except Exception:
                vol_val = float("nan")
            if vol_val != vol_val:  # NaN guard
                result.messages.append(f"  ⚠️  GATE [volatility] skipped — {vf_metric.upper()} not yet warm.")
            elif vf_min is not None and vol_val < float(vf_min):
                return _block("volatility", f"{vf_metric.upper()} {vol_val:.3f} < band min {vf_min}.")
            elif vf_max is not None and vol_val > float(vf_max):
                return _block("volatility", f"{vf_metric.upper()} {vol_val:.3f} > band max {vf_max}.")
            else:
                result.messages.append(
                    f"  ✅ GATE [volatility] → {vf_metric.upper()} {vol_val:.3f} inside band."
                )
        else:
            result.messages.append("  ⚠️  GATE [volatility] skipped — too few candles.")

    # ── Gate 11: regime ────────────────────────────────────────────────────────
    # Parity with the simulator regime gate: same classify_regime() logic. The
    # live evaluator classifies the latest window (== the simulator's regime
    # label at the final bar). No-raise: skip if the classifier is unavailable.
    regime_allowed = getattr(gates, "regime_filter_allowed", None)
    if regime_allowed:
        allowed_set = {str(r).lower() for r in regime_allowed}
        try:
            from engine.regime import classify_regime
            detected = str(classify_regime(df).get("type", "ranging"))
        except Exception:
            detected = None
        if detected is None:
            result.messages.append("  ⚠️  GATE [regime] skipped — classifier unavailable.")
        elif detected not in allowed_set:
            return _block("regime", f"regime '{detected}' not in allowed {sorted(allowed_set)}.")
        else:
            result.messages.append(f"  ✅ GATE [regime] → regime '{detected}' allowed.")

    # ── Gate 12: relative_strength ──────────────────────────────────────────────
    # Parity with the simulator RS gate: RS = (sym ret) / (ref ret) over window.
    # Needs REF_close merged into the df; skip (no-raise) when absent.
    rs_window = getattr(gates, "rs_filter_window", None)
    if rs_window and "REF_close" in df.columns:
        w = int(rs_window)
        close_s, ref_s = _series(df, "close"), _series(df, "REF_close")
        if close_s is not None and ref_s is not None and len(close_s) > w:
            min_ratio = float(getattr(gates, "rs_filter_min_ratio", 1.0) or 1.0)
            c_now, c_then = float(close_s.iloc[-1]), float(close_s.iloc[-1 - w])
            r_now, r_then = float(ref_s.iloc[-1]), float(ref_s.iloc[-1 - w])
            if c_then > 0 and r_then > 0 and r_now > 0:
                rs_ratio = (c_now / c_then) / (r_now / r_then)
                if rs_ratio < min_ratio:
                    return _block(
                        "relative_strength",
                        f"RS {rs_ratio:.3f} < required {min_ratio} (underperforming reference).",
                    )
                result.messages.append(f"  ✅ GATE [relative_strength] → RS {rs_ratio:.3f} >= {min_ratio}.")
            else:
                result.messages.append("  ⚠️  GATE [relative_strength] skipped — non-positive prices.")
        else:
            result.messages.append("  ⚠️  GATE [relative_strength] skipped — too few candles.")

    # ── Gate 13: event filter (blackout dates) ──────────────────────────────────
    skip_dates = getattr(gates, "event_skip_dates", None)
    if skip_dates:
        skip_set = {str(d)[:10] for d in skip_dates}
        try:
            bar_date = str(df.index[-1])[:10]
        except Exception:
            bar_date = None
        if bar_date and bar_date in skip_set:
            return _block("event", f"date {bar_date} is a configured blackout (earnings/expiry/macro).")
        if bar_date:
            result.messages.append(f"  ✅ GATE [event] → {bar_date} not a blackout date.")

    # ── Gate 14: lunch-lull skip ────────────────────────────────────────────────
    # Parity with the simulator: UTC minute-of-day in [start, end] → block. The
    # simulator computes minutes as index.hour*60+index.minute, so do the same.
    ll_start = getattr(gates, "lunch_lull_start_utc", None)
    ll_end = getattr(gates, "lunch_lull_end_utc", None)
    if ll_start is not None and ll_end is not None:
        try:
            ts = df.index[-1]
            bar_min = int(ts.hour) * 60 + int(ts.minute)
        except Exception:
            bar_min = None
        if bar_min is not None and int(ll_start) <= bar_min <= int(ll_end):
            return _block("lunch_lull", f"bar at {bar_min // 60:02d}:{bar_min % 60:02d} UTC is inside the lunch-lull skip window.")
        if bar_min is not None:
            result.messages.append("  ✅ GATE [lunch_lull] → outside the lunch-lull window.")

    if result.passed:
        result.messages.append("✅ All entry gates passed.")
    return result
