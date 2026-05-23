"""
Strategy YAML loader for the quant engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any

import yaml

from engine.kb_signals import KB_REGISTRY_AVAILABLE

logger = logging.getLogger(__name__)


def _normalise_market(market: str) -> str:
    m = str(market or "").lower().strip()
    if any(token in m for token in ("crypto", "coin", "bitcoin", "btc", "cryptocurrency")):
        return "crypto"
    if any(token in m for token in ("indian_ind", "index", "indices", "nifty", "sensex")):
        return "indian_indices"
    if any(token in m for token in ("indian", "nse", "bse")):
        return "indian_stocks"
    if any(token in m for token in ("us_stock", "nasdaq", "nyse", "american")):
        return "us_stocks"
    return m


def _normalise_objective(obj: str) -> str:
    """Normalise objective to 'intraday' or 'positional'."""
    val = str(obj or "").lower().strip()
    if val in ("intraday", "day_trade", "daytrade", "scalp"):
        return "intraday"
    if val in ("positional", "swing", "long_term", "multiday", "multi_day"):
        return "positional"
    # Default to positional for daily-bar strategies
    return "positional"


def _normalise_legacy_entry_condition(condition: str) -> str:
    if not condition:
        return condition

    return re.sub(
        r"\bCLOSE\s*>\s*HIGH\b",
        "CLOSE > OPENING_RANGE_HIGH(4)",
        condition,
        flags=re.IGNORECASE,
    )


def _to_float(value: Any, field_name: str, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy field '{field_name}' must be numeric.") from exc


def _to_int(value: Any, field_name: str, default: int) -> int:
    if value in (None, ""):
        return default
    # Handle plain int/float directly
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    # Handle strings: only accept purely numeric strings.
    # Descriptive strings like "1 trading day" or "5 trading days" are NOT numeric
    # limits — they are display labels. Return the default (0 = unlimited) for those.
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        # Non-numeric string — silently use the default instead of crashing.
        logger.debug(
            "Strategy field '%s' has non-numeric value %r — using default %s.",
            field_name, value, default,
        )
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy field '{field_name}' must be an integer.") from exc


# Phase 8b — time_exit supports a small whitelist of timezones to keep the
# loader dependency-free (no zoneinfo / tzdata on Windows). NSE doesn't
# observe DST so a fixed offset is correct for the entire trading window.
_TIME_EXIT_TZ_UTC_OFFSET_MINUTES = {
    "Asia/Kolkata":  330,    # IST = UTC+5:30
    "Asia/Calcutta": 330,    # legacy alias
    "UTC":           0,
}
_TIME_EXIT_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$")


def _parse_time_exit_spec(raw: Any) -> dict[str, Any] | None:
    """Parse the optional time_exit block (Phase 8b).

    Shape:
        time_exit:
          exit_time: "15:15"           # HH:MM in the given timezone
          timezone:  "Asia/Kolkata"    # optional; default IST

    Returns a dict including a precomputed `utc_minutes_of_day` so the
    simulator can do a fast int comparison per bar without dragging
    timezone math into the hot loop. None when the block is absent.
    """
    if raw in (None, "", {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("time_exit must be a mapping or omitted entirely.")

    raw_time = str(raw.get("exit_time") or "").strip()
    if not raw_time:
        raise ValueError("time_exit.exit_time is required (format HH:MM).")
    m = _TIME_EXIT_PATTERN.match(raw_time)
    if not m:
        raise ValueError(
            f"time_exit.exit_time must be HH:MM (24-hour), got {raw_time!r}."
        )
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"time_exit.exit_time HH:MM out of range: {raw_time!r}.")

    tz = str(raw.get("timezone") or "Asia/Kolkata").strip()
    offset = _TIME_EXIT_TZ_UTC_OFFSET_MINUTES.get(tz)
    if offset is None:
        raise ValueError(
            f"time_exit.timezone {tz!r} not supported. Use one of "
            f"{sorted(_TIME_EXIT_TZ_UTC_OFFSET_MINUTES)}."
        )

    # Convert HH:MM in local tz to UTC minutes-of-day. Modulo 1440 handles
    # wrap cases (e.g. 02:00 IST = 20:30 UTC the prior day).
    utc_minutes = ((hh * 60 + mm) - offset) % (24 * 60)

    return {
        "exit_time":          raw_time,
        "timezone":           tz,
        "utc_minutes_of_day": utc_minutes,
    }


def _parse_time_to_utc_minutes(raw_time: Any, tz: str = "Asia/Kolkata") -> int | None:
    """Convert an HH:MM local-time string to UTC minutes-of-day. None if absent or malformed."""
    if not raw_time:
        return None
    m = _TIME_EXIT_PATTERN.match(str(raw_time).strip())
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        return None
    offset = _TIME_EXIT_TZ_UTC_OFFSET_MINUTES.get(str(tz).strip(), 330)  # default IST
    return ((hh * 60 + mm) - offset) % (24 * 60)


_STOP_LOSS_TYPES = {"percent", "structural", "atr"}
_STOP_LOSS_ANCHORS = {
    "opening_range_low",   # anchor at LOW of first N bars in the session (long SL)
    "opening_range_high",  # anchor at HIGH of first N bars in the session (short SL — reserved)
    "prev_n_bar_low",      # anchor at MIN(LOW, N) at entry (long SL)
    "prev_n_bar_high",     # anchor at MAX(HIGH, N) at entry (short SL — reserved)
}
_TRAILING_TYPES = {"percent", "atr", "ema", "chandelier"}


def _parse_stop_loss_spec(raw: Any) -> dict[str, Any] | None:
    """Validate the stop_loss block. Returns None when block is absent so the
    simulator falls back to the legacy `stop_loss_percent` field."""
    if raw in (None, "", {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("stop_loss must be a mapping or omitted entirely.")

    sl_type = str(raw.get("type") or "").lower().strip()
    if sl_type not in _STOP_LOSS_TYPES:
        raise ValueError(
            f"stop_loss.type must be one of {sorted(_STOP_LOSS_TYPES)}, got {sl_type!r}"
        )

    spec: dict[str, Any] = {"type": sl_type}

    if sl_type == "percent":
        pct = _to_float(raw.get("pct"), "stop_loss.pct", 0.0)
        if pct <= 0:
            raise ValueError("stop_loss.pct must be > 0 for type=percent")
        spec["pct"] = pct
        return spec

    if sl_type == "structural":
        anchor = str(raw.get("anchor") or "").lower().strip()
        if anchor not in _STOP_LOSS_ANCHORS:
            raise ValueError(
                f"stop_loss.anchor must be one of {sorted(_STOP_LOSS_ANCHORS)}, got {anchor!r}"
            )
        spec["anchor"] = anchor
        spec["padding_pct"] = max(0.0, _to_float(raw.get("padding_pct", 0.0), "stop_loss.padding_pct", 0.0))
        if anchor in {"opening_range_low", "opening_range_high"}:
            opening_bars = _to_int(raw.get("opening_bars", 3), "stop_loss.opening_bars", 3)
            if opening_bars <= 0:
                raise ValueError("stop_loss.opening_bars must be > 0")
            spec["opening_bars"] = opening_bars
        else:  # prev_n_bar_*
            window = _to_int(raw.get("window", 5), "stop_loss.window", 5)
            if window <= 0:
                raise ValueError("stop_loss.window must be > 0")
            spec["window"] = window
        return spec

    if sl_type == "atr":
        multiplier = _to_float(raw.get("multiplier", 1.5), "stop_loss.multiplier", 1.5)
        window = _to_int(raw.get("window", 14), "stop_loss.window", 14)
        if multiplier <= 0:
            raise ValueError("stop_loss.multiplier must be > 0 for type=atr")
        if window <= 0:
            raise ValueError("stop_loss.window must be > 0 for type=atr")
        spec["multiplier"] = multiplier
        spec["window"] = window
        return spec

    return None  # unreachable — sl_type already validated above


def _parse_trailing_stop_spec(raw: Any) -> dict[str, Any] | None:
    """Validate the trailing_stop block. Returns None when absent (no trailing)."""
    if raw in (None, "", {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("trailing_stop must be a mapping or omitted entirely.")

    ts_type = str(raw.get("type") or "").lower().strip()
    if ts_type not in _TRAILING_TYPES:
        raise ValueError(
            f"trailing_stop.type must be one of {sorted(_TRAILING_TYPES)}, got {ts_type!r}"
        )

    spec: dict[str, Any] = {"type": ts_type}
    # Common: only start trailing once the trade has earned at least this %.
    activate = _to_float(raw.get("activate_after_pct", 0.0), "trailing_stop.activate_after_pct", 0.0)
    spec["activate_after_pct"] = max(0.0, activate)

    if ts_type == "percent":
        distance = _to_float(raw.get("distance_pct"), "trailing_stop.distance_pct", 0.0)
        if distance <= 0:
            raise ValueError("trailing_stop.distance_pct must be > 0 for type=percent")
        spec["distance_pct"] = distance
        return spec

    if ts_type in {"atr", "chandelier"}:
        multiplier = _to_float(raw.get("multiplier", 2.0), "trailing_stop.multiplier", 2.0)
        window = _to_int(raw.get("window", 14), "trailing_stop.window", 14)
        if multiplier <= 0:
            raise ValueError("trailing_stop.multiplier must be > 0")
        if window <= 0:
            raise ValueError("trailing_stop.window must be > 0")
        spec["multiplier"] = multiplier
        spec["window"] = window
        return spec

    if ts_type == "ema":
        window = _to_int(raw.get("window", 20), "trailing_stop.window", 20)
        if window <= 0:
            raise ValueError("trailing_stop.window must be > 0 for type=ema")
        spec["window"] = window
        return spec

    return None  # unreachable


def _parse_htf_rules(raw: Any) -> tuple[HtfRule, ...]:
    """Parse the htf: block (Phase 5) into a tuple of HtfRule.

    Empty/missing yields an empty tuple. Each entry must declare both a
    timeframe (e.g. "1d") and a condition string. Validation here means
    malformed declarations fail at load-time, not silently mid-backtest.
    """
    if raw in (None, "", [], {}):
        return ()
    if isinstance(raw, dict):
        # Single-rule shortcut: htf: { timeframe: 1d, condition: "..." }
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("htf must be a list of {timeframe, condition} mappings.")

    rules: list[HtfRule] = []
    seen_timeframes: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each htf entry must be a mapping with timeframe + condition.")
        tf = str(item.get("timeframe") or "").strip().lower()
        cond = str(item.get("condition") or "").strip()
        if not tf:
            raise ValueError("htf entry missing 'timeframe'.")
        if not cond:
            raise ValueError(f"htf entry for timeframe={tf!r} missing 'condition'.")
        if tf in seen_timeframes:
            raise ValueError(f"duplicate htf timeframe {tf!r} — declare each HTF only once.")
        seen_timeframes.add(tf)
        rules.append(HtfRule(timeframe=tf, condition=cond))
    return tuple(rules)


def _normalise_signal_rules(raw: Any) -> list[dict[str, Any]]:
    """Parse entry_signals / exit_signals lists from strategy YAML."""
    if not raw:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        params = item.get("params")
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"Signal '{name}' params must be a mapping.")
        out.append({"name": name, "params": dict(params or {})})
    return out


def _infer_max_holding_candles(objective: str, timeframe: str) -> int | None:
    """
    Derive a sensible max_holding_candles from objective + timeframe.

    NSE/BSE session: 9:15 AM – 3:30 PM IST = 375 minutes (NOT 390 which is the US market).

    Intraday:
      - 1m  → 375  (full NSE session = 375 one-minute bars)
      - 5m  → 75   (375 / 5)
      - 10m → 37   (375 / 10, floor)
      - 15m → 25   (375 / 15)
      - 30m → 12   (375 / 30, floor)
      - 1h  → 6    (375 / 60, floor)
      - 1d  → 1    (must close same day bar)

    Positional:
      - 1d  → 20   (~1 month of trading days)
      - 1h  → 40   (~1 week of hourly bars)
      - sub-hour → None (let stop/TP do the work)
    """
    tf = str(timeframe or "1d").lower().strip()
    obj = str(objective or "positional").lower()

    if obj == "intraday":
        _bars_per_session = {
            "1m": 375, "5m": 75, "10m": 37, "15m": 25,
            "30m": 12, "1h": 6, "1d": 1,
        }
        return _bars_per_session.get(tf, 25)

    # Positional
    if tf == "1d":
        return 20   # ~1 calendar month
    if tf in ("1h", "2h", "4h"):
        return 40   # ~1 week of hourly bars
    # Sub-hour positional: no hard cap, rely on stop/TP
    return None


@dataclass(frozen=True)
class HtfRule:
    """One higher-timeframe entry gate (Phase 5).

    Each rule pairs a higher timeframe (e.g. "1d") with a condition string
    evaluated against THAT timeframe's OHLCV. The simulator only allows an
    entry when every HTF rule passes on the most recently completed HTF bar
    at the time of the LTF entry signal — strict no-look-ahead.
    """
    timeframe: str
    condition: str


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    symbol: str
    market: str
    timeframe: str
    objective: str              # 'intraday' | 'positional'
    entry_condition: str
    exit_condition: str
    indicators: dict[str, Any]
    stop_loss: float
    take_profit: float
    daily_loss_cap_pct: float   # Circuit-breaker: max daily portfolio loss
    max_trades_per_day: int     # 0 = unlimited
    max_holding_candles: int | None   # None = unlimited
    # When entry_evaluation_mode / exit_evaluation_mode is "registry", conditions are
    # evaluated via RuleRegistry.evaluate() — any registered KB signal may be used.
    entry_evaluation_mode: str = "formula"   # "formula" | "registry"
    exit_evaluation_mode: str = "formula"
    entry_signal_rules: list[dict[str, Any]] | None = None
    exit_signal_rules: list[dict[str, Any]] | None = None
    # Phase 3 — Structural SL + trailing SL specs.
    # Both are optional dicts; when omitted the simulator uses the legacy
    # `stop_loss` percent (initial SL) with no trailing.
    #
    # stop_loss_spec:
    #   {"type": "percent",    "pct": 1.5}
    #   {"type": "structural", "anchor": "opening_range_low", "opening_bars": 3, "padding_pct": 0.1}
    #   {"type": "structural", "anchor": "prev_n_bar_low",    "window": 5,        "padding_pct": 0.0}
    #   {"type": "atr",        "multiplier": 1.5, "window": 14}
    #
    # trailing_stop_spec:
    #   {"type": "percent",     "distance_pct": 0.8, "activate_after_pct": 0.5}
    #   {"type": "atr",         "multiplier": 2.0,   "window": 14}
    #   {"type": "ema",         "window": 20}
    #   {"type": "chandelier",  "multiplier": 3.0,   "window": 14}
    stop_loss_spec: dict[str, Any] | None = None
    trailing_stop_spec: dict[str, Any] | None = None
    # Phase 4 — optional reference symbol whose OHLCV is fetched alongside the
    # trade symbol and joined into the main df with a REF_ prefix. This is what
    # lets formulas reference Nifty/Bank-Nifty/sector indices via REF_CLOSE,
    # RS(n), etc. Caller (the API layer) is responsible for fetching the
    # reference series and passing it into run_backtest().
    reference_symbol: str | None = None
    # Phase 5 — optional higher-timeframe entry gates. Each rule says
    # "the entry signal only counts if THIS condition holds on the last
    # CLOSED bar of THIS timeframe". Empty tuple = no HTF gating.
    htf_rules: tuple[HtfRule, ...] = ()
    # Phase 6 — optional structural-pattern parameter overrides. Maps
    # pattern_name (e.g. "swing_high", "bos_bullish") to a params dict.
    # Defaults are used when keys are missing or the block is absent.
    # Patterns are auto-detected from condition formulas; this block only
    # tunes their parameters (e.g. swing window=7 instead of default 5).
    patterns: dict[str, dict[str, Any]] | None = None
    # Phase 8b — optional wall-clock intraday cutoff. When set, the simulator
    # force-exits any open position when a bar's timestamp falls at or after
    # the configured time, AND blocks new entries past that time. Format:
    #   {"exit_time": "15:15", "timezone": "Asia/Kolkata",
    #    "utc_minutes_of_day": 585}   (utc_minutes precomputed by the loader)
    time_exit_spec: dict[str, Any] | None = None
    # Phase 10 — direction constraint
    direction: str = "both"   # "long_only" | "short_only" | "both"
    # Phase 10 — session entry window (precomputed UTC minutes of day).
    # entry_window_start_utc blocks entries BEFORE this time (e.g. 09:30 = 4*60+30=270 UTC).
    # entry_window_end_utc blocks NEW entries AFTER this time (does not force-exit).
    # Both are independent of time_exit_spec (which force-exits open trades).
    entry_window_start_utc: int | None = None
    entry_window_end_utc: int | None = None
    # Phase 10 — risk circuit breakers
    max_consecutive_losses: int = 0       # 0 = disabled; stop trading after N consecutive losses
    cooldown_bars_after_loss: int = 0     # 0 = disabled; wait N bars after a loss before re-entering
    cooldown_bars_after_profit: int = 0   # 0 = disabled; wait N bars after a profit before re-entering
    # Phase 10 — entry gate controls
    max_spread_bps: float = 0.0           # 0 = disabled; reject entry if estimated spread > N bps
    gap_filter: str = "none"              # "none" | "ignore_gap_up" | "ignore_gap_down" | "ignore_both"
    gap_threshold_pct: float = 0.5        # % open vs prev close that counts as a gap
    entry_confirmation_bars: int = 1      # signal must hold for N consecutive closed bars before entry
    rsi_entry_band_min: float | None = None   # RSI must be >= this to enter (None = no gate)
    rsi_entry_band_max: float | None = None   # RSI must be <= this to enter (None = no gate)
    volume_ratio_threshold: float | None = None  # volume must be >= N × avg volume (None = no gate)
    # Phase 10 — position sizing
    position_sizing_mode: str = "fixed_fractional"   # "fixed_fractional" | "risk_based" | "fixed_units"
    max_capital_allocation_pct: float = 100.0        # max % of balance in one trade (100 = no cap)
    per_trade_risk_pct: float = 2.0                  # % of balance to risk per trade (for risk_based mode)


def _build_strategy_config(strategy: dict[str, Any]) -> StrategyConfig:
    variables = strategy.get("variables", {}) or {}
    risk = strategy.get("risk_management", {}) or {}
    profile = strategy.get("profile", {}) or {}

    entry_signal_rules = _normalise_signal_rules(strategy.get("entry_signals"))
    exit_signal_rules = _normalise_signal_rules(strategy.get("exit_signals"))

    raw_entry_mode = str(strategy.get("entry_evaluation_mode") or "").lower().strip()
    raw_exit_mode = str(strategy.get("exit_evaluation_mode") or "").strip().lower()

    if raw_entry_mode in ("formula", "registry"):
        entry_evaluation_mode = raw_entry_mode
    else:
        entry_evaluation_mode = "registry" if entry_signal_rules else "formula"

    if raw_exit_mode in ("formula", "registry"):
        exit_evaluation_mode = raw_exit_mode
    else:
        exit_evaluation_mode = "registry" if exit_signal_rules else "formula"

    # ── Registry availability fallback ────────────────────────────────────────
    # The strategy YAML always contains both registry signal rules AND formula
    # condition strings.  When the KB package cannot be imported in this
    # environment (e.g. separate quant-engine container), silently fall back to
    # the formula string so the backtest still runs instead of crashing.
    if entry_evaluation_mode == "registry" and not KB_REGISTRY_AVAILABLE:
        fallback = str(
            strategy.get("entry_condition")
            or (strategy.get("entry") or {}).get("condition")
            or ""
        ).strip()
        if fallback:
            logger.warning(
                "loader|entry_mode=registry but stretus_kb unavailable — "
                "falling back to formula mode | condition=%r", fallback,
            )
            entry_evaluation_mode = "formula"

    if exit_evaluation_mode == "registry" and not KB_REGISTRY_AVAILABLE:
        fallback = str(
            strategy.get("exit_condition")
            or (strategy.get("exit") or {}).get("condition")
            or ""
        ).strip()
        if fallback:
            logger.warning(
                "loader|exit_mode=registry but stretus_kb unavailable — "
                "falling back to formula mode | condition=%r", fallback,
            )
            exit_evaluation_mode = "formula"

    entry_condition = _normalise_legacy_entry_condition(
        str(strategy.get("entry_condition") or strategy.get("entry", {}).get("condition") or "").strip()
    )
    exit_condition = str(
        strategy.get("exit_condition")
        or strategy.get("exit", {}).get("condition")
        or "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
    ).strip()

    stop_loss = _to_float(
        risk.get("stop_loss_percent", strategy.get("stop_loss_pct", variables.get("STOP_LOSS_TARGET"))),
        "stop_loss_percent",
        2.0,
    )
    take_profit = _to_float(
        risk.get("take_profit_percent", strategy.get("take_profit_pct", variables.get("TAKE_PROFIT_TARGET"))),
        "take_profit_percent",
        5.0,
    )

    if stop_loss <= 0:
        raise ValueError("stop_loss_percent must be greater than zero.")
    if take_profit <= 0:
        raise ValueError("take_profit_percent must be greater than zero.")

    if entry_evaluation_mode == "registry":
        if not entry_signal_rules:
            raise ValueError(
                "entry_evaluation_mode is 'registry' but entry_signals is missing or empty."
            )
    elif not entry_condition:
        raise ValueError("Strategy YAML is missing a non-empty entry condition.")

    if exit_evaluation_mode == "registry" and not exit_signal_rules:
        raise ValueError(
            "exit_evaluation_mode is 'registry' but exit_signals is missing or empty."
        )

    indicators = strategy.get("indicators", {}) or {}
    if not isinstance(indicators, dict):
        raise ValueError("Strategy indicators must be a mapping.")

    # Objective parsing
    raw_objective = (
        profile.get("objective")
        or strategy.get("objective")
        or "positional"
    )
    objective = _normalise_objective(str(raw_objective or "positional"))

    timeframe = str(strategy.get("timeframe") or "1d").strip()

    # Daily loss cap: profile > strategy-level > default by objective
    daily_loss_cap_pct = _to_float(
        profile.get("daily_loss_cap_pct") or strategy.get("daily_loss_cap_pct"),
        "daily_loss_cap_pct",
        2.0 if objective == "intraday" else 3.0,
    )

    # Max trades per day (intraday only makes sense).
    # Support both "max_trades_per_day" and the shorthand "max_trade" from the strategy draft.
    # IMPORTANT: use `is not None` checks, NOT `or`, because 0 is a valid value (= unlimited)
    # and `0 or next_value` would incorrectly skip an explicit 0.
    def _first_not_none(*vals: Any) -> Any:
        for v in vals:
            if v is not None:
                return v
        return None

    raw_max_trades = _first_not_none(
        profile.get("max_trades_per_day"),   # explicit numeric field (preferred)
        profile.get("max_trade"),             # display label alias, e.g. "10" or "1 trading day"
        strategy.get("max_trades_per_day"),
        strategy.get("max_trade"),
    )
    max_trades_per_day = _to_int(raw_max_trades, "max_trades_per_day", 0)

    # Max holding candles — YAML can override, otherwise derive from objective+timeframe
    raw_max_holding = (
        profile.get("max_holding_candles")
        or strategy.get("max_holding_candles")
        or risk.get("max_holding_candles")
    )
    if raw_max_holding is not None:
        max_holding_candles: int | None = _to_int(raw_max_holding, "max_holding_candles", 0) or None
    else:
        max_holding_candles = _infer_max_holding_candles(objective, timeframe)

    # Phase 3 — structural SL + trailing SL specs (both optional).
    # When stop_loss block is absent, fall back to the legacy stop_loss percent.
    stop_loss_spec = _parse_stop_loss_spec(
        strategy.get("stop_loss") or risk.get("stop_loss")
    )
    trailing_stop_spec = _parse_trailing_stop_spec(
        strategy.get("trailing_stop") or risk.get("trailing_stop")
    )

    # Phase 4 — optional reference symbol. Whitespace-strip and uppercase so
    # "^nsei" and "^NSEI" are equivalent. None when omitted.
    raw_ref = strategy.get("reference_symbol") or strategy.get("benchmark_symbol")
    reference_symbol = str(raw_ref).strip().upper() if raw_ref else None
    if reference_symbol == "":
        reference_symbol = None

    # Phase 5 — optional higher-timeframe entry gates.
    htf_rules = _parse_htf_rules(strategy.get("htf") or strategy.get("higher_timeframes"))

    # Phase 6 — optional structural-pattern parameter overrides.
    raw_patterns = strategy.get("patterns")
    if raw_patterns is None:
        patterns_overrides: dict[str, dict[str, Any]] | None = None
    elif isinstance(raw_patterns, dict):
        patterns_overrides = {}
        for name, params in raw_patterns.items():
            if not isinstance(params, dict):
                raise ValueError(f"patterns.{name} must be a mapping of parameters.")
            patterns_overrides[str(name).strip().lower()] = dict(params)
    else:
        raise ValueError("patterns must be a mapping of pattern_name -> params, or omitted.")

    # Phase 8b — optional wall-clock intraday cutoff.
    time_exit_spec = _parse_time_exit_spec(
        strategy.get("time_exit") or (strategy.get("exits") or {}).get("time_exit")
    )

    # Phase 10 — direction
    raw_direction = str(strategy.get("direction") or profile.get("direction") or "both").lower().strip()
    direction = raw_direction if raw_direction in ("long_only", "short_only", "both") else "both"

    # Phase 10 — entry window (from YAML "entry_window" block or flat fields)
    entry_window_raw = strategy.get("entry_window") or {}
    entry_window_start_utc: int | None = None
    entry_window_end_utc:   int | None = None
    if isinstance(entry_window_raw, dict):
        entry_window_start_utc = _parse_time_to_utc_minutes(
            entry_window_raw.get("start") or entry_window_raw.get("start_time"),
            entry_window_raw.get("timezone", "Asia/Kolkata"),
        )
        entry_window_end_utc = _parse_time_to_utc_minutes(
            entry_window_raw.get("end") or entry_window_raw.get("end_time"),
            entry_window_raw.get("timezone", "Asia/Kolkata"),
        )

    # Phase 10 — risk circuit breakers
    max_consecutive_losses  = _to_int(
        _first_not_none(strategy.get("max_consecutive_losses"), profile.get("max_consecutive_losses")),
        "max_consecutive_losses", 0,
    )
    cooldown_bars_after_loss = _to_int(
        _first_not_none(strategy.get("cooldown_bars_after_loss"), profile.get("cooldown_bars_after_loss")),
        "cooldown_bars_after_loss", 0,
    )
    cooldown_bars_after_profit = _to_int(
        _first_not_none(strategy.get("cooldown_bars_after_profit"), profile.get("cooldown_bars_after_profit")),
        "cooldown_bars_after_profit", 0,
    )

    # Phase 10 — entry gate controls
    max_spread_bps = _to_float(
        _first_not_none(strategy.get("max_spread_bps"), profile.get("max_spread_bps")),
        "max_spread_bps", 0.0,
    )
    raw_gap_filter = str(
        _first_not_none(strategy.get("gap_filter"), profile.get("gap_filter")) or "none"
    ).lower().strip()
    gap_filter = raw_gap_filter if raw_gap_filter in (
        "none", "ignore_gap_up", "ignore_gap_down", "ignore_both"
    ) else "none"
    gap_threshold_pct = _to_float(
        _first_not_none(strategy.get("gap_threshold_pct"), profile.get("gap_threshold_pct")),
        "gap_threshold_pct", 0.5,
    )
    entry_confirmation_bars = _to_int(
        _first_not_none(strategy.get("entry_confirmation_bars"), profile.get("entry_confirmation_bars")),
        "entry_confirmation_bars", 1,
    )
    rsi_band_raw = strategy.get("rsi_entry_band") or {}
    rsi_entry_band_min: float | None = None
    rsi_entry_band_max: float | None = None
    if isinstance(rsi_band_raw, dict):
        _rmin = rsi_band_raw.get("min")
        _rmax = rsi_band_raw.get("max")
        rsi_entry_band_min = float(_rmin) if _rmin is not None else None
        rsi_entry_band_max = float(_rmax) if _rmax is not None else None
    else:
        # also accept flat fields
        _rmin = _first_not_none(strategy.get("rsi_entry_band_min"), profile.get("rsi_entry_band_min"))
        _rmax = _first_not_none(strategy.get("rsi_entry_band_max"), profile.get("rsi_entry_band_max"))
        rsi_entry_band_min = float(_rmin) if _rmin is not None else None
        rsi_entry_band_max = float(_rmax) if _rmax is not None else None
    _vrt = _first_not_none(strategy.get("volume_ratio_threshold"), profile.get("volume_ratio_threshold"))
    volume_ratio_threshold: float | None = float(_vrt) if _vrt is not None else None

    # Phase 10 — position sizing
    raw_psm = str(
        _first_not_none(strategy.get("position_sizing_mode"), profile.get("position_sizing_mode")) or "fixed_fractional"
    ).lower().strip()
    position_sizing_mode = raw_psm if raw_psm in (
        "fixed_fractional", "risk_based", "fixed_units"
    ) else "fixed_fractional"
    max_capital_allocation_pct = _to_float(
        _first_not_none(strategy.get("max_capital_allocation_pct"), profile.get("max_capital_allocation_pct")),
        "max_capital_allocation_pct", 100.0,
    )
    per_trade_risk_pct = _to_float(
        _first_not_none(strategy.get("per_trade_risk_pct"), profile.get("per_trade_risk_pct")),
        "per_trade_risk_pct", 2.0,
    )

    config = StrategyConfig(
        name=str(strategy.get("name") or "Unknown Strategy"),
        symbol=str(strategy.get("symbol") or "").strip(),
        market=_normalise_market(str(strategy.get("market") or "")),
        timeframe=timeframe,
        objective=objective,
        entry_condition=entry_condition,
        exit_condition=exit_condition,
        indicators=indicators,
        stop_loss=stop_loss,
        take_profit=take_profit,
        daily_loss_cap_pct=daily_loss_cap_pct,
        max_trades_per_day=max_trades_per_day,
        max_holding_candles=max_holding_candles,
        entry_evaluation_mode=entry_evaluation_mode,
        exit_evaluation_mode=exit_evaluation_mode,
        entry_signal_rules=entry_signal_rules or None,
        exit_signal_rules=exit_signal_rules or None,
        stop_loss_spec=stop_loss_spec,
        trailing_stop_spec=trailing_stop_spec,
        reference_symbol=reference_symbol,
        htf_rules=htf_rules,
        patterns=patterns_overrides,
        time_exit_spec=time_exit_spec,
        direction=direction,
        entry_window_start_utc=entry_window_start_utc,
        entry_window_end_utc=entry_window_end_utc,
        max_consecutive_losses=max_consecutive_losses,
        cooldown_bars_after_loss=cooldown_bars_after_loss,
        cooldown_bars_after_profit=cooldown_bars_after_profit,
        max_spread_bps=max_spread_bps,
        gap_filter=gap_filter,
        gap_threshold_pct=gap_threshold_pct,
        entry_confirmation_bars=max(1, entry_confirmation_bars),
        rsi_entry_band_min=rsi_entry_band_min,
        rsi_entry_band_max=rsi_entry_band_max,
        volume_ratio_threshold=volume_ratio_threshold,
        position_sizing_mode=position_sizing_mode,
        max_capital_allocation_pct=max_capital_allocation_pct,
        per_trade_risk_pct=per_trade_risk_pct,
    )

    logger.info(
        "Loaded strategy config | name=%s symbol=%s timeframe=%s market=%s objective=%s "
        "max_holding_candles=%s daily_loss_cap_pct=%.1f max_trades_per_day=%s",
        config.name,
        config.symbol,
        config.timeframe,
        config.market,
        config.objective,
        config.max_holding_candles,
        config.daily_loss_cap_pct,
        config.max_trades_per_day,
    )
    return config


def load_strategy(yaml_path: str) -> StrategyConfig:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy YAML not found: {yaml_path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "strategy" not in data:
        raise ValueError(f"Invalid strategy YAML: missing 'strategy' key in {yaml_path}")

    strategy = data.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError(f"Invalid strategy YAML: 'strategy' must be an object in {yaml_path}")

    return _build_strategy_config(strategy)


def load_strategy_from_content(yaml_content: str) -> StrategyConfig:
    """Parse a strategy directly from YAML text (no filesystem access required)."""
    data = yaml.safe_load(yaml_content)

    if not isinstance(data, dict) or "strategy" not in data:
        raise ValueError("Invalid strategy YAML: missing 'strategy' key.")

    strategy = data.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("Invalid strategy YAML: 'strategy' must be an object.")

    return _build_strategy_config(strategy)
