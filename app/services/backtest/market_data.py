"""
Market-data request extraction, validation, and OHLCV normalization helpers.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Any
import logging

import httpx
import yaml

from app.core.config import get_settings
from app.core.timing import time_step
from app.schemas.backtest import BacktestTriggerRequest
from app.services.backtest.market_data_grpc import (
    GRPC_OHLCV_INTERVALS,
    fetch_ohlcv_chunk_grpc,
    open_grpc_channel,
    use_grpc_transport,
)
from app.services.backtest.ohlcv_cache import (
    get_cached_records,
    save_records as save_ohlcv_cache,
)
from app.services.strategy.builder import (
    SUPPORTED_USER_TIMEFRAMES,
    UNSUPPORTED_USER_TIMEFRAME_MESSAGE,
    _timeframe_to_minutes,
    resolve_supported_user_timeframe,
)
logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_MARKET_DATA_INTERVALS = tuple(SUPPORTED_USER_TIMEFRAMES)
OHLCV_ENDPOINT_PATH = "/api/v1/marketdata/ohlcv"
OHLCV_SLOT_FETCH_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class StrategyMarketDataRequest:
    yaml_path: str
    raw_symbol: str
    symbol: str
    interval: str
    from_utc: str
    to_utc: str
    # True when the caller supplied both from_utc and to_utc (custom range).
    user_specified_window: bool = False


@dataclass(frozen=True)
class OhlcvFetchSlot:
    index: int
    from_utc: str
    to_utc: str
    label: str = ""

# ── Lookback computation ───────────────────────────────────────────────────────

# Bars per calendar day for NSE/BSE, computed from the timeframe rather than a
# fixed lookup so ANY interval (2m, 7m, 45m, 4h, …) gets a correct value.
# Indian market session: 9:15 AM – 3:30 PM IST = 375 minutes (NOT 390 = US).
_NSE_SESSION_MINUTES = 375.0


def _bars_per_calendar_day(interval: str) -> float:
    """Approx. number of bars one NSE trading day yields at ``interval``.

    Intraday: session_minutes / timeframe_minutes (e.g. 5m → 75, 45m → 8.33).
    Daily-or-coarser: 1 bar per day. Defensive fallback for unparseable input.
    """
    minutes = _timeframe_to_minutes(interval) or 0
    if minutes <= 0:
        return 26.0
    if minutes >= 1440:
        return 1.0
    return _NSE_SESSION_MINUTES / minutes


def _interval_buffer_days(interval: str, *, daily_default: int = 90) -> int:
    """Extra calendar-day warm-up buffer; coarser timeframes need a wider window
    because each day yields fewer bars. Monotonic in the timeframe so arbitrary
    intervals slot in sensibly (e.g. 45m → 21, 7m → 7)."""
    minutes = _timeframe_to_minutes(interval) or 0
    if minutes >= 1440:
        return daily_default
    if minutes >= 60:
        return 21
    if minutes >= 15:
        return 14
    return 7

# Warm-up safety multiplier: we want 2× the indicator warm-up as eligible candles
_WARMUP_SAFETY_FACTOR = 2


def _extract_max_indicator_period(indicators: dict) -> int:
    """
    Given an indicators dict like {"SMA": [20, 50], "RSI": [14]},
    return the maximum look-back period across all indicators.
    """
    max_period = 0
    for indicator, periods in (indicators or {}).items():
        ind = str(indicator).upper()
        if ind in ("SMA", "EMA", "RSI", "BB_UPPER", "BB_LOWER", "BB", "ATR"):
            for n in (periods or []):
                try:
                    max_period = max(max_period, int(n))
                except (TypeError, ValueError):
                    pass
        elif ind == "MACD":
            max_period = max(max_period, 26 + 9)  # EMA(26) + signal(9)
    return max_period


def _max_warmup_bars(strategy: dict[str, Any], indicators: dict) -> int:
    """Maximum indicator/signal warm-up bars from YAML indicators and KB signal rules."""
    max_period = _extract_max_indicator_period(indicators)
    try:
        from engine.kb_signals import estimate_kb_warmup

        entry_rules = strategy.get("entry_signals") or strategy.get("entry_signal_rules") or []
        exit_rules = strategy.get("exit_signals") or strategy.get("exit_signal_rules") or []
        if entry_rules:
            max_period = max(max_period, estimate_kb_warmup(entry_rules))
        if exit_rules:
            max_period = max(max_period, estimate_kb_warmup(exit_rules))
    except ImportError:
        pass
    return max_period


def _compute_user_range_padding_days(
    interval: str,
    indicators: dict,
    *,
    strategy: dict[str, Any] | None = None,
    objective: str = "positional",
) -> int:
    """
    Calendar days to fetch *before* a user-specified simulation window for warm-up.

    Capped so a short custom range does not pull the full default history.
    """
    del objective
    strategy_payload = strategy or {}
    max_period = _max_warmup_bars(strategy_payload, indicators)
    bars_per_day = _bars_per_calendar_day(interval)
    buffer_days = _interval_buffer_days(interval)
    min_pad = max(7, settings.backtest_user_range_min_padding_days)
    max_pad = max(min_pad, settings.backtest_user_range_max_padding_days)

    if max_period <= 0:
        padding = min_pad
    else:
        total_candles_needed = max_period * _WARMUP_SAFETY_FACTOR
        padding = int(total_candles_needed / bars_per_day) + buffer_days
        padding = max(min_pad, padding)

    result = min(padding, max_pad)
    logger.info(
        "📏 User-range padding | interval=%s max_warmup_bars=%s padding_days=%s "
        "(min=%s max=%s)",
        interval,
        max_period,
        result,
        min_pad,
        max_pad,
    )
    return result


def _compute_required_lookback_days(
    interval: str,
    indicators: dict,
    objective: str = "positional",
) -> int:
    """
    Legacy full-window lookback (default backtest range). Prefer
    ``_compute_user_range_padding_days`` for user-specified windows.
    """
    max_period = _extract_max_indicator_period(indicators)
    bars_per_day = _bars_per_calendar_day(interval)
    buffer_days = _interval_buffer_days(interval)

    if max_period <= 0:
        return max(settings.backtest_default_lookback_days, 90)

    total_candles_needed = max_period * _WARMUP_SAFETY_FACTOR
    required_calendar_days = int(total_candles_needed / bars_per_day) + buffer_days

    floor = max(settings.backtest_default_lookback_days, 90)
    result = max(required_calendar_days, floor)

    logger.info(
        "📏 Lookback computed | interval=%s max_indicator_period=%s bars_per_day=%.1f "
        "required_days=%s (floor=%s)",
        interval,
        max_period,
        bars_per_day,
        result,
        floor,
    )
    return result


def _default_backtest_window(
    interval: str = "1d",
    indicators: dict | None = None,
    objective: str = "positional",
) -> tuple[str, str]:
    """Return the full configured backtest date range.

    The full range is always fetched — HTTP 429 is avoided by chunked fetching
    inside fetch_ohlcv_records(), not by reducing the range.
    """
    from quant_engine.engine.config import (
        BACKTEST_MARKET_DATA_FROM_UTC,
        BACKTEST_MARKET_DATA_TO_UTC,
    )
    _ = interval, indicators, objective
    return BACKTEST_MARKET_DATA_FROM_UTC, BACKTEST_MARKET_DATA_TO_UTC


def _resolve_market_data_endpoint(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("HISTORICAL_DATA_URL is not configured.")
    if normalized.endswith(OHLCV_ENDPOINT_PATH):
        return normalized
    return f"{normalized}{OHLCV_ENDPOINT_PATH}"


def _to_utc_iso(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _load_strategy_yaml(yaml_path: str) -> dict[str, Any]:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy YAML not found: {yaml_path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict) or "strategy" not in payload:
        raise ValueError(f"Invalid strategy YAML: missing 'strategy' section in {yaml_path}")

    strategy = payload.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError(f"Invalid strategy YAML: 'strategy' must be an object in {yaml_path}")

    return strategy


def _normalize_symbol_for_market_data(raw_symbol: str) -> str:
    """Project a KB-canonical symbol into the broker-native form market data expects.

      RELIANCE.NS     → RELIANCE   (strip Indian-exchange suffix)
      NSE:RELIANCE    → RELIANCE   (strip venue prefix)
      BTC_USDT        → BTCUSDT    (strip underscore on canonical crypto pairs)
    """
    symbol = str(raw_symbol or "").strip().upper()
    if not symbol:
        return ""

    if ":" in symbol:
        _, symbol = symbol.split(":", 1)

    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        symbol = symbol[:-3]

    # Handle underscore variant stored in DB (e.g. RELIANCE_NS, HDFC_BO).
    if symbol.endswith("_NS") or symbol.endswith("_BO"):
        symbol = symbol[:-3]

    # Canonical crypto pair (BASE_QUOTE) → broker-native (BASEQUOTE).
    # Guarded by a strict regex so we never accidentally collapse a name with
    # a stray underscore that happens to look like a pair.
    if re.fullmatch(r"[A-Z0-9]+_[A-Z0-9]+", symbol):
        symbol = symbol.replace("_", "")

    return symbol


def _is_canonical_crypto_symbol(raw_symbol: str) -> bool:
    """True if raw_symbol is the KB-canonical BASE_QUOTE form (e.g. BTC_USDT)."""
    return bool(re.fullmatch(r"[A-Z0-9]+_[A-Z0-9]+", str(raw_symbol or "").upper().strip()))


def _chunk_days_for(raw_symbol: str, settings_obj) -> int:
    """Pick the OHLCV fetch chunk size per asset class.

    Crypto venues (24/7 markets, dense candles) reject the equity-sized window,
    so we use a smaller default. The raw KB symbol (BTC_USDT) is the signal
    because the normalized form (BTCUSDT) has already collapsed the underscore.
    """
    if _is_canonical_crypto_symbol(raw_symbol):
        return settings_obj.backtest_fetch_chunk_days_crypto
    return settings_obj.backtest_fetch_chunk_days


def _normalize_interval(raw_interval: str) -> str:
    resolved_interval, validation_message = resolve_supported_user_timeframe(raw_interval)
    if not resolved_interval:
        raise ValueError(validation_message or UNSUPPORTED_USER_TIMEFRAME_MESSAGE)
    return resolved_interval


def _parse_utc_timestamp(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp.") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _half_year_boundary_end(cursor: datetime) -> datetime:
    if cursor.month <= 6:
        return datetime(cursor.year, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    return datetime(cursor.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def _next_half_year_start(slot_end: datetime) -> datetime:
    if slot_end.month <= 6:
        return datetime(slot_end.year, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    return datetime(slot_end.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _month_boundary_end(cursor: datetime) -> datetime:
    if cursor.month == 12:
        next_month = datetime(cursor.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        next_month = datetime(cursor.year, cursor.month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return next_month - timedelta(seconds=1)


def _build_ohlcv_fetch_slots(from_utc: str, to_utc: str) -> list[OhlcvFetchSlot]:
    start_dt = _parse_utc_timestamp(from_utc, "from_utc")
    end_dt = _parse_utc_timestamp(to_utc, "to_utc")
    if start_dt is None or end_dt is None:
        raise ValueError("from_utc and to_utc are required for OHLCV slot fetches.")
    if start_dt >= end_dt:
        raise ValueError("from_utc must be earlier than to_utc.")

    slots: list[OhlcvFetchSlot] = []
    cursor = start_dt
    while cursor <= end_dt:
        slot_end = min(_half_year_boundary_end(cursor), end_dt)
        slots.append(
            OhlcvFetchSlot(
                index=len(slots) + 1,
                from_utc=_to_utc_iso(cursor),
                to_utc=_to_utc_iso(slot_end),
            )
        )
        if slot_end >= end_dt:
            break
        cursor = _next_half_year_start(slot_end)

    return slots


def _build_monthly_fallback_slots(slot: OhlcvFetchSlot) -> list[OhlcvFetchSlot]:
    start_dt = _parse_utc_timestamp(slot.from_utc, "from_utc")
    end_dt = _parse_utc_timestamp(slot.to_utc, "to_utc")
    if start_dt is None or end_dt is None:
        return []

    fallback_slots: list[OhlcvFetchSlot] = []
    cursor = start_dt
    while cursor <= end_dt:
        month_end = min(_month_boundary_end(cursor), end_dt)
        fallback_slots.append(
            OhlcvFetchSlot(
                index=slot.index,
                from_utc=_to_utc_iso(cursor),
                to_utc=_to_utc_iso(month_end),
                label=f"{slot.index}.{len(fallback_slots) + 1}",
            )
        )
        if month_end >= end_dt:
            break
        cursor = month_end + timedelta(seconds=1)

    return fallback_slots


def _resolve_market_data_window(
    from_utc: str | None,
    to_utc: str | None,
    *,
    interval: str = "1d",
    indicators: dict | None = None,
    objective: str = "positional",
) -> tuple[str, str]:
    from app.services.backtest.backtest_window import resolve_backtest_window

    resolved_from, resolved_to = resolve_backtest_window(
        from_utc,
        to_utc,
        interval=interval,
        indicators=indicators,
        objective=objective,
    )
    if from_utc or to_utc:
        logger.info(
            "🕒 Resolved backtest window | requested_from=%s requested_to=%s "
            "resolved_from=%s resolved_to=%s",
            from_utc,
            to_utc,
            resolved_from,
            resolved_to,
        )
    return resolved_from, resolved_to


def _normalize_reference_symbol_for_market_data(raw: str | None) -> str | None:
    """Reference symbols come from strategy YAML in user-friendly form (e.g.
    "^NSEI" for Nifty 50). The market-data API expects the bare ticker in the
    same shape it uses for stocks — so we strip the leading caret if present
    while preserving everything else (no .NS / .BO suffixes for indices)."""
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None
    if text.startswith("^"):
        text = text[1:]
    return text or None


# ── Phase 11: 1-minute execution helpers ─────────────────────────────────────
# When a backtest runs with 1-minute execution, the MAIN (and reference) series
# is fetched at this resolution; the quant engine resamples it up to the
# strategy timeframe for signals and walks the minute bars for fills/SL/TP.
INTRABAR_EXECUTION_INTERVAL = "1m"


def resolve_intrabar_execution(
    overrides: BacktestTriggerRequest | dict[str, Any] | None,
    *,
    signal_interval: str | None = None,
) -> bool:
    """Decide whether this backtest fetches 1-minute data and runs the engine's
    1-minute execution path.

    The request's `intrabar_execution` flag is tri-state:
      • explicit True/False  → honoured as-is (per-run override).
      • None (the default)    → AUTO: on for any strategy timeframe coarser than
        1m, off when the strategy already trades on 1m (1m IS its execution TF).

    `signal_interval` is the strategy (signal) timeframe; required to resolve the
    AUTO case. When omitted and the flag is None, defaults to off."""
    explicit: bool | None = None
    if isinstance(overrides, BacktestTriggerRequest):
        explicit = getattr(overrides, "intrabar_execution", None)
    elif isinstance(overrides, dict):
        explicit = overrides.get("intrabar_execution")

    interval_norm = str(signal_interval or "").strip().lower()

    # Safety: an arbitrary timeframe (e.g. 2m, 7m, 45m) cannot be fetched
    # natively from the data provider — only 1m can, and the engine resamples
    # it up. So even if the caller explicitly disabled intrabar execution, force
    # it on whenever the interval is neither 1m nor a provider-native interval;
    # otherwise the raw interval would be sent to the provider and rejected.
    natively_fetchable = (
        interval_norm == INTRABAR_EXECUTION_INTERVAL
        or interval_norm in GRPC_OHLCV_INTERVALS
    )
    if interval_norm and interval_norm != INTRABAR_EXECUTION_INTERVAL and not natively_fetchable:
        return True

    if explicit is not None:
        return bool(explicit)

    # AUTO: anything coarser than 1m executes on 1m for accuracy.
    if not signal_interval:
        return False
    return interval_norm != INTRABAR_EXECUTION_INTERVAL


def build_main_fetch_request(
    market_data_request: StrategyMarketDataRequest,
    *,
    intrabar_execution: bool,
) -> StrategyMarketDataRequest:
    """Return the request used to fetch the MAIN OHLCV series.

    With 1-minute execution on, the main series is fetched at 1m (the engine
    resamples it to the strategy timeframe). Otherwise it is fetched at the
    strategy timeframe exactly as before. The original request's interval is
    preserved everywhere else (it remains the reported "signal" timeframe)."""
    if intrabar_execution and market_data_request.interval != INTRABAR_EXECUTION_INTERVAL:
        return replace(market_data_request, interval=INTRABAR_EXECUTION_INTERVAL)
    return market_data_request


def extract_reference_market_data_request(
    main_request: StrategyMarketDataRequest,
    yaml_path: str,
    *,
    fetch_interval: str | None = None,
) -> StrategyMarketDataRequest | None:
    """Build a StrategyMarketDataRequest for the strategy's reference_symbol,
    sharing the main request's time window. Returns None when the strategy
    doesn't declare a reference.

    `fetch_interval` overrides the timeframe the reference is fetched at — used
    by 1-minute execution to fetch the reference at 1m so it resamples
    consistently with the main series inside the engine. Defaults to the main
    request's interval."""
    strategy = _load_strategy_yaml(yaml_path)
    raw_ref = strategy.get("reference_symbol") or strategy.get("benchmark_symbol")
    normalized = _normalize_reference_symbol_for_market_data(raw_ref)
    if normalized is None:
        return None
    return StrategyMarketDataRequest(
        yaml_path=yaml_path,
        raw_symbol=str(raw_ref),
        symbol=normalized,
        interval=fetch_interval or main_request.interval,
        from_utc=main_request.from_utc,
        to_utc=main_request.to_utc,
        user_specified_window=main_request.user_specified_window,
    )


def extract_htf_market_data_requests(
    main_request: StrategyMarketDataRequest,
    yaml_path: str,
) -> list[StrategyMarketDataRequest]:
    """Build one StrategyMarketDataRequest per HTF timeframe declared by the
    strategy. Reuses the trade symbol and the main window. Returns [] when no
    HTF rules are declared.

    Each declared HTF must be a supported user timeframe (5m, 15m, 1h, 1d…).
    Unsupported timeframes raise — fail fast rather than silently return
    incomplete data.
    """
    strategy = _load_strategy_yaml(yaml_path)
    raw_htf = strategy.get("htf") or strategy.get("higher_timeframes")
    if not raw_htf:
        return []
    if isinstance(raw_htf, dict):
        raw_htf = [raw_htf]
    if not isinstance(raw_htf, list):
        raise ValueError("strategy.htf must be a list of {timeframe, condition} entries.")

    requests: list[StrategyMarketDataRequest] = []
    seen_intervals: set[str] = set()
    for entry in raw_htf:
        if not isinstance(entry, dict):
            continue
        raw_tf = str(entry.get("timeframe") or "").strip()
        if not raw_tf:
            raise ValueError("strategy.htf entry missing 'timeframe'.")
        # Normalise via the same path the main interval uses; raises on unsupported.
        interval = _normalize_interval(raw_tf)
        if interval == main_request.interval:
            # An HTF that matches the main timeframe is meaningless — skip
            # silently rather than fetch the same data twice.
            continue
        if interval in seen_intervals:
            continue
        seen_intervals.add(interval)
        requests.append(StrategyMarketDataRequest(
            yaml_path=yaml_path,
            raw_symbol=main_request.raw_symbol,
            symbol=main_request.symbol,
            interval=interval,
            from_utc=main_request.from_utc,
            to_utc=main_request.to_utc,
            user_specified_window=main_request.user_specified_window,
        ))
    return requests


async def fetch_auxiliary_ohlcv(
    main_request: StrategyMarketDataRequest,
    *,
    main_fetch_interval: str | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, list[dict[str, Any]]] | None]:
    """Phase 7 entry point — fetch reference + HTF OHLCV alongside the main
    series. Returns (reference_ohlcv, htf_ohlcv) ready to forward to the
    quant engine. Both are None when the strategy declares neither.

    Fetches happen concurrently (asyncio.gather) so the extra latency is the
    SLOWEST single fetch, not the sum.

    `main_fetch_interval` (Phase 11) overrides the timeframe the REFERENCE
    series is fetched at — set to "1m" for 1-minute execution so the reference
    resamples consistently with the main series. HTF series are unaffected:
    they keep their own declared higher timeframes, and the "skip HTF that
    equals main" check still compares against the strategy (signal) timeframe.
    """
    yaml_path = main_request.yaml_path
    ref_request = extract_reference_market_data_request(
        main_request, yaml_path, fetch_interval=main_fetch_interval,
    )
    htf_requests = extract_htf_market_data_requests(main_request, yaml_path)

    if ref_request is None and not htf_requests:
        return None, None

    aux_requests: list[StrategyMarketDataRequest] = []
    if ref_request is not None:
        aux_requests.append(ref_request)
    aux_requests.extend(htf_requests)

    logger.info(
        "📡 Fetching auxiliary market data | reference=%s htf_intervals=%s",
        ref_request.symbol if ref_request else None,
        [r.interval for r in htf_requests],
    )

    results = await asyncio.gather(
        *(fetch_ohlcv_records(req) for req in aux_requests),
        return_exceptions=False,
    )

    reference_ohlcv: list[dict[str, Any]] | None = None
    htf_ohlcv: dict[str, list[dict[str, Any]]] | None = None
    cursor = 0
    if ref_request is not None:
        reference_ohlcv = results[cursor]
        cursor += 1
    if htf_requests:
        htf_ohlcv = {req.interval: results[cursor + idx] for idx, req in enumerate(htf_requests)}

    return reference_ohlcv, htf_ohlcv


def extract_strategy_market_data_request(
    yaml_path: str,
    overrides: BacktestTriggerRequest | dict[str, Any] | None = None,
) -> StrategyMarketDataRequest:
    strategy = _load_strategy_yaml(yaml_path)

    if isinstance(overrides, BacktestTriggerRequest):
        override_payload = overrides.model_dump(exclude_none=True)
    else:
        override_payload = dict(overrides or {})

    raw_symbol  = str(override_payload.get("symbol")   or strategy.get("symbol")    or "").strip()
    raw_interval = str(override_payload.get("interval") or strategy.get("timeframe") or "").strip()
    symbol = _normalize_symbol_for_market_data(raw_symbol)

    if not raw_symbol:
        raise ValueError(f"Strategy YAML at {yaml_path} is missing strategy.symbol")
    if not raw_interval:
        raise ValueError(f"Strategy YAML at {yaml_path} is missing strategy.timeframe")

    interval = _normalize_interval(raw_interval)
    profile = strategy.get("profile") if isinstance(strategy.get("profile"), dict) else {}
    raw_objective = (
        override_payload.get("objective")
        or profile.get("objective")
        or strategy.get("objective")
        or "positional"
    )
    objective = str(raw_objective or "positional").strip().lower()
    indicators = strategy.get("indicators") if isinstance(strategy.get("indicators"), dict) else {}
    user_specified_window = bool(
        str(override_payload.get("from_utc") or "").strip()
        or str(override_payload.get("to_utc") or "").strip()
    )

    from_utc, to_utc = _resolve_market_data_window(
        override_payload.get("from_utc"),
        override_payload.get("to_utc"),
        interval=interval,
        indicators=indicators,
        objective=objective,
    )

    logger.info(
        "🧭 Market data request ready | symbol=%s interval=%s sim_from=%s sim_to=%s "
        "user_range=%s",
        symbol,
        interval,
        from_utc,
        to_utc,
        user_specified_window,
    )

    return StrategyMarketDataRequest(
        yaml_path=yaml_path,
        raw_symbol=raw_symbol,
        symbol=symbol,
        interval=interval,
        from_utc=from_utc,
        to_utc=to_utc,
        user_specified_window=user_specified_window,
    )


def _extract_record_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "ohlcv", "candles", "items", "result", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError("Market data response did not contain a recognizable OHLCV list.")


def _pick_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    lowered = {str(k).lower(): value for k, value in record.items()}
    for key in keys:
        if key.lower() in lowered and lowered[key.lower()] is not None:
            return lowered[key.lower()]
    return None


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        raise ValueError("OHLCV record is missing timestamp")

    if isinstance(value, (int, float)):
        if value > 10_000_000_000:
            timestamp = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        else:
            timestamp = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return _to_utc_iso(timestamp)

    text = str(value).strip()
    if not text:
        raise ValueError("OHLCV record has an empty timestamp")

    normalized = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return _to_utc_iso(dt)


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OHLCV record has invalid {field_name} value: {value!r}") from exc


def _normalize_record(record: Any) -> dict[str, Any]:
    if isinstance(record, (list, tuple)) and len(record) >= 6:
        timestamp, open_, high, low, close, volume = record[:6]
        source = {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    elif isinstance(record, dict):
        source = record
    else:
        raise ValueError("OHLCV record must be a dict or a 6-item list.")

    normalized = {
        "timestamp": _normalize_timestamp(
            _pick_value(source, "timestamp", "datetime", "time", "date", "ts")
        ),
        "open": _coerce_float(_pick_value(source, "open", "o"), "open"),
        "high": _coerce_float(_pick_value(source, "high", "h"), "high"),
        "low": _coerce_float(_pick_value(source, "low", "l"), "low"),
        "close": _coerce_float(_pick_value(source, "close", "c"), "close"),
        "volume": _coerce_float(_pick_value(source, "volume", "vol", "v"), "volume"),
    }
    _validate_price_row(normalized)
    return normalized


def _validate_price_row(record: dict[str, Any]) -> None:
    open_ = float(record["open"])
    high = float(record["high"])
    low = float(record["low"])
    close = float(record["close"])
    volume = float(record["volume"])

    if min(open_, high, low, close) < 0:
        raise ValueError("OHLCV record contains negative price values.")
    if high < max(open_, close, low):
        raise ValueError("OHLCV record has high lower than other price fields.")
    if low > min(open_, close, high):
        raise ValueError("OHLCV record has low greater than other price fields.")
    if volume < 0:
        raise ValueError("OHLCV record contains negative volume.")


def normalize_ohlcv_payload(payload: Any) -> list[dict[str, Any]]:
    records = [_normalize_record(record) for record in _extract_record_list(payload)]
    records.sort(key=lambda row: row["timestamp"])
    if not records:
        raise ValueError("Market data response did not return any OHLCV records.")
    return records


def _dedup_and_sort_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by timestamp (keep first occurrence) and sort ascending.

    Designed to run inside ``asyncio.to_thread`` after a chunked fetch.
    """
    seen: set = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("timestamp")
        if ts in seen:
            continue
        seen.add(ts)
        out.append(row)
    out.sort(key=lambda r: r["timestamp"])
    return out


class _gc_disabled:
    """Context manager that disables cyclic GC for bulk allocation phases.

    Refcounting still frees memory; we just suspend the gen-2 sweeps that
    grow O(N) with len(all_rows). Restores the prior GC state on exit.
    """
    def __enter__(self):
        import gc
        self._gc = gc
        self._was_enabled = gc.isenabled()
        if self._was_enabled:
            gc.disable()
        return self

    def __exit__(self, *exc_info):
        if self._was_enabled:
            self._gc.enable()
        return False


def _build_date_chunks(from_utc: str, to_utc: str, chunk_days: int) -> list[tuple[str, str]]:
    """Split [from_utc, to_utc] into sequential chunks of at most chunk_days each."""
    start = _parse_utc_timestamp(from_utc, "from_utc")
    end   = _parse_utc_timestamp(to_utc,   "to_utc")
    if start is None or end is None or start >= end:
        return [(from_utc, to_utc)]

    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunks.append((
            cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))
        cursor = chunk_end + timedelta(seconds=1)
    return chunks


async def _fetch_single_chunk(
    client: httpx.AsyncClient,
    endpoint: str,
    symbol: str,
    interval: str,
    from_utc: str,
    to_utc: str,
    chunk_idx: int,
    total_chunks: int,
) -> list[dict[str, Any]]:
    """Fetch one chunk and return normalized records. Raises RuntimeError on HTTP error."""
    params = {"symbol": symbol, "from": from_utc, "to": to_utc, "interval": interval}
    
    chunk_start = time.perf_counter()
    logger.info(
        "📡 Fetching chunk %d/%d | symbol=%s interval=%s from=%s to=%s",
        chunk_idx, total_chunks, symbol, interval, from_utc, to_utc,
    )
    
    try:
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        
        chunk_duration = time.perf_counter() - chunk_start
        logger.info(
            "⏱️  TIMING|step=fetch_chunk_%d|status=COMPLETE|duration=%.4fs|symbol=%s|interval=%s",
            chunk_idx, chunk_duration, symbol, interval,
        )
        
    except httpx.HTTPStatusError as exc:
        chunk_duration = time.perf_counter() - chunk_start
        logger.exception(
            "⏱️  TIMING|step=fetch_chunk_%d|status=FAILED|duration=%.4fs|"
            "Market data service returned HTTP %s (chunk %d/%d)",
            chunk_idx, chunk_duration,
            exc.response.status_code, chunk_idx, total_chunks,
        )
        raise RuntimeError(
            f"Market data service rejected the request with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        chunk_duration = time.perf_counter() - chunk_start
        logger.exception(
            "⏱️  TIMING|step=fetch_chunk_%d|status=FAILED|duration=%.4fs|"
            "Market data request failed (chunk %d/%d)",
            chunk_idx, chunk_duration, chunk_idx, total_chunks
        )
        raise RuntimeError("Failed to fetch market data from the upstream service.") from exc

    # Both ``response.json()`` and ``normalize_ohlcv_payload`` are sync CPU
    # work; for 43k-row chunks they were blocking the event loop for seconds
    # and starving the gRPC/HTTP receive side. Offload to a worker thread.
    def _parse_and_normalize() -> list[dict[str, Any]]:
        try:
            return normalize_ohlcv_payload(response.json())
        except ValueError:
            return []

    return await asyncio.to_thread(_parse_and_normalize)


async def fetch_ohlcv_records(
    request: StrategyMarketDataRequest | str,
    overrides: BacktestTriggerRequest | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch OHLCV records for the given request, splitting the date range into
    chunks of settings.backtest_fetch_chunk_days to avoid HTTP 429 on large ranges."""
    import asyncio

    fetch_start = time.perf_counter()
    
    resolved = (
        extract_strategy_market_data_request(request, overrides=overrides)
        if isinstance(request, str)
        else request
    )

    from app.services.backtest.backtest_window import extend_fetch_start

    indicators: dict = {}
    objective = "positional"
    strategy_payload: dict[str, Any] = {}
    if resolved.yaml_path:
        try:
            strategy_payload = _load_strategy_yaml(resolved.yaml_path)
            indicators = (
                strategy_payload.get("indicators")
                if isinstance(strategy_payload.get("indicators"), dict)
                else {}
            )
            profile = (
                strategy_payload.get("profile")
                if isinstance(strategy_payload.get("profile"), dict)
                else {}
            )
            objective = str(
                profile.get("objective") or strategy_payload.get("objective") or "positional"
            ).strip().lower()
        except Exception:
            pass
    fetch_from_utc = extend_fetch_start(
        resolved.from_utc,
        user_specified_window=resolved.user_specified_window,
        interval=resolved.interval,
        indicators=indicators,
        strategy=strategy_payload or None,
        objective=objective,
    )
    fetch_request = replace(resolved, from_utc=fetch_from_utc)

    if fetch_from_utc != resolved.from_utc:
        logger.info(
            "📐 Warm-up padding applied | symbol=%s interval=%s "
            "sim_from=%s → fetch_from=%s (delta: cache key uses fetch_from)",
            resolved.symbol, resolved.interval, resolved.from_utc, fetch_from_utc,
        )

    # ── Fix 3: cache hit short-circuits the entire HTTP fetch ─────────────────
    with time_step("check_ohlcv_cache", extra_context={
        "symbol": fetch_request.symbol,
        "interval": fetch_request.interval,
    }):
        cached = get_cached_records(
            fetch_request.symbol,
            fetch_request.interval,
            fetch_request.from_utc,
            fetch_request.to_utc,
        )
    
    if cached is not None:
        fetch_duration = time.perf_counter() - fetch_start
        logger.info(
            "⏱️  TIMING|step=fetch_ohlcv_records|status=CACHE_HIT|duration=%.4fs|"
            "symbol=%s|interval=%s|rows=%d",
            fetch_duration, fetch_request.symbol, fetch_request.interval, len(cached),
        )
        return cached

    chunk_days = max(7, _chunk_days_for(fetch_request.raw_symbol, settings))
    chunks = _build_date_chunks(fetch_request.from_utc, fetch_request.to_utc, chunk_days)
    use_grpc = use_grpc_transport(settings)

    if use_grpc:
        logger.info(
            "📡 Fetching market data via gRPC | symbol=%s interval=%s "
            "fetch_from=%s fetch_to=%s sim_from=%s sim_to=%s "
            "total_chunks=%d chunk_days=%d target=%s",
            resolved.symbol,
            resolved.interval,
            fetch_request.from_utc,
            fetch_request.to_utc,
            resolved.from_utc,
            resolved.to_utc,
            len(chunks),
            chunk_days,
            settings.market_data_grpc_target,
        )
    else:
        endpoint = _resolve_market_data_endpoint(settings.historical_data_url)
        logger.info(
            "📡 Fetching market data via HTTP | symbol=%s interval=%s "
            "fetch_from=%s fetch_to=%s sim_from=%s sim_to=%s "
            "total_chunks=%d chunk_days=%d endpoint=%s",
            resolved.symbol,
            resolved.interval,
            fetch_request.from_utc,
            fetch_request.to_utc,
            resolved.from_utc,
            resolved.to_utc,
            len(chunks),
            chunk_days,
            endpoint,
        )

    all_rows: list[dict[str, Any]] = []

    # GC-disabled scope covers the entire bulk-allocation phase (every chunk
    # plus the final dedup/sort). With it enabled, gen-2 sweeps were scaling
    # with total accumulated rows and stalling the event loop seconds at a
    # time by chunk ~20.
    with _gc_disabled():
        if use_grpc:
            from app.proto.gen.marketdata_pb2_grpc import MarketDataServiceStub

            timeout = settings.market_data_grpc_timeout_seconds
            async with open_grpc_channel(settings) as channel:
                stub = MarketDataServiceStub(channel)
                for idx, (chunk_from, chunk_to) in enumerate(chunks, start=1):
                    raw_rows = await fetch_ohlcv_chunk_grpc(
                        stub,
                        symbol=resolved.symbol,
                        interval=resolved.interval,
                        from_utc=chunk_from,
                        to_utc=chunk_to,
                        chunk_idx=idx,
                        total_chunks=len(chunks),
                        timeout_seconds=timeout,
                    )
                    if raw_rows:
                        try:
                            chunk_rows = await asyncio.to_thread(
                                normalize_ohlcv_payload, raw_rows
                            )
                        except ValueError:
                            chunk_rows = []
                    else:
                        chunk_rows = []
                    all_rows.extend(chunk_rows)
                    if idx < len(chunks):
                        await asyncio.sleep(0.3)
        else:
            endpoint = _resolve_market_data_endpoint(settings.historical_data_url)
            async with httpx.AsyncClient(timeout=settings.historical_data_timeout_seconds) as client:
                for idx, (chunk_from, chunk_to) in enumerate(chunks, start=1):
                    chunk_rows = await _fetch_single_chunk(
                        client,
                        endpoint,
                        resolved.symbol,
                        resolved.interval,
                        chunk_from,
                        chunk_to,
                        idx,
                        len(chunks),
                    )
                    all_rows.extend(chunk_rows)

                    if idx < len(chunks):
                        await asyncio.sleep(0.3)  # brief pause between chunks

        # Final dedup + sort in a worker thread — keeps the event loop
        # responsive even when total row count is in the millions.
        all_rows = await asyncio.to_thread(_dedup_and_sort_records, all_rows)

    if not all_rows:
        raise ValueError("Market data response did not return any OHLCV records.")

    fetch_duration = time.perf_counter() - fetch_start
    logger.info(
        "⏱️  TIMING|step=fetch_ohlcv_records|status=COMPLETE|duration=%.4fs|"
        "symbol=%s|interval=%s|total_rows=%d|chunks=%d",
        fetch_duration, resolved.symbol, resolved.interval, len(all_rows), len(chunks),
    )

    # Persist for next time using fetch_request keys (same key as the lookup above).
    # IMPORTANT: must use fetch_request (warm-up extended dates), NOT resolved (sim dates).
    # Using resolved here was the original bug — lookup key ≠ save key → permanent miss.
    with time_step("save_ohlcv_cache", extra_context={
        "symbol": fetch_request.symbol,
        "interval": fetch_request.interval,
        "rows": len(all_rows),
    }):
        save_ohlcv_cache(
            fetch_request.symbol, fetch_request.interval, fetch_request.from_utc, fetch_request.to_utc, all_rows,
        )

    return all_rows
