"""
Market-data request extraction, validation, and OHLCV normalization helpers.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import logging

import httpx
import yaml

from app.core.config import get_settings
from app.schemas.backtest import BacktestTriggerRequest
from app.services.backtest.ohlcv_cache import (
    get_cached_records,
    save_records as save_ohlcv_cache,
)
from app.services.strategy.builder import (
    SUPPORTED_USER_TIMEFRAMES,
    UNSUPPORTED_USER_TIMEFRAME_MESSAGE,
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


@dataclass(frozen=True)
class OhlcvFetchSlot:
    index: int
    from_utc: str
    to_utc: str
    label: str = ""

# ── Lookback computation ───────────────────────────────────────────────────────

# Bars per calendar day for NSE/BSE.
# Indian market session: 9:15 AM – 3:30 PM IST = 375 minutes (NOT 390 which is the US market).
_BARS_PER_CALENDAR_DAY: dict[str, float] = {
    "1m":  375.0,   # 375 one-minute bars per NSE session
    "5m":  75.0,    # 375 / 5
    "10m": 37.0,    # 375 / 10 = 37.5 → floor
    "15m": 25.0,    # 375 / 15
    "30m": 12.0,    # 375 / 30 = 12.5 → floor
    "1h":  6.25,    # 375 / 60 = 6.25
    "1d":  1.0,
}

# Extra calendar-day buffer beyond indicator warm-up:
# 1d bars have fewer bars per day so we add a large buffer
_INTERVAL_BUFFER_DAYS: dict[str, int] = {
    "1m":  7,
    "5m":  7,
    "10m": 7,
    "15m": 14,
    "30m": 14,
    "1h":  21,
    "1d":  90,  # monthly-style buffer for daily strategies
}

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


def _compute_required_lookback_days(
    interval: str,
    indicators: dict,
    objective: str = "positional",
) -> int:
    """
    Compute the minimum number of calendar days of data needed for indicators
    to have warmed up AND still have enough tradeable bars.

    Formula:
      max_period × safety_factor  →  total_candles_needed
      total_candles_needed / bars_per_calendar_day  →  calendar_days_needed
      + interval_buffer_days  →  final

    For intraday objectives with sub-daily bars, the configured backtest window
    is usually ample. For daily bars with SMA(50), we need ~200+ days.
    """
    max_period = _extract_max_indicator_period(indicators)
    bars_per_day = _BARS_PER_CALENDAR_DAY.get(interval, 26.0)
    buffer_days  = _INTERVAL_BUFFER_DAYS.get(interval, 30)

    if max_period <= 0:
        return max(settings.backtest_default_lookback_days, 90)

    total_candles_needed  = max_period * _WARMUP_SAFETY_FACTOR
    required_calendar_days = int(total_candles_needed / bars_per_day) + buffer_days

    floor = max(settings.backtest_default_lookback_days, 90)
    result = max(required_calendar_days, floor)

    logger.info(
        "📏 Lookback computed | interval=%s max_indicator_period=%s bars_per_day=%.1f "
        "required_days=%s (floor=%s)",
        interval, max_period, bars_per_day, result, floor,
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
    symbol = str(raw_symbol or "").strip().upper()
    if not symbol:
        return ""

    if ":" in symbol:
        _, symbol = symbol.split(":", 1)

    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        symbol = symbol[:-3]

    return symbol


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
    start_dt = _parse_utc_timestamp(from_utc, "from_utc")
    end_dt = _parse_utc_timestamp(to_utc, "to_utc")
    if start_dt is not None and end_dt is not None and start_dt >= end_dt:
        raise ValueError("from_utc must be earlier than to_utc.")

    default_from, default_to = _default_backtest_window(
        interval=interval,
        indicators=indicators,
        objective=objective,
    )
    if from_utc or to_utc:
        logger.info(
            "🕒 Using configured backtest window "
            "| requested_from=%s requested_to=%s enforced_from=%s enforced_to=%s",
            from_utc,
            to_utc,
            default_from,
            default_to,
        )
    return default_from, default_to


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


def extract_reference_market_data_request(
    main_request: StrategyMarketDataRequest,
    yaml_path: str,
) -> StrategyMarketDataRequest | None:
    """Build a StrategyMarketDataRequest for the strategy's reference_symbol,
    sharing the main request's interval and time window. Returns None when the
    strategy doesn't declare a reference."""
    strategy = _load_strategy_yaml(yaml_path)
    raw_ref = strategy.get("reference_symbol") or strategy.get("benchmark_symbol")
    normalized = _normalize_reference_symbol_for_market_data(raw_ref)
    if normalized is None:
        return None
    return StrategyMarketDataRequest(
        yaml_path=yaml_path,
        raw_symbol=str(raw_ref),
        symbol=normalized,
        interval=main_request.interval,         # reference uses the same TF as main
        from_utc=main_request.from_utc,
        to_utc=main_request.to_utc,
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
        ))
    return requests


async def fetch_auxiliary_ohlcv(
    main_request: StrategyMarketDataRequest,
) -> tuple[list[dict[str, Any]] | None, dict[str, list[dict[str, Any]]] | None]:
    """Phase 7 entry point — fetch reference + HTF OHLCV alongside the main
    series. Returns (reference_ohlcv, htf_ohlcv) ready to forward to the
    quant engine. Both are None when the strategy declares neither.

    Fetches happen concurrently (asyncio.gather) so the extra latency is the
    SLOWEST single fetch, not the sum.
    """
    yaml_path = main_request.yaml_path
    ref_request = extract_reference_market_data_request(main_request, yaml_path)
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

    from_utc, to_utc = _resolve_market_data_window(
        override_payload.get("from_utc"),
        override_payload.get("to_utc"),
        interval=interval,
        indicators=indicators,
        objective=objective,
    )

    logger.info(
        "🧭 Market data request ready | symbol=%s interval=%s from=%s to=%s",
        symbol,
        interval,
        from_utc,
        to_utc,
    )

    return StrategyMarketDataRequest(
        yaml_path=yaml_path,
        raw_symbol=raw_symbol,
        symbol=symbol,
        interval=interval,
        from_utc=from_utc,
        to_utc=to_utc,
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
    logger.info(
        "📡 Fetching chunk %d/%d | symbol=%s interval=%s from=%s to=%s",
        chunk_idx, total_chunks, symbol, interval, from_utc, to_utc,
    )
    try:
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Market data service returned HTTP %s (chunk %d/%d)",
            exc.response.status_code, chunk_idx, total_chunks,
        )
        raise RuntimeError(
            f"Market data service rejected the request with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("Market data request failed (chunk %d/%d)", chunk_idx, total_chunks)
        raise RuntimeError("Failed to fetch market data from the upstream service.") from exc

    try:
        return normalize_ohlcv_payload(response.json())
    except ValueError:
        # Empty chunk (no trading days in that window) — skip silently
        return []


async def fetch_ohlcv_records(
    request: StrategyMarketDataRequest | str,
    overrides: BacktestTriggerRequest | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch OHLCV records for the given request, splitting the date range into
    chunks of settings.backtest_fetch_chunk_days to avoid HTTP 429 on large ranges."""
    import asyncio

    resolved = (
        extract_strategy_market_data_request(request, overrides=overrides)
        if isinstance(request, str)
        else request
    )

    # ── Fix 3: cache hit short-circuits the entire HTTP fetch ─────────────────
    cached = get_cached_records(
        resolved.symbol, resolved.interval, resolved.from_utc, resolved.to_utc,
    )
    if cached is not None:
        return cached

    endpoint    = _resolve_market_data_endpoint(settings.historical_data_url)
    chunk_days  = max(7, settings.backtest_fetch_chunk_days)
    chunks      = _build_date_chunks(resolved.from_utc, resolved.to_utc, chunk_days)

    logger.info(
        "📡 Fetching market data | symbol=%s interval=%s from=%s to=%s "
        "total_chunks=%d chunk_days=%d endpoint=%s",
        resolved.symbol, resolved.interval,
        resolved.from_utc, resolved.to_utc,
        len(chunks), chunk_days, endpoint,
    )

    all_rows: list[dict[str, Any]] = []
    seen_timestamps: set = set()

    async with httpx.AsyncClient(timeout=settings.historical_data_timeout_seconds) as client:
        for idx, (chunk_from, chunk_to) in enumerate(chunks, start=1):
            chunk_rows = await _fetch_single_chunk(
                client, endpoint,
                resolved.symbol, resolved.interval,
                chunk_from, chunk_to,
                idx, len(chunks),
            )
            for row in chunk_rows:
                ts = row.get("timestamp")
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    all_rows.append(row)

            if idx < len(chunks):
                await asyncio.sleep(0.3)  # brief pause between chunks

    all_rows.sort(key=lambda r: r["timestamp"])

    if not all_rows:
        raise ValueError("Market data response did not return any OHLCV records.")

    logger.info(
        "✅ Market data received | symbol=%s interval=%s total_rows=%d chunks=%d",
        resolved.symbol, resolved.interval, len(all_rows), len(chunks),
    )

    # Persist for next time. Best-effort; won't fail the request on cache errors.
    save_ohlcv_cache(
        resolved.symbol, resolved.interval, resolved.from_utc, resolved.to_utc, all_rows,
    )

    return all_rows
