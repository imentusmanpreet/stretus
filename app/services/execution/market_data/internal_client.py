"""
app.services.execution.market_data.internal_client
────────────────────────────────────────────────────
Single market-data client that routes ALL requests through our own
stretus-backend service (user-gateway /api/v1/marketdata/*).

This replaces the direct Upstox and Binance clients for execution market data.
Adding a new broker now only requires changes in stretus-backend — this client
and everything above it (MarketDataService, StrategyEvaluator) are unaffected.

Endpoints consumed
──────────────────
  OHLCV    GET {BASE}/api/v1/marketdata/ohlcv
             ?symbol=&from=&to=&timeframe=
  LTP      GET {BASE}/api/v1/marketdata/trade/latest
             ?symbol=
  Circuit  — not yet exposed; returns None (evaluator falls back to
             percentage-based defaults via lookup_instrument_defaults).

DataFrame contract (same as UpstoxClient / BinanceClient)
──────────────────────────────────────────────────────────
  - Index:   UTC-aware DatetimeIndex, oldest → newest, no duplicates.
  - Columns: open, high, low, close, volume  (all lowercase).
  - Length:  at most ``lookback`` rows (most recent ``lookback`` bars).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd

from app.core.config import get_settings
from app.services.execution.market_data.base import MarketDataClient

logger = logging.getLogger(__name__)

_OHLCV_PATH  = "/api/v1/marketdata/ohlcv"
_TRADE_PATH  = "/api/v1/marketdata/trade/latest"


def _resolve_base_url(base_url: str) -> str:
    """Normalize HISTORICAL_DATA_URL to a gateway base URL (no OHLCV path suffix)."""
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith(_OHLCV_PATH):
        return normalized[: -len(_OHLCV_PATH)]
    return normalized

# Calendar-day safety buffers when computing the fetch window from a bare
# ``lookback`` (number of bars). These mirror the logic in upstox_client so
# the same bar counts are honoured regardless of which client is used.
_NSE_SESSION_MINUTES = 375.0   # 09:15–15:30 IST
_CRYPTO_MINUTES_PER_DAY = 1440.0  # 24/7


def _timeframe_to_minutes(timeframe: str) -> int:
    """Convert a timeframe string (1m, 15m, 1h, 4h, 1d) to minutes."""
    tf = str(timeframe or "").strip().lower()
    if tf.endswith("m"):
        try:
            return int(tf[:-1])
        except ValueError:
            return 1
    if tf.endswith("h"):
        try:
            return int(tf[:-1]) * 60
        except ValueError:
            return 60
    if tf.endswith("d"):
        return 1440
    return 1


def _date_window(lookback: int, timeframe: str, *, is_crypto: bool = False) -> tuple[str, str]:
    """Return (from_rfc3339, to_rfc3339) covering at least ``lookback`` bars."""
    tf_mins = _timeframe_to_minutes(timeframe)
    session_mins = _CRYPTO_MINUTES_PER_DAY if is_crypto else _NSE_SESSION_MINUTES
    if tf_mins >= 1440:
        # Daily or coarser: each bar spans one calendar day.
        calendar_days = lookback * 2 + 14   # 2× safety + holiday buffer
    else:
        bars_per_day = session_mins / tf_mins
        trading_days = max(2, int(lookback / bars_per_day) + 1)
        calendar_days = trading_days * 2 + 14

    to_dt   = datetime.now(tz=timezone.utc)
    from_dt = to_dt - timedelta(days=calendar_days)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return from_dt.strftime(fmt), to_dt.strftime(fmt)


def _extract_bars(payload: object) -> list[dict]:
    """Extract OHLCV bar list from gateway JSON (dict or bare list)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "ohlcv", "candles", "items", "result", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _response_to_df(bars: list[dict]) -> pd.DataFrame:
    """Convert the gateway's OHLCV bar list to the standard DataFrame contract."""
    if not bars:
        raise ValueError("Internal market data service returned an empty OHLCV list.")

    rows = []
    for b in bars:
        try:
            rows.append({
                "timestamp": b["ts"],
                "open":   float(b["open"]),
                "high":   float(b["high"]),
                "low":    float(b["low"]),
                "close":  float(b["close"]),
                "volume": float(b["volume"]),
            })
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed bars

    if not rows:
        raise ValueError("No usable OHLCV bars after parsing internal service response.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


class InternalMarketDataClient(MarketDataClient):
    """
    Unified market-data client backed by stretus-backend (user-gateway).

    Broker / exchange selection is the gateway's concern. When a new broker is
    added, only stretus-backend changes — this client and all callers are
    unaffected.

    ``adapter_symbol`` is accepted for interface compatibility but is not sent
    to the gateway (the gateway resolves broker-native symbols via ref_data
    internally).
    """

    adapter_id = "internal"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        is_crypto: bool = False,
    ) -> None:
        settings = get_settings()
        self._base = _resolve_base_url(
            base_url or settings.historical_data_url or ""
        )
        self._timeout = timeout if timeout is not None else settings.market_data_timeout_seconds
        self._is_crypto = is_crypto
        if not self._base:
            raise RuntimeError(
                "HISTORICAL_DATA_URL is not configured. "
                "Set it to the stretus-backend user-gateway base URL."
            )
        logger.debug(
            "internal_client|init|base_url=%s|is_crypto=%s",
            self._base, self._is_crypto,
        )

    # ── Candles ───────────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        adapter_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        from_utc, to_utc = _date_window(lookback, timeframe, is_crypto=self._is_crypto)
        params = {
            "symbol":    symbol,
            "from":      from_utc,
            "to":        to_utc,
            "timeframe": timeframe,
        }

        logger.info(
            "internal_client|fetch_candles|symbol=%s|timeframe=%s|lookback=%d"
            "|from=%s|to=%s",
            symbol, timeframe, lookback, from_utc, to_utc,
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._base + _OHLCV_PATH, params=params)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Internal market data service returned HTTP {resp.status_code} "
                f"for symbol={symbol} timeframe={timeframe}: {resp.text[:400]}"
            )

        payload = resp.json()
        bars = _extract_bars(payload)

        df = _response_to_df(bars)

        logger.info(
            "internal_client|fetch_candles|ok|symbol=%s|timeframe=%s|rows=%d|last=%s",
            symbol, timeframe, len(df), df.index[-1] if not df.empty else "n/a",
        )

        if df.empty:
            raise ValueError(
                f"Internal market data service returned no usable candles "
                f"for {symbol} ({timeframe})."
            )

        return df.tail(lookback)

    # ── LTP ───────────────────────────────────────────────────────────────────

    async def fetch_ltp(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> float:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._base + _TRADE_PATH,
                params={"symbol": symbol},
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Internal market data service returned HTTP {resp.status_code} "
                f"for LTP of symbol={symbol}: {resp.text[:400]}"
            )

        payload = resp.json()
        data = payload.get("data") or payload
        ltp = data.get("ltp") if isinstance(data, dict) else None

        if ltp is None:
            raise ValueError(
                f"No 'ltp' in internal service response for symbol={symbol}: {payload}"
            )

        price = float(ltp)
        if price <= 0:
            raise ValueError(
                f"Internal market data service returned non-positive LTP for {symbol}: {price}"
            )

        logger.info("internal_client|fetch_ltp|symbol=%s|ltp=%.10g", symbol, price)
        return price

    # ── Circuit limits ────────────────────────────────────────────────────────

    async def fetch_circuit_limits(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> Optional[dict]:
        # Circuit limits are not yet exposed by the internal service.
        # The evaluator falls back to percentage-based defaults from
        # ref_data.system_configs via lookup_instrument_defaults().
        return None
