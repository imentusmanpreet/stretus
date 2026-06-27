"""
app/services/data/provider.py — the lazy, streaming, cached OHLCV access layer.

Why this exists
---------------
The static single-symbol path fetches all OHLCV for one symbol and hands a dict to the
engine. That does not scale to a 5,000-symbol universe (§4): we must read **only what we
need, when we need it**, and cache aggressively. The :class:`DataProvider` Protocol is the
seam that makes the store swappable — an injected fetcher (Phase A), Parquet-on-disk
(Phase B), ClickHouse (later) — without any caller change (§4.2).

Two read shapes mirror the two-tier data model (§3.2):
  * **screening read** (Tier A) — daily (or scan-timeframe) bars for a candidate symbol
    up to ``asof``; cheap, cached, batched; used only at the refresh cadence;
  * **execution read** (Tier B) — strategy-timeframe (or 1-min) bars for a single member
    over a bounded window; fetched only when a symbol becomes a member, evicted when it
    leaves and is flat.

Invariants honored: no look-ahead (reads are bounded by ``to``); bounded memory (callers
pull per-symbol slices, never the full matrix); transparent caching keyed precisely on
``(symbol, timeframe, from, to)``. KB-free.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

# An injectable async OHLCV fetcher: (symbol, interval, from_iso, to_iso) -> records.
# Mirrors the signature used by app.services.discovery.scanner so the same backtest
# market-data fetcher (concurrent, chunked, cached) can back the provider.
FetchOhlcv = Callable[[str, str, str, str], Awaitable[list[dict[str, Any]]]]


def records_to_frame(records: list[dict[str, Any]] | None) -> pd.DataFrame | None:
    """Normalize OHLCV records → a UTC-naive, timestamp-indexed, float DataFrame.

    Returns ``None`` when records are missing/empty or lack a usable timestamp/columns.
    Identical normalization to the scanner's ``_df_from_records`` so screen conditions
    evaluate on the exact same shape the engine indicators expect.
    """
    if not records:
        return None
    df = pd.DataFrame.from_records(records)
    if "timestamp" not in df.columns:
        return None
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").tz_localize(None)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if not keep:
        return None
    return df[keep].astype(float)


@runtime_checkable
class DataProvider(Protocol):
    """Lazy, streaming OHLCV access. The store behind it is swappable (§4.1/§4.2).

    Implementations MUST:
      * never return bars with ``timestamp > to_iso`` (no look-ahead — Invariant 4);
      * tolerate per-symbol failure by returning ``None`` rather than raising, so one
        missing feed does not sink a whole resolution (Invariant: scanner tolerance);
      * be safe to call concurrently.
    """

    async def fetch_screening(
        self, symbol: str, *, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        """Tier-A read: daily/scan-timeframe bars for one candidate up to ``to_iso``."""
        ...

    async def fetch_execution(
        self, symbol: str, *, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        """Tier-B read: strategy-timeframe bars for one member over a bounded window."""
        ...


class CachingFetchProvider:
    """A :class:`DataProvider` backed by an injected async ``fetch_ohlcv`` callable.

    Phase-A implementation: wraps the existing backtest market-data fetcher (concurrent,
    chunked, cached at the HTTP layer) and adds an in-process frame cache keyed precisely
    on ``(symbol, timeframe, from, to)`` so repeated reads within a resolution/backtest are
    free (§12.3). The same callable is injectable in tests (no network), exactly as
    ``scanner.scan_universe`` does. Phase B swaps this for a Parquet-backed provider with
    no caller change.

    ``evict`` drops a symbol's cached frames once it is flat and out of the universe, so
    peak memory scales with active members, not pool size (Invariant 6).
    """

    def __init__(self, fetch_ohlcv: FetchOhlcv) -> None:
        self._fetch = fetch_ohlcv
        self._cache: dict[tuple[str, str, str, str], pd.DataFrame | None] = {}

    async def _read(
        self, symbol: str, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        key = (symbol, timeframe, from_iso, to_iso)
        if key in self._cache:
            return self._cache[key]
        try:
            records = await self._fetch(symbol, timeframe, from_iso, to_iso)
        except Exception as exc:  # noqa: BLE001 — per-symbol tolerance: log + return None
            logger.warning("provider|fetch_failed|symbol=%s|tf=%s|err=%s", symbol, timeframe, exc)
            self._cache[key] = None
            return None
        frame = records_to_frame(records)
        self._cache[key] = frame
        return frame

    async def fetch_screening(
        self, symbol: str, *, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        return await self._read(symbol, timeframe, from_iso, to_iso)

    async def fetch_execution(
        self, symbol: str, *, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        return await self._read(symbol, timeframe, from_iso, to_iso)

    def evict(self, symbol: str) -> int:
        """Drop all cached frames for ``symbol``; returns how many entries were freed."""
        keys = [k for k in self._cache if k[0] == symbol]
        for k in keys:
            del self._cache[k]
        if keys:
            logger.debug("provider|evict|symbol=%s|frames=%d", symbol, len(keys))
        return len(keys)
