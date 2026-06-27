"""
app/services/data/parquet_store.py — a columnar Parquet-on-disk OHLCV store (§4.2).

Beyond a few hundred symbols, broker-fetch + pandas does not scale; a dynamic universe
needs range-sliced reads by ``(symbol, timeframe, time range)`` from a local columnar store.
This implements the :class:`app.services.data.provider.DataProvider` Protocol on
Parquet-on-disk, so it drops in behind the resolver/portfolio loop with **no caller change**
(that is the point of the abstraction — ClickHouse can replace it later the same way).

Layout: ``<root>/<timeframe>/<safe_symbol>.parquet``, one file per (symbol, timeframe),
timestamp-indexed. Reads are range-sliced and bounded at ``to`` (no look-ahead, Invariant 4).
Ingestion is idempotent (upsert + dedupe by timestamp) and resumable (§4.4). On a read miss
an optional ``fallback_fetch`` lazily fetches and ingests, so the store warms transparently.

Requires a Parquet engine (``pyarrow``). The import is guarded so importing this module never
breaks environments without it — only constructing the store does, with a clear message.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import pandas as pd

from .provider import FetchOhlcv, records_to_frame

logger = logging.getLogger(__name__)

_SAFE_SYMBOL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:  # pragma: no cover - environment dependent
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def _safe_symbol(symbol: str) -> str:
    """Filesystem-safe filename stem for a symbol (e.g. ``M&M.NS`` → ``M_M.NS``)."""
    return _SAFE_SYMBOL_RE.sub("_", symbol.strip())


class ParquetOhlcvStore:
    """Range-sliced columnar OHLCV store behind the :class:`DataProvider` Protocol.

    Args:
      root_dir: directory rooting the store (created on demand).
      fallback_fetch: optional async fetcher used on a read miss to lazily fetch+ingest
        a symbol's bars (mirrors the scanner's injectable fetcher). When ``None``, a miss
        returns ``None`` (per-symbol tolerance — never raises into the resolver).
    """

    def __init__(self, root_dir: str | Path, *, fallback_fetch: FetchOhlcv | None = None) -> None:
        if not _parquet_engine_available():
            raise ImportError(
                "ParquetOhlcvStore needs a Parquet engine. Install pyarrow "
                "(`pip install pyarrow`). The DataProvider Protocol lets you use "
                "CachingFetchProvider instead until then — same interface."
            )
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fallback = fallback_fetch
        self._lock = asyncio.Lock()

    def _path(self, symbol: str, timeframe: str) -> Path:
        return self.root / timeframe.strip().lower() / f"{_safe_symbol(symbol)}.parquet"

    # ── ingestion (idempotent, resumable) ─────────────────────────────────────
    def ingest(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Upsert ``df`` (timestamp-indexed OHLCV) for ``(symbol, timeframe)``.

        Merges with any existing rows, de-duplicates by timestamp (last wins), and sorts —
        so re-running a backfill is a no-op and an incremental top-up just appends. Returns
        the total row count after the upsert.
        """
        if df is None or df.empty:
            return 0
        path = self._path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = df.copy()
        incoming.index = pd.to_datetime(incoming.index)
        if path.exists():
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index)
            combined = pd.concat([existing, incoming])
        else:
            combined = incoming
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.to_parquet(path)
        logger.debug("parquet|ingest|symbol=%s|tf=%s|rows=%d", symbol, timeframe, len(combined))
        return len(combined)

    def _read_slice(
        self, symbol: str, timeframe: str, from_ts: pd.Timestamp, to_ts: pd.Timestamp
    ) -> pd.DataFrame | None:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        # Range-slice, bounded at `to` → no look-ahead (Invariant 4).
        sliced = df.loc[(df.index >= from_ts) & (df.index <= to_ts)]
        return sliced if not sliced.empty else None

    async def _read(
        self, symbol: str, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        from_ts = pd.to_datetime(from_iso).tz_localize(None) if pd.to_datetime(from_iso).tzinfo else pd.to_datetime(from_iso)
        to_ts = pd.to_datetime(to_iso).tz_localize(None) if pd.to_datetime(to_iso).tzinfo else pd.to_datetime(to_iso)
        sliced = self._read_slice(symbol, timeframe, from_ts, to_ts)
        if sliced is not None:
            return sliced
        if self._fallback is None:
            return None
        # Lazy fetch + ingest, then re-slice. Serialized per-store to avoid duplicate fetches.
        async with self._lock:
            sliced = self._read_slice(symbol, timeframe, from_ts, to_ts)
            if sliced is not None:
                return sliced
            try:
                records = await self._fallback(symbol, timeframe, from_iso, to_iso)
            except Exception as exc:  # noqa: BLE001 — per-symbol tolerance
                logger.warning("parquet|fallback_fetch_failed|symbol=%s|err=%s", symbol, exc)
                return None
            frame = records_to_frame(records)
            if frame is None or frame.empty:
                return None
            self.ingest(symbol, timeframe, frame)
            return self._read_slice(symbol, timeframe, from_ts, to_ts)

    async def fetch_screening(
        self, symbol: str, *, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        return await self._read(symbol, timeframe, from_iso, to_iso)

    async def fetch_execution(
        self, symbol: str, *, timeframe: str, from_iso: str, to_iso: str
    ) -> pd.DataFrame | None:
        return await self._read(symbol, timeframe, from_iso, to_iso)
