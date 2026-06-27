"""Phase B — Parquet columnar OHLCV store behind the DataProvider Protocol (§4.2).

Skips when no Parquet engine (pyarrow/fastparquet) is installed — the store is ready,
the dependency is an install step. Covers idempotent ingest, range-sliced reads bounded
at `to` (no look-ahead), and lazy fetch+ingest on a miss.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.services.data.parquet_store import _parquet_engine_available

pytestmark = pytest.mark.skipif(
    not _parquet_engine_available(), reason="no Parquet engine (pyarrow/fastparquet) installed"
)

ASOF = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _frame(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp(ASOF).tz_localize(None), periods=n, freq="D")
    return pd.DataFrame(
        {"open": range(n), "high": range(n), "low": range(n),
         "close": range(n), "volume": [100] * n},
        index=idx,
    ).astype(float)


def test_ingest_is_idempotent(tmp_path):
    from app.services.data.parquet_store import ParquetOhlcvStore
    store = ParquetOhlcvStore(tmp_path)
    n1 = store.ingest("AAA", "1d", _frame(30))
    n2 = store.ingest("AAA", "1d", _frame(30))  # same rows again
    assert n1 == n2 == 30  # dedupe by timestamp — no duplication


@pytest.mark.asyncio
async def test_range_sliced_read_bounded_at_to(tmp_path):
    from app.services.data.parquet_store import ParquetOhlcvStore
    store = ParquetOhlcvStore(tmp_path)
    store.ingest("AAA", "1d", _frame(30))
    mid = (pd.Timestamp(ASOF).tz_localize(None) - timedelta(days=10))
    out = await store.fetch_screening(
        "AAA", timeframe="1d",
        from_iso="2025-01-01T00:00:00Z", to_iso=mid.isoformat() + "Z")
    assert out is not None
    # No bar after `to` (no look-ahead).
    assert out.index.max() <= mid


@pytest.mark.asyncio
async def test_lazy_fallback_fetch_and_ingest(tmp_path):
    from app.services.data.parquet_store import ParquetOhlcvStore

    async def fallback(symbol, interval, from_iso, to_iso):
        return [
            {"timestamp": (ASOF - timedelta(days=i)).isoformat().replace("+00:00", "Z"),
             "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for i in range(20)
        ]

    store = ParquetOhlcvStore(tmp_path, fallback_fetch=fallback)
    out = await store.fetch_execution(
        "BBB", timeframe="1d", from_iso="2025-01-01T00:00:00Z",
        to_iso=ASOF.isoformat().replace("+00:00", "Z"))
    assert out is not None and len(out) > 0
    # Now cached on disk — a second read needs no fallback.
    again = await store.fetch_execution(
        "BBB", timeframe="1d", from_iso="2025-01-01T00:00:00Z",
        to_iso=ASOF.isoformat().replace("+00:00", "Z"))
    assert again is not None and len(again) == len(out)


@pytest.mark.asyncio
async def test_missing_symbol_returns_none_without_fallback(tmp_path):
    from app.services.data.parquet_store import ParquetOhlcvStore
    store = ParquetOhlcvStore(tmp_path)
    out = await store.fetch_screening(
        "NOPE", timeframe="1d", from_iso="2025-01-01T00:00:00Z",
        to_iso=ASOF.isoformat().replace("+00:00", "Z"))
    assert out is None
