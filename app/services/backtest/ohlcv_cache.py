"""
OHLCV fetch cache.

What this saves:
  list[dict] of normalized OHLCV rows — exactly what fetch_ohlcv_records() returns —
  serialized as gzipped pickle on local disk.

Why pickle.gz on disk (not parquet, not Redis):
  - Zero new dependencies (parquet would require pyarrow in the main app env;
    Redis would require infra). pickle + gzip is in the stdlib.
  - OHLCV is bulk numeric data (210k rows × 6 cols ≈ 5–15 MB gzipped). Disk is
    plenty fast — pickle.load is faster than parquet for this access pattern
    (read whole file once into memory).
  - If we ever need shared cache across multiple engine workers, swap to Redis
    or S3. For a single-server setup disk is the right call.

Smart TTL:
  - Window ends in the past → cache forever. Historical bars don't change.
  - Window includes today    → 1h TTL. Today's last bar is still updating.

Cache size guard:
  - Soft cap (CACHE_MAX_BYTES, default 5 GB). When exceeded, oldest files are
    deleted by mtime until under cap. No LRU bookkeeping — mtime is good enough.

Cache key:
  MD5 of (symbol, interval, from_utc, to_utc). Deterministic per request.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where to put the cache files. Lives next to the existing quant_engine/cache
# directory so we don't sprout a second cache root.
CACHE_DIR = Path(__file__).resolve().parents[3] / "quant_engine" / "cache" / "ohlcv_fetch"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 1 hour for windows that include today (today's last bar is still being printed).
CACHE_TTL_TODAY_SECONDS = int(os.getenv("BACKTEST_CACHE_TTL_TODAY_SECONDS", "3600"))

# 5 GB soft cap on the cache directory.
CACHE_MAX_BYTES = int(os.getenv("BACKTEST_CACHE_MAX_BYTES", str(5 * 1024 * 1024 * 1024)))


def _cache_key(symbol: str, interval: str, from_utc: str, to_utc: str) -> str:
    raw = f"{symbol}|{interval}|{from_utc}|{to_utc}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl.gz"


def _window_touches_today(to_utc: str) -> bool:
    """True if the requested window's end is today or in the future (UTC)."""
    try:
        end = datetime.fromisoformat(str(to_utc).replace("Z", "+00:00"))
    except ValueError:
        # If we can't parse, be conservative and treat as "live" (short TTL).
        return True
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return end.date() >= datetime.now(timezone.utc).date()


def _is_fresh(path: Path, to_utc: str) -> bool:
    if not path.exists():
        return False
    if not _window_touches_today(to_utc):
        # Past window — historical bars are immutable, cache never expires.
        return True
    age_seconds = datetime.now().timestamp() - path.stat().st_mtime
    return age_seconds <= CACHE_TTL_TODAY_SECONDS


def get_cached_records(
    symbol: str, interval: str, from_utc: str, to_utc: str,
) -> list[dict[str, Any]] | None:
    """Return cached rows if a fresh entry exists for this exact window, else None."""
    path = _cache_path(_cache_key(symbol, interval, from_utc, to_utc))
    if not _is_fresh(path, to_utc):
        if path.exists():
            logger.info(
                "🗑️  OHLCV cache stale | symbol=%s interval=%s from=%s to=%s",
                symbol, interval, from_utc, to_utc,
            )
        return None
    try:
        with gzip.open(path, "rb") as fh:
            rows = pickle.load(fh)
        if not isinstance(rows, list):
            raise TypeError(f"unexpected cache payload type: {type(rows).__name__}")
        logger.info(
            "✅ OHLCV cache HIT | symbol=%s interval=%s from=%s to=%s rows=%d",
            symbol, interval, from_utc, to_utc, len(rows),
        )
        return rows
    except Exception as exc:
        logger.warning("Failed to read OHLCV cache %s: %s — refetching", path.name, exc)
        return None


def save_records(
    symbol: str, interval: str, from_utc: str, to_utc: str,
    records: list[dict[str, Any]],
) -> None:
    """Persist the fetched rows. Best-effort — logs and continues on failure."""
    if not records:
        return
    path = _cache_path(_cache_key(symbol, interval, from_utc, to_utc))
    try:
        with gzip.open(path, "wb", compresslevel=4) as fh:
            pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "💾 OHLCV cached | symbol=%s interval=%s from=%s to=%s rows=%d",
            symbol, interval, from_utc, to_utc, len(records),
        )
        _enforce_size_cap()
    except Exception as exc:
        logger.warning("Failed to save OHLCV cache for %s/%s: %s", symbol, interval, exc)


def _enforce_size_cap() -> None:
    """Delete oldest files (by mtime) until directory size is under CACHE_MAX_BYTES."""
    files = sorted(CACHE_DIR.glob("*.pkl.gz"), key=lambda p: p.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    if total <= CACHE_MAX_BYTES:
        return
    deleted = 0
    while files and total > CACHE_MAX_BYTES:
        oldest = files.pop(0)
        size = oldest.stat().st_size
        try:
            oldest.unlink()
            total -= size
            deleted += 1
        except OSError as exc:
            logger.warning("Failed to evict cache file %s: %s", oldest.name, exc)
    if deleted:
        logger.info("OHLCV cache size cap reached — evicted %d oldest files", deleted)


def clear_cache() -> int:
    """Delete every cached file. Returns number deleted."""
    count = 0
    for f in CACHE_DIR.glob("*.pkl.gz"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    logger.info("Cleared %d OHLCV cache files", count)
    return count
