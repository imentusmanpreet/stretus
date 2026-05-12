"""
app/core/signal_performance_cache.py
─────────────────────────────────────
Phase 3: Data-driven signal parameter selection.

After every successful backtest the caller records which signal parameters
were used and what win-rate they produced.  The next time the same
(signal, symbol, timeframe) combination is requested, we return the
parameter set that historically gave the best win-rate — instead of always
falling back to the YAML defaults.

How it works
────────────
1. record_performance() is called from the backtest completion path with
   the signal name, stock symbol, timeframe, parameters used, and the
   win-rate returned by the quant engine.

2. get_best_params() is called from param_resolver.py before building a
   strategy.  If we have ≥ MIN_TRADES_FOR_CONFIDENCE trades recorded for
   this (signal, symbol, timeframe) key AND the stored params differ from
   the YAML default, we return the historically best params.  Otherwise we
   fall back to the supplied default.

Storage
───────
A single JSON file at app/core/signal_performance_cache.json.
The file is created automatically on the first write.
It is intentionally gitignored — it is runtime data, not source code.

Thread safety
─────────────
Reads and writes are wrapped in a file lock (fcntl on Linux/macOS).
On Windows the lock is skipped (only one process is expected there).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).parent / "signal_performance_cache.json"

# Minimum number of recorded trades before we trust the cached params
# over the YAML defaults.  Below this threshold we keep using defaults.
_MIN_TRADES_FOR_CONFIDENCE = 10


# ── File-level lock helpers ───────────────────────────────────────────────────

def _lock(fh):
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass


def _unlock(fh):
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass


# ── Internal I/O ──────────────────────────────────────────────────────────────

def _load() -> dict[str, Any]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            _lock(fh)
            data = json.load(fh)
            _unlock(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("signal_performance_cache|load_error: %s", exc)
        return {}


def _save(data: dict[str, Any]) -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            _lock(fh)
            json.dump(data, fh, indent=2)
            _unlock(fh)
    except Exception as exc:
        logger.warning("signal_performance_cache|save_error: %s", exc)


def _key(signal_name: str, symbol: str, timeframe: str) -> str:
    return f"{signal_name}:{symbol}:{timeframe}"


# ── Public API ────────────────────────────────────────────────────────────────

def get_best_params(
    signal_name: str,
    symbol: str,
    timeframe: str,
    fallback_params: dict,
) -> dict:
    """
    Return the historically best parameters for this (signal, symbol, timeframe).

    Falls back to fallback_params when:
    - the cache has no entry for this key, OR
    - fewer than MIN_TRADES_FOR_CONFIDENCE trades are recorded (not enough data), OR
    - the cache file cannot be read.

    This is called on every signal planning step so it must be fast.
    The JSON load is ~0.2 ms for a small cache — acceptable for a one-time
    call per strategy build.
    """
    if not symbol or not timeframe:
        return fallback_params

    cache = _load()
    entry = cache.get(_key(signal_name, symbol, timeframe))

    if not isinstance(entry, dict):
        logger.debug(
            "🔧 signal_performance_cache|miss|signal=%s|symbol=%s|tf=%s|reason=no_entry",
            signal_name, symbol, timeframe,
        )
        return fallback_params

    if entry.get("total_trades", 0) < _MIN_TRADES_FOR_CONFIDENCE:
        logger.debug(
            "🔧 signal_performance_cache|miss|signal=%s|symbol=%s|tf=%s"
            "|reason=insufficient_trades|recorded=%d|required=%d",
            signal_name, symbol, timeframe,
            entry.get("total_trades", 0),
            _MIN_TRADES_FOR_CONFIDENCE,
        )
        return fallback_params

    cached_params = entry.get("best_params")
    if not isinstance(cached_params, dict) or not cached_params:
        logger.debug(
            "🔧 signal_performance_cache|miss|signal=%s|symbol=%s|tf=%s|reason=empty_params",
            signal_name, symbol, timeframe,
        )
        return fallback_params

    logger.info(
        "🏆 signal_performance_cache|hit|signal=%s|symbol=%s|tf=%s"
        "|win_rate=%.2f%%|trades=%d|params=%s",
        signal_name, symbol, timeframe,
        entry.get("win_rate", 0.0) * 100,
        entry.get("total_trades", 0),
        cached_params,
    )
    return cached_params


def record_performance(
    signal_name: str,
    symbol: str,
    timeframe: str,
    params: dict,
    win_rate: float,
    total_trades: int,
) -> None:
    """
    Record backtest performance for a (signal, symbol, timeframe) combination.

    Call this after every completed backtest so the cache accumulates data.
    The entry is only updated when the new win_rate beats the stored best,
    or when we have too few trades to trust the existing entry.

    Example call (from chat_service.py after backtest completes):

        from app.core.signal_performance_cache import record_performance

        for signal in plan.get("entry", []) + plan.get("exit", []):
            record_performance(
                signal_name  = signal["name"],
                symbol       = builder.format_symbol(),
                timeframe    = builder.timeframe or "",
                params       = signal.get("params", {}),
                win_rate     = backtest_metrics.get("win_rate", 0.0) / 100.0,
                total_trades = backtest_metrics.get("total_trades", 0),
            )
    """
    if not signal_name or not symbol or not timeframe or total_trades < 1:
        return

    cache = _load()
    cache_key = _key(signal_name, symbol, timeframe)
    existing = cache.get(cache_key, {})

    existing_win_rate = float(existing.get("win_rate", -1.0))
    existing_trades = int(existing.get("total_trades", 0))

    should_update = (
        win_rate > existing_win_rate
        or existing_trades < _MIN_TRADES_FOR_CONFIDENCE
    )

    if not should_update:
        return

    cache[cache_key] = {
        "signal_name":  signal_name,
        "symbol":       symbol,
        "timeframe":    timeframe,
        "best_params":  dict(params),
        "win_rate":     round(float(win_rate), 4),
        "total_trades": int(total_trades),
        "last_updated": date.today().isoformat(),
    }
    _save(cache)

    logger.info(
        "signal_performance_cache|recorded|signal=%s|symbol=%s|tf=%s"
        "|win_rate=%.2f|trades=%d",
        signal_name, symbol, timeframe, win_rate, total_trades,
    )


def all_entries() -> list[dict]:
    """Return all stored entries — useful for admin/debug endpoints."""
    return list(_load().values())


def clear_cache() -> None:
    """Wipe the entire cache — useful in tests."""
    _save({})
