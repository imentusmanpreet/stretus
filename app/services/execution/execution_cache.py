"""
app/services/execution/execution_cache.py
──────────────────────────────────────────
Thread-safe in-process TTL cache for market data.

Each cache entry expires after `ttl_seconds` (default: settings.market_data_cache_ttl_seconds).
Separate stores exist for candles, LTP, and circuit limits so each can be
invalidated independently.

Usage:
    cache = MarketDataCache()
    cache.set_candles("RELIANCE.NS", "5m", df)
    df = cache.get_candles("RELIANCE.NS", "5m")   # None if expired
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import pandas as pd

from app.core.config import get_settings

settings = get_settings()


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl

    def is_alive(self) -> bool:
        return time.monotonic() < self.expires_at


class MarketDataCache:
    """
    Per-instance in-memory TTL cache (singleton used by MarketDataService).

    All public methods are thread-safe.
    """

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        self._ttl = float(ttl_seconds or settings.market_data_cache_ttl_seconds)
        self._candles: dict[str, _CacheEntry] = {}
        self._ltp: dict[str, _CacheEntry] = {}
        self._circuit: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # ── Candles ───────────────────────────────────────────────────────────────

    def _candles_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}::{timeframe}"

    def get_candles(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        with self._lock:
            entry = self._candles.get(self._candles_key(symbol, timeframe))
            if entry and entry.is_alive():
                return entry.value
            return None

    def set_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._candles[self._candles_key(symbol, timeframe)] = _CacheEntry(df, self._ttl)

    # ── LTP ───────────────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> Optional[float]:
        with self._lock:
            entry = self._ltp.get(symbol)
            if entry and entry.is_alive():
                return entry.value
            return None

    def set_ltp(self, symbol: str, ltp: float) -> None:
        with self._lock:
            self._ltp[symbol] = _CacheEntry(ltp, self._ttl)

    # ── Circuit limits ────────────────────────────────────────────────────────

    def get_circuit(self, symbol: str) -> Optional[dict]:
        with self._lock:
            entry = self._circuit.get(symbol)
            if entry and entry.is_alive():
                return entry.value
            return None

    def set_circuit(self, symbol: str, data: dict) -> None:
        with self._lock:
            self._circuit[symbol] = _CacheEntry(data, self._ttl)

    # ── Invalidation ─────────────────────────────────────────────────────────

    def invalidate(self, symbol: str, timeframe: Optional[str] = None) -> None:
        with self._lock:
            if timeframe:
                self._candles.pop(self._candles_key(symbol, timeframe), None)
            self._ltp.pop(symbol, None)
            self._circuit.pop(symbol, None)
