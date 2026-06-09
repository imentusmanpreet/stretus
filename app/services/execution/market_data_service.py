"""
app/services/execution/market_data_service.py
──────────────────────────────────────────────
Asset-class-aware live market-data service used by the execution layer.

Architecture
────────────
Per-venue clients live in ``app.services.execution.market_data`` and implement
a shared ``MarketDataClient`` interface (Upstox v2 for equity_cash, Binance
Spot REST for crypto_spot). This module is the thin caching/orchestration
layer above them:

  Public API (back-compat preserved)
  ──────────────────────────────────
    MarketDataService.fetch_candles(symbol, timeframe, lookback,
                                    db_instrument_key=None,
                                    asset_class=AssetClass.equity_cash)
    MarketDataService.fetch_ltp(symbol, db_instrument_key=None,
                                    asset_class=AssetClass.equity_cash)
    MarketDataService.fetch_circuit_limits(symbol, db_instrument_key=None,
                                    asset_class=AssetClass.equity_cash)

  Legacy module-level helpers
  ───────────────────────────
    _upstox_instrument_key(symbol)   — preserved for tests + back-compat.

All responses are cached in-process per (symbol, timeframe, asset_class).
On a network error the last cached value is returned as a fallback.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.schemas.execution import AssetClass
from app.services.execution.execution_cache import MarketDataCache
from app.services.execution.market_data import MarketDataClient, get_market_data_client

# Re-export for back-compat with callers/tests that import the legacy helper
# from this module (e.g. tests/test_api/test_execution_market_data_service.py).
from app.services.execution.market_data.upstox_client import (  # noqa: F401
    UpstoxClient,
    _upstox_instrument_key,
)

logger = logging.getLogger(__name__)

# Shared in-process cache; keys are namespaced by asset class so equity and
# crypto entries never collide even when symbols happen to overlap.
_cache = MarketDataCache()


def _cache_key(symbol: str, asset_class: AssetClass) -> str:
    """Compose a cache-namespaced key so asset classes are isolated."""
    return f"{asset_class.value}::{symbol}"


class MarketDataService:
    """
    Unified, asset-class-aware market-data client for the execution service.

    Instantiate once per request; the underlying per-venue clients are
    cached globally so HTTP sessions are reused.
    """

    def __init__(self) -> None:
        # Pre-warm the equity client so existing call sites that pass no
        # ``asset_class`` keep their warm-start latency.
        self._equity_client: MarketDataClient = get_market_data_client(AssetClass.equity_cash)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _client_for(self, asset_class: AssetClass) -> MarketDataClient:
        if asset_class == AssetClass.equity_cash:
            return self._equity_client
        return get_market_data_client(asset_class)

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        db_instrument_key: Optional[str] = None,
        *,
        asset_class: AssetClass = AssetClass.equity_cash,
    ) -> pd.DataFrame:
        """
        Return ``lookback`` OHLCV candles at ``timeframe`` resolution.

        ``db_instrument_key`` is the venue-native adapter symbol resolved from
        ``ref_data.instrument_adapter_mappings`` (e.g. ``"NSE_EQ|INE002A01018"``
        for Upstox, ``"BTCUSDT"`` for Binance). Callers should always pass it
        when available — clients only fall back to best-effort derivation when
        ``None``.

        Falls back to stale cache on network error.
        """
        key = _cache_key(symbol, asset_class)
        cached = _cache.get_candles(key, timeframe)
        if cached is not None:
            logger.debug(
                "cache hit | candles asset_class=%s symbol=%s timeframe=%s rows=%d",
                asset_class.value, symbol, timeframe, len(cached),
            )
            return cached

        client = self._client_for(asset_class)
        try:
            df = await client.fetch_candles(
                symbol, timeframe, lookback, adapter_symbol=db_instrument_key,
            )
            _cache.set_candles(key, timeframe, df)
            return df
        except Exception as exc:
            logger.warning(
                "fetch_candles failed | asset_class=%s symbol=%s timeframe=%s — %s. "
                "Checking stale cache.",
                asset_class.value, symbol, timeframe, exc,
            )
            stale = _cache.get_candles(key, timeframe)
            if stale is not None:
                return stale
            raise RuntimeError(
                f"No market data available for {symbol} ({timeframe}, "
                f"asset_class={asset_class.value}): {exc}"
            ) from exc

    async def fetch_ltp(
        self,
        symbol: str,
        db_instrument_key: Optional[str] = None,
        *,
        asset_class: AssetClass = AssetClass.equity_cash,
    ) -> float:
        """
        Return the last traded price. Falls back to the most recent cached
        candle close when the venue endpoint is unavailable.
        """
        key = _cache_key(symbol, asset_class)
        cached = _cache.get_ltp(key)
        if cached is not None:
            return cached

        client = self._client_for(asset_class)
        try:
            ltp = await client.fetch_ltp(symbol, adapter_symbol=db_instrument_key)
            _cache.set_ltp(key, ltp)
            return ltp
        except Exception as exc:
            logger.warning(
                "LTP unavailable | asset_class=%s symbol=%s adapter=%s — %s. "
                "Falling back to candle close.",
                asset_class.value, symbol, client.adapter_id, exc,
            )

        for tf in ("1m", "5m", "15m", "30m", "1h", "1d"):
            df = _cache.get_candles(key, tf)
            if df is not None and not df.empty:
                ltp = float(df["Close"].iloc[-1])
                _cache.set_ltp(key, ltp)
                logger.info(
                    "LTP from candle close | asset_class=%s symbol=%s tf=%s ltp=%.10g",
                    asset_class.value, symbol, tf, ltp,
                )
                return ltp

        raise RuntimeError(
            f"Cannot determine LTP for {symbol} (asset_class={asset_class.value}): "
            "venue unavailable and no candle cache."
        )

    async def fetch_circuit_limits(
        self,
        symbol: str,
        db_instrument_key: Optional[str] = None,
        *,
        asset_class: AssetClass = AssetClass.equity_cash,
    ) -> Optional[dict]:
        """
        Return ``{"upper_circuit": float, "lower_circuit": float}`` from the
        venue when available, else ``None``. For crypto the values come from
        the Binance 24h ticker (soft sanity band); for equity they come from
        the Upstox ``market-quote/quotes`` endpoint (hard circuit limits).
        """
        key = _cache_key(symbol, asset_class)
        cached = _cache.get_circuit(key)
        if cached is not None:
            return cached

        client = self._client_for(asset_class)
        try:
            data = await client.fetch_circuit_limits(
                symbol, adapter_symbol=db_instrument_key,
            )
            if data:
                _cache.set_circuit(key, data)
                logger.info(
                    "circuit limits | asset_class=%s symbol=%s upper=%s lower=%s",
                    asset_class.value, symbol,
                    data.get("upper_circuit"), data.get("lower_circuit"),
                )
            return data
        except Exception as exc:
            logger.warning(
                "fetch_circuit_limits failed | asset_class=%s symbol=%s — %s. "
                "Evaluator will use defaults.",
                asset_class.value, symbol, exc,
            )
            return None
