"""
app.services.execution.market_data.resilient_client
─────────────────────────────────────────────────────
A :class:`MarketDataClient` that composes an ordered list of clients into a primary→fallback
chain. Each call tries the primary; on a RAISED failure (transport/auth/empty payload) it logs a
WARNING and tries the next client, and so on. The first success wins; if all fail, the last
error is re-raised.

This is how the live evaluator stays robust without going blind: e.g. ``[UpstoxClient,
BacktestFeedClient]`` serves live broker data when available and transparently degrades to the
proven backtest feed when the broker API fails (token expiry, rate-limit, outage) — never
silently (every fallback is logged).

Important: a fallback is taken only on an EXCEPTION. A client that legitimately returns ``None``
(e.g. ``fetch_circuit_limits`` when a venue exposes no bands) is a valid result, NOT a failure,
so the chain stops there rather than masking it with a downstream client.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.services.execution.market_data.base import MarketDataClient

logger = logging.getLogger(__name__)


class ResilientMarketDataClient(MarketDataClient):
    """Primary→fallback chain over several :class:`MarketDataClient`s."""

    def __init__(self, clients: list[MarketDataClient], *, ltp_fallback: bool = False) -> None:
        clients = [c for c in clients if c is not None]
        if not clients:
            raise ValueError("ResilientMarketDataClient requires at least one client.")
        self._clients = clients
        # LTP is the price we DECIDE/TRADE on — it must be live, never a stale historical close.
        # So by default the fallback chain applies ONLY to candles/circuit (closed bars are safe);
        # fetch_ltp uses the live primary alone and fails closed if it can't get a live price.
        self._ltp_fallback = ltp_fallback
        chain = "→".join(c.adapter_id for c in clients)
        self.adapter_id = f"resilient({chain}; ltp={'chain' if ltp_fallback else clients[0].adapter_id})"

    async def _attempt(self, method: str, symbol: str, *args, **kwargs):
        last_exc: Exception | None = None
        for i, client in enumerate(self._clients):
            try:
                result = await getattr(client, method)(symbol, *args, **kwargs)
                if i > 0:
                    logger.warning(
                        "🛟 market_data fallback | %s served %s(%s) after %d primary failure(s)",
                        client.adapter_id, method, symbol, i,
                    )
                return result
            except Exception as exc:  # noqa: BLE001 — try the next source, surface if none work
                last_exc = exc
                more = i < len(self._clients) - 1
                logger.warning(
                    "⚠️ market_data | %s.%s(%s) failed — %s%s",
                    client.adapter_id, method, symbol, exc,
                    " → trying fallback" if more else " (no fallback left)",
                )
        assert last_exc is not None
        raise last_exc

    async def fetch_candles(
        self, symbol: str, timeframe: str, lookback: int, adapter_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        return await self._attempt("fetch_candles", symbol, timeframe, lookback, adapter_symbol=adapter_symbol)

    async def fetch_ltp(self, symbol: str, adapter_symbol: Optional[str] = None) -> float:
        if self._ltp_fallback:
            return await self._attempt("fetch_ltp", symbol, adapter_symbol=adapter_symbol)
        # Fail-closed: a live price or nothing — never trade on a stale historical close.
        primary = self._clients[0]
        try:
            return await primary.fetch_ltp(symbol, adapter_symbol=adapter_symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "⛔ market_data | %s LTP unavailable for %s and LTP fallback is DISABLED "
                "(no stale-price trading) — failing closed: %s", primary.adapter_id, symbol, exc,
            )
            raise

    async def fetch_circuit_limits(self, symbol: str, adapter_symbol: Optional[str] = None) -> Optional[dict]:
        return await self._attempt("fetch_circuit_limits", symbol, adapter_symbol=adapter_symbol)
