"""
app.services.execution.market_data.factory
──────────────────────────────────────────
Single entry point for picking the right ``MarketDataClient`` for a given
asset class. The cached instance per asset class is process-wide so HTTP
sessions and connection pools are reused.
"""
from __future__ import annotations

import logging
import threading

from app.schemas.execution import AssetClass
from app.services.execution.market_data.base import MarketDataClient
from app.services.execution.market_data.binance_client import BinanceClient
from app.services.execution.market_data.upstox_client import UpstoxClient

logger = logging.getLogger(__name__)

_CLIENTS: dict[AssetClass, MarketDataClient] = {}
_CLIENTS_LOCK = threading.Lock()


def get_market_data_client(asset_class: AssetClass) -> MarketDataClient:
    """Return the singleton ``MarketDataClient`` for ``asset_class``.

    Raises ``ValueError`` for unsupported asset classes — the live evaluator
    catches and surfaces this as a user-facing error before any market-data
    call is attempted.
    """
    with _CLIENTS_LOCK:
        cached = _CLIENTS.get(asset_class)
        if cached is not None:
            return cached

        if asset_class == AssetClass.equity_cash:
            client: MarketDataClient = UpstoxClient()
        elif asset_class == AssetClass.crypto_spot:
            client = BinanceClient()
        else:
            raise ValueError(
                f"No market-data client is registered for asset_class={asset_class!r}."
            )

        _CLIENTS[asset_class] = client
        logger.info(
            "Market-data client created | asset_class=%s adapter_id=%s class=%s",
            asset_class.value, client.adapter_id, client.__class__.__name__,
        )
        return client


def reset_market_data_clients() -> None:
    """Test-only helper: drop cached clients so ``get_market_data_client`` rebuilds them."""
    with _CLIENTS_LOCK:
        _CLIENTS.clear()
