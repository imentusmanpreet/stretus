"""
app.services.execution.market_data.factory
Single entry point for picking the right MarketDataClient for a given
asset class. Clients are cached per (tenant_id, asset_class) so each tenant
gets isolated URLs/credentials while HTTP sessions are still reused.

When no tenant context is set (tests, scripts), falls back to env-based clients.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from app.core.config import get_settings
from app.core.tenant_context import TenantContext, get_current_tenant
from app.schemas.execution import AssetClass
from app.services.execution.market_data.backtest_feed_client import BacktestFeedClient
from app.services.execution.market_data.base import MarketDataClient
from app.services.execution.market_data.internal_client import InternalMarketDataClient
from app.services.execution.market_data.resilient_client import ResilientMarketDataClient

logger = logging.getLogger(__name__)

_CLIENTS: dict[tuple[str, str], MarketDataClient] = {}
_CLIENTS_LOCK = threading.Lock()


def _cache_key(tenant_id: str, asset_class: AssetClass) -> tuple[str, str]:
    return tenant_id, asset_class.value


def _build_client(asset_class: AssetClass, tenant: Optional[TenantContext]) -> MarketDataClient:
    # All asset classes are routed through InternalMarketDataClient, which calls
    # stretus-backend (user-gateway). The gateway owns broker selection — adding a
    # new broker requires no changes here or anywhere in the AI service.
    #
    # Equity additionally honours EQUITY_MARKET_DATA_SOURCE so the live gateway feed
    # can degrade to the proven backtest feed instead of leaving a strategy blind.
    is_crypto = asset_class == AssetClass.crypto_spot
    primary = InternalMarketDataClient(is_crypto=is_crypto)
    if asset_class == AssetClass.equity_cash:
        return _wrap_equity_source(primary)
    return primary


def _wrap_equity_source(primary: MarketDataClient) -> MarketDataClient:
    """Apply the configured equity data-source policy (EQUITY_MARKET_DATA_SOURCE).

    The live broker feed is reached through the stretus-backend gateway
    (``InternalMarketDataClient``), passed in as ``primary``:

      * ``upstox`` (live)    → the gateway live client as-is.
      * ``backtest_feed``    → the historical backtest feed only (paper/parity, no broker token).
      * ``resilient`` (default) → gateway primary with the backtest feed as a logged fallback, so a
        gateway/broker outage degrades gracefully instead of leaving the strategy blind.
    """
    source = (get_settings().equity_market_data_source or "resilient").strip().lower()
    if source == "upstox":
        return primary
    if source == "backtest_feed":
        return BacktestFeedClient()
    return ResilientMarketDataClient([primary, BacktestFeedClient()])


def get_market_data_client(
    asset_class: AssetClass,
    *,
    tenant: Optional[TenantContext] = None,
) -> MarketDataClient:
    """Return a cached ``MarketDataClient`` for ``asset_class`` and tenant."""
    tenant = tenant or get_current_tenant()
    tenant_id = tenant.tenant_id if tenant else "legacy"

    key = _cache_key(tenant_id, asset_class)
    with _CLIENTS_LOCK:
        cached = _CLIENTS.get(key)
        if cached is not None:
            return cached

        client = _build_client(asset_class, tenant)
        _CLIENTS[key] = client
        logger.info(
            "tenant_config|event=market_data_client_created|tenant_id=%s|"
            "asset_class=%s|adapter_id=%s|class=%s",
            tenant_id,
            asset_class.value,
            client.adapter_id,
            client.__class__.__name__,
        )
        return client


def reset_market_data_clients() -> None:
    """Test-only helper: drop cached clients so ``get_market_data_client`` rebuilds them."""
    with _CLIENTS_LOCK:
        _CLIENTS.clear()
