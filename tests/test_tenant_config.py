"""
tests/test_tenant_config.py
Unit tests for multi-tenant config (no DB required for key helpers / legacy path).
"""
from __future__ import annotations

from app.core.config import Settings
from app.core.tenant_context import (
    TenantContext,
    broker_credential_key,
    default_adapter_id_for_context,
    reset_current_tenant,
    set_current_tenant,
)
from app.schemas.execution import AssetClass
from app.services.execution.market_data.factory import (
    get_market_data_client,
    reset_market_data_clients,
)
from app.services.execution.tenant_config_service import (
    _build_legacy_context,
    _legacy_adapter_from_env,
    clear_tenant_config_cache,
)


class TestCredentialKeys:
    def test_broker_credential_key_format(self) -> None:
        tid = "11111111-1111-1111-1111-111111111111"
        assert (
            broker_credential_key(tid, "upstox_rest", "access_token")
            == f"tenant.{tid}.broker.credentials.upstox_rest.access_token"
        )


class TestLegacyFallback:
    def test_legacy_context_uses_env(self) -> None:
        settings = Settings(
            market_data_url="https://api.upstox.com/v2",
            upstox_access_token="test-token",
            crypto_market_data_url="https://api.binance.com",
        )
        ctx = _build_legacy_context(settings)
        assert ctx.is_legacy is True
        assert ctx.tenant_id == "legacy"
        upstox = ctx.adapter_for(AssetClass.equity_cash)
        assert upstox is not None
        assert upstox.base_url == "https://api.upstox.com/v2"
        assert upstox.access_token == "test-token"
        assert upstox.credential_source == "env_fallback"

    def test_legacy_adapter_from_env_upstox(self) -> None:
        settings = Settings(upstox_access_token="abc")
        cfg = _legacy_adapter_from_env("upstox_rest", settings)
        assert cfg.access_token == "abc"


class TestTenantContextAdapterSelection:
    def test_default_adapter_without_context(self) -> None:
        assert default_adapter_id_for_context(AssetClass.equity_cash) == "upstox_rest"

    def test_tenant_context_uses_gateway_with_resilient_equity(self, monkeypatch) -> None:
        from app.core.tenant_context import AdapterRuntimeConfig

        # Market data for every asset class now flows through the stretus-backend gateway
        # (InternalMarketDataClient); the gateway — not the tenant adapter — owns broker
        # selection. The default equity policy still wraps that gateway client in a resilient
        # (gateway→backtest_feed) chain. Force the policy + gateway settings so the test is
        # independent of any ambient .env (a local EQUITY_MARKET_DATA_SOURCE would change it).
        import app.services.execution.market_data.factory as fac
        import app.services.execution.market_data.internal_client as internal
        fake_settings = type("S", (), {
            "equity_market_data_source": "resilient",
            "historical_data_url": "http://gateway.test",
            "market_data_timeout_seconds": 5.0,
        })()
        monkeypatch.setattr(fac, "get_settings", lambda: fake_settings)
        monkeypatch.setattr(internal, "get_settings", lambda: fake_settings)

        ctx = TenantContext(
            tenant_id="t1",
            adapters={
                "upstox_rest": AdapterRuntimeConfig(
                    adapter_id="upstox_rest",
                    base_url="https://api.upstox.com/v2",
                )
            },
        )
        token = set_current_tenant(ctx)
        try:
            # Adapter selection metadata is unchanged; only the market-data client routing moved.
            assert default_adapter_id_for_context(AssetClass.equity_cash) == "upstox_rest"
            client = get_market_data_client(AssetClass.equity_cash, tenant=ctx)
            from app.services.execution.market_data.backtest_feed_client import BacktestFeedClient
            from app.services.execution.market_data.internal_client import InternalMarketDataClient
            from app.services.execution.market_data.resilient_client import ResilientMarketDataClient
            assert isinstance(client, ResilientMarketDataClient)
            assert isinstance(client._clients[0], InternalMarketDataClient)  # live gateway primary
            assert isinstance(client._clients[1], BacktestFeedClient)        # proven fallback
        finally:
            reset_current_tenant(token)
            reset_market_data_clients()
            clear_tenant_config_cache()
