"""
app/core/tenant_context.py
Request-scoped tenant context for multi-tenant broker/market-data config.

Callers (strategy / order-execution services) pass ``x-tenant-id``.
Single-tenant deployments set ``TENANT_CODE`` in env instead.

Log prefix for all tenant resolution: ``tenant_config|`` (grep-friendly for debugging).
"""
from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Optional

from app.schemas.execution import AssetClass

logger = logging.getLogger(__name__)

_LOG_PREFIX = "tenant_config|"

# Default adapter per asset class when tenant_adapters has no explicit mapping.
# Must match ref_data seed adapter ids.
DEFAULT_ADAPTER_BY_ASSET_CLASS: dict[AssetClass, str] = {
    AssetClass.equity_cash: "upstox_rest",
    AssetClass.crypto_spot: "binance_rest",
}


def broker_credential_key(tenant_id: str, adapter_id: str, field_name: str) -> str:
    """Build ref_data.system_configs key for tenant-scoped broker credentials."""
    return f"tenant.{tenant_id}.broker.credentials.{adapter_id}.{field_name}"


@dataclass(frozen=True)
class AdapterRuntimeConfig:
    adapter_id: str
    base_url: str
    access_token: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    auth_scheme: Optional[str] = None
    credential_source: str = "unknown"  # db | env_fallback


@dataclass
class TenantContext:
    tenant_id: str
    tenant_code: Optional[str] = None
    enabled_asset_classes: list[str] = field(default_factory=list)
    adapters: dict[str, AdapterRuntimeConfig] = field(default_factory=dict)
    is_legacy: bool = False

    def adapter_for(self, asset_class: AssetClass) -> Optional[AdapterRuntimeConfig]:
        preferred = DEFAULT_ADAPTER_BY_ASSET_CLASS.get(asset_class)
        if preferred and preferred in self.adapters:
            return self.adapters[preferred]
        for adapter_id, cfg in self.adapters.items():
            if asset_class == AssetClass.equity_cash and "upstox" in adapter_id:
                return cfg
            if asset_class == AssetClass.crypto_spot and "binance" in adapter_id:
                return cfg
        if preferred and self.is_legacy:
            return self.adapters.get(preferred)
        return None

    def default_adapter_id(self, asset_class: AssetClass) -> str:
        cfg = self.adapter_for(asset_class)
        if cfg is not None:
            return cfg.adapter_id
        return DEFAULT_ADAPTER_BY_ASSET_CLASS[asset_class]

    def log_summary(self) -> str:
        adapter_ids = sorted(self.adapters.keys())
        return (
            f"{_LOG_PREFIX}tenant_id={self.tenant_id}|code={self.tenant_code or '-'}|"
            f"legacy={self.is_legacy}|adapters={adapter_ids}|"
            f"asset_classes={self.enabled_asset_classes}"
        )


_current_tenant: ContextVar[Optional[TenantContext]] = ContextVar(
    "current_tenant_context", default=None
)


def get_current_tenant() -> Optional[TenantContext]:
    return _current_tenant.get()


def set_current_tenant(ctx: TenantContext) -> Token:
    logger.debug("%scontext_set|%s", _LOG_PREFIX, ctx.log_summary())
    return _current_tenant.set(ctx)


def reset_current_tenant(token: Token) -> None:
    _current_tenant.reset(token)


def default_adapter_id_for_context(asset_class: AssetClass) -> str:
    ctx = get_current_tenant()
    if ctx is not None:
        return ctx.default_adapter_id(asset_class)
    return DEFAULT_ADAPTER_BY_ASSET_CLASS[asset_class]
