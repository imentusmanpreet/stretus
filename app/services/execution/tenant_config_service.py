"""
app/services/execution/tenant_config_service.py
Load per-tenant broker URLs and credentials from ref_data only.

Credential keys (ref_data.system_configs):
  tenant.<uuid>.broker.credentials.<adapter_id>.access_token   - Upstox / bearer
  tenant.<uuid>.broker.credentials.<adapter_id>.api_key        - CEX / api_key
  tenant.<uuid>.broker.credentials.<adapter_id>.secret_key     - CEX HMAC

When DB values are missing, falls back to legacy env vars (UPSTOX_*, MARKET_DATA_URL)
so existing single-tenant deployments keep working. All paths log with ``tenant_config|``.

See also: stretus-backend/pkg/exchange/cex/vault.go (global CEX pattern; we use
tenant-prefixed keys here per product decision).
"""
from __future__ import annotations

import logging
import time
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.tenant_context import (
    AdapterRuntimeConfig,
    DEFAULT_ADAPTER_BY_ASSET_CLASS,
    TenantContext,
    broker_credential_key,
)
from app.schemas.execution import AssetClass

logger = logging.getLogger(__name__)

_LOG_PREFIX = "tenant_config|"
_CACHE_TTL_SECONDS = 30.0
_tenant_cache: dict[str, tuple[float, TenantContext]] = {}


def _cache_get(tenant_id: str) -> Optional[TenantContext]:
    entry = _tenant_cache.get(tenant_id)
    if entry is None:
        return None
    expires_at, ctx = entry
    if time.monotonic() > expires_at:
        _tenant_cache.pop(tenant_id, None)
        return None
    return ctx


def _cache_set(tenant_id: str, ctx: TenantContext) -> None:
    _tenant_cache[tenant_id] = (time.monotonic() + _CACHE_TTL_SECONDS, ctx)


def clear_tenant_config_cache() -> None:
    """Test helper: drop cached tenant configs."""
    _tenant_cache.clear()


async def _lookup_tenant_by_code(db: AsyncSession, code: str) -> Optional[tuple[str, str]]:
    row = await db.execute(
        text(
            """
            SELECT id::text, code
            FROM ref_data.tenants
            WHERE code = :code AND status = 'active'
            LIMIT 1
            """
        ),
        {"code": code.strip()},
    )
    result = row.fetchone()
    if not result:
        return None
    return str(result[0]), str(result[1])


async def _validate_tenant_id(db: AsyncSession, tenant_id: str) -> Optional[tuple[str, str]]:
    try:
        UUID(tenant_id)
    except ValueError:
        logger.warning("%sevent=invalid_tenant_id|value=%s", _LOG_PREFIX, tenant_id)
        return None
    row = await db.execute(
        text(
            """
            SELECT id::text, code
            FROM ref_data.tenants
            WHERE id = :id::uuid AND status = 'active'
            LIMIT 1
            """
        ),
        {"id": tenant_id},
    )
    result = row.fetchone()
    if not result:
        logger.warning("%sevent=tenant_not_found|tenant_id=%s", _LOG_PREFIX, tenant_id)
        return None
    return str(result[0]), str(result[1])


async def _load_system_config(db: AsyncSession, key: str) -> Optional[str]:
    row = await db.execute(
        text("SELECT value FROM ref_data.system_configs WHERE key = :key LIMIT 1"),
        {"key": key},
    )
    result = row.fetchone()
    if not result:
        return None
    value = str(result[0] or "").strip()
    return value or None


async def _load_tenant_adapters(
    db: AsyncSession, tenant_id: str
) -> list[tuple[str, str, str]]:
    """Return (adapter_id, base_url, auth_scheme) for enabled tenant adapters."""
    rows = await db.execute(
        text(
            """
            SELECT ta.adapter_id,
                   COALESCE(a.base_url, ''),
                   COALESCE(a.auth_scheme, '')
            FROM ref_data.tenant_adapters ta
            JOIN ref_data.adapters a ON a.id = ta.adapter_id
            WHERE ta.tenant_id = :tenant_id::uuid
              AND ta.is_enabled = true
              AND a.is_active = true
            ORDER BY ta.adapter_id
            """
        ),
        {"tenant_id": tenant_id},
    )
    return [(str(r[0]), str(r[1] or ""), str(r[2] or "")) for r in rows.fetchall()]


async def _load_entitlements(db: AsyncSession, tenant_id: str) -> list[str]:
    rows = await db.execute(
        text(
            """
            SELECT asset_class_id
            FROM ref_data.tenant_asset_class_entitlements
            WHERE tenant_id = :tenant_id::uuid AND is_enabled = true
            ORDER BY asset_class_id
            """
        ),
        {"tenant_id": tenant_id},
    )
    return [str(r[0]) for r in rows.fetchall()]


def _legacy_adapter_from_env(
    adapter_id: str,
    settings: Settings,
) -> AdapterRuntimeConfig:
    """Build adapter config from env vars - backward-compatible single-tenant path."""
    if "upstox" in adapter_id:
        base = (settings.market_data_url or "https://api.upstox.com/v2").rstrip("/")
        token = settings.upstox_access_token or None
        return AdapterRuntimeConfig(
            adapter_id=adapter_id,
            base_url=base,
            access_token=token,
            api_key=settings.upstox_api_key or None,
            auth_scheme="api_key_secret",
            credential_source="env_fallback",
        )
    if "binance" in adapter_id:
        base = (settings.crypto_market_data_url or "https://api.binance.com").rstrip("/")
        return AdapterRuntimeConfig(
            adapter_id=adapter_id,
            base_url=base,
            auth_scheme="api_key_secret",
            credential_source="env_fallback",
        )
    return AdapterRuntimeConfig(
        adapter_id=adapter_id,
        base_url="",
        credential_source="env_fallback",
    )


async def _build_adapter_config(
    db: AsyncSession,
    tenant_id: str,
    adapter_id: str,
    base_url: str,
    auth_scheme: str,
    settings: Settings,
) -> AdapterRuntimeConfig:
    access_token = await _load_system_config(
        db, broker_credential_key(tenant_id, adapter_id, "access_token")
    )
    api_key = await _load_system_config(
        db, broker_credential_key(tenant_id, adapter_id, "api_key")
    )
    secret_key = await _load_system_config(
        db, broker_credential_key(tenant_id, adapter_id, "secret_key")
    )

    credential_source = "db"
    if not any((access_token, api_key, secret_key)):
        legacy = _legacy_adapter_from_env(adapter_id, settings)
        logger.info(
            "%sevent=credential_env_fallback|tenant_id=%s|adapter_id=%s|"
            "hint=seed ref_data.system_configs with tenant.<uuid>.broker.credentials.*",
            _LOG_PREFIX,
            tenant_id,
            adapter_id,
        )
        access_token = legacy.access_token
        api_key = legacy.api_key
        secret_key = legacy.secret_key
        credential_source = "env_fallback"

    resolved_base = base_url.rstrip("/") if base_url else ""
    if not resolved_base:
        legacy = _legacy_adapter_from_env(adapter_id, settings)
        resolved_base = legacy.base_url
        logger.info(
            "%sevent=base_url_env_fallback|tenant_id=%s|adapter_id=%s|base_url=%s",
            _LOG_PREFIX,
            tenant_id,
            adapter_id,
            resolved_base,
        )

    cfg = AdapterRuntimeConfig(
        adapter_id=adapter_id,
        base_url=resolved_base,
        access_token=access_token,
        api_key=api_key,
        secret_key=secret_key,
        auth_scheme=auth_scheme or None,
        credential_source=credential_source,
    )
    logger.debug(
        "%sevent=adapter_resolved|tenant_id=%s|adapter_id=%s|base_url=%s|"
        "has_token=%s|has_api_key=%s|credential_source=%s",
        _LOG_PREFIX,
        tenant_id,
        adapter_id,
        resolved_base,
        bool(access_token),
        bool(api_key),
        credential_source,
    )
    return cfg


def _build_legacy_context(settings: Settings) -> TenantContext:
    adapters: dict[str, AdapterRuntimeConfig] = {}
    for asset_class in AssetClass:
        adapter_id = DEFAULT_ADAPTER_BY_ASSET_CLASS[asset_class]
        adapters[adapter_id] = _legacy_adapter_from_env(adapter_id, settings)

    ctx = TenantContext(
        tenant_id="legacy",
        tenant_code=None,
        enabled_asset_classes=[ac.value for ac in AssetClass],
        adapters=adapters,
        is_legacy=True,
    )
    logger.info(
        "%sevent=legacy_mode|reason=no_tenant_context|adapters=%s|"
        "hint=set TENANT_CODE or pass x-tenant-id header",
        _LOG_PREFIX,
        sorted(adapters.keys()),
    )
    return ctx


async def load_tenant_context(
    db: AsyncSession,
    *,
    tenant_id_header: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> TenantContext:
    """
    Resolve tenant context for the current request.

    Priority:
      1. ``x-tenant-id`` header (UUID)
      2. ``TENANT_CODE`` env ? lookup UUID in ref_data.tenants
      3. Legacy env fallback (UPSTOX_*, MARKET_DATA_URL, ...)
    """
    settings = settings or get_settings()
    resolved_id: Optional[str] = None
    resolved_code: Optional[str] = None

    header = (tenant_id_header or "").strip()
    if header:
        validated = await _validate_tenant_id(db, header)
        if validated:
            resolved_id, resolved_code = validated
        else:
            logger.warning(
                "%sevent=header_tenant_invalid|header=%s|falling_back=tenant_code_or_legacy",
                _LOG_PREFIX,
                header,
            )

    if resolved_id is None and settings.tenant_code:
        by_code = await _lookup_tenant_by_code(db, settings.tenant_code)
        if by_code:
            resolved_id, resolved_code = by_code
            logger.info(
                "%sevent=tenant_from_env|tenant_code=%s|tenant_id=%s",
                _LOG_PREFIX,
                settings.tenant_code,
                resolved_id,
            )
        else:
            logger.warning(
                "%sevent=tenant_code_not_found|tenant_code=%s",
                _LOG_PREFIX,
                settings.tenant_code,
            )

    if resolved_id is None:
        return _build_legacy_context(settings)

    if settings.tenant_code and resolved_code and resolved_code != settings.tenant_code:
        env_lookup = await _lookup_tenant_by_code(db, settings.tenant_code)
        if env_lookup and env_lookup[0] != resolved_id:
            logger.error(
                "%sevent=tenant_mismatch|header_tenant_id=%s|header_code=%s|"
                "env_tenant_code=%s|env_tenant_id=%s",
                _LOG_PREFIX,
                resolved_id,
                resolved_code,
                settings.tenant_code,
                env_lookup[0],
            )

    cached = _cache_get(resolved_id)
    if cached is not None:
        logger.debug("%sevent=cache_hit|tenant_id=%s", _LOG_PREFIX, resolved_id)
        return cached

    entitlements = await _load_entitlements(db, resolved_id)
    adapter_rows = await _load_tenant_adapters(db, resolved_id)

    adapters: dict[str, AdapterRuntimeConfig] = {}
    for adapter_id, base_url, auth_scheme in adapter_rows:
        adapters[adapter_id] = await _build_adapter_config(
            db, resolved_id, adapter_id, base_url, auth_scheme, settings
        )

    if not adapters:
        logger.warning(
            "%sevent=no_tenant_adapters|tenant_id=%s|using_default_adapters_with_env_fallback",
            _LOG_PREFIX,
            resolved_id,
        )
        for asset_class in AssetClass:
            adapter_id = DEFAULT_ADAPTER_BY_ASSET_CLASS[asset_class]
            adapters[adapter_id] = await _build_adapter_config(
                db, resolved_id, adapter_id, "", "", settings
            )

    ctx = TenantContext(
        tenant_id=resolved_id,
        tenant_code=resolved_code,
        enabled_asset_classes=entitlements,
        adapters=adapters,
        is_legacy=False,
    )
    _cache_set(resolved_id, ctx)
    logger.info("%sevent=loaded|%s", _LOG_PREFIX, ctx.log_summary())
    return ctx
