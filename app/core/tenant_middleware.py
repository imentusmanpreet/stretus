"""
app/core/tenant_middleware.py
Attach ``TenantContext`` to each API request via contextvars.

Skipped paths: /health*, /docs, /openapi.json, /redoc, /
Internal callers (strategy / order-execution) should send ``x-tenant-id``.
"""
from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import get_settings
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.db.session import AsyncSessionLocal
from app.services.execution.tenant_config_service import load_tenant_context

logger = logging.getLogger(__name__)

_LOG_PREFIX = "tenant_config|"

_SKIP_PREFIXES = (
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
)


def _should_skip(path: str) -> bool:
    if path == "/":
        return True
    return any(path.startswith(prefix) for prefix in _SKIP_PREFIXES)


class TenantContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if _should_skip(request.url.path):
            return await call_next(request)

        tenant_header = request.headers.get("x-tenant-id")
        token = None
        try:
            async with AsyncSessionLocal() as db:
                ctx = await load_tenant_context(
                    db,
                    tenant_id_header=tenant_header,
                    settings=get_settings(),
                )
            token = set_current_tenant(ctx)
            logger.info(
                "%sevent=request|method=%s|path=%s|tenant_id=%s|legacy=%s",
                _LOG_PREFIX,
                request.method,
                request.url.path,
                ctx.tenant_id,
                ctx.is_legacy,
            )
            return await call_next(request)
        finally:
            if token is not None:
                reset_current_tenant(token)
