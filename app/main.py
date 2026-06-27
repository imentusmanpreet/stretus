"""
app/main.py
────────────
Stretus FastAPI application factory.

Startup loads the structured KB (app/kb) into memory. No ChromaDB / embeddings.
"""
from __future__ import annotations

# quant_engine/ has 26 internal cross-imports of the form `from engine.*`.
# They resolve only when `quant_engine/` is itself on sys.path. tests/conftest.py
# does this for pytest; production Uvicorn startup needs the same injection or
# any code path touching the discovery scanner / backtest engine fails with
# `ModuleNotFoundError: No module named 'engine'`.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPO_ROOT, _REPO_ROOT / "quant_engine"):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging_setup import configure_logging
from app.core.tenant_middleware import TenantContextMiddleware
from app.core.errors import install_exception_handlers
from app.api.v1.routes import chat, strategy, backtest, execution
from app.db.session import AsyncSessionLocal
from app.kb import kb
from app.services.execution.risk_execution_config_service import (
    ensure_default_risk_execution_config,
)

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)


async def _seed_risk_execution_if_needed() -> None:
    try:
        async with AsyncSessionLocal() as db:
            snapshot = await ensure_default_risk_execution_config(db)
            await db.commit()
            logger.info(
                "🛡️ Risk/execution config ready | scope=%s scope_id=%s updated_at=%s",
                snapshot.config_scope,
                snapshot.scope_id,
                snapshot.updated_at,
            )
    except Exception as exc:
        logger.warning("⚠️ Risk/execution config seed skipped | error=%s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.strategy_folder, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    # LLM info
    from app.services.ai.llm import LLMService
    llm = LLMService()
    info = llm.info()
    logger.info(
        "🚀 Stretus API ready | provider=%s model=%s tenant_code=%s",
        info["provider"].upper(),
        info["active_model"],
        settings.tenant_code or "(multi-tenant — use x-tenant-id)",
    )
    logger.info(
        "tenant_config|event=startup|tenant_code=%s|market_data_grpc=%s",
        settings.tenant_code or "",
        settings.market_data_grpc_target or "(not set)",
    )
    logger.info("  🚀 🚀  LLM details | %s", {k: v for k, v in info.items() if k != "provider"})
    logger.info("the info is %s", info)
    logger.info("Test LOG")
    if info["provider"] == "ollama":
        logger.info("🔌 Ollama endpoint configured | url=%s", info["ollama_url"])

    logger.info(
        "🧠 KB ready | signals=%d stocks=%d timeframes=%d",
        len(kb.signals), len(kb.stocks), len(kb.timeframes.supported),
    )

    from app.services.backtest.market_data_grpc import use_grpc_transport

    if use_grpc_transport(settings):
        logger.info(
            "📡 Backtest market data | transport=grpc target=%s timeout=%ss secure=%s",
            settings.market_data_grpc_target,
            settings.market_data_grpc_timeout_seconds,
            settings.market_data_grpc_secure,
        )
    else:
        logger.info(
            "📡 Backtest market data | transport=http url=%s timeout=%ss",
            settings.historical_data_url or "(not set)",
            settings.historical_data_timeout_seconds,
        )

    await _seed_risk_execution_if_needed()

    yield
    
    logger.info("👋 Stretus API shutting down")


app = FastAPI(
    title="Stretus Trading API",
    description="""
## Stretus — AI-Powered Trading Strategy Builder

### LLM Providers Supported
- **Groq Cloud** — fast, free tier, set `LLM_PROVIDER=groq` and `GROQ_API_KEY`
- **Ollama Local** — private, no API key, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL`
- **Auto** — tries Groq first, falls back to Ollama, set `LLM_PROVIDER=auto`

### Workflow
1. `POST /api/v1/strategy/chats` — start a new chat session
2. `POST /api/v1/strategy/chats/{chat_id}/messages` — chat with AI (collect requirements)
3. `POST /api/v1/strategy/strategies` — confirm strategy + generate YAML
4. `GET  /api/v1/strategy/strategies/{strategy_id}` — get full strategy_object
5. `POST /api/v1/strategy/backtest` — run backtest
""",
    version="1.0.0",
    debug=settings.app_debug,
    lifespan=lifespan,
)
install_exception_handlers(app)

app.add_middleware(TenantContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_debug else ["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,      prefix="/api/v1/strategy", tags=["💬 Chat"])
app.include_router(strategy.router,  prefix="/api/v1/strategy", tags=["📋 Strategy"])
app.include_router(backtest.router,  prefix="/api/v1/strategy", tags=["📊 Backtest"])
app.include_router(execution.router, prefix="/api/v1/strategy", tags=["⚡ Execution"])


@app.get("/health", tags=["🔧 Health"])
async def health():
    return {"status": "ok", "service": "stretus", "version": "1.0.0"}


@app.get("/health/kb", tags=["🔧 Health"])
async def kb_health():
    """Summary of the in-memory knowledge base."""
    return {
        "status":     "ok",
        "signals":    len(kb.signals),
        "stocks":     [s.symbol for s in kb.stocks.values() if s.enabled],
        "timeframes": kb.timeframes.supported,
    }


@app.post("/api/v1/admin/kb/reload", tags=["🔧 Admin"])
async def admin_kb_reload():
    """Hot-reload all KB files (signals, stocks, timeframes, etc.) without
    restarting the process. Use after editing app/kb/*.yaml."""
    kb.reload()
    from app.core.supported_stocks import reload_supported_stocks
    reload_supported_stocks()
    logger.info(
        "🔄 KB reloaded | signals=%d stocks=%d",
        len(kb.signals), len(kb.stocks),
    )
    return {
        "status":  "reloaded",
        "signals": len(kb.signals),
        "stocks":  len(kb.stocks),
    }


@app.get("/health/llm", tags=["🔧 Health"])
async def llm_health():
    from app.services.ai.llm import LLMService
    llm    = LLMService()
    status = await llm.health_check()
    return {"config": llm.info(), "status": status}


@app.get("/health/llm/models", tags=["🔧 Health"])
async def list_models():
    from app.services.ai.llm import OLLAMA_MODELS, LLMService
    import httpx
    pulled = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                pulled = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return {
        "ollama_models": [
            {"model": m, "description": d, "pulled": any(m in p for p in pulled),
             "pull_command": f"ollama pull {m}"}
            for m, d in OLLAMA_MODELS.items()
        ],
        "ollama_pulled": pulled,
    }


@app.get("/", tags=["🔧 Health"])
async def root():
    return {
        "message": "Stretus Trading API",
        "docs":    "/docs",
        "health":  "/health",
        "kb":      "/health/kb",
        "llm":     "/health/llm",
    }
