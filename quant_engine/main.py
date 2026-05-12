"""
Stretus Quant Engine.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from logging_setup import configure_logging
from engine.metrics import build_execution_error_result
from engine.runner import run_backtest

configure_logging()
logger = logging.getLogger(__name__)

_running: dict[str, dict[str, str]] = {}
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
_DEFAULT_ERROR_MESSAGES = {
    400: "Bad request. Please review the submitted data and try again.",
    404: "The requested resource was not found.",
    422: "Validation failed. Please review the submitted data and try again.",
    429: (
        "Rate limit exceeded. You have reached your API usage limit. "
        "Please retry after some time."
    ),
    500: "Something went wrong on our side. Please retry after some time.",
}


def _error_content(status_code: int, message: str | None = None) -> dict[str, dict[str, Any]]:
    resolved_message = " ".join(str(message or "").split()).strip()
    if not resolved_message:
        resolved_message = _DEFAULT_ERROR_MESSAGES.get(status_code, _DEFAULT_ERROR_MESSAGES[500])
    return {"error": {"code": int(status_code), "message": resolved_message}}


def _detail_message(detail: Any) -> str | None:
    if isinstance(detail, str):
        text = " ".join(detail.split()).strip()
        return text or None
    return None


async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_content(exc.status_code, _detail_message(exc.detail)),
    )


async def _validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    issues: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        message = _detail_message(error.get("msg")) or "Invalid value."
        issues.append(f"{location}: {message}" if location else message)

    return JSONResponse(
        status_code=422,
        content=_error_content(422, f"Validation failed. {'; '.join(issues)}"),
    )


async def _unexpected_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=400,
            content=_error_content(400, str(exc)),
        )

    logger.exception("Unhandled quant engine exception")
    return JSONResponse(status_code=500, content=_error_content(500))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Quant engine ready | fastapi_callback_url=%s", FASTAPI_URL)
    yield
    logger.info("👋 Quant engine shutting down")


app = FastAPI(
    title="Stretus Quant Engine",
    description="Backtest calculator that consumes strategy YAML plus OHLCV payloads.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_exception_handler(HTTPException, _http_exception_handler)
app.add_exception_handler(RequestValidationError, _validation_exception_handler)
app.add_exception_handler(Exception, _unexpected_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class RunConfig(BaseModel):
    starting_balance: float = 10000.0
    slippage_bps: float = 5.0
    commission_bps: float = 2.0
    max_holding_candles: Optional[int] = None
    # Strategy objective and risk controls
    objective: str = "positional"           # 'intraday' | 'positional'
    daily_loss_cap_pct: float = 0.0         # 0 = disabled; e.g. 2.0 = stop trading if portfolio drops 2% today
    max_trades_per_day: int = 0             # 0 = unlimited; enforced for intraday only
    # Indian market: Securities Transaction Tax (NSE/BSE regulatory cost)
    # Intraday equity: 0% on buy, 0.025% on sell side
    # Delivery equity: 0.1% on both buy and sell
    stt_intraday_sell_pct: float = 0.025
    stt_delivery_pct: float = 0.1


class MarketDataRequestPayload(BaseModel):
    symbol: str
    interval: str
    from_utc: str
    to_utc: str


class RunRequest(BaseModel):
    backtest_id: str
    strategy_id: str
    yaml_content: str
    ohlcv_data: list[dict[str, Any]] | list[list[Any]]
    run_config: RunConfig = Field(default_factory=RunConfig)
    market_data_request: MarketDataRequestPayload
    # Phase 7 — caller-supplied auxiliary OHLCV. Both optional; the engine
    # validates them against cfg.reference_symbol / cfg.htf_rules and raises
    # a clear error if a strategy declares one but the corresponding payload
    # is missing.
    reference_ohlcv: Optional[list[dict[str, Any]] | list[list[Any]]] = None
    htf_ohlcv: Optional[dict[str, list[dict[str, Any]] | list[list[Any]]]] = None


class RunSyncRequest(BaseModel):
    yaml_content: str
    ohlcv_data: list[dict[str, Any]] | list[list[Any]]
    run_config: RunConfig = Field(default_factory=RunConfig)
    market_data_request: MarketDataRequestPayload
    backtest_ref_id: Optional[str] = None
    reference_ohlcv: Optional[list[dict[str, Any]] | list[list[Any]]] = None
    htf_ohlcv: Optional[dict[str, list[dict[str, Any]] | list[list[Any]]]] = None


class RunResponse(BaseModel):
    backtest_id: str
    status: str
    message: str


@app.post("/run", response_model=RunResponse)
async def start_backtest(
    body: RunRequest,
    background_tasks: BackgroundTasks,
):
    if not body.ohlcv_data:
        raise HTTPException(status_code=400, detail="ohlcv_data is required.")

    logger.info(
        "📥 Async backtest queued | backtest_id=%s strategy_id=%s symbol=%s interval=%s rows=%s",
        body.backtest_id,
        body.strategy_id,
        body.market_data_request.symbol,
        body.market_data_request.interval,
        len(body.ohlcv_data),
    )

    _running[body.backtest_id] = {
        "status": "running",
        "started_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }

    background_tasks.add_task(
        _execute_and_callback,
        body.backtest_id,
        body.yaml_content,
        body.ohlcv_data,
        body.run_config.model_dump(),
        body.market_data_request.model_dump(),
        body.reference_ohlcv,
        body.htf_ohlcv,
    )

    return RunResponse(
        backtest_id=body.backtest_id,
        status="running",
        message="Backtest started. Results will be posted back to FastAPI.",
    )


@app.post("/run-sync")
async def run_sync(
    body: RunSyncRequest,
):
    if not body.ohlcv_data:
        raise HTTPException(status_code=400, detail="ohlcv_data is required.")

    logger.info(
        "🧮 Sync backtest started | ref=%s symbol=%s interval=%s rows=%s",
        body.backtest_ref_id,
        body.market_data_request.symbol,
        body.market_data_request.interval,
        len(body.ohlcv_data),
    )

    try:
        loop = asyncio.get_running_loop()
        # Use a lambda so we can pass keyword args through run_in_executor —
        # run_backtest's reference_ohlcv / htf_ohlcv are kw-only optionals.
        result = await loop.run_in_executor(
            None,
            lambda: run_backtest(
                body.yaml_content,
                body.ohlcv_data,
                body.run_config.model_dump(),
                body.market_data_request.model_dump(),
                body.backtest_ref_id,
                reference_ohlcv=body.reference_ohlcv,
                htf_ohlcv=body.htf_ohlcv,
            ),
        )
        return result
    except Exception as exc:
        logger.exception("❌ Sync backtest failed | ref=%s", body.backtest_ref_id)
        return build_execution_error_result(
            backtest_ref_id=body.backtest_ref_id or "sync-backtest-error",
            reason=str(exc),
            starting_balance=body.run_config.starting_balance,
            start_utc=body.market_data_request.from_utc,
            end_utc=body.market_data_request.to_utc,
            config={
                **body.market_data_request.model_dump(),
                **body.run_config.model_dump(),
            },
        )


async def _execute_and_callback(
    backtest_id: str,
    yaml_content: str,
    ohlcv_data: list[dict[str, Any]] | list[list[Any]],
    run_config: dict[str, Any],
    market_data_request: dict[str, Any],
    reference_ohlcv: list[dict[str, Any]] | list[list[Any]] | None = None,
    htf_ohlcv: dict[str, Any] | None = None,
) -> None:
    try:
        logger.info("⚙️ Executing async backtest | backtest_id=%s", backtest_id)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_backtest(
                yaml_content,
                ohlcv_data,
                run_config,
                market_data_request,
                backtest_id,
                reference_ohlcv=reference_ohlcv,
                htf_ohlcv=htf_ohlcv,
            ),
        )
        _running[backtest_id] = {
            "status": "completed",
            "started_at": _running.get(backtest_id, {}).get("started_at", _utc_now_iso()),
            "updated_at": _utc_now_iso(),
        }
        await _post_result(backtest_id, result)
    except Exception as exc:
        logger.exception("❌ Async backtest failed | backtest_id=%s", backtest_id)
        _running[backtest_id] = {
            "status": "failed",
            "started_at": _running.get(backtest_id, {}).get("started_at", _utc_now_iso()),
            "updated_at": _utc_now_iso(),
        }
        error_result = build_execution_error_result(
            backtest_ref_id=backtest_id,
            reason=str(exc),
            starting_balance=float(run_config.get("starting_balance", 10000.0)),
            start_utc=str(market_data_request.get("from_utc")),
            end_utc=str(market_data_request.get("to_utc")),
            config={**market_data_request, **run_config},
        )
        await _post_result(backtest_id, error_result)


async def _post_result(backtest_id: str, result: dict) -> None:
    """
    POST result back to FastAPI with automatic retry.

    Retries up to 3 times with exponential backoff (2s, 4s) before giving up.
    If all attempts fail the result is lost — the backtest row stays 'running'
    in the DB until it is manually cleaned up or the job is re-queued.
    """
    url = f"{FASTAPI_URL}/api/v1/strategy/backtest/{backtest_id}/result"
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(url, json=result)
                response.raise_for_status()
            logger.info(
                "quant_engine|event=result_posted|backtest_id=%s|http_status=%s|attempt=%s/%s",
                backtest_id,
                response.status_code,
                attempt,
                max_attempts,
            )
            return
        except Exception as exc:
            if attempt < max_attempts:
                wait_seconds = 2 ** attempt   # 2s, then 4s
                logger.warning(
                    "⚠️ Result callback failed | backtest_id=%s attempt=%s/%s error=%s retry_in=%ss",
                    backtest_id,
                    attempt,
                    max_attempts,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
            else:
                logger.error(
                    "❌ Result callback exhausted retries | backtest_id=%s attempts=%s error=%s",
                    backtest_id,
                    max_attempts,
                    exc,
                )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "stretus-quant-engine",
        "version": "3.0.0",
        "fastapi_url": FASTAPI_URL,
    }


@app.get("/status/{backtest_id}")
async def get_status(backtest_id: str):
    return {
        "backtest_id": backtest_id,
        **_running.get(backtest_id, {"status": "unknown"}),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
