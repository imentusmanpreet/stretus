"""
Backtesting endpoints.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import build_error_object
from app.db.models.strategy import Backtest, BacktestStatus, Strategy, StrategyStatus
from app.db.session import AsyncSessionLocal, get_db
from app.schemas.backtest import (
    BacktestResultResponse,
    BacktestTriggerRequest,
    BacktestTriggerResponse,
)
from app.services.backtest import (
    apply_sanitized_result_to_row,
    extract_strategy_market_data_request,
    fetch_auxiliary_ohlcv,
    fetch_ohlcv_records,
    normalize_backtest_metric_aliases,
    queue_quant_backtest,
    summarize_backtest_for_db,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_backtest_result_payload(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return result

    sanitized = dict(result)
    sanitized.pop("failure", None)
    normalize_backtest_metric_aliases(sanitized)
    return sanitized


async def _mark_backtest_failed(backtest_id: str, exc: Exception) -> None:
    async with AsyncSessionLocal() as db:
        try:
            backtest = await db.get(Backtest, uuid.UUID(backtest_id))
            if not backtest:
                return
            backtest.status = BacktestStatus.failed
            backtest.error_message = str(exc)
            backtest.completed_at = _utcnow()
            strategy = await db.get(Strategy, backtest.strategy_id)
            if strategy:
                strategy.status = StrategyStatus.confirmed
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception(
                "backtest_persist|source=orch_failure|outcome=db_update_failed|backtest_id=%s",
                backtest_id,
            )


async def _call_quant_engine(backtest_id: str, strategy_id: str, yaml_path: str, run_request: BacktestTriggerRequest) -> None:
    try:
        logger.info(
            "backtest_orch|stage=prep|backtest_id=%s|strategy_id=%s",
            backtest_id,
            strategy_id,
        )
        market_data_request = extract_strategy_market_data_request(yaml_path, overrides=run_request)
        ohlcv_data = await fetch_ohlcv_records(market_data_request)
        # Phase 7 — when the strategy declares reference_symbol or htf rules,
        # fetch those series in parallel and forward to the engine. Both are
        # None when the strategy uses neither, so legacy strategies see no
        # behavior change.
        reference_ohlcv, htf_ohlcv = await fetch_auxiliary_ohlcv(market_data_request)
        await queue_quant_backtest(
            backtest_id=backtest_id,
            strategy_id=strategy_id,
            yaml_path=yaml_path,
            ohlcv_data=ohlcv_data,
            run_config=run_request,
            market_data_request={
                "symbol": market_data_request.symbol,
                "interval": market_data_request.interval,
                "from_utc": market_data_request.from_utc,
                "to_utc": market_data_request.to_utc,
            },
            reference_ohlcv=reference_ohlcv,
            htf_ohlcv=htf_ohlcv,
        )
        logger.info(
            "backtest_orch|stage=quant_queued|backtest_id=%s|symbol=%s|interval=%s|candles=%s"
            "|aux_reference=%s|aux_htf=%s",
            backtest_id,
            market_data_request.symbol,
            market_data_request.interval,
            len(ohlcv_data),
            (len(reference_ohlcv) if reference_ohlcv is not None else 0),
            ({tf: len(rows) for tf, rows in htf_ohlcv.items()} if htf_ohlcv else {}),
        )
    except Exception as exc:
        logger.exception(
            "backtest_orch|stage=failed|backtest_id=%s|error_type=%s",
            backtest_id,
            type(exc).__name__,
        )
        await _mark_backtest_failed(backtest_id, exc)


@router.post(
    "/backtest",
    response_model=BacktestTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_backtest(
    body: BacktestTriggerRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        strategy_uuid = uuid.UUID(body.strategy_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="strategy_id must be a valid UUID.") from exc

    strategy = await db.get(Strategy, strategy_uuid)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found.")

    if strategy.status not in (StrategyStatus.confirmed, StrategyStatus.backtest_complete):
        raise HTTPException(
            status_code=400,
            detail=f"Strategy must be in 'confirmed' state. Current: {strategy.status.value}",
        )
    if not strategy.yaml_path:
        raise HTTPException(status_code=400, detail="Strategy has no YAML file. Confirm first.")

    backtest = Backtest(
        id=uuid.uuid4(),
        strategy_id=strategy.id,
        status=BacktestStatus.running,
        started_at=_utcnow(),
    )
    db.add(backtest)

    strategy.status = StrategyStatus.backtesting
    await db.commit()

    background_tasks.add_task(
        _call_quant_engine,
        str(backtest.id),
        str(strategy.id),
        strategy.yaml_path,
        body,
    )

    logger.info(
        "backtest_persist|source=trigger|outcome=queued|backtest_id=%s|strategy_id=%s|"
        "symbol=%s|timeframe=%s|db=committed",
        backtest.id,
        strategy.id,
        strategy.symbol,
        strategy.timeframe,
    )
    return BacktestTriggerResponse(
        backtest_id=str(backtest.id),
        strategy_id=str(strategy.id),
        status="running",
        message="Backtest queued. Poll GET /backtest/{backtest_id} for results.",
    )


@router.get("/backtest/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest_result(
    backtest_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        backtest_uuid = uuid.UUID(backtest_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="backtest_id must be a valid UUID.") from exc

    backtest = await db.get(Backtest, backtest_uuid)
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found.")

    error_payload = None
    if backtest.status == BacktestStatus.failed and backtest.error_message:
        error_payload = build_error_object(500, backtest.error_message)

    return BacktestResultResponse(
        backtest_id=str(backtest.id),
        strategy_id=str(backtest.strategy_id),
        status=backtest.status.value,
        backtest_result=_sanitize_backtest_result_payload(backtest.result_json) or None,
        error_message=(
            error_payload.get("message")
            if isinstance(error_payload, dict)
            else backtest.error_message
        ),
        error=error_payload,
        created_at=backtest.created_at.isoformat(),
        completed_at=backtest.completed_at.isoformat() if backtest.completed_at else None,
    )


@router.put("/backtest/{backtest_id}/result", status_code=status.HTTP_200_OK)
async def receive_backtest_result(
    backtest_id: str,
    result: dict,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        backtest_uuid = uuid.UUID(backtest_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="backtest_id must be a valid UUID.") from exc

    backtest = await db.get(Backtest, backtest_uuid)
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found.")

    logger.info(
        "backtest_persist|source=http_callback|stage=recv|backtest_id=%s|payload_keys=%s",
        backtest_id,
        list(result.keys()) if isinstance(result, dict) else "non-dict",
    )

    sanitized_result = _sanitize_backtest_result_payload(result)
    strategy = await db.get(Strategy, backtest.strategy_id)

    if not isinstance(sanitized_result, dict) or not sanitized_result:
        backtest.status = BacktestStatus.failed
        backtest.error_message = "Backtest result payload was empty or invalid."
        backtest.result_json = None
        backtest.completed_at = _utcnow()
        if strategy:
            strategy.status = StrategyStatus.confirmed
        stored: dict | None = None
    else:
        stored = summarize_backtest_for_db(sanitized_result) or {}
        apply_sanitized_result_to_row(backtest=backtest, strategy=strategy, sanitized=stored)

    # Commit here so a successful response means the row is durable (get_db will commit again as no-op)
    try:
        await db.commit()
        logger.info(
            "backtest_persist|source=http_callback|outcome=db_committed|backtest_id=%s|row_status=%s|"
            "strategy_id=%s",
            backtest_id,
            backtest.status.value,
            backtest.strategy_id,
        )
    except Exception:
        logger.exception(
            "backtest_persist|source=http_callback|outcome=db_commit_failed|backtest_id=%s",
            backtest_id,
        )
        raise

    m = stored.get("metrics", {}) if isinstance(stored, dict) else {}
    total_trades = m.get("total_trades") if isinstance(m, dict) else None
    logger.info(
        "backtest_persist|source=http_callback|metrics|backtest_id=%s|total_trades=%s|engine_ref=%s",
        backtest_id,
        total_trades,
        stored.get("backtest_ref_id") if isinstance(stored, dict) else None,
    )
    return {"message": "Result stored."}
