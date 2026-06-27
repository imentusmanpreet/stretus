"""
Backtesting endpoints.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import yaml
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
from app.core.timing import BacktestTimer
from app.services.backtest import (
    INTRABAR_EXECUTION_INTERVAL,
    apply_sanitized_result_to_row,
    build_main_fetch_request,
    extract_strategy_market_data_request,
    fetch_auxiliary_ohlcv,
    fetch_ohlcv_records,
    normalize_backtest_metric_aliases,
    queue_quant_backtest,
    resolve_intrabar_execution,
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
                "💥 backtest_persist|source=orch_failure|outcome=db_update_failed|backtest_id=%s",
                backtest_id,
            )


def _extract_universe_block(yaml_content: str) -> dict | None:
    """Return the top-level ``universe:`` block from a strategy YAML, else ``None``.

    Plain PyYAML read — the app process detects dynamic mode without importing the engine.
    Tolerant: malformed/universe-free YAML returns ``None`` (static path unperturbed).
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("universe")
    return block if isinstance(block, dict) else None


def _backtest_market_fetcher():
    """Async ``(symbol, interval, from_iso, to_iso) -> records`` over the backtest feed.

    Mirrors ``scanner._default_fetch_ohlcv`` so the dynamic resolver + Tier-B loader reuse the
    same concurrent/chunked/cached market-data path as the rest of the backtest service.
    """
    async def fetch(symbol: str, interval: str, from_utc: str, to_utc: str):
        from app.services.backtest.market_data import (
            StrategyMarketDataRequest,
            fetch_ohlcv_records,
        )
        request = StrategyMarketDataRequest(
            yaml_path="", raw_symbol=symbol,
            symbol=symbol.replace(".NS", "").replace(".BO", ""),
            interval=interval, from_utc=from_utc, to_utc=to_utc,
        )
        return await fetch_ohlcv_records(request)

    return fetch


async def _run_dynamic_backtest_and_persist(
    *,
    backtest_id: str,
    strategy_id: str,
    yaml_content: str,
    universe_block: dict,
    market_data_request,
    run_request: BacktestTriggerRequest,
    timer: BacktestTimer,
) -> None:
    """Resolve the universe, load member data, run the portfolio backtest, persist the result.

    Reuses the existing result-persistence helpers (``summarize_backtest_for_db`` +
    ``apply_sanitized_result_to_row``) so the dynamic result lands on the same Backtest row
    shape (additive — Invariant 10). Failures mark the row failed with the root cause.
    """
    from app.core.config import get_settings
    from app.services.backtest.result_store import (
        apply_sanitized_result_to_row,
        summarize_backtest_for_db,
    )
    from app.services.universe.backtest_orchestrator import run_dynamic_backtest
    from app.services.universe.membership import MembershipPoolProvider, SqlMembershipStore
    from app.strategy.spec import UniverseSpec

    try:
        # Keep only real UniverseSpec fields — a YAML universe block from an older build may
        # carry a display-only `summary`, which extra="forbid" would reject.
        universe_block = {k: v for k, v in (universe_block or {}).items()
                          if k in set(UniverseSpec.model_fields)}
        universe = UniverseSpec.model_validate(universe_block)
        settings = get_settings()
        window_from = _parse_iso(market_data_request.from_utc)
        window_to = _parse_iso(market_data_request.to_utc)
        # index/sector/f_and_o resolve point-in-time from the membership store; watchlist/
        # crypto_all need no provider (the orchestrator resolves them directly).
        pool_provider = None
        if universe.source.kind in ("index", "sector", "f_and_o"):
            pool_provider = MembershipPoolProvider(SqlMembershipStore(AsyncSessionLocal))

        mdr = {
            "symbol": market_data_request.symbol, "interval": market_data_request.interval,
            "from_utc": market_data_request.from_utc, "to_utc": market_data_request.to_utc,
        }
        with timer.step("dynamic_universe_portfolio_backtest", {"source": universe.source.kind}):
            result = await run_dynamic_backtest(
                universe=universe, template_yaml=yaml_content,
                window_from=window_from, window_to=window_to,
                timeframe=market_data_request.interval,
                fetch_ohlcv=_backtest_market_fetcher(), market_data_request=mdr,
                run_config={"starting_balance": float(getattr(run_request, "starting_balance", 100000.0) or 100000.0)},
                pool_provider=pool_provider, settings=settings,
                warmup_bars=settings.dynamic_universe_warmup_bars,
                backtest_ref_id=backtest_id, run_id=backtest_id,
            )

        async with AsyncSessionLocal() as db:
            backtest = await db.get(Backtest, uuid.UUID(backtest_id))
            if backtest:
                strategy = await db.get(Strategy, backtest.strategy_id)
                sanitized = summarize_backtest_for_db(result) or {}
                apply_sanitized_result_to_row(backtest=backtest, strategy=strategy, sanitized=sanitized)
                await db.commit()
        logger.info(
            "✅ backtest_orch|stage=dynamic_complete|backtest_id=%s|members=%s|survivorship=%s",
            backtest_id, len(result.get("members") or []), result.get("survivorship_mode"),
        )
    except Exception as exc:  # noqa: BLE001 — surface the root cause on the row
        logger.exception("💥 backtest_orch|stage=dynamic_failed|backtest_id=%s", backtest_id)
        await _mark_backtest_failed(backtest_id, exc)


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def _call_quant_engine(backtest_id: str, strategy_id: str, yaml_path: str, run_request: BacktestTriggerRequest) -> None:
    timer = BacktestTimer(backtest_id)
    
    try:
        logger.info(
            "🚀 backtest_orch|stage=prep|backtest_id=%s|strategy_id=%s",
            backtest_id,
            strategy_id,
        )
        
        # Step 1: Extract market data request from strategy YAML
        with timer.step("extract_market_data_request", {"strategy_id": strategy_id}):
            market_data_request = extract_strategy_market_data_request(yaml_path, overrides=run_request)

        # Dynamic-universe branch: a strategy that names a SELECTION RULE carries a
        # top-level ``universe:`` block. It runs the portfolio path (resolve → Tier-B load →
        # /run-portfolio-sync) instead of the single-symbol flow. Detection is a plain YAML
        # read (the app process needn't import the engine). Static specs fall straight through
        # unchanged (Invariant 2).
        yaml_content = Path(yaml_path).read_text(encoding="utf-8")
        universe_block = _extract_universe_block(yaml_content)
        if universe_block is not None:
            await _run_dynamic_backtest_and_persist(
                backtest_id=backtest_id, strategy_id=strategy_id,
                yaml_content=yaml_content, universe_block=universe_block,
                market_data_request=market_data_request, run_request=run_request, timer=timer,
            )
            return

        # Phase 11: 1-minute execution. AUTO-on for any non-1m strategy (the
        # request flag can force it on/off). When on, the main (and reference)
        # series is fetched at 1m; the engine resamples to the strategy timeframe
        # for signals and walks the minute bars for fills/SL/TP. The resolved
        # concrete decision is written back onto run_request so the engine's
        # run_config receives a definite bool (never the AUTO sentinel).
        intrabar_execution = resolve_intrabar_execution(
            run_request, signal_interval=market_data_request.interval,
        )
        run_request = run_request.model_copy(update={"intrabar_execution": intrabar_execution})
        main_fetch_request = build_main_fetch_request(
            market_data_request, intrabar_execution=intrabar_execution,
        )

        # Step 2: Fetch main OHLCV data (chunked)
        with timer.step("fetch_main_ohlcv", {
            "symbol": main_fetch_request.symbol,
            "interval": main_fetch_request.interval,
            "from_utc": main_fetch_request.from_utc,
            "to_utc": main_fetch_request.to_utc,
            "intrabar_execution": intrabar_execution,
        }):
            ohlcv_data = await fetch_ohlcv_records(main_fetch_request)

        # Step 3: Fetch auxiliary OHLCV (reference + HTF) in parallel
        with timer.step("fetch_auxiliary_ohlcv", {
            "symbol": market_data_request.symbol,
        }):
            reference_ohlcv, htf_ohlcv = await fetch_auxiliary_ohlcv(
                market_data_request,
                main_fetch_interval=INTRABAR_EXECUTION_INTERVAL if intrabar_execution else None,
            )
        
        # Step 4: Queue quant engine execution
        with timer.step("queue_quant_engine", {
            "candles": len(ohlcv_data),
            "reference_candles": len(reference_ohlcv) if reference_ohlcv else 0,
            "htf_timeframes": list(htf_ohlcv.keys()) if htf_ohlcv else [],
        }):
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
        
        # Generate and log timing summary
        summary = timer.summary()
        
        # Log additional visible summary
        logger.info("=" * 100)
        logger.info("🎯 API LAYER TIMING SUMMARY - Backtest ID: %s", backtest_id)
        logger.info("=" * 100)
        logger.info("⏱️  Total Duration: %s", summary["overall_duration_formatted"])
        logger.info("🔢 Steps Completed: %d", summary["step_count"])
        logger.info("-" * 100)
        for step in summary["steps"]:
            percentage = (step["duration_seconds"] / summary["overall_duration_seconds"] * 100) if summary["overall_duration_seconds"] > 0 else 0
            logger.info("  ⏳ %-32s : %12s (%6.2f%%)", step["step"], step["duration_formatted"], percentage)
        logger.info("=" * 100)

        logger.info(
            "✅ backtest_orch|stage=quant_queued|backtest_id=%s|symbol=%s|interval=%s|candles=%s"
            "|aux_reference=%s|aux_htf=%s|total_duration=%s",
            backtest_id,
            market_data_request.symbol,
            market_data_request.interval,
            len(ohlcv_data),
            (len(reference_ohlcv) if reference_ohlcv is not None else 0),
            ({tf: len(rows) for tf, rows in htf_ohlcv.items()} if htf_ohlcv else {}),
            summary["overall_duration_formatted"],
        )
    except Exception as exc:
        # Generate summary even on failure
        try:
            summary = timer.summary()
            logger.info("=" * 100)
            logger.info("❌ API LAYER TIMING SUMMARY (FAILED) - Backtest ID: %s", backtest_id)
            logger.info("=" * 100)
            logger.info("⏱️  Total Duration: %s", summary["overall_duration_formatted"])
            logger.info("🔢 Steps Completed: %d", summary["step_count"])
            logger.info("-" * 100)
            for step in summary["steps"]:
                percentage = (step["duration_seconds"] / summary["overall_duration_seconds"] * 100) if summary["overall_duration_seconds"] > 0 else 0
                status_icon = "✅" if step["status"] == "COMPLETE" else "❌"
                logger.info("  %s %-33s : %12s (%6.2f%%)", status_icon, step["step"], step["duration_formatted"], percentage)
            logger.info("=" * 100)
        except Exception:
            pass  # Don't let summary generation hide the real error
        
        logger.exception(
            "💥 backtest_orch|stage=failed|backtest_id=%s|error_type=%s",
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

    # Multi-asset note: this endpoint is the asynchronous single-asset path (the
    # quant engine posts each result back independently via PUT .../result, so
    # there is no point to assemble the cross-asset wrapper + summary here).
    # Multi-asset runs go through the chat flow, which runs assets sequentially
    # and returns the MultiAssetBacktestResult wrapper. A single symbol is
    # accepted as a convenience override.
    if body.symbols:
        _syms = [s.strip() for s in body.symbols if s and s.strip()]
        if len(_syms) > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multi-asset backtests are run through the chat flow "
                    "(send 'symbols' with the message). This endpoint supports "
                    "a single asset per request."
                ),
            )
        if _syms and not body.symbol:
            body = body.model_copy(update={"symbol": _syms[0]})

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
        "🆕 backtest_persist|source=trigger|outcome=queued|backtest_id=%s|strategy_id=%s|"
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
        "📥 backtest_persist|source=http_callback|stage=recv|backtest_id=%s|payload_keys=%s",
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
            "💾 backtest_persist|source=http_callback|outcome=db_committed|backtest_id=%s|row_status=%s|"
            "strategy_id=%s",
            backtest_id,
            backtest.status.value,
            backtest.strategy_id,
        )
    except Exception:
        logger.exception(
            "❌ backtest_persist|source=http_callback|outcome=db_commit_failed|backtest_id=%s",
            backtest_id,
        )
        raise

    m = stored.get("metrics", {}) if isinstance(stored, dict) else {}
    total_trades = m.get("total_trades") if isinstance(m, dict) else None
    logger.info(
        "📊 backtest_persist|source=http_callback|metrics|backtest_id=%s|total_trades=%s|engine_ref=%s",
        backtest_id,
        total_trades,
        stored.get("backtest_ref_id") if isinstance(stored, dict) else None,
    )
    return {"message": "Result stored."}
