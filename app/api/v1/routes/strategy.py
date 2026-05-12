"""
app/api/v1/routes/strategy.py
──────────────────────────────
POST /strategies                → confirm strategy draft, generate YAML, save to DB
GET  /strategies/{strategy_id} → fetch a saved strategy with full strategy_object
GET  /strategies                → list all strategies for a chat
"""
from __future__ import annotations

import copy
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.strategy import Chat, ChatMessage, Strategy, StrategyStatus, ChatRole
from app.schemas.strategy import StrategyConfirmRequest, StrategyConfirmResponse
from app.services.execution.risk_execution_config_service import (
    SESSION_SCOPE,
    STRATEGY_SCOPE,
    RiskExecutionConfigSnapshot,
    build_risk_execution_response,
    compose_risk_execution_values,
    resolve_active_risk_execution_config,
    upsert_risk_execution_config,
)
from app.services.strategy.builder import StrategyBuilder
from app.services.strategy.yaml_generator import generate_yaml

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_risk_execution_values_for_builder(
    builder: "StrategyBuilder",
    base_config: RiskExecutionConfigSnapshot,
) -> dict[str, object]:
    return compose_risk_execution_values(
        max_trades=base_config.max_trades,
        daily_loss_cap=base_config.daily_loss_cap,
        execution_mode=base_config.execution_mode,
        per_trade_risk=base_config.per_trade_risk,
        trading_window=base_config.trading_window,
        position_sizing=base_config.position_sizing,
        risk_validation=base_config.risk_validation,
        stop_loss_pct=base_config.stop_loss_pct,
        take_profit_pct=base_config.take_profit_pct,
        minimum_trade_value=base_config.minimum_trade_value,
    )


async def _hydrate_builder_risk_execution_config(
    db: AsyncSession,
    builder: "StrategyBuilder",
    *,
    session_id: uuid.UUID,
    strategy_id: str | None = None,
) -> RiskExecutionConfigSnapshot:
    snapshot = await resolve_active_risk_execution_config(
        db,
        session_id=str(session_id),
        strategy_id=strategy_id,
    )
    builder.set_risk_execution_config(snapshot.to_builder_context())
    return snapshot


async def _persist_risk_execution_config(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    strategy_id: uuid.UUID,
    builder: "StrategyBuilder",
) -> RiskExecutionConfigSnapshot:
    active_config = await _hydrate_builder_risk_execution_config(
        db,
        builder,
        session_id=session_id,
        strategy_id=str(strategy_id),
    )
    values = _build_risk_execution_values_for_builder(builder, active_config)
    await upsert_risk_execution_config(
        db,
        config_scope=SESSION_SCOPE,
        scope_id=str(session_id),
        session_id=str(session_id),
        strategy_id=str(strategy_id),
        **values,
    )
    snapshot = await upsert_risk_execution_config(
        db,
        config_scope=STRATEGY_SCOPE,
        scope_id=str(strategy_id),
        session_id=str(session_id),
        strategy_id=str(strategy_id),
        **values,
    )
    builder.set_risk_execution_config(snapshot.to_builder_context())
    return snapshot


def _strategy_config_from_object(strategy_object: dict | None) -> dict | None:
    if not isinstance(strategy_object, dict):
        return None

    return {
        "name": strategy_object.get("name", "strategy"),
        "asset": strategy_object.get("asset"),
        "entry": copy.deepcopy(strategy_object.get("entry", [])),
        "exit": copy.deepcopy(strategy_object.get("exit", [])),
        "reward_factor": strategy_object.get("reward_factor"),
    }


def _extract_strategy_parts(payload: dict | None) -> tuple[dict | None, dict | None]:
    if not payload:
        return None, None

    context = payload.get("context") if isinstance(payload, dict) else None
    if isinstance(context, dict):
        strategy_config = context.get("strategy_config")
        strategy_object = context.get("strategy_object")
        if not isinstance(strategy_config, dict):
            strategy_config = _strategy_config_from_object(strategy_object)
        return (
            strategy_config if isinstance(strategy_config, dict) else None,
            strategy_object if isinstance(strategy_object, dict) else None,
        )

    if "strategy_object" in payload and isinstance(payload.get("strategy_object"), dict):
        strategy_object = payload["strategy_object"]
        return _strategy_config_from_object(strategy_object), strategy_object

    return None, payload if isinstance(payload, dict) else None


def _find_latest_strategy_message(messages: list[ChatMessage]) -> ChatMessage | None:
    for message in reversed(messages):
        strategy_config, strategy_object = _extract_strategy_parts(message.strategy_json)
        if strategy_config or strategy_object:
            return message
    return None


# ─────────────────────────────────────────────────────────────────────────────
# POST /strategies  — confirm draft, generate YAML, save
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/strategies",
    response_model=StrategyConfirmResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Confirm a strategy draft and generate YAML",
)
async def confirm_strategy(
    body: StrategyConfirmRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    # 1. Validate chat
    chat = await db.get(Chat, uuid.UUID(body.session_id))
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")

    # 2. Find the latest assistant message that still carries the assembled strategy payload
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == chat.id,
            ChatMessage.role == ChatRole.assistant,
            ChatMessage.strategy_json.isnot(None),
        )
        .order_by(ChatMessage.created_at.asc())
    )
    last_msg = _find_latest_strategy_message(list(result.scalars().all()))

    if not last_msg or not last_msg.strategy_json:
        raise HTTPException(
            status_code=400,
            detail=(
                "No assembled strategy found in this chat. "
                "Complete collect_user_input, plan_signals, and assemble_strategy first."
            ),
        )

    existing_strategy = None
    if last_msg.strategy_id:
        existing_strategy = await db.get(Strategy, last_msg.strategy_id)

    # 3. Rebuild builder & validate
    builder = StrategyBuilder()
    if last_msg.strategy_draft:
        builder.merge_preview(last_msg.strategy_draft)
    strategy_config, strategy_object = _extract_strategy_parts(last_msg.strategy_json)
    if strategy_config:
        builder.merge_preview(
            {
                "symbol": strategy_config.get("asset"),
                "entry_condition": (last_msg.strategy_draft or {}).get("entry_condition"),
                "exit_condition": (last_msg.strategy_draft or {}).get("exit_condition"),
            }
        )
    await _hydrate_builder_risk_execution_config(
        db,
        builder,
        session_id=chat.id,
        strategy_id=str(last_msg.strategy_id) if last_msg.strategy_id else None,
    )
    builder.apply_defaults()

    missing = builder.missing_fields()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy is incomplete. Missing fields: {', '.join(missing)}",
        )

    if existing_strategy:
        await _persist_risk_execution_config(
            db,
            session_id=chat.id,
            strategy_id=existing_strategy.id,
            builder=builder,
        )
        logger.info(
            "strategy_persist|source=http_confirm|outcome=unchanged_id|session_id=%s|strategy_id=%s",
            chat.id,
            existing_strategy.id,
        )
        return StrategyConfirmResponse(
            strategy_id=str(existing_strategy.id),
            name=existing_strategy.name,
            symbol=existing_strategy.symbol,
            market=existing_strategy.market,
            timeframe=existing_strategy.timeframe,
            yaml_path=existing_strategy.yaml_path or "",
            status=existing_strategy.status.value,
        )

    # 4. Generate YAML
    yaml_path = generate_yaml(builder)

    # 5. Persist to DB — use session_id (not chat_id)
    stored_config = strategy_config or {
        "name": f"{builder.format_symbol().lower()}_strategy",
        "asset": builder.format_symbol(),
        "entry": [],
        "exit": [],
        "reward_factor": round(float(builder.take_profit) / float(builder.stop_loss), 2)
        if builder.stop_loss
        else None,
    }

    strategy = Strategy(
        id=uuid.uuid4(),
        session_id=chat.id,          # ← correct field name
        name=stored_config.get("name") or f"{builder.format_symbol()} {builder.timeframe} Strategy",
        symbol=builder.format_symbol(),
        market=builder.market,
        timeframe=builder.timeframe or "1d",
        status=StrategyStatus.confirmed,
        yaml_path=yaml_path,
        strategy_config=stored_config,
    )
    db.add(strategy)
    await db.flush()

    await _persist_risk_execution_config(
        db,
        session_id=chat.id,
        strategy_id=strategy.id,
        builder=builder,
    )

    # Tag source message
    last_msg.strategy_id = strategy.id

    logger.info(
        "strategy_persist|source=http_confirm|outcome=created|session_id=%s|strategy_id=%s|yaml_path=%s",
        chat.id,
        strategy.id,
        yaml_path,
    )

    return StrategyConfirmResponse(
        strategy_id=str(strategy.id),
        name=strategy.name,
        symbol=strategy.symbol,
        market=strategy.market,
        timeframe=builder.timeframe,
        yaml_path=yaml_path,
        status=strategy.status.value,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /strategies/{strategy_id}  — full strategy_object response
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/strategies/{strategy_id}",
    summary="Get a saved strategy with full strategy_object",
)
async def get_strategy(
    strategy_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    strategy = await db.get(Strategy, uuid.UUID(strategy_id))
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found.")

    # Find the source chat message that has the full strategy_object payload
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == strategy.session_id,
            ChatMessage.strategy_id == strategy.id,
            ChatMessage.strategy_json.isnot(None),
        )
        .order_by(ChatMessage.created_at.asc())
    )
    source_msg = _find_latest_strategy_message(list(result.scalars().all()))

    # Also try last message with strategy_json in the session
    if not source_msg:
        result2 = await db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.session_id == strategy.session_id,
                ChatMessage.strategy_json.isnot(None),
            )
            .order_by(ChatMessage.created_at.asc())
        )
        source_msg = _find_latest_strategy_message(list(result2.scalars().all()))

    # Build full response
    strategy_object = None
    if source_msg and source_msg.strategy_json:
        sj = source_msg.strategy_json
        _, extracted_object = _extract_strategy_parts(sj)
        if extracted_object:
            runtime_snapshot = await resolve_active_risk_execution_config(
                db,
                session_id=str(strategy.session_id),
                strategy_id=str(strategy.id),
            )
            extracted_object = copy.deepcopy(extracted_object)
            extracted_object["risk_and_execution"] = build_risk_execution_response(
                runtime_snapshot
            )
            strategy_object = {"strategy_object": extracted_object}

    return {
        "strategy_id":     str(strategy.id),
        "session_id":      str(strategy.session_id),   # ← session_id, not chat_id
        "name":            strategy.name,
        "symbol":          strategy.symbol,
        "market":          strategy.market,
        "timeframe":       strategy.timeframe,
        "status":          strategy.status.value,
        "yaml_path":       strategy.yaml_path,
        "strategy_config": strategy.strategy_config,
        "strategy_object": strategy_object,            # ← full strategy_object here
        "created_at":      strategy.created_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /strategies?session_id=...  — list all strategies for a chat
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/strategies",
    summary="List all strategies for a chat session",
)
async def list_strategies(
    session_id: str = Query(..., description="The chat session_id to filter strategies by"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Strategy)
        .where(Strategy.session_id == uuid.UUID(session_id))
        .order_by(Strategy.created_at.desc())
    )
    strategies = result.scalars().all()

    return [
        {
            "strategy_id": str(s.id),
            "name":        s.name,
            "symbol":      s.symbol,
            "market":      s.market,
            "timeframe":   s.timeframe,
            "status":      s.status.value,
            "created_at":  s.created_at.isoformat(),
        }
        for s in strategies
    ]


# ── Debug endpoints ───────────────────────────────────────────────────────────


@router.get(
    "/debug/last-plan",
    summary="Return the planner decision trace for a session",
    description=(
        "Returns the structured decision trace from the most recent planner "
        "run for `session_id`. Useful for diagnosing why a particular signal "
        "was picked. Trace store is bounded (last 256 sessions, in-memory)."
    ),
    tags=["🔧 Debug"],
)
async def get_last_plan_trace(session_id: str = Query(..., description="Chat session id")):
    from app.planner.trace import recall_trace
    trace = recall_trace(session_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no planner trace found for session_id={session_id!r}",
        )
    return {"session_id": session_id, "trace": trace}

