"""
app/api/v1/routes/chat.py
══════════════════════════════════════════════════════════════════════
Chat API endpoints.

CHANGES:
  - chat_id renamed to session_id in all routes and responses.
  - POST /chats/{session_id}/messages now returns 202 immediately with
    AsyncMessageAcceptedResponse and fires AI processing in background
    via FastAPI BackgroundTasks.
  - GET /chats/{session_id}/messages fetches full history from DB
    (AI responses are saved there by the background task).
"""
from __future__ import annotations

import copy
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import build_error_object
from app.db.session import get_db
from app.schemas.chat import (
    ChatCreateRequest,
    ChatCreateResponse,
    MessageRequest,
    AsyncMessageAcceptedResponse,
    ChatHistoryResponse,
    ChatMessageItem,
    ChatListResponse,
    ChatListItem,
)
from app.services.execution.risk_execution_config_service import (
    build_risk_execution_response,
    resolve_active_risk_execution_config,
)
from app.services.backtest import normalize_backtest_metric_aliases
from app.services.chat.chat_service import (
    create_chat_session,
    get_chat,
    get_chat_history,
    get_current_mode,
    get_chat_status,
    list_chats,
    delete_chat,
    accept_message,
    cancel_ai_processing,
    run_ai_processing,
    _draft_state,
)

router = APIRouter()
logger = logging.getLogger(__name__)
_HIDDEN_STRATEGY_DRAFT_KEYS = {
    "kb_query",
    "kb_snippets",
    "daily_loss_cap_pct",
    "max_trade",
    "daily_loss_cap",
    "position_sizing",
    "execution_mode",
    "trading_window",
    "risk_validation",
    "per_trade_risk",
    "risk_reward",
    "max_trades",
    "risk_execution_config",
}
_HIDDEN_SIGNAL_PLAN_KEYS = {"kb_query", "kb_snippets", "rationale"}
_BACKTEST_METRIC_ORDER = (
    "num_days",
    "total_return_pct",
    "win_rate",
    "average_outcome_per_trade",
    "average_holding_duration",
    "irr_daily",
    "calmar_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "total_trades",
    "annual_return",
    "sortino_ratio",
    "ending_balance",
    "irr_annualized",
    "avg_daily_return",
    "volatility_pct",
    "downside_deviation_pct",
    "profit_factor",
    "var_95_pct",
    "expected_shortfall_95_pct",
    "trades_per_month",
    "longest_losing_streak",
    "starting_balance",
    "max_drawdown_duration",
    "worst_drawdown_start_date",
    "worst_drawdown_end_date",
    "drawdown_duration_days",
    "recovery_date",
    "recovery_time_days",
    "backtest_trades",
)


def _public_strategy_draft(draft: dict | None) -> dict | None:
    if not isinstance(draft, dict):
        return draft

    public_draft = {
        key: value
        for key, value in draft.items()
        if key not in _HIDDEN_STRATEGY_DRAFT_KEYS
    }

    signal_plan = public_draft.get("signal_plan")
    if isinstance(signal_plan, dict):
        public_draft["signal_plan"] = {
            key: value
            for key, value in signal_plan.items()
            if key not in _HIDDEN_SIGNAL_PLAN_KEYS
        }

    public_draft.pop("risk_and_execution", None)

    return public_draft


def _ordered_backtest_metrics(metrics: dict | None) -> dict | None:
    if not isinstance(metrics, dict):
        return metrics

    ordered_metrics = {
        key: metrics[key]
        for key in _BACKTEST_METRIC_ORDER
        if key in metrics
    }
    ordered_metrics.update(
        {
            key: value
            for key, value in metrics.items()
            if key not in ordered_metrics
        }
    )
    return ordered_metrics


def _ordered_backtest_result(backtest_result: dict | None) -> dict | None:
    if not isinstance(backtest_result, dict):
        return backtest_result

    ordered_result = copy.deepcopy(backtest_result)
    ordered_result.pop("failure", None)
    normalize_backtest_metric_aliases(ordered_result)
    ordered_result["metrics"] = _ordered_backtest_metrics(ordered_result.get("metrics"))
    return ordered_result


def _public_strategy_json(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return payload

    public_payload = dict(payload)
    context = public_payload.get("context")
    if isinstance(context, dict):
        public_context = {
            key: value
            for key, value in context.items()
            if key not in {
                "strategy_config",
                "strategy_id",
                "next_state",
                "current_mode",
                "conversation_history",
                "yaml_path",
            }
        }
        public_payload["context"] = public_context

    backtest_result = public_payload.get("backtest_result")
    if isinstance(backtest_result, dict):
        public_payload["backtest_result"] = _ordered_backtest_result(backtest_result)

    return public_payload


def _strategy_object_nodes(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []

    nodes: list[dict] = []
    direct_strategy_object = payload.get("strategy_object")
    if isinstance(direct_strategy_object, dict):
        nodes.append(direct_strategy_object)

    context = payload.get("context")
    if isinstance(context, dict):
        context_strategy_object = context.get("strategy_object")
        if isinstance(context_strategy_object, dict):
            nodes.append(context_strategy_object)

    return nodes


def _overlay_runtime_risk_execution(
    payload: dict | None,
    risk_and_execution: dict | None,
) -> dict | None:
    """Fill missing risk fields only — never replace an assembled risk block."""
    if not isinstance(payload, dict) or not isinstance(risk_and_execution, dict):
        return payload

    updated_payload = copy.deepcopy(payload)
    strategy_objects = _strategy_object_nodes(updated_payload)
    if not strategy_objects:
        return payload

    for strategy_object in strategy_objects:
        existing = strategy_object.get("risk_and_execution")
        if isinstance(existing, dict) and existing:
            # Preserve user/assembled SL/TP/window; backfill only null/missing keys.
            merged = dict(risk_and_execution)
            for key, value in existing.items():
                if value is not None:
                    merged[key] = value
            strategy_object["risk_and_execution"] = merged
        else:
            strategy_object["risk_and_execution"] = dict(risk_and_execution)

    return updated_payload


def _history_message_payload(message) -> dict:
    state = _draft_state(message.strategy_draft, message.strategy_json)
    error_payload = None
    if message.role.value == "assistant":
        strategy_json = message.strategy_json if isinstance(message.strategy_json, dict) else None
        if message.status and message.status.value == "failed":
            if strategy_json and isinstance(strategy_json.get("error"), dict):
                error_payload = strategy_json["error"]
            elif message.error_message:
                error_payload = build_error_object(500, message.error_message)

    payload = {
        "id": str(message.id),
        "role": message.role.value,
        "state": state,
        "content": message.content,
        "status": (
            "processing"
            if message.status and message.status.value == "pending"
            else (message.status.value if message.status else "completed")
        ) if message.role.value == "assistant" else None,
        "error_message": (
            error_payload.get("message")
            if isinstance(error_payload, dict)
            else message.error_message
        ) if message.role.value == "assistant" else None,
        "error": error_payload if message.role.value == "assistant" else None,
        "strategy_draft": (
            _public_strategy_draft(message.strategy_draft)
            if message.role.value == "assistant" and state != "backtest_complete"
            else None
        ),
        "created_at": message.created_at.isoformat() if message.role.value == "assistant" else None,
        "updated_at": (
            message.updated_at.isoformat() if message.updated_at else None
        ) if message.role.value == "assistant" else None,
    }

    if message.role.value == "assistant":
        public_strategy_json = _public_strategy_json(message.strategy_json)
        if isinstance(public_strategy_json, dict):
            if isinstance(public_strategy_json.get("error"), dict):
                payload["error"] = public_strategy_json["error"]
            remaining_payload = {
                key: value
                for key, value in public_strategy_json.items()
                if key != "error"
            }
            if set(remaining_payload.keys()) == {"backtest_result"}:
                payload["backtest_result"] = remaining_payload["backtest_result"]
            elif remaining_payload:
                payload["strategy_json"] = remaining_payload

    return payload


async def _history_message_payload_with_runtime_risk_execution(
    db: AsyncSession,
    session_id: str,
    message,
    snapshot_cache: dict[tuple[str, str | None], dict],
) -> dict:
    payload = _history_message_payload(message)
    if message.role.value != "assistant":
        return payload

    public_strategy_json = payload.get("strategy_json")
    if not isinstance(public_strategy_json, dict):
        return payload
    if not _strategy_object_nodes(public_strategy_json):
        return payload

    strategy_id = str(message.strategy_id) if getattr(message, "strategy_id", None) else None
    cache_key = (session_id, strategy_id)

    try:
        runtime_risk_execution = snapshot_cache.get(cache_key)
        if runtime_risk_execution is None:
            snapshot = await resolve_active_risk_execution_config(
                db,
                session_id=session_id,
                strategy_id=strategy_id,
            )
            runtime_risk_execution = build_risk_execution_response(snapshot)
            snapshot_cache[cache_key] = runtime_risk_execution

        payload["strategy_json"] = _overlay_runtime_risk_execution(
            public_strategy_json,
            runtime_risk_execution,
        )
    except Exception as exc:
        logger.warning(
            "chat_history risk_execution hydration skipped | session_id=%s message_id=%s error=%s",
            session_id,
            getattr(message, "id", None),
            exc,
        )

    return payload


def _history_current_mode(messages) -> str:
    for message in reversed(messages):
        state = _draft_state(message.strategy_draft, message.strategy_json)
        if state:
            return state
    return "collect_user_input"


# ── POST /chats ────────────────────────────────────────────────────────────────
@router.post(
    "/chats",
    response_model=ChatCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_chat(
    body: ChatCreateRequest = ChatCreateRequest(),
    db: AsyncSession = Depends(get_db),
):
    chat = await create_chat_session(db, title=body.title)
    status_value = await get_chat_status(db, str(chat.id))
    return ChatCreateResponse(
        session_id=str(chat.id),
        title=chat.title or "New Strategy",
        status=status_value,
        created_at=chat.created_at.isoformat(),
    )


# ── GET /chats ─────────────────────────────────────────────────────────────────
@router.get(
    "/chats",
    response_model=ChatListResponse,
    summary="List all chat sessions",
)
async def list_all_chats(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_chats(db, limit=limit)
    return ChatListResponse(total=len(rows), chats=[ChatListItem(**r) for r in rows])


# ── GET /chats/{session_id} ────────────────────────────────────────────────────
@router.get("/chats/{session_id}", summary="Get a single chat session")
async def get_single_chat(session_id: str, db: AsyncSession = Depends(get_db)):
    chat = await get_chat(db, session_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    status_value = await get_chat_status(db, session_id)
    return {
        "session_id": str(chat.id),
        "title":      chat.title or "Untitled",
        "status":     status_value,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
    }


# ── POST /chats/{session_id}/messages ──────────────────────────────────────────
@router.post(
    "/chats/{session_id}/messages",
    response_model=AsyncMessageAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send a user message — returns immediately, AI processes in background",
)
async def send_message(
    session_id: str,
    body: MessageRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    1. Validates session exists.
    2. Saves the user message to DB instantly (status=processing).
    3. Returns 202 with AsyncMessageAcceptedResponse immediately.
    4. Background task calls LLM and saves assistant response to DB.

    To get the AI response, call:
      GET /chats/{session_id}/messages
    and look for the latest assistant message with status='completed'.
    """
    chat = await get_chat(db, session_id)
    if not chat:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Create one first with POST /chats.",
        )

    current_mode = await get_current_mode(db, session_id)

    # Save user message, get ID back immediately
    user_msg = await accept_message(db, session_id, body.content)

    # Fire background AI processing
    # NOTE: db is NOT passed — background task opens its own session
    # to avoid sharing the request-scoped session (which closes on response).
    background_tasks.add_task(
        run_ai_processing,
        session_id,
        str(user_msg.id),
        body.content,
    )

    return AsyncMessageAcceptedResponse(
        message_id=str(user_msg.id),
        session_id=session_id,
        message="Your message is being processed.",
        current_mode=current_mode,
        created_at=user_msg.created_at.isoformat(),
    )


# ── GET /chats/{session_id}/messages ───────────────────────────────────────────
@router.get(
    "/chats/{session_id}/messages",
    response_model=ChatHistoryResponse,
    response_model_exclude_none=True,
    summary="Get full message history — poll this to retrieve AI responses",
)
async def get_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the full message history for the session.

    Assistant messages will appear here once the background task completes.
    Check status field per message:
      - processing → AI is working
      - completed  → AI response ready, read 'content'
      - failed     → something went wrong, read 'error_message'
    """
    chat = await get_chat(db, session_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    messages = await get_chat_history(db, session_id)
    status_value = await get_chat_status(db, session_id)
    snapshot_cache: dict[tuple[str, str | None], dict] = {}
    response_messages = []
    for message in messages:
        response_messages.append(
            ChatMessageItem(
                **await _history_message_payload_with_runtime_risk_execution(
                    db,
                    session_id,
                    message,
                    snapshot_cache,
                )
            )
        )

    return ChatHistoryResponse(
        session_id=session_id,
        title=chat.title or "Untitled",
        current_mode=_history_current_mode(messages),
        status=status_value,
        message_count=len(messages),
        messages=response_messages,
    )


# ── POST /chats/{session_id}/messages/pause ───────────────────────────────────
@router.post(
    "/chats/{session_id}/messages/pause",
    summary="Abort the in-flight AI generation for this session",
)
async def pause_message(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Cancels the AI generation that is currently producing a response for this
    session. Any partial assistant output is discarded — only messages that
    were fully completed before the pause remain in history. After this call
    the session is free to accept a new prompt.
    """
    chat = await get_chat(db, session_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    cancelled = await cancel_ai_processing(session_id)
    status_value = await get_chat_status(db, session_id)
    return {
        "session_id": session_id,
        "cancelled": cancelled,
        "status": status_value,
        "message": (
            "AI generation aborted. Partial response discarded; you can send a new message."
            if cancelled
            else "No active AI generation to pause."
        ),
    }


# ── DELETE /chats/{session_id} ─────────────────────────────────────────────────
@router.delete(
    "/chats/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session and all its messages",
)
async def remove_chat(session_id: str, db: AsyncSession = Depends(get_db)):
    deleted = await delete_chat(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
