"""
ChatService — the main orchestration layer for the strategy chat flow.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.assistant_responses import compose_assistant_response
from app.core.supported_stocks import (
    SUPPORTED_STOCKS_DISPLAY,
    is_supported_stock_symbol,
    unsupported_stock_validation,
)
from app.core.errors import build_error_content, normalize_exception
from app.db.models.strategy import Chat, ChatMessage, ChatRole, MessageStatus, Strategy, StrategyStatus
from app.db.session import AsyncSessionLocal
from app.services.agent.router import AgentRouter
from app.services.ai.parser import (
    detect_stock_advice_request,
    detect_supported_stocks_question,
    detect_user_confirmation,
)
from app.services.ai.system_prompt import SYSTEM_INSTRUCTION
from app.services.backtest import (
    extract_strategy_market_data_request,
    fetch_auxiliary_ohlcv,
    fetch_ohlcv_records,
    insert_chat_backtest_row,
    insert_failed_chat_backtest,
    run_quant_backtest_sync,
    summarize_backtest_for_db,
)
from app.services.backtest.market_data import (
    StrategyMarketDataRequest,
    _normalize_symbol_for_market_data,
)
from app.schemas.backtest import BacktestTriggerRequest
from app.services.chat.strategy_flow import (
    build_assemble_strategy_reply,
    build_backtest_error_reply,
    build_backtest_ready_reminder,
    build_backtest_result_reply,
    build_collect_user_input_reply,
    build_final_strategy_payload,
    build_grounded_clarification_reply,
    build_input_modification_invalid_field_reply,
    build_pause_workflow_reply,
    build_plan_signals_reminder,
    build_plan_signals_reply,
    build_sebi_compliance_message,
    select_direct_llm_reply,
    build_welcome_message,
)
from app.core.signal_performance_cache import record_performance
from app.core.config import get_settings
from app.kb.compat import resolve_supported_stock
from app.planner.legacy_bridge import (
    NoValidCandidate,
    UnsupportedStock,
    UnsupportedTimeframe,
    plan_signals_v2,
)
from app.planner.strategy_assembler import (
    build_retrieval_meta,
    build_strategy_config,
    build_strategy_object,
    enrich_plan_with_ohlcv,
)
from app.services.execution.risk_execution_config_service import (
    SESSION_SCOPE,
    STRATEGY_SCOPE,
    RiskExecutionConfigSnapshot,
    build_risk_execution_response,
    compose_risk_execution_values,
    resolve_active_risk_execution_config,
    upsert_risk_execution_config,
)
from app.services.strategy.builder import (
    CORE_USER_INPUT_FIELDS,
    StrategyBuilder,
    UNSUPPORTED_USER_TIMEFRAME_CODE,
    extract_company_name_query,
    extract_exchange_hint,
    extract_goal_text,
    extract_strategy_details,
    resolve_supported_user_timeframe,
    unsupported_user_timeframe_validation_facts,
)
from app.services.strategy.yaml_generator import generate_yaml

FLOW_MODEL_NAME = "stretus-kb-planner"
logger = logging.getLogger(__name__)

# Tracks the asyncio.Task running run_ai_processing per session_id so that
# an incoming pause request can cancel the in-flight generation.
_active_generations: dict[str, asyncio.Task] = {}


async def cancel_ai_processing(session_id: str) -> bool:
    """Abort the in-flight AI generation for ``session_id``.

    Returns True when a running task was cancelled, False when no generation
    was in flight. Awaits the cancelled task so the user message is freed up
    (status reset, partial work rolled back) before returning.
    """
    task = _active_generations.get(session_id)
    if task is None or task.done():
        return False
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    return True

_GREETING_ONLY_RE = re.compile(r"^\s*(hi+|hello+|hey+|hii+|namaste)\s*[!.]*\s*$", re.IGNORECASE)
_EXPLICIT_STOCK_CUE_RE = re.compile(
    r"\b(?:NSE|BSE)\s*[:\-]|\b[A-Za-z][A-Za-z0-9&-]{1,19}\.(?:NS|BO)\b",
    re.IGNORECASE,
)
_IGNORED_OPTIONAL_INPUT_RE = re.compile(
    r"\b(?:daily\s+loss\s+cap|max\s+daily\s+loss|risk\s+per\s+day|max\s+trades?|"
    r"trade\s+duration|max\s+trade(?:\s+duration)?)\b",
    re.IGNORECASE,
)
_INPUT_MODIFICATION_RE = re.compile(
    r"\b(?:modify|change|update|edit|revise|correct|adjust|redo)\b"
    r".{0,80}\b(?:input|inputs|field|fields|detail|details|stock|symbol|ticker|company|"
    r"timeframe|time\s*frame|objective|trade\s*type|trading\s*style|sentiment|"
    r"market\s*view|experience|goal|intent)\b|"
    r"\b(?:input|inputs|field|fields|detail|details)\b"
    r".{0,80}\b(?:modify|change|update|edit|revise|correct|adjust|redo)\b",
    re.IGNORECASE,
)
_INPUT_MODIFICATION_FILLER_RE = re.compile(
    r"\b(?:i|me|my|we|want|wants|would|like|need|to|the|a|an|this|that|please|"
    r"can|could|you|let|lets|let's|modify|change|update|edit|revise|correct|adjust|redo|"
    r"input|inputs|field|fields|detail|details|value|values|saved|previous|previously|captured|"
    r"strategy|setup|of|for)\b",
    re.IGNORECASE,
)
_INPUT_MODIFICATION_REFUSAL_RE = re.compile(
    r"\b(?:no\s+(?:changes?|change|edits?|updates?|modifications?|update|modify|change\s+needed))\b|"
    r"\b(?:nothing\s+(?:to\s+(?:change|update|modify|edit))|"
    r"don'?t\s+(?:want\s+to\s+)?(?:change|update|modify|edit|choose|pick|select)|"
    r"do\s+not\s+(?:want\s+to\s+)?(?:change|update|modify|edit|choose|pick|select)|"
    r"keep\s+(?:them|it|all|everything)\s+(?:as\s+is|same|unchanged)?|"
    r"leave\s+(?:them|it|all)\s+(?:alone|as\s+is|unchanged|same)|"
    r"(?:everything|all|all\s+(?:of\s+)?(?:them|it))\s+(?:is\s+)?(?:fine|good|correct|right|ok|okay))\b|"
    r"\b(?:no\s+(?:thanks?|thank\s+you))\b|"
    r"^\s*(?:no|nope|nah)\s*[\.!,]?\s*$",
    re.IGNORECASE,
)
_INPUT_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "symbol",
        re.compile(
            r"\b(?:stock(?:\s+name)?|symbol|ticker|company(?:\s+name)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "timeframe",
        re.compile(r"\b(?:time\s*frame|timeframe|interval)\b", re.IGNORECASE),
    ),
    (
        "objective",
        re.compile(r"\b(?:objective|trade\s*type|trading\s*style|style)\b", re.IGNORECASE),
    ),
    (
        "sentiment",
        re.compile(r"\b(?:sentiment|market\s*view|view|bias)\b", re.IGNORECASE),
    ),
    (
        "experience",
        re.compile(r"\b(?:experience|trading\s*experience|level)\b", re.IGNORECASE),
    ),
    (
        "goal",
        re.compile(r"\b(?:goal|trading\s*goal|intent|intention|setup\s*goal)\b", re.IGNORECASE),
    ),
)


def _signal_eval_window(interval: str) -> tuple[str, str]:
    """Return a short (from, to) UTC ISO range for signal param estimation.

    Uses settings.signal_eval_lookback_days (default 30) so 1-minute fetches
    stay well within rate limits while still providing enough bars for ATR/param
    estimation.  The full backtest window is NOT used here.
    """
    settings = get_settings()
    days = max(7, settings.signal_eval_lookback_days)
    now = datetime.now(tz=timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)
    start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"


def _coerce_agent_float_param(params: dict[str, Any], key: str) -> float | None:
    if params.get(key) is None:
        return None
    value = float(params[key])
    if key == "daily_loss_cap":
        if value < 0:
            raise ValueError("daily_loss_cap cannot be negative.")
    elif value <= 0:
        raise ValueError(f"{key} must be greater than 0.")
    return value


def _coerce_agent_int_param(params: dict[str, Any], key: str) -> int | None:
    if params.get(key) is None:
        return None
    value = int(params[key])
    if value < 0:
        raise ValueError(f"{key} cannot be negative.")
    return value


def _merge_agent_risk_config(
    builder: "StrategyBuilder",
    base_config: RiskExecutionConfigSnapshot,
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    merged = dict(base_config.to_builder_context())
    existing = getattr(builder, "risk_execution_config", None)
    if isinstance(existing, dict):
        merged.update({key: value for key, value in existing.items() if value is not None})

    changed: list[str] = []
    for key in ("stop_loss_pct", "take_profit_pct", "daily_loss_cap", "per_trade_risk"):
        value = _coerce_agent_float_param(params, key)
        if value is not None:
            merged[key] = value
            changed.append(key)

    max_trades = _coerce_agent_int_param(params, "max_trades")
    if max_trades is not None:
        merged["max_trades"] = max_trades
        changed.append("max_trades")

    return merged, changed


def _apply_risk_config_to_builder(builder: "StrategyBuilder", config: dict[str, Any]) -> None:
    builder.set_risk_execution_config(config)
    if config.get("stop_loss_pct") is not None:
        builder.stop_loss = float(config["stop_loss_pct"])
    if config.get("take_profit_pct") is not None:
        builder.take_profit = float(config["take_profit_pct"])
    if config.get("daily_loss_cap") is not None:
        builder.daily_loss_cap = float(config["daily_loss_cap"])
    if config.get("max_trades") is not None:
        builder.max_trade = str(int(config["max_trades"]))


def _build_risk_update_reply(
    config: dict[str, Any],
    changed_fields: list[str],
    *,
    needs_reassembly: bool,
) -> str:
    labels = {
        "stop_loss_pct": "stop loss",
        "take_profit_pct": "take profit",
        "daily_loss_cap": "daily loss cap",
        "per_trade_risk": "per-trade risk",
        "max_trades": "max trades",
    }
    pieces: list[str] = []
    for field in changed_fields:
        label = labels.get(field, field)
        value = config.get(field)
        suffix = "" if field == "max_trades" else "%"
        pieces.append(f"{label}: {_compact_number(value)}{suffix}")

    details = "; ".join(pieces) if pieces else "the requested risk settings"
    reply = f"Updated {details}."
    if needs_reassembly:
        reply += (
            " I will not run a backtest from the older assembled strategy. "
            "Please confirm if you want me to reassemble the strategy with these settings."
        )
    return reply


def _build_market_data_reply(
    *,
    symbol: str,
    interval: str,
    from_utc: str,
    to_utc: str,
    rows: list[dict[str, Any]],
) -> str:
    first = rows[0]
    last = rows[-1]
    first_close = float(first["close"])
    last_close = float(last["close"])
    change_pct = ((last_close - first_close) / first_close * 100.0) if first_close else 0.0
    high = max(float(row["high"]) for row in rows)
    low = min(float(row["low"]) for row in rows)
    volume = sum(float(row["volume"]) for row in rows[-min(len(rows), 20):])

    return (
        f"Market data fetched for {symbol} on {interval}: {len(rows)} candles from "
        f"{from_utc} to {to_utc}. Latest close is {_compact_number(last_close)}, "
        f"range high/low is {_compact_number(high)}/{_compact_number(low)}, "
        f"period change is {_compact_number(round(change_pct, 2))}%, and recent volume "
        f"across the latest {min(len(rows), 20)} candles is {_compact_number(round(volume, 2))}."
    )


_REPEAT_LOOP_THRESHOLD = 2  # if the prior 2 assistant messages are identical, escalate.


def _normalize_for_repeat_compare(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _recent_assistant_texts(messages: list[ChatMessage], limit: int) -> list[str]:
    collected: list[str] = []
    for message in reversed(messages):
        role_obj = getattr(message, "role", None)
        role_value = getattr(role_obj, "value", role_obj)
        if role_value != "assistant":
            continue
        normalized = _normalize_for_repeat_compare(getattr(message, "content", ""))
        if not normalized:
            continue
        collected.append(normalized)
        if len(collected) >= limit:
            break
    return collected


def _is_repeat_loop(candidate: str, recent: list[str]) -> bool:
    """Return True if `candidate` matches the bot's last few assistant replies."""
    if not candidate or len(recent) < _REPEAT_LOOP_THRESHOLD:
        return False
    needle = _normalize_for_repeat_compare(candidate)
    if not needle:
        return False
    return all(prev == needle for prev in recent[:_REPEAT_LOOP_THRESHOLD])


def _agent_question_text(route: dict | None) -> str | None:
    if not isinstance(route, dict):
        return None
    params = route.get("agent_tool_parameters")
    if not isinstance(params, dict):
        return None
    question = " ".join(str(params.get("question") or "").split()).strip()
    return question or None


def _prefer_agent_question_text(route: dict | None, fallback_text: str) -> str:
    return _agent_question_text(route) or fallback_text


async def create_chat_session(db: AsyncSession, title: Optional[str] = None) -> Chat:
    chat = Chat(id=uuid.uuid4(), title=title or "New Strategy")
    db.add(chat)
    await db.flush()

    db.add(
        ChatMessage(
            id=uuid.uuid4(),
            session_id=chat.id,
            role=ChatRole.system,
            content=SYSTEM_INSTRUCTION,
            model="system",
            status=MessageStatus.completed,
        )
    )

    welcome_builder = StrategyBuilder()
    db.add(
        ChatMessage(
            id=uuid.uuid4(),
            session_id=chat.id,
            role=ChatRole.assistant,
            content=build_welcome_message(),
            model=FLOW_MODEL_NAME,
            strategy_draft=welcome_builder.to_draft_json(
                mode_override="collect_user_input",
                processing_status="awaiting_user_input",
            ),
            status=MessageStatus.completed,
            is_final=True,
        )
    )
    return chat


async def get_chat(db: AsyncSession, session_id: str) -> Optional[Chat]:
    try:
        return await db.get(Chat, uuid.UUID(session_id))
    except (ValueError, AttributeError):
        return None


def _normalize_message_status(status: Optional[MessageStatus]) -> str:
    if status == MessageStatus.pending:
        return "processing"
    if status:
        return status.value
    return "completed"


async def get_chat_status(db: AsyncSession, session_id: str) -> str:
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, AttributeError):
        return "completed"

    result = await db.execute(
        select(ChatMessage.status)
        .where(
            ChatMessage.session_id == session_uuid,
            ChatMessage.role != ChatRole.system,
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(1)
    )
    latest_status = result.scalar_one_or_none()
    return _normalize_message_status(latest_status)


async def list_chats(db: AsyncSession, limit: int = 50) -> list[dict]:
    result = await db.execute(select(Chat).order_by(Chat.created_at.desc()).limit(limit))
    chats = result.scalars().all()
    rows = []

    for chat in chats:
        count_result = await db.execute(
            select(func.count(ChatMessage.id)).where(
                ChatMessage.session_id == chat.id,
                ChatMessage.role != ChatRole.system,
            )
        )
        msg_count = count_result.scalar() or 0
        status_result = await db.execute(
            select(ChatMessage.status)
            .where(
                ChatMessage.session_id == chat.id,
                ChatMessage.role != ChatRole.system,
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
        latest_status = status_result.scalar_one_or_none()
        rows.append(
            {
                "session_id": str(chat.id),
                "title": chat.title or "Untitled",
                "status": _normalize_message_status(latest_status),
                "message_count": msg_count,
                "created_at": chat.created_at.isoformat(),
                "updated_at": chat.updated_at.isoformat(),
            }
        )

    return rows


async def delete_chat(db: AsyncSession, session_id: str) -> bool:
    chat = await get_chat(db, session_id)
    if not chat:
        return False
    await db.delete(chat)
    return True


async def get_chat_history(db: AsyncSession, session_id: str) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == uuid.UUID(session_id),
            ChatMessage.role != ChatRole.system,
        )
        .order_by(ChatMessage.created_at.asc())
    )
    return result.scalars().all()


async def get_current_mode(db: AsyncSession, session_id: str) -> str:
    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == uuid.UUID(session_id),
            ChatMessage.role != ChatRole.system,
        )
        .order_by(ChatMessage.created_at.desc())
    )
    messages = result.scalars().all()

    for message in messages:
        mode = _draft_state(message.strategy_draft, message.strategy_json)
        if mode:
            return mode

    return "collect_user_input"


def _draft_state(draft: Optional[dict], strategy_json: Optional[dict] = None) -> Optional[str]:
    if draft and draft.get("mode"):
        return draft["mode"]
    context = (strategy_json or {}).get("context", {})
    return context.get("current_mode")


def _generate_chat_title(draft: dict) -> str:
    market_labels = {
        "indian_indices": "Indian Indices",
        "indian_stocks": "Indian Stocks",
        "us_stocks": "US Stocks",
        "crypto": "Crypto",
        "commodity_market": "Commodity Market",
    }
    symbol = (draft.get("symbol") or "Unknown").upper()
    timeframe = draft.get("timeframe") or ""
    market = draft.get("market") or ""
    label = market_labels.get(market, market.replace("_", " ").title())
    asset_part = " ".join(part for part in [symbol, timeframe] if part).strip()
    if asset_part and label:
        return f"{asset_part} - {label}"
    return asset_part or label or "New Strategy"


def _is_placeholder_title(title: Optional[str]) -> bool:
    return title in (None, "", "New Strategy")


def _is_greeting_message(message: str) -> bool:
    return bool(_GREETING_ONLY_RE.match(message or ""))


def _derive_max_trades_from_builder(builder: "StrategyBuilder") -> int:
    """
    Derive max_trades_per_day from the strategy builder's objective and max_trade field.

    Returns 0 (unlimited) for positional strategies.
    For intraday strategies, attempts to parse a number from the max_trade string
    (e.g. '10 trades' → 10). Falls back to 0 (unlimited) if not parseable.
    """
    import re as _re
    objective = str(getattr(builder, "objective", None) or "positional").lower()
    if objective != "intraday":
        return 0

    max_trade_str = str(getattr(builder, "max_trade", None) or "")
    match = _re.search(r"(\d+)\s*trade", max_trade_str, _re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback: if just a plain integer is stored
    plain_match = _re.search(r"^\s*(\d+)\s*$", max_trade_str)
    if plain_match:
        return int(plain_match.group(1))

    return 0   # unlimited


def _build_risk_execution_values_for_builder(
    builder: "StrategyBuilder",
    base_config: RiskExecutionConfigSnapshot,
) -> dict[str, object]:
    risk_cfg = dict(getattr(builder, "risk_execution_config", None) or {})
    stop_loss_pct = (
        float(builder.stop_loss)
        if builder.stop_loss is not None
        else float(risk_cfg.get("stop_loss_pct", base_config.stop_loss_pct))
    )
    take_profit_pct = (
        float(builder.take_profit)
        if builder.take_profit is not None
        else float(risk_cfg.get("take_profit_pct", base_config.take_profit_pct))
    )
    daily_loss_cap = (
        float(builder.daily_loss_cap)
        if builder.daily_loss_cap is not None
        else float(risk_cfg.get("daily_loss_cap", base_config.daily_loss_cap))
    )
    max_trades = int(risk_cfg.get("max_trades", base_config.max_trades))
    if builder.max_trade is not None:
        derived_max_trades = _derive_max_trades_from_builder(builder)
        if derived_max_trades > 0:
            max_trades = derived_max_trades

    return compose_risk_execution_values(
        max_trades=max_trades,
        daily_loss_cap=daily_loss_cap,
        execution_mode=str(risk_cfg.get("execution_mode", base_config.execution_mode)),
        per_trade_risk=float(risk_cfg.get("per_trade_risk", base_config.per_trade_risk)),
        trading_window=str(risk_cfg.get("trading_window", base_config.trading_window)),
        position_sizing=str(risk_cfg.get("position_sizing", base_config.position_sizing)),
        risk_validation=str(risk_cfg.get("risk_validation", base_config.risk_validation)),
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        minimum_trade_value=float(
            risk_cfg.get("minimum_trade_value", base_config.minimum_trade_value)
        ),
    )


def _risk_snapshot_from_values(
    values: dict[str, object],
    *,
    config_scope: str,
    scope_id: str,
    session_id: str | None = None,
    strategy_id: str | None = None,
) -> RiskExecutionConfigSnapshot:
    return RiskExecutionConfigSnapshot(
        config_scope=config_scope,
        scope_id=scope_id,
        session_id=session_id,
        strategy_id=strategy_id,
        max_trades=int(values["max_trades"]),
        risk_reward=(
            float(values["risk_reward"])
            if values.get("risk_reward") is not None
            else None
        ),
        daily_loss_cap=float(values["daily_loss_cap"]),
        execution_mode=str(values["execution_mode"]),
        per_trade_risk=float(values["per_trade_risk"]),
        trading_window=str(values["trading_window"]),
        position_sizing=str(values["position_sizing"]),
        risk_validation=str(values["risk_validation"]),
        stop_loss_pct=float(values["stop_loss_pct"]),
        take_profit_pct=float(values["take_profit_pct"]),
        minimum_trade_value=float(values["minimum_trade_value"]),
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
    context = snapshot.to_builder_context()
    existing = getattr(builder, "risk_execution_config", None)
    if isinstance(existing, dict):
        context.update({key: value for key, value in existing.items() if value is not None})
    builder.set_risk_execution_config(context)
    return snapshot


async def _upsert_builder_risk_execution_config(
    db: AsyncSession,
    builder: "StrategyBuilder",
    *,
    session_id: uuid.UUID,
    strategy_id: str | None = None,
) -> RiskExecutionConfigSnapshot:
    active_config = await _hydrate_builder_risk_execution_config(
        db,
        builder,
        session_id=session_id,
        strategy_id=strategy_id,
    )
    values = _build_risk_execution_values_for_builder(builder, active_config)

    await upsert_risk_execution_config(
        db,
        config_scope=SESSION_SCOPE,
        scope_id=str(session_id),
        session_id=str(session_id),
        strategy_id=strategy_id,
        **values,
    )
    effective_session_snapshot = _risk_snapshot_from_values(
        values,
        config_scope=SESSION_SCOPE,
        scope_id=str(session_id),
        session_id=str(session_id),
        strategy_id=strategy_id,
    )
    builder.set_risk_execution_config(effective_session_snapshot.to_builder_context())

    if strategy_id:
        await upsert_risk_execution_config(
            db,
            config_scope=STRATEGY_SCOPE,
            scope_id=strategy_id,
            session_id=str(session_id),
            strategy_id=strategy_id,
            **values,
        )
        effective_strategy_snapshot = _risk_snapshot_from_values(
            values,
            config_scope=STRATEGY_SCOPE,
            scope_id=strategy_id,
            session_id=str(session_id),
            strategy_id=strategy_id,
        )
        builder.set_risk_execution_config(effective_strategy_snapshot.to_builder_context())
        return effective_strategy_snapshot

    return effective_session_snapshot


def _fallback_title_from_message(user_content: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", user_content).strip(" \t\r\n.,!?-:")
    cleaned = re.sub(r"^(?:hi+|hello+|hey+|namaste)[,\s]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\r\n.,!?-:")
    if not cleaned:
        return None

    words = cleaned.split()
    title = " ".join(words[:8])
    if len(words) > 8 or len(cleaned) > len(title):
        title = title.rstrip(".,!?") + "..."
    return title


def _generate_chat_title_from_message(user_content: str, builder: StrategyBuilder) -> Optional[str]:
    symbol = builder.format_symbol() or builder.symbol
    if symbol:
        return _generate_chat_title(
            {
                "market": builder.market,
                "symbol": symbol,
                "timeframe": builder.timeframe,
            }
        )
    if _is_greeting_message(user_content) or detect_user_confirmation(user_content):
        return None
    return _fallback_title_from_message(user_content)


_CORE_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "market",
    "symbol",
    "timeframe",
    "sentiment",
    "experience",
    "objective",
    "goal",
)
_CAPTURE_ACK_LABELS: dict[str, str] = {
    "symbol": "stock",
    "timeframe": "timeframe",
    "objective": "trade type",
    "sentiment": "market view",
    "experience": "experience level",
    "goal": "trading goal",
}

# Natural-language templates used to acknowledge captured/modified fields.
# The selector below picks one deterministically based on the conversation turn
# so wording varies across turns but stays idempotent within the same turn
# (refreshing won't change the phrasing).
#
# Two summary forms are exposed so templates can pair with the right grammar:
#   {summary_to}      → "timeframe to 10m"
#                       "stock to TCS and timeframe to 15m"
#                       (paired with action verbs: set/updated/captured)
#   {summary_is}      → "timeframe is 10m"
#                       "stock is TCS and timeframe is 15m"
#                       (used for declarative phrasing)
#   {summary_is_cap}  → same as {summary_is} with first letter capitalised
_CAPTURE_INITIAL_TEMPLATES: tuple[str, ...] = (
    "Got it — set {summary_to}.",
    "Noted. {summary_is_cap}.",
    "Captured {summary_to}.",
    "Saved {summary_to}.",
    "Done. {summary_is_cap}.",
    "Locked in — {summary_to}.",
)
_CAPTURE_MODIFY_TEMPLATES: tuple[str, ...] = (
    "Updated {summary_to}.",
    "Switched {summary_to}.",
    "Changed {summary_to}.",
    "Done. {summary_is_cap}.",
    "Got it. {summary_is_cap}.",
    "Noted — {summary_to}.",
)


def _core_snapshot(builder: StrategyBuilder) -> tuple[Optional[str], ...]:
    return tuple(getattr(builder, field, None) for field in _CORE_SNAPSHOT_FIELDS)


def _format_capture_value(builder: StrategyBuilder, field: str, value: Any) -> str:
    if field == "symbol":
        return str(builder.format_symbol() or value)
    text = str(value or "").strip()
    if field == "goal" and len(text) > 60:
        return f'"{text[:57].rstrip()}..."'
    return text


def _join_capture_parts(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _build_capture_acknowledgement(
    builder: StrategyBuilder,
    before_snapshot: tuple[Optional[str], ...],
    after_snapshot: tuple[Optional[str], ...],
    *,
    was_modification: bool,
    turn_index: int = 0,
) -> Optional[str]:
    """Build a varied, natural one-line ack of fields the user just provided/changed.

    Returns ``None`` when nothing changed or the new value did not actually land
    on the builder (for example because of a validation error).  The phrasing
    is selected deterministically from a curated pool using ``turn_index`` so
    the user sees variety across turns instead of the same prefix every time.
    """
    parts_to: list[str] = []
    parts_is: list[str] = []
    for index, field in enumerate(_CORE_SNAPSHOT_FIELDS):
        if field not in _CAPTURE_ACK_LABELS:
            continue
        before_value = before_snapshot[index] if index < len(before_snapshot) else None
        after_value = after_snapshot[index] if index < len(after_snapshot) else None
        if not after_value or before_value == after_value:
            continue
        label = _CAPTURE_ACK_LABELS[field]
        rendered_value = _format_capture_value(builder, field, after_value)
        if not rendered_value:
            continue
        parts_to.append(f"{label} to {rendered_value}")
        parts_is.append(f"{label} is {rendered_value}")

    if not parts_to:
        return None

    summary_to = _join_capture_parts(parts_to)
    summary_is = _join_capture_parts(parts_is)
    summary_is_cap = _capitalize_first(summary_is)

    templates = (
        _CAPTURE_MODIFY_TEMPLATES if was_modification else _CAPTURE_INITIAL_TEMPLATES
    )
    chosen = templates[max(0, turn_index) % len(templates)]
    return chosen.format(
        summary_to=summary_to,
        summary_is=summary_is,
        summary_is_cap=summary_is_cap,
    )


def _has_explicit_stock_cue(message: str) -> bool:
    return bool(_EXPLICIT_STOCK_CUE_RE.search(message or ""))


def _contains_ignored_optional_input(message: str) -> bool:
    return bool(_IGNORED_OPTIONAL_INPUT_RE.search(message or ""))


def _detect_input_modification_request(message: str, route_intent: str | None = None) -> bool:
    return route_intent == "modify_input" or bool(_INPUT_MODIFICATION_RE.search(message or ""))


def _extract_input_modification_fields(message: str) -> list[str]:
    fields: list[str] = []
    for field, pattern in _INPUT_FIELD_PATTERNS:
        if pattern.search(message or ""):
            fields.append(field)
    return fields


def _remove_input_field_labels(message: str, fields: list[str]) -> str:
    cleaned = message or ""
    for field, pattern in _INPUT_FIELD_PATTERNS:
        if field in fields:
            cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _is_input_modification_selection_only(message: str, fields: list[str]) -> bool:
    if not fields:
        return False
    cleaned = _remove_input_field_labels(message, fields)
    cleaned = _INPUT_MODIFICATION_FILLER_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", cleaned)
    return not cleaned.strip()


def _is_input_modification_refusal(message: str) -> bool:
    """User explicitly refuses to modify ('no changes', 'leave it', 'dont want to update')."""
    return bool(_INPUT_MODIFICATION_REFUSAL_RE.search(message or ""))


def _join_human_readable(parts: list[str]) -> str:
    cleaned = [part for part in parts if part]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _summarize_interpreted_values(
    builder: StrategyBuilder,
    route: dict,
    recognized_fields: set[str],
) -> str:
    parts: list[str] = []

    if "symbol" in recognized_fields:
        symbol = builder.format_symbol() or builder.symbol or str(route.get("stock_query") or "").strip()
        if symbol:
            parts.append(f"stock as {symbol}")
    if "timeframe" in recognized_fields:
        timeframe = builder.timeframe or str(route.get("timeframe_input") or "").strip()
        if timeframe:
            parts.append(f"timeframe as {timeframe}")
    if "objective" in recognized_fields and builder.objective:
        parts.append(f"trade type as {builder.objective}")
    if "sentiment" in recognized_fields and builder.sentiment:
        parts.append(f"market view as {builder.sentiment}")
    if "experience" in recognized_fields and builder.experience:
        parts.append(f"experience as {builder.experience}")
    if "goal" in recognized_fields and builder.goal:
        parts.append(f"goal as {builder.goal}")

    return _join_human_readable(parts)


def _draft_processing_status(builder: StrategyBuilder, mode: str) -> str:
    if mode == "assemble_strategy":
        return "awaiting_confirmation"
    if mode == "backtest_complete":
        return "complete"
    if mode == "plan_signals":
        return "awaiting_confirmation"
    if mode == "backtest_confirmation":
        return "awaiting_confirmation"
    if not builder.is_user_input_complete():
        return "awaiting_user_input"
    return "awaiting_confirmation"


def _strategy_context(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    context = payload.get("context")
    return context if isinstance(context, dict) else None


def _find_latest_strategy_payload_message(messages: list[ChatMessage]) -> Optional[ChatMessage]:
    for message in reversed(messages):
        context = _strategy_context(message.strategy_json)
        if not context:
            continue
        if context.get("strategy_object") or context.get("yaml_path") or context.get("strategy_id"):
            return message
    return None


def _find_latest_backtest_result_message(messages: list[ChatMessage]) -> Optional[ChatMessage]:
    for message in reversed(messages):
        strategy_json = message.strategy_json if isinstance(message.strategy_json, dict) else None
        if isinstance((strategy_json or {}).get("backtest_result"), dict):
            return message
    return None


def _extract_latest_backtest_result(messages: list[ChatMessage]) -> Optional[dict]:
    message = _find_latest_backtest_result_message(messages)
    if not message:
        return None
    strategy_json = message.strategy_json if isinstance(message.strategy_json, dict) else {}
    backtest_result = strategy_json.get("backtest_result")
    return backtest_result if isinstance(backtest_result, dict) else None


def _extract_explicit_stock_query(message: str) -> Optional[str]:
    if not message:
        return None

    explicit_suffix = re.search(r"\b([A-Za-z][A-Za-z0-9&-]{1,19})\.(?:NS|BO)\b", message, re.IGNORECASE)
    if explicit_suffix:
        return explicit_suffix.group(1).upper()

    explicit_exchange = re.search(r"\b(?:NSE|BSE)\s*[:\-]\s*([A-Za-z][A-Za-z0-9&-]{1,19})\b", message, re.IGNORECASE)
    if explicit_exchange:
        return explicit_exchange.group(1).upper()

    natural_exchange = re.search(
        r"\b([A-Za-z][A-Za-z0-9&-]{1,19})\s+(?:on|in)\s+(?:NSE|BSE)\b",
        message,
        re.IGNORECASE,
    )
    if natural_exchange:
        return natural_exchange.group(1).upper()

    return None


async def accept_message(db: AsyncSession, session_id: str, user_content: str) -> ChatMessage:
    user_msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=uuid.UUID(session_id),
        role=ChatRole.user,
        content=user_content,
        status=MessageStatus.processing,
    )
    db.add(user_msg)
    await db.flush()
    return user_msg


async def run_ai_processing(session_id: str, user_message_id: str, user_content: str) -> None:
    session_uuid = uuid.UUID(session_id)
    user_msg_uuid = uuid.UUID(user_message_id)
    logger.info(
        "🚀 chat_flow|event=processing_start|session_id=%s|message_id=%s|content_len=%d",
        session_id,
        user_message_id,
        len(user_content),
    )

    current_task = asyncio.current_task()
    if current_task is not None:
        _active_generations[session_id] = current_task
        current_task.add_done_callback(
            lambda t, sid=session_id: _active_generations.pop(sid, None)
            if _active_generations.get(sid) is t
            else None
        )

    async with AsyncSessionLocal() as db:
        try:
            user_msg = await db.get(ChatMessage, user_msg_uuid)
            if user_msg:
                user_msg.status = MessageStatus.processing
                await db.flush()

            history_result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_uuid)
                .order_by(ChatMessage.created_at.asc())
                .limit(80)
            )
            all_messages = history_result.scalars().all()
            chat_obj = await db.get(Chat, session_uuid)
            logger.info(
                "📖 chat_flow|event=history_loaded|session_id=%s|message_count=%d",
                session_id,
                len(all_messages),
            )

            builder = StrategyBuilder()
            last_draft = next(
                (message.strategy_draft for message in reversed(all_messages) if message.strategy_draft),
                None,
            )
            latest_strategy_message = _find_latest_strategy_payload_message(all_messages)
            latest_strategy_context = (
                _strategy_context(latest_strategy_message.strategy_json)
                if latest_strategy_message
                else None
            )
            latest_backtest_result = _extract_latest_backtest_result(all_messages)
            previous_state = "collect_user_input"
            if last_draft:
                previous_state = last_draft.get("mode", "collect_user_input")
                builder.merge_preview(last_draft)
                logger.info(
                    "🔄 chat_flow|event=state_restored|session_id=%s|previous_state=%s"
                    "|symbol=%s|timeframe=%s|goal=%r",
                    session_id,
                    previous_state,
                    getattr(builder, "symbol", None),
                    getattr(builder, "timeframe", None),
                    (getattr(builder, "goal", None) or "")[:60],
                )
            else:
                logger.info(
                    "🆕 chat_flow|event=fresh_session_turn|session_id=%s|previous_state=%s",
                    session_id,
                    previous_state,
                )

            await _hydrate_builder_risk_execution_config(
                db,
                builder,
                session_id=session_uuid,
                strategy_id=str((latest_strategy_context or {}).get("strategy_id") or "") or None,
            )
            logger.info(
                "⚙️ chat_flow|event=risk_config_hydrated|session_id=%s|strategy_id=%s",
                session_id,
                (latest_strategy_context or {}).get("strategy_id"),
            )

            agent_decision = await AgentRouter().decide(
                session_id=session_id,
                user_message=user_content,
                builder=builder,
                previous_state=previous_state,
                recent_messages=all_messages,
                latest_strategy_context=latest_strategy_context,
                latest_backtest_result=latest_backtest_result,
            )
            route = agent_decision.to_legacy_route()
            route_intent = route.get("intent", "collect_input")
            route_reply_text = (route.get("reply_text") or "").strip() or None
            logger.info(
                "🧭 chat_flow|event=agent_decision|session_id=%s|intent=%s"
                "|tool=%s|source=%s|recognized_fields=%s|needs_clarification=%s",
                session_id,
                route_intent,
                route.get("agent_tool"),
                route.get("agent_source"),
                route.get("recognized_fields") or [],
                bool(route.get("needs_clarification")),
            )

            if detect_supported_stocks_question(user_content):
                logger.info(
                    "📚 chat_flow|event=supported_stocks_question|session_id=%s",
                    session_id,
                )
                assistant_text = (
                    f"Currently supported stocks are: {SUPPORTED_STOCKS_DISPLAY}. "
                    "Please pick one to start building a strategy."
                )
                draft_mode = previous_state
                processing_status = _draft_processing_status(builder, draft_mode)
                draft = builder.to_draft_json(
                    mode_override=draft_mode,
                    processing_status=processing_status,
                )
                if user_msg:
                    user_msg.strategy_draft = draft
                    user_msg.status = MessageStatus.completed

                assistant_msg = ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role=ChatRole.assistant,
                    content=assistant_text,
                    model=FLOW_MODEL_NAME,
                    strategy_draft=draft,
                    strategy_json=None,
                    status=MessageStatus.completed,
                    is_final=True,
                    parent_message_id=user_msg_uuid,
                )
                db.add(assistant_msg)
                await db.commit()
                return

            if route_intent == "stock_advice_request" or detect_stock_advice_request(user_content):
                logger.info(
                    "🛡️ chat_flow|event=sebi_compliance_intercept|session_id=%s|intent=%s",
                    session_id,
                    route_intent,
                )
                assistant_text = build_sebi_compliance_message()
                draft_mode = previous_state
                processing_status = _draft_processing_status(builder, draft_mode)
                draft = builder.to_draft_json(
                    mode_override=draft_mode,
                    processing_status=processing_status,
                )
                if user_msg:
                    user_msg.strategy_draft = draft
                    user_msg.status = MessageStatus.completed

                assistant_msg = ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role=ChatRole.assistant,
                    content=assistant_text,
                    model=FLOW_MODEL_NAME,
                    strategy_draft=draft,
                    strategy_json=None,
                    status=MessageStatus.completed,
                    is_final=True,
                    parent_message_id=user_msg_uuid,
                )
                db.add(assistant_msg)
                await db.commit()
                return

            if route_intent == "user_rejection":
                logger.info(
                    "🛑 chat_flow|event=strategy_rejected|session_id=%s|state=%s|tool=%s",
                    session_id,
                    previous_state,
                    route.get("agent_tool"),
                )
                agent_params = route.get("agent_tool_parameters") or {}
                assistant_text = (
                    str(agent_params.get("question") or "").strip()
                    or str(route.get("reply_text") or "").strip()
                    or str(agent_params.get("reason") or "").strip()
                )
                draft = builder.to_draft_json(
                    mode_override=previous_state,
                    processing_status="awaiting_user_input",
                )
                draft["agent_decision"] = {
                    "tool": route.get("agent_tool"),
                    "parameters": route.get("agent_tool_parameters") or {},
                    "source": route.get("agent_source"),
                }
                draft["user_rejection"] = {
                    "detected": True,
                    "halted_previous_flow": True,
                    "reason": (route.get("agent_tool_parameters") or {}).get("reason"),
                }
                if user_msg:
                    user_msg.strategy_draft = draft
                    user_msg.status = MessageStatus.completed

                assistant_msg = ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role=ChatRole.assistant,
                    content=assistant_text,
                    model=FLOW_MODEL_NAME,
                    strategy_draft=draft,
                    strategy_json=None,
                    status=MessageStatus.completed,
                    is_final=True,
                    parent_message_id=user_msg_uuid,
                )
                db.add(assistant_msg)
                await db.commit()
                return

            if route_intent == "pause_workflow":
                logger.info(
                    "⏸️ chat_flow|event=workflow_paused|session_id=%s|state=%s|tool=%s",
                    session_id,
                    previous_state,
                    route.get("agent_tool"),
                )
                assistant_text = build_pause_workflow_reply(previous_state)
                draft = builder.to_draft_json(
                    mode_override=previous_state,
                    processing_status="paused",
                )
                draft["agent_decision"] = {
                    "tool": route.get("agent_tool"),
                    "parameters": route.get("agent_tool_parameters") or {},
                    "source": route.get("agent_source"),
                }
                draft["workflow_paused"] = True
                if user_msg:
                    user_msg.strategy_draft = draft
                    user_msg.status = MessageStatus.completed

                assistant_msg = ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role=ChatRole.assistant,
                    content=assistant_text,
                    model=FLOW_MODEL_NAME,
                    strategy_draft=draft,
                    strategy_json=None,
                    status=MessageStatus.completed,
                    is_final=True,
                    parent_message_id=user_msg_uuid,
                )
                db.add(assistant_msg)
                await db.commit()
                return

            if route_intent == "new_strategy":
                logger.info(
                    "🆕 chat_flow|event=new_strategy_requested|session_id=%s|tool=%s",
                    session_id,
                    route.get("agent_tool"),
                )
                builder = StrategyBuilder()
                previous_state = "collect_user_input"
                latest_strategy_context = None
                await _hydrate_builder_risk_execution_config(
                    db,
                    builder,
                    session_id=session_uuid,
                )
                if not route.get("recognized_fields"):
                    assistant_text = build_collect_user_input_reply(
                        builder,
                        user_content,
                        preface="Sure. Let's create a new strategy with new inputs.",
                    )
                    draft = builder.to_draft_json(
                        mode_override="collect_user_input",
                        processing_status=_draft_processing_status(builder, "collect_user_input"),
                    )
                    draft["new_strategy_requested"] = True
                    draft["agent_decision"] = {
                        "tool": route.get("agent_tool"),
                        "parameters": route.get("agent_tool_parameters") or {},
                        "source": route.get("agent_source"),
                    }
                    if user_msg:
                        user_msg.strategy_draft = draft
                        user_msg.status = MessageStatus.completed

                    assistant_msg = ChatMessage(
                        id=uuid.uuid4(),
                        session_id=session_uuid,
                        role=ChatRole.assistant,
                        content=assistant_text,
                        model=FLOW_MODEL_NAME,
                        strategy_draft=draft,
                        strategy_json=None,
                        status=MessageStatus.completed,
                        is_final=True,
                        parent_message_id=user_msg_uuid,
                    )
                    db.add(assistant_msg)
                    await db.commit()
                    return
                route_intent = "collect_input"
                route["intent"] = "collect_input"

            if route_intent == "market_inquiry":
                params = dict(route.get("agent_tool_parameters") or {})
                raw_symbol = str(params.get("symbol") or builder.symbol or "").strip()
                raw_interval = str(params.get("interval") or builder.timeframe or "1d").strip()
                assistant_state = previous_state
                draft = builder.to_draft_json(
                    mode_override=assistant_state,
                    processing_status=_draft_processing_status(builder, assistant_state),
                )
                draft["agent_decision"] = {
                    "tool": route.get("agent_tool"),
                    "parameters": params,
                    "source": route.get("agent_source"),
                }

                if not raw_symbol:
                    assistant_text = (
                        "Which stock should I fetch market data for? You can also include "
                        "the timeframe, for example INFY.NS on 15m."
                    )
                    draft["processing_status"] = "awaiting_user_input"
                else:
                    try:
                        stock_match = await resolve_supported_stock(raw_symbol)
                        resolved_symbol = str(
                            (stock_match or {}).get("symbol") or raw_symbol
                        ).strip().upper()
                        if not is_supported_stock_symbol(resolved_symbol):
                            raise ValueError("This stock is not currently supported for market data.")

                        interval, validation_message = resolve_supported_user_timeframe(raw_interval)
                        if not interval:
                            raise ValueError(validation_message or "Unsupported timeframe.")

                        default_from, default_to = _signal_eval_window(interval)
                        from_utc = str(params.get("from_utc") or default_from).strip()
                        to_utc = str(params.get("to_utc") or default_to).strip()
                        market_data_request = StrategyMarketDataRequest(
                            yaml_path="",
                            raw_symbol=resolved_symbol,
                            symbol=_normalize_symbol_for_market_data(resolved_symbol),
                            interval=interval,
                            from_utc=from_utc,
                            to_utc=to_utc,
                        )
                        logger.info(
                            "📡 chat_flow|event=agent_market_data_fetching|session_id=%s"
                            "|symbol=%s|interval=%s|from=%s|to=%s",
                            session_id,
                            market_data_request.symbol,
                            market_data_request.interval,
                            market_data_request.from_utc,
                            market_data_request.to_utc,
                        )
                        ohlcv_data = await fetch_ohlcv_records(market_data_request)
                        assistant_text = _build_market_data_reply(
                            symbol=resolved_symbol,
                            interval=interval,
                            from_utc=from_utc,
                            to_utc=to_utc,
                            rows=ohlcv_data,
                        )
                        draft["processing_status"] = "complete"
                        draft["market_data_summary"] = {
                            "symbol": resolved_symbol,
                            "interval": interval,
                            "from_utc": from_utc,
                            "to_utc": to_utc,
                            "candles": len(ohlcv_data),
                            "latest_close": ohlcv_data[-1].get("close") if ohlcv_data else None,
                        }
                    except Exception as market_exc:
                        logger.warning(
                            "⚠️ chat_flow|event=agent_market_data_failed|session_id=%s|err=%s",
                            session_id,
                            str(market_exc)[:160],
                            exc_info=True,
                        )
                        assistant_text = (
                            "I could not fetch market data for that request. "
                            f"{str(market_exc)}"
                        )
                        draft["processing_status"] = "failed"
                        draft["market_data_error"] = str(market_exc)

                if user_msg:
                    user_msg.strategy_draft = draft
                    user_msg.status = MessageStatus.completed

                assistant_msg = ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role=ChatRole.assistant,
                    content=assistant_text,
                    model=FLOW_MODEL_NAME,
                    strategy_draft=draft,
                    strategy_json=None,
                    status=MessageStatus.completed,
                    is_final=True,
                    parent_message_id=user_msg_uuid,
                )
                db.add(assistant_msg)
                await db.commit()
                return

            if route_intent == "risk_execution_update":
                params = dict(route.get("agent_tool_parameters") or {})
                assistant_state = previous_state
                try:
                    active_config = await resolve_active_risk_execution_config(
                        db,
                        session_id=str(session_uuid),
                        strategy_id=str((latest_strategy_context or {}).get("strategy_id") or "") or None,
                    )
                    merged_risk_config, changed_risk_fields = _merge_agent_risk_config(
                        builder,
                        active_config,
                        params,
                    )
                    if not changed_risk_fields:
                        assistant_text = (
                            "Which risk or execution setting should I update? "
                            "You can change stop loss, take profit, daily loss cap, "
                            "per-trade risk, or max trades."
                        )
                        processing_status = "awaiting_user_input"
                    else:
                        _apply_risk_config_to_builder(builder, merged_risk_config)
                        needs_reassembly = bool(
                            latest_strategy_context
                            or previous_state in {
                                "assemble_strategy",
                                "backtest_confirmation",
                                "backtest_complete",
                            }
                        )
                        if builder.signal_plan:
                            assistant_state = "plan_signals" if needs_reassembly else previous_state
                        elif builder.is_user_input_complete() and builder.user_input_confirmed:
                            assistant_state = "plan_signals"
                        else:
                            assistant_state = "collect_user_input"
                        assistant_text = _build_risk_update_reply(
                            merged_risk_config,
                            changed_risk_fields,
                            needs_reassembly=needs_reassembly,
                        )
                        processing_status = (
                            "awaiting_confirmation"
                            if assistant_state == "plan_signals"
                            else _draft_processing_status(builder, assistant_state)
                        )
                except Exception as risk_exc:
                    logger.warning(
                        "⚠️ chat_flow|event=agent_risk_update_failed|session_id=%s|err=%s",
                        session_id,
                        str(risk_exc)[:160],
                        exc_info=True,
                    )
                    merged_risk_config = dict(builder.risk_execution_config or {})
                    changed_risk_fields = []
                    assistant_text = (
                        "I could not update those risk settings. "
                        f"{str(risk_exc)}"
                    )
                    processing_status = "awaiting_user_input"

                draft = builder.to_draft_json(
                    mode_override=assistant_state,
                    processing_status=processing_status,
                )
                if builder.daily_loss_cap is not None:
                    draft["daily_loss_cap_pct"] = builder.daily_loss_cap
                if builder.max_trade is not None:
                    draft["max_trade"] = builder.max_trade
                if merged_risk_config:
                    draft["risk_execution_config"] = merged_risk_config
                draft["risk_execution_update"] = {
                    "changed_fields": changed_risk_fields,
                    "requires_reassembly": bool(
                        latest_strategy_context
                        or previous_state in {
                            "assemble_strategy",
                            "backtest_confirmation",
                            "backtest_complete",
                        }
                    ),
                }
                draft["agent_decision"] = {
                    "tool": route.get("agent_tool"),
                    "parameters": params,
                    "source": route.get("agent_source"),
                }
                if user_msg:
                    user_msg.strategy_draft = draft
                    user_msg.status = MessageStatus.completed

                assistant_msg = ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role=ChatRole.assistant,
                    content=assistant_text,
                    model=FLOW_MODEL_NAME,
                    strategy_draft=draft,
                    strategy_json=None,
                    status=MessageStatus.completed,
                    is_final=True,
                    parent_message_id=user_msg_uuid,
                )
                db.add(assistant_msg)
                await db.commit()
                return

            before_snapshot = _core_snapshot(builder)
            input_was_already_confirmed = builder.user_input_confirmed
            missing_fields_before_update = builder.missing_user_input_fields()
            user_confirmed = bool(route.get("is_confirmation")) or detect_user_confirmation(user_content)

            builder.clear_validation_state()

            parsed_builder = StrategyBuilder()
            extract_strategy_details(user_content, parsed_builder)

            relevant_message = bool(route.get("is_relevant"))
            recognized_fields = set(route.get("recognized_fields") or [])
            ignored_optional_input = _contains_ignored_optional_input(user_content)

            modification_fields = _extract_input_modification_fields(user_content)
            explicit_input_modification_request = _detect_input_modification_request(
                user_content,
                route_intent,
            )
            awaiting_modification_selection = (
                builder.input_modification_requested
                and not builder.pending_input_modification_fields
            )
            awaiting_modification_values = (
                builder.input_modification_requested
                and bool(builder.pending_input_modification_fields)
            )
            modification_selection_only = _is_input_modification_selection_only(
                user_content,
                modification_fields,
            )
            skip_field_extraction = False
            input_modification_invalid_selection = False
            input_modification_turn = bool(
                explicit_input_modification_request
                or awaiting_modification_selection
                or awaiting_modification_values
            )
            modification_value_fields: list[str] = []

            if awaiting_modification_selection:
                relevant_message = True
                # Escape hatch: if the user confirms, signals "no changes", or
                # asks a general/clarification question while we are still waiting
                # for them to pick a field, clear the modification request and let
                # the regular confirmation path proceed. Without this, any reply
                # that does not name one of the six fields traps the user in a
                # "Please choose one of the six strategy inputs..." loop.
                user_refused_modification = _is_input_modification_refusal(user_content)
                if (
                    (
                        user_confirmed
                        or user_refused_modification
                        or route_intent in {"confirmation", "general_chat", "clarification"}
                    )
                    and not modification_fields
                ):
                    builder.clear_input_modification_request()
                    awaiting_modification_selection = False
                    awaiting_modification_values = False
                    input_modification_turn = bool(explicit_input_modification_request)
                    if user_refused_modification:
                        # Treat explicit refusal as confirmation of the existing inputs.
                        user_confirmed = True
                    logger.info(
                        "🔁 chat_flow|event=modification_selection_cleared|session_id=%s"
                        "|reason=%s",
                        session_id,
                        "refusal" if user_refused_modification else (
                            "confirmation" if user_confirmed else "non_modification_intent"
                        ),
                    )
                elif modification_fields and modification_selection_only:
                    builder.request_input_modification(modification_fields)
                    skip_field_extraction = True
                elif modification_fields:
                    modification_value_fields = modification_fields
                else:
                    builder.request_input_modification()
                    skip_field_extraction = True
                    input_modification_invalid_selection = True
            elif explicit_input_modification_request:
                relevant_message = True
                if modification_fields and modification_selection_only:
                    builder.request_input_modification(modification_fields)
                    skip_field_extraction = True
                elif modification_fields:
                    modification_value_fields = modification_fields
                else:
                    builder.request_input_modification()
                    skip_field_extraction = True
            elif awaiting_modification_values:
                relevant_message = True
                modification_value_fields = list(builder.pending_input_modification_fields)

            if not skip_field_extraction:
                explicit_stock_query = _extract_explicit_stock_query(user_content)
                stock_query = explicit_stock_query
                if not stock_query and ("symbol" in recognized_fields or route_intent == "collect_input"):
                    stock_query = route.get("stock_query")
                if (
                    not stock_query
                    and builder.input_modification_requested
                    and "symbol" in builder.pending_input_modification_fields
                    and route_intent not in {"general_chat", "clarification"}
                ):
                    stock_query = user_content
                if not stock_query and _has_explicit_stock_cue(user_content):
                    stock_query = extract_company_name_query(user_content)
                if stock_query:
                    relevant_message = True
                    stock_match = await resolve_supported_stock(stock_query)
                    if stock_match and stock_match.get("symbol"):
                        resolved_symbol = str(stock_match["symbol"]).strip()
                        exchange_hint = extract_exchange_hint(user_content)
                        if exchange_hint and not resolved_symbol.upper().endswith((".NS", ".BO")):
                            resolved_symbol = f"{resolved_symbol}.{exchange_hint}"
                        builder.symbol = resolved_symbol
                        builder.set_symbol_validation(None)
                        logger.info(
                            "✅ chat_flow|event=stock_resolved|session_id=%s|query=%r|symbol=%s",
                            session_id,
                            stock_query,
                            resolved_symbol,
                        )
                    else:
                        builder.symbol = None
                        code, facts = unsupported_stock_validation()
                        builder.set_symbol_validation(code, facts)
                        logger.warning(
                            "❌ chat_flow|event=stock_not_found|session_id=%s|query=%r",
                            session_id,
                            stock_query,
                        )

                if builder.symbol and not is_supported_stock_symbol(builder.symbol):
                    builder.symbol = None
                    code, facts = unsupported_stock_validation()
                    builder.set_symbol_validation(code, facts)
                    relevant_message = True
                    logger.warning(
                        "❌ chat_flow|event=stock_unsupported|session_id=%s|symbol=%s",
                        session_id,
                        builder.symbol,
                    )

                timeframe_input = route.get("timeframe_input")
                if timeframe_input:
                    relevant_message = True
                    resolved_timeframe, validation_message = resolve_supported_user_timeframe(timeframe_input)
                    if resolved_timeframe:
                        builder.timeframe = resolved_timeframe
                        builder.set_timeframe_validation(None)
                        logger.info(
                            "✅ chat_flow|event=timeframe_resolved|session_id=%s|input=%r|resolved=%s",
                            session_id,
                            timeframe_input,
                            resolved_timeframe,
                        )
                    elif validation_message:
                        builder.timeframe = None
                        builder.set_timeframe_validation(
                            UNSUPPORTED_USER_TIMEFRAME_CODE,
                            unsupported_user_timeframe_validation_facts(),
                        )
                        logger.warning(
                            "❌ chat_flow|event=timeframe_unsupported|session_id=%s|input=%r",
                            session_id,
                            timeframe_input,
                        )
                elif parsed_builder.timeframe:
                    builder.timeframe = parsed_builder.timeframe
                    relevant_message = True
                elif parsed_builder.timeframe_validation_message:
                    builder.timeframe = None
                    builder.set_timeframe_validation(
                        parsed_builder.timeframe_validation_code,
                        parsed_builder.timeframe_validation_facts,
                        message=parsed_builder.timeframe_validation_message,
                    )
                    relevant_message = True

                sentiment = route.get("sentiment") or parsed_builder.sentiment
                if sentiment:
                    builder.sentiment = sentiment
                    relevant_message = True

                experience = route.get("experience") or parsed_builder.experience
                if experience:
                    builder.experience = experience
                    relevant_message = True

                objective = route.get("objective") or parsed_builder.objective
                if objective:
                    builder.objective = objective
                    relevant_message = True

                goal = route.get("goal") or parsed_builder.goal
                if not goal and missing_fields_before_update == ["goal"]:
                    fallback_goal = extract_goal_text(user_content)
                    if not fallback_goal:
                        compact_goal = re.sub(r"\s+", " ", user_content).strip(" \t\r\n.,!?-:")
                        if (
                            route_intent == "collect_input"
                            and not user_confirmed
                            and not recognized_fields
                            and not _is_greeting_message(user_content)
                            and len(compact_goal.split()) >= 4
                        ):
                            fallback_goal = compact_goal
                    goal = fallback_goal
                if goal:
                    builder.goal = re.sub(r"\s+", " ", str(goal)).strip()
                    relevant_message = True

                if modification_value_fields:
                    builder.input_modification_requested = True
                    builder.pending_input_modification_fields = [
                        field
                        for field in modification_value_fields
                        if field in CORE_USER_INPUT_FIELDS
                    ]
                    builder.refresh_pending_input_modification_fields()

            if ignored_optional_input:
                relevant_message = True

            after_snapshot = _core_snapshot(builder)
            snapshot_changed = after_snapshot != before_snapshot
            user_state = previous_state
            logger.info(
                "🏷️ chat_flow|event=fields_extracted|session_id=%s|confirmed=%s|snapshot_changed=%s"
                "|symbol=%s|timeframe=%s|sentiment=%s|experience=%s|objective=%s|goal=%r",
                session_id,
                user_confirmed,
                snapshot_changed,
                getattr(builder, "symbol", None),
                getattr(builder, "timeframe", None),
                getattr(builder, "sentiment", None),
                getattr(builder, "experience", None),
                getattr(builder, "objective", None),
                (getattr(builder, "goal", None) or "")[:60],
            )

            if chat_obj and _is_placeholder_title(chat_obj.title) and not _is_greeting_message(user_content):
                should_update_title = route_intent not in {"general_chat", "clarification"} or bool(builder.symbol)
                if should_update_title:
                    generated_title = _generate_chat_title_from_message(user_content, builder)
                    if generated_title:
                        chat_obj.title = generated_title
                        logger.info(
                            "💬 chat_flow|event=chat_title_updated|session_id=%s|title=%r",
                            session_id,
                            generated_title,
                        )

            reset_requires_reconfirmation = bool(
                input_modification_turn
                or input_was_already_confirmed
                or previous_state in {
                    "plan_signals",
                    "assemble_strategy",
                    "backtest_confirmation",
                    "backtest_complete",
                }
            )
            if snapshot_changed:
                builder.reset_generated_strategy_state()
                user_state = "collect_user_input"
                logger.info(
                    "🔁 chat_flow|event=state_reset|session_id=%s"
                    "|reason=input_changed|new_state=collect_user_input",
                    session_id,
                )
            elif input_modification_turn:
                builder.reset_generated_strategy_state()
                user_state = "collect_user_input"
                logger.info(
                    "🔁 chat_flow|event=state_reset|session_id=%s"
                    "|reason=input_modification_requested|new_state=collect_user_input",
                    session_id,
                )

            if reset_requires_reconfirmation and (snapshot_changed or input_modification_turn):
                user_confirmed = False

            if (
                route.get("needs_clarification")
                and not input_modification_turn
                and not builder.symbol_validation_message
                and not builder.timeframe_validation_message
                and not user_confirmed
            ):
                missing_fields = builder.missing_user_input_fields()
                clarification_field = route.get("clarification_field") or (
                    missing_fields[0] if missing_fields else None
                )
                builder.set_input_validation(
                    "validation.low_confidence_clarification",
                    {
                        "interpreted_values": _summarize_interpreted_values(builder, route, recognized_fields),
                        "missing_field": clarification_field,
                    },
                )
                logger.info(
                    "📋 chat_flow|event=clarification_requested|session_id=%s|field=%s",
                    session_id,
                    clarification_field,
                )

            if (
                route_intent == "invalid_value"
                and not input_modification_invalid_selection
                and not builder.symbol_validation_message
                and not builder.timeframe_validation_message
                and not builder.input_validation_message
                and not user_confirmed
            ):
                invalid_field = route.get("invalid_field")
                missing_fields = builder.missing_user_input_fields()
                builder.set_input_validation(
                    "validation.invalid_input",
                    {
                        "field_name": invalid_field or (missing_fields[0] if missing_fields else None),
                    },
                )
                logger.warning(
                    "📋 chat_flow|event=invalid_input|session_id=%s|field=%s",
                    session_id,
                    invalid_field or (missing_fields[0] if missing_fields else None),
                )
            elif (
                not builder.is_user_input_complete()
                and not builder.symbol_validation_message
                and not builder.timeframe_validation_message
                and not builder.input_validation_message
                and not relevant_message
                and not ignored_optional_input
                and not input_modification_invalid_selection
                and not user_confirmed
                and route_intent not in {"general_chat", "clarification"}
                and not _is_greeting_message(user_content)
            ):
                missing_fields = builder.missing_user_input_fields()
                builder.set_input_validation(
                    "validation.invalid_input",
                    {
                        "field_name": missing_fields[0] if missing_fields else None,
                    },
                )
                logger.info(
                    "📋 chat_flow|event=missing_field_prompt|session_id=%s|missing=%s",
                    session_id,
                    missing_fields,
                )

            if builder.is_user_input_complete():
                builder.apply_defaults()
                logger.info(
                    "✅ chat_flow|event=user_input_complete|session_id=%s"
                    "|symbol=%s|timeframe=%s|sentiment=%s|experience=%s|objective=%s",
                    session_id,
                    getattr(builder, "symbol", None),
                    getattr(builder, "timeframe", None),
                    getattr(builder, "sentiment", None),
                    getattr(builder, "experience", None),
                    getattr(builder, "objective", None),
                )
            if user_msg:
                user_msg.strategy_draft = builder.to_draft_json(
                    mode_override=user_state,
                    processing_status="received",
                )

            assistant_state = "collect_user_input"
            strategy_json_to_save = None
            strategy_id_to_save = None
            ref_backtest_id = None
            persisted_backtest_uuid: uuid.UUID | None = None

            if user_state in {"assemble_strategy", "backtest_confirmation"} and user_confirmed:
                _sid = (latest_strategy_context or {}).get("strategy_id")
                logger.info(
                    "📈 chat_flow|event=backtest_user_trigger|session_id=%s|user_state=%s|strategy_id=%s",
                    session_id,
                    user_state,
                    _sid,
                )
                try:
                    yaml_path = str((latest_strategy_context or {}).get("yaml_path") or "").strip()
                    if not yaml_path:
                        raise ValueError("The assembled strategy YAML could not be found for this session.")

                    # Build run request with objective and risk controls from strategy builder
                    _objective         = str(getattr(builder, "objective", None) or "positional").lower()
                    _daily_loss_cap    = float(getattr(builder, "daily_loss_cap", None) or 0.0)
                    _max_trades_per_day = _derive_max_trades_from_builder(builder)

                    run_request = BacktestTriggerRequest(
                        strategy_id=str((latest_strategy_context or {}).get("strategy_id") or uuid.uuid4()),
                        objective=_objective,
                        daily_loss_cap_pct=_daily_loss_cap,
                        max_trades_per_day=_max_trades_per_day,
                    )
                    logger.info(
                        "⚙️ chat_flow|event=backtest_run_config|session_id=%s"
                        "|objective=%s|daily_loss_cap=%.1f%%|max_trades=%d|yaml=%s",
                        session_id,
                        _objective,
                        _daily_loss_cap,
                        _max_trades_per_day,
                        yaml_path,
                    )
                    backtest_started_at = datetime.now(timezone.utc)
                    market_data_request = extract_strategy_market_data_request(yaml_path, overrides=run_request)
                    logger.info(
                        "📡 chat_flow|event=market_data_fetching|session_id=%s"
                        "|symbol=%s|interval=%s|from=%s|to=%s",
                        session_id,
                        market_data_request.symbol,
                        market_data_request.interval,
                        market_data_request.from_utc,
                        market_data_request.to_utc,
                    )
                    ohlcv_data = await fetch_ohlcv_records(market_data_request)
                    logger.info(
                        "📡 chat_flow|event=market_data_ready|session_id=%s"
                        "|symbol=%s|interval=%s|candles=%d",
                        session_id,
                        market_data_request.symbol,
                        market_data_request.interval,
                        len(ohlcv_data),
                    )

                    # Enrich signal parameters using actual price/volume statistics
                    # before running the backtest. This replaces universal YAML defaults
                    # with values calibrated to this specific stock and timeframe.
                    # Requires >= 60 bars; falls back to YAML defaults silently on any error.
                    if builder.signal_plan and len(ohlcv_data) >= 60:
                        logger.info(
                            "✨ chat_flow|event=signal_enrichment_start|session_id=%s"
                            "|bars=%d|signals=%s|objective=%s|timeframe=%s",
                            session_id,
                            len(ohlcv_data),
                            (builder.signal_plan or {}).get("signals_used", []),
                            builder.objective or "intraday",
                            builder.timeframe or "15m",
                        )
                        try:
                            enriched_plan = enrich_plan_with_ohlcv(
                                plan=builder.signal_plan,
                                ohlcv_records=ohlcv_data,
                                objective=builder.objective or "intraday",
                                timeframe=builder.timeframe or "15m",
                            )
                            builder.signal_plan = enriched_plan
                            builder.entry_condition = enriched_plan.get("entry_condition")
                            builder.exit_condition = enriched_plan.get("exit_condition")
                            yaml_path = generate_yaml(builder)
                            logger.info(
                                "✅ chat_flow|event=signal_params_enriched|session_id=%s"
                                "|bars=%d|yaml_path=%s|entry=%r|exit=%r",
                                session_id,
                                len(ohlcv_data),
                                yaml_path,
                                (enriched_plan.get("entry_condition") or "")[:80],
                                (enriched_plan.get("exit_condition") or "")[:80],
                            )
                        except Exception:
                            logger.warning(
                                "⚠️ chat_flow|event=signal_enrichment_failed|session_id=%s"
                                " — falling back to YAML defaults",
                                session_id,
                                exc_info=True,
                            )
                    elif builder.signal_plan:
                        logger.info(
                            "⚠️ chat_flow|event=signal_enrichment_skipped|session_id=%s"
                            "|reason=insufficient_bars|bars=%d|min_required=60",
                            session_id,
                            len(ohlcv_data),
                        )

                    logger.info(
                        "📈 chat_flow|event=backtest_running|session_id=%s|yaml=%s",
                        session_id,
                        yaml_path,
                    )
                    # Phase 7 — fetch reference / HTF OHLCV when the strategy
                    # declares them (Phases 4 / 5). Both are None for legacy
                    # strategies, so this is purely additive.
                    reference_ohlcv, htf_ohlcv = await fetch_auxiliary_ohlcv(market_data_request)
                    backtest_result = await run_quant_backtest_sync(
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
                    m = (backtest_result or {}).get("metrics") or {}
                    logger.info(
                        "🎯 chat_flow|event=backtest_complete|session_id=%s|engine_ref=%s"
                        "|pass=%s|total_trades=%s|win_rate=%s%%|total_return=%s%%",
                        session_id,
                        (backtest_result or {}).get("backtest_ref_id"),
                        (backtest_result or {}).get("pass"),
                        m.get("total_trades") if isinstance(m, dict) else None,
                        round(float(m.get("win_rate", 0) or 0), 1) if isinstance(m, dict) else None,
                        round(float(m.get("total_return_pct", 0) or 0), 2) if isinstance(m, dict) else None,
                    )
                    result_summary = summarize_backtest_for_db(backtest_result) or {}

                    # Feed backtest outcome back into the performance cache so future
                    # signal planning can use real win-rate data instead of YAML defaults.
                    if builder.signal_plan:
                        _bm = (backtest_result or {}).get("metrics") or {}
                        _wr = float(_bm.get("win_rate", 0) or 0) / 100.0
                        _tt = int(_bm.get("total_trades", 0) or 0)
                        _sym = builder.format_symbol() or ""
                        _tf  = builder.timeframe or ""
                        for _sig in (
                            builder.signal_plan.get("entry", [])
                            + builder.signal_plan.get("exit", [])
                        ):
                            record_performance(
                                signal_name=_sig.get("name", ""),
                                symbol=_sym,
                                timeframe=_tf,
                                params=_sig.get("params", {}),
                                win_rate=_wr,
                                total_trades=_tt,
                            )
                        logger.info(
                            "💾 chat_flow|event=signal_perf_recorded|session_id=%s"
                            "|symbol=%s|tf=%s|win_rate=%.1f%%|total_trades=%d|signals=%s",
                            session_id, _sym, _tf, _wr * 100, _tt,
                            [s.get("name") for s in (
                                builder.signal_plan.get("entry", [])
                                + builder.signal_plan.get("exit", [])
                            )],
                        )

                    if _sid:
                        try:
                            strategy_uuid = uuid.UUID(str(_sid))
                            persisted_backtest_uuid = await insert_chat_backtest_row(
                                db,
                                strategy_id=strategy_uuid,
                                summary=result_summary,
                                started_at=backtest_started_at,
                            )
                            logger.info(
                                "💾 chat_flow|event=backtest_db_saved|session_id=%s"
                                "|backtest_id=%s|strategy_id=%s",
                                session_id,
                                persisted_backtest_uuid,
                                strategy_uuid,
                            )
                        except Exception:
                            logger.exception(
                                "❌ chat_flow|event=backtest_db_save_failed|session_id=%s|strategy_id=%s",
                                session_id,
                                _sid,
                            )
                    else:
                        logger.warning(
                            "⚠️ chat_flow|event=backtest_db_skipped|reason=no_strategy_id|session_id=%s",
                            session_id,
                        )

                    assistant_text = build_backtest_result_reply(builder, backtest_result)
                    assistant_state = "backtest_complete"
                    # Full result (incl. metrics.backtest_trades) for API/clients; DB row uses result_summary only
                    strategy_json_to_save = {"backtest_result": backtest_result}
                    draft = builder.to_draft_json(
                        mode_override="backtest_complete",
                        processing_status="complete",
                    )
                    ref_backtest_id = str(backtest_result.get("backtest_ref_id") or "")
                    if ref_backtest_id:
                        draft["backtest_ref_id"] = ref_backtest_id
                except Exception as backtest_exc:
                    logger.error(
                        "❌ chat_flow|event=backtest_failed|session_id=%s|error=%s",
                        session_id,
                        str(backtest_exc)[:200],
                        exc_info=True,
                    )
                    if _sid:
                        try:
                            await insert_failed_chat_backtest(
                                db,
                                strategy_id=uuid.UUID(str(_sid)),
                                error_message=str(backtest_exc),
                            )
                        except Exception:
                            logger.exception(
                                "❌ chat_flow|event=backtest_failure_row_failed|session_id=%s|strategy_id=%s",
                                session_id,
                                _sid,
                            )
                    else:
                        logger.warning(
                            "⚠️ chat_flow|event=backtest_failure_unpersisted|session_id=%s|reason=no_strategy_id",
                            session_id,
                        )
                    assistant_text = build_backtest_error_reply(str(backtest_exc))
                    assistant_state = (
                        "assemble_strategy"
                        if user_state == "assemble_strategy"
                        else "backtest_confirmation"
                    )
                    draft = builder.to_draft_json(
                        mode_override=assistant_state,
                        processing_status=_draft_processing_status(builder, assistant_state),
                    )
                    draft["backtest_error"] = str(backtest_exc)
            elif builder.signal_plan and user_confirmed and previous_state in {"plan_signals", "assemble_strategy"}:
                logger.info(
                    "🏗️ chat_flow|event=strategy_assembly_start|session_id=%s"
                    "|symbol=%s|timeframe=%s|signals=%s",
                    session_id,
                    getattr(builder, "symbol", None),
                    getattr(builder, "timeframe", None),
                    (builder.signal_plan or {}).get("signals_used", []),
                )
                builder.apply_defaults()
                assembled_config_snapshot = await _upsert_builder_risk_execution_config(
                    db,
                    builder,
                    session_id=session_uuid,
                )
                builder.set_risk_execution_config(
                    assembled_config_snapshot.to_builder_context()
                )
                strategy_payload = build_strategy_object(builder)
                strategy_payload["strategy_object"]["risk_and_execution"] = (
                    build_risk_execution_response(assembled_config_snapshot)
                )
                strategy_config = build_strategy_config(builder, strategy_payload)
                stored_config = strategy_config or {
                    "name": f"{builder.format_symbol().lower()}_strategy",
                    "asset": builder.format_symbol(),
                    "entry": [],
                    "exit": [],
                    "reward_factor": round(float(builder.take_profit) / float(builder.stop_loss), 2)
                    if builder.stop_loss
                    else None,
                }
                yaml_path = generate_yaml(builder)
                logger.info(
                    "📝 chat_flow|event=strategy_yaml_written|session_id=%s|yaml_path=%s",
                    session_id,
                    yaml_path,
                )

                strategy_name = (
                    stored_config.get("name")
                    or f"{builder.format_symbol()} {builder.timeframe} Strategy"
                )
                strategy_symbol = builder.format_symbol()
                strategy_market = builder.market or "indian_stocks"
                strategy_timeframe = builder.timeframe or "1d"

                existing_strategy = None
                existing_strategy_id = (latest_strategy_context or {}).get("strategy_id")
                if existing_strategy_id:
                    try:
                        existing_strategy = await db.get(Strategy, uuid.UUID(str(existing_strategy_id)))
                    except (ValueError, TypeError, AttributeError):
                        existing_strategy = None

                if existing_strategy:
                    strategy = existing_strategy
                    strategy.name = strategy_name
                    strategy.symbol = strategy_symbol
                    strategy.market = strategy_market
                    strategy.timeframe = strategy_timeframe
                    strategy.status = StrategyStatus.confirmed
                    strategy.yaml_path = yaml_path
                    strategy.strategy_config = stored_config
                    logger.info(
                        "🔄 chat_flow|event=strategy_updated|session_id=%s|strategy_id=%s|name=%s",
                        session_id,
                        existing_strategy.id,
                        strategy_name,
                    )
                else:
                    strategy = Strategy(
                        id=uuid.uuid4(),
                        session_id=session_uuid,
                        name=strategy_name,
                        symbol=strategy_symbol,
                        market=strategy_market,
                        timeframe=strategy_timeframe,
                        status=StrategyStatus.confirmed,
                        yaml_path=yaml_path,
                        strategy_config=stored_config,
                    )
                    db.add(strategy)

                await db.flush()
                strategy_id_to_save = strategy.id
                latest_config_snapshot = await _upsert_builder_risk_execution_config(
                    db,
                    builder,
                    session_id=session_uuid,
                    strategy_id=str(strategy.id),
                )
                strategy_payload["strategy_object"]["risk_and_execution"] = (
                    build_risk_execution_response(latest_config_snapshot)
                )
                logger.info(
                    "💾 chat_flow|event=strategy_persisted|session_id=%s"
                    "|strategy_id=%s|name=%s|symbol=%s|timeframe=%s|yaml_path=%s|status=%s",
                    session_id,
                    strategy.id,
                    strategy_name,
                    strategy_symbol,
                    strategy_timeframe,
                    strategy.yaml_path,
                    strategy.status.value,
                )

                assistant_text = build_assemble_strategy_reply(builder, strategy_config)
                strategy_json_to_save = build_final_strategy_payload(
                    session_id=session_id,
                    user_id=str(chat_obj.user_id) if chat_obj and chat_obj.user_id else None,
                    strategy_object=strategy_payload["strategy_object"],
                    strategy_config=stored_config,
                    strategy_id=str(strategy.id),
                    yaml_path=yaml_path,
                    current_mode="assemble_strategy",
                    next_state="backtest_confirmation",
                )
                assistant_state = "assemble_strategy"
                draft = builder.to_draft_json(
                    mode_override="assemble_strategy",
                    processing_status=_draft_processing_status(builder, "assemble_strategy"),
                )
                retrieval_meta = build_retrieval_meta(strategy_payload)
                draft["kb_signals_used"] = retrieval_meta.get("signals_used", [])
                draft["kb_signals_available"] = retrieval_meta.get("signals_available", 0)
            else:
                if builder.is_user_input_complete() and not builder.user_input_confirmed and user_confirmed:
                    builder.user_input_confirmed = True

                _structured_clarification_topics = {
                    "tutorial",
                    "onboarding",
                    "purpose_overview",
                    "capability_examples",
                    "ambiguous",
                    "status",
                    "collected_inputs",
                    "signal_plan",
                    "strategy",
                    "backtest",
                    "next_step",
                }
                _clarification_topic = route.get("clarification_topic")
                _confidence = (route.get("confidence") or "high").strip().lower()

                # Low-confidence safety net: if the router was unsure and we did
                # not pick up any concrete strategy field from the message, force
                # the ambiguous-clarification flow so the user gets a structured
                # "what would you like to do?" reply instead of a confidently-
                # wrong free-form LLM message.
                if (
                    _confidence == "low"
                    and not recognized_fields
                    and not user_confirmed
                    and not input_modification_turn
                    and not builder.symbol_validation_message
                    and not builder.timeframe_validation_message
                    and not builder.input_validation_message
                ):
                    _clarification_topic = _clarification_topic or "ambiguous"
                    route_intent = "clarification"

                grounded_clarification_reply = None
                if (
                    (
                        route_intent == "clarification"
                        or _clarification_topic in _structured_clarification_topics
                    )
                    and not input_modification_turn
                    and not user_confirmed
                    and not builder.symbol_validation_message
                    and not builder.timeframe_validation_message
                    and not builder.input_validation_message
                ):
                    grounded_clarification_reply = build_grounded_clarification_reply(
                        builder,
                        current_state=user_state,
                        clarification_topic=_clarification_topic,
                        latest_backtest_result=latest_backtest_result,
                        latest_strategy_context=latest_strategy_context,
                    )

                direct_llm_reply = None
                if not input_modification_turn:
                    direct_llm_reply = select_direct_llm_reply(
                        route_intent=route_intent,
                        route_reply_text=route_reply_text,
                        user_confirmed=user_confirmed,
                        recognized_fields=recognized_fields,
                        builder=builder,
                        snapshot_changed=snapshot_changed,
                    )

                if input_modification_invalid_selection:
                    assistant_text = build_input_modification_invalid_field_reply()
                    assistant_state = "collect_user_input"
                    draft = builder.to_draft_json(
                        mode_override="collect_user_input",
                        processing_status=_draft_processing_status(builder, "collect_user_input"),
                    )
                    logger.info(
                        "📋 chat_flow|event=input_modification_field_prompt_repeated|session_id=%s",
                        session_id,
                    )
                elif grounded_clarification_reply:
                    builder.set_input_validation(None)
                    assistant_text = grounded_clarification_reply
                    assistant_state = user_state
                    draft = builder.to_draft_json(
                        mode_override=assistant_state,
                        processing_status=_draft_processing_status(builder, assistant_state),
                    )
                    if isinstance(draft, dict):
                        draft["router_reasoning"] = route.get("reasoning")
                        draft["router_confidence"] = _confidence
                        draft["clarification_topic"] = _clarification_topic
                    logger.info(
                        "💬 chat_flow|event=clarification_reply_sent|session_id=%s|topic=%s|confidence=%s",
                        session_id,
                        _clarification_topic,
                        _confidence,
                    )
                elif direct_llm_reply:
                    builder.set_input_validation(None)
                    assistant_text = direct_llm_reply
                    assistant_state = user_state
                    draft = builder.to_draft_json(
                        mode_override=assistant_state,
                        processing_status=_draft_processing_status(builder, assistant_state),
                    )
                    logger.info(
                        "💬 chat_flow|event=direct_llm_reply_used|session_id=%s|intent=%s",
                        session_id,
                        route_intent,
                    )
                elif user_state in {"assemble_strategy", "backtest_confirmation"}:
                    assistant_text = build_backtest_ready_reminder()
                    assistant_state = (
                        "assemble_strategy"
                        if user_state == "assemble_strategy"
                        else "backtest_confirmation"
                    )
                    draft = builder.to_draft_json(
                        mode_override=assistant_state,
                        processing_status=_draft_processing_status(builder, assistant_state),
                    )
                    logger.info(
                        "⏳ chat_flow|event=backtest_reminder_sent|session_id=%s|user_state=%s",
                        session_id,
                        user_state,
                    )
                elif user_state == "backtest_complete":
                    assistant_text = compose_assistant_response(
                        "workflow.backtest_already_available"
                    )
                    assistant_state = "backtest_complete"
                    draft = builder.to_draft_json(
                        mode_override="backtest_complete",
                        processing_status="complete",
                    )
                    logger.info(
                        "⏳ chat_flow|event=backtest_already_done_reply|session_id=%s",
                        session_id,
                    )
                elif not builder.is_user_input_complete() or not builder.user_input_confirmed:
                    capture_ack = _build_capture_acknowledgement(
                        builder,
                        before_snapshot,
                        after_snapshot,
                        was_modification=input_modification_turn,
                        turn_index=len(all_messages),
                    )
                    assistant_text = build_collect_user_input_reply(
                        builder,
                        user_content,
                        preface=capture_ack,
                    )
                    assistant_state = "collect_user_input"
                    draft = builder.to_draft_json(
                        mode_override="collect_user_input",
                        processing_status=_draft_processing_status(builder, "collect_user_input"),
                    )
                    logger.info(
                        "📋 chat_flow|event=collecting_input|session_id=%s|missing=%s|ack=%s",
                        session_id,
                        builder.missing_user_input_fields(),
                        bool(capture_ack),
                    )
                elif not builder.signal_plan:
                    builder.apply_defaults()
                    logger.info(
                        "📊 chat_flow|event=signal_planning_start|session_id=%s"
                        "|symbol=%s|goal=%r|sentiment=%s|objective=%s|timeframe=%s",
                        session_id,
                        getattr(builder, "symbol", None),
                        (getattr(builder, "goal", None) or "")[:60],
                        getattr(builder, "sentiment", None),
                        getattr(builder, "objective", None),
                        getattr(builder, "timeframe", None),
                    )

                    # ── Fetch OHLCV early so params can be calibrated before ──────
                    # showing the signal plan to the user — no YAML needed yet,
                    # we build the market-data request directly from the builder.
                    _planning_ohlcv: list | None = None
                    if builder.symbol and builder.timeframe:
                        try:
                            _from, _to = _signal_eval_window(interval=builder.timeframe)
                            _planning_mdr = StrategyMarketDataRequest(
                                yaml_path="",
                                raw_symbol=builder.symbol,
                                symbol=_normalize_symbol_for_market_data(builder.symbol),
                                interval=builder.timeframe,
                                from_utc=_from,
                                to_utc=_to,
                            )
                            logger.info(
                                "📡 chat_flow|event=planning_ohlcv_fetching|session_id=%s"
                                "|symbol=%s|tf=%s|from=%s|to=%s",
                                session_id,
                                _planning_mdr.symbol,
                                _planning_mdr.interval,
                                _from,
                                _to,
                            )
                            _planning_ohlcv = await fetch_ohlcv_records(_planning_mdr)
                            logger.info(
                                "📡 chat_flow|event=planning_ohlcv_ready|session_id=%s"
                                "|symbol=%s|tf=%s|bars=%d",
                                session_id,
                                _planning_mdr.symbol,
                                _planning_mdr.interval,
                                len(_planning_ohlcv),
                            )
                        except Exception:
                            logger.warning(
                                "⚠️ chat_flow|event=planning_ohlcv_failed|session_id=%s"
                                " — planner will fall back to card defaults for params and SL/TP",
                                session_id,
                                exc_info=True,
                            )

                    logger.info(
                        "🧠 chat_flow|event=planner_start|session_id=%s"
                        "|symbol=%s|tf=%s|sentiment=%s|experience=%s|goal=%r|ohlcv_bars=%s",
                        session_id,
                        builder.symbol,
                        builder.timeframe,
                        builder.sentiment,
                        builder.experience,
                        (builder.goal or "")[:60],
                        len(_planning_ohlcv) if _planning_ohlcv else 0,
                    )
                    try:
                        plan = await plan_signals_v2(
                            builder,
                            ohlcv_records=_planning_ohlcv,
                            session_id=session_id,
                        )
                    except (UnsupportedStock, UnsupportedTimeframe, NoValidCandidate) as exc:
                        logger.error(
                            "❌ chat_flow|event=planner_failed|session_id=%s|err=%s",
                            session_id, exc,
                        )
                        raise

                    logger.info(
                        "✅ chat_flow|event=signal_planning_done|session_id=%s"
                        "|signals=%s|available=%d|sl=%s%%|tp=%s%%|entry=%r|exit=%r",
                        session_id,
                        plan.get("signals_used", []),
                        plan.get("signals_available", 0),
                        plan.get("_sl_pct"),
                        plan.get("_tp_pct"),
                        (plan.get("entry_condition") or "")[:80],
                        (plan.get("exit_condition") or "")[:80],
                    )
                    builder.apply_signal_plan(plan)
                    assistant_text = build_plan_signals_reply(builder, plan)
                    assistant_state = "plan_signals"
                    draft = builder.to_draft_json(
                        mode_override="plan_signals",
                        processing_status="awaiting_confirmation",
                    )
                    draft["kb_signals_used"] = plan.get("signals_used", [])
                    draft["kb_signals_available"] = plan.get("signals_available", 0)
                else:
                    assistant_text = build_plan_signals_reminder(builder)
                    assistant_state = "plan_signals"
                    draft = builder.to_draft_json(
                        mode_override="plan_signals",
                        processing_status="awaiting_confirmation",
                    )
                    logger.info(
                        "⏳ chat_flow|event=signal_plan_reminder_sent|session_id=%s",
                        session_id,
                    )

            if isinstance(draft, dict):
                draft.setdefault(
                    "agent_decision",
                    {
                        "tool": route.get("agent_tool"),
                        "parameters": route.get("agent_tool_parameters") or {},
                        "source": route.get("agent_source"),
                    },
                )
                assistant_text = _prefer_agent_question_text(route, assistant_text)

            # Loop detection: if we are about to send the exact same text the bot
            # has already sent on its last few turns, the user is stuck. Rephrase
            # and offer concrete options instead of repeating verbatim.
            recent_assistant_texts = _recent_assistant_texts(
                all_messages, _REPEAT_LOOP_THRESHOLD
            )
            if _is_repeat_loop(assistant_text, recent_assistant_texts):
                logger.warning(
                    "🔁 chat_flow|event=repeat_loop_detected|session_id=%s"
                    "|prior_count=%d|state=%s",
                    session_id,
                    len(recent_assistant_texts),
                    assistant_state,
                )
                assistant_text = (
                    "It looks like we are going in circles, so let me reset the question.\n\n"
                    f"{assistant_text}\n\n"
                    "If you would like to keep the current inputs and proceed, reply 'proceed'. "
                    "If you want to change something, name the field — for example "
                    "'change timeframe to 5m' or 'change stock to TCS'. "
                    "If something else is on your mind, please tell me in your own words."
                )
                if isinstance(draft, dict):
                    draft["loop_recovery_applied"] = True

            if user_msg:
                user_msg.status = MessageStatus.completed

            assistant_msg = ChatMessage(
                id=uuid.uuid4(),
                session_id=session_uuid,
                role=ChatRole.assistant,
                content=assistant_text,
                model=FLOW_MODEL_NAME,
                strategy_draft=draft,
                strategy_json=strategy_json_to_save,
                status=MessageStatus.completed,
                is_final=True,
                parent_message_id=user_msg_uuid,
                strategy_id=strategy_id_to_save,
                backtest_id=persisted_backtest_uuid,
                ref_backtest_id=ref_backtest_id,
            )
            db.add(assistant_msg)

            await db.commit()
            logger.info(
                "🎉 chat_flow|event=processing_complete|session_id=%s|message_id=%s|final_state=%s",
                session_id,
                user_message_id,
                assistant_state,
            )

        except asyncio.CancelledError:
            logger.info(
                "⏹️ chat_flow|event=processing_cancelled|session_id=%s|message_id=%s",
                session_id,
                user_message_id,
            )
            try:
                await db.rollback()
            except Exception:
                pass
            async with AsyncSessionLocal() as cancel_db:
                try:
                    cancelled_user_msg = await cancel_db.get(ChatMessage, user_msg_uuid)
                    if (
                        cancelled_user_msg
                        and cancelled_user_msg.status == MessageStatus.processing
                    ):
                        cancelled_user_msg.status = MessageStatus.completed
                        await cancel_db.commit()
                except Exception:
                    await cancel_db.rollback()
            raise

        except Exception as exc:
            logger.exception(
                "💥 chat_flow|event=processing_failed|session_id=%s|message_id=%s|error=%s",
                session_id,
                user_message_id,
                str(exc)[:200],
            )
            app_error = normalize_exception(exc)
            await db.rollback()
            async with AsyncSessionLocal() as err_db:
                try:
                    err_user_msg = await err_db.get(ChatMessage, user_msg_uuid)
                    if err_user_msg:
                        err_user_msg.status = MessageStatus.failed
                    err_db.add(
                        ChatMessage(
                            id=uuid.uuid4(),
                            session_id=session_uuid,
                            role=ChatRole.assistant,
                            content=app_error.message,
                            status=MessageStatus.failed,
                            error_message=app_error.message,
                            strategy_json=build_error_content(
                                app_error.status_code,
                                app_error.message,
                            ),
                            is_final=True,
                            parent_message_id=user_msg_uuid,
                        )
                    )
                    await err_db.commit()
                except Exception:
                    await err_db.rollback()
