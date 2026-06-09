"""
ChatService — the main orchestration layer for the strategy chat flow.
"""
from __future__ import annotations

import asyncio
import json
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
from app.services.ai.system_prompt import SYSTEM_INSTRUCTION, build_system_instruction
from app.services.backtest import (
    INTRABAR_EXECUTION_INTERVAL,
    build_main_fetch_request,
    extract_strategy_market_data_request,
    fetch_auxiliary_ohlcv,
    fetch_ohlcv_records,
    insert_chat_backtest_row,
    insert_failed_chat_backtest,
    resolve_intrabar_execution,
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
    build_backtest_earliest_date_reply,
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
from app.kb.compat import AMBIGUOUS_STOCK_VALIDATION_CODE, resolve_supported_stock
from app.planner.legacy_bridge import (
    NoValidCandidate,
    UnsupportedStock,
    UnsupportedTimeframe,
    plan_signals_v2,
)
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.constraint_compiler import (
    apply_semantic_constraints,
    needs_manual_review,
)
from app.planner.execution_orchestrator import ExecutionOrchestrator
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
    build_risk_and_execution_from_builder,
    build_risk_execution_response,
    sync_builder_risk_from_state,
    compose_risk_execution_values,
    resolve_active_risk_execution_config,
    upsert_risk_execution_config,
)
from app.services.strategy.builder import (
    CORE_USER_INPUT_FIELDS,
    MARKET_CONFIG,
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
from app.core.token_tracker import log_token_summary
# Phase 9b — discovery integration helpers (thin wrappers around the
# orchestrator that return a DiscoveryStepResult so the integration here
# stays a small if-block instead of a state machine).
from app.services.discovery.chat_integration import (
    handle_pending_tie_break,
    maybe_dispatch_discovery,
)

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


def _format_signal_plan_lines(plan: dict[str, Any] | None) -> str:
    """Render the trader-facing summary of a signal_plan dict."""
    plan = plan or {}
    entry_items = [
        str(s.get("name"))
        for s in plan.get("entry") or []
        if isinstance(s, dict) and s.get("name")
    ]
    exit_items = [
        str(s.get("name"))
        for s in plan.get("exit") or []
        if isinstance(s, dict) and s.get("name")
    ]
    entry_line = ", ".join(entry_items) if entry_items else "(none)"
    exit_line = ", ".join(exit_items) if exit_items else "(none)"
    return f"- Entry: {entry_line}\n- Exit: {exit_line}"


def _format_candidates(candidates: tuple[str, ...] | list[str] | None) -> str:
    """Pretty-print a short list of suggested signal names."""
    items = list(candidates or [])
    if not items:
        return ""
    shown = items[:6]
    return "Suggestions: " + ", ".join(shown) + (" …" if len(items) > 6 else "")


def _llm_hallucinated_signal_names(
    changes: list[dict[str, Any]], user_content: str
) -> bool:
    """
    True when the LLM filled `changes` with `signal_name` values that the
    trader never typed — e.g. the user said "change signals" and the LLM
    guessed "macd" / "rsi". We detect this by checking whether the family
    prefix of every guessed signal_name (e.g. "macd_positive" → "macd")
    appears in the user message. If none do, the LLM hallucinated and we
    should surface the suggestion list instead of running the resolver.
    """
    if not changes:
        return False
    haystack = (user_content or "").lower()
    for change in changes:
        signal_name = str(change.get("signal_name") or "").strip().lower()
        if not signal_name:
            continue
        # Match either the full name or its family prefix anywhere in the
        # user's message — handles "macd", "macd_positive", "rsi_oversold".
        family = signal_name.split("_", 1)[0]
        if signal_name in haystack or (family and family in haystack):
            return False
    return True


def _parse_rms_agent_value(raw: Any) -> tuple[float | None, str]:
    """Parse RMS tool values: plain numbers or ``{"value": N, "source": "user"}``."""
    if raw is None:
        return None, "user"
    if isinstance(raw, dict):
        try:
            val = float(raw["value"])
        except (KeyError, TypeError, ValueError):
            return None, str(raw.get("source") or "user")
        return val, str(raw.get("source") or "user")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, "user"
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return _parse_rms_agent_value(parsed)
        except json.JSONDecodeError:
            pass
        try:
            return float(text.rstrip("%").strip()), "user"
        except ValueError:
            return None, "user"
    try:
        return float(raw), "user"
    except (TypeError, ValueError):
        return None, "user"


def _coerce_agent_float_param(params: dict[str, Any], key: str) -> float | None:
    if params.get(key) is None:
        return None
    value, _ = _parse_rms_agent_value(params[key])
    if value is None:
        value = float(params[key])
    if key == "daily_loss_cap":
        if value < 0:
            raise ValueError("daily_loss_cap cannot be negative.")
    elif value <= 0:
        raise ValueError(f"{key} must be greater than 0.")
    return value


def _apply_agent_risk_param_overrides(
    builder: "StrategyBuilder",
    params: dict[str, Any],
) -> bool:
    """Apply agent ``stop_loss`` / ``take_profit`` objects onto builder + risk config."""
    attr_by_rms_key = {
        "stop_loss_pct": "stop_loss",
        "take_profit_pct": "take_profit",
    }
    alias_to_rms_key = (
        ("stop_loss", "stop_loss_pct"),
        ("take_profit", "take_profit_pct"),
        ("stop_loss_pct", "stop_loss_pct"),
        ("take_profit_pct", "take_profit_pct"),
    )
    existing_rms = dict(builder.risk_execution_config or {})
    existing_sources = dict(existing_rms.get("rms_sources", {}))
    changed = False
    for agent_key, rms_key in alias_to_rms_key:
        if agent_key not in params:
            continue
        val, source = _parse_rms_agent_value(params[agent_key])
        if val is None or val <= 0:
            continue
        existing_rms[rms_key] = float(val)
        existing_sources[rms_key] = source or "user"
        attr = attr_by_rms_key.get(rms_key)
        if attr:
            setattr(builder, attr, float(val))
            builder.mark_phase10_user_override(attr)
        changed = True
    if changed:
        existing_rms["rms_sources"] = existing_sources
        builder.risk_execution_config = existing_rms
    return changed


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
    rms_aliases = (
        ("stop_loss_pct", "stop_loss_pct"),
        ("take_profit_pct", "take_profit_pct"),
        ("stop_loss", "stop_loss_pct"),
        ("take_profit", "take_profit_pct"),
        ("daily_loss_cap", "daily_loss_cap"),
        ("per_trade_risk", "per_trade_risk"),
    )
    for agent_key, cfg_key in rms_aliases:
        if params.get(agent_key) is None:
            continue
        if agent_key in ("daily_loss_cap", "per_trade_risk"):
            value = _coerce_agent_float_param(params, agent_key)
            source = "user"
        else:
            value, source = _parse_rms_agent_value(params[agent_key])
            if value is None or value <= 0:
                continue
        if value is not None:
            merged[cfg_key] = value
            sources = dict(merged.get("rms_sources") or {})
            sources[cfg_key] = source
            merged["rms_sources"] = sources
            if cfg_key not in changed:
                changed.append(cfg_key)

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


def _enabled_asset_classes(capabilities: Optional[dict]) -> list[str]:
    """Extract asset_class_id strings of enabled capabilities. Empty list when
    capabilities is None / missing / contains only disabled entries."""
    return [
        item.get("asset_class_id")
        for item in (capabilities or {}).get("asset_classes", [])
        if item.get("enabled") and item.get("asset_class_id")
    ]


async def create_chat_session(
    db: AsyncSession,
    title: Optional[str] = None,
    capabilities: Optional[dict] = None,
) -> Chat:
    chat = Chat(
        id=uuid.uuid4(),
        title=title or "New Strategy",
        capabilities=capabilities,
    )
    db.add(chat)
    await db.flush()

    enabled_asset_classes = _enabled_asset_classes(capabilities)

    db.add(
        ChatMessage(
            id=uuid.uuid4(),
            session_id=chat.id,
            role=ChatRole.system,
            content=build_system_instruction(enabled_asset_classes),
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
            content=build_welcome_message(enabled_asset_classes),
            model=FLOW_MODEL_NAME,
            strategy_draft=welcome_builder.to_draft_json(
                mode_override="collect_user_input",
                processing_status="awaiting_user_input",
            ),
            status=MessageStatus.completed,
            is_final=True,
        )
    )
    
    # Log token summary at session start
    log_token_summary(str(chat.id), "session_start")
    
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


def _resolve_strategy_source_prompt(
    builder: "StrategyBuilder",
    user_content: str,
    all_messages: list[ChatMessage] | None = None,
) -> str:
    """Return the best available representation of the user's literal strategy
    description for downstream extractors/validators.

    Priority order:
        1. builder.original_user_prompt — set on first substantive turn and
           persisted across turns via to_draft_json/merge_preview.
        2. First substantive user message in chat history (≥ 6 words, not a
           greeting, not a confirmation reply). Defends against the case
           where original_user_prompt was lost / never set.
        3. The current user_content.
        4. builder.goal — the (lossy) agent-summarized version.

    Always returns a string (possibly empty); never None.
    """
    raw = (getattr(builder, "original_user_prompt", None) or "").strip()
    if raw and len(raw.split()) >= 6:
        return raw

    if all_messages:
        for msg in all_messages:
            if getattr(msg, "role", None) != ChatRole.user:
                continue
            content = (msg.content or "").strip()
            if not content:
                continue
            if _is_greeting_message(content):
                continue
            if detect_user_confirmation(content):
                continue
            if len(content.split()) >= 6:
                return content

    current = (user_content or "").strip()
    if current and len(current.split()) >= 6 and not detect_user_confirmation(current):
        return current

    return (getattr(builder, "goal", None) or current or "").strip()


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
    from app.services.execution.risk_execution_config_service import (
        build_risk_execution_values_from_builder,
    )

    return build_risk_execution_values_from_builder(builder, base_config)


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
    # User SL/TP on builder always win over hydrated DB defaults.
    if getattr(builder, "stop_loss", None) is not None:
        context["stop_loss_pct"] = float(builder.stop_loss)
    if getattr(builder, "take_profit", None) is not None:
        context["take_profit_pct"] = float(builder.take_profit)
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


# Phase 9g — generic English words that must NEVER be treated as
# tickers. They show up in natural-language discovery prompts like
# "create a strategy on NSE stock..." and would otherwise short-
# circuit the chat flow with a spurious "unsupported stock" error
# before discovery dispatches.
_NON_TICKER_WORDS = frozenset({
    "strategy", "strategies", "stock", "stocks", "share", "shares",
    "trade", "trades", "trading", "company", "companies", "equity",
    "equities", "security", "securities", "market", "markets",
    "setup", "system", "position", "intraday", "swing", "scalp",
    "scalping", "future", "futures", "option", "options",
    "any", "some", "all", "the", "this", "that", "these", "those",
    "investment", "investments", "portfolio", "asset", "assets",
    "anything", "something", "everything", "an",
})


# Generic nouns that follow "NSE"/"BSE" when the user is describing
# market scope, not a specific ticker (e.g. "...on NSE stock", "...in
# BSE company"). Used to suppress the natural-exchange regex match.
_NSE_SCOPE_NOUN_RE = re.compile(
    r"\s+(?:stock|stocks|share|shares|equity|equities|securit\w*|company|"
    r"companies|listed|firm|firms|ticker|tickers)\b",
    re.IGNORECASE,
)


# Phase 9k — primitive extractors for compositional discovery. Each
# function detects ONE concept (volume spike, RSI threshold, pullback,
# VWAP confirmation, …) in the user's prose and returns either None or
# a `{name, params}` dict. The aggregate _extract_discovery_conditions
# helper runs them all and returns the ordered list. The orchestrator
# then renders that list into AST conditions and runs the scanner with
# EXACTLY those constraints — nothing implicit, nothing the user
# didn't ask for.

# Phase 9h — extract user-typed discovery thresholds from prose. The
# orchestrator then substitutes them into the preset's condition
# placeholders so the user's "1.2x volume" actually lowers the scanner
# threshold (instead of the preset's hardcoded 2x being applied
# regardless of what the user typed).
_VOLUME_MULTIPLIER_RE = re.compile(
    # Catches: "1.5x", "2X", "3×", "1.2 x", "above 2x", "more than 1.5x",
    # "2 times", "above 3 times", "2x higher", "3x average", "3x normal".
    r"\b(\d+(?:\.\d+)?)\s*(?:[xX×]|times?)\b",
)
_NEAR_52W_HIGH_PCT_RE = re.compile(
    r"\bwithin\s+(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?"
    r"52[-\s]?week[s]?[-\s]?high\b",
    re.IGNORECASE,
)
_NEAR_52W_LOW_PCT_RE = re.compile(
    r"\bwithin\s+(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?"
    r"52[-\s]?week[s]?[-\s]?low\b",
    re.IGNORECASE,
)
_VOLUME_CONTEXT_RE = re.compile(
    r"\b(?:volume|spike|relative\s+volume|rel\s*vol)\b",
    re.IGNORECASE,
)

# Phase 9i — the high/low window is tunable. Users phrase it as
# "20-day high", "10 week breakdown", "6-month breakout", "1-year high".
# We only honor the match when the surrounding context mentions
# high/low/breakout/breakdown/verge so a bare "5 days" in unrelated
# prose can't accidentally set the window.
_LOOKBACK_WINDOW_RE = re.compile(
    r"\b(\d+)[-\s]+(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)
_LOOKBACK_CONTEXT_RE = re.compile(
    r"\b(?:high|low|breakout|breakdown|breaking|verge)\b",
    re.IGNORECASE,
)
_BARS_PER_UNIT: dict[str, int] = {
    # Trading-day conversions used to map calendar units to scan bars
    # (the scanner runs on the preset's `scan_timeframe`, which is
    # daily for volume_breakout_52w).
    "day":    1, "days":   1,
    "week":   5, "weeks":  5,
    "month":  21, "months": 21,
    "year":   252, "years": 252,
}


# Issue 2 — recognise messages where the user explicitly narrows the
# discovery universe to a hand-picked stock list, e.g.:
#   "choose among these stocks: TCS, Infosys, Reliance"
#   "scan only HDFC Bank, ICICI, SBI"
#   "from these stocks - TCS, INFY which matches"
# The trigger phrase must appear; otherwise stray symbol mentions in
# a prose message (e.g. "RELIANCE did 1.8× volume yesterday") won't
# silently constrain the scan.
_STOCK_LIST_TRIGGER_RE = re.compile(
    r"\b("
    r"choose\s+(?:among|from|between)"
    r"|pick\s+(?:from|among|between)"
    r"|select\s+(?:from|among|between)"
    r"|scan\s+only"
    r"|from\s+these\s+stocks"
    r"|among\s+these\s+stocks"
    r"|restrict\s+to"
    r"|limit\s+(?:to|the\s+scan\s+to)"
    r"|only\s+these\s+stocks"
    r")\b",
    re.IGNORECASE,
)

# Separators that can sit between list items. Hyphen and colon require
# surrounding whitespace so company names containing them ("L&T",
# "Larsen & Toubro", a future ":NS" suffix) aren't split apart — except
# colon is also allowed without a leading space because users often
# type "stocks: TCS, INFY" with no space before the colon.
_LIST_ITEM_SPLIT_RE = re.compile(
    r"\s*(?:,|/|\bor\b|\band\b|;|•|\||\s-\s|\s:\s|:\s+|^-\s|\s-$)\s*",
    re.IGNORECASE,
)


def _extract_user_supplied_stock_list(message: str) -> list[str] | None:
    """Detect a user-supplied stock-list narrowing and resolve each name
    to a canonical KB symbol (with the .NS suffix the scanner expects).

    Returns:
      list[str]  — when the trigger phrase matched AND at least one
                   name resolved to a known KB stock.
      None       — when no trigger phrase fired or nothing resolved.
                   The caller treats None as "don't touch the override"
                   so a prior turn's narrowing stays sticky.
    """
    if not message:
        return None
    if not _STOCK_LIST_TRIGGER_RE.search(message):
        return None

    # Lazy import of the KB so this module stays light at import time
    # and we don't pay for the KB load when the trigger phrase never
    # fires. (Bypasses the legacy app.services.knowledge.stock_matcher
    # module which is currently broken — its top-level imports a
    # non-existent `embedder` submodule.)
    try:
        from app.kb import kb as _kb
    except Exception:
        return None

    # Take the slice of text from the first trigger phrase to the end;
    # the list almost always sits after the trigger ("choose among
    # these stocks: A, B, C ...").
    match = _STOCK_LIST_TRIGGER_RE.search(message)
    tail = message[match.end():] if match else message

    # Drop trailing clauses that change the scope back ("...which
    # matches above condition and timeframe should be 1min"). NOTE:
    # ':' is NOT a stop phrase — users often type "from these stocks:
    # TCS, INFY" where the colon is the LIST INTRODUCER, not a
    # boundary. The split regex treats ':' as a separator instead.
    for stop_phrase in (
        " which ", " that ", " and timeframe", " and tf",
        " for the ", " with the ", " whose ", "\n",
    ):
        idx = tail.lower().find(stop_phrase)
        if 0 < idx < len(tail):
            tail = tail[:idx]

    # Strip leading punctuation/separators.
    tail = re.sub(r"^[\s\-:–—,]+", "", tail).strip()
    if not tail:
        return None

    raw_items = [item.strip(" .'\"`-") for item in _LIST_ITEM_SPLIT_RE.split(tail)]
    raw_items = [item for item in raw_items if item]
    if not raw_items:
        return None

    resolved: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        # kb.lookup_stock handles direct symbol match, manual aliases
        # ("hdfc bank"), and auto-derived aliases ("infosys" → INFY.NS).
        # Items that don't resolve are silently skipped — we can't scan
        # a stock the KB doesn't know about, and the user already typed
        # other items that did resolve.
        stock = _kb.lookup_stock(item)
        if stock is None:
            continue
        canonical = stock.symbol
        if canonical and canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)

    return resolved or None


def _extract_discovery_parameter_overrides(message: str) -> dict[str, float]:
    """Pull user-typed discovery thresholds out of free-form chat.
    Returns the override dict (possibly empty). Only sets a key when
    the parsed value passes basic sanity bounds — out-of-range numbers
    are dropped so a typo (e.g. "200x") can't break the scanner."""
    if not message:
        return {}
    overrides: dict[str, float] = {}

    # Volume multiplier ("1.2x", "2x", "1.5×"). Only honor when the
    # surrounding text mentions volume / spike / relative volume so a
    # bare "1x" in unrelated prose doesn't get mis-classified.
    if _VOLUME_CONTEXT_RE.search(message):
        m = _VOLUME_MULTIPLIER_RE.search(message)
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                val = None
            if val is not None and 0.5 <= val <= 10.0:
                overrides["volume_multiplier"] = val

    # 52-week proximity ("within 5% of 52-week high / low"). Convert
    # percentage to the multiplicative factor the preset expects.
    m_hi = _NEAR_52W_HIGH_PCT_RE.search(message)
    if m_hi:
        try:
            pct = float(m_hi.group(1))
        except ValueError:
            pct = None
        if pct is not None and 0.0 < pct <= 25.0:
            overrides["near_52w_high_factor"] = round(1.0 - pct / 100.0, 6)

    m_lo = _NEAR_52W_LOW_PCT_RE.search(message)
    if m_lo:
        try:
            pct = float(m_lo.group(1))
        except ValueError:
            pct = None
        if pct is not None and 0.0 < pct <= 25.0:
            overrides["near_52w_low_factor"] = round(1.0 + pct / 100.0, 6)

    # Phase 9i — lookback window for the high/low test. Only honored
    # when the user's wording connects the duration to a high/low/
    # breakout context (e.g. "near 20-day high", "on verge of 26 week
    # breakout"). Bounds 5..1260 cover 1 trading week to 5 trading
    # years; out-of-range values are dropped so e.g. "5000 days" can't
    # blow up the OHLCV fetch.
    if _LOOKBACK_CONTEXT_RE.search(message):
        m_win = _LOOKBACK_WINDOW_RE.search(message)
        if m_win:
            try:
                n = int(m_win.group(1))
            except ValueError:
                n = None
            unit = m_win.group(2).lower()
            multiplier = _BARS_PER_UNIT.get(unit)
            if n is not None and multiplier is not None:
                bars = n * multiplier
                if 5 <= bars <= 1260:
                    overrides["lookback_window_bars"] = float(bars)

    return overrides


# ── Phase 9k — compositional discovery primitives ───────────────────────────


# RSI: catches "RSI above 60", "RSI > 60", "RSI is greater than 60",
# "RSI exceeds 60", "RSI in overbought zone above 70", etc.
_RSI_BAND_DISCOVERY_RE = re.compile(
    r"rsi\s+(?:between|from)\s+\d+\.?\d*\s+(?:and|to|-)\s+\d+\.?\d*",
    re.IGNORECASE,
)
_RSI_ABOVE_RE = re.compile(
    r"\brsi[\s\(\d\)]{0,8}\b[^.]{0,40}?"
    r"(?:above|>|>=|over|greater\s+than|exceed(?:ing|s)?|higher\s+than|crosses?\s+(?:above|over))"
    r"\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RSI_BELOW_RE = re.compile(
    r"\brsi[\s\(\d\)]{0,8}\b[^.]{0,40}?"
    r"(?:below|<|<=|under|less\s+than|lower\s+than|crosses?\s+(?:below|under))"
    r"\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# VWAP: "above VWAP", "above the VWAP", "price > VWAP", "VWAP confirmation",
# "closes above VWAP", "is above VWAP", "trading above VWAP"
_VWAP_ABOVE_RE = re.compile(
    r"\b(?:above|over|>)\s+(?:the\s+)?vwap\b"
    r"|\bvwap\s+confirmation\b"
    r"|\b(?:close[sd]?|closes?|closing|trading)\s+above\s+(?:the\s+)?vwap\b"
    r"|\bprice\s+(?:is\s+)?above\s+(?:the\s+)?vwap\b"
    r"|\bcandle\s+(?:close[sd]?\s+)?above\s+(?:the\s+)?vwap\b"
    r"|\babove\s+vwap\s+(?:line|level)?\b",
    re.IGNORECASE,
)
_VWAP_BELOW_RE = re.compile(
    r"\b(?:below|under|<)\s+(?:the\s+)?vwap\b"
    r"|\b(?:close[sd]?|closes?|closing|trading)\s+below\s+(?:the\s+)?vwap\b"
    r"|\bprice\s+(?:is\s+)?below\s+(?:the\s+)?vwap\b"
    r"|\bcandle\s+(?:close[sd]?\s+)?below\s+(?:the\s+)?vwap\b",
    re.IGNORECASE,
)
# EMA: "above 20 EMA", "above EMA(20)", "above the 50 day EMA",
# "above 200-day EMA", "above 9 ema", "price > EMA(20)"
# EMA — `\b` is dropped before `>` because `>` isn't a word character
# and would prevent the boundary from matching. Use explicit `\b` only
# for word verbs (above/over). Order: alt 1 catches `above 20 EMA`,
# alt 2 catches `above EMA(20)` or `> EMA(20)`, alt 3 catches the
# verb-prefixed form (`price above 20 EMA`, `price > EMA(20)`).
_ABOVE_EMA_RE = re.compile(
    r"(?:\babove|\bover|>)\s+(?:the\s+)?(\d+)[\s-]?(?:day[\s-]+)?ema\b"
    r"|(?:\babove|\bover|>)\s+ema\s*\(?\s*(\d+)\s*\)?"
    r"|\b(?:close[sd]?|closes?|trading|price)\s+(?:is\s+)?(?:above|over|>)\s+(?:the\s+)?(\d+)[\s-]?(?:day[\s-]+)?ema\b"
    r"|\b(?:close[sd]?|closes?|trading|price)\s+(?:is\s+)?(?:above|over|>)\s+ema\s*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)
_BELOW_EMA_RE = re.compile(
    r"(?:\bbelow|\bunder|<)\s+(?:the\s+)?(\d+)[\s-]?(?:day[\s-]+)?ema\b"
    r"|(?:\bbelow|\bunder|<)\s+ema\s*\(?\s*(\d+)\s*\)?"
    r"|\b(?:close[sd]?|closes?|trading|price)\s+(?:is\s+)?(?:below|under|<)\s+(?:the\s+)?(\d+)[\s-]?(?:day[\s-]+)?ema\b"
    r"|\b(?:close[sd]?|closes?|trading|price)\s+(?:is\s+)?(?:below|under|<)\s+ema\s*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)
# Day high / low: "near day high", "close to today's high",
# "approaching day's high", "at session high", "near intraday high"
_NEAR_DAY_HIGH_RE = re.compile(
    r"\b(?:near|close\s+to|approaching|at|stock\s+is\s+close\s+to)"
    r"\s+(?:the\s+)?(?:day(?:\'s)?|today(?:\'s)?|session|intraday)\s+high\b",
    re.IGNORECASE,
)
_NEAR_DAY_LOW_RE = re.compile(
    r"\b(?:near|close\s+to|approaching|at|stock\s+is\s+close\s+to)"
    r"\s+(?:the\s+)?(?:day(?:\'s)?|today(?:\'s)?|session|intraday)\s+low\b",
    re.IGNORECASE,
)
# Pullback context: bias for long vs short variant
_PULLBACK_LONG_CONTEXT_RE = re.compile(
    r"\b(?:bullish|long|up(?:trend|side)?|bull|buying)\b",
    re.IGNORECASE,
)
_PULLBACK_SHORT_CONTEXT_RE = re.compile(
    r"\b(?:bearish|short|down(?:trend|side)?|bear|weak\s+recovery|selling)\b",
    re.IGNORECASE,
)
# Pullback detection: "low pullback", "shallow pullback", "pullback < 1%",
# "pullback is less than 2%", "minor retrace", "shallow retracement",
# "pullback is shallow", "weak recovery after pullback"
_PULLBACK_RE = re.compile(
    r"\b(?:low|shallow|small|minor|tiny|slight|short)\s+(?:pull[-\s]?back|retrace(?:ment)?)\b"
    r"|\b(?:pull[-\s]?back|retrace(?:ment)?)\s+is\s+(?:shallow|small|minor|low|tiny|slight|short)\b"
    r"|\b(?:pull[-\s]?back|retrace(?:ment)?)\s+(?:is\s+)?(?:less\s+than|under|<|below|<=)\s*\d+(?:\.\d+)?\s*%"
    r"|\b(?:pull[-\s]?back|retrace(?:ment)?)\s+of\s+(?:less\s+than\s+|under\s+|<\s*|below\s+)?\d+(?:\.\d+)?\s*%"
    r"|\bweak\s+recovery\s+(?:after\s+)?(?:pull[-\s]?back|retrace(?:ment)?)\b"
    r"|\b(?:limited|controlled|brief)\s+pull[-\s]?back\b",
    re.IGNORECASE,
)
# 52-week high (or any window-N high): "near 52-week high", "near 20-day high",
# "approaching 52w high", "verge of 52 weeks high", "close to year high",
# "near yearly peak", "near record high", "at 52-week peak"
_NEAR_52W_HIGH_RE = re.compile(
    r"\b(?:near|close\s+to|approaching|on\s+verge\s+of|verge\s+of|at|nearing)\s+"
    r"(?:the\s+)?\d*\s*"
    r"(?:[-\s]?week|w\b|[-\s]?year|y\b|[-\s]?month|m\b|[-\s]?day|d\b|year(?:ly)?|recent|record|all[-\s]?time)?"
    r"[\s-]?(?:high|peak|top)\b"
    r"|\b(?:near|close\s+to|approaching)\s+\d+[-\s]?(?:day|week|month|year)s?\s+(?:high|peak|top)\b"
    r"|\bnear\s+52[-\s]?week[s]?\s+high\b",
    re.IGNORECASE,
)
_NEAR_52W_LOW_RE = re.compile(
    r"\b(?:near|close\s+to|approaching|on\s+verge\s+of|verge\s+of|at|nearing)\s+"
    r"(?:the\s+)?\d*\s*"
    r"(?:[-\s]?week|w\b|[-\s]?year|y\b|[-\s]?month|m\b|[-\s]?day|d\b|year(?:ly)?|recent|record|all[-\s]?time)?"
    r"[\s-]?(?:low|bottom)\b"
    r"|\b(?:near|close\s+to|approaching)\s+\d+[-\s]?(?:day|week|month|year)s?\s+(?:low|bottom)\b"
    r"|\bnear\s+52[-\s]?week[s]?\s+low\b",
    re.IGNORECASE,
)
# Above/breaking high: "breaking 52-week high", "breaks above 52w high",
# "above all-time high", "new 52-week high", "fresh breakout above 52w high"
_ABOVE_52W_HIGH_RE = re.compile(
    r"\b(?:break(?:ing|s|out)?|fresh\s+breakout|new(?:ly)?(?:\s+made)?|crossing|crosses)\s+"
    r"(?:above|over|past|through)?\s*"
    r"(?:the\s+)?(?:52[-\s]?week[s]?|52w|\d+[-\s]?(?:day|week|month|year)s?|year(?:ly)?|all[-\s]?time|record)\s+(?:highs?|peaks?)\b"
    r"|\b(?:above|over)\s+(?:the\s+)?(?:52[-\s]?week[s]?|52w|\d+[-\s]?(?:day|week|month|year)s?|year(?:ly)?|all[-\s]?time)\s+(?:highs?|peaks?)\b"
    r"|\bnew\s+52[-\s]?week[s]?\s+highs?\b",
    re.IGNORECASE,
)
_BELOW_52W_LOW_RE = re.compile(
    r"\b(?:break(?:ing|s|down)?|fresh\s+breakdown|new(?:ly)?(?:\s+made)?|crossing|crosses)\s+"
    r"(?:below|under|past|through)?\s*"
    r"(?:the\s+)?(?:52[-\s]?week[s]?|52w|\d+[-\s]?(?:day|week|month|year)s?|year(?:ly)?|all[-\s]?time|record)\s+(?:lows?|bottoms?)\b"
    r"|\b(?:below|under)\s+(?:the\s+)?(?:52[-\s]?week[s]?|52w|\d+[-\s]?(?:day|week|month|year)s?|year(?:ly)?|all[-\s]?time)\s+(?:lows?|bottoms?)\b"
    r"|\bnew\s+52[-\s]?week[s]?\s+lows?\b",
    re.IGNORECASE,
)
_HIGH_LOW_MENTION_RE = re.compile(
    r"\b(?:high\s+or\s+low|low\s+or\s+high|extreme[s]?)\b",
    re.IGNORECASE,
)


def _extract_discovery_conditions(
    message: str,
    overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Parse the user's prose into an ordered list of discovery
    primitives. Each item is `{name, params}` ready for
    `app.services.discovery.primitives.render_primitive`.

    Phase 9k — REPLACES the old "always run the preset's hardcoded
    conditions" behavior. The orchestrator runs ONLY the primitives
    extracted here, so a prompt that mentions volume but not 52-week
    proximity gets a volume-only scan (no implicit 52w clause).

    `overrides` is the dict from `_extract_discovery_parameter_overrides`;
    primitives consume them so e.g. `volume_spike` honors the user's
    "1.2x" multiplier and `near_52w_high` honors "within 5%".
    """
    if not message:
        return []
    overrides = overrides or {}
    conditions: list[dict[str, Any]] = []

    # ── Volume ─────────────────────────────────────────────────────────
    if _VOLUME_CONTEXT_RE.search(message):
        multiplier = overrides.get("volume_multiplier")
        if multiplier is not None:
            conditions.append(
                {"name": "volume_spike", "params": {"multiplier": float(multiplier)}}
            )
        elif re.search(
            r"\bhigh[-\s]+volume\b"
            r"|\bstrong[-\s]+volume\b"
            r"|\belevated[-\s]+volume\b",
            message, re.IGNORECASE,
        ):
            # User said "high/strong/elevated volume" (with hyphen or
            # space) without a multiplier — default 2x.
            conditions.append(
                {"name": "volume_spike", "params": {"multiplier": 2.0}}
            )
        elif re.search(r"\b(?:volume\s+spike|spike\s+up|relative\s+volume)\b",
                       message, re.IGNORECASE):
            # "volume spike" without a number — default 2x.
            conditions.append(
                {"name": "volume_spike", "params": {"multiplier": 2.0}}
            )

    # ── 52-week / lookback proximity ──────────────────────────────────
    window = overrides.get("lookback_window_bars")
    near_high_factor = overrides.get("near_52w_high_factor")
    near_low_factor  = overrides.get("near_52w_low_factor")

    # "above 52-week high" / "breaking 52-week high" → above_52w_high
    if _ABOVE_52W_HIGH_RE.search(message):
        params: dict[str, Any] = {}
        if window is not None:
            params["window"] = float(window)
        conditions.append({"name": "above_52w_high", "params": params})
    elif _NEAR_52W_HIGH_RE.search(message) or (
        _HIGH_LOW_MENTION_RE.search(message) and re.search(r"\b(?:52|year|week)\b", message, re.IGNORECASE)
    ):
        params = {}
        if window is not None:
            params["window"] = float(window)
        if near_high_factor is not None:
            params["factor"] = float(near_high_factor)
        conditions.append({"name": "near_52w_high", "params": params})

    if _BELOW_52W_LOW_RE.search(message):
        params = {}
        if window is not None:
            params["window"] = float(window)
        conditions.append({"name": "below_52w_low", "params": params})
    elif _NEAR_52W_LOW_RE.search(message) or (
        _HIGH_LOW_MENTION_RE.search(message) and re.search(r"\b(?:52|year|week)\b", message, re.IGNORECASE)
    ):
        params = {}
        if window is not None:
            params["window"] = float(window)
        if near_low_factor is not None:
            params["factor"] = float(near_low_factor)
        conditions.append({"name": "near_52w_low", "params": params})

    # ── Pullback ──────────────────────────────────────────────────────
    if _PULLBACK_RE.search(message):
        # Bias: bearish/short context → short pullback, else default to long.
        if _PULLBACK_SHORT_CONTEXT_RE.search(message):
            conditions.append({"name": "shallow_pullback_short", "params": {}})
        else:
            conditions.append({"name": "shallow_pullback_long", "params": {}})

    # ── RSI ───────────────────────────────────────────────────────────
    # Skip threshold extraction when user specified an RSI band (55–75).
    # Require threshold >= 20 so volume multipliers (e.g. 1.5×) are never
    # bound to rsi_above.
    _has_rsi_band = bool(_RSI_BAND_DISCOVERY_RE.search(message))
    if not _has_rsi_band:
        m_rsi_hi = _RSI_ABOVE_RE.search(message)
        if m_rsi_hi:
            try:
                threshold = float(m_rsi_hi.group(1))
            except ValueError:
                threshold = None
            if threshold is not None and 20.0 <= threshold < 100.0:
                conditions.append(
                    {"name": "rsi_above", "params": {"threshold": threshold}}
                )
        m_rsi_lo = _RSI_BELOW_RE.search(message)
        if m_rsi_lo:
            try:
                threshold = float(m_rsi_lo.group(1))
            except ValueError:
                threshold = None
            if threshold is not None and 0.0 < threshold <= 80.0:
                conditions.append(
                    {"name": "rsi_below", "params": {"threshold": threshold}}
                )

    # ── VWAP ──────────────────────────────────────────────────────────
    if _VWAP_ABOVE_RE.search(message):
        conditions.append({"name": "above_vwap", "params": {}})
    if _VWAP_BELOW_RE.search(message):
        conditions.append({"name": "below_vwap", "params": {}})

    # ── EMA ───────────────────────────────────────────────────────────
    m_ema_hi = _ABOVE_EMA_RE.search(message)
    if m_ema_hi:
        # Period may be in any of the 4 capture groups (one per alt).
        period_str = next(
            (g for g in m_ema_hi.groups() if g),
            None,
        )
        try:
            period = int(period_str) if period_str else 20
        except (TypeError, ValueError):
            period = 20
        if 1 <= period <= 200:
            conditions.append(
                {"name": "above_ema", "params": {"period": period}}
            )
    m_ema_lo = _BELOW_EMA_RE.search(message)
    if m_ema_lo:
        period_str = next(
            (g for g in m_ema_lo.groups() if g),
            None,
        )
        try:
            period = int(period_str) if period_str else 20
        except (TypeError, ValueError):
            period = 20
        if 1 <= period <= 200:
            conditions.append(
                {"name": "below_ema", "params": {"period": period}}
            )

    # ── Day high / low ────────────────────────────────────────────────
    if _NEAR_DAY_HIGH_RE.search(message):
        conditions.append({"name": "near_day_high", "params": {}})
    if _NEAR_DAY_LOW_RE.search(message):
        conditions.append({"name": "near_day_low", "params": {}})

    return conditions


# Phase 9l — context detector. The LLM extractor is only invoked when
# the message LOOKS like discovery prose so we don't burn an LLM call
# on every single chat turn (e.g. when the user just sends a timeframe
# like "5m" or a bare confirmation like "ok").
_DISCOVERY_CONTEXT_RE = re.compile(
    r"\b(?:volume|spike|rsi|vwap|ema|sma|bollinger|macd|"
    r"pull[-\s]?back|retrace(?:ment)?|breakout|breakdown|breaking|"
    r"high|low|peak|bottom|top|verge|near|above|below|over|under|"
    r"52[-\s]?week|year(?:ly)?|month|week|day|all[-\s]?time|"
    r"momentum|overbought|oversold|relative\s+strength|"
    r"day\s+high|session\s+high|intraday\s+high|day\s+low|session\s+low|"
    r"strong\s+volume|high\s+volume|elevated\s+volume|relative\s+volume|"
    r"scanner|screener|filter|stocks?\s+(?:where|with|that))\b",
    re.IGNORECASE,
)


# Per-process cache so identical messages don't burn duplicate LLM
# calls within a session. Trimmed to the last 256 entries (LRU-ish).
_LLM_EXTRACTOR_CACHE: dict[str, list[dict[str, Any]]] = {}
_LLM_EXTRACTOR_CACHE_MAX = 256


def _looks_like_discovery_prose(message: str) -> bool:
    """Cheap context check before paying for an LLM call."""
    if not message or len(message.strip()) < 5:
        return False
    return bool(_DISCOVERY_CONTEXT_RE.search(message))


def _merge_condition_lists(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine two parsed condition lists. Primary entries win when
    the same primitive name appears on both sides — secondary's
    contribution is only the primitives primary missed."""
    seen = {c["name"] for c in primary if isinstance(c, dict) and c.get("name")}
    merged = list(primary)
    for c in secondary:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name or name in seen:
            continue
        merged.append(c)
        seen.add(name)
    return merged


async def _extract_discovery_conditions_hybrid(
    message: str,
    overrides: dict[str, float] | None = None,
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid regex + LLM extractor.

    Behavior:
      • Regex extractor runs first — fast, deterministic, free.
      • If the message clearly isn't discovery prose, return regex
        result (which is also empty in this case) — saves the LLM call.
      • If regex captured ≥ 2 primitives, trust it and skip the LLM.
        The combo of "user typed something specific" + "regex matched
        multiple things" is a strong signal we already have the right
        list.
      • Otherwise call the LLM extractor and merge: regex is canonical
        for primitives it caught, LLM fills in the rest.

    LLM responses are cached per-process so a repeated identical
    message (e.g. user resends after a network hiccup) doesn't burn
    a second call.
    """
    regex_conditions = _extract_discovery_conditions(message, overrides)

    if not _looks_like_discovery_prose(message):
        return regex_conditions
    if len(regex_conditions) >= 2:
        return regex_conditions

    cache_key = (message or "").strip().lower()
    if cache_key in _LLM_EXTRACTOR_CACHE:
        cached = _LLM_EXTRACTOR_CACHE[cache_key]
        return _merge_condition_lists(regex_conditions, cached) or regex_conditions

    try:
        from app.services.discovery.llm_extractor import extract_via_llm
        llm_conditions = await extract_via_llm(message)
    except Exception as exc:
        logger.info(
            "chat_flow|event=llm_extractor_error|session_id=%s|err=%s",
            session_id, exc,
        )
        llm_conditions = []

    # Reapply user-typed parameter overrides on top of the LLM's
    # output — the LLM may have used the primitive's defaults even
    # when the user explicitly typed a different number.
    if llm_conditions and overrides:
        llm_conditions = _apply_overrides_to_llm_conditions(llm_conditions, overrides)

    if llm_conditions:
        if len(_LLM_EXTRACTOR_CACHE) >= _LLM_EXTRACTOR_CACHE_MAX:
            # Drop oldest entry to keep memory bounded.
            try:
                _LLM_EXTRACTOR_CACHE.pop(next(iter(_LLM_EXTRACTOR_CACHE)))
            except StopIteration:
                pass
        _LLM_EXTRACTOR_CACHE[cache_key] = llm_conditions
        logger.info(
            "chat_flow|event=llm_extractor_supplemented|session_id=%s|"
            "regex_count=%d|llm_count=%d|llm_names=%s",
            session_id, len(regex_conditions), len(llm_conditions),
            [c.get("name") for c in llm_conditions],
        )

    return _merge_condition_lists(regex_conditions, llm_conditions)


def _apply_overrides_to_llm_conditions(
    conditions: list[dict[str, Any]],
    overrides: dict[str, float],
) -> list[dict[str, Any]]:
    """When the regex extractor pulled out a numeric override
    (volume_multiplier=1.2, near_52w_high_factor=0.95, …), make sure
    the LLM's primitive uses that value rather than its own default.
    The user typed it explicitly; the LLM might have rounded or
    skipped it."""
    out: list[dict[str, Any]] = []
    for c in conditions:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        params = dict(c.get("params") or {})
        if name == "volume_spike" and "volume_multiplier" in overrides:
            params["multiplier"] = float(overrides["volume_multiplier"])
        elif name in ("near_52w_high", "above_52w_high") and "lookback_window_bars" in overrides:
            params["window"] = float(overrides["lookback_window_bars"])
            if name == "near_52w_high" and "near_52w_high_factor" in overrides:
                params["factor"] = float(overrides["near_52w_high_factor"])
        elif name in ("near_52w_low", "below_52w_low") and "lookback_window_bars" in overrides:
            params["window"] = float(overrides["lookback_window_bars"])
            if name == "near_52w_low" and "near_52w_low_factor" in overrides:
                params["factor"] = float(overrides["near_52w_low_factor"])
        out.append({"name": name, "params": params})
    return out


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
        candidate = natural_exchange.group(1)
        # Phase 9g — generic English words ("strategy", "stock", etc.)
        # are never tickers. Reject so discovery prompts like "create
        # strategy on NSE stock..." don't get misparsed.
        if candidate.lower() in _NON_TICKER_WORDS:
            return None
        # Phase 9g — if NSE/BSE is followed by a generic noun ("NSE
        # stock", "BSE company"), the user is describing scope, not a
        # specific ticker. Drop the match.
        tail = message[natural_exchange.end():]
        if _NSE_SCOPE_NOUN_RE.match(tail):
            return None
        return candidate.upper()

    return None


def _stock_choice_key(value: Any) -> str:
    text = str(value or "").strip()
    if ":" in text:
        _, text = text.split(":", 1)
    text = re.sub(r"\.(?:NS|BO)$", "", text, flags=re.IGNORECASE)
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _resolve_pending_stock_ambiguity_choice(
    message: str,
    builder: StrategyBuilder,
) -> str | None:
    """Resolve replies like "1" after an ambiguous stock-prefix prompt."""
    if builder.symbol_validation_code != AMBIGUOUS_STOCK_VALIDATION_CODE:
        return None
    options = builder.symbol_validation_facts.get("stock_options")
    if not isinstance(options, list) or not options:
        return None

    text = " ".join(str(message or "").split()).strip()
    if not text:
        return None

    if re.fullmatch(r"\d+", text):
        index = int(text)
        if 1 <= index <= len(options) and isinstance(options[index - 1], dict):
            return str(options[index - 1].get("symbol") or "").strip() or None
        return None

    choice_key = _stock_choice_key(text)
    if not choice_key:
        return None

    for option in options:
        if not isinstance(option, dict):
            continue
        symbol = str(option.get("symbol") or "").strip()
        display_name = str(option.get("display_name") or "").strip()
        candidate_keys = {
            _stock_choice_key(symbol),
            _stock_choice_key(symbol.split(".", 1)[0]),
            _stock_choice_key(display_name),
        }
        if choice_key in candidate_keys:
            return symbol or None

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

            # Phase 9e — early preset detection. We scan the user message
            # for a strategy preset BEFORE the agent router runs so the
            # agent sees the full picture (and so the downstream
            # is_user_input_complete() check exempts symbol when discovery
            # would supply it). Without this, the agent sees no preset +
            # no symbol and asks the user for a specific stock — which is
            # exactly what discovery is meant to avoid.
            if not builder.strategy_preset:
                try:
                    from app.kb import kb as _kb_for_preset
                    _early_preset = _kb_for_preset.detect_preset_in_text(user_content)
                except Exception:
                    _early_preset = None
                if _early_preset is not None:
                    builder.strategy_preset = _early_preset.name
                    logger.info(
                        "🎯 chat_flow|event=preset_detected_pre_router|"
                        "session_id=%s|preset=%s",
                        session_id, _early_preset.name,
                    )

            # Issue 2 — recognise an explicit "choose among these stocks:
            # A, B, C" message and persist the resolved symbols on the
            # builder so the next discovery run scans only those stocks.
            # Sticky across turns; the user can broaden by typing a
            # fresh list (or by saying "scan all" — TODO, not in this
            # change).
            _new_universe = _extract_user_supplied_stock_list(user_content)
            if _new_universe:
                builder.discovery_universe_override = _new_universe
                logger.info(
                    "🎯 chat_flow|event=discovery_universe_override_captured|"
                    "session_id=%s|symbols=%s",
                    session_id, _new_universe,
                )

            # Phase 9h — parse user-typed discovery thresholds (e.g.
            # "volume spike 1.2x", "within 5% of 52-week high") and
            # stash them on the builder. The orchestrator substitutes
            # them into the preset's condition placeholders before the
            # scanner runs, so the user actually gets the threshold
            # they asked for instead of the preset's hardcoded default.
            _new_overrides = _extract_discovery_parameter_overrides(user_content)
            if _new_overrides:
                merged_overrides = dict(builder.discovery_parameter_overrides or {})
                merged_overrides.update(_new_overrides)
                builder.discovery_parameter_overrides = merged_overrides
                logger.info(
                    "🎚 chat_flow|event=discovery_param_overrides_captured|"
                    "session_id=%s|overrides=%s",
                    session_id, _new_overrides,
                )

            # Phase 9k/9l — parse the user's prose into a list of
            # discovery primitives. Hybrid pipeline:
            #   1. Regex extractor runs first (fast, deterministic).
            #   2. If regex finds 0-1 primitives BUT the message looks
            #      like discovery prose, the LLM extractor runs as a
            #      fallback — covers any phrasing the regex missed.
            #   3. The merged list is stashed on builder.discovery_conditions
            #      and the orchestrator runs the scanner with EXACTLY those
            #      constraints (no implicit clauses).
            #
            # Note (post-PR#71 follow-up): the previous version of this
            # block contained an `_preset_carries_authoritative_conditions`
            # guard that suppressed extraction whenever the pinned preset
            # had a `discovery.conditions:` block (e.g. volume_breakout_52w).
            # That guard reversed Phase 9k/9l's design — it caused the
            # preset's full conditions (with implicit 52-week clauses) to
            # be applied even when the user only typed "volume spike 1.2x",
            # which is the exact behaviour Phase 9k was created to fix.
            # The original concern ("OR-disjunction wiped") was a misread:
            # when the user supplies primitives, the orchestrator CORRECTLY
            # replaces the preset's conditions entirely with the rendered
            # primitive list. When the user supplies nothing, the preset's
            # OR-disjunction still applies because builder.discovery_conditions
            # stays None and the orchestrator falls through to the preset.
            _new_conditions = await _extract_discovery_conditions_hybrid(
                user_content, _new_overrides, session_id=session_id,
            )
            if _new_conditions:
                builder.discovery_conditions = _new_conditions
                logger.info(
                    "🧩 chat_flow|event=discovery_conditions_captured|"
                    "session_id=%s|conditions=%s",
                    session_id, [c["name"] for c in _new_conditions],
                )

            agent_decision = await AgentRouter().decide(
                session_id=session_id,
                user_message=user_content,
                builder=builder,
                previous_state=previous_state,
                recent_messages=all_messages,
                latest_strategy_context=latest_strategy_context,
                latest_backtest_result=latest_backtest_result,
                asset_classes=_enabled_asset_classes(
                    chat_obj.capabilities if chat_obj else None
                ),
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

            if route_intent == "modify_signals":
                from app.services.strategy.signal_modifier import (
                    apply_signal_request,
                )

                params = dict(route.get("agent_tool_parameters") or {})

                # The tool schema requires `changes: [...]`. Older shapes
                # (top-level slot/signal_name) are folded into a single-item
                # list so a stray legacy call still works.
                raw_changes = params.get("changes")
                if isinstance(raw_changes, list) and raw_changes:
                    changes = [c for c in raw_changes if isinstance(c, dict)]
                else:
                    legacy_slot = params.get("slot")
                    legacy_signal = params.get("signal_name")
                    if legacy_slot or legacy_signal:
                        changes = [{
                            "slot": legacy_slot,
                            "signal_name": legacy_signal,
                            "replace_name": params.get("replace_name"),
                            "action": params.get("action"),
                        }]
                    else:
                        changes = []

                logger.info(
                    "🔧 chat_flow|event=modify_signal_request|session_id=%s"
                    "|change_count=%d",
                    session_id,
                    len(changes),
                )

                # Always force plan_signals so that the next "yes" re-assembles
                # the strategy with the new signals and shows the preview card,
                # instead of jumping directly to backtest with the old yaml.
                assistant_state = "plan_signals"
                assistant_text: str
                per_change_outcomes: list[dict[str, Any]] = []
                applied_messages: list[str] = []
                pending_prompts: list[str] = []
                error_messages: list[str] = []

                # ── SDL-first modify ──────────────────────────────────────────
                # If this strategy was built via the SDL flow, edit the SDL
                # ticket directly (holistic + accurate) rather than the legacy
                # per-signal modifier. The user's raw message is the change
                # instruction ("change RSI to 20", "use EMA instead of SMA").
                # Falls back to legacy when there is no SDL on the builder or the
                # SDL modify can't run — so legacy-built strategies are unaffected.
                _sdl_modified = False
                if getattr(builder, "_sdl", None) is not None:
                    try:
                        from app.planner.sdl_flow import try_sdl_modify
                        _mod = await try_sdl_modify(
                            user_content, builder, session_id=session_id,
                        )
                    except Exception as _exc:
                        logger.warning(
                            "⚠️ chat_flow|event=sdl_modify_error|session_id=%s|err=%s",
                            session_id, _exc,
                        )
                        _mod = None
                    if _mod is not None and _mod.used_sdl:
                        _sdl_modified = True
                        builder.signal_plan = _mod.signal_plan or builder.signal_plan
                        if _mod.validation_ok:
                            assistant_text = (
                                (_mod.readback_text or "").rstrip()
                                + "\n\nConfirm to assemble the strategy, or request "
                                "another change."
                            )
                        else:
                            assistant_text = (
                                (_mod.readback_text or "").rstrip()
                                or "I updated the strategy, but it needs a fix "
                                "before backtest."
                            )
                        per_change_outcomes.append({
                            "status": "sdl_modified",
                            "match_pct": _mod.match_pct,
                            "valid": _mod.validation_ok,
                        })
                        logger.info(
                            "✅ chat_flow|event=sdl_modify_used|session_id=%s"
                            "|match=%.0f%%|valid=%s",
                            session_id, _mod.match_pct, _mod.validation_ok,
                        )

                if _sdl_modified:
                    pass  # SDL path already set assistant_text + signal_plan
                elif not builder.signal_plan:
                    assistant_text = (
                        "There is no signal plan to modify yet. Please ask me "
                        "to plan signals first, then you can swap individual "
                        "entry / exit signals."
                    )
                    per_change_outcomes.append({"status": "no_plan"})
                elif not changes or _llm_hallucinated_signal_names(changes, user_content):
                    from app.services.strategy.signal_suggestions import (
                        format_signal_suggestions,
                    )
                    assistant_text = format_signal_suggestions(session_id, n=5)
                    per_change_outcomes.append({"status": "bad_params"})
                else:
                    for change in changes:
                        slot = str(change.get("slot") or "").strip().lower()
                        signal_name = str(change.get("signal_name") or "").strip()
                        replace_name = (
                            str(change.get("replace_name") or "").strip() or None
                        )
                        action = str(change.get("action") or "").strip().lower()

                        outcome: dict[str, Any] = {
                            "slot": slot,
                            "signal_name": signal_name,
                            "action": action or "replace",
                            "replace_name": replace_name,
                        }

                        logger.info(
                            "🔧 chat_flow|modify_signal_item|session_id=%s"
                            "|slot=%s|signal=%s|action=%s|replace=%s",
                            session_id,
                            slot,
                            signal_name,
                            action or "(default)",
                            replace_name or "-",
                        )

                        if slot not in ("entry", "exit") or not signal_name:
                            error_messages.append(
                                "Skipped a change with missing slot or signal "
                                f"name (slot={slot or '?'}, signal={signal_name or '?'})."
                            )
                            outcome["status"] = "bad_params"
                            per_change_outcomes.append(outcome)
                            continue

                        if action == "remove":
                            try:
                                builder.remove_signal(slot, signal_name)
                                applied_messages.append(
                                    f"Removed {slot} signal '{signal_name}'."
                                )
                                outcome["status"] = "removed"
                            except Exception as exc:
                                error_messages.append(
                                    f"Could not remove '{signal_name}' from "
                                    f"{slot}: {exc}"
                                )
                                outcome["status"] = "error"
                                outcome["error"] = str(exc)
                            per_change_outcomes.append(outcome)
                            continue

                        # Default to 'replace' when the LLM does not specify —
                        # matches the most common phrasing ("change entry
                        # signal to X" / "use Y for exit"), which wipes the
                        # slot and sets the new signal. Only 'add' preserves
                        # prior entries; anything else falls back to 'replace'.
                        apply_action = action if action in ("replace", "add") else "replace"
                        try:
                            result = apply_signal_request(
                                builder,
                                slot,  # type: ignore[arg-type]
                                signal_name,
                                replace_name=replace_name,
                                action=apply_action,
                            )
                        except Exception as exc:
                            logger.warning(
                                "modify_signal|unexpected_error|session_id=%s|err=%s",
                                session_id,
                                str(exc)[:160],
                            )
                            error_messages.append(
                                f"Could not apply '{signal_name}' to {slot}: {exc}"
                            )
                            outcome["status"] = "error"
                            outcome["error"] = str(exc)
                            per_change_outcomes.append(outcome)
                            continue

                        outcome["status"] = result.status
                        if result.candidates:
                            outcome["candidates"] = list(result.candidates)
                        if result.resolved_name:
                            outcome["resolved_name"] = result.resolved_name

                        if result.status == "applied":
                            applied_messages.append(result.message)
                        elif result.status == "needs_choice":
                            pending_prompts.append(result.message)
                        elif result.status == "wrong_slot":
                            pending_prompts.append(
                                f"{result.message}\n"
                                + _format_candidates(result.candidates)
                            )
                        else:  # not_found
                            pending_prompts.append(result.message)

                        per_change_outcomes.append(outcome)

                    # Compose a single trader-facing reply that combines every
                    # outcome — applied changes show first, then any pending
                    # disambiguation prompts, then per-item errors.
                    if applied_messages and not pending_prompts and not error_messages:
                        assistant_text = (
                            " ".join(applied_messages)
                            + " Updated plan:\n"
                            + _format_signal_plan_lines(builder.signal_plan)
                            + "\n\nConfirm to assemble the strategy, or "
                            "request another change."
                        )
                    elif applied_messages:
                        sections: list[str] = [" ".join(applied_messages)]
                        sections.append(
                            "Updated plan so far:\n"
                            + _format_signal_plan_lines(builder.signal_plan)
                        )
                        if pending_prompts:
                            sections.extend(pending_prompts)
                        if error_messages:
                            sections.append("\n".join(error_messages))
                        assistant_text = "\n\n".join(sections)
                    elif pending_prompts or error_messages:
                        sections = list(pending_prompts)
                        if error_messages:
                            sections.append("\n".join(error_messages))
                        assistant_text = "\n\n".join(sections)
                    else:
                        assistant_text = "I could not apply any signal changes."

                draft = builder.to_draft_json(
                    mode_override=assistant_state,
                    processing_status=_draft_processing_status(builder, assistant_state),
                )
                draft["agent_decision"] = {
                    "tool": route.get("agent_tool"),
                    "parameters": params,
                    "source": route.get("agent_source"),
                }
                draft["modify_signal_outcomes"] = per_change_outcomes

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
                        if stock_match and stock_match.get("ambiguous"):
                            facts = (
                                stock_match.get("validation_facts")
                                if isinstance(stock_match.get("validation_facts"), dict)
                                else {}
                            )
                            assistant_text = compose_assistant_response(
                                "validation.ambiguous_stock",
                                **facts,
                            )
                            draft["processing_status"] = "awaiting_user_input"
                            draft["symbol_validation_code"] = AMBIGUOUS_STOCK_VALIDATION_CODE
                            draft["symbol_validation_facts"] = facts
                            draft["symbol_validation_message"] = assistant_text
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
                        # Release the DB connection for the duration of this network
                        # fetch (~20s). user_msg will be detached on close — re-fetch
                        # it below in the now-fresh session before mutating.
                        await db.commit()
                        await db.close()
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

                # Session was released around the network fetch above; user_msg is
                # now detached. Re-fetch it so the mutations below get persisted.
                if user_msg is not None and user_msg_uuid is not None:
                    user_msg = await db.get(ChatMessage, user_msg_uuid)
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
            pending_stock_choice = _resolve_pending_stock_ambiguity_choice(user_content, builder)

            parsed_builder = StrategyBuilder()
            extract_strategy_details(user_content, parsed_builder)

            # ── Early semantic extraction ─────────────────────────────────
            # Run BEFORE planning so that builder.semantic_intent is populated
            # when _resolve_preset is called.  The SemanticExtractor identifies
            # the *primary strategy framework* (ORB, EMA_PULLBACK, …) from the
            # prose structure, which takes priority over KB keyword-scoring in
            # the planner (see Pipeline._SEMANTIC_FAMILY_TO_PRESET).
            # The normalizer then merges the extraction with parsed_builder
            # state to produce a CanonicalSemanticIntent — the single source
            # of truth that all downstream systems will consume.
            try:
                from app.planner.semantic_normalizer import SemanticIntentNormalizer
                _early_sem = SemanticExtractor().extract(user_content)
                if _early_sem and _early_sem.extraction_quality_score > 0:
                    _canonical = SemanticIntentNormalizer().normalize(
                        _early_sem,
                        strategy_preset=parsed_builder.strategy_preset,
                        timeframe=parsed_builder.timeframe,
                        sentiment=parsed_builder.sentiment,
                        stop_loss_spec=parsed_builder.stop_loss_spec,
                        trailing_stop_spec=parsed_builder.trailing_stop_spec,
                        risk_execution_config=parsed_builder.risk_execution_config,
                        source_prompt=user_content[:1000],
                    )
                    parsed_builder.semantic_intent = _canonical.dict()
                    # Propagate structural SL to parsed_builder so the pipeline
                    # can use it even when the LLM tool call missed it.
                    _crm = _canonical.risk_model
                    if _crm:
                        if not parsed_builder.stop_loss_spec and _crm.stop_loss:
                            _csl = _crm.stop_loss
                            parsed_builder.stop_loss_spec = {
                                "type": _csl.type,
                                "anchor": _csl.anchor,
                                "source": "semantic",
                                "atr_multiple": _csl.atr_multiple,
                            }
                        if not parsed_builder.trailing_stop_spec and _crm.trailing_stop and _crm.trailing_stop.enabled:
                            _cts = _crm.trailing_stop
                            parsed_builder.trailing_stop_spec = {
                                "type": _cts.type,
                                "activate_after_pct": _cts.activate_after_pct,
                                "ema_period": _cts.ema_period,
                                "source": "semantic",
                            }
                    # ── Phase 2 — propagate execution parameters from semantic
                    # extractor into parsed_builder's typed fields so they
                    # flow into builder during the propagation block below.
                    _sem = _early_sem
                    if _sem.direction and not parsed_builder.direction:
                        parsed_builder.direction = _sem.direction
                    if _sem.rsi_entry_band_min is not None and parsed_builder.rsi_entry_band_min is None:
                        parsed_builder.rsi_entry_band_min = _sem.rsi_entry_band_min
                    if _sem.rsi_entry_band_max is not None and parsed_builder.rsi_entry_band_max is None:
                        parsed_builder.rsi_entry_band_max = _sem.rsi_entry_band_max
                    if _sem.volume_ratio_threshold is not None and parsed_builder.volume_ratio_threshold is None:
                        parsed_builder.volume_ratio_threshold = _sem.volume_ratio_threshold
                    if _sem.position_sizing_mode and not parsed_builder.position_sizing_mode:
                        parsed_builder.position_sizing_mode = _sem.position_sizing_mode
                    if _sem.max_consecutive_losses is not None and parsed_builder.max_consecutive_losses is None:
                        parsed_builder.max_consecutive_losses = _sem.max_consecutive_losses
                    if _sem.cooldown_bars_after_loss is not None and parsed_builder.cooldown_bars_after_loss is None:
                        parsed_builder.cooldown_bars_after_loss = _sem.cooldown_bars_after_loss
                    if _sem.cooldown_bars_after_profit is not None and parsed_builder.cooldown_bars_after_profit is None:
                        parsed_builder.cooldown_bars_after_profit = _sem.cooldown_bars_after_profit
                    if _sem.max_spread_bps is not None and parsed_builder.max_spread_bps is None:
                        parsed_builder.max_spread_bps = _sem.max_spread_bps
                    if _sem.gap_filter and not parsed_builder.gap_filter:
                        parsed_builder.gap_filter = _sem.gap_filter
                    if _sem.gap_threshold_pct is not None and parsed_builder.gap_threshold_pct is None:
                        parsed_builder.gap_threshold_pct = _sem.gap_threshold_pct
                    if _sem.entry_confirmation_bars is not None and parsed_builder.entry_confirmation_bars is None:
                        parsed_builder.entry_confirmation_bars = _sem.entry_confirmation_bars
                    if _sem.max_capital_allocation_pct is not None and parsed_builder.max_capital_allocation_pct is None:
                        parsed_builder.max_capital_allocation_pct = _sem.max_capital_allocation_pct
            except Exception as _sem_err:
                logger.debug(
                    "chat_flow|early_semantic_extraction_skipped|err=%s", _sem_err
                )

            relevant_message = bool(route.get("is_relevant"))
            recognized_fields = set(route.get("recognized_fields") or [])
            if pending_stock_choice:
                builder.symbol = pending_stock_choice
                builder.sync_market_from_symbol()
                builder.set_symbol_validation(None)
                builder.clear_validation_state()
                relevant_message = True
                recognized_fields.add("symbol")
                route_intent = "collect_input"
                route["intent"] = "collect_input"
                logger.info(
                    "✅ chat_flow|event=stock_ambiguity_choice_resolved|session_id=%s|symbol=%s",
                    session_id,
                    pending_stock_choice,
                )
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
                if pending_stock_choice:
                    stock_query = None
                if stock_query:
                    relevant_message = True
                    stock_match = await resolve_supported_stock(stock_query)
                    if stock_match and stock_match.get("ambiguous"):
                        builder.symbol = None
                        builder.set_symbol_validation(
                            str(stock_match.get("validation_code") or AMBIGUOUS_STOCK_VALIDATION_CODE),
                            stock_match.get("validation_facts")
                            if isinstance(stock_match.get("validation_facts"), dict)
                            else {},
                        )
                        logger.info(
                            "❓ chat_flow|event=stock_prefix_ambiguous|session_id=%s|query=%r|matches=%d",
                            session_id,
                            stock_query,
                            len(stock_match.get("matches") or []),
                        )
                    elif stock_match and stock_match.get("symbol"):
                        resolved_symbol = str(stock_match["symbol"]).strip()
                        exchange_hint = extract_exchange_hint(user_content)
                        if exchange_hint and not resolved_symbol.upper().endswith((".NS", ".BO")):
                            resolved_symbol = f"{resolved_symbol}.{exchange_hint}"
                        builder.symbol = resolved_symbol
                        builder.sync_market_from_symbol()
                        builder.set_symbol_validation(None)
                        logger.info(
                            "✅ chat_flow|event=stock_resolved|session_id=%s|query=%r|symbol=%s|asset_class=%s",
                            session_id,
                            stock_query,
                            resolved_symbol,
                            builder.asset_class,
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

                # Capture the user's raw, full strategy description on the
                # first substantive turn. This survives subsequent `goal`
                # summaries from the agent router so semantic extraction can
                # always go back to the user's literal phrasing (e.g. "buy
                # when price crosses above 20 SMA, 3:7 reward risk ratio").
                # We never overwrite once set — the original prompt is
                # immutable. (Persisted via to_draft_json/merge_preview.)
                if getattr(builder, "original_user_prompt", None) is None:
                    raw = re.sub(r"\s+", " ", user_content or "").strip()
                    if (
                        not _is_greeting_message(user_content)
                        and len(raw.split()) >= 6
                        and not detect_user_confirmation(user_content)
                    ):
                        builder.original_user_prompt = raw
                        logger.info(
                            "📝 chat_flow|event=original_user_prompt_captured"
                            "|session_id=%s|len=%d|preview=%r",
                            session_id, len(raw), raw[:80],
                        )

                # Propagate strategy_preset — three sources in priority order:
                #   1. Agent explicitly named a preset in tool params (highest)
                #   2. KB keyword detection on the user message (parsed_builder)
                #   3. Already set from a prior turn (preserved, never overwrite)
                agent_preset = (route.get("strategy_preset") or "").strip().lower()
                if agent_preset:
                    builder.strategy_preset = agent_preset
                    relevant_message = True
                elif parsed_builder.strategy_preset:
                    # A freshly detected preset from the current message always
                    # supersedes a stale preset stored from a prior turn
                    # (e.g. an old wrong match like "relative_strength" that
                    # was persisted in the draft).
                    if builder.strategy_preset != parsed_builder.strategy_preset:
                        logger.info(
                            "chat|collect_input|preset_override|old=%r|new=%r",
                            builder.strategy_preset,
                            parsed_builder.strategy_preset,
                        )
                    builder.strategy_preset = parsed_builder.strategy_preset
                    relevant_message = True

                # Propagate RMS fields extracted by _extract_rms_from_text.
                # First-write-wins per field: a user-supplied value in a prior
                # turn (source="user") is never overwritten by a new extraction.
                parsed_rms = dict(parsed_builder.risk_execution_config or {})
                parsed_sources = dict(parsed_rms.pop("rms_sources", {}))
                if parsed_rms:
                    existing_rms = dict(builder.risk_execution_config or {})
                    existing_sources = dict(existing_rms.get("rms_sources", {}))
                    changed = False
                    for field, val in parsed_rms.items():
                        if existing_sources.get(field) == "user":
                            continue
                        existing_rms[field] = val
                        existing_sources[field] = parsed_sources.get(field, "user")
                        changed = True
                    if changed:
                        existing_rms["rms_sources"] = existing_sources
                        builder.risk_execution_config = existing_rms
                        # stop_loss / take_profit are read-through properties over
                        # risk_execution_config — writing the dict above is the
                        # single source of truth, no mirror needed. daily_loss_cap
                        # is still a plain attribute.
                        if "daily_loss_cap" in parsed_rms and builder.daily_loss_cap is None:
                            builder.daily_loss_cap = parsed_rms["daily_loss_cap"]
                        relevant_message = True

                # ── Phase 2 — apply execution parameter overrides ───────────────
                # Priority:  agent_tool_parameters (LLM-extracted, highest)
                #            > parsed_builder (regex / semantic extraction)
                #            > existing builder value (preserved from prior turns)
                _p2_agent = dict(route.get("agent_tool_parameters") or {})

                def _p2_str(key: str) -> str | None:
                    v = _p2_agent.get(key)
                    return str(v).strip() if v is not None else None

                def _p2_int(key: str) -> int | None:
                    v = _p2_agent.get(key)
                    try:
                        return int(v) if v is not None else None
                    except (TypeError, ValueError):
                        return None

                def _p2_float(key: str) -> float | None:
                    v = _p2_agent.get(key)
                    try:
                        return float(v) if v is not None else None
                    except (TypeError, ValueError):
                        return None

                _p2_updates: list[tuple[str, object]] = [
                    ("direction",               _p2_str("direction") or parsed_builder.direction),
                    ("entry_window_start",       _p2_str("entry_window_start") or parsed_builder.entry_window_start),
                    ("entry_window_end",         _p2_str("entry_window_end") or parsed_builder.entry_window_end),
                    ("max_consecutive_losses",   _p2_int("max_consecutive_losses") if _p2_agent.get("max_consecutive_losses") is not None else parsed_builder.max_consecutive_losses),
                    ("cooldown_bars_after_loss", _p2_int("cooldown_bars_after_loss") if _p2_agent.get("cooldown_bars_after_loss") is not None else parsed_builder.cooldown_bars_after_loss),
                    ("cooldown_bars_after_profit", _p2_int("cooldown_bars_after_profit") if _p2_agent.get("cooldown_bars_after_profit") is not None else parsed_builder.cooldown_bars_after_profit),
                    ("max_spread_bps",           _p2_float("max_spread_bps") if _p2_agent.get("max_spread_bps") is not None else parsed_builder.max_spread_bps),
                    ("gap_filter",               _p2_str("gap_filter") or parsed_builder.gap_filter),
                    ("gap_threshold_pct",        _p2_float("gap_threshold_pct") if _p2_agent.get("gap_threshold_pct") is not None else parsed_builder.gap_threshold_pct),
                    ("entry_confirmation_bars",  _p2_int("entry_confirmation_bars") if _p2_agent.get("entry_confirmation_bars") is not None else parsed_builder.entry_confirmation_bars),
                    ("rsi_entry_band_min",       _p2_float("rsi_entry_band_min") if _p2_agent.get("rsi_entry_band_min") is not None else parsed_builder.rsi_entry_band_min),
                    ("rsi_entry_band_max",       _p2_float("rsi_entry_band_max") if _p2_agent.get("rsi_entry_band_max") is not None else parsed_builder.rsi_entry_band_max),
                    ("volume_ratio_threshold",   _p2_float("volume_ratio_threshold") if _p2_agent.get("volume_ratio_threshold") is not None else parsed_builder.volume_ratio_threshold),
                    ("position_sizing_mode",     _p2_str("position_sizing_mode") or parsed_builder.position_sizing_mode),
                    ("max_capital_allocation_pct", _p2_float("max_capital_allocation_pct") if _p2_agent.get("max_capital_allocation_pct") is not None else parsed_builder.max_capital_allocation_pct),
                ]
                for _field, _val in _p2_updates:
                    # Always honour a value that came from the current turn
                    # (either the agent's tool call or the regex/semantic
                    # extractor). Previously this only wrote when the builder
                    # field was None, which silently dropped user updates
                    # after tier defaults had filled the slot. Tier-derived
                    # fields are marked so apply_tier_execution_defaults()
                    # won't overwrite them later when experience changes.
                    if _val is not None:
                        setattr(builder, _field, _val)
                        builder.mark_phase10_user_override(_field)
                        relevant_message = True

                if _apply_agent_risk_param_overrides(builder, _p2_agent):
                    relevant_message = True

                # Propagate structural SL + trailing stop specs from parsed_builder.
                # User-specified specs from the current message override stored ones.
                if parsed_builder.stop_loss_spec:
                    builder.stop_loss_spec = parsed_builder.stop_loss_spec
                    relevant_message = True
                if parsed_builder.trailing_stop_spec:
                    builder.trailing_stop_spec = parsed_builder.trailing_stop_spec
                    relevant_message = True

                # Propagate semantic intent extracted this turn onto the builder.
                # A freshly extracted intent (quality > 0) always supersedes any
                # stale intent stored from a prior turn so that the planner always
                # sees the most recent structural reading of the user's message.
                if parsed_builder.semantic_intent:
                    builder.semantic_intent = parsed_builder.semantic_intent
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

            if route_intent == "run_backtest" and user_state in {
                "assemble_strategy",
                "backtest_confirmation",
                "backtest_complete",
            }:
                from app.services.backtest.backtest_window import (
                    BacktestWindowError,
                    earliest_backtest_from_display,
                    resolve_backtest_window,
                )

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

                    _objective         = str(getattr(builder, "objective", None) or "positional").lower()
                    _daily_loss_cap    = float(getattr(builder, "daily_loss_cap", None) or 0.0)
                    _max_trades_per_day = _derive_max_trades_from_builder(builder)

                    _bt_from = route.get("backtest_from_utc") or None
                    _bt_to = route.get("backtest_to_utc") or None
                    _from_utc, _to_utc = resolve_backtest_window(_bt_from, _bt_to, objective=_objective)

                    run_request = BacktestTriggerRequest(
                        strategy_id=str((latest_strategy_context or {}).get("strategy_id") or uuid.uuid4()),
                        from_utc=_bt_from,
                        to_utc=_bt_to,
                        objective=_objective,
                        daily_loss_cap_pct=_daily_loss_cap,
                        max_trades_per_day=_max_trades_per_day,
                    )
                    logger.info(
                        "⚙️ chat_flow|event=backtest_run_config|session_id=%s"
                        "|objective=%s|daily_loss_cap=%.1f%%|max_trades=%d|from=%s|to=%s|yaml=%s",
                        session_id,
                        _objective,
                        _daily_loss_cap,
                        _max_trades_per_day,
                        _from_utc,
                        _to_utc,
                        yaml_path,
                    )
                    backtest_started_at = datetime.now(timezone.utc)
                    market_data_request = extract_strategy_market_data_request(yaml_path, overrides=run_request)
                    run_request = run_request.model_copy(
                        update={
                            "from_utc": market_data_request.from_utc,
                            "to_utc": market_data_request.to_utc,
                        }
                    )
                    # Phase 11 — 1-minute execution. AUTO-on for any non-1m
                    # strategy. The main series fed to the engine is fetched at
                    # 1m below; signal enrichment further down stays on the
                    # strategy-timeframe series so its stats remain calibrated to
                    # the strategy's own bars. The resolved decision is written
                    # back onto run_request so the engine run_config gets a
                    # definite bool (never the AUTO sentinel).
                    intrabar_execution = resolve_intrabar_execution(
                        run_request, signal_interval=market_data_request.interval,
                    )
                    run_request = run_request.model_copy(update={"intrabar_execution": intrabar_execution})
                    logger.info(
                        "📡 chat_flow|event=market_data_fetching|session_id=%s"
                        "|symbol=%s|interval=%s|from=%s|to=%s",
                        session_id,
                        market_data_request.symbol,
                        market_data_request.interval,
                        market_data_request.from_utc,
                        market_data_request.to_utc,
                    )
                    # Release the DB connection for the duration of the long network
                    # calls below (fetch_ohlcv_records, fetch_auxiliary_ohlcv,
                    # run_quant_backtest_sync — up to ~3 minutes for big crypto runs).
                    # SQLAlchemy auto-reacquires when the next db op runs at
                    # insert_chat_backtest_row(...) below. Prevents pool starvation
                    # under concurrent load.
                    await db.commit()
                    await db.close()
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
                            # SDL path: use artifact_to_yaml so the engine
                            # receives the exact contract the SDL compiled.
                            _bt_artifact = getattr(builder, "_sdl_artifact", None)
                            if _bt_artifact is not None:
                                try:
                                    from app.planner.evaluator import artifact_to_yaml as _a2y
                                    import tempfile, os as _os
                                    _yaml_str = _a2y(_bt_artifact)
                                    _tmp = tempfile.NamedTemporaryFile(
                                        mode="w", suffix=".yaml", delete=False
                                    )
                                    _tmp.write(_yaml_str)
                                    _tmp.close()
                                    yaml_path = _tmp.name
                                    logger.info(
                                        "📦 chat_flow|event=sdl_yaml_written|session_id=%s"
                                        "|artifact=%.8s|path=%s",
                                        session_id, _bt_artifact.artifact_id, yaml_path,
                                    )
                                except Exception as _y_err:
                                    logger.warning(
                                        "⚠️ chat_flow|event=sdl_yaml_fallback|err=%s",
                                        _y_err,
                                    )
                                    yaml_path = generate_yaml(builder)
                            else:
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
                    reference_ohlcv, htf_ohlcv = await fetch_auxiliary_ohlcv(
                        market_data_request,
                        main_fetch_interval=INTRABAR_EXECUTION_INTERVAL if intrabar_execution else None,
                    )
                    # Phase 11 — when 1-minute execution is on, fetch the 1m
                    # series for the engine (it resamples to the strategy
                    # timeframe for signals and walks minutes for fills/SL/TP).
                    # Enrichment above already ran on the strategy-timeframe
                    # `ohlcv_data`, so its calibration is unaffected.
                    if intrabar_execution:
                        engine_ohlcv = await fetch_ohlcv_records(
                            build_main_fetch_request(market_data_request, intrabar_execution=True)
                        )
                    else:
                        engine_ohlcv = ohlcv_data
                    backtest_result = await run_quant_backtest_sync(
                        yaml_path=yaml_path,
                        ohlcv_data=engine_ohlcv,
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
                    draft["backtest_window"] = {
                        "from_utc": _from_utc,
                        "to_utc": _to_utc,
                    }
                    ref_backtest_id = str(backtest_result.get("backtest_ref_id") or "")
                    if ref_backtest_id:
                        draft["backtest_ref_id"] = ref_backtest_id
                    
                    # Log token usage summary at backtest completion
                    log_token_summary(session_id, "backtest_complete")
                except BacktestWindowError as window_exc:
                    assistant_text = build_backtest_earliest_date_reply(
                        earliest_backtest_from_display(),
                    )
                    assistant_state = (
                        "assemble_strategy"
                        if user_state == "assemble_strategy"
                        else "backtest_confirmation"
                    )
                    draft = builder.to_draft_json(
                        mode_override=assistant_state,
                        processing_status=_draft_processing_status(builder, assistant_state),
                    )
                    draft["backtest_error"] = str(window_exc)
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
                sync_builder_risk_from_state(builder)
                builder.apply_defaults()
                strategy_payload = build_strategy_object(builder)
                await _upsert_builder_risk_execution_config(
                    db,
                    builder,
                    session_id=session_uuid,
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
                # SDL path: emit exact compiled artifact YAML for the engine.
                _asm_artifact = getattr(builder, "_sdl_artifact", None)
                if _asm_artifact is not None:
                    try:
                        from app.planner.evaluator import artifact_to_yaml as _a2y_asm
                        import tempfile as _tf
                        _yaml_str_asm = _a2y_asm(_asm_artifact)
                        _tmp_asm = _tf.NamedTemporaryFile(
                            mode="w", suffix=".yaml", delete=False
                        )
                        _tmp_asm.write(_yaml_str_asm)
                        _tmp_asm.close()
                        yaml_path = _tmp_asm.name
                        logger.info(
                            "📦 chat_flow|event=sdl_yaml_asm|session_id=%s"
                            "|artifact=%.8s|path=%s",
                            session_id, _asm_artifact.artifact_id, yaml_path,
                        )
                    except Exception as _ya_err:
                        logger.warning(
                            "⚠️ chat_flow|sdl_yaml_asm_fallback|err=%s", _ya_err
                        )
                        yaml_path = generate_yaml(builder)
                else:
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
                sync_builder_risk_from_state(builder)
                strategy_payload["strategy_object"]["risk_and_execution"] = (
                    build_risk_and_execution_from_builder(builder)
                )
                strategy_payload["strategy_object"]["reward_factor"] = (
                    round(float(builder.take_profit) / float(builder.stop_loss), 2)
                    if builder.stop_loss and builder.take_profit
                    else strategy_payload["strategy_object"].get("reward_factor")
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

                # Fidelity validation at final assembly — defense in depth.
                # Same prompt-vs-plan comparison as the plan_signals path, but
                # here we only attach the findings to the draft for the UI to
                # render. The strategy has already been persisted at this point
                # so we don't block; we just make sure the user can see any
                # remaining substitutions.
                final_fidelity_findings: list = []
                try:
                    from app.planner.fidelity_validator import (
                        validate_strategy_fidelity as _validate_fidelity_final,
                    )
                    # When the strategy was built via the SDL flow, hand the
                    # fidelity validator the SDL provenance so it trusts what the
                    # user actually stated (SL/TP/RR) instead of re-deriving it
                    # from the unreliable legacy rms_sources and raising false
                    # "system default" clarifications.
                    _sdl_for_fidelity = getattr(builder, "_sdl", None)
                    _sdl_field_sources = (
                        dict((_sdl_for_fidelity.provenance.field_sources or {}))
                        if _sdl_for_fidelity is not None else None
                    )
                    final_fidelity_findings = _validate_fidelity_final(
                        _resolve_strategy_source_prompt(
                            builder, user_content, all_messages,
                        ),
                        signal_plan=builder.signal_plan,
                        risk_execution_config=builder.risk_execution_config,
                        stop_loss_spec=getattr(builder, "stop_loss_spec", None),
                        sdl_field_sources=_sdl_field_sources,
                    )
                    logger.info(
                        "🔎 chat_flow|event=fidelity_check_done_at_assembly|session_id=%s"
                        "|critical=%d|warning=%d",
                        session_id,
                        sum(1 for f in final_fidelity_findings if f.severity == "critical"),
                        sum(1 for f in final_fidelity_findings if f.severity == "warning"),
                    )
                except Exception as _fid_err2:
                    logger.warning(
                        "⚠️ chat_flow|event=fidelity_check_error_at_assembly|session_id=%s|err=%s",
                        session_id, str(_fid_err2)[:200],
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
                    asset_class=builder.asset_class,
                )
                assistant_state = "assemble_strategy"
                draft = builder.to_draft_json(
                    mode_override="assemble_strategy",
                    processing_status=_draft_processing_status(builder, "assemble_strategy"),
                )
                retrieval_meta = build_retrieval_meta(strategy_payload)
                draft["kb_signals_used"] = retrieval_meta.get("signals_used", [])
                draft["kb_signals_available"] = retrieval_meta.get("signals_available", 0)
                if final_fidelity_findings:
                    draft["fidelity_findings"] = [
                        {
                            "severity": f.severity,
                            "code": f.code,
                            "field": f.field,
                            "message": f.message,
                            "user_value": f.user_value,
                            "assembled_value": f.assembled_value,
                        }
                        for f in final_fidelity_findings
                    ]
                    draft["fidelity_summary"] = {
                        "critical": sum(1 for f in final_fidelity_findings if f.severity == "critical"),
                        "warning":  sum(1 for f in final_fidelity_findings if f.severity == "warning"),
                        "requires_user_clarification": any(
                            f.severity == "critical" for f in final_fidelity_findings
                        ),
                    }
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

                # Phase 9b — discovery dispatch.
                # Both helpers are no-ops unless the active preset declares
                # a discovery block. handle_pending_tie_break runs first so
                # a user reply like "1" or "highest relative volume" is
                # interpreted as a tie-break choice instead of new strategy
                # input. When tie-break resolves to a single symbol, the
                # helper returns symbol_resolved=True; we DON'T short-circuit
                # in that case so the existing flow can pick up the new
                # symbol and continue to signal planning the same turn.
                discovery_step = await handle_pending_tie_break(builder, user_content)
                if not discovery_step.handled:
                    discovery_step = await maybe_dispatch_discovery(builder)

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
                # Phase 9e — discovery owns the turn whenever it has something
                # to say (awaiting tie-break OR scan produced none/multiple).
                # MUST be evaluated BEFORE grounded_clarification_reply and
                # direct_llm_reply, otherwise the agent router's "ask for a
                # specific stock" clarification short-circuits the discovery
                # prompt that would actually answer the user's request.
                # symbol_resolved=True falls through so the rest of the flow
                # picks up the newly-set symbol.
                elif discovery_step.handled and not discovery_step.symbol_resolved:
                    assistant_text = discovery_step.assistant_text
                    assistant_state = discovery_step.assistant_state or "collect_user_input"
                    draft = builder.to_draft_json(
                        mode_override=discovery_step.draft_mode,
                        processing_status=_draft_processing_status(builder, assistant_state),
                    )
                    logger.info(
                        "🔎 chat_flow|event=discovery_step|session_id=%s|state=%s|preset=%s",
                        session_id,
                        assistant_state,
                        builder.strategy_preset,
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
                    # ── Ask-before-default: require user confirmation on RMS ──
                    # Per the Stretus prompt §3: never silently fill stop_loss
                    # or take_profit. If both are absent and the user hasn't
                    # explicitly said "use defaults", pause and ask. We check
                    # rms_sources so the question is only asked once — once the
                    # user accepts or provides values, sources are marked and we
                    # fall through to planning on the next turn.
                    rms_sources = (builder.risk_execution_config or {}).get("rms_sources", {})

                    # Check if the user explicitly gave a structural SL (any preset,
                    # any strategy — no hardcoding of specific names).
                    from app.kb import kb as _kb_rms
                    _preset_obj = (
                        _kb_rms.lookup_preset(builder.strategy_preset)
                        if builder.strategy_preset else None
                    )
                    _preset_sl = getattr(_preset_obj, "stop_loss", None) or {}
                    has_structural_sl = (
                        builder.stop_loss_spec is not None
                        or "structural_sl" in rms_sources
                        or (isinstance(_preset_sl, dict) and _preset_sl.get("type") == "structural")
                    )
                    has_rr = "risk_reward" in rms_sources

                    # Structural SL + RR is a valid complete risk spec — no % needed
                    sl_missing = (
                        builder.stop_loss is None
                        and "stop_loss_pct" not in rms_sources
                        and not has_structural_sl
                    )
                    # When the user set RR, TP will be computed from SL × RR at planning
                    tp_missing = (
                        builder.take_profit is None
                        and "take_profit_pct" not in rms_sources
                        and not has_rr
                        and not has_structural_sl
                    )
                    user_said_use_defaults = re.search(
                        r"\b(use\s+defaults?|default\s+values?|go\s+with\s+defaults?"
                        r"|ok\s+defaults?|use\s+these\s+defaults?|proceed\s+with\s+defaults?"
                        r"|defaults?\s+(?:are\s+)?(?:fine|ok|good)|apply\s+defaults?)\b",
                        user_content, re.IGNORECASE,
                    )
                    if (sl_missing or tp_missing) and not user_said_use_defaults:
                        from app.kb.compat import get_risk_defaults
                        from app.services.strategy.risk_defaults import (
                            DEFAULT_RISK_REWARD,
                            resolve_default_stop_loss,
                            resolve_default_take_profit,
                        )
                        default_sl = float(
                            (builder.risk_execution_config or {}).get("stop_loss_pct")
                            or resolve_default_stop_loss(builder.objective, builder.experience).value
                        )
                        default_tp_rr = DEFAULT_RISK_REWARD
                        default_tp = round(
                            resolve_default_take_profit(
                                builder.objective, builder.experience, stop_loss=default_sl
                            ).value,
                            2,
                        )
                        default_dlc = 3.0
                        missing_lines = []
                        if sl_missing:
                            missing_lines.append(
                                f"• **Stop loss** — suggested default **{default_sl}%** "
                                f"(standard intraday ATR-based buffer)"
                            )
                        if tp_missing:
                            missing_lines.append(
                                f"• **Take profit** — suggested default **{default_tp}%** "
                                f"(1:{default_tp_rr} risk:reward on the above SL)"
                            )
                        dlc_missing = builder.daily_loss_cap is None and "daily_loss_cap" not in rms_sources
                        if dlc_missing:
                            missing_lines.append(
                                f"• **Daily loss cap** — suggested default **{default_dlc}%** of capital"
                            )
                        missing_block = "\n".join(missing_lines)
                        assistant_text = (
                            "Before I plan the signals, I need your call on risk. "
                            "You haven't specified:\n\n"
                            f"{missing_block}\n\n"
                            "**What would you like to do?**\n"
                            "1. Use these defaults — just say *\"use defaults\"*\n"
                            "2. Give me your own numbers — e.g. *\"SL 1%, TP 2%\"*\n"
                            "3. Mix — tell me which ones to keep and which to change"
                        )
                        assistant_state = "collect_user_input"
                        draft = builder.to_draft_json(
                            mode_override="collect_user_input",
                            processing_status="awaiting_user_input",
                        )
                        draft["rms_pending_confirmation"] = True
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
                        logger.info(
                            "⚖️ chat_flow|event=rms_clarification_sent|session_id=%s"
                            "|sl_missing=%s|tp_missing=%s",
                            session_id, sl_missing, tp_missing,
                        )
                        return

                    # Mark any still-missing RMS as accepted defaults so apply_defaults()
                    # knows the user is aware of them.
                    if sl_missing or tp_missing:
                        existing_rms = dict(builder.risk_execution_config or {})
                        existing_sources = dict(existing_rms.get("rms_sources", {}))
                        if sl_missing:
                            existing_sources["stop_loss_pct"] = "user_confirmed_default"
                        if tp_missing:
                            existing_sources["take_profit_pct"] = "user_confirmed_default"
                        existing_rms["rms_sources"] = existing_sources
                        builder.risk_execution_config = existing_rms

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

                    # ════════════════════════════════════════════════════════
                    # SDL PATH — try the SDL selector first.
                    # When the user's message is a full strategy description,
                    # the SDL selector (one LLM call) fills a typed form from
                    # the catalog, validates it, compiles it to the existing
                    # engine contract, and shows the user a read-back with
                    # match%.  Nothing is silently dropped or invented.
                    # On any failure the code falls through to the legacy pipeline.
                    # ════════════════════════════════════════════════════════
                    _sdl_prompt = _resolve_strategy_source_prompt(
                        builder, user_content, all_messages,
                    )
                    try:
                        from app.planner.sdl_flow import try_sdl_plan
                        _sdl_result = await try_sdl_plan(
                            _sdl_prompt, builder, session_id=session_id,
                        )
                    except Exception as _sdl_exc:
                        logger.warning(
                            "⚠️ chat_flow|event=sdl_flow_error|session_id=%s|err=%s",
                            session_id, _sdl_exc,
                        )
                        _sdl_result = None

                    if _sdl_result and _sdl_result.used_sdl:
                        # SDL path succeeded — use the SDL plan directly.
                        plan = _sdl_result.signal_plan
                        logger.info(
                            "✅ chat_flow|event=sdl_plan_used|session_id=%s"
                            "|match=%.0f%%|valid=%s|signals=%s",
                            session_id, _sdl_result.match_pct,
                            _sdl_result.validation_ok,
                            plan.get("signals_used", []),
                        )
                        # Skip the legacy pipeline and catalog picker entirely —
                        # jump straight to apply_signal_plan below.
                        semantic_instructions = None
                        semantic_gates_applied = []
                    else:
                        # ── Legacy pipeline (fallback) ─────────────────────
                        if _sdl_result:
                            logger.info(
                                "🔄 chat_flow|event=sdl_fallback|session_id=%s"
                                "|reason=%s — running legacy pipeline",
                                session_id, (_sdl_result.skip_reason or "unknown"),
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

                    # ───── Phase 14 + semantic: legacy path only ────────────
                    # Skip catalog picker and semantic extraction when the SDL
                    # path succeeded — the SDL selector already captured intent
                    # more accurately than the regex + ranker pipeline.
                    _used_sdl_path = bool(_sdl_result and _sdl_result.used_sdl)

                    if not _used_sdl_path:
                        # ── Phase 14: Catalog-driven signal picker ───────────
                        try:
                            from app.kb import kb as _kb_global
                            from app.planner.catalog_signal_picker import (
                                pick_plan_from_catalog,
                                merge_picker_with_preset,
                                auto_fill_missing_families,
                            )
                            from app.planner.semantic_extractor import SemanticExtractor as _SE
                            _picker_prompt = _resolve_strategy_source_prompt(
                                builder, user_content, all_messages,
                            )
                            _picker_sem = _SE().extract(_picker_prompt)
                            _picker_pick = pick_plan_from_catalog(
                                _picker_prompt,
                                kb=_kb_global,
                                timeframe=builder.timeframe,
                                sentiment=(builder.sentiment or "bullish"),
                                exit_on_opposite=bool(getattr(_picker_sem, "exit_on_opposite", False)),
                            )
                            plan, _merge_mode = merge_picker_with_preset(
                                plan, _picker_pick.signal_plan,
                                picker_confidence=_picker_pick.confidence,
                            )
                            plan["_catalog_picker_confidence"] = _picker_pick.confidence
                            plan["_catalog_picker_merge_mode"] = _merge_mode
                            _auto_audit = auto_fill_missing_families(
                                plan, _picker_prompt,
                                kb=_kb_global,
                                timeframe=builder.timeframe,
                                sentiment=(builder.sentiment or "bullish"),
                            )
                            if _auto_audit:
                                plan["_catalog_auto_fill"] = _auto_audit
                            logger.info(
                                "🧭 chat_flow|event=catalog_picker_done|session_id=%s"
                                "|confidence=%.3f|merge_mode=%s|auto_fill=%d",
                                session_id, _picker_pick.confidence, _merge_mode,
                                len(_auto_audit),
                            )
                        except Exception as _picker_err:
                            logger.warning(
                                "⚠️ chat_flow|event=catalog_picker_error|session_id=%s|err=%s",
                                session_id, str(_picker_err)[:200],
                            )
                        # ── End Phase 14 ──────────────────────────────────────

                        # ── Semantic extraction + constraint compiler ─────────
                        semantic_instructions = None
                        semantic_gates_applied = []
                        try:
                            semantic_extractor = SemanticExtractor()
                            semantic_instructions = semantic_extractor.extract(user_content)

                            logger.info(
                                "📊 chat_flow|event=semantic_extraction_done|session_id=%s"
                                "|family=%s|htf_rules=%d|quality=%.2f",
                                session_id,
                                semantic_instructions.strategy_family,
                                len(semantic_instructions.htf_rules),
                                semantic_instructions.extraction_quality_score,
                            )

                            orchestrator = ExecutionOrchestrator()
                            plan = orchestrator.apply_semantic_gates(plan, semantic_instructions)
                            semantic_gates_applied = plan.get("_semantic_gates_applied", [])

                            plan = apply_semantic_constraints(
                                plan,
                                builder,
                                semantic_instructions=semantic_instructions,
                                source_prompt=user_content,
                            )
                            semantic_gates_applied.extend(
                                plan.get("_constraint_compiler_applied") or []
                            )

                            logger.info(
                                "⚙️ chat_flow|event=semantic_gates_applied|session_id=%s|gates=%s",
                                session_id, ", ".join(semantic_gates_applied),
                            )
                        except Exception as e:
                            logger.warning(
                                "⚠️ chat_flow|event=semantic_extraction_error|session_id=%s|error=%s",
                                session_id, str(e),
                            )
                    # ── End legacy-only block ─────────────────────────────────
                        # Continue without semantic enhancement if extraction fails
                        pass
                    # ===== END: Semantic extraction and orchestration =====

                    logger.info(
                        "✅ chat_flow|event=signal_planning_done|session_id=%s"
                        "|signals=%s|available=%d|sl=%s%%|tp=%s%%|entry=%r|exit=%r|semantic_gates=%d",
                        session_id,
                        plan.get("signals_used", []),
                        plan.get("signals_available", 0),
                        plan.get("_sl_pct"),
                        plan.get("_tp_pct"),
                        (plan.get("entry_condition") or "")[:80],
                        (plan.get("exit_condition") or "")[:80],
                        len(semantic_gates_applied),
                    )
                    builder.apply_signal_plan(plan)

                    # ── Post-planning semantic sync-back (legacy path only) ───
                    # Skip for SDL path: artifact fields are already on builder.
                    if not _used_sdl_path and semantic_instructions:
                        sem_rr = plan.get("_semantic_risk_reward")
                        if sem_rr and sem_rr.get("ratio"):
                            existing_rms = builder.risk_execution_config or {}
                            rms_sources = existing_rms.get("rms_sources", {})
                            if rms_sources.get("risk_reward") != "user":
                                updated_rms = dict(existing_rms)
                                updated_rms["risk_reward"] = sem_rr["ratio"]
                                updated_sources = dict(rms_sources)
                                updated_sources["risk_reward"] = "semantic"
                                updated_rms["rms_sources"] = updated_sources
                                builder.risk_execution_config = updated_rms

                        # 2. Re-normalize the canonical intent now that we have
                        #    the confirmed preset, timeframe, and post-plan RMS
                        #    so the stored object is accurate for future turns.
                        try:
                            from app.planner.semantic_normalizer import SemanticIntentNormalizer
                            _post_canonical = SemanticIntentNormalizer().normalize(
                                semantic_instructions,
                                strategy_preset=builder.strategy_preset,
                                timeframe=builder.timeframe,
                                sentiment=builder.sentiment,
                                stop_loss_spec=builder.stop_loss_spec,
                                trailing_stop_spec=builder.trailing_stop_spec,
                                risk_execution_config=builder.risk_execution_config,
                                # Single source of truth for "the user's
                                # literal prompt". Walks builder →
                                # all_messages → user_content → goal.
                                source_prompt=_resolve_strategy_source_prompt(
                                    builder, user_content, all_messages,
                                )[:1000],
                            )
                            builder.semantic_intent = _post_canonical.dict()
                        except Exception as _norm_err:
                            logger.debug(
                                "chat_flow|post_semantic_normalize_failed|err=%s",
                                _norm_err,
                            )
                            # Fall back to storing raw SemanticInstructions dict
                            builder.semantic_intent = semantic_instructions.dict()
                    # ── Fidelity / SDL validation ─────────────────────────
                    # SDL path:    use SDL's own validation results — no regex
                    #              heuristics needed; provenance is authoritative.
                    # Legacy path: run the regex-based fidelity validator.
                    fidelity_findings: list = []
                    fidelity_user_message: str | None = None

                    if _used_sdl_path and _sdl_result:
                        # Map SDL validation errors → fidelity_findings shape
                        # so the rest of the draft/reply logic is unchanged.
                        for _ve in (_sdl_result.validation_errors or []):
                            from app.planner.fidelity_validator import FidelityFinding
                            fidelity_findings.append(FidelityFinding(
                                severity="critical",
                                code=_ve.get("code", "sdl_validation_error"),
                                message=_ve.get("message", ""),
                                field=_ve.get("field"),
                            ))
                        if not _sdl_result.validation_ok and fidelity_findings:
                            from app.planner.fidelity_validator import format_user_message as _fmt_f
                            fidelity_user_message = _fmt_f(fidelity_findings)
                    else:
                        try:
                            from app.planner.fidelity_validator import (
                                validate_strategy_fidelity,
                                format_user_message as _format_fidelity_message,
                            )
                            fidelity_source_prompt = _resolve_strategy_source_prompt(
                                builder, user_content, all_messages,
                            )
                            fidelity_findings = validate_strategy_fidelity(
                                fidelity_source_prompt,
                                signal_plan=plan,
                                risk_execution_config=builder.risk_execution_config,
                                stop_loss_spec=getattr(builder, "stop_loss_spec", None),
                            )
                            fidelity_user_message = _format_fidelity_message(
                                fidelity_findings
                            )
                            logger.info(
                                "🔎 chat_flow|event=fidelity_check_done|session_id=%s"
                                "|critical=%d|warning=%d|info=%d|prompt_len=%d|prompt_preview=%r",
                                session_id,
                                sum(1 for f in fidelity_findings if f.severity == "critical"),
                                sum(1 for f in fidelity_findings if f.severity == "warning"),
                                sum(1 for f in fidelity_findings if f.severity == "info"),
                                len(fidelity_source_prompt),
                                fidelity_source_prompt[:120],
                            )
                        except Exception as _fid_err:
                            logger.warning(
                                "⚠️ chat_flow|event=fidelity_check_error|session_id=%s|err=%s",
                                session_id, str(_fid_err)[:200],
                            )

                    # ── Build reply text ──────────────────────────────────────
                    # SDL path: read-back (Built/Assumed/Couldn't-do/match%) is the
                    #   primary content.  We prepend it to the plan summary.
                    # Legacy path: existing build_plan_signals_reply output.
                    assistant_text = build_plan_signals_reply(builder, plan)
                    if _used_sdl_path and _sdl_result and _sdl_result.readback_text:
                        assistant_text = (
                            _sdl_result.readback_text
                            + "\n\n"
                            + assistant_text
                        )
                        logger.info(
                            "📝 chat_flow|event=sdl_readback_prepended|session_id=%s"
                            "|match=%.0f%%",
                            session_id, _sdl_result.match_pct,
                        )

                    assistant_state = "plan_signals"
                    plan_status = "awaiting_confirmation"

                    # If any CRITICAL fidelity/SDL-validation finding, prepend the
                    # clarifying questions and block auto-advance to backtest.
                    critical_findings = [
                        f for f in fidelity_findings if f.severity == "critical"
                    ]
                    if critical_findings and fidelity_user_message:
                        assistant_text = (
                            f"{fidelity_user_message}\n\n"
                            "I have a draft plan ready below, but I'd like to "
                            "fix the items above first. Reply with the corrections "
                            "(e.g. \"use SMA 20\", \"RR is 3:7\", \"SL 1.5%\") and "
                            "I'll re-plan.\n\n"
                            f"{assistant_text}"
                        )
                        plan_status = "fidelity_review_required"
                        logger.warning(
                            "⚠️ chat_flow|event=fidelity_critical|session_id=%s"
                            "|count=%d|codes=%s",
                            session_id,
                            len(critical_findings),
                            ", ".join(f.code for f in critical_findings),
                        )
                    elif fidelity_user_message:
                        assistant_text = f"{assistant_text}\n\n{fidelity_user_message}"

                    if not _used_sdl_path and semantic_instructions and needs_manual_review(
                        semantic_instructions.extraction_quality_score
                    ):
                        # Don't downgrade a fidelity flag back to manual_review;
                        # fidelity is the stronger signal.
                        if plan_status == "awaiting_confirmation":
                            plan_status = "manual_review_required"
                        logger.warning(
                            "⚠️ chat_flow|event=low_semantic_quality|session_id=%s|score=%.3f",
                            session_id,
                            semantic_instructions.extraction_quality_score,
                        )
                    draft = builder.to_draft_json(
                        mode_override="plan_signals",
                        processing_status=plan_status,
                    )
                    draft["kb_signals_used"] = plan.get("signals_used", [])
                    draft["kb_signals_available"] = plan.get("signals_available", 0)

                    # Expose fidelity findings on the draft so the API/UI can
                    # render structured clarification prompts (one per finding)
                    # instead of the plain-text fallback message.
                    if fidelity_findings:
                        draft["fidelity_findings"] = [
                            {
                                "severity": f.severity,
                                "code": f.code,
                                "field": f.field,
                                "message": f.message,
                                "user_value": f.user_value,
                                "assembled_value": f.assembled_value,
                            }
                            for f in fidelity_findings
                        ]
                        draft["fidelity_summary"] = {
                            "critical": sum(1 for f in fidelity_findings if f.severity == "critical"),
                            "warning":  sum(1 for f in fidelity_findings if f.severity == "warning"),
                            "info":     sum(1 for f in fidelity_findings if f.severity == "info"),
                            "requires_user_clarification": any(
                                f.severity == "critical" for f in fidelity_findings
                            ),
                        }

                    # ── SDL provenance on draft (SDL path only) ───────────
                    if _used_sdl_path and _sdl_result:
                        _sdl = _sdl_result.sdl
                        if _sdl:
                            prov = _sdl.provenance
                            draft["sdl_provenance"] = {
                                "field_sources":         dict(prov.field_sources or {}),
                                "unmapped_details":      [u.model_dump() for u in (prov.unmapped_details or [])],
                                "clarifications_needed": [c.model_dump() for c in (prov.clarifications_needed or [])],
                            }
                            draft["sdl_match_pct"]     = round(_sdl_result.match_pct, 1)
                            draft["sdl_version"]        = _sdl.version
                            draft["sdl_content_hash"]   = _sdl.content_hash
                            draft["sdl_readback"]       = _sdl_result.readback_text
                            if _sdl_result.engine_gaps:
                                draft["sdl_engine_gaps"] = _sdl_result.engine_gaps
                        if _sdl_result.artifact:
                            draft["sdl_artifact_id"] = _sdl_result.artifact.artifact_id

                    # ── Semantic extraction results on draft (legacy path) ────
                    elif semantic_instructions:
                        draft["semantic_extraction"] = {
                            "quality_score": semantic_instructions.extraction_quality_score,
                            "strategy_family": semantic_instructions.strategy_family,
                            "htf_rules": [r.dict() for r in semantic_instructions.htf_rules],
                            "indicators": semantic_instructions.indicators,
                            "adx_threshold": (
                                semantic_instructions.volume_momentum.momentum.adx_threshold
                                if semantic_instructions.volume_momentum and semantic_instructions.volume_momentum.momentum
                                else None
                            ),
                            "structural_sl": semantic_instructions.stop_loss.dict() if semantic_instructions.stop_loss else None,
                            "risk_reward": semantic_instructions.risk_reward.dict() if semantic_instructions.risk_reward else None,
                            "candle_confirmation": semantic_instructions.candle_confirmation.dict() if semantic_instructions.candle_confirmation else None,
                        }
                        draft["semantic_gates_applied"] = semantic_gates_applied
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

            from app.services.chat.inputs_snapshot import append_inputs_snapshot

            assistant_text, draft = append_inputs_snapshot(
                assistant_text,
                builder,
                state=assistant_state,
                draft=draft if isinstance(draft, dict) else None,
            )

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
