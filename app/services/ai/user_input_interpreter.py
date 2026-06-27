"""
LLM-assisted routing and interpretation for chat turns.

Every user message is sent here first so the model can understand the
conversation context before rule-based validation is applied.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.errors import AppError
from app.services.ai.llm import LLMService
from app.services.strategy.builder import StrategyBuilder

logger = logging.getLogger(__name__)

_ALLOWED_INTENTS = {
    "collect_input",
    "general_chat",
    "clarification",
    "stock_advice_request",
    "confirmation",
    "run_backtest",
    "modify_input",
    "new_strategy",
    "user_rejection",
    "pause_workflow",
    "invalid_value",
}
_ALLOWED_CLARIFICATION_TOPICS = {
    "status",
    "collected_inputs",
    "signal_plan",
    "strategy",
    "backtest",
    "next_step",
    "assistant_scope",
    "educational",
    # New onboarding-aware topics — let the LLM route meta-questions to a
    # structured templated reply instead of free-form text.
    "tutorial",
    "onboarding",
    "purpose_overview",
    "capability_examples",
    "ambiguous",
}

_ALLOWED_CONFIDENCE_LEVELS = {"high", "medium", "low"}

_FIELD_NAME_ALIASES = {
    "stock": "symbol",
    "stock_name": "symbol",
    "stock name": "symbol",
    "company": "symbol",
    "company_name": "symbol",
    "company name": "symbol",
    "symbol": "symbol",
    "ticker": "symbol",
    "timeframe": "timeframe",
    "time frame": "timeframe",
    "objective": "objective",
    "trade_type": "objective",
    "trade type": "objective",
    "sentiment": "sentiment",
    "market_view": "sentiment",
    "market view": "sentiment",
    "experience": "experience",
    "trading_experience": "experience",
    "trading experience": "experience",
    "goal": "goal",
    "trading_goal": "goal",
    "trading goal": "goal",
    "intent": "goal",
    "intention": "goal",
    "setup_goal": "goal",
    "setup goal": "goal",
}

_EXPERIENCE_MAP = {
    # Beginner — anyone signaling little/no experience.
    "beginner": "beginner",
    "novice": "beginner",
    "new": "beginner",
    "newbie": "beginner",
    "starter": "beginner",
    "fresher": "beginner",
    "naive": "beginner",
    "naïve": "beginner",
    "first time": "beginner",
    "first-time": "beginner",
    "first timer": "beginner",
    "amateur": "beginner",
    "rookie": "beginner",
    "noob": "beginner",
    "n00b": "beginner",
    "learner": "beginner",
    "student": "beginner",
    "illiterate": "beginner",
    "uneducated": "beginner",
    "untrained": "beginner",
    "inexperienced": "beginner",
    "no experience": "beginner",
    "zero experience": "beginner",
    "i am new": "beginner",
    "i'm new": "beginner",
    "low": "beginner",
    "basic": "beginner",
    "entry level": "beginner",
    "entry-level": "beginner",
    # Intermediate — moderate familiarity.
    "intermediate": "intermediate",
    "mid": "intermediate",
    "medium": "intermediate",
    "moderate": "intermediate",
    "average": "intermediate",
    "comfortable": "intermediate",
    "familiar": "intermediate",
    "some experience": "intermediate",
    "decent": "intermediate",
    "okay": "intermediate",
    # Expert — high proficiency.
    "expert": "expert",
    "advanced": "expert",
    "pro": "expert",
    "professional": "expert",
    "seasoned": "expert",
    "experienced": "expert",
    "veteran": "expert",
    "master": "expert",
    "high": "expert",
    "skilled": "expert",
    "proficient": "expert",
}

_OBJECTIVE_MAP = {
    "intraday": "intraday",
    "day": "intraday",
    "day_trade": "intraday",
    "day trading": "intraday",
    "same_day": "intraday",
    "quick_profit": "intraday",
    "short_term": "intraday",
    "positional": "positional",
    "swing": "positional",
    "carry": "positional",
    "multi_day": "positional",
    "longer_hold": "positional",
    "long_term": "positional",
}

_SENTIMENT_MAP = {
    "bullish": "bullish",
    "positive": "bullish",
    "optimistic": "bullish",
    "bearish": "bearish",
    "negative": "bearish",
    "cautious": "bearish",
}


def _clean_text(value: Any) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _clean_backtest_symbols(value: Any) -> list[str] | None:
    """Normalise the LLM's `backtest_symbols` into a clean list (or None).

    Accepts a JSON array, or a single comma/whitespace-delimited string (the
    model sometimes returns 'ETH_USDC, BTC_USDC'). Trims, upper-cases and
    de-duplicates while preserving order. Returns None when nothing usable —
    callers treat None as "single-asset / use the strategy symbol".
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        sym = str(item or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            cleaned.append(sym)
    return cleaned or None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    try:
        return json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _normalise_label(value: Any, mapping: dict[str, str]) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None

    direct = mapping.get(cleaned.lower())
    if direct:
        return direct

    compact = cleaned.lower().replace(" ", "_")
    return mapping.get(compact)


def _normalise_intent(value: Any) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return "collect_input"

    compact = cleaned.lower().replace(" ", "_")
    if compact in _ALLOWED_INTENTS:
        return compact
    if compact in {"general", "smalltalk", "small_talk", "chat"}:
        return "general_chat"
    if compact in {"clarify", "question"}:
        return "clarification"
    if compact in {"advice", "recommendation", "recommend"}:
        return "stock_advice_request"
    if compact in {"confirm", "approved"}:
        return "confirmation"
    if compact in {"run_backtest", "backtest", "execute_backtest"}:
        return "run_backtest"
    if compact in {
        "modify",
        "modify_inputs",
        "modify_input",
        "edit_input",
        "edit_inputs",
        "change_input",
        "change_inputs",
    }:
        return "modify_input"
    if compact in {
        "new",
        "fresh",
        "new_strategy",
        "fresh_strategy",
        "new_setup",
        "fresh_setup",
        "another_strategy",
        "another_setup",
        "start_over",
        "restart_strategy",
    }:
        return "new_strategy"
    if compact in {
        "reject",
        "rejection",
        "user_rejection",
        "negative_feedback",
        "dissatisfied",
        "dissatisfaction",
        "strategy_rejection",
        "backtest_rejection",
        "result_rejection",
        "market_view_rejection",
    }:
        return "user_rejection"
    if compact in {"pause", "cancel", "stop", "hold", "wait", "pause_workflow"}:
        return "pause_workflow"
    if compact in {"invalid", "irrelevant"}:
        return "invalid_value"
    return "collect_input"


def _normalise_field_name(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None

    direct = _FIELD_NAME_ALIASES.get(cleaned.lower())
    if direct:
        return direct

    compact = cleaned.lower().replace("_", " ")
    return _FIELD_NAME_ALIASES.get(compact)


def _normalise_clarification_topic(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None

    compact = cleaned.lower().replace(" ", "_")
    if compact in _ALLOWED_CLARIFICATION_TOPICS:
        return compact
    if compact in {"inputs", "captured_inputs", "collected"}:
        return "collected_inputs"
    if compact in {"signals", "plan"}:
        return "signal_plan"
    if compact in {"backtest_result", "results"}:
        return "backtest"
    if compact in {"scope", "capabilities"}:
        return "assistant_scope"
    if compact in {"education", "learning", "explanation"}:
        return "educational"
    if compact in {
        "tutorial_request",
        "walkthrough",
        "walk_through",
        "how_to_use",
        "how_to",
        "guide",
        "step_by_step",
        "step_guide",
    }:
        return "tutorial"
    if compact in {
        "new_user",
        "fresh_user",
        "getting_started",
        "get_started",
        "start_here",
        "i_am_new",
        "first_time",
    }:
        return "onboarding"
    if compact in {
        "purpose",
        "what_is_this",
        "what_does_this_do",
        "tool_purpose",
        "scope_overview",
        "what_can_you_do",
    }:
        return "purpose_overview"
    if compact in {
        "examples",
        "example",
        "sample",
        "samples",
        "demo",
        "what_can_i_ask",
        "show_examples",
    }:
        return "capability_examples"
    if compact in {
        "unclear",
        "unsure",
        "vague",
        "confused",
        "low_confidence",
    }:
        return "ambiguous"
    return None


def _normalise_confidence(value: Any) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return "high"
    compact = cleaned.lower().replace(" ", "_")
    if compact in _ALLOWED_CONFIDENCE_LEVELS:
        return compact
    if compact in {"hi", "strong", "certain", "sure"}:
        return "high"
    if compact in {"med", "ok", "okay", "uncertain", "maybe"}:
        return "medium"
    if compact in {"lo", "weak", "vague", "unclear", "guess"}:
        return "low"
    return "high"


def _signal_plan_summary(builder: StrategyBuilder) -> dict[str, Any] | None:
    if not builder.signal_plan:
        return None

    return {
        "entry": [
            item.get("name")
            for item in builder.signal_plan.get("entry", [])
            if isinstance(item, dict) and item.get("name")
        ],
        "exit": [
            item.get("name")
            for item in builder.signal_plan.get("exit", [])
            if isinstance(item, dict) and item.get("name")
        ],
        "rationale": _clean_text(builder.signal_plan.get("rationale")),
    }


def _count_user_messages(messages: list[Any]) -> int:
    count = 0
    for message in messages or []:
        role_obj = getattr(message, "role", None)
        role = getattr(role_obj, "value", role_obj)
        if str(role or "").lower() == "user":
            count += 1
    return count


def _builder_has_any_field(builder: StrategyBuilder) -> bool:
    return any(
        getattr(builder, field, None)
        for field in ("symbol", "timeframe", "objective", "sentiment", "experience", "goal")
    )


def _is_new_user_session(builder: StrategyBuilder, user_message_count: int) -> bool:
    """Best-effort flag: user has barely engaged with the workflow yet.

    True when the user has not provided any strategy field, no signal plan
    exists, and they have sent fewer than three messages. The router LLM uses
    this as a context signal to bias toward tutorial / onboarding replies for
    meta-questions like 'how to use this tool'.
    """
    if _builder_has_any_field(builder):
        return False
    if builder.signal_plan:
        return False
    return user_message_count <= 2


def _current_state(
    builder: StrategyBuilder,
    previous_state: str,
    *,
    user_message_count: int = 0,
) -> dict[str, Any]:
    return {
        "mode": previous_state,
        "symbol": builder.format_symbol() or builder.symbol,
        "timeframe": builder.timeframe,
        "objective": builder.objective,
        "goal": builder.goal,
        "sentiment": builder.sentiment,
        "experience": builder.experience,
        "user_input_confirmed": builder.user_input_confirmed,
        "has_signal_plan": bool(builder.signal_plan),
        "signal_plan_summary": _signal_plan_summary(builder),
        "user_message_count": user_message_count,
        "is_new_user_session": _is_new_user_session(builder, user_message_count),
        "has_any_field": _builder_has_any_field(builder),
    }


def _serialise_recent_messages(messages: list[Any], limit: int = 6) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []

    for message in reversed(messages):
        role_obj = getattr(message, "role", None)
        role = getattr(role_obj, "value", role_obj)
        if role == "system":
            continue

        content = _clean_text(getattr(message, "content", ""))
        if not content:
            continue

        draft = getattr(message, "strategy_draft", None) or {}
        state = draft.get("mode") or ""
        if len(content) > 280:
            content = content[:277].rstrip() + "..."

        history.append(
            {
                "role": str(role or ""),
                "state": state,
                "content": content,
            }
        )
        if len(history) >= limit:
            break

    history.reverse()
    return history


async def route_user_message(
    user_message: str,
    builder: StrategyBuilder,
    previous_state: str,
    recent_messages: list[Any],
) -> dict[str, Any]:
    """
    Route a user turn before any rule-based logic is applied.

    The LLM can answer normal conversation, detect policy-sensitive requests,
    extract strategy fields, and decide whether an invalid-value fallback is
    actually appropriate.
    """
    result: dict[str, Any] = {
        "intent": "collect_input",
        "reply_text": None,
        "stock_query": None,
        "timeframe_input": None,
        "objective": None,
        "goal": None,
        "sentiment": None,
        "experience": None,
        "invalid_field": None,
        "is_relevant": False,
        "is_confirmation": False,
        "needs_clarification": False,
        "clarification_field": None,
        "clarification_topic": None,
        "recognized_fields": [],
        "reasoning": None,
        "confidence": "high",
        "backtest_from_utc": None,
        "backtest_to_utc": None,
        # Multi-asset backtest: the assets to run when the user names more than
        # one in a run_backtest request. These drive the backtest only — they are
        # NOT a strategy-symbol change, so they must NOT populate stock_query.
        "backtest_symbols": None,
        # Risk-management settings — the LLM is the primary extractor for these.
        # Each is a number copied verbatim from the user's message, or None when
        # the user did not state it. Validated downstream before being stored.
        "stop_loss_pct": None,
        "take_profit_pct": None,
        "risk_reward": None,
        "multi_take_profit": None,
        "per_trade_risk": None,
        "daily_loss_cap": None,
        "max_trades": None,
    }

    llm = LLMService()
    user_message_count = _count_user_messages(recent_messages)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the first-pass routing brain for Stretus, an Indian-equity and crypto strategy assistant.\n"
                "Every user message comes to you before any rule-based logic.\n"
                "Your job is to understand the user in context first, then decide the best next action.\n\n"
                "Reason before you route. Do this in two steps inside the JSON output:\n"
                "  1. reasoning — one or two sentences in plain English describing what the user is trying to do, and what their context says about it. Use the conversation state (mode, missing fields, is_new_user_session, has_any_field) and the recent conversation, not just the latest message.\n"
                "  2. confidence — your confidence in the routing decision: high, medium, or low. If two interpretations are roughly equally plausible, mark it medium. If you genuinely cannot tell what the user wants, mark it low and route as clarification with topic 'ambiguous'.\n\n"
                "Allowed intents:\n"
                "- collect_input: the user provided one or more strategy-input values\n"
                "- general_chat: small talk, assistant identity questions, or general educational questions\n"
                "- clarification: the user asked about the current strategy-building process, collected values, current plan, current strategy, latest backtest, or next step\n"
                "- stock_advice_request: the user asked which stock to buy/sell, asked for tips, calls, or recommendations\n"
                "- confirmation: the user approved the current summary or next step\n"
                "- run_backtest: the user explicitly asked to run or execute a historical backtest "
                "(e.g. 'run backtest', 'backtest now', 'test on history from DATE to DATE'). "
                "Do NOT use run_backtest for generic yes/ok/proceed after assembly — those are confirmation only.\n"
                "- modify_input: the user wants to change one or more previously captured strategy inputs\n"
                "- new_strategy: the user wants to start a separate new strategy or reset the current strategy\n"
                "- user_rejection: the user is dissatisfied with or rejects the current strategy, signal plan, market view, or backtest result\n"
                "- pause_workflow: the user wants to stop, cancel, wait, or pause the workflow\n"
                "- invalid_value: the user seems to be answering the flow, but the response is unusable and not a genuine question\n\n"
                "Important behavior rules:\n"
                "- Be LLM-first. Do not assume every message is a field answer.\n"
                "- Understand intent from meaning, not from exact word matches. Affirmations can be subtle: 'i am happy', 'all good', 'looks fine', 'nothing to change', 'no changes needed', 'move to next step', 'lets go', 'proceed', 'plan', 'next', 'create and assemble' all mean confirmation when the prior assistant message asked for approval.\n"
                "- 'no' alone is a refusal, but 'no changes', 'no edits needed', 'no, looks good' are CONFIRMATIONS — the user is rejecting the *idea of changes*, not the *workflow*.\n"
                "- If the user types something unclear or off-topic while a confirmation is pending, prefer is_confirmation = false and clarification with topic 'next_step' over invalid_value, so the assistant can re-state the choice instead of looping.\n"
                "- Never ask the user to 'pick one of the six fields' more than twice. If they have already declined to pick a field once, treat further non-field replies as confirmation/general_chat and let the workflow advance.\n"
                "- If the user asks something normal like 'who are you', 'what can you do', or 'what is EMA', prefer general_chat or clarification.\n"
                "- If the user expresses uncertainty about whether the strategy will be profitable ('will this make profit', 'is this good', 'will it work'), classify as clarification with topic 'next_step' or 'educational' — do NOT classify as user_rejection just because the user is unsure.\n"
                "- If the user mixes a normal question with field values, still extract the field values.\n"
                "- Never recommend a stock or give buy/sell advice. Use stock_advice_request for that.\n"
                "- Stretus supports a large universe: 100+ Indian equities on NSE and 100+ crypto spot pairs (for example BTC/USDT, ETH/USDT, SOL/USDC). Do NOT gatekeep symbols yourself. Whenever the user names any stock, company, ticker, or crypto pair — Indian equity OR crypto — capture it verbatim in stock_query and let the backend validate it against the full universe. Never reject, 'correct', or claim a symbol is unsupported on your own; the app handles validation and will tell the user if something is not available. Crypto pairs such as BTC_USDT, BTC/USDT, ETH/USDT are fully supported — always capture them as the symbol. Because you do not give buy/sell advice (see above), never fabricate a specific ticker as a recommendation.\n"
                "- Re-validate the symbol on EVERY turn. A stock that was rejected earlier in this conversation may have failed only because of a typo or unusual word order — that earlier 'not supported' reply is NOT ground truth. Whenever the user names or re-states a stock (even one you previously thought was unsupported, e.g. after they retype or correct it), you MUST put that name in stock_query and let the backend decide again. Never reuse a past 'unsupported' verdict and never tell the user a stock is unsupported yourself.\n"
                "- If the user asks 'what stocks do you support', 'list the stocks', 'which stocks are available', or any equivalent meta-question about the universe, set intent to clarification with clarification_topic = assistant_scope and leave reply_text null. The app will return the canonical supported list directly.\n"
                "- Use invalid_value only when the message is not a genuine question and does not provide a usable value.\n"
                "- If you infer some values but still need explicit confirmation before the workflow should continue, set needs_clarification to true.\n"
                "- When needs_clarification is true, set clarification_field to the field that needs confirmation and keep reply_text as null.\n"
                "- If the current mode is collect_user_input and all six fields are already captured (missing_fields is empty) and the user says something that means 'proceed', 'plan', 'build', 'start', or any action command (e.g. 'plan signal', 'build strategy', 'generate signals', 'yes', 'ok', 'go ahead'), set intent to confirmation and is_confirmation to true.\n"
                "- If the current mode is plan_signals and the user gives a natural go-ahead to assemble, set intent to confirmation and is_confirmation to true.\n"
                "- If the current mode is assemble_strategy or backtest_confirmation and the user only affirms without asking to run a backtest, set intent to confirmation (not run_backtest).\n"
                "- When intent is run_backtest, set backtest_from_utc and backtest_to_utc to UTC ISO strings "
                "(YYYY-MM-DDTHH:MM:SSZ) for any explicit calendar range in the message. "
                "Use start-of-day UTC for the start date and end-of-day UTC (23:59:59Z) for the end date. "
                "If the user requests a backtest but gives no dates, leave both date fields null. "
                "If they give only a start or only an end date, set that field and leave the other null "
                "(the backend fills the missing bound with the product default).\n"
                "- When intent is run_backtest and the user names one or more assets to backtest "
                "(e.g. 'backtest ETH_USDC and BTC_USDC from ...', 'run it on SBIN, TCS'), put those "
                "assets in the backtest_symbols array using their canonical symbol form "
                "(e.g. [\"ETH_USDC\", \"BTC_USDC\"]). These drive the backtest only — they are NOT a "
                "request to change the strategy's symbol, so leave stock_query null in that case. "
                "Omit backtest_symbols (null) when the user names no asset or only confirms a re-run.\n"
                "- At any mode, if the user asks to modify, edit, update, change, revise, or correct previously captured inputs, set intent to modify_input.\n"
                "- If the user asks for a new, fresh, another, restart, or start-over strategy, set intent to new_strategy.\n"
                "- If the user gives negative feedback about the current strategy, signal plan, indicators, market view, or backtest result, set intent to user_rejection.\n"
                "- If the user says stop, cancel, wait, hold on, not now, or pause, set intent to pause_workflow and do not set is_confirmation true.\n"
                "- If the user names one of the six core fields while asking to modify inputs, extract any new value they provide for that field. If they only name the field and give no new value, leave the value fields null.\n"
                "- Treat compact timeframe units case-insensitively. For example, 15M means 15 minutes and 1D means 1 day.\n"
                "- Do not use month-based timeframes in the user flow.\n"
                "- Keep timeframe_input as the raw phrase the user gave. Do not snap, round, or substitute it with a supported value yourself. The app will validate it against the supported list and tell the user clearly when it is not supported.\n"
                "- Timeframes are flexible: any interval from 1 minute up to 1 day is supported (for example 1m, 2m, 5m, 7m, 15m, 45m, 1h, 4h, 1d) — NOT just a fixed list. Do not tell the user a timeframe in this range is unsupported. Sub-minute (e.g. '30 seconds') and multi-day/weekly (e.g. '3d', '1w') are out of range. Whatever the user says, pass their phrase verbatim in timeframe_input — never snap or substitute it; the app validates the 1m–1d range and replies if it is out of range.\n"
                "- The six core fields are: symbol, timeframe, objective, sentiment, experience, and goal.\n"
                "- Goal is a free-form natural-language summary of what kind of setup the trader wants.\n"
                "- Capture goal when the user describes intent such as breakout, trend-following, reversal, high conviction, low false signals, confirmation, or controlled risk.\n"
                "- Keep goal as one short plain-language phrase or sentence.\n"
                "- Capture risk-management settings when the user states them, copying the NUMBER exactly as written:\n"
                "    * stop_loss_pct — stop-loss percent, e.g. 'stop loss 0.5%', 'SL: 1%' → 0.5, 1.\n"
                "    * take_profit_pct — take-profit/target percent, e.g. 'take profit 1%', 'target 2%' → 1, 2.\n"
                "    * risk_reward — risk:reward as a single reward-per-risk number; '1:2 RR' or '2R' → 2; '1:3' → 3. Use this ONLY when the user names a single target with no per-tranche sizes.\n"
                "    * multi_take_profit — when the user specifies MULTIPLE exit targets with sizes (e.g. 'exit 30% at 1R, 30% at 2R, 40% at 4R' or 'take half at 1.5R, rest at 3R'), emit an array of {\"value\": <R-multiple>, \"exit_fraction\": <0-to-1>} objects in ascending value order. If the user says 'half' treat it as 0.5. If fractions are not stated, distribute equally. Set risk_reward to null when multi_take_profit is non-null.\n"
                "    * per_trade_risk — percent of capital risked per trade, e.g. 'risk 1% per trade', 'Per Trade Risk: 1%' → 1.\n"
                "    * daily_loss_cap — daily loss cap percent, e.g. 'daily loss cap 4%', 'stop after 3% daily loss' → 4, 3.\n"
                "    * max_trades — max trades per day as an integer, e.g. 'max 10 trades per day' → 10.\n"
                "- Copy these numbers verbatim. NEVER infer, round, derive, or fill a default — if the user did not state a field, leave it null. Do not compute take_profit from stop_loss and risk_reward; only capture what is written.\n"
                "- Do not ask follow-up questions about these risk fields; capture them silently when present.\n"
                "- Do not ask follow-up questions about trade duration if the user mentions it.\n"
                "- Map objective to only intraday or positional.\n"
                "- Map sentiment to only bullish or bearish.\n"
                "- Map experience to only beginner, intermediate, or expert.\n"
                "- Infer experience semantically from intent, not exact words. "
                "Treat phrases that signal little or no trading knowledge — for example "
                "'naive', 'new', 'newbie', 'novice', 'amateur', 'rookie', 'first time', "
                "'illiterate', 'uneducated', 'no idea', 'i don't know anything', 'just starting' "
                "— as beginner. Treat 'comfortable', 'some experience', 'familiar', 'okay', "
                "'medium', 'moderate' as intermediate. Treat 'advanced', 'pro', 'expert', "
                "'seasoned', 'veteran', 'experienced', 'master' as expert.\n"
                "- Never reject a self-described experience answer just because it does not "
                "use one of the exact three labels — always map it to the closest of "
                "beginner, intermediate, or expert when the meaning is reasonably clear.\n"
                "- Keep stock_query as the raw company or symbol phrase, not the full sentence.\n"
                "- reply_text should be short and written in standard business English.\n"
                "- Use simple, direct, and easy-to-understand sentences.\n"
                "- Avoid slang, filler, hype, idioms, or overly casual phrasing.\n"
                "- If some fields are still missing, reply_text may end with one clear next-step question when appropriate.\n\n"
                "Clarification topics — pick the MOST specific match. Prefer the structured topics over assistant_scope/educational so the app can render a templated reply instead of relying on free-form text.\n"
                "- status: the user asked for current status or where the workflow stands\n"
                "- collected_inputs: the user asked what has been captured so far\n"
                "- signal_plan: the user asked about the signal plan or selected signals\n"
                "- strategy: the user asked about the assembled strategy or current setup\n"
                "- backtest: the user asked about backtest status or results\n"
                "- next_step: the user asked what happens next\n"
                "- tutorial: the user wants a step-by-step walkthrough of how to use the platform (for example 'how to use stretus', 'how to use this tool', 'walk me through it', 'step by step', 'how do I start')\n"
                "- onboarding: the user signals they are new and does not yet know what to do (for example 'i am new', 'i don't know anything', 'help me start', 'kuch nahi pata', 'where to begin')\n"
                "- purpose_overview: the user asked what this tool is for, what it does, or its scope (for example 'what is the purpose of this tool', 'what does this do', 'what is stretus')\n"
                "- capability_examples: the user asked for examples of what they can ask or do (for example 'show me examples', 'what can i ask', 'give me a sample request')\n"
                "- ambiguous: you genuinely cannot tell what the user wants. Use this together with confidence = low.\n"
                "- assistant_scope: fallback only when none of the above structured topics fit\n"
                "- educational: the user asked for a general concept explanation, e.g. 'what is RSI', 'explain EMA crossover'\n"
                "- When clarification can be answered from current workflow state, set clarification_topic and keep reply_text as null.\n"
                "- Use reply_text for clarification only when the answer is not stateful, such as assistant scope or educational help, AND no structured topic fits.\n\n"
                "Few-shot examples (illustrative, not exhaustive):\n"
                "Example A — new user asking how to use the tool:\n"
                "  state: { mode: collect_user_input, is_new_user_session: true, has_any_field: false }\n"
                "  user message: 'how to use this tool'\n"
                "  → reasoning: 'New user wants a walkthrough of the platform.' intent: clarification, clarification_topic: tutorial, confidence: high.\n"
                "Example B — new user asking about the tool's purpose:\n"
                "  state: { mode: collect_user_input, is_new_user_session: true }\n"
                "  user message: 'what is the purpose of this tool?'\n"
                "  → reasoning: 'User wants the scope/overview of what the tool does.' intent: clarification, clarification_topic: purpose_overview, confidence: high.\n"
                "Example C — new user signalling they are clueless:\n"
                "  state: { mode: collect_user_input, is_new_user_session: true }\n"
                "  user message: 'i dont know anything to use this can you help me'\n"
                "  → reasoning: 'User is brand new and asking for guidance on getting started.' intent: clarification, clarification_topic: onboarding, confidence: high.\n"
                "Example D — value answer mid-flow:\n"
                "  state: { mode: collect_user_input, missing_fields: [symbol, timeframe, ...] }\n"
                "  user message: 'TCS'\n"
                "  → reasoning: 'User is naming a supported stock as the symbol field.' intent: collect_input, stock_query: 'TCS', confidence: high.\n"
                "Example E — vague request you cannot route:\n"
                "  state: { mode: collect_user_input, is_new_user_session: true }\n"
                "  user message: 'kuch karo bhai'\n"
                "  → reasoning: 'Too vague to route reliably; better to ask the user what they actually want.' intent: clarification, clarification_topic: ambiguous, confidence: low.\n"
                "Example F — confirmation after summary:\n"
                "  state: { mode: collect_user_input, missing_fields: [], user_input_confirmed: false }\n"
                "  user message: 'everything is fine, proceed to next'\n"
                "  → reasoning: 'User approves the captured inputs and wants the next step.' intent: confirmation, is_confirmation: true, confidence: high.\n\n"
                "Return JSON only in exactly this shape:\n"
                "{"
                "\"reasoning\": \"\","
                "\"confidence\": \"high\","
                "\"intent\": \"collect_input\","
                "\"reply_text\": null,"
                "\"stock_query\": null,"
                "\"timeframe_input\": null,"
                "\"objective\": null,"
                "\"goal\": null,"
                "\"sentiment\": null,"
                "\"experience\": null,"
                "\"invalid_field\": null,"
                "\"is_relevant\": false,"
                "\"is_confirmation\": false,"
                "\"needs_clarification\": false,"
                "\"clarification_field\": null,"
                "\"clarification_topic\": null,"
                "\"backtest_from_utc\": null,"
                "\"backtest_to_utc\": null,"
                "\"stop_loss_pct\": null,"
                "\"take_profit_pct\": null,"
                "\"risk_reward\": null,"
                "\"multi_take_profit\": null,"
                "\"per_trade_risk\": null,"
                "\"daily_loss_cap\": null,"
                "\"max_trades\": null"
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current state: {json.dumps(_current_state(builder, previous_state, user_message_count=user_message_count), ensure_ascii=True)}\n"
                f"Missing core fields: {json.dumps(builder.missing_user_input_fields(), ensure_ascii=True)}\n"
                f"Recent conversation: {json.dumps(_serialise_recent_messages(recent_messages), ensure_ascii=True)}\n"
                f"User message: {user_message}"
            ),
        },
    ]

    logger.info(
        "🧠 Routing brain | state=%s | user: %s",
        previous_state or "—",
        " ".join(str(user_message or "").split())[:160] or "<empty>",
    )
    try:
        payload = _extract_json_object(await llm.chat(messages, fast=True)) or {}
    except AppError:
        raise
    except Exception:
        logger.exception("💥 Routing brain failed to parse LLM output — using safe defaults")
        payload = {}

    result["reasoning"] = _clean_text(payload.get("reasoning"))
    result["confidence"] = _normalise_confidence(payload.get("confidence"))
    result["intent"] = _normalise_intent(payload.get("intent"))
    result["reply_text"] = _clean_text(payload.get("reply_text"))
    result["stock_query"] = _clean_text(payload.get("stock_query"))
    result["timeframe_input"] = _clean_text(payload.get("timeframe_input"))
    result["objective"] = _normalise_label(payload.get("objective"), _OBJECTIVE_MAP)
    result["goal"] = _clean_text(payload.get("goal"))
    result["sentiment"] = _normalise_label(payload.get("sentiment"), _SENTIMENT_MAP)
    result["experience"] = _normalise_label(payload.get("experience"), _EXPERIENCE_MAP)
    result["invalid_field"] = _normalise_field_name(payload.get("invalid_field"))
    result["is_confirmation"] = _coerce_bool(payload.get("is_confirmation"))
    result["needs_clarification"] = _coerce_bool(payload.get("needs_clarification"))
    result["clarification_field"] = _normalise_field_name(payload.get("clarification_field"))
    result["clarification_topic"] = _normalise_clarification_topic(payload.get("clarification_topic"))
    result["backtest_from_utc"] = _clean_text(payload.get("backtest_from_utc"))
    result["backtest_to_utc"] = _clean_text(payload.get("backtest_to_utc"))
    result["backtest_symbols"] = _clean_backtest_symbols(payload.get("backtest_symbols"))

    # Safety net (legacy/fallback path): if the model lumped multiple backtest
    # assets into stock_query (comma-separated) for a run_backtest request instead
    # of using backtest_symbols, recover them here so the run isn't misread as a
    # single-symbol change and reset to collect_user_input. Conservative: only
    # fires for run_backtest with a comma-delimited multi-symbol stock_query.
    if (
        result["intent"] == "run_backtest"
        and not result["backtest_symbols"]
        and result["stock_query"]
        and "," in result["stock_query"]
    ):
        recovered = _clean_backtest_symbols(result["stock_query"])
        if recovered and len(recovered) > 1:
            result["backtest_symbols"] = recovered
            result["stock_query"] = None

    # Risk-management numbers — coerce defensively (the LLM may echo "1%" or
    # "1.0" as a string). Stays None when absent; validated before storage.
    for _rms_key in ("stop_loss_pct", "take_profit_pct", "risk_reward",
                     "per_trade_risk", "daily_loss_cap"):
        raw = payload.get(_rms_key)
        if isinstance(raw, str):
            raw = raw.strip().rstrip("%").strip()
        result[_rms_key] = _coerce_float(raw)
    _max_trades = payload.get("max_trades")
    if isinstance(_max_trades, str):
        _max_trades = _max_trades.strip()
    _max_trades = _coerce_float(_max_trades)
    result["max_trades"] = int(_max_trades) if _max_trades is not None else None

    # multi_take_profit — a list of {value, exit_fraction} dicts or null.
    # Validate defensively: keep only well-formed dicts with numeric value.
    _mtp_raw = payload.get("multi_take_profit")
    if isinstance(_mtp_raw, list) and len(_mtp_raw) >= 2:
        _mtp_clean: list[dict] = []
        for _r in _mtp_raw:
            if not isinstance(_r, dict):
                continue
            try:
                _v = float(_r.get("value") or 0)
                _f = float(_r.get("exit_fraction") or 0)
            except (TypeError, ValueError):
                continue
            if _v > 0:
                _mtp_clean.append({"value": _v, "exit_fraction": max(0.0, min(1.0, _f))})
        result["multi_take_profit"] = _mtp_clean if len(_mtp_clean) >= 2 else None
    else:
        result["multi_take_profit"] = None

    recognized_fields: list[str] = []
    if result["stock_query"]:
        recognized_fields.append("symbol")
    if result["timeframe_input"]:
        recognized_fields.append("timeframe")
    if result["objective"]:
        recognized_fields.append("objective")
    if result["goal"]:
        recognized_fields.append("goal")
    if result["sentiment"]:
        recognized_fields.append("sentiment")
    if result["experience"]:
        recognized_fields.append("experience")

    result["recognized_fields"] = recognized_fields
    result["is_relevant"] = (
        _coerce_bool(payload.get("is_relevant"))
        or bool(recognized_fields)
        or bool(result["reply_text"])
        or result["intent"] in {
            "general_chat",
            "clarification",
            "stock_advice_request",
            "confirmation",
            "run_backtest",
            "new_strategy",
            "user_rejection",
            "pause_workflow",
        }
    )

    # Safety net: if the LLM marked confidence as low but did not pick a
    # clarification topic, force the ambiguous-clarification path so the user
    # gets a structured "what would you like to do?" question instead of a
    # confidently-wrong reply.
    if result["confidence"] == "low" and not recognized_fields:
        if result["intent"] not in {"clarification", "general_chat"}:
            result["intent"] = "clarification"
        if result["intent"] == "clarification" and not result["clarification_topic"]:
            result["clarification_topic"] = "ambiguous"
        result["needs_clarification"] = True

    logger.info(
        "🎯 Routed → intent=%s confidence=%s%s%s | why: %s",
        result["intent"],
        result["confidence"],
        f" fields={result['recognized_fields']}" if result["recognized_fields"] else "",
        f" topic={result['clarification_topic']}" if result["clarification_topic"] else "",
        _clean_text(result["reasoning"]) or "—",
    )
    return result


async def interpret_collect_user_input(user_message: str, builder: StrategyBuilder) -> dict[str, Any]:
    """
    Backward-compatible wrapper for callers that only want extracted fields.
    """
    return await route_user_message(user_message, builder, "collect_user_input", [])
