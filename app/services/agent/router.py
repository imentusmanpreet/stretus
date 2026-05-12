"""LLM tool router for the Stretus agent."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any

from app.services.agent.prompt import MASTER_AGENTIC_SYSTEM_PROMPT
from app.services.agent.state import build_agent_state
from app.services.agent.tool_catalog import (
    AGENT_TOOL_NAMES,
    AGENT_TOOL_SCHEMAS,
    AgentToolName,
)
from app.services.ai.llm import LLMService
from app.services.strategy.builder import StrategyBuilder

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """One structured backend action selected by the agent."""

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    assistant_text: str | None = None
    source: str = "tool_call"
    legacy_route: dict[str, Any] | None = None

    def to_legacy_route(self) -> dict[str, Any]:
        """Bridge tool decisions into the current chat_service fields."""
        if self.legacy_route is not None:
            route = dict(self.legacy_route)
            route.setdefault("agent_tool", self.tool_name)
            route.setdefault("agent_tool_parameters", dict(self.parameters or {}))
            route.setdefault("agent_source", self.source)
            return route

        params = dict(self.parameters or {})
        route: dict[str, Any] = {
            "intent": "collect_input",
            "reply_text": None,
            "stock_query": None,
            "timeframe_input": None,
            "objective": None,
            "goal": None,
            "sentiment": None,
            "experience": None,
            "invalid_field": None,
            "is_relevant": True,
            "is_confirmation": False,
            "needs_clarification": False,
            "clarification_field": None,
            "clarification_topic": None,
            "recognized_fields": [],
            "agent_tool": self.tool_name,
            "agent_tool_parameters": params,
            "agent_source": self.source,
        }

        if self.tool_name in {
            AgentToolName.MODIFY_STRATEGY_INPUTS.value,
            AgentToolName.START_NEW_STRATEGY.value,
        }:
            route["intent"] = (
                "new_strategy"
                if self.tool_name == AgentToolName.START_NEW_STRATEGY.value
                else "collect_input"
            )
            for param_key, route_key in (
                ("symbol", "stock_query"),
                ("timeframe", "timeframe_input"),
                ("objective", "objective"),
                ("goal", "goal"),
                ("sentiment", "sentiment"),
                ("experience", "experience"),
            ):
                if params.get(param_key) is not None:
                    route[route_key] = params[param_key]
                    route["recognized_fields"].append(
                        "symbol" if param_key == "symbol" else param_key
                    )
            return route

        if self.tool_name == AgentToolName.ASK_USER_FOR_CLARIFICATION.value:
            route["intent"] = "clarification"
            route["reply_text"] = self.assistant_text or params.get("question")
            route["needs_clarification"] = True
            route["clarification_field"] = params.get("field")
            return route

        if self.tool_name in {
            AgentToolName.PLAN_STRATEGY_SIGNALS.value,
            AgentToolName.ASSEMBLE_STRATEGY.value,
            AgentToolName.RUN_BACKTEST.value,
        }:
            route["intent"] = "confirmation"
            route["is_confirmation"] = True
            return route

        if self.tool_name == AgentToolName.GET_BACKTEST_RESULT.value:
            route["intent"] = "clarification"
            route["clarification_topic"] = "backtest"
            return route

        if self.tool_name == AgentToolName.FETCH_MARKET_DATA.value:
            route["intent"] = "market_inquiry"
            return route

        if self.tool_name == AgentToolName.UPDATE_RISK_EXECUTION_CONFIG.value:
            route["intent"] = "risk_execution_update"
            return route

        if self.tool_name == AgentToolName.MARK_STRATEGY_REJECTED.value:
            route["intent"] = "user_rejection"
            route["reply_text"] = self.assistant_text or params.get("question") or params.get("reason")
            return route

        if self.tool_name == AgentToolName.PAUSE_WORKFLOW.value:
            route["intent"] = "pause_workflow"
            route["reply_text"] = self.assistant_text or params.get("reason")
            return route

        if self.tool_name == AgentToolName.RESPOND_WITH_SAFETY_BOUNDARY.value:
            route["intent"] = "stock_advice_request"
            return route

        if self.tool_name == AgentToolName.RESPOND_TEXT.value:
            route["intent"] = "general_chat"
            route["reply_text"] = self.assistant_text or params.get("message")
            return route

        route["intent"] = "clarification"
        route["reply_text"] = "Please clarify what you would like to do next."
        route["needs_clarification"] = True
        route["clarification_field"] = "other"
        return route


class AgentRouter:
    """Select the next backend action using tool/function calling."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    async def decide(
        self,
        *,
        session_id: str,
        user_message: str,
        builder: StrategyBuilder,
        previous_state: str,
        recent_messages: list[Any],
        latest_strategy_context: dict | None = None,
        latest_backtest_result: dict | None = None,
    ) -> AgentDecision:
        state = build_agent_state(
            session_id=session_id,
            builder=builder,
            previous_state=previous_state,
            latest_strategy_context=latest_strategy_context,
            latest_backtest_result=latest_backtest_result,
            recent_messages=recent_messages,
        )
        messages = [
            {"role": "system", "content": MASTER_AGENTIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Conversation state JSON:\n"
                    f"{json.dumps(state, ensure_ascii=True)}\n\n"
                    f"User message:\n{user_message}"
                ),
            },
        ]

        try:
            response = await self._llm.chat_with_tools(messages, AGENT_TOOL_SCHEMAS)
            decision = _decision_from_tool_response(response)
            if decision is not None:
                _ensure_session_id(decision, session_id)
                return decision
        except Exception as exc:
            logger.warning(
                "agent_router|tool_call_failed|session_id=%s|err=%s|fallback=legacy_router",
                session_id,
                str(exc)[:160],
            )

        legacy_route = await _legacy_route_user_message(
            user_message,
            builder,
            previous_state,
            recent_messages,
        )
        decision = _decision_from_legacy_route(legacy_route, state)
        _ensure_session_id(decision, session_id)
        return decision


def _has_any_core_input(params: dict[str, Any]) -> bool:
    return any(
        params.get(field) is not None
        for field in ("symbol", "timeframe", "objective", "sentiment", "experience", "goal")
    )


def _ensure_session_id(decision: AgentDecision, session_id: str) -> None:
    if decision.tool_name != AgentToolName.RESPOND_TEXT.value:
        decision.parameters.setdefault("session_id", session_id)


def _decision_from_tool_response(response: dict[str, Any]) -> AgentDecision | None:
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        name = str(first.get("name") or "").strip()
        args = first.get("arguments") if isinstance(first.get("arguments"), dict) else {}
        if name in AGENT_TOOL_NAMES:
            return AgentDecision(tool_name=name, parameters=dict(args), source="native_tool_call")

    parsed = _parse_json_tool_call(response.get("content"))
    if parsed is not None:
        return parsed

    content = " ".join(str(response.get("content") or "").split()).strip()
    if content:
        return AgentDecision(
            tool_name=AgentToolName.RESPOND_TEXT.value,
            parameters={"message": content},
            assistant_text=content,
            source="assistant_text",
        )
    return None


def _parse_json_tool_call(content: Any) -> AgentDecision | None:
    text = str(content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        import re

        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("tool_name") or payload.get("name") or "").strip()
    params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    if name not in AGENT_TOOL_NAMES:
        return None
    return AgentDecision(tool_name=name, parameters=dict(params), source="json_tool_call")


async def _legacy_route_user_message(
    user_message: str,
    builder: StrategyBuilder,
    previous_state: str,
    recent_messages: list[Any],
) -> dict[str, Any]:
    from app.services.ai.user_input_interpreter import route_user_message

    return await route_user_message(user_message, builder, previous_state, recent_messages)


def _decision_from_legacy_route(route: dict[str, Any], state: dict[str, Any]) -> AgentDecision:
    intent = str(route.get("intent") or "collect_input")
    session_id = str(state.get("session_id") or "")
    params: dict[str, Any] = {"session_id": session_id}

    for route_key, param_key in (
        ("stock_query", "symbol"),
        ("timeframe_input", "timeframe"),
        ("objective", "objective"),
        ("goal", "goal"),
        ("sentiment", "sentiment"),
        ("experience", "experience"),
    ):
        if route.get(route_key) is not None:
            params[param_key] = route[route_key]

    if intent == "stock_advice_request":
        tool_name = AgentToolName.RESPOND_WITH_SAFETY_BOUNDARY.value
    elif intent == "modify_input":
        tool_name = AgentToolName.MODIFY_STRATEGY_INPUTS.value
    elif intent == "new_strategy":
        tool_name = AgentToolName.START_NEW_STRATEGY.value
    elif intent == "user_rejection":
        tool_name = AgentToolName.MARK_STRATEGY_REJECTED.value
        if route.get("reply_text"):
            params["question"] = route["reply_text"]
            params["reason"] = route["reply_text"]
        else:
            params["reason"] = "strategy_rejection"
        params["rejected_artifact"] = "unknown"
    elif intent == "clarification":
        tool_name = AgentToolName.ASK_USER_FOR_CLARIFICATION.value
        params["question"] = route.get("reply_text") or "Please clarify what you would like to do next."
        if route.get("clarification_field"):
            params["field"] = route["clarification_field"]
    elif intent == "general_chat":
        tool_name = AgentToolName.RESPOND_TEXT.value
        params = {"message": route.get("reply_text") or "I can help you build and backtest an Indian stock strategy."}
    elif intent == "confirmation" and route.get("is_confirmation"):
        mode = str(state.get("mode") or "")
        if mode == "plan_signals" and state.get("has_signal_plan"):
            tool_name = AgentToolName.ASSEMBLE_STRATEGY.value
        elif mode in {"assemble_strategy", "backtest_confirmation"}:
            tool_name = AgentToolName.RUN_BACKTEST.value
            strategy = state.get("strategy") if isinstance(state.get("strategy"), dict) else {}
            if strategy.get("strategy_id"):
                params["strategy_id"] = strategy["strategy_id"]
        else:
            tool_name = AgentToolName.PLAN_STRATEGY_SIGNALS.value
    else:
        tool_name = AgentToolName.MODIFY_STRATEGY_INPUTS.value if _has_any_core_input(params) else AgentToolName.ASK_USER_FOR_CLARIFICATION.value
        if tool_name == AgentToolName.ASK_USER_FOR_CLARIFICATION.value:
            params["question"] = route.get("reply_text") or "Please share the next strategy input."

    return AgentDecision(
        tool_name=tool_name,
        parameters=params,
        assistant_text=route.get("reply_text"),
        source="legacy_router_fallback",
        legacy_route=route,
    )
