from __future__ import annotations

import asyncio

from app.services.agent.router import AgentDecision, AgentRouter
from app.services.agent.tool_catalog import AGENT_TOOL_SCHEMAS, AgentToolName
from app.services.strategy.builder import StrategyBuilder


def test_agent_tool_catalog_contains_required_backend_actions() -> None:
    names = {tool["function"]["name"] for tool in AGENT_TOOL_SCHEMAS}

    assert {
        "modify_strategy_inputs",
        "ask_user_for_clarification",
        "plan_strategy_signals",
        "assemble_strategy",
        "run_backtest",
        "mark_strategy_rejected",
        "start_new_strategy",
    } <= names


def test_agent_decision_maps_tool_call_to_legacy_route_fields() -> None:
    route = AgentDecision(
        tool_name=AgentToolName.MODIFY_STRATEGY_INPUTS.value,
        parameters={
            "session_id": "s1",
            "symbol": "INFY",
            "timeframe": "5m",
            "sentiment": "bullish",
        },
    ).to_legacy_route()

    assert route["intent"] == "collect_input"
    assert route["stock_query"] == "INFY"
    assert route["timeframe_input"] == "5m"
    assert route["sentiment"] == "bullish"
    assert set(route["recognized_fields"]) == {"symbol", "timeframe", "sentiment"}


def test_agent_decision_halts_on_strategy_rejection() -> None:
    route = AgentDecision(
        tool_name=AgentToolName.MARK_STRATEGY_REJECTED.value,
        parameters={
            "session_id": "s1",
            "reason": "Backtest is weak",
            "question": "I have paused this strategy. What would you like to change next?",
        },
    ).to_legacy_route()

    assert route["intent"] == "user_rejection"
    assert route["is_confirmation"] is False
    assert route["reply_text"] == "I have paused this strategy. What would you like to change next?"


def test_mark_strategy_rejected_requires_ai_question() -> None:
    rejected_tool = next(
        tool
        for tool in AGENT_TOOL_SCHEMAS
        if tool["function"]["name"] == "mark_strategy_rejected"
    )

    assert "question" in rejected_tool["function"]["parameters"]["required"]


def test_agent_decision_maps_market_and_risk_tools_to_executor_intents() -> None:
    market_route = AgentDecision(
        tool_name=AgentToolName.FETCH_MARKET_DATA.value,
        parameters={"symbol": "INFY.NS", "interval": "15m", "purpose": "market_inquiry"},
    ).to_legacy_route()
    risk_route = AgentDecision(
        tool_name=AgentToolName.UPDATE_RISK_EXECUTION_CONFIG.value,
        parameters={"session_id": "s1", "stop_loss_pct": 1.5},
    ).to_legacy_route()

    assert market_route["intent"] == "market_inquiry"
    assert risk_route["intent"] == "risk_execution_update"


def test_agent_router_uses_native_tool_call_response() -> None:
    class FakeLLM:
        async def chat_with_tools(self, messages, tools, *, tool_choice="auto"):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "name": "plan_strategy_signals",
                        "arguments": {"session_id": "session-1"},
                    }
                ],
            }

    builder = StrategyBuilder()
    builder.symbol = "INFY.NS"
    builder.timeframe = "15m"
    builder.objective = "intraday"
    builder.sentiment = "bullish"
    builder.experience = "expert"
    builder.goal = "breakout with controlled risk"
    builder.user_input_confirmed = True

    decision = asyncio.run(
        AgentRouter(FakeLLM()).decide(
            session_id="session-1",
            user_message="go ahead",
            builder=builder,
            previous_state="collect_user_input",
            recent_messages=[],
        )
    )

    assert decision.tool_name == "plan_strategy_signals"
    assert decision.to_legacy_route()["is_confirmation"] is True
