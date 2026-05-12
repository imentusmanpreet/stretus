from __future__ import annotations

from app.services.chat.chat_service import _prefer_agent_question_text
from app.services.chat.strategy_flow import select_direct_llm_reply
from app.services.strategy.builder import StrategyBuilder


def test_select_direct_llm_reply_uses_predefined_flow_for_collect_input() -> None:
    builder = StrategyBuilder()
    builder.symbol = "TCS.NS"

    reply = select_direct_llm_reply(
        route_intent="collect_input",
        route_reply_text="I noted TCS. What timeframe should I use?",
        user_confirmed=False,
        recognized_fields={"symbol"},
        builder=builder,
        snapshot_changed=True,
    )

    assert reply is None


def test_select_direct_llm_reply_keeps_predefined_validation_for_bad_timeframe() -> None:
    builder = StrategyBuilder()
    builder.timeframe_validation_message = "Unsupported timeframe"

    reply = select_direct_llm_reply(
        route_intent="collect_input",
        route_reply_text="I understood your request, but 12M is not supported here.",
        user_confirmed=False,
        recognized_fields={"timeframe"},
        builder=builder,
        snapshot_changed=False,
    )

    assert reply is None


def test_select_direct_llm_reply_skips_smalltalk_reply_when_fields_were_captured() -> None:
    builder = StrategyBuilder()
    builder.symbol = "TCS.NS"

    reply = select_direct_llm_reply(
        route_intent="general_chat",
        route_reply_text="Sure, I can help with that.",
        user_confirmed=False,
        recognized_fields={"symbol"},
        builder=builder,
        snapshot_changed=True,
    )

    assert reply is None


def test_agent_question_overrides_grounded_clarification_copy() -> None:
    route = {
        "agent_tool_parameters": {
            "field": "timeframe",
            "reason": "Invalid timeframe",
            "question": (
                "The timeframe 2m is not supported. Please choose from the following "
                "intervals: 1m, 5m, 10m, 15m, 30m, 1h, 1d."
            ),
        }
    }

    assert _prefer_agent_question_text(route, "Please confirm the timeframe to continue.") == (
        "The timeframe 2m is not supported. Please choose from the following "
        "intervals: 1m, 5m, 10m, 15m, 30m, 1h, 1d."
    )
