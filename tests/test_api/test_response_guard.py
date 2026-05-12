from __future__ import annotations

from app.core.assistant_responses import compose_assistant_response
from app.services.chat.response_guard import guard_dynamic_assistant_reply
from app.services.chat.strategy_flow import select_direct_llm_reply
from app.services.strategy.builder import StrategyBuilder


def test_guard_dynamic_assistant_reply_trims_casual_opener() -> None:
    assert guard_dynamic_assistant_reply("general_chat", "Sure, I can explain EMA for you.") == (
        "I can explain EMA for you."
    )


def test_guard_dynamic_assistant_reply_replaces_unsafe_advice_with_boundary_message() -> None:
    assert guard_dynamic_assistant_reply("general_chat", "You should buy this stock today.") == (
        compose_assistant_response("safety.stock_advice_boundary")
    )


def test_select_direct_llm_reply_does_not_use_dynamic_invalid_value_copy() -> None:
    builder = StrategyBuilder()

    reply = select_direct_llm_reply(
        route_intent="invalid_value",
        route_reply_text="That input does not work. Please try again.",
        user_confirmed=False,
        recognized_fields=set(),
        builder=builder,
        snapshot_changed=False,
    )

    assert reply is None
