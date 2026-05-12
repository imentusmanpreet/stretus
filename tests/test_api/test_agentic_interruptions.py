from __future__ import annotations

from app.services.ai.parser import detect_user_confirmation
from app.services.chat.strategy_flow import build_pause_workflow_reply


def test_negative_action_does_not_count_as_confirmation() -> None:
    assert detect_user_confirmation("run backtest")
    assert not detect_user_confirmation("don't run backtest")
    assert not detect_user_confirmation("No, don't run it")


def test_pause_reply_blocks_backtest_confirmation() -> None:
    assert build_pause_workflow_reply("backtest_confirmation") == (
        "Understood. I will not run the backtest. "
        "Tell me what you would like to do next: modify the strategy, review it, or pause here."
    )
