from __future__ import annotations


def _should_run_backtest(route_intent: str, user_state: str) -> bool:
    """Mirror of chat_service backtest trigger gate."""
    return route_intent == "run_backtest" and user_state in {
        "assemble_strategy",
        "backtest_confirmation",
        "backtest_complete",
    }


def test_generic_confirmation_does_not_trigger_backtest():
    assert not _should_run_backtest("confirmation", "assemble_strategy")
    assert not _should_run_backtest("confirmation", "backtest_confirmation")


def test_explicit_run_backtest_triggers():
    assert _should_run_backtest("run_backtest", "backtest_confirmation")
    assert _should_run_backtest("run_backtest", "assemble_strategy")
    assert _should_run_backtest("run_backtest", "backtest_complete")
