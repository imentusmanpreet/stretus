from __future__ import annotations

from app.core.assistant_responses import (
    ASSISTANT_RESPONSE_CODES,
    build_low_confidence_clarification_message,
    compose_assistant_response,
    normalize_backtest_failure_reason,
)


def test_assistant_response_code_count_matches_target_architecture() -> None:
    # 18 baseline codes + 5 onboarding-aware clarification topics
    # (tutorial, onboarding, purpose_overview, capability_examples, ambiguous).
    assert len(ASSISTANT_RESPONSE_CODES) == 23


def test_build_low_confidence_clarification_message_uses_standard_copy() -> None:
    assert build_low_confidence_clarification_message(
        interpreted_values="stock as TCS.NS and timeframe as 30m",
        missing_field="objective",
    ) == (
        "I understood the following from your message: stock as TCS.NS and timeframe as 30m. "
        "Please confirm the trade type to continue."
    )


def test_compose_assistant_response_returns_backtest_already_available_copy() -> None:
    assert compose_assistant_response("workflow.backtest_already_available") == (
        "A backtest result is already available for this strategy. "
        "Please review the existing result before starting a new run."
    )


def test_compose_assistant_response_renders_backtest_metrics() -> None:
    assert compose_assistant_response(
        "workflow.backtest_complete",
        asset="TCS.NS",
        passed=True,
        total_return_pct=8.4,
        win_rate=61.2,
        total_trades=14,
        overall_grade="B+",
    ) == (
        "The backtest for TCS.NS is complete.\n"
        "Result: Passed\n"
        "Strategy return: 8.40%\n"
        "Win rate: 61.20%\n"
        "Total trades: 14\n"
        "Overall grade: B+\n"
        "Please review the results before proceeding with any further changes."
    )


def test_normalize_backtest_failure_reason_humanizes_known_failures() -> None:
    assert normalize_backtest_failure_reason(
        "Market data request failed because the configured backtest window did not match OHLCV data."
    ) == (
        "The required market data could not be retrieved for the selected configuration. "
        "Please try again or adjust the stock or timeframe."
    )
