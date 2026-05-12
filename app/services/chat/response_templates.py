"""
Shared constants and response-family identifiers for assistant messaging.
"""
from __future__ import annotations

from typing import Final, Literal

SUPPORTED_STOCK_SELECTION_PROMPT: Final[str] = (
    "TCS, Infosys, Reliance, Adani, HDFC Bank, NHPC, Suzlon, GMR Airports, or Vodafone Idea"
)
GOAL_EXAMPLES_TEXT: Final[str] = "Steady profit, Long-term steady growth, Aggressive"

FIELD_LABELS: Final[dict[str, str]] = {
    "symbol": "stock name or symbol",
    "timeframe": "timeframe",
    "objective": "trade type",
    "sentiment": "market view",
    "experience": "trading experience",
    "goal": "trading goal",
}

FIELD_EXAMPLES: Final[dict[str, str]] = {
    "symbol": SUPPORTED_STOCK_SELECTION_PROMPT,
    "timeframe": "15 minute, 30 minute, 1 hour, or 1 day",
    "objective": "intraday or positional",
    "sentiment": "bullish or bearish",
    "experience": "beginner, intermediate, or expert",
    "goal": GOAL_EXAMPLES_TEXT,
}

AssistantResponseCode = Literal[
    "collect_input.welcome",
    "collect_input.ask_symbol",
    "collect_input.ask_timeframe",
    "collect_input.ask_objective",
    "collect_input.ask_sentiment",
    "collect_input.ask_experience",
    "collect_input.ask_goal",
    "validation.invalid_input",
    "validation.low_confidence_clarification",
    "safety.stock_advice_boundary",
    "validation.unsupported_stock",
    "validation.unsupported_timeframe",
    "workflow.input_summary_confirmation",
    "workflow.signal_plan_ready",
    "workflow.strategy_ready_for_backtest",
    "workflow.backtest_complete",
    "workflow.backtest_failed",
    "workflow.backtest_already_available",
    "clarification.tutorial",
    "clarification.onboarding",
    "clarification.purpose_overview",
    "clarification.capability_examples",
    "clarification.ambiguous",
]

ASSISTANT_RESPONSE_CODES: Final[tuple[AssistantResponseCode, ...]] = (
    "collect_input.welcome",
    "collect_input.ask_symbol",
    "collect_input.ask_timeframe",
    "collect_input.ask_objective",
    "collect_input.ask_sentiment",
    "collect_input.ask_experience",
    "collect_input.ask_goal",
    "validation.invalid_input",
    "validation.low_confidence_clarification",
    "safety.stock_advice_boundary",
    "validation.unsupported_stock",
    "validation.unsupported_timeframe",
    "workflow.input_summary_confirmation",
    "workflow.signal_plan_ready",
    "workflow.strategy_ready_for_backtest",
    "workflow.backtest_complete",
    "workflow.backtest_failed",
    "workflow.backtest_already_available",
    "clarification.tutorial",
    "clarification.onboarding",
    "clarification.purpose_overview",
    "clarification.capability_examples",
    "clarification.ambiguous",
)
