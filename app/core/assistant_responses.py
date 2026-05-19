"""
Compatibility wrapper for the centralized assistant response composer.
"""
from __future__ import annotations

from app.services.chat.response_composer import (
    build_ambiguous_stock_message,
    build_invalid_input_message,
    build_low_confidence_clarification_message,
    build_sebi_boundary_message,
    build_unsupported_stock_message,
    build_unsupported_timeframe_message,
    compose_response as compose_assistant_response,
    normalize_backtest_failure_reason,
)
from app.services.chat.response_templates import (
    ASSISTANT_RESPONSE_CODES,
    AssistantResponseCode,
    GOAL_EXAMPLES_TEXT,
    SUPPORTED_STOCK_SELECTION_PROMPT,
)

__all__ = [
    "ASSISTANT_RESPONSE_CODES",
    "AssistantResponseCode",
    "GOAL_EXAMPLES_TEXT",
    "SUPPORTED_STOCK_SELECTION_PROMPT",
    "build_ambiguous_stock_message",
    "build_invalid_input_message",
    "build_low_confidence_clarification_message",
    "build_sebi_boundary_message",
    "build_unsupported_stock_message",
    "build_unsupported_timeframe_message",
    "compose_assistant_response",
    "normalize_backtest_failure_reason",
]
