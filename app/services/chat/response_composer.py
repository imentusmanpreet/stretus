"""
Central response composer for assistant messaging.
"""
from __future__ import annotations

from typing import Any
import re

from app.services.chat.response_templates import (
    ASSISTANT_RESPONSE_CODES,
    AssistantResponseCode,
    FIELD_EXAMPLES,
    FIELD_LABELS,
    GOAL_EXAMPLES_TEXT,
    SUPPORTED_STOCK_SELECTION_PROMPT,
)


def _compact_text(value: Any, default: str = "") -> str:
    text = " ".join(str(value or "").split()).strip()
    return text or default


def _preface_lines(*lines: str) -> str:
    cleaned = [line for line in (_compact_text(item) for item in lines) if line]
    return "\n".join(cleaned)


def _format_percent(value: Any) -> str | None:
    try:
        if value is None or value == "":
            return None
        return f"{float(value):.2f}%"
    except Exception:
        return None


def _format_int(value: Any) -> str | None:
    try:
        if value is None or value == "":
            return None
        return str(int(value))
    except Exception:
        return None


def normalize_backtest_failure_reason(reason: str | None) -> str:
    message = _compact_text(reason)
    if not message:
        return "The backtest could not be completed for the selected configuration. Please review the inputs and try again."

    lowered = message.lower()
    if "quant engine" in lowered or "run-sync" in lowered:
        return "The quant engine service is currently unavailable. Please retry shortly."
    if (
        "market data" in lowered
        or "ohlcv" in lowered
        or "configured backtest window" in lowered
        or "missing strategy.symbol" in lowered
        or "missing strategy.timeframe" in lowered
    ):
        return (
            "The required market data could not be retrieved for the selected configuration. "
            "Please try again or adjust the stock or timeframe."
        )
    if "yaml" in lowered or "strategy configuration" in lowered:
        return "The strategy configuration could not be prepared for backtesting. Please reassemble the strategy and try again."
    if "no trades were executed" in lowered:
        return (
            "The strategy did not trigger any trades during the selected backtest window. "
            "Please review the strategy rules or try a different stock or timeframe."
        )
    if "minimum required" in lowered or "below the required" in lowered or "profit factor" in lowered:
        return message.rstrip(".") + ". Please review the strategy rules and try again."
    return message.rstrip(".") + ". Please review the configuration and try again."


def compose_response(code: AssistantResponseCode, **facts: Any) -> str:
    if code not in ASSISTANT_RESPONSE_CODES:
        raise ValueError(f"Unsupported assistant response code: {code}")

    if code == "collect_input.welcome":
        return (
            "Welcome to Stretus — your AI-powered platform for building and backtesting trading strategies without coding.\n\n"
            "Choose a stock to get started, and Stretus will create a data-driven strategy tailored to your preferences and market outlook.\n\n"
            "You’ll receive:\n"
            "• Optimized signal selection\n"
            "• Clear entry and exit rules\n"
            "• Risk and reward parameters\n"
            "• Backtest performance insights\n\n"
            f"Currently supported stocks: {SUPPORTED_STOCK_SELECTION_PROMPT}.\n\n"
            "Required Inputs (with examples):\n"
            "• Stock Selection: TCS\n"
            "• Timeframe: 5 Min / 15 Min / 1 Hour / Daily\n"
            "• Market View: Bullish / Bearish\n"
            "• Trading Style: Intraday / Positional\n"
            "• Experience Level: Beginner / Intermediate / Advanced\n"
            "• Trading Goal: Quick Profits / Swing Gains / Long-Term Growth / Low Risk Income"
        )

    if code == "validation.invalid_input":
        resolved_field = _compact_text(facts.get("field_name"), "value")
        label = FIELD_LABELS.get(resolved_field, resolved_field.replace("_", " "))
        example = FIELD_EXAMPLES.get(resolved_field, "please share the requested field in a simple format")
        return (
            f"I could not identify a valid {label} from your response. "
            f"Please provide it again. Example: {example}."
        )

    if code == "validation.low_confidence_clarification":
        interpreted_values = _compact_text(facts.get("interpreted_values"))
        resolved_field = _compact_text(facts.get("missing_field"), "detail")
        label = FIELD_LABELS.get(resolved_field, resolved_field.replace("_", " "))
        if interpreted_values:
            return (
                f"I understood the following from your message: {interpreted_values}. "
                f"Please confirm the {label} to continue."
            )
        return f"Please confirm the {label} to continue."

    if code == "safety.stock_advice_boundary":
        return (
            "I cannot provide personalized stock advice or buy/sell recommendations. "
            "However, I can help you build and evaluate a strategy for a stock you select.\n\n"
            f"Currently supported stocks: {SUPPORTED_STOCK_SELECTION_PROMPT}.\n"
            "Please pick one of these to continue."
        )

    if code == "validation.unsupported_stock":
        supported_stocks_display = _compact_text(facts.get("supported_stocks_display"))
        return (
            "This stock is not currently supported for strategy creation and backtesting.\n\n"
            f"Currently supported stocks are: {supported_stocks_display}. "
            "Please select one of these to continue."
        )

    if code == "validation.unsupported_timeframe":
        supported_timeframes = _compact_text(facts.get("supported_timeframes"))
        return (
            "The selected timeframe is not currently supported. "
            f"Please choose one of the supported intervals: {supported_timeframes}."
        )

    if code in {
        "collect_input.ask_symbol",
        "collect_input.ask_timeframe",
        "collect_input.ask_objective",
        "collect_input.ask_sentiment",
        "collect_input.ask_experience",
        "collect_input.ask_goal",
    }:
        preface = _preface_lines(
            facts.get("preface"),
            (
                f"I have recorded the following so far: {_compact_text(facts.get('captured_summary'))}."
                if _compact_text(facts.get("captured_summary"))
                else ""
            ),
        )
        if code == "collect_input.ask_symbol":
            question = f"Which Indian stock would you like to analyze? {SUPPORTED_STOCK_SELECTION_PROMPT}."
        elif code == "collect_input.ask_timeframe":
            asset = _compact_text(facts.get("asset"))
            supported_timeframes = _compact_text(facts.get("supported_timeframes"))
            if asset:
                question = f"Please confirm the timeframe for {asset}. Supported intervals are {supported_timeframes}."
            else:
                question = f"Please confirm the timeframe. Supported intervals are {supported_timeframes}."
        elif code == "collect_input.ask_objective":
            asset = _compact_text(facts.get("asset"))
            question = (
                f"Should this be an intraday or positional strategy for {asset}?"
                if asset
                else "Should this strategy be intraday or positional?"
            )
        elif code == "collect_input.ask_sentiment":
            asset = _compact_text(facts.get("asset"))
            question = (
                f"What is your market view on {asset}: bullish or bearish?"
                if asset
                else "What is your market view: bullish or bearish?"
            )
        elif code == "collect_input.ask_experience":
            question = "How would you describe your trading experience: beginner, intermediate, or expert?"
        else:
            asset = _compact_text(facts.get("asset"))
            question = (
                f"Please describe your trading goal for {asset} so we can tailor the strategy accordingly.\n"
                f"Examples: {GOAL_EXAMPLES_TEXT}.\n"
                "Enter your goal in simple terms to proceed."
                if asset
                else (
                    "Please describe your trading goal so we can tailor the strategy accordingly.\n"
                    f"Examples: {GOAL_EXAMPLES_TEXT}.\n"
                    "Enter your goal in simple terms to proceed."
                )
            )
        return f"{preface}\n{question}".strip() if preface else question

    if code == "workflow.input_summary_confirmation":
        # When a comprehensive summary body has already been rendered by the
        # caller (built from the SemanticExtractor over the user's full
        # prompt), use it verbatim — it covers every dimension the user can
        # have mentioned. The compact 6-line fallback below is used only on
        # legacy paths where no extractor has run.
        summary_text = facts.get("summary_text")
        if isinstance(summary_text, str) and summary_text.strip():
            return summary_text.strip()

        asset = _compact_text(facts.get("asset"), "this asset")
        timeframe = _compact_text(facts.get("timeframe"))
        objective = _compact_text(facts.get("objective"))
        sentiment = _compact_text(facts.get("sentiment"))
        experience = _compact_text(facts.get("experience"))
        goal = _compact_text(facts.get("goal"))
        return (
            f"I have captured the required inputs for {asset} on {timeframe} in the Indian stock market.\n"
            f"Objective: {objective}\n"
            f"Sentiment: {sentiment}\n"
            f"Experience: {experience}\n"
            f"Goal: {goal}\n\n"
            "Please confirm if these details are correct. I will then plan the signals."
        )

    if code == "workflow.missing_critical_inputs":
        items = facts.get("missing_items") or []
        if not isinstance(items, (list, tuple)) or not items:
            return (
                "Before I can build the signals, I need a couple of details "
                "that were not in your prompt. Please describe your stop loss "
                "and exit condition."
            )
        bullets: list[str] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            label = _compact_text(entry.get("label"), "")
            question = _compact_text(entry.get("question"), "")
            if label and question:
                bullets.append(f"- {label}: {question}")
            elif label:
                bullets.append(f"- {label}")
            elif question:
                bullets.append(f"- {question}")
        body = "\n".join(bullets) if bullets else "- exit rules"
        return (
            "Before I can build the entry and exit signals, I need a few "
            "details that were not in your prompt. I am not filling these "
            "in with defaults — please tell me what you want:\n"
            f"{body}\n\n"
            "Once you reply, I will update the summary and ask you to "
            "confirm before building."
        )

    if code == "workflow.signal_plan_ready":
        asset = _compact_text(facts.get("asset"), "this asset")
        reminder = bool(facts.get("reminder"))
        if reminder:
            return (
                f'The signal plan for "{asset}" is ready. '
                "Please confirm if you would like me to assemble the strategy."
            )
        return (
            f'I have reviewed the embedded knowledge base for "{asset}" and prepared the signal plan. '
            "Please confirm if you would like me to assemble the strategy."
        )

    if code == "workflow.strategy_ready_for_backtest":
        asset = _compact_text(facts.get("asset"), "this asset")
        entry_names = _compact_text(facts.get("entry_names"))
        exit_names = _compact_text(facts.get("exit_names"))
        if entry_names or exit_names:
            return (
                f"The strategy for {asset} has been assembled.\n"
                f"Entry signals: {entry_names or 'n/a'}\n"
                f"Exit signals: {exit_names or 'n/a'}\n"
                "The strategy is ready for backtesting. Please confirm if you would like me to proceed."
            )
        return "The strategy is ready for backtesting. Please confirm if you would like me to proceed."

    if code == "workflow.backtest_complete":
        asset = _compact_text(facts.get("asset"), "this asset")
        passed = bool(facts.get("passed"))
        total_return_pct = _format_percent(facts.get("total_return_pct"))
        win_rate_pct = _format_percent(facts.get("win_rate"))
        total_trades = _format_int(facts.get("total_trades"))
        grade = _compact_text(facts.get("overall_grade"))
        reason = _compact_text(facts.get("failure_reason"))

        lines = [
            f"The backtest for {asset} is complete.",
            f"Result: {'Passed' if passed else 'Did not pass'}",
        ]
        if total_return_pct:
            lines.append(f"Strategy return: {total_return_pct}")
        if win_rate_pct:
            lines.append(f"Win rate: {win_rate_pct}")
        if total_trades:
            lines.append(f"Total trades: {total_trades}")
        if grade:
            lines.append(f"Overall grade: {grade}")
        if not passed and reason:
            lines.append(f"Reason: {reason}")
        lines.append("Please review the results before proceeding with any further changes.")
        return "\n".join(lines)

    if code == "workflow.backtest_failed":
        reason = normalize_backtest_failure_reason(facts.get("reason"))
        return f"I was unable to complete the backtest. {reason}"

    if code == "workflow.backtest_already_available":
        return (
            "A backtest result is already available for this strategy. "
            "Please review the existing result before starting a new run."
        )

    if code == "clarification.tutorial":
        supported_timeframes = _compact_text(facts.get("supported_timeframes"))
        return (
            "Here is how Stretus works, step by step:\n"
            f"1. Pick a stock from the supported universe ({SUPPORTED_STOCK_SELECTION_PROMPT}).\n"
            f"2. Choose a {FIELD_LABELS['timeframe']} ({supported_timeframes or FIELD_EXAMPLES['timeframe']}).\n"
            f"3. Tell me your {FIELD_LABELS['objective']} ({FIELD_EXAMPLES['objective']}).\n"
            f"4. Share your {FIELD_LABELS['sentiment']} ({FIELD_EXAMPLES['sentiment']}).\n"
            f"5. Share your {FIELD_LABELS['experience']} ({FIELD_EXAMPLES['experience']}).\n"
            f"6. Describe your {FIELD_LABELS['goal']} in your own words ({FIELD_EXAMPLES['goal']}).\n\n"
            "Once these are captured, I will plan signals from the knowledge base, "
            "assemble the strategy, and run a backtest you can review.\n\n"
            "Whenever you are ready, name a stock from the list above and I will guide you through the rest."
        )

    if code == "clarification.onboarding":
        return (
            "Welcome. I can help you in three ways:\n"
            "1. Build and backtest a trading strategy. Just name one of the supported stocks "
            f"({SUPPORTED_STOCK_SELECTION_PROMPT}) and I will walk you through the inputs.\n"
            "2. Explain how this tool works step by step. Ask 'how to use this tool' or 'walk me through it'.\n"
            "3. Explain a trading concept (for example RSI, EMA crossover, ATR). Just ask 'what is RSI?'.\n\n"
            "Which of these would you like to start with?"
        )

    if code == "clarification.purpose_overview":
        supported_timeframes = _compact_text(facts.get("supported_timeframes"))
        return (
            "Stretus is an AI assistant that helps you design, validate, assemble, "
            "and backtest algorithmic trading strategies for Indian equities — without coding.\n\n"
            "What I can do for you:\n"
            "- Plan entry and exit signals using a knowledge base of vetted indicators.\n"
            "- Apply risk-aware defaults like stop loss, take profit, and daily loss cap based on your experience.\n"
            "- Run a historical backtest and report return, win rate, total trades, and a grade.\n"
            "- Let you modify, reject, or restart at any step.\n\n"
            f"Currently supported stocks: {SUPPORTED_STOCK_SELECTION_PROMPT}.\n"
            f"Supported timeframes: {supported_timeframes or FIELD_EXAMPLES['timeframe']}.\n\n"
            "Tell me a stock when you are ready, or ask 'how to use this tool' for a step-by-step walkthrough."
        )

    if code == "clarification.capability_examples":
        return (
            "Here are some things you can ask me:\n"
            "- 'Create an intraday bullish TCS strategy on 15m for a beginner with breakout goal'\n"
            "- 'Change timeframe to 5m and keep everything else'\n"
            "- 'Plan the signals' / 'assemble the strategy' / 'run the backtest'\n"
            "- 'What stocks do you support?'\n"
            "- 'What is RSI?' or 'Explain EMA crossover'\n\n"
            "Pick any of these or describe what you want in your own words."
        )

    if code == "clarification.ambiguous":
        return (
            "I want to make sure I help with the right thing. Could you tell me which of these you would like?\n"
            "1. Build or test a trading strategy (just name a supported stock to start).\n"
            "2. Learn how this tool works (a step-by-step walkthrough).\n"
            "3. Understand a trading concept (for example, ask 'what is RSI?').\n"
            "4. Something else — please describe it in your own words.\n\n"
            f"For reference, the supported stocks are: {SUPPORTED_STOCK_SELECTION_PROMPT}."
        )

    raise ValueError(f"Unsupported assistant response code: {code}")


def build_invalid_input_message(field_name: str | None) -> str:
    return compose_response("validation.invalid_input", field_name=field_name)


def build_low_confidence_clarification_message(
    *,
    interpreted_values: str,
    missing_field: str | None,
) -> str:
    return compose_response(
        "validation.low_confidence_clarification",
        interpreted_values=interpreted_values,
        missing_field=missing_field,
    )


def build_sebi_boundary_message() -> str:
    return compose_response("safety.stock_advice_boundary")


def build_unsupported_stock_message(supported_stocks_display: str) -> str:
    return compose_response(
        "validation.unsupported_stock",
        supported_stocks_display=supported_stocks_display,
    )


def build_unsupported_timeframe_message(supported_timeframes: str) -> str:
    return compose_response(
        "validation.unsupported_timeframe",
        supported_timeframes=supported_timeframes,
    )
