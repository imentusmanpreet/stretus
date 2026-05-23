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


def _bare_stock_symbol(value: Any) -> str:
    text = _compact_text(value)
    return text.split(".", 1)[0] if text else ""


def _format_stock_option(option: Any, index: int) -> str:
    if isinstance(option, dict):
        symbol = _bare_stock_symbol(option.get("symbol")).lower()
        display_name = _compact_text(option.get("display_name"))
    else:
        symbol = _bare_stock_symbol(option).lower()
        display_name = ""
    if symbol and display_name:
        return f"{index}. {symbol} ({display_name})"
    return f"{index}. {symbol or display_name}"


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


def compose_response(code: AssistantResponseCode, **facts: Any) -> str:  # noqa: C901
    if code not in ASSISTANT_RESPONSE_CODES:
        raise ValueError(f"Unsupported assistant response code: {code}")

    if code == "collect_input.welcome":
        return (
            "## Welcome to Stretus\n\n"
            "Your AI-powered platform for building and backtesting **algorithmic trading strategies** on Indian equities \u2014 no coding required.\n\n"
            "Tell me a stock to get started and I\'ll design a data-driven strategy tailored to your style and market outlook.\n\n"
            "**What you\'ll get:**\n"
            "- Ranked entry & exit signal selection from a vetted knowledge base\n"
            "- Risk parameters calibrated to your experience level\n"
            "- Historical backtest with return, win rate, and grade\n\n"
            "**I\'ll need these 6 inputs:**\n\n"
            "| Input | Example |\n"
            "|-------|---------|\n"
            "| Stock | TCS, Reliance, HDFC Bank |\n"
            "| Timeframe | 5m / 15m / 1h / 1d |\n"
            "| Trade type | Intraday / Positional |\n"
            "| Market view | Bullish / Bearish |\n"
            "| Experience | Beginner / Intermediate / Expert |\n"
            "| Goal | Quick profits / Swing gains / Low-risk income |\n\n"
            "Which stock would you like to start with?"
        )

    if code == "validation.invalid_input":
        resolved_field = _compact_text(facts.get("field_name"), "value")
        label = FIELD_LABELS.get(resolved_field, resolved_field.replace("_", " "))
        example = FIELD_EXAMPLES.get(resolved_field, "please share the requested field in a simple format")
        return (
            f"Couldn\'t parse a valid **{label}** from that. "
            f"Try again \u2014 example: `{example}`"
        )

    if code == "validation.low_confidence_clarification":
        interpreted_values = _compact_text(facts.get("interpreted_values"))
        resolved_field = _compact_text(facts.get("missing_field"), "detail")
        label = FIELD_LABELS.get(resolved_field, resolved_field.replace("_", " "))
        if interpreted_values:
            return (
                f"Picked up: **{interpreted_values}**\n\n"
                f"Please confirm the **{label}** to continue."
            )
        return f"Please confirm the **{label}** to continue."

    if code == "safety.stock_advice_boundary":
        return (
            "I don\'t provide buy/sell calls or stock tips \u2014 that\'s outside my scope.\n\n"
            "What I *can* do is help you build and backtest a rules-based strategy for any stock you choose.\n\n"
            f"**Supported stocks:** {SUPPORTED_STOCK_SELECTION_PROMPT}\n\n"
            "Pick one to get started."
        )

    if code == "validation.unsupported_stock":
        supported_stocks_display = _compact_text(facts.get("supported_stocks_display"))
        return (
            "That stock isn\'t in my universe yet.\n\n"
            f"**Currently supported:** {supported_stocks_display}\n\n"
            "Please select one of these to continue."
        )

    if code == "validation.ambiguous_stock":
        stock_query = _compact_text(facts.get("stock_query"), "that")
        options = facts.get("stock_options")
        if not isinstance(options, list):
            options = []
        option_lines = "\n".join(
            f"- {_format_stock_option(option, idx)}"
            for idx, option in enumerate(options, start=1)
        )
        if option_lines:
            return (
                f"Found multiple matches for **\'{stock_query}\'**:\n\n"
                f"{option_lines}\n\n"
                "Which one did you mean?"
            )
        return (
            f"**\'{stock_query}\'** matched several symbols. "
            "Please type the full symbol or company name."
        )

    if code == "validation.unsupported_timeframe":
        supported_timeframes = _compact_text(facts.get("supported_timeframes"))
        return (
            "That timeframe isn\'t supported.\n\n"
            f"**Supported intervals:** {supported_timeframes}\n\n"
            "Pick one to continue."
        )

    if code in {
        "collect_input.ask_symbol",
        "collect_input.ask_timeframe",
        "collect_input.ask_objective",
        "collect_input.ask_sentiment",
        "collect_input.ask_experience",
        "collect_input.ask_goal",
    }:
        summary = _compact_text(facts.get("captured_summary"))
        preface_parts: list[str] = []
        if _compact_text(facts.get("preface")):
            preface_parts.append(_compact_text(facts.get("preface")))
        if summary:
            preface_parts.append(f"*Captured so far: {summary}*")

        if code == "collect_input.ask_symbol":
            question = (
                f"Which Indian stock would you like to build a strategy for?\n\n"
                f"**Available:** {SUPPORTED_STOCK_SELECTION_PROMPT}"
            )
        elif code == "collect_input.ask_timeframe":
            asset = _compact_text(facts.get("asset"))
            supported_timeframes = _compact_text(facts.get("supported_timeframes"))
            question = (
                f"What timeframe for **{asset}**? Supported: `{supported_timeframes}`"
                if asset
                else f"What timeframe? Supported: `{supported_timeframes}`"
            )
        elif code == "collect_input.ask_objective":
            asset = _compact_text(facts.get("asset"))
            question = (
                f"**{asset}** \u2014 intraday or positional?"
                if asset
                else "Intraday or positional?"
            )
        elif code == "collect_input.ask_sentiment":
            asset = _compact_text(facts.get("asset"))
            question = (
                f"Market view on **{asset}**: bullish or bearish?"
                if asset
                else "Market view: bullish or bearish?"
            )
        elif code == "collect_input.ask_experience":
            question = "Trading experience level: **beginner**, **intermediate**, or **expert**?"
        else:
            asset = _compact_text(facts.get("asset"))
            question = (
                f"What\'s your trading goal for **{asset}**?\n\n"
                f"*Examples: {GOAL_EXAMPLES_TEXT}*"
                if asset
                else (
                    f"What\'s your trading goal?\n\n"
                    f"*Examples: {GOAL_EXAMPLES_TEXT}*"
                )
            )

        preface = "\n\n".join(preface_parts)
        return f"{preface}\n\n{question}".strip() if preface else question

    if code == "workflow.input_summary_confirmation":
        asset = _compact_text(facts.get("asset"), "this asset")
        timeframe = _compact_text(facts.get("timeframe"))
        objective = _compact_text(facts.get("objective"))
        sentiment = _compact_text(facts.get("sentiment"))
        experience = _compact_text(facts.get("experience"))
        goal = _compact_text(facts.get("goal"))
        return (
            f"## Strategy Setup \u2014 {asset}\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Stock | **{asset}** |\n"
            f"| Timeframe | `{timeframe}` |\n"
            f"| Trade type | {objective} |\n"
            f"| Market view | {sentiment} |\n"
            f"| Experience | {experience} |\n"
            f"| Goal | {goal} |\n\n"
            "All inputs locked in. **Confirm to plan the signals**, or tell me what to change."
        )

    if code == "workflow.signal_plan_ready":
        asset = _compact_text(facts.get("asset"), "this asset")
        reminder = bool(facts.get("reminder"))
        if reminder:
            return (
                f"Signal plan for **{asset}** is ready and waiting.\n\n"
                "Confirm to assemble the strategy."
            )
        return (
            f"## Signal Plan Ready \u2014 {asset}\n\n"
            "Scanned the knowledge base, ranked the signals, locked in the best fit for your setup.\n\n"
            "**Confirm to assemble the strategy** \u2014 I\'ll wire up entry, exit, and risk parameters."
        )

    if code == "workflow.strategy_ready_for_backtest":
        asset = _compact_text(facts.get("asset"), "this asset")
        entry_names = _compact_text(facts.get("entry_names"))
        exit_names = _compact_text(facts.get("exit_names"))
        if entry_names or exit_names:
            return (
                f"## Strategy Assembled \u2014 {asset}\n\n"
                f"| Role | Signals |\n"
                f"|------|---------|\n"
                f"| Entry | `{entry_names or 'n/a'}` |\n"
                f"| Exit | `{exit_names or 'n/a'}` |\n\n"
                "Strategy is built and ready. **Confirm to run the backtest.**"
            )
        return (
            "## Strategy Ready\n\n"
            "All parameters locked in. **Confirm to run the backtest.**"
        )

    if code == "workflow.backtest_complete":
        asset = _compact_text(facts.get("asset"), "this asset")
        passed = bool(facts.get("passed"))
        total_return_pct = _format_percent(facts.get("total_return_pct"))
        win_rate_pct = _format_percent(facts.get("win_rate"))
        total_trades = _format_int(facts.get("total_trades"))
        grade = _compact_text(facts.get("overall_grade"))
        reason = _compact_text(facts.get("failure_reason"))

        verdict = "\u2705 **Passed**" if passed else "\u274c **Did not pass**"
        lines = [
            f"## Backtest Complete \u2014 {asset}",
            "",
            f"**Verdict:** {verdict}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if total_return_pct:
            lines.append(f"| Return | {total_return_pct} |")
        if win_rate_pct:
            lines.append(f"| Win Rate | {win_rate_pct} |")
        if total_trades:
            lines.append(f"| Total Trades | {total_trades} |")
        if grade:
            lines.append(f"| Grade | **{grade}** |")
        if not passed and reason:
            lines.append("")
            lines.append(f"> \u26a0\ufe0f **Reason:** {reason}")
        lines.append("")
        lines.append("Review the results above. Want to tweak parameters and re-run, or start a new setup?")
        return "\n".join(lines)

    if code == "workflow.backtest_failed":
        reason = normalize_backtest_failure_reason(facts.get("reason"))
        return f"## Backtest Failed\n\n> \u26a0\ufe0f {reason}\n\nReview the inputs and try again."

    if code == "workflow.backtest_already_available":
        return (
            "A backtest result is already available for this strategy.\n\n"
            "Review the existing result before triggering a new run."
        )

    if code == "clarification.tutorial":
        supported_timeframes = _compact_text(facts.get("supported_timeframes"))
        _tf = supported_timeframes or FIELD_EXAMPLES["timeframe"]
        _obj = FIELD_EXAMPLES["objective"]
        _sent = FIELD_EXAMPLES["sentiment"]
        _exp = FIELD_EXAMPLES["experience"]
        _goal = FIELD_EXAMPLES["goal"]
        return (
            "## How Stretus Works\n\n"
            "**6 steps from idea to backtest:**\n\n"
            f"1. **Stock** \u2014 pick from the supported universe ({SUPPORTED_STOCK_SELECTION_PROMPT})\n"
            f"2. **Timeframe** \u2014 {_tf}\n"
            f"3. **Trade type** \u2014 {_obj}\n"
            f"4. **Market view** \u2014 {_sent}\n"
            f"5. **Experience** \u2014 {_exp}\n"
            f"6. **Goal** \u2014 describe it in plain English ({_goal})\n\n"
            "Once I have those, I\'ll:\n"
            "- Pull the best-fit signals from the knowledge base\n"
            "- Assemble the full strategy with entry, exit, and risk rules\n"
            "- Run a historical backtest and report the results\n\n"
            "Name a stock whenever you\'re ready."
        )

    if code == "clarification.onboarding":
        return (
            "## Getting Started\n\n"
            "Three ways I can help:\n\n"
            f"1. **Build a strategy** \u2014 name any supported stock ({SUPPORTED_STOCK_SELECTION_PROMPT}) and I\'ll guide you through it\n"
            "2. **Learn how it works** \u2014 ask *\'walk me through it\'* for a step-by-step walkthrough\n"
            "3. **Understand a concept** \u2014 ask *\'what is RSI?\'*, *\'explain ORB\'*, etc.\n\n"
            "What would you like to do?"
        )

    if code == "clarification.purpose_overview":
        supported_timeframes = _compact_text(facts.get("supported_timeframes"))
        _tf = supported_timeframes or FIELD_EXAMPLES["timeframe"]
        return (
            "## What Stretus Does\n\n"
            "AI-powered strategy design and backtesting for **Indian equities** \u2014 no coding.\n\n"
            "**Core capabilities:**\n"
            "- Select entry & exit signals from a vetted indicator knowledge base\n"
            "- Apply risk parameters calibrated to your experience and volatility\n"
            "- Backtest on historical data \u2014 return, win rate, grade\n"
            "- Modify, reject, or restart at any step\n\n"
            f"**Universe:** {SUPPORTED_STOCK_SELECTION_PROMPT}\n\n"
            f"**Timeframes:** `{_tf}`\n\n"
            "Name a stock to start, or ask *\'how to use this tool\'* for the full walkthrough."
        )

    if code == "clarification.capability_examples":
        return (
            "## What You Can Ask\n\n"
            "**Start a strategy:**\n"
            "- *\'Intraday bullish TCS strategy on 15m for an intermediate trader\'*\n"
            "- *\'Build an ORB setup on HDFC Bank with 1:2 RR\'*\n\n"
            "**Modify inputs:**\n"
            "- *\'Change timeframe to 5m, keep everything else\'*\n"
            "- *\'Switch to bearish and re-plan\'*\n\n"
            "**Drive the workflow:**\n"
            "- *\'Plan the signals\'* \u2192 *\'Assemble the strategy\'* \u2192 *\'Run the backtest\'*\n\n"
            "**Learn:**\n"
            "- *\'What is RSI?\'*, *\'Explain EMA crossover\'*, *\'What stocks do you support?\'*\n\n"
            "Describe what you want in your own words \u2014 I\'ll figure out the rest."
        )

    if code == "clarification.ambiguous":
        return (
            "What would you like to do?\n\n"
            "1. **Build a strategy** \u2014 name a stock and I\'ll guide you\n"
            "2. **Learn how it works** \u2014 ask for a walkthrough\n"
            "3. **Understand a concept** \u2014 *\'what is RSI?\'*, *\'explain ORB\'*\n"
            "4. **Something else** \u2014 describe it\n\n"
            f"*Supported stocks: {SUPPORTED_STOCK_SELECTION_PROMPT}*"
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


def build_ambiguous_stock_message(stock_query: str, stock_options: list[dict[str, Any]]) -> str:
    return compose_response(
        "validation.ambiguous_stock",
        stock_query=stock_query,
        stock_options=stock_options,
    )


def build_unsupported_timeframe_message(supported_timeframes: str) -> str:
    return compose_response(
        "validation.unsupported_timeframe",
        supported_timeframes=supported_timeframes,
    )
