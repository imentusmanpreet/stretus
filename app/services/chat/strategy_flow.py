"""
Conversation helpers for the KB-driven strategy assistant.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.chat.response_composer import (
    build_invalid_input_message,
    build_low_confidence_clarification_message,
    build_sebi_boundary_message,
    compose_response,
)
from app.services.chat.response_guard import guard_dynamic_assistant_reply
from app.services.chat.response_templates import FIELD_LABELS
from app.services.strategy.builder import (
    CORE_USER_INPUT_FIELDS,
    StrategyBuilder,
    SUPPORTED_USER_TIMEFRAME_TEXT,
)

DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_USER_ID = "72abe85e-cd08-4e63-bd45-d54bb1d68478"
_GREETING_ONLY_RE = re.compile(r"^\s*(hi+|hello+|hey+|hii+|namaste)\s*[!.]*\s*$", re.IGNORECASE)
_INPUT_MODIFICATION_OPTIONS_TEXT = (
    "stock name or symbol, timeframe, trade type, market view, trading experience, or trading goal"
)


def build_welcome_message() -> str:
    return compose_response("collect_input.welcome")


def _asset_label(builder: StrategyBuilder) -> str:
    return builder.format_symbol() or builder.symbol or "this asset"


def build_invalid_user_input_message(field_name: str | None) -> str:
    return build_invalid_input_message(field_name)


def select_direct_llm_reply(
    route_intent: str,
    route_reply_text: str | None,
    user_confirmed: bool,
    recognized_fields: set[str],
    builder: StrategyBuilder,
    snapshot_changed: bool,
) -> str | None:
    if not route_reply_text or user_confirmed:
        return None

    if builder.symbol_validation_message or builder.timeframe_validation_message:
        return None

    if route_intent == "invalid_value":
        return None

    if route_intent == "collect_input":
        return None

    if route_intent in {"general_chat", "clarification"}:
        if recognized_fields or snapshot_changed:
            return None
        return guard_dynamic_assistant_reply(route_intent, route_reply_text)

    return None


def build_sebi_compliance_message() -> str:
    return build_sebi_boundary_message()


def build_pause_workflow_reply(current_state: str | None = None) -> str:
    state = (current_state or "").strip().lower()
    if state in {"assemble_strategy", "backtest_confirmation"}:
        return (
            "Understood. I will not run the backtest. "
            "Tell me what you would like to do next: modify the strategy, review it, or pause here."
        )
    if state == "plan_signals":
        return (
            "Understood. I will not assemble the strategy yet. "
            "Tell me what you would like to do next: modify inputs, review the signal plan, or pause here."
        )
    return "Understood. I will pause the workflow. What would you like to focus on next?"


def build_low_confidence_clarification_reply(
    *,
    interpreted_values: str,
    missing_field: str | None,
) -> str:
    return build_low_confidence_clarification_message(
        interpreted_values=interpreted_values,
        missing_field=missing_field,
    )


def _captured_input_summary(builder: StrategyBuilder) -> str:
    parts: list[str] = []
    asset = _asset_label(builder)
    if builder.symbol:
        parts.append(f"stock: {asset}")
    if builder.timeframe:
        parts.append(f"timeframe: {builder.timeframe}")
    if builder.objective:
        parts.append(f"trade type: {builder.objective}")
    if builder.sentiment:
        parts.append(f"market view: {builder.sentiment}")
    if builder.experience:
        parts.append(f"experience: {builder.experience}")
    if builder.goal:
        parts.append(f"goal: {builder.goal}")
    return "; ".join(parts)


def build_input_modification_field_prompt(builder: StrategyBuilder) -> str:
    summary = _captured_input_summary(builder)
    if summary:
        return (
            f"I can update the saved inputs. Current inputs: {summary}.\n"
            f"Which input would you like to change: {_INPUT_MODIFICATION_OPTIONS_TEXT}?"
        )
    return (
        "I can update the strategy inputs. "
        f"Which input would you like to change: {_INPUT_MODIFICATION_OPTIONS_TEXT}?"
    )


def build_input_modification_invalid_field_reply() -> str:
    return (
        "Please choose one of the six strategy inputs to change: "
        f"{_INPUT_MODIFICATION_OPTIONS_TEXT}."
    )


def _input_modification_preface(builder: StrategyBuilder) -> str:
    pending = [
        FIELD_LABELS.get(field, field.replace("_", " "))
        for field in CORE_USER_INPUT_FIELDS
        if field in set(builder.pending_input_modification_fields or [])
    ]
    if not pending:
        return "I will keep the other inputs unchanged."
    if len(pending) == 1:
        return f"I will keep the other inputs unchanged. Please provide the updated {pending[0]}."
    return "I will keep the other inputs unchanged. Please provide the updated values."


def _next_user_input_question(
    builder: StrategyBuilder,
    missing: list[str],
    *,
    preface: str | None = None,
    include_captured_summary: bool = False,
) -> str:
    if not missing:
        return compose_response(
            "collect_input.ask_symbol",
            preface=preface,
            captured_summary=_captured_input_summary(builder) if include_captured_summary else None,
        )

    next_field = missing[0]
    asset = _asset_label(builder)
    shared_facts: dict[str, Any] = {
        "preface": preface,
        "captured_summary": _captured_input_summary(builder) if include_captured_summary else None,
    }

    if next_field == "symbol":
        return compose_response("collect_input.ask_symbol", **shared_facts)
    if next_field == "timeframe":
        return compose_response(
            "collect_input.ask_timeframe",
            asset=asset if builder.symbol else None,
            supported_timeframes=SUPPORTED_USER_TIMEFRAME_TEXT,
            **shared_facts,
        )
    if next_field == "objective":
        return compose_response(
            "collect_input.ask_objective",
            asset=asset if builder.symbol else None,
            **shared_facts,
        )
    if next_field == "sentiment":
        return compose_response(
            "collect_input.ask_sentiment",
            asset=asset if builder.symbol else None,
            **shared_facts,
        )
    if next_field == "experience":
        return compose_response("collect_input.ask_experience", **shared_facts)
    if next_field == "goal":
        return compose_response(
            "collect_input.ask_goal",
            asset=asset if builder.symbol else None,
            **shared_facts,
        )
    return compose_response("collect_input.ask_symbol", **shared_facts)


def _is_greeting_only(user_message: str | None) -> bool:
    if not user_message:
        return False
    return bool(_GREETING_ONLY_RE.match(user_message))


def build_collect_user_input_reply(
    builder: StrategyBuilder,
    user_message: str | None = None,
    *,
    include_captured_summary: bool = False,
    preface: str | None = None,
) -> str:
    if builder.input_modification_requested and not builder.pending_input_modification_fields:
        return build_input_modification_field_prompt(builder)

    missing = builder.missing_user_input_fields()

    if not builder.is_user_input_complete():
        if builder.symbol_validation_message:
            return builder.symbol_validation_message
        if builder.input_validation_message:
            return builder.input_validation_message
        if builder.timeframe_validation_message:
            return builder.timeframe_validation_message

        effective_preface = preface
        if _is_greeting_only(user_message):
            effective_preface = (
                "Hello. I can assist you with building a strategy for the Indian stock market."
            )
        elif builder.input_modification_requested and builder.pending_input_modification_fields:
            effective_preface = effective_preface or _input_modification_preface(builder)

        return _next_user_input_question(
            builder,
            missing,
            preface=effective_preface,
            include_captured_summary=include_captured_summary,
        )

    builder.apply_defaults()
    summary_text: str | None = None
    if isinstance(builder.prompt_summary, dict):
        candidate = builder.prompt_summary.get("text")
        if isinstance(candidate, str) and candidate.strip():
            summary_text = candidate.strip()
    return compose_response(
        "workflow.input_summary_confirmation",
        asset=_asset_label(builder),
        timeframe=builder.timeframe,
        objective=builder.objective,
        sentiment=builder.sentiment,
        experience=builder.experience,
        goal=builder.goal,
        summary_text=summary_text,
    )


def build_missing_critical_inputs_reply(builder: StrategyBuilder) -> str:
    return compose_response(
        "workflow.missing_critical_inputs",
        missing_items=list(builder.missing_critical_inputs or []),
    )


def build_plan_signals_reply(builder: StrategyBuilder, plan: dict) -> str:
    return compose_response(
        "workflow.signal_plan_ready",
        asset=_asset_label(builder),
    )


def build_plan_signals_reminder(builder: StrategyBuilder) -> str:
    return compose_response(
        "workflow.signal_plan_ready",
        asset=_asset_label(builder),
        reminder=True,
    )


def build_assemble_strategy_reply(builder: StrategyBuilder, strategy_config: dict) -> str:
    entry = strategy_config.get("entry", [])
    exit_ = strategy_config.get("exit", [])

    entry_names = ", ".join(item["name"] for item in entry) if entry else "n/a"
    exit_names = ", ".join(item["name"] for item in exit_) if exit_ else "n/a"

    return compose_response(
        "workflow.strategy_ready_for_backtest",
        asset=_asset_label(builder),
        entry_names=entry_names,
        exit_names=exit_names,
    )


def build_backtest_ready_reminder() -> str:
    return compose_response("workflow.strategy_ready_for_backtest")


def build_backtest_result_reply(builder: StrategyBuilder, backtest_result: dict) -> str:
    metrics = backtest_result.get("metrics") if isinstance(backtest_result, dict) else {}
    assessment = backtest_result.get("assessment") if isinstance(backtest_result, dict) else {}

    return compose_response(
        "workflow.backtest_complete",
        asset=_asset_label(builder),
        passed=backtest_result.get("pass"),
        total_return_pct=(metrics or {}).get("total_return_pct"),
        win_rate=(metrics or {}).get("win_rate"),
        total_trades=(metrics or {}).get("total_trades"),
        overall_grade=(assessment or {}).get("overall_grade"),
        failure_reason=backtest_result.get("failure_reason"),
    )


def build_backtest_error_reply(reason: str) -> str:
    return compose_response(
        "workflow.backtest_failed",
        reason=reason,
    )


def _signal_names(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return ", ".join(
        str(item.get("name")).strip()
        for item in items
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    )


def _strategy_payload_for_reply(latest_strategy_context: dict | None) -> dict:
    if not isinstance(latest_strategy_context, dict):
        return {}

    strategy_config = latest_strategy_context.get("strategy_config")
    if isinstance(strategy_config, dict):
        return strategy_config

    strategy_object = latest_strategy_context.get("strategy_object")
    if isinstance(strategy_object, dict):
        return strategy_object

    return {}


def build_grounded_clarification_reply(
    builder: StrategyBuilder,
    *,
    current_state: str,
    clarification_topic: str | None,
    latest_backtest_result: dict | None = None,
    latest_strategy_context: dict | None = None,
) -> str | None:
    topic = (clarification_topic or "status").strip().lower()
    state = (current_state or builder.get_mode()).strip().lower()

    if topic == "backtest" and latest_backtest_result:
        return build_backtest_result_reply(builder, latest_backtest_result)

    strategy_payload = _strategy_payload_for_reply(latest_strategy_context)
    if topic == "strategy":
        if strategy_payload:
            return build_assemble_strategy_reply(builder, strategy_payload)
        if state in {"assemble_strategy", "backtest_confirmation"}:
            return build_backtest_ready_reminder()

    if topic == "signal_plan":
        if state in {"assemble_strategy", "backtest_confirmation"} and strategy_payload:
            return build_assemble_strategy_reply(builder, strategy_payload)
        if builder.signal_plan:
            return build_plan_signals_reminder(builder)

    # Onboarding-aware topics: render structured templated replies that pull
    # their content from the live data sources (supported stocks, supported
    # timeframes, field labels). Nothing about these answers is hardcoded
    # marketing copy — change the universe or the timeframes and these reply
    # automatically reflect the new state.
    if topic == "tutorial":
        return compose_response(
            "clarification.tutorial",
            supported_timeframes=SUPPORTED_USER_TIMEFRAME_TEXT,
        )
    if topic == "onboarding":
        return compose_response("clarification.onboarding")
    if topic == "purpose_overview":
        return compose_response(
            "clarification.purpose_overview",
            supported_timeframes=SUPPORTED_USER_TIMEFRAME_TEXT,
        )
    if topic == "capability_examples":
        return compose_response("clarification.capability_examples")
    if topic == "ambiguous":
        return compose_response("clarification.ambiguous")

    if topic in {"assistant_scope", "educational"}:
        return None

    if latest_backtest_result and state == "backtest_complete":
        return build_backtest_result_reply(builder, latest_backtest_result)

    if state in {"assemble_strategy", "backtest_confirmation"}:
        if strategy_payload:
            return build_assemble_strategy_reply(builder, strategy_payload)
        return build_backtest_ready_reminder()

    if builder.signal_plan:
        return build_plan_signals_reminder(builder)

    return build_collect_user_input_reply(
        builder,
        include_captured_summary=True,
        preface="Here is the current status.",
    )


def build_final_strategy_payload(
    session_id: str,
    user_id: str | None,
    strategy_object: dict,
    strategy_config: dict | None = None,
    strategy_id: str | None = None,
    yaml_path: str | None = None,
    current_mode: str = "assemble_strategy",
    next_state: str = "backtest_confirmation",
) -> dict:
    resolved_user_id = user_id or DEFAULT_USER_ID

    return {
        "context": {
            "session_id": session_id,
            "org_id": DEFAULT_ORG_ID,
            "user_id": resolved_user_id,
            "processing_status": "complete",
            "current_mode": current_mode,
            "next_state": next_state,
            "strategy_object": strategy_object,
            "strategy_config": strategy_config,
            "strategy_id": strategy_id,
            "yaml_path": yaml_path,
        },
        "success": True,
    }
