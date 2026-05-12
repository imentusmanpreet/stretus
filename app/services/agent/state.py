"""State projection helpers for the agent prompt."""
from __future__ import annotations

from typing import Any

from app.services.strategy.builder import StrategyBuilder


def _clean_text(value: Any) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _signal_names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [
        str(item.get("name")).strip()
        for item in items
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]


def _signal_plan_summary(builder: StrategyBuilder) -> dict[str, Any] | None:
    plan = builder.signal_plan if isinstance(builder.signal_plan, dict) else None
    if not plan:
        return None
    return {
        "entry": _signal_names(plan.get("entry")),
        "exit": _signal_names(plan.get("exit")),
        "signals_used": list(plan.get("signals_used") or []),
        "entry_condition": _clean_text(builder.entry_condition or plan.get("entry_condition")),
        "exit_condition": _clean_text(builder.exit_condition or plan.get("exit_condition")),
    }


def _backtest_summary(backtest_result: dict | None) -> dict[str, Any] | None:
    if not isinstance(backtest_result, dict):
        return None
    metrics = backtest_result.get("metrics") if isinstance(backtest_result.get("metrics"), dict) else {}
    assessment = backtest_result.get("assessment") if isinstance(backtest_result.get("assessment"), dict) else {}
    return {
        "backtest_ref_id": backtest_result.get("backtest_ref_id"),
        "passed": backtest_result.get("pass"),
        "failure_reason": backtest_result.get("failure_reason"),
        "total_return_pct": metrics.get("total_return_pct"),
        "win_rate": metrics.get("win_rate"),
        "total_trades": metrics.get("total_trades"),
        "overall_grade": assessment.get("overall_grade"),
    }


def serialise_recent_messages(messages: list[Any], limit: int = 6) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in reversed(messages):
        role_obj = getattr(message, "role", None)
        role = getattr(role_obj, "value", role_obj)
        if role == "system":
            continue

        content = _clean_text(getattr(message, "content", ""))
        if not content:
            continue

        draft = getattr(message, "strategy_draft", None) or {}
        state = draft.get("mode") or ""
        if len(content) > 320:
            content = content[:317].rstrip() + "..."

        history.append({"role": str(role or ""), "state": state, "content": content})
        if len(history) >= limit:
            break

    history.reverse()
    return history


def build_agent_state(
    *,
    session_id: str,
    builder: StrategyBuilder,
    previous_state: str,
    latest_strategy_context: dict | None = None,
    latest_backtest_result: dict | None = None,
    recent_messages: list[Any] | None = None,
) -> dict[str, Any]:
    context = latest_strategy_context if isinstance(latest_strategy_context, dict) else {}
    return {
        "session_id": session_id,
        "mode": previous_state or builder.get_mode(),
        "market": builder.market or "indian_stocks",
        "inputs": {
            "symbol": builder.format_symbol() or builder.symbol,
            "timeframe": builder.timeframe,
            "objective": builder.objective,
            "sentiment": builder.sentiment,
            "experience": builder.experience,
            "goal": builder.goal,
        },
        "missing_inputs": builder.missing_user_input_fields(),
        "user_input_confirmed": bool(builder.user_input_confirmed),
        "has_signal_plan": bool(builder.signal_plan),
        "signal_plan": _signal_plan_summary(builder),
        "strategy": {
            "strategy_id": context.get("strategy_id"),
            "yaml_path": context.get("yaml_path"),
            "has_strategy_object": bool(context.get("strategy_object")),
        },
        "latest_backtest": _backtest_summary(latest_backtest_result),
        "pending_input_modification": {
            "requested": bool(builder.input_modification_requested),
            "fields": list(builder.pending_input_modification_fields or []),
        },
        "recent_messages": serialise_recent_messages(recent_messages or []),
    }
