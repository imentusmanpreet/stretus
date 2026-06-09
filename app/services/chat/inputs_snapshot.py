"""
Progressive stored-inputs summary for chat replies.

Values are read from the same ``strategy_draft`` payload persisted each turn
(merged with a fresh ``builder.to_draft_json()`` so per-turn draft extras
like ``backtest_window`` are included).
"""
from __future__ import annotations

import json
from typing import Any

from app.services.chat.response_templates import FIELD_LABELS
from app.services.strategy.builder import CORE_USER_INPUT_FIELDS, StrategyBuilder

_SUMMARY_SKIP_MARKERS = (
    "### Stored inputs",
)

# Keys shown in the compact table (order matters). Each entry is (draft_key, label).
_CORE_SUMMARY_KEYS: tuple[tuple[str, str], ...] = (
    ("symbol", "Stock"),
    ("timeframe", "Timeframe"),
    ("objective", "Trade type"),
    ("sentiment", "Market view"),
    ("experience", "Experience"),
    ("goal", "Goal"),
    ("strategy_preset", "Preset"),
    ("discovered_symbol", "Discovered symbol"),
)

def _format_signal_names(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return ", ".join(
        str(item.get("name")).strip()
        for item in items
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    )


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return not text or text.lower() in {
            "none",
            "null",
            "nil",
            "n/a",
            "na",
            "not applicable",
            "unknown",
            "unspecified",
            "not specified",
        }
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    return False


def _row(label: str, value: Any) -> dict[str, str] | None:
    if _is_empty(value):
        return None
    text = str(value).strip() if isinstance(value, str) else str(value)
    return {"label": label, "value": text}


def _working_draft(
    draft: dict[str, Any] | None,
    builder: StrategyBuilder,
    *,
    state: str,
) -> dict[str, Any]:
    """Merge persisted draft extras with the latest builder snapshot.

    Builder fields (``to_draft_json``) win on conflict; keys only present on
    the turn draft (e.g. ``backtest_window``, ``kb_signals_used``) are kept.
    """
    fresh = builder.to_draft_json(mode_override=state or builder.get_mode())
    working = dict(fresh)
    if isinstance(draft, dict):
        for key, value in draft.items():
            if key == "inputs_snapshot":
                continue
            if key not in fresh:
                working[key] = value
    return working


_USER_RMS_SOURCES = frozenset({"user", "user_confirmed_default"})
_RMS_PCT_BUILDER_FIELDS = {
    "stop_loss_pct": "stop_loss",
    "take_profit_pct": "take_profit",
}


def _rms_pct_from_agent_decision(draft: dict[str, Any], rms_key: str) -> float | None:
    """Read pending SL/TP from agent tool params when draft fields are still null."""
    agent_key = "stop_loss" if rms_key == "stop_loss_pct" else "take_profit"
    decision = draft.get("agent_decision")
    if not isinstance(decision, dict):
        return None
    params = decision.get("parameters")
    if not isinstance(params, dict) or agent_key not in params:
        return None
    raw = params[agent_key]
    if isinstance(raw, dict):
        try:
            return float(raw["value"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("value") is not None:
                return float(parsed["value"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        try:
            return float(raw.strip().rstrip("%"))
        except ValueError:
            return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _user_supplied_rms_pct(draft: dict[str, Any], rms_key: str) -> float | None:
    """Return SL/TP only when the user (or agent on their behalf) set them.

    Tier/system defaults applied by ``apply_defaults`` are intentionally hidden
    from the progressive summary until the user engages with risk inputs.
    """
    risk = draft.get("risk_execution_config")
    if not isinstance(risk, dict):
        risk = {}
    sources = risk.get("rms_sources") or {}
    source = sources.get(rms_key)

    overrides = draft.get("_phase10_user_overrides") or []
    builder_field = _RMS_PCT_BUILDER_FIELDS.get(rms_key)
    if builder_field and builder_field in overrides:
        val = draft.get(rms_key)
        if val is not None:
            return float(val)

    if source in _USER_RMS_SOURCES:
        val = draft.get(rms_key)
        if val is None:
            val = risk.get(rms_key)
        if val is not None:
            return float(val)

    return _rms_pct_from_agent_decision(draft, rms_key)


def _summary_rows_from_draft(draft: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for key, label in _CORE_SUMMARY_KEYS:
        row = _row(label, draft.get(key))
        if row:
            rows.append(row)

    plan = draft.get("signal_plan")
    if isinstance(plan, dict):
        entry_names = _format_signal_names(plan.get("entry"))
        exit_names = _format_signal_names(plan.get("exit"))
        if entry_names:
            rows.append({"label": "Entry signals", "value": entry_names})
        elif draft.get("entry_condition"):
            rows.append({"label": "Entry condition", "value": str(draft["entry_condition"])})
        if exit_names:
            rows.append({"label": "Exit signals", "value": exit_names})
        elif draft.get("exit_condition"):
            rows.append({"label": "Exit condition", "value": str(draft["exit_condition"])})
    else:
        if draft.get("entry_condition"):
            rows.append({"label": "Entry condition", "value": str(draft["entry_condition"])})
        if draft.get("exit_condition"):
            rows.append({"label": "Exit condition", "value": str(draft["exit_condition"])})

    risk = draft.get("risk_execution_config")
    if not isinstance(risk, dict):
        risk = {}
    sources = risk.get("rms_sources") or {}

    sl = _user_supplied_rms_pct(draft, "stop_loss_pct")
    tp = _user_supplied_rms_pct(draft, "take_profit_pct")
    daily_cap = (
        risk.get("daily_loss_cap_pct") or risk.get("daily_loss_cap")
        if sources.get("daily_loss_cap") in _USER_RMS_SOURCES
        else None
    )
    max_trades = (
        risk.get("max_trades_per_day") or risk.get("max_trades")
        if sources.get("max_trades") in _USER_RMS_SOURCES
        else None
    )

    for label, raw in (
        ("Stop loss", f"{sl}%" if sl is not None else None),
        ("Take profit", f"{tp}%" if tp is not None else None),
        ("Daily loss cap", f"{daily_cap}%" if daily_cap not in (None, 0, 0.0) else None),
        ("Max trades / day", str(max_trades) if max_trades not in (None, 0) else None),
    ):
        row = _row(label, raw)
        if row:
            rows.append(row)

    window = draft.get("backtest_window")
    if isinstance(window, dict) and window.get("from_utc") and window.get("to_utc"):
        rows.append({
            "label": "Backtest range",
            "value": f"{window['from_utc']} ? {window['to_utc']}",
        })

    return rows


def _full_draft_details(draft: dict[str, Any]) -> dict[str, Any]:
    """Complete ``strategy_draft`` for the UI expander (no nested snapshot)."""
    full = dict(draft)
    full.pop("inputs_snapshot", None)
    return full


def build_inputs_snapshot_from_draft(
    draft: dict[str, Any],
    *,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    """Build summary + details from a strategy_draft dict."""
    missing_fields = list(missing or [])
    return {
        "state": str(draft.get("mode") or ""),
        "summary_rows": _summary_rows_from_draft(draft),
        "details": _full_draft_details(draft),
        "core_field_order": list(CORE_USER_INPUT_FIELDS),
        "missing_fields": missing_fields,
        "missing_field_labels": [FIELD_LABELS.get(f, f) for f in missing_fields],
    }


def build_inputs_snapshot(builder: StrategyBuilder, *, state: str = "") -> dict[str, Any]:
    """Convenience wrapper: snapshot from builder's current draft JSON."""
    draft = builder.to_draft_json(mode_override=state or builder.get_mode())
    return build_inputs_snapshot_from_draft(
        draft,
        missing=builder.missing_user_input_fields(),
    )


def format_inputs_summary_markdown(snapshot: dict[str, Any]) -> str:
    rows = snapshot.get("summary_rows") or []
    if not rows:
        return ""

    lines = [
        "### Stored inputs",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    for row in rows:
        label = str(row.get("label", "")).replace("|", "\\|")
        value = str(row.get("value", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def should_append_inputs_footer(assistant_text: str) -> bool:
    text = str(assistant_text or "")
    return not any(marker in text for marker in _SUMMARY_SKIP_MARKERS)


def append_inputs_snapshot(
    assistant_text: str,
    builder: StrategyBuilder,
    *,
    state: str,
    draft: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Append summary table and attach ``inputs_snapshot`` on strategy_draft."""
    working_draft = _working_draft(draft, builder, state=state)
    snapshot = build_inputs_snapshot_from_draft(
        working_draft,
        missing=builder.missing_user_input_fields(),
    )
    working_draft["inputs_snapshot"] = snapshot

    if not snapshot.get("summary_rows"):
        return assistant_text, working_draft

    if not should_append_inputs_footer(assistant_text):
        return assistant_text, working_draft

    footer = format_inputs_summary_markdown(snapshot)
    body = str(assistant_text or "").rstrip()
    if body:
        return f"{body}\n\n---\n\n{footer}", working_draft
    return footer, working_draft


def format_details_json(snapshot: dict[str, Any]) -> str:
    details = snapshot.get("details") or {}
    if not details:
        return "{}"
    return json.dumps(details, indent=2, ensure_ascii=False, default=str)
