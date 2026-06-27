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


def build_welcome_message(asset_classes: list[str] | None = None) -> str:
    """Welcome message scoped to the asset classes enabled for this chat.

    Pass the asset_class_id strings of *enabled* capabilities (e.g.
    ["equity_cash", "crypto_spot"]). When omitted, defaults to the legacy
    equity-only copy.
    """
    return compose_response("collect_input.welcome", asset_classes=asset_classes or [])


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
    return compose_response(
        "workflow.input_summary_confirmation",
        asset=_asset_label(builder),
        timeframe=builder.timeframe,
        objective=builder.objective,
        sentiment=builder.sentiment,
        experience=builder.experience,
        goal=builder.goal,
    )


def build_plan_signals_reply(builder: StrategyBuilder, plan: dict) -> str:
    return compose_response(
        "workflow.signal_plan_ready",
        asset=_asset_label(builder),
        experience=builder.experience,
        trade_type=builder.objective,
    )


def build_plan_signals_reminder(builder: StrategyBuilder) -> str:
    return compose_response(
        "workflow.signal_plan_ready",
        asset=_asset_label(builder),
        experience=builder.experience,
        trade_type=builder.objective,
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


def build_backtest_result_reply(
    builder: StrategyBuilder,
    backtest_result: dict,
    *,
    multi_result: Any = None,
    display_symbol: str | None = None,
) -> str:
    # Multi-asset: when more than one asset ran, lead with a side-by-side
    # comparison table. `multi_result` is a MultiAssetBacktestResult (duck-typed
    # here to avoid a circular import).
    summary = getattr(multi_result, "summary", None)
    if summary and len(summary) > 1:
        return _build_multi_asset_reply(multi_result)

    metrics = backtest_result.get("metrics") if isinstance(backtest_result, dict) else {}
    assessment = backtest_result.get("assessment") if isinstance(backtest_result, dict) else {}

    # `display_symbol` is set only for a single-asset run whose symbol was
    # overridden via the run_backtest `symbols` arg (e.g. "run backtest for tcs"
    # on a HINDALCO strategy). The override changes which OHLCV is simulated but
    # NOT the session strategy, so we label the header with the asset the user
    # actually asked for. None → legacy behaviour (the strategy's own symbol).
    return compose_response(
        "workflow.backtest_complete",
        asset=display_symbol or _asset_label(builder),
        passed=backtest_result.get("pass"),
        total_return_pct=(metrics or {}).get("total_return_pct"),
        win_rate=(metrics or {}).get("win_rate"),
        total_trades=(metrics or {}).get("total_trades"),
        overall_grade=(assessment or {}).get("overall_grade"),
        failure_reason=backtest_result.get("failure_reason"),
    )


def _build_multi_asset_reply(multi_result: Any) -> str:
    """Compact markdown comparison of per-asset key metrics.

    One row per asset: total return, annualized return, volatility, max drawdown —
    the four metrics requested for cross-asset comparison.
    """
    def _pct(value: float | None) -> str:
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    rows = [
        "| Asset | Total Return | Annualized | Volatility | Max Drawdown | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in multi_result.summary:
        if row.failure_reason and not row.backtest_ref_id:
            rows.append(
                f"| {row.symbol} | — | — | — | — | ❌ {row.failure_reason[:40]} |"
            )
            continue
        rows.append(
            f"| {row.symbol} | {_pct(row.total_return_pct)} | {_pct(row.annual_return)} "
            f"| {_pct(row.volatility_pct)} | {_pct(row.max_drawdown)} "
            f"| {'✅ Pass' if row.pass_ else '⚠️ Fail'} |"
        )

    num = getattr(multi_result, "num_assets", len(multi_result.summary))
    header = f"Backtest complete for {num} asset(s). Here's how they compare:"
    return "\n".join([header, "", *rows])


def build_backtest_error_reply(reason: str) -> str:
    return compose_response(
        "workflow.backtest_failed",
        reason=reason,
    )


def build_backtest_earliest_date_reply(earliest_date: str) -> str:
    return compose_response(
        "workflow.backtest_earliest_date_unsupported",
        earliest_date=earliest_date,
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
    asset_class: str | None = None,
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
            "asset_class": asset_class,
            "strategy_object": strategy_object,
            "strategy_config": strategy_config,
            "strategy_id": strategy_id,
            "yaml_path": yaml_path,
        },
        "success": True,
    }


# ── Phase 9 — discovery / dynamic-symbol replies ────────────────────────────


def build_discovery_no_match_reply(
    asset_label_hint: str = "stocks",
    *,
    parameters_used: dict | None = None,
    conditions_used: list | None = None,
    scan_diagnostics: dict | None = None,
) -> str:
    """The scanner found zero candidates matching the preset's discovery
    conditions. Surface ENOUGH context that the user can take an
    informed next step rather than guessing.

    Phase 9h — surface the active threshold summary.
    Phase 9k — list the parsed primitives so the user sees the exact
               constraints applied (no "you also have to be near 52w
               high" surprise).
    Phase 9m — when scan_diagnostics is supplied (asof, scanned_count,
               failed_fetches, fail_counts), use them to:
                 • show how many stocks were checked and on what asof
                   date (so the user can spot stale data fast),
                 • show per-condition failure counts (e.g. "18/19
                   stocks failed: volume ≥ 1.2× the 20-day average"),
                 • generate suggestions tailored to the active
                   conditions only (e.g. "lower the volume multiplier"
                   when only volume_spike is active — no more
                   misleading "relax 52-week proximity" text when no
                   52-week clause was applied).
    """
    # Phase 9n — if the universe is empty because EVERY fetch failed,
    # the constraint-relaxation suggestions are misleading (the user's
    # thresholds were never even checked). Detect this case and emit a
    # totally different message focused on the infrastructure error.
    if _is_total_fetch_failure(scan_diagnostics):
        return _build_fetch_failure_reply(asset_label_hint, scan_diagnostics)

    lines = [f"No {asset_label_hint} matched the strategy's discovery criteria today."]

    # 1. What did we actually look for?
    primitive_lines = _primitive_descriptions(conditions_used)
    if primitive_lines:
        lines.append("Constraints applied: " + "; ".join(primitive_lines) + ".")
    elif parameters_used:
        thresholds = _format_threshold_summary(parameters_used)
        if thresholds:
            lines.append(f"Thresholds used: {thresholds}.")

    # 2. Diagnostic snapshot (asof + universe size + per-condition fails)
    if scan_diagnostics:
        diag_line = _format_scan_diagnostics(scan_diagnostics, conditions_used or [])
        if diag_line:
            lines.append(diag_line)

    # 3. Concrete next steps
    lines.append("You can either:")
    for hint in _build_relax_hints(conditions_used, parameters_used, scan_diagnostics):
        lines.append(f"  • {hint}")

    return "\n".join(lines)


def _is_total_fetch_failure(scan_diagnostics: dict | None) -> bool:
    """True when EVERY universe fetch failed (no data reached the
    scanner, so threshold-relaxation hints would be misleading)."""
    if not scan_diagnostics:
        return False
    scanned = scan_diagnostics.get("scanned_count") or 0
    failed = scan_diagnostics.get("failed_fetches") or 0
    fetch_errors = scan_diagnostics.get("fetch_errors") or {}
    if scanned <= 0:
        return False
    return failed >= scanned and bool(fetch_errors)


def _build_fetch_failure_reply(
    asset_label_hint: str, scan_diagnostics: dict,
) -> str:
    """User-facing message for the case where the scanner couldn't
    fetch ANY market data. Quotes a sample error so the user / ops
    can take an informed next step (retry, check service health,
    file a ticket) instead of fiddling with thresholds that aren't
    the problem."""
    scanned = int(scan_diagnostics.get("scanned_count") or 0)
    fetch_errors = scan_diagnostics.get("fetch_errors") or {}
    sample_symbol = next(iter(fetch_errors), None)
    sample_error = fetch_errors.get(sample_symbol) if sample_symbol else None

    lines = [
        f"I couldn't scan any {asset_label_hint} right now — every "
        f"market-data fetch failed.",
    ]
    if scanned:
        lines.append(f"Tried {scanned} symbols; none returned OHLCV records.")
    if sample_symbol and sample_error:
        lines.append(f"Example error ({sample_symbol}): {sample_error}")
    asof = scan_diagnostics.get("asof_iso")
    if asof:
        date = str(asof).split("T", 1)[0] if "T" in str(asof) else asof
        lines.append(f"As-of: {date}.")
    lines.append("This is an infrastructure issue, not a strategy problem. You can either:")
    lines.append("  • Wait a minute and retry — transient network or rate-limit issues clear quickly")
    lines.append("  • Pin a specific stock manually so the scanner is bypassed")
    lines.append("  • Switch to a different strategy preset that doesn't run discovery")
    lines.append("  • Report the example error above if it persists for more than a few minutes")
    return "\n".join(lines)


def _primitive_descriptions(conditions_used: list | None) -> list[str]:
    if not conditions_used:
        return []
    try:
        from app.services.discovery.primitives import primitive_descriptions
        return primitive_descriptions(conditions_used) or []
    except Exception:
        return []


def _format_scan_diagnostics(diag: dict, conditions_used: list) -> str:
    """Multi-line diagnostic summary so the user can see exactly which
    stocks were scanned, which failed which step, and (if available)
    which were fetched but rejected by which condition. Replaces the
    earlier one-line "Scanned N stocks (M excluded ...)" black box.
    """
    asof = diag.get("asof_iso")
    scanned = diag.get("scanned_count")
    failed_fetches = diag.get("failed_fetches") or 0
    fail_counts = diag.get("fail_counts") or {}
    fetch_errors = diag.get("fetch_errors") or {}
    scanned_symbols: list[str] = list(diag.get("scanned_symbols") or [])
    failed_fetch_symbols: list[str] = list(diag.get("failed_fetch_symbols") or [])
    failed_condition_by_symbol: dict[str, str] = dict(
        diag.get("failed_condition_by_symbol") or {}
    )

    lines: list[str] = []

    # Headline: how many, on what date.
    if isinstance(scanned, int) and scanned > 0:
        headline = f"Scanned {scanned} stock{'s' if scanned != 1 else ''}"
        if asof:
            date = str(asof).split("T", 1)[0] if "T" in str(asof) else asof
            headline += f" (as of {date})"
        headline += "."
        lines.append(headline)

    # The actual universe — names so the user isn't guessing.
    if scanned_symbols:
        lines.append(f"  Universe: {', '.join(_pretty_symbols(scanned_symbols))}.")

    # Fetch failures — show every symbol that couldn't be fetched, not
    # just one example. fetch_errors still dedups by error text so we
    # can quote a sample error message.
    if failed_fetch_symbols:
        sample_error = next(iter(fetch_errors.values()), None) if fetch_errors else None
        line = (
            f"  Couldn't fetch data ({len(failed_fetch_symbols)}): "
            f"{', '.join(_pretty_symbols(failed_fetch_symbols))}."
        )
        if sample_error:
            line += f" Example error: {sample_error}"
        lines.append(line)
    elif failed_fetches:
        # Counter says fetches failed but no symbol list (older diagnostics).
        # Fall back to the example-only line.
        example = next(iter(fetch_errors.values()), None) if fetch_errors else None
        line = f"  {failed_fetches} stocks excluded due to data fetch errors."
        if example:
            line += f" Example: {example}"
        lines.append(line)

    # Per-condition breakdown: show which stocks failed each step,
    # not just the aggregate count. Limits each list to a few names
    # so the message stays scannable when the universe is large.
    if failed_condition_by_symbol:
        per_cond: dict[str, list[str]] = {}
        for sym, cond in failed_condition_by_symbol.items():
            per_cond.setdefault(cond, []).append(sym)
        # Sort by descending fail-count so the binding constraint
        # appears first.
        for cond, syms in sorted(per_cond.items(), key=lambda kv: -len(kv[1])):
            desc = _condition_to_description(cond, conditions_used) or cond
            lines.append(
                f"  Failed «{desc}» ({len(syms)}): "
                f"{', '.join(_pretty_symbols(syms))}."
            )
    else:
        # No per-symbol breakdown available (older scan or all fetches
        # failed) — fall back to the aggregate "worst condition" line.
        worst = _worst_failing_condition(fail_counts, conditions_used)
        if worst:
            worst_cond, worst_count = worst
            worst_desc = _condition_to_description(worst_cond, conditions_used)
            if worst_count > 0:
                lines.append(f"  Most stocks ({worst_count}) failed: {worst_desc}.")

    return "\n".join(lines)


def _pretty_symbols(symbols: list[str]) -> list[str]:
    """Strip the .NS exchange suffix from a list of symbols so the
    user sees "TCS, INFY" instead of "TCS.NS, INFY.NS". Preserves
    order and uniqueness; non-string entries are skipped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        if not isinstance(sym, str):
            continue
        clean = sym.split(".", 1)[0] if "." in sym else sym
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


def _worst_failing_condition(
    fail_counts: dict, conditions_used: list,
) -> tuple[str, int] | None:
    """Pick the condition that the most stocks failed at. Returns
    (condition_string, count) or None."""
    if not isinstance(fail_counts, dict) or not fail_counts:
        return None
    items = [(k, int(v)) for k, v in fail_counts.items() if isinstance(v, (int, float))]
    if not items:
        return None
    worst = max(items, key=lambda kv: kv[1])
    if worst[1] <= 0:
        return None
    return worst


def _condition_to_description(condition: str, conditions_used: list) -> str:
    """Map an AST condition string back to its human description if
    possible. Falls back to the raw condition string otherwise."""
    if not conditions_used:
        return condition
    try:
        from app.services.discovery.primitives import (
            primitive_descriptions, render_conditions,
        )
    except Exception:
        return condition
    try:
        rendered = render_conditions(conditions_used)
    except Exception:
        return condition
    descs = _primitive_descriptions(conditions_used)
    for r, d in zip(rendered, descs):
        if r.strip() == condition.strip():
            return d
    return condition


def _build_relax_hints(
    conditions_used: list | None,
    parameters_used: dict | None,
    scan_diagnostics: dict | None,
) -> list[str]:
    """Generate next-step suggestions based ONLY on the constraints
    actually applied. No more "relax 52-week proximity" when 52-week
    isn't an active constraint."""
    hints: list[str] = []
    names = (
        [str(c.get("name") or "") for c in conditions_used if isinstance(c, dict)]
        if conditions_used else []
    )

    has_volume   = "volume_spike" in names
    has_near_high = "near_52w_high" in names
    has_near_low  = "near_52w_low"  in names
    has_above_high = "above_52w_high" in names
    has_below_low  = "below_52w_low"  in names
    has_rsi_above  = "rsi_above"  in names
    has_rsi_below  = "rsi_below"  in names
    has_pullback = ("shallow_pullback_long" in names
                    or "shallow_pullback_short" in names)

    if has_volume:
        # Suggest a lower multiplier.
        cur = _condition_param_value(conditions_used, "volume_spike", "multiplier", 2.0)
        suggestion = max(1.0, round(cur - 0.3, 1))
        if suggestion < cur:
            hints.append(
                f"Lower the volume threshold (e.g. drop from {cur:g}× to {suggestion:g}×)"
            )
    if has_near_high or has_near_low:
        window_phrase = _active_window_phrase(parameters_used)
        hints.append(
            f"Widen the {window_phrase} proximity (e.g. from 2% to 5%)"
        )
    if has_above_high or has_below_low:
        hints.append(
            "Switch the breakout to proximity (`near 52-week high` instead of `breaking above`)"
        )
    if has_rsi_above:
        cur = _condition_param_value(conditions_used, "rsi_above", "threshold", 60.0)
        suggestion = max(40.0, cur - 10.0)
        if suggestion < cur:
            hints.append(
                f"Lower the RSI threshold (e.g. drop from {cur:g} to {suggestion:g})"
            )
    if has_rsi_below:
        cur = _condition_param_value(conditions_used, "rsi_below", "threshold", 40.0)
        suggestion = min(60.0, cur + 10.0)
        if suggestion > cur:
            hints.append(
                f"Raise the RSI threshold (e.g. lift from {cur:g} to {suggestion:g})"
            )
    if has_pullback:
        hints.append("Drop the pullback constraint (some setups don't pull back at all)")

    # Generic fallback when we can't propose anything specific (e.g.
    # only above_vwap is active and there's no numeric tunable).
    if not hints:
        hints.append("Loosen the constraints — type a new prompt with relaxed thresholds")

    # Always include these last two so the user has multiple paths.
    hints.append("Switch to a different strategy preset")
    hints.append("Pin a specific stock manually so the scanner is bypassed")

    # Stale data hint when the asof date is more than 3 days old.
    if scan_diagnostics:
        asof = scan_diagnostics.get("asof_iso")
        if asof and _asof_looks_stale(asof):
            hints.insert(
                0,
                f"The latest data the scanner saw is dated {str(asof).split('T', 1)[0]} — "
                "it may be stale; try again later if the market is closed",
            )

    return hints


def _condition_param_value(
    conditions_used: list, name: str, param: str, default: float,
) -> float:
    """Pull the user's param value for a primitive (or the default
    when not specified)."""
    for c in conditions_used or []:
        if isinstance(c, dict) and c.get("name") == name:
            params = c.get("params") if isinstance(c.get("params"), dict) else {}
            try:
                return float(params.get(param, default))
            except (TypeError, ValueError):
                return default
    return default


def _asof_looks_stale(asof_iso: str) -> bool:
    """True when the asof date is more than 3 calendar days old. Used
    to surface a 'data may be stale' hint in the no-match reply."""
    try:
        from datetime import datetime, timezone, timedelta
        text = str(asof_iso).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt) > timedelta(days=3)
    except Exception:
        return False


def _active_window_phrase(parameters_used: dict | None) -> str:
    """Return the window phrase ("52-week", "20-day", …) implied by
    the active discovery parameters; defaults to "52-week" so the
    pre-9i hint phrasing is preserved when no params are supplied."""
    if parameters_used:
        raw = parameters_used.get("lookback_window_bars")
        if raw is not None:
            try:
                bars = int(float(raw))
            except (TypeError, ValueError):
                bars = 252
            if bars and bars != 252:
                return _format_window_phrase(bars)
    return "52-week"


def _format_threshold_summary(params: dict) -> str:
    """Render a human-readable summary of the discovery thresholds in
    use. Translates the internal factor params back to percentages so
    the user sees the same units they typed.

    Phase 9i — the high/low window is also tunable. When the user
    chose a non-default window, the percentages reference the same
    window phrase (e.g. "within 5% of 20-day high") so the message
    matches the actual scan."""
    window_phrase = "52-week"
    window_bars_raw = params.get("lookback_window_bars")
    if window_bars_raw is not None:
        try:
            bars = int(float(window_bars_raw))
        except (TypeError, ValueError):
            bars = 252
        if bars and bars != 252:
            window_phrase = _format_window_phrase(bars)
    parts: list[str] = []
    vm = params.get("volume_multiplier")
    if vm is not None:
        parts.append(f"volume ≥ {float(vm):g}× the 20-day average")
    hi = params.get("near_52w_high_factor")
    if hi is not None:
        pct = max(0.0, (1.0 - float(hi)) * 100.0)
        parts.append(f"within {pct:g}% of {window_phrase} high")
    lo = params.get("near_52w_low_factor")
    if lo is not None:
        pct = max(0.0, (float(lo) - 1.0) * 100.0)
        parts.append(f"within {pct:g}% of {window_phrase} low")
    return "; ".join(parts)


def _format_window_phrase(bars: int) -> str:
    """Render a bar count as a user-friendly window phrase. Mirrors the
    helper in app.services.discovery.orchestrator (kept duplicated to
    avoid a chat→discovery import cycle)."""
    if bars % 252 == 0 and bars >= 252:
        years = bars // 252
        return f"{years}-year" if years > 1 else "1-year"
    if bars % 21 == 0 and bars >= 42:
        months = bars // 21
        return f"{months}-month"
    if bars % 5 == 0 and bars >= 25:
        weeks = bars // 5
        return f"{weeks}-week"
    return f"{bars}-day"


def build_discovery_choice_prompt(scan_result: dict) -> str:
    """The scanner returned multiple candidates. Format them + the tie-break
    options so the user can pick a method.

    `scan_result` is the dict produced by ScanResult.to_dict() (kept as a
    dict here so this builder doesn't need to import the discovery types)."""
    candidates = scan_result.get("candidates") or []
    options    = scan_result.get("tie_break_options") or []

    if not candidates:
        return "The scanner returned no candidates. Try different criteria."
    if not options:
        return (
            f"Found {len(candidates)} candidates but no tie-break options were "
            "configured for this preset. Please pick a stock manually."
        )

    cand_lines = []
    for c in candidates:
        sym  = c.get("symbol", "?")
        name = c.get("display_name") or sym
        m    = c.get("metrics") or {}
        # Pick a couple of metrics to surface in-line so the user has
        # context. Falling back to "—" for any that didn't compute.
        rv   = m.get("relative_volume")
        d52  = m.get("distance_to_52w_high_pct")
        rsi  = m.get("rsi_14")
        bits = []
        if rv is not None and rv == rv:        # not NaN
            bits.append(f"vol×{rv:.1f}")
        if d52 is not None and d52 == d52:
            bits.append(f"−{d52:.1f}% from 52w high")
        if rsi is not None and rsi == rsi:
            bits.append(f"RSI {rsi:.0f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        cand_lines.append(f"  • {name} ({sym}){suffix}")
    cand_block = "\n".join(cand_lines)

    opt_lines = []
    for i, o in enumerate(options, start=1):
        label = o.get("label") or o.get("method", "?")
        desc  = o.get("description") or ""
        opt_lines.append(f"  {i}. {label}" + (f" — {desc}" if desc else ""))
    opt_block = "\n".join(opt_lines)

    return (
        f"I found {len(candidates)} stocks matching the strategy criteria today:\n"
        f"{cand_block}\n\n"
        "How should I pick one? Reply with the number or method name:\n"
        f"{opt_block}"
    )


def build_discovery_resolved_reply(symbol: str, display_name: str, method: str) -> str:
    """Confirmation message after the user picks a tie-break method and we
    resolve to a single symbol."""
    pretty_method = method.replace("_", " ")
    return (
        f"Picked **{display_name}** (`{symbol}`) using *{pretty_method}*. "
        "Continuing with the strategy build."
    )
