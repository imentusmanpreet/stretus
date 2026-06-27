"""
app/services/chat/direct_strategy_flow.py — narrow-branch strategy synthesis (flag ON).

This is the ONE place the ``use_direct_strategy_path`` flag swaps in. When enabled,
the chat flow's existing routing, input gathering, risk config, edits, backtest and
persistence stay exactly as they are; only the *synthesis* step changes: instead of
the KB planner (``plan_signals_v2`` + ``build_strategy_object``/``config``), we:

  1. 🧠 generate a strict ``StrategySpec`` from the gathered context (LLM + repair loop),
  2. 🛡️ which is already validated against the live engine grammar by the generator,
  3. 🧭 reconcile identity fields (symbol/timeframe/market/objective) to the values the
        existing input-gathering already confirmed, so the YAML and the persisted
        strategy object never diverge,
  4. 🏗️ populate the SAME ``StrategyBuilder`` the downstream reads — so the unchanged
        ``build_strategy_object``/``build_strategy_config`` keep working as-is,
  5. 📝 write the engine YAML straight from the spec (clean formula mode).

Fail-closed: if generation/validation fails after repairs, we return ``ok=False`` with
a user-facing message and DO NOT populate the builder — the caller surfaces a
clarification instead of assembling a half-built strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.services.ai.strategy_prompt import build_strategy_system_prompt
from app.strategy.generator import generate_strategy
from app.strategy.spec import StrategySpec
from app.strategy.validator import ValidationResult
from app.strategy.yaml_writer import write_spec_yaml

logger = logging.getLogger(__name__)


def _apply_universe_to_builder(builder: Any, spec: StrategySpec) -> None:
    """Store the universe RULE (pure ``UniverseSpec`` dump — must stay re-validatable) and
    the AI's own plain-words ``intent_summary`` SEPARATELY on the builder.

    The summary is kept OFF the universe dict on purpose: ``builder.universe`` is fed back into
    ``UniverseSpec.model_validate`` (resolver + YAML), which is ``extra='forbid'`` — any stray
    key would crash validation. The response layer merges the summary back in for display.
    """
    builder.universe = spec.universe.model_dump(mode="json", exclude_none=True)
    builder.universe_summary = (spec.intent_summary or "").strip() or None


def _log_spec_breakdown(spec: StrategySpec, *, session_id: str | None, stage: str) -> None:
    """Pretty, human-readable breakdown of the spec the AI authored — universe highlighted.

    Logged at INFO so a tester can see, at a glance, exactly what was built (instrument vs
    selection rule, the conditions, the risk) without digging through the raw JSON.
    """
    lines = [
        f"┌─🧠 STRATEGY AUTHORED ({stage}) | session={session_id} ─────────────────",
        f"│  name        : {spec.name}",
        f"│  market/tf   : {spec.market} / {spec.timeframe}  ({spec.objective}, {spec.direction})",
    ]
    if spec.is_dynamic and spec.universe is not None:
        u = spec.universe
        lines += [
            "│  🌌 UNIVERSE  : DYNAMIC (rule-selected, no fixed symbol)",
            f"│     source    : {u.source.kind}" + (f" '{u.source.name}'" if u.source.name else ""),
            f"│     rank by   : {u.rank.by} ({u.rank.order})"
            + (f", window={u.rank.window}" if u.rank.window else ""),
            f"│     screen    : {u.screen or '(none)'}",
            f"│     take      : {u.take}   max_positions: {u.max_positions or u.take}",
            f"│     refresh   : {u.refresh.cadence}" + (f" @ {u.refresh.at}" if u.refresh.at else ""),
        ]
        if u.source.kind == "watchlist" and u.source.symbols:
            lines.append(f"│     symbols   : {u.source.symbols}")
    else:
        lines.append(f"│  📈 SYMBOL    : {spec.symbol}")
    lines += [
        f"│  entry       : {spec.entry_condition}",
        f"│  exit        : {spec.exit_condition or '(none)'}",
        f"│  stop_loss   : {spec.stop_loss.type} = {spec.stop_loss.value}"
        + (f" (window {spec.stop_loss.window})" if spec.stop_loss.window else ""),
        f"│  take_profit : {(spec.take_profit.type + ' = ' + str(spec.take_profit.value)) if spec.take_profit else '(trailing/none)'}",
        f"│  intent      : {(spec.intent_summary or '')[:200]}",
        "└──────────────────────────────────────────────────────────────────────",
    ]
    logger.info("\n%s", "\n".join(lines))


@dataclass
class DirectSynthesisResult:
    """Outcome of the direct synthesis step, consumed by the chat router."""

    ok: bool
    spec: StrategySpec | None = None
    yaml_path: str | None = None
    notes: list[str] = field(default_factory=list)          # user-facing assumed-value lines
    error_message: str | None = None                        # user-facing failure summary
    validation: ValidationResult | None = None              # raw result for logging/tests


# ── Context the LLM sees (built from the already-gathered builder state) ──────

def _builder_context_message(builder: Any) -> str:
    """A single instruction summarising what the existing flow already gathered.

    The system prompt supplies the grammar + schema + rules; this supplies the
    user's intent + the confirmed parameters. Anything absent here the LLM should
    assume sensibly and tag ``source:"assumed"``.
    """
    g = lambda name: getattr(builder, name, None)  # noqa: E731
    request = g("original_user_prompt") or g("goal") or "Build a trading strategy."

    known: list[str] = []
    def _add(label: str, value: Any) -> None:
        if value is not None and str(value).strip():
            known.append(f"- {label}: {value}")

    _add("symbol", builder.format_symbol() if hasattr(builder, "format_symbol") else g("symbol"))
    _add("market", g("market"))
    _add("timeframe", g("timeframe"))
    _add("objective", g("objective"))
    _add("sentiment", g("sentiment"))
    _add("experience", g("experience"))
    _add("goal", g("goal"))
    _add("direction", g("direction"))
    # Only surface SL/TP as user-given when provenance says so. A hydrated platform
    # DEFAULT must NOT be presented as user input — otherwise the LLM echoes the
    # default (e.g. a 0.5% take-profit) instead of assuming a sensible one. The raw
    # user prompt is always included above, so genuinely user-given values are still
    # available to the model to extract.
    rms_sources = (g("risk_execution_config") or {}).get("rms_sources", {}) or {}
    if rms_sources.get("stop_loss_pct") == "user":
        _add("stop_loss_pct (user-specified)", g("stop_loss"))

    # Multi-TP ladder: include rungs instead of the (now-zeroed) scalar.
    # rms_sources["take_profit_pct"] == "multi_tp" is set by constraint_compiler
    # when a partial-exit ladder is detected — suppress the scalar in that case.
    partial_exits = (getattr(builder, "execution_layers", None) or {}).get("partial_exits")
    if partial_exits:
        _add(
            "multi_take_profit_ladder (user-specified)",
            "; ".join(
                f"rung {i+1}: {pe.get('trigger')}={pe.get('value')}"
                + (f" exit={pe['size_pct']}%" if pe.get("size_pct") else "")
                for i, pe in enumerate(partial_exits)
            ),
        )
    elif rms_sources.get("take_profit_pct") == "user":
        _add("take_profit_pct (user-specified)", g("take_profit"))

    known_block = "\n".join(known) if known else "- (none gathered yet)"
    return (
        "Build a single trading strategy that captures this request, and return ONLY a "
        "StrategySpec JSON.\n\n"
        f"User's request:\n{request}\n\n"
        "Confirmed parameters (use these verbatim; assume sensible defaults for anything "
        "missing and tag those fields source:\"assumed\"):\n"
        f"{known_block}"
    )


# ── Keep the spec's identity aligned with the confirmed builder state ─────────

def _reconcile_identity(spec: StrategySpec, builder: Any) -> StrategySpec:
    """Override spec identity fields with the values the input-gathering confirmed.

    Identity (symbol/timeframe/market/objective) is owned by the existing flow; the
    spec owns the trading *logic*. This keeps the engine YAML (from the spec) and the
    persisted strategy_object (from the builder) describing the same instrument.
    """
    updates: dict[str, Any] = {}
    # Dynamic-universe specs own NO single symbol — the universe rule is the instrument
    # selector. Never force a symbol onto them (it would erase the universe and break the
    # one-of contract). Only non-dynamic specs reconcile their confirmed symbol.
    if not spec.is_dynamic:
        confirmed_symbol = builder.format_symbol() if hasattr(builder, "format_symbol") else getattr(builder, "symbol", None)
        if confirmed_symbol:
            updates["symbol"] = confirmed_symbol
    if getattr(builder, "timeframe", None):
        updates["timeframe"] = builder.timeframe
    if getattr(builder, "market", None):
        updates["market"] = builder.market
    if getattr(builder, "objective", None):
        updates["objective"] = builder.objective
    if not updates:
        return spec
    # Identity-only fields don't affect condition compilation, so no re-validation needed.
    return spec.model_copy(update=updates)


# ── Map the validated spec onto the builder the downstream already reads ──────

def populate_builder_from_spec(builder: Any, spec: StrategySpec) -> None:
    """Write the spec's logic onto ``builder`` so build_strategy_object works unchanged.

    Pure mutation, no I/O — directly unit-testable. The engine's full truth comes from
    the spec YAML; the builder carries the subset the persistence/UI layer reads.
    """
    builder.entry_condition = spec.entry_condition
    builder.exit_condition = spec.exit_condition or None
    builder.short_entry_condition = spec.short_entry_condition or None
    builder.short_exit_condition = spec.short_exit_condition or None
    builder.direction = spec.direction
    # Dynamic universe: carry the rule onto the builder so it persists across turns and
    # the engine YAML emits the `universe:` block (backtest route → portfolio path).
    if spec.is_dynamic and spec.universe is not None:
        _apply_universe_to_builder(builder, spec)

    # Risk (properties backed by the builder's risk store).
    builder.stop_loss = spec.resolved_stop_loss_pct()
    builder.take_profit = spec.resolved_take_profit_pct()
    builder.stop_loss_spec = spec.stop_loss_engine_spec()
    if spec.trailing_stop is not None:
        builder.trailing_stop_spec = spec.trailing_stop.to_engine_dict()
    if spec.trailing_take_profit is not None:
        builder.trailing_take_profit_spec = spec.trailing_take_profit.to_engine_dict()

    if spec.reference_symbol:
        builder.reference_symbol = spec.reference_symbol
    if spec.htf_rules:
        builder.htf_rules = [
            {"timeframe": r.timeframe, "condition": r.condition} for r in spec.htf_rules
        ]
    if spec.position_sizing_mode:
        builder.position_sizing_mode = spec.position_sizing_mode
    if spec.max_capital_allocation_pct is not None:
        builder.max_capital_allocation_pct = spec.max_capital_allocation_pct

    # Gate subset the builder carries (engine gets the full set from the YAML).
    _apply_gates_to_builder(builder, spec.gates)

    # Formula-mode signal_plan (no KB signals) in the legacy shape build_strategy_object reads.
    builder.signal_plan = {
        "entry_condition": spec.entry_condition,
        "exit_condition": spec.exit_condition,
        "short_entry_condition": spec.short_entry_condition,
        "short_exit_condition": spec.short_exit_condition,
        "signals_used": [],
        "signals_available": 0,
        "entry": [],
        "exit": [],
    }


def _set_if(builder: Any, attr: str, value: Any) -> None:
    if value is not None:
        setattr(builder, attr, value)


def _apply_gates_to_builder(builder: Any, gates: Any) -> None:
    """Map the gate fields the builder carries (engine gets the full set from YAML)."""
    if gates is None:
        return
    _set_if(builder, "entry_window_start", gates.entry_window_start)
    _set_if(builder, "entry_window_end", gates.entry_window_end)
    _set_if(builder, "gap_filter", gates.gap_filter)
    _set_if(builder, "gap_threshold_pct", gates.gap_threshold_pct)
    _set_if(builder, "entry_confirmation_bars", gates.entry_confirmation_bars)
    _set_if(builder, "rsi_entry_band_min", gates.rsi_entry_band_min)
    _set_if(builder, "rsi_entry_band_max", gates.rsi_entry_band_max)
    _set_if(builder, "volume_ratio_threshold", gates.volume_ratio_threshold)
    _set_if(builder, "max_consecutive_losses", gates.max_consecutive_losses)
    _set_if(builder, "cooldown_bars_after_loss", gates.cooldown_bars_after_loss)
    _set_if(builder, "cooldown_bars_after_profit", gates.cooldown_bars_after_profit)


# ── Chat-flow integration (narrow branch: reuse apply_signal_plan downstream) ─

def spec_to_plan(spec: StrategySpec) -> dict:
    """Build the legacy-shaped ``signal_plan`` dict that ``builder.apply_signal_plan``
    consumes — so the existing assembly path keeps working unchanged. Formula mode:
    no KB signals, conditions carried verbatim, SL/trailing/ref/HTF as underscore keys.
    """
    plan: dict[str, Any] = {
        "entry_condition": spec.entry_condition,
        "exit_condition": spec.exit_condition or "",
        "short_entry_condition": spec.short_entry_condition or "",
        "short_exit_condition": spec.short_exit_condition or "",
        "signals_used": [],
        "signals_available": 0,
        "entry": [],
        "exit": [],
    }
    sl_spec = spec.stop_loss_engine_spec()
    if sl_spec:
        plan["_stop_loss_spec"] = sl_spec
    if spec.trailing_stop is not None:
        plan["_trailing_stop_spec"] = spec.trailing_stop.to_engine_dict()
    if spec.trailing_take_profit is not None:
        plan["_trailing_take_profit_spec"] = spec.trailing_take_profit.to_engine_dict()
    if spec.reference_symbol:
        plan["_reference_symbol"] = spec.reference_symbol
    if spec.htf_rules:
        plan["_htf_rules"] = [
            {"timeframe": r.timeframe, "condition": r.condition} for r in spec.htf_rules
        ]
    return plan


def apply_spec_extras_to_builder(builder: Any, spec: StrategySpec) -> None:
    """Set the fields ``apply_signal_plan`` does NOT (risk %, direction, gates, sizing).

    Conditions + SL/trailing/ref/HTF specs are applied downstream by
    ``builder.apply_signal_plan(spec_to_plan(spec))``; this fills the rest.
    """
    builder.direction = spec.direction
    builder.stop_loss = spec.resolved_stop_loss_pct()
    builder.take_profit = spec.resolved_take_profit_pct()
    # Dynamic universe: carry the rule (persisted → emitted into the engine YAML).
    if spec.is_dynamic and spec.universe is not None:
        _apply_universe_to_builder(builder, spec)
    # The spec is the source of truth now — keep risk provenance accurate so the
    # read-back/config shows the right value AND who set it (user vs assumed).
    store = getattr(builder, "risk_execution_config", None)
    if isinstance(store, dict):
        rms = store.setdefault("rms_sources", {})
        # Only a *percent* stop/target is the user's literal percentage. A typed stop
        # (ATR/structure) or an R-multiple TP carries the real intent in
        # stop_loss_spec / risk_reward — the percent is just an engine fallback, so
        # don't tag that fallback "user" (it would render as "2%" and mask the true
        # "1 × ATR" / "1.5R" in the readback).
        sl_user_pct = spec.stop_loss.source == "user" and spec.stop_loss.type == "percent"
        tp = spec.take_profit  # may be None when a trailing_take_profit supplies the exit
        tp_user_pct = tp is not None and tp.source == "user" and tp.type == "percent"
        rms["stop_loss_pct"] = "user" if sl_user_pct else "default"
        rms["take_profit_pct"] = "user" if tp_user_pct else "default"
        # Surface a risk:reward target as first-class user intent so the readback
        # shows "1.5R" instead of a stale/derived percentage.
        if tp is not None and tp.type == "risk_reward":
            rr = tp.numeric()
            if rr and rr > 0:
                store["risk_reward"] = rr
                rms["risk_reward"] = "user" if tp.source == "user" else "default"
    if spec.position_sizing_mode:
        builder.position_sizing_mode = spec.position_sizing_mode
    if spec.max_capital_allocation_pct is not None:
        builder.max_capital_allocation_pct = spec.max_capital_allocation_pct
    # Declared indicators + exact params for the read-back (e.g. MACD {12,26,9}).
    if spec.indicators:
        builder._direct_indicators = {
            ind.name: dict(ind.params or {}) for ind in spec.indicators
        }
    _apply_gates_to_builder(builder, spec.gates)


async def plan_via_direct(
    builder: Any,
    *,
    history: list[dict] | None = None,
    llm: Any = None,
    session_id: str | None = None,
    turn: int = 0,
    max_repairs: int | None = None,
) -> tuple[StrategySpec | None, dict | None, str | None]:
    """Narrow-branch planning step: produce a validated spec + a legacy ``plan`` dict.

    Returns ``(spec, plan, None)`` on success (spec stashed on ``builder._direct_spec``,
    extras applied, notes on ``builder._direct_notes``), or ``(None, None, message)`` when
    synthesis can't produce a runnable strategy after repairs.
    """
    settings = get_settings()
    if max_repairs is None:
        max_repairs = settings.direct_strategy_max_repairs
    if llm is None:
        from app.services.ai.llm import LLMService
        llm = LLMService()

    messages: list[dict] = [{"role": "user", "content": _builder_context_message(builder)}]
    if history:
        messages = [*history, *messages]

    logger.info("🧭 direct_flow | plan_via_direct | session=%s | symbol=%s",
                session_id, getattr(builder, "symbol", None))
    try:
        spec, result = await generate_strategy(
            messages,
            llm=llm,
            system_prompt=build_strategy_system_prompt(),
            max_repairs=max_repairs,
            max_output_tokens=settings.direct_strategy_max_output_tokens,
            reasoning_effort=settings.direct_strategy_reasoning_effort,
            session_id=session_id,
            turn=turn,
        )
    except Exception as exc:  # noqa: BLE001 — never crash the chat turn
        logger.error("❌ direct_flow | plan_via_direct crashed | session=%s | %r", session_id, exc)
        return None, None, "Something went wrong while building the strategy. Please try again."

    if spec is None:
        logger.warning("⚠️ direct_flow | plan_via_direct failed | session=%s | codes=%s",
                       session_id, [e.code for e in result.errors])
        return None, None, _error_for_user(result)

    spec = _reconcile_identity(spec, builder)
    apply_spec_extras_to_builder(builder, spec)
    builder._direct_spec = spec
    builder._direct_notes = _notes_for_user(result)
    _log_spec_breakdown(spec, session_id=session_id, stage="plan_via_direct")
    logger.info(
        "✅ direct_flow | plan_via_direct ok | session=%s | %s",
        session_id, ("universe=" + spec.universe.source.kind) if spec.is_dynamic else ("symbol=" + str(spec.symbol)),
    )
    return spec, spec_to_plan(spec), None


# ── User-facing summaries ─────────────────────────────────────────────────────

def _notes_for_user(result: ValidationResult) -> list[str]:
    """Turn assumed-value / reduction notes into short lines the chat can show."""
    return [f"⚠️ {n.message}" for n in result.notes]


def _error_for_user(result: ValidationResult) -> str:
    """A friendly summary when synthesis can't produce a runnable strategy."""
    if not result.errors:
        return "I couldn't build a valid strategy from that. Could you rephrase the idea?"
    bullets = "\n".join(f"• {e.message}" for e in result.errors[:4])
    return (
        "I couldn't turn that into a backtest-ready strategy yet:\n"
        f"{bullets}\n"
        "Could you clarify or simplify the idea?"
    )


# ── Orchestration ─────────────────────────────────────────────────────────────

async def synthesize(
    builder: Any,
    *,
    history: list[dict] | None = None,
    llm: Any = None,
    session_id: str | None = None,
    turn: int = 0,
    max_repairs: int | None = None,
) -> DirectSynthesisResult:
    """Generate + validate + render a strategy from the gathered builder context.

    On success: populates ``builder``, writes the engine YAML, returns ``ok=True`` with
    ``yaml_path`` and any assumed-value ``notes``. On failure: ``ok=False`` with a
    user-facing ``error_message``; the builder is left untouched.
    """
    settings = get_settings()
    if max_repairs is None:
        max_repairs = settings.direct_strategy_max_repairs
    if llm is None:
        from app.services.ai.llm import LLMService
        llm = LLMService()

    messages: list[dict] = [{"role": "user", "content": _builder_context_message(builder)}]
    if history:
        messages = [*history, *messages]

    logger.info(
        "🧭 direct_flow | synthesis start | session=%s | symbol=%s tf=%s",
        session_id, getattr(builder, "symbol", None), getattr(builder, "timeframe", None),
    )

    try:
        spec, result = await generate_strategy(
            messages,
            llm=llm,
            system_prompt=build_strategy_system_prompt(),
            max_repairs=max_repairs,
            max_output_tokens=settings.direct_strategy_max_output_tokens,
            reasoning_effort=settings.direct_strategy_reasoning_effort,
            session_id=session_id,
            turn=turn,
        )
    except Exception as exc:  # noqa: BLE001 — never let synthesis crash the chat turn
        logger.error("❌ direct_flow | synthesis crashed | session=%s | %r", session_id, exc)
        return DirectSynthesisResult(
            ok=False,
            error_message="Something went wrong while building the strategy. Please try again.",
        )

    if spec is None:
        logger.warning(
            "⚠️ direct_flow | synthesis failed after repairs | session=%s | codes=%s",
            session_id, [e.code for e in result.errors],
        )
        return DirectSynthesisResult(
            ok=False, error_message=_error_for_user(result), validation=result,
        )

    spec = _reconcile_identity(spec, builder)
    populate_builder_from_spec(builder, spec)
    _log_spec_breakdown(spec, session_id=session_id, stage="synthesize")

    try:
        yaml_path = write_spec_yaml(spec, session_id=session_id, turn=turn)
    except Exception as exc:  # noqa: BLE001 — write failure shouldn't crash the turn
        logger.error("❌ direct_flow | YAML write failed | session=%s | %r", session_id, exc)
        return DirectSynthesisResult(
            ok=False,
            spec=spec,
            error_message="I built the strategy but couldn't save it for backtesting. Please retry.",
            validation=result,
        )

    notes = _notes_for_user(result)
    logger.info(
        "✅ direct_flow | synthesis complete | session=%s | symbol=%s | yaml=%s | notes=%d",
        session_id, spec.symbol, yaml_path, len(notes),
    )
    return DirectSynthesisResult(
        ok=True, spec=spec, yaml_path=yaml_path, notes=notes, validation=result,
    )
