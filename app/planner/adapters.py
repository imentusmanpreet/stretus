"""
app/planner/adapters.py — Convert StrategyPlan into the wire-shape
expected by /api/v1/strategy/evaluate/execute (and by the existing
chat response composer).

The execution layer is unchanged. This adapter is the seam.
"""
from __future__ import annotations

from typing import Any

from app.kb.schemas import PickedSignal, StrategyPlan
from app.planner.formulas import render_formula


def _signal_to_obj(picked: PickedSignal | None) -> dict[str, Any] | None:
    if picked is None:
        return None
    return {
        "type": picked.name,
        "params": dict(picked.params),
    }


def _signal_to_plan_entry(picked: PickedSignal, signal_type: str = "TRIGGER") -> dict[str, Any]:
    """Shape used by the existing strategy_object/signal_plan blocks."""
    return {
        "name":        picked.name,
        "params":      dict(picked.params),
        "timeframe":   picked.timeframe,
        "signal_type": signal_type,
    }


def plan_to_strategy_config(plan: StrategyPlan) -> dict[str, Any]:
    """Build the dict accepted by /api/v1/strategy/evaluate/execute."""
    entry_block: dict[str, Any] = {
        "trigger": _signal_to_obj(plan.entry_trigger),
        "filters": [_signal_to_obj(f) for f in plan.entry_filters],
    }

    return {
        "strategy_config": {
            "symbol":        plan.stock.symbol,
            "timeframe":     plan.timeframe,
            "strategy_type": "intraday" if plan.timeframe in {"1m", "3m", "5m", "10m", "15m", "30m", "1h"} else "positional",
            "entry":         entry_block,
            "exit": {
                "trigger": _signal_to_obj(plan.exit_trigger),
                "filters": [_signal_to_obj(f) for f in plan.exit_filters],
            },
            "sl_tp": {
                "stop_loss_pct":   plan.sl_pct,
                "take_profit_pct": plan.tp_pct,
            },
            "risk": dict(plan.risk),
        },
    }


def plan_to_signal_plan(plan: StrategyPlan) -> dict[str, Any]:
    """Build the dict that fits into builder.signal_plan / strategy_object.

    Compatible with the existing chat response composer and YAML generator
    so drop-in replacement is safe.
    """
    entries = [_signal_to_plan_entry(plan.entry_trigger, "TRIGGER")]
    for picked in plan.entry_filters:
        entries.append(_signal_to_plan_entry(picked, "FILTER"))

    exits = [_signal_to_plan_entry(plan.exit_trigger, "TRIGGER")]
    for picked in plan.exit_filters:
        exits.append(_signal_to_plan_entry(picked, "FILTER"))

    signals_used = [plan.entry_trigger.name]
    signals_used.extend(p.name for p in plan.entry_filters)
    signals_used.append(plan.exit_trigger.name)
    signals_used.extend(p.name for p in plan.exit_filters)

    entry_parts = [render_formula(plan.entry_trigger.name, plan.entry_trigger.params)]
    for picked in plan.entry_filters:
        entry_parts.append(render_formula(picked.name, picked.params))
    entry_condition = " AND ".join(p for p in entry_parts if p) or None

    exit_parts = [render_formula(plan.exit_trigger.name, plan.exit_trigger.params)]
    for picked in plan.exit_filters:
        exit_parts.append(render_formula(picked.name, picked.params))
    exit_condition = " AND ".join(p for p in exit_parts if p) or None

    return {
        "entry":            entries,
        "exit":             exits,
        "entry_condition":  entry_condition,
        "exit_condition":   exit_condition,
        "signals_used":     signals_used,
        "signals_available": 0,                # filled by caller from kb if needed
        "rationale":        _rationale(plan),
        # Phase 3 — when a preset pinned structural / trailing SL specs, surface
        # them on the plan so the builder can copy them into the strategy YAML.
        # Underscore prefix keeps them out of the public _SIGNAL_PLAN_PUBLIC_KEYS
        # filter while still being addressable by builder.apply_signal_plan.
        "_stop_loss_spec":     plan.stop_loss_spec,
        "_trailing_stop_spec": plan.trailing_stop_spec,
        # Phase 4 — reference symbol passes through the same way so the
        # strategy YAML carries it and the API layer knows what to fetch.
        "_reference_symbol":   plan.reference_symbol,
        # Phase 5 — higher-timeframe rules ride along so the API layer knows
        # which HTF series to fetch and the YAML carries the gate definitions.
        "_htf_rules":          list(plan.htf_rules) if plan.htf_rules else [],
        # Phase 8b — wall-clock cutoff (e.g. 15:15 IST) carried into the YAML
        # so the simulator force-exits intraday positions on time.
        "_time_exit":          dict(plan.time_exit) if plan.time_exit else None,
    }


def _rationale(plan: StrategyPlan) -> str:
    if plan.entry_filters:
        names = ", ".join(p.name for p in plan.entry_filters)
        entry_filt = f" with {names} as entry confirmation filter{'s' if len(plan.entry_filters) > 1 else ''}"
    else:
        entry_filt = ""
    if plan.exit_filters:
        names = ", ".join(p.name for p in plan.exit_filters)
        exit_filt = f" and {names} as exit confirmation filter{'s' if len(plan.exit_filters) > 1 else ''}"
    else:
        exit_filt = ""
    return (
        f"For a {plan.sentiment} {plan.timeframe} setup on {plan.stock.display_name}, "
        f"selected {plan.entry_trigger.name} as the entry trigger{entry_filt}, "
        f"and {plan.exit_trigger.name} as the exit{exit_filt}. "
        f"SL={plan.sl_pct}% TP={plan.tp_pct}% (volatility-scaled)."
    )
