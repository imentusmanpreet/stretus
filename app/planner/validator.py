"""
app/planner/validator.py — Post-pick invariants.

A plan that violates any of these invariants must NEVER be shipped. We raise
PlanInvariantViolation so the chat layer surfaces a clear error instead of
sending the user a broken strategy.
"""
from __future__ import annotations

from app.kb.schemas import StrategyPlan
from app.planner.filters import opposite_direction


class PlanInvariantViolation(RuntimeError):
    """The planner produced a logically inconsistent plan — never ship it."""


def validate_plan(plan: StrategyPlan) -> None:
    """Raise on any invariant violation. Return None on success."""

    # 1. Entry trigger direction must align with sentiment.
    if plan.entry_trigger.direction not in {plan.sentiment, "neutral"}:
        raise PlanInvariantViolation(
            f"entry_trigger '{plan.entry_trigger.name}' direction={plan.entry_trigger.direction} "
            f"does not align with sentiment={plan.sentiment}"
        )

    # 2. Exit trigger direction must be opposite to sentiment (or neutral).
    expected_exit_dir = opposite_direction(plan.sentiment)
    if plan.exit_trigger.direction not in {expected_exit_dir, "neutral"}:
        raise PlanInvariantViolation(
            f"exit_trigger '{plan.exit_trigger.name}' direction={plan.exit_trigger.direction} "
            f"is not opposite of sentiment={plan.sentiment} (expected {expected_exit_dir})"
        )

    # 3. Every entry filter must align with sentiment too.
    for picked in plan.entry_filters:
        if picked.direction not in {plan.sentiment, "neutral"}:
            raise PlanInvariantViolation(
                f"entry_filter '{picked.name}' direction={picked.direction} "
                f"does not align with sentiment={plan.sentiment}"
            )

    # 4. Every exit filter must be opposite to sentiment (or neutral) — same rule as exit_trigger.
    for picked in plan.exit_filters:
        if picked.direction not in {expected_exit_dir, "neutral"}:
            raise PlanInvariantViolation(
                f"exit_filter '{picked.name}' direction={picked.direction} "
                f"is not opposite of sentiment={plan.sentiment} (expected {expected_exit_dir})"
            )

    # 5. No signal may appear in two roles simultaneously.
    role_names = [plan.entry_trigger.name]
    role_names.extend(p.name for p in plan.entry_filters)
    role_names.append(plan.exit_trigger.name)
    role_names.extend(p.name for p in plan.exit_filters)
    if len(set(role_names)) != len(role_names):
        raise PlanInvariantViolation(
            f"signal appears in multiple roles: {role_names}"
        )

    # 6. SL/TP must be positive and TP > SL (otherwise risk:reward < 1).
    if plan.sl_pct <= 0 or plan.tp_pct <= 0:
        raise PlanInvariantViolation(f"sl_pct/tp_pct must be positive: sl={plan.sl_pct} tp={plan.tp_pct}")
    if plan.tp_pct < plan.sl_pct:
        raise PlanInvariantViolation(
            f"tp_pct ({plan.tp_pct}) must be >= sl_pct ({plan.sl_pct}) for a positive risk:reward"
        )
