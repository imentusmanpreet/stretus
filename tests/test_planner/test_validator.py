"""Plan invariant tests."""
from __future__ import annotations

import pytest

from app.kb import kb
from app.kb.schemas import Intent, PickedSignal, StrategyPlan
from app.planner.validator import PlanInvariantViolation, validate_plan


def _picked(card_name: str, role: str) -> PickedSignal:
    card = kb.signals[card_name]
    params = card.params_by_timeframe.get("15m", {}) or next(iter(card.params_by_timeframe.values()))
    return PickedSignal.from_card(card, role, dict(params), "15m")


def _plan(entry: str, filt: str | None, exit_: str, sentiment: str = "bullish") -> StrategyPlan:
    return StrategyPlan(
        stock=kb.stocks["HDFCBANK.NS"],
        timeframe="15m",
        sentiment=sentiment,
        intent=Intent(**kb.intent_taxonomy.defaults),
        entry_trigger=_picked(entry, "entry_trigger"),
        entry_filters=[_picked(filt, "entry_filter")] if filt else [],
        exit_trigger=_picked(exit_, "exit_trigger"),
        sl_pct=1.5,
        tp_pct=3.0,
        risk={},
    )


def test_valid_bullish_plan_passes():
    validate_plan(_plan("ema_cross_up", "price_above_ema", "rsi_cross_down"))


def test_bearish_entry_for_bullish_sentiment_raises():
    with pytest.raises(PlanInvariantViolation):
        validate_plan(_plan("ema_cross_down", None, "rsi_cross_down", sentiment="bullish"))


def test_bullish_exit_for_bullish_sentiment_raises():
    # exit_trigger must be bearish (or neutral) for a bullish strategy
    with pytest.raises(PlanInvariantViolation):
        validate_plan(_plan("ema_cross_up", None, "ema_cross_up", sentiment="bullish"))


def test_duplicate_signal_across_roles_raises():
    with pytest.raises(PlanInvariantViolation):
        validate_plan(_plan("ema_above", "ema_above", "rsi_cross_down"))


def test_negative_sl_raises():
    plan = _plan("ema_cross_up", None, "rsi_cross_down")
    plan.sl_pct = -1.0
    with pytest.raises(PlanInvariantViolation):
        validate_plan(plan)


def test_tp_less_than_sl_raises():
    plan = _plan("ema_cross_up", None, "rsi_cross_down")
    plan.sl_pct = 2.0
    plan.tp_pct = 1.0
    with pytest.raises(PlanInvariantViolation):
        validate_plan(plan)
