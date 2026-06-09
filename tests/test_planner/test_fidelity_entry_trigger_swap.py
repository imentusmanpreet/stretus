"""
Phase 5 — fidelity check: the entry TRIGGER must use the indicator the user
asked to enter on. Catches the real bug where "enter above the upper Bollinger
Band" was assembled with a Keltner-channel breakout trigger (a sibling family),
which the family-presence check missed because a bb_squeeze filter made the BB
family look "used".
"""
from __future__ import annotations

from app.planner.fidelity_validator import validate_strategy_fidelity


def _codes(prompt: str, plan: dict) -> set[str]:
    findings = validate_strategy_fidelity(prompt, signal_plan=plan, risk_execution_config={})
    return {f.code for f in findings}


def test_bollinger_request_with_keltner_trigger_is_flagged():
    plan = {
        "entry": [
            {"name": "keltner_breakout_up", "signal_type": "TRIGGER"},
            {"name": "bb_squeeze", "signal_type": "FILTER"},   # makes BB family "present"
        ],
        "exit": [],
        "entry_condition": "CLOSE > KC_UPPER(20)",
    }
    codes = _codes(
        "Use Bollinger Bands. Enter long when price closes above the upper Bollinger Band.",
        plan,
    )
    assert "entry_trigger_swap" in codes


def test_bollinger_request_with_bollinger_trigger_is_clean():
    plan = {
        "entry": [{"name": "bb_breakout_up", "signal_type": "TRIGGER"}],
        "exit": [],
        "entry_condition": "CLOSE > BB_UPPER(20)",
    }
    assert "entry_trigger_swap" not in _codes(
        "Enter long when price closes above the upper Bollinger Band.", plan
    )


def test_explicit_keltner_request_is_not_flagged():
    plan = {
        "entry": [{"name": "keltner_breakout_up", "signal_type": "TRIGGER"}],
        "exit": [],
        "entry_condition": "CLOSE > KC_UPPER(20)",
    }
    assert "entry_trigger_swap" not in _codes(
        "Enter on a Keltner channel breakout.", plan
    )


def test_sma_ema_entry_trigger_swap_is_flagged():
    plan = {
        "entry": [{"name": "ema_cross_up", "signal_type": "TRIGGER"}],
        "exit": [],
        "entry_condition": "EMA(20) > EMA(50)",
    }
    assert "entry_trigger_swap" in _codes("Enter on an SMA crossover.", plan)


def test_no_swap_when_indicator_not_a_sibling():
    """ATR mentioned for a stop, entry is a Donchian breakout — different roles,
    not a sibling swap, so the trigger-swap check stays silent."""
    plan = {
        "entry": [{"name": "donchian_breakout_up", "signal_type": "TRIGGER"}],
        "exit": [],
        "entry_condition": "CLOSE > DONCHIAN_UPPER(20)",
    }
    assert "entry_trigger_swap" not in _codes(
        "Breakout strategy using ATR for the stop loss.", plan
    )
