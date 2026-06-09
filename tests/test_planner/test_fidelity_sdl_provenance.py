"""Fidelity must trust the SDL provenance for risk-field origin.

Regression for the ICICIBANK case: the user wrote "Stop Loss: 0.75%, Take
Profit: 1.5%, Risk Reward: 1:2" and the SDL captured them (field_sources=user),
but the legacy fidelity validator re-derived origin from the unreliable
rms_sources (system_default) and raised 3 FALSE "please confirm" criticals that
blocked assembly. When the SDL says a risk field is user-stated, trust it.
"""
from __future__ import annotations

from app.planner.fidelity_validator import validate_strategy_fidelity

_PROMPT = "Stop Loss: 0.75% Take Profit: 1.5% Risk Reward: 1:2"
_REC = {
    "stop_loss_pct": 0.75,
    "take_profit_pct": 1.5,
    "risk_reward": 2.0,
    "rms_sources": {
        "stop_loss_pct": "system_default",
        "take_profit_pct": "system_default",
        "risk_reward": "system_default",
    },
}
_PLAN = {"entry_condition": "EMA(9) > EMA(21) AND CLOSE > VWAP", "entry": [], "exit": []}


def _codes(**kw) -> set[str]:
    return {f.code for f in validate_strategy_fidelity(
        _PROMPT, signal_plan=_PLAN, risk_execution_config=_REC, **kw)}


def test_legacy_path_still_flags_when_no_sdl_provenance():
    # Backward compatible: without SDL provenance the legacy behaviour is intact.
    codes = _codes()
    assert "sl_system_default" in codes
    assert "tp_system_default" in codes
    assert "rr_not_from_user" in codes


def test_sdl_user_sourced_risk_suppresses_false_criticals():
    fs = {
        "risk.stop_loss": "user", "risk.stop_loss.value": "user",
        "risk.take_profit": "user", "risk.take_profit.value": "user",
    }
    codes = _codes(sdl_field_sources=fs)
    assert "sl_system_default" not in codes
    assert "tp_system_default" not in codes
    assert "rr_not_from_user" not in codes   # RR implied by user SL+TP


def test_partial_provenance_only_suppresses_that_field():
    # User stated SL but not TP → suppress SL, still flag TP.
    codes = _codes(sdl_field_sources={"risk.stop_loss.value": "user"})
    assert "sl_system_default" not in codes
    assert "tp_system_default" in codes


def test_explicit_rr_provenance_suppresses_rr():
    codes = _codes(sdl_field_sources={"risk.take_profit.ratio": "user"})
    assert "rr_not_from_user" not in codes
