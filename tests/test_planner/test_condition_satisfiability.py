"""
Phase 5 — validation gate: an assembled entry rule that can NEVER fire (so the
strategy would take zero trades) must be caught before backtest.

Covers the real bug seen in the wild: the planner injected a collapsed RSI band
``RSI(14) >= 0 AND RSI(14) <= 0`` which can never be true.
"""
from __future__ import annotations

import pytest

from app.planner.condition_satisfiability import (
    check_entry_satisfiable,
    entry_fire_count,
)
from app.planner.fidelity_validator import validate_strategy_fidelity


# The actual broken condition the assembler produced for the ETHUSDC strategy.
USER_BROKEN = (
    "CLOSE > VWAP AND VOL > AVG(VOL, 20) * 1.0 AND CLOSE > EMA(20) "
    "AND RSI(14) >= 0 AND RSI(14) <= 0"
)

UNSATISFIABLE = [
    USER_BROKEN,
    "CLOSE > EMA(20) AND CLOSE < EMA(20)",   # direct contradiction
    "RSI(14) >= 70 AND RSI(14) <= 30",       # reversed band (min > max)
    "RSI(14) > 100",                          # RSI can't exceed 100
]

SATISFIABLE = [
    "CLOSE > VWAP AND VOL > AVG(VOL, 20) * 1.0 AND CLOSE > EMA(20)",
    "RSI(14) < 30",
    "RSI(14) > 70",
    "CCI(20) > 100",
    "CDL_ENGULFING > 0",            # candlestick (sparse) still fires on the probe
    "CDL_HAMMER > 0",
    "CLOSE > VWAP AND RSI(14) < 35 AND VOL > AVG(VOL, 20) * 1.5",  # strict but valid
    "CLOSE > DONCHIAN_UPPER(20)",
    "STOCH_K < 20",
    "MACD > MACD_SIGNAL",
    "EMA(20) > EMA(50)",
]


@pytest.mark.parametrize("cond", UNSATISFIABLE)
def test_unsatisfiable_conditions_are_flagged(cond):
    assert entry_fire_count(cond) == 0
    finding = check_entry_satisfiable(cond)
    assert finding is not None
    assert finding["severity"] == "critical"
    assert finding["code"] == "entry_never_fires"


@pytest.mark.parametrize("cond", SATISFIABLE)
def test_valid_conditions_are_not_flagged(cond):
    assert (entry_fire_count(cond) or 0) > 0, f"{cond} should fire on the probe"
    assert check_entry_satisfiable(cond) is None


INVALID_FUNCTION = [
    # Real bug: planner emitted a Keltner-channel function that isn't wired up.
    "CLOSE > KC_UPPER(20) AND VOL > AVG(VOL, 20) * 1.5",
    "CLOSE < KC_LOWER(20)",
    "BB_WIDTH(20) < 0.5",          # bb_squeeze card references this unknown fn
    "CMF(20) > 0",                  # chaikin money flow — not wired
]


@pytest.mark.parametrize("cond", INVALID_FUNCTION)
def test_invalid_function_conditions_are_flagged(cond):
    finding = check_entry_satisfiable(cond)
    assert finding is not None
    assert finding["severity"] == "critical"
    assert finding["code"] == "entry_condition_invalid"


def test_valid_bollinger_band_condition_passes():
    # The user asked for Bollinger Bands; BB_UPPER is the correct (wired) name.
    cond = "CLOSE > BB_UPPER(20) AND VOL > AVG(VOL, 20) * 1.5"
    assert check_entry_satisfiable(cond) is None


def test_empty_condition_is_ignored():
    assert entry_fire_count("") is None
    assert check_entry_satisfiable("") is None


def test_fidelity_validator_surfaces_never_fires():
    """End-to-end: the fidelity gate must raise the entry_never_fires finding for
    the real broken ETHUSDC condition."""
    findings = validate_strategy_fidelity(
        "Trade ETHUSDC, long when above VWAP and 20 EMA, enter on engulfing.",
        signal_plan={"entry": [], "exit": [], "entry_condition": USER_BROKEN},
        risk_execution_config={},
    )
    codes = {f.code for f in findings}
    assert "entry_never_fires" in codes
    bad = next(f for f in findings if f.code == "entry_never_fires")
    assert bad.severity == "critical"


def test_fidelity_validator_passes_a_firing_condition():
    findings = validate_strategy_fidelity(
        "Long when price above VWAP with above-average volume.",
        signal_plan={
            "entry": [],
            "exit": [],
            "entry_condition": "CLOSE > VWAP AND VOL > AVG(VOL, 20) * 1.0 AND CLOSE > EMA(20)",
        },
        risk_execution_config={},
    )
    assert "entry_never_fires" not in {f.code for f in findings}
