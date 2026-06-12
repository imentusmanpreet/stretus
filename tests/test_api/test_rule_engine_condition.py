"""Tests for rule_engine._eval_rule handling of formula-string condition triggers."""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.execution.rule_engine import RuleEngine


def _make_df(close: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": close, "open": close, "high": close, "low": close, "volume": [1000.0] * len(close)})


def test_condition_trigger_passes_when_formula_true():
    engine = RuleEngine()
    df = _make_df([100.0] * 30)
    # RSI on flat series stays around 50; CLOSE > 90 is always true here
    rule = {"type": "condition", "params": {"formula": "CLOSE > 90"}}
    result, msgs = engine._eval_rule(df, rule, "ENTRY trigger")
    assert result is True
    assert "condition" in msgs[-1]
    assert "PASS" in msgs[-1]


def test_condition_trigger_fails_when_formula_false():
    engine = RuleEngine()
    df = _make_df([50.0] * 30)
    rule = {"type": "condition", "params": {"formula": "CLOSE > 90"}}
    result, msgs = engine._eval_rule(df, rule, "ENTRY trigger")
    assert result is False
    assert "FAIL" in msgs[-1]


def test_condition_trigger_missing_formula_returns_false():
    engine = RuleEngine()
    df = _make_df([100.0] * 5)
    rule = {"type": "condition", "params": {}}
    result, msgs = engine._eval_rule(df, rule, "ENTRY trigger")
    assert result is False
    assert "no formula param" in msgs[-1]


def test_condition_trigger_invalid_formula_returns_false():
    engine = RuleEngine()
    df = _make_df([100.0] * 5)
    rule = {"type": "condition", "params": {"formula": "UNKNOWN_FUNC(99) > 0"}}
    result, msgs = engine._eval_rule(df, rule, "ENTRY trigger")
    # Unknown function → evaluate_condition returns False (logs a warning once)
    assert result is False


def test_unknown_kb_signal_still_returns_false():
    """Regression: non-condition unknown signals still fail as before."""
    engine = RuleEngine()
    df = _make_df([100.0] * 5)
    rule = {"type": "nonexistent_signal_xyz", "params": {}}
    result, msgs = engine._eval_rule(df, rule, "ENTRY trigger")
    assert result is False
    assert "unknown signal" in msgs[-1].lower()


def test_entry_block_with_formula_condition():
    engine = RuleEngine()
    df = _make_df([200.0] * 30)
    entry_block = {
        "trigger": {"type": "condition", "params": {"formula": "CLOSE > 100"}},
        "filters": [],
    }
    result, msgs = engine.evaluate_entry(df, entry_block)
    assert result is True
    assert any("All entry conditions passed" in m for m in msgs)
