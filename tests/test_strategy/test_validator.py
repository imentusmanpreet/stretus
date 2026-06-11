"""
Tests for the strict validator.

Split into two groups:
  * fail-closed behaviour — runs everywhere (it asserts what happens when the
    engine grammar is unavailable), so it is meaningful even without TA-Lib.
  * engine-backed checks — gated behind ``engine_available`` (needs TA-Lib +
    quant_engine), so they run in CI/Docker where the engine is present.
"""
from __future__ import annotations

import pytest

from app.strategy import engine_bridge, validator
from app.strategy.spec import StopLoss, StrategySpec, TakeProfit

ENGINE = engine_bridge.is_available()
requires_engine = pytest.mark.skipif(not ENGINE, reason="quant engine (TA-Lib) not available")


def _spec(**overrides) -> StrategySpec:
    base = dict(
        name="t", symbol="INFY.NS", market="indian_stocks", timeframe="15m",
        objective="intraday", direction="long_only",
        entry_condition="CLOSE > EMA(20) AND RSI(14) > 60",
        exit_condition="RSI(14) < 50",
        stop_loss=StopLoss(type="percent", value=1.5, source="user"),
        take_profit=TakeProfit(type="risk_reward", value=2.0, source="user"),
    )
    base.update(overrides)
    return StrategySpec(**base)


# ── fail-closed (runs without the engine) ─────────────────────────────────────

def test_engine_unavailable_fails_closed(monkeypatch):
    """No engine ⇒ exactly one blocking engine_unavailable error, never a silent pass."""
    monkeypatch.setattr(engine_bridge, "is_available", lambda: False)
    result = validator.validate_spec(_spec())
    assert not result.ok
    assert [e.code for e in result.errors] == ["engine_unavailable"]


# ── engine-backed checks (CI/Docker) ──────────────────────────────────────────

@requires_engine
def test_valid_spec_passes():
    result = validator.validate_spec(_spec())
    assert result.ok, result.as_repair_text()


@requires_engine
def test_unknown_function_is_rejected():
    result = validator.validate_spec(_spec(entry_condition="KC_UPPER(20) > CLOSE"))
    assert any(e.code == "condition_invalid" for e in result.errors)


@requires_engine
def test_unknown_bare_identifier_is_rejected():
    result = validator.validate_spec(_spec(entry_condition="FOO > CLOSE"))
    assert any(e.code == "unknown_identifier" for e in result.errors)


@requires_engine
def test_grammatically_valid_entry_passes_without_behavioural_probe():
    """The 800-bar satisfiability probe was removed — a grammatically valid entry passes
    even if it would rarely/never fire; the real backtest is the ground truth on trades."""
    result = validator.validate_spec(_spec(entry_condition="RSI(14) >= 70 AND RSI(14) <= 30"))
    assert result.ok
    assert not any(
        x.code in ("entry_never_fires", "entry_may_be_too_selective")
        for x in (*result.errors, *result.notes)
    )


@requires_engine
def test_both_direction_requires_short_leg():
    result = validator.validate_spec(_spec(direction="both"))
    assert any(e.code == "missing_short_leg" for e in result.errors)


@requires_engine
def test_unsupported_timeframe_is_rejected():
    # 2m is now valid (range-based 1m..1d). Use an out-of-range timeframe.
    result = validator.validate_spec(_spec(timeframe="2w"))
    assert any(e.code == "unsupported_timeframe" for e in result.errors)


@requires_engine
def test_assumed_values_become_notes_not_errors():
    s = _spec(
        stop_loss=StopLoss(type="percent", value=2.0, source="assumed"),
        take_profit=TakeProfit(type="percent", value=4.0, source="assumed"),
    )
    result = validator.validate_spec(s)
    assert result.ok
    codes = {n.code for n in result.notes}
    assert "assumed_stop_loss" in codes and "assumed_take_profit" in codes
