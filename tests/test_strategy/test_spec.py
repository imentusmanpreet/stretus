"""
Engine-independent tests for StrategySpec — run everywhere (no TA-Lib needed).

These cover the pure-Python contract: rich-RM resolution, RR→percent, and the
formula-mode YAML rendering the engine loader consumes.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.strategy.spec import RiskManagement, StopLoss, StrategySpec, TakeProfit


def _spec(**overrides) -> StrategySpec:
    base = dict(
        name="INFY 15m momentum",
        symbol="INFY.NS",
        market="indian_stocks",
        timeframe="15m",
        objective="intraday",
        direction="long_only",
        entry_condition="CLOSE > EMA(20) AND RSI(14) > 60",
        exit_condition="RSI(14) < 50",
        stop_loss=StopLoss(type="percent", value=1.5, source="user"),
        take_profit=TakeProfit(type="risk_reward", value=2.0, source="user"),
    )
    base.update(overrides)
    return StrategySpec(**base)


def test_risk_reward_resolves_to_percent():
    s = _spec()
    assert s.resolved_stop_loss_pct() == 1.5
    assert s.resolved_take_profit_pct() == 3.0  # 1.5 × 2


def test_string_values_are_coerced():
    s = _spec(stop_loss=StopLoss(type="percent", value="2%"), take_profit=TakeProfit(type="percent", value="4"))
    assert s.resolved_stop_loss_pct() == 2.0
    assert s.resolved_take_profit_pct() == 4.0


def test_percent_stop_maps_to_engine_spec():
    s = _spec()
    assert s.stop_loss_engine_spec() == {"type": "percent", "pct": 1.5}


def test_atr_stop_maps_to_engine_spec():
    s = _spec(stop_loss=StopLoss(type="atr", value=2.0, window=14))
    spec = s.stop_loss_engine_spec()
    assert spec["type"] == "atr" and spec["multiplier"] == 2.0 and spec["window"] == 14


def test_structure_stop_uses_directional_anchor():
    s = _spec(direction="long_only", stop_loss=StopLoss(type="structure", window=5))
    spec = s.stop_loss_engine_spec()
    assert spec["type"] == "structural" and spec["anchor"] == "prev_n_bar_low"
    s_short = _spec(direction="short_only", stop_loss=StopLoss(type="structure"))
    assert s_short.stop_loss_engine_spec()["anchor"] == "prev_n_bar_high"


def test_unsupported_stop_types_fall_back_to_percent():
    for t in ("fixed_points", "indicator_based"):
        s = _spec(stop_loss=StopLoss(type=t, value=10))
        assert s.stop_loss_engine_spec() is None  # no native engine spec
        assert s.resolved_stop_loss_pct() > 0     # but a runnable percent always exists


def test_yaml_dict_is_formula_mode_never_registry():
    d = _spec().to_engine_yaml_dict()
    assert d["entry_evaluation_mode"] == "formula"
    assert d["exit_evaluation_mode"] == "formula"
    assert "entry_signals" not in d and "exit_signals" not in d
    assert d["entry"]["condition"] == "CLOSE > EMA(20) AND RSI(14) > 60"
    assert d["risk_management"] == {"stop_loss_percent": 1.5, "take_profit_percent": 3.0}


def test_yaml_dict_carries_profile_and_risk_management():
    s = _spec(
        risk_management=RiskManagement(
            risk_per_trade_pct=1.0, daily_loss_cap_pct=3.0, max_trades_per_day=3
        )
    )
    d = s.to_engine_yaml_dict()
    assert d["profile"]["objective"] == "intraday"
    assert d["profile"]["max_trades_per_day"] == 3
    assert d["profile"]["daily_loss_cap_pct"] == 3.0
    assert d["per_trade_risk_pct"] == 1.0


def test_extra_fields_are_forbidden():
    with pytest.raises(PydanticValidationError):
        StopLoss(type="percent", value=1, bogus=True)
    with pytest.raises(PydanticValidationError):
        _spec(unexpected_field="x")


def test_stop_loss_is_required():
    # stop_loss has no default — a spec without it is a Pydantic error.
    with pytest.raises(PydanticValidationError):
        StrategySpec(
            name="x", symbol="INFY.NS", market="indian_stocks", timeframe="15m",
            objective="intraday", direction="long_only", entry_condition="CLOSE > EMA(20)",
            take_profit=TakeProfit(type="percent", value=2.0),
        )


def test_take_profit_is_optional_when_trailing_present():
    from app.strategy.spec import TrailingTakeProfit
    # No static take_profit, but a trailing one supplies the profit exit.
    s = _spec(take_profit=None, trailing_take_profit=TrailingTakeProfit(distance_pct=0.5, activate_after_pct=1.0))
    # No static target → resolved percent is 0 and the engine reads "no cap".
    assert s.resolved_take_profit_pct() == 0.0
    assert s.to_engine_yaml_dict()["risk_management"]["take_profit_percent"] == 0.0
    assert s.to_engine_yaml_dict()["trailing_take_profit"]["distance_pct"] == 0.5
