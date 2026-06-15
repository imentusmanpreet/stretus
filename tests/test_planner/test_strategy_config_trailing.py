"""The persisted strategy_config (what the live evaluator reads back) must carry
the trailing exit specs. SL/TP scalars live in risk_execution_config; the trailing
blocks live in strategy_config so strategy_evaluator can attach them to the bracket
order's trailing block for the OMS.
"""
from __future__ import annotations

from app.planner.strategy_assembler import build_strategy_config
from app.services.strategy.builder import StrategyBuilder


def _builder() -> StrategyBuilder:
    b = StrategyBuilder()
    b.market = "crypto"
    b.symbol = "BTC_USDT"
    b.goal = "scalp"
    return b


_PAYLOAD = {"strategy_object": {"name": "x", "entry": [], "exit": []}}


def test_trailing_take_profit_persisted_into_strategy_config():
    b = _builder()
    b.trailing_take_profit_spec = {"type": "percent", "distance_pct": 0.5, "activate_after_pct": 1.0}
    cfg = build_strategy_config(b, _PAYLOAD)
    assert cfg["trailing_take_profit"]["distance_pct"] == 0.5
    assert cfg["trailing_take_profit"]["activate_after_pct"] == 1.0


def test_trailing_stop_persisted_into_strategy_config():
    b = _builder()
    b.trailing_stop_spec = {"type": "percent", "distance_pct": 1.0}
    cfg = build_strategy_config(b, _PAYLOAD)
    assert cfg["trailing_stop"]["distance_pct"] == 1.0


def test_no_trailing_keys_when_unset():
    cfg = build_strategy_config(_builder(), _PAYLOAD)
    assert "trailing_take_profit" not in cfg
    assert "trailing_stop" not in cfg
