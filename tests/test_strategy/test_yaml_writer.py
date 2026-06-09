"""
yaml_writer (render) tests.

The write + shape checks run everywhere; the engine round-trip (loading the YAML back
through the real engine loader) is gated behind engine availability.
"""
from __future__ import annotations

import yaml

from app.strategy import engine_bridge
from app.strategy.spec import StopLoss, StrategySpec, TakeProfit
from app.strategy.yaml_writer import write_spec_yaml

import pytest

requires_engine = pytest.mark.skipif(
    not engine_bridge.is_available(), reason="quant engine (TA-Lib) not available"
)


def _spec() -> StrategySpec:
    return StrategySpec(
        name="INFY 15m momentum", symbol="INFY.NS", market="indian_stocks",
        timeframe="15m", objective="intraday", direction="long_only",
        entry_condition="CLOSE > EMA(20) AND RSI(14) > 60",
        exit_condition="RSI(14) < 50",
        stop_loss=StopLoss(type="percent", value=1.5, source="user"),
        take_profit=TakeProfit(type="risk_reward", value=2.0, source="user"),
    )


def test_writes_file_with_strategy_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.strategy.yaml_writer._candidate_strategy_folders", lambda: [str(tmp_path)]
    )
    path = write_spec_yaml(_spec())
    assert path.endswith(".yaml")
    doc = yaml.safe_load(open(path).read())
    assert "strategy" in doc
    strat = doc["strategy"]
    assert strat["entry_evaluation_mode"] == "formula"
    assert strat["entry"]["condition"] == "CLOSE > EMA(20) AND RSI(14) > 60"
    assert strat["risk_management"]["take_profit_percent"] == 3.0


@requires_engine
def test_engine_loads_written_yaml(tmp_path, monkeypatch):
    """The written YAML must load cleanly through the real engine loader (the contract)."""
    monkeypatch.setattr(
        "app.strategy.yaml_writer._candidate_strategy_folders", lambda: [str(tmp_path)]
    )
    path = write_spec_yaml(_spec())
    engine_bridge._ensure_engine_on_path()
    from engine.loader import load_strategy  # type: ignore[import-not-found]

    cfg = load_strategy(path)
    assert cfg.symbol == "INFY.NS"
    assert cfg.timeframe == "15m"
    assert cfg.entry_evaluation_mode == "formula"
    assert cfg.stop_loss > 0 and cfg.take_profit > 0
