"""Phase 3 builder integration: SL specs propagate from preset → builder → YAML."""
from __future__ import annotations

from app.services.strategy.builder import StrategyBuilder


def _populate_minimum(b: StrategyBuilder) -> None:
    b.symbol = "RELIANCE.NS"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "ORB structural test"
    b.stop_loss = 1.5
    b.take_profit = 3.0


def test_builder_writes_stop_loss_spec_to_yaml():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.stop_loss_spec = {
        "type": "structural",
        "anchor": "opening_range_low",
        "opening_bars": 3,
        "padding_pct": 0.1,
    }

    yaml_dict = b.to_yaml_dict()
    assert "stop_loss" in yaml_dict["strategy"]
    assert yaml_dict["strategy"]["stop_loss"]["anchor"] == "opening_range_low"
    # Legacy stop_loss_percent stays for the % fallback the simulator uses
    # when the structural spec can't be resolved at entry time.
    assert yaml_dict["strategy"]["risk_management"]["stop_loss_percent"] == 1.5


def test_builder_writes_trailing_stop_spec_to_yaml():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.trailing_stop_spec = {"type": "ema", "window": 20, "activate_after_pct": 0.5}

    yaml_dict = b.to_yaml_dict()
    assert "trailing_stop" in yaml_dict["strategy"]
    assert yaml_dict["strategy"]["trailing_stop"]["type"] == "ema"
    assert yaml_dict["strategy"]["trailing_stop"]["window"] == 20


def test_builder_omits_blocks_when_specs_unset():
    b = StrategyBuilder()
    _populate_minimum(b)
    yaml_dict = b.to_yaml_dict()
    assert "stop_loss" not in yaml_dict["strategy"]
    assert "trailing_stop" not in yaml_dict["strategy"]


def test_apply_signal_plan_captures_specs_from_underscore_keys():
    """The legacy_bridge attaches `_stop_loss_spec` / `_trailing_stop_spec`
    to the signal plan. apply_signal_plan should hoist them onto the builder."""
    b = StrategyBuilder()
    _populate_minimum(b)
    b.apply_signal_plan({
        "entry": [], "exit": [],
        "_stop_loss_spec":     {"type": "atr", "multiplier": 1.5, "window": 14},
        "_trailing_stop_spec": {"type": "percent", "distance_pct": 0.8},
    })
    assert b.stop_loss_spec == {"type": "atr", "multiplier": 1.5, "window": 14}
    assert b.trailing_stop_spec == {"type": "percent", "distance_pct": 0.8}
    yaml_dict = b.to_yaml_dict()
    assert yaml_dict["strategy"]["stop_loss"]["type"] == "atr"
    assert yaml_dict["strategy"]["trailing_stop"]["type"] == "percent"


# ── Phase 4: reference_symbol propagation ────────────────────────────────────


def test_builder_writes_reference_symbol_to_yaml():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.reference_symbol = "^NSEI"
    yaml_dict = b.to_yaml_dict()
    assert yaml_dict["strategy"]["reference_symbol"] == "^NSEI"


def test_apply_signal_plan_captures_reference_symbol():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.apply_signal_plan({
        "entry": [], "exit": [],
        "_reference_symbol": "^nsei",            # planner may pass lowercase
    })
    assert b.reference_symbol == "^NSEI"          # builder normalizes
    yaml_dict = b.to_yaml_dict()
    assert yaml_dict["strategy"]["reference_symbol"] == "^NSEI"


def test_builder_omits_reference_symbol_when_unset():
    b = StrategyBuilder()
    _populate_minimum(b)
    yaml_dict = b.to_yaml_dict()
    assert "reference_symbol" not in yaml_dict["strategy"]
