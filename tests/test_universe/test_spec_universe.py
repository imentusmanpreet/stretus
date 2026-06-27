"""
Phase-A spec contract tests for the dynamic universe (engine-independent).

Covers the one-of validator (symbol XOR universe), the symbol-agnostic template render,
the static-path guard, and that ``universe`` is present in the LLM JSON schema (§16).
"""
from __future__ import annotations

import pytest

from app.strategy.spec import StrategySpec, UniverseSpec


def _base(**overrides) -> dict:
    base = dict(
        name="dyn",
        market="indian_stocks",
        timeframe="5m",
        objective="intraday",
        direction="long_only",
        entry_condition="CLOSE > VWAP",
        stop_loss={"type": "percent", "value": 1.0, "source": "user"},
        take_profit={"type": "percent", "value": 2.0, "source": "user"},
    )
    base.update(overrides)
    return base


def _universe(**overrides) -> dict:
    u = dict(
        source={"kind": "index", "name": "NIFTY500"},
        rank={"by": "rvol", "order": "desc"},
        take=10,
        max_positions=5,
        screen=["CLOSE > VWAP"],
    )
    u.update(overrides)
    return u


def test_static_path_still_valid_and_unchanged():
    s = StrategySpec(**_base(symbol="TCS.NS"))
    assert s.is_dynamic is False
    assert s.to_engine_yaml_dict()["symbol"] == "TCS.NS"


def test_dynamic_path_one_of_accepts_universe_only():
    s = StrategySpec(**_base(universe=_universe()))
    assert s.is_dynamic is True
    assert s.symbol is None


def test_one_of_rejects_both_symbol_and_universe():
    with pytest.raises(Exception):
        StrategySpec(**_base(symbol="TCS", universe=_universe()))


def test_one_of_rejects_neither():
    with pytest.raises(Exception):
        StrategySpec(**_base())


def test_template_render_stamps_member_symbol():
    s = StrategySpec(**_base(universe=_universe()))
    body = s.to_engine_template_dict("RELIANCE.NS")
    assert body["symbol"] == "RELIANCE.NS"
    # The rest of the body is symbol-agnostic: a different member differs ONLY in symbol.
    other = s.to_engine_template_dict("INFY.NS")
    body.pop("symbol"); other.pop("symbol")
    assert body == other


def test_to_engine_yaml_dict_guards_on_dynamic_spec():
    s = StrategySpec(**_base(universe=_universe()))
    with pytest.raises(ValueError):
        s.to_engine_yaml_dict()


def test_universe_in_llm_schema():
    schema = StrategySpec.json_schema_for_llm()
    assert "universe" in schema["properties"]


def test_effective_max_positions_honors_cap_and_take():
    u = UniverseSpec(**_universe(take=3, max_positions=10))
    # bounded by take (3) even though max_positions says 10
    assert u.effective_max_positions() == 3
    # platform cap reduces further
    assert u.effective_max_positions(cap=2) == 2
