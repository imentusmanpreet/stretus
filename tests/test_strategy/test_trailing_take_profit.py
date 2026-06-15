"""Phase 2 — trailing take-profit through the chat → spec → builder → YAML spine.

Covers:
  * the StrategySpec model rendering `trailing_take_profit` into the engine dict,
  * the StrategyBuilder persisting it (draft round-trip, signal-plan hoist, YAML),
  * the validator (engine-legal type + the trailing-stop/take-profit conflict),
  * and an end-to-end check that the emitted YAML is parsed by the engine loader
    into `trailing_take_profit_spec` (ties Phase 2 back to the Phase-1 backtest).
"""
from __future__ import annotations

import pytest
import yaml

from app.services.strategy.builder import StrategyBuilder
from app.strategy import engine_bridge, validator
from app.strategy.spec import (
    StopLoss,
    StrategySpec,
    TakeProfit,
    TrailingStop,
    TrailingTakeProfit,
)

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


def _populate_minimum(b: StrategyBuilder) -> None:
    b.symbol = "RELIANCE.NS"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "trailing take-profit test"
    b.stop_loss = 1.5
    b.take_profit = 3.0


# ── Spec model → engine dict ─────────────────────────────────────────────────


def test_spec_renders_trailing_take_profit_block():
    s = _spec(trailing_take_profit=TrailingTakeProfit(distance_pct=2.0, activate_after_pct=5.0))
    d = s.to_engine_yaml_dict()
    assert d["trailing_take_profit"] == {
        "type": "percent",
        "distance_pct": 2.0,
        "activate_after_pct": 5.0,
    }


def test_spec_omits_block_when_unset():
    assert "trailing_take_profit" not in _spec().to_engine_yaml_dict()


def test_spec_to_engine_dict_drops_none_fields():
    # activate_after_pct omitted → not rendered (immediate trailing).
    tp = TrailingTakeProfit(distance_pct=1.5)
    assert tp.to_engine_dict() == {"type": "percent", "distance_pct": 1.5}


def test_spec_forbids_extra_fields():
    from pydantic import ValidationError as PydErr
    with pytest.raises(PydErr):
        TrailingTakeProfit(distance_pct=1.0, bogus=True)


# ── Builder persistence ──────────────────────────────────────────────────────


def test_builder_writes_trailing_take_profit_to_yaml():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.trailing_take_profit_spec = {"type": "percent", "distance_pct": 2.0, "activate_after_pct": 3.0}

    strat = b.to_yaml_dict()["strategy"]
    assert strat["trailing_take_profit"]["type"] == "percent"
    assert strat["trailing_take_profit"]["distance_pct"] == 2.0
    assert strat["trailing_take_profit"]["activate_after_pct"] == 3.0


def test_builder_omits_block_when_unset():
    b = StrategyBuilder()
    _populate_minimum(b)
    assert "trailing_take_profit" not in b.to_yaml_dict()["strategy"]


def test_apply_signal_plan_hoists_underscore_key():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.apply_signal_plan({
        "entry": [], "exit": [],
        "_trailing_take_profit_spec": {"type": "percent", "distance_pct": 1.5},
    })
    assert b.trailing_take_profit_spec == {"type": "percent", "distance_pct": 1.5}
    assert b.to_yaml_dict()["strategy"]["trailing_take_profit"]["distance_pct"] == 1.5


def test_draft_round_trip_preserves_spec():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.trailing_take_profit_spec = {"type": "percent", "distance_pct": 2.5, "activate_after_pct": 4.0}

    draft = b.to_draft_json()
    assert draft["trailing_take_profit_spec"] == b.trailing_take_profit_spec

    restored = StrategyBuilder()
    restored.merge_preview(draft)
    assert restored.trailing_take_profit_spec == b.trailing_take_profit_spec


# ── Validator (engine-backed) ────────────────────────────────────────────────


@requires_engine
def test_valid_trailing_take_profit_passes():
    result = validator.validate_spec(
        _spec(trailing_take_profit=TrailingTakeProfit(distance_pct=2.0, activate_after_pct=5.0))
    )
    assert result.ok, result.as_repair_text()


@requires_engine
def test_non_engine_legal_type_is_rejected():
    result = validator.validate_spec(
        _spec(trailing_take_profit=TrailingTakeProfit(type="atr", distance_pct=2.0))
    )
    assert any(e.code == "unsupported_trailing_take_profit_type" for e in result.errors)


@requires_engine
def test_take_profit_optional_with_trailing_passes():
    # No static take_profit, trailing supplies the exit — valid.
    result = validator.validate_spec(
        _spec(take_profit=None, trailing_take_profit=TrailingTakeProfit(distance_pct=0.5, activate_after_pct=1.0))
    )
    assert result.ok, result.as_repair_text()


@requires_engine
def test_no_profit_exit_at_all_is_rejected():
    # Neither a static target nor a trailing one → a strategy with no profit exit.
    result = validator.validate_spec(_spec(take_profit=None))
    assert any(e.code == "missing_profit_exit" for e in result.errors)


@requires_engine
def test_trailing_stop_and_take_profit_conflict_is_rejected():
    result = validator.validate_spec(
        _spec(
            trailing_stop=TrailingStop(type="percent", distance_pct=1.0),
            trailing_take_profit=TrailingTakeProfit(distance_pct=2.0),
        )
    )
    assert any(e.code == "trailing_stop_and_take_profit_conflict" for e in result.errors)


# ── End-to-end: emitted YAML is parsed by the engine loader ──────────────────


@requires_engine
def test_emitted_block_is_parsed_by_engine_loader():
    """The exact block the spec/builder emits must round-trip into the engine
    loader's `trailing_take_profit_spec` — proving Phase 2 feeds Phase 1. We inject
    it into the minimal loader-legal YAML shape so the check isolates key
    compatibility (not unrelated required fields)."""
    from engine.loader import load_strategy_from_content

    tp_block = TrailingTakeProfit(distance_pct=2.0, activate_after_pct=5.0).to_engine_dict()
    strat = {
        "name": "t",
        "symbol": "INFY.NS",
        "market": "indian_stocks",
        "timeframe": "5m",
        "entry": {"condition": "CLOSE > EMA(20)"},
        "risk_management": {"stop_loss_percent": 1.5, "take_profit_percent": 3.0},
        "trailing_take_profit": tp_block,
    }
    cfg = load_strategy_from_content(yaml.safe_dump({"strategy": strat}))
    assert cfg.trailing_take_profit_spec == {
        "type": "percent",
        "distance_pct": 2.0,
        "activate_after_pct": 5.0,
    }
