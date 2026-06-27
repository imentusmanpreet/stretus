"""Phase 2 — trailing take-profit through the chat → spec → builder → YAML spine.

Covers:
  * the StrategySpec model rendering `trailing_take_profit` into the engine dict,
  * the StrategyBuilder persisting it (draft round-trip, signal-plan hoist, YAML),
  * the validator (engine-legal type + both trailing lines may coexist, with a
    non-blocking warning when one line is dominated and can never fire),
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


def test_percent_trailing_take_profit_requires_distance_pct():
    # distance_pct is the give-back amount itself — a percent block with only
    # activate_after_pct is undefined and must be rejected at the contract
    # (otherwise the engine loader crashes the backtest later).
    from pydantic import ValidationError as PydErr
    with pytest.raises(PydErr, match="distance_pct"):
        TrailingTakeProfit(type="percent", activate_after_pct=0.2)


def test_percent_trailing_stop_requires_distance_pct():
    from pydantic import ValidationError as PydErr
    with pytest.raises(PydErr, match="distance_pct"):
        TrailingStop(type="percent", activate_after_pct=0.2)


def test_full_spec_rejects_percent_trailing_stop_without_distance():
    # Mirrors the real chat path: the LLM emits a trailing_stop missing distance_pct;
    # StrategySpec.model_validate must reject it so the generator repair loop fixes it.
    from pydantic import ValidationError as PydErr
    with pytest.raises(PydErr, match="distance_pct"):
        _spec(trailing_stop=TrailingStop(type="percent", activate_after_pct=0.2))


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


def test_builder_drops_distance_less_percent_trailing_stop_from_yaml():
    # Several extraction paths can produce a percent trailing block carrying only an
    # activation and no distance (the ETH/USDT crash). The YAML boundary must DROP it
    # rather than emit it and crash the engine loader — the backtest runs without it.
    b = StrategyBuilder()
    _populate_minimum(b)
    b.trailing_stop_spec = {"type": "percent", "activate_after_pct": 0.2, "source": "user"}
    strat = b.to_yaml_dict()["strategy"]
    assert "trailing_stop" not in strat            # malformed leg dropped, no crash


def test_builder_keeps_valid_percent_trailing_stop():
    b = StrategyBuilder()
    _populate_minimum(b)
    b.trailing_stop_spec = {"type": "percent", "distance_pct": 0.2, "activate_after_pct": 0.2}
    strat = b.to_yaml_dict()["strategy"]
    assert strat["trailing_stop"]["distance_pct"] == 0.2


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
def test_trailing_stop_alone_is_a_valid_profit_exit():
    # "Ride until stopped out": a trailing stop ratchets above entry and books the
    # gain on a reversal, so it is a complete exit on its own — no static target or
    # trailing take-profit required. (Mirrors the user prompt "once up 2%, trail my
    # stop by 0.5%", which must map to trailing_stop, not trailing_take_profit.)
    result = validator.validate_spec(
        _spec(
            take_profit=None,
            trailing_stop=TrailingStop(type="percent", distance_pct=0.5, activate_after_pct=2.0),
        )
    )
    assert result.ok, result.as_repair_text()


@requires_engine
def test_trailing_stop_and_take_profit_both_allowed_staged():
    # Both trailing exits may coexist (a staged trail). Here the stop trails wide
    # (2%, immediate) and the take-profit trails tight (1%) but only after +8% — so
    # neither dominates the other and there is NO warning.
    result = validator.validate_spec(
        _spec(
            trailing_stop=TrailingStop(type="percent", distance_pct=2.0),
            trailing_take_profit=TrailingTakeProfit(distance_pct=1.0, activate_after_pct=8.0),
        )
    )
    assert result.ok, result.as_repair_text()
    assert not any(n.code == "dominated_trailing_line" for n in result.notes)


@requires_engine
def test_dominated_trailing_line_warns_but_passes():
    # The NAUKRI/Reliance case: a 1% trailing stop after +3% always fires before a
    # 2% trailing take-profit after +8% — the TP can never trigger. Allowed, but the
    # user is warned (non-blocking).
    result = validator.validate_spec(
        _spec(
            trailing_stop=TrailingStop(type="percent", distance_pct=1.0, activate_after_pct=3.0),
            trailing_take_profit=TrailingTakeProfit(distance_pct=2.0, activate_after_pct=8.0),
        )
    )
    assert result.ok, result.as_repair_text()              # not blocked
    warnings = [n for n in result.notes if n.code == "dominated_trailing_line"]
    assert len(warnings) == 1
    assert warnings[0].field == "trailing_take_profit"     # the TP is the dead line


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
