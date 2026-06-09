"""
Tests for the narrow-branch synthesis helper (app/services/chat/direct_strategy_flow).

  * populate_builder_from_spec — pure mapping, runs everywhere.
  * synthesize happy path     — needs the engine (validator runs inside the generator).
  * synthesize failure path   — builder is left untouched, runs everywhere.
"""
from __future__ import annotations

import json

import pytest

from app.services.chat import direct_strategy_flow as flow
from app.services.strategy.builder import StrategyBuilder
from app.strategy import engine_bridge
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


def _builder() -> StrategyBuilder:
    b = StrategyBuilder()
    b.symbol = "INFY"
    b.market = "indian_stocks"
    b.timeframe = "15m"
    b.objective = "intraday"
    b.original_user_prompt = "INFY momentum intraday"
    return b


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, session_id=None, max_tokens=None, reasoning_effort=None):  # noqa: ANN001
        self.calls += 1
        return self.reply


def test_populate_builder_maps_spec_onto_builder():
    builder = _builder()
    spec = _spec()
    flow.populate_builder_from_spec(builder, spec)

    assert builder.entry_condition == spec.entry_condition
    assert builder.exit_condition == spec.exit_condition
    assert builder.direction == "long_only"
    assert builder.stop_loss == 1.5
    assert builder.take_profit == 3.0  # 1.5 × 2.0 risk_reward
    assert builder.signal_plan["signals_used"] == []
    assert builder.signal_plan["entry_condition"] == spec.entry_condition


def test_macd_params_emitted_in_engine_indicators_block():
    """Non-default MACD params must reach the engine YAML indicators block."""
    from app.strategy.spec import IndicatorSpec

    spec = _spec(
        entry_condition="MACD > MACD_SIGNAL",
        indicators=[IndicatorSpec(name="MACD", params={"fast": 5, "slow": 35, "signal": 5})],
    )
    block = spec.to_engine_yaml_dict()["indicators"]
    assert block == {"MACD": {"fast": 5, "slow": 35, "signal": 5}}


@requires_engine
def test_macd_params_actually_change_the_computed_column():
    """Engine computes a DIFFERENT MACD with custom params (no silent default)."""
    from app.planner.condition_satisfiability import _probe_df
    from app.strategy import engine_bridge
    engine_bridge._ensure_engine_on_path()
    from engine.indicators import add_all_indicators

    df = _probe_df()
    default = add_all_indicators(df.copy(), {"MACD": {}})["MACD"]
    custom = add_all_indicators(df.copy(), {"MACD": {"fast": 5, "slow": 35, "signal": 5}})["MACD"]
    # If params were ignored, these would be identical.
    assert not default.equals(custom)


def test_declared_indicator_params_reach_the_readback():
    """MACD(12,26,9) requested → params visible in extract_indicators, not '[]'."""
    from app.strategy.spec import IndicatorSpec

    builder = _builder()
    spec = _spec(
        entry_condition="MACD > MACD_SIGNAL AND PREV(MACD,1) <= PREV(MACD_SIGNAL,1)",
        indicators=[
            IndicatorSpec(name="MACD", params={"fast": 12, "slow": 26, "signal": 9},
                          purpose="momentum trigger"),
        ],
    )
    flow.apply_spec_extras_to_builder(builder, spec)

    indicators = builder.extract_indicators()
    assert indicators["MACD"] == {"fast": 12, "slow": 26, "signal": 9}


@requires_engine
async def test_synthesize_happy_path_populates_and_writes_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_FOLDER", str(tmp_path))
    builder = _builder()
    fake = _FakeLLM(json.dumps(_spec().model_dump()))

    result = await flow.synthesize(builder, llm=fake, session_id="s1", max_repairs=0)

    assert result.ok, result.error_message
    assert result.yaml_path and result.yaml_path.endswith(".yaml")
    assert builder.entry_condition == "CLOSE > EMA(20) AND RSI(14) > 60"
    assert builder.stop_loss == 1.5


def test_default_take_profit_is_not_passed_as_user_specified():
    """A hydrated platform default (TP 0.5%) must NOT reach the LLM as user input."""
    builder = _builder()
    builder.risk_execution_config = {
        "stop_loss_pct": 4.0,
        "take_profit_pct": 0.5,
        "rms_sources": {"stop_loss_pct": "user", "take_profit_pct": "default"},
    }
    msg = flow._builder_context_message(builder)
    assert "stop_loss_pct (user-specified): 4.0" in msg   # genuinely user-given
    assert "take_profit_pct (user-specified)" not in msg  # default → LLM assumes instead


async def test_synthesize_failure_leaves_builder_untouched():
    builder = _builder()
    fake = _FakeLLM("this is not json at all")

    result = await flow.synthesize(builder, llm=fake, session_id="s2", max_repairs=0)

    assert not result.ok
    assert result.error_message
    assert builder.entry_condition is None  # builder never populated on failure
