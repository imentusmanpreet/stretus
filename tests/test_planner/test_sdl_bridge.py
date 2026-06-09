"""
tests/test_planner/test_sdl_bridge.py — SDL bridge + flow tests.

Covers:
  - is_strategy_description() detection heuristic
  - artifact_to_signal_plan() produces a valid legacy shape
  - populate_builder_from_artifact() writes the right builder fields
  - try_sdl_plan() uses_sdl=True on strategy descriptions with mocked LLM
  - try_sdl_plan() uses_sdl=False for non-strategy messages
  - try_sdl_plan() never raises — degrades gracefully
  - sdl_readback_text is populated and non-empty
  - validation failures surface in result.validation_errors
"""
import asyncio
import pytest

from app.planner.sdl import (
    SDL,
    EntrySpec,
    ExitSpec,
    GatesSpec,
    HTFRule,
    Leg,
    Provenance,
    RegimeGate,
    RiskSpec,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    VolatilityGate,
)
from app.planner.compiler import StrategyArtifact, compile_sdl
from app.planner.sdl_bridge import (
    is_strategy_description,
    artifact_to_signal_plan,
    populate_builder_from_artifact,
)
from app.planner.sdl_selector import clear_cache, set_llm_override
from app.planner.catalog_schema import invalidate_menu_cache


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def reset():
    clear_cache()
    set_llm_override(None)
    invalidate_menu_cache()
    yield
    clear_cache()
    set_llm_override(None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _eth_sdl() -> SDL:
    return SDL(
        context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
        universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(
                    trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})
                ),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=2.0),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        provenance=Provenance(field_sources={
            "universe.symbol": "user",
            "legs.0.entry.trigger": "user",
            "risk.stop_loss": "user",
            "risk.take_profit": "user",
        }),
    )


def _eth_artifact() -> StrategyArtifact:
    return compile_sdl(_eth_sdl())


class _MockBuilder:
    """Minimal StrategyBuilder-like object for testing populate_builder_from_artifact."""
    def __init__(self):
        self.entry_condition = None
        self.exit_condition = None
        self.stop_loss = None
        self.take_profit = None
        self.stop_loss_spec = None
        self.trailing_stop_spec = None
        self.htf_rules = None
        self.reference_symbol = None
        self.regime_filter_allowed = None
        self.entry_window_start_utc = None
        self.entry_window_end_utc = None
        self.volume_ratio_threshold = None
        self.vol_filter_metric = None
        self.vol_filter_window = None
        self.vol_filter_min = None
        self.vol_filter_max = None
        self._sdl = None
        self._sdl_artifact = None
        self._sdl_readback = None


# ── is_strategy_description tests ─────────────────────────────────────────────

class TestIsStrategyDescription:
    def test_full_strategy_is_true(self):
        assert is_strategy_description(
            "Buy ETH on 15m when RSI drops below 30. Stop loss 2%, take profit 2:1."
        )

    def test_structured_format_is_true(self):
        assert is_strategy_description(
            "Long Entry: Price breaks above range high. Stop Loss: 1.5%. "
            "Take Profit: 2:1 RR. Timeframe: 15m."
        )

    def test_indicator_plus_risk_is_true(self):
        assert is_strategy_description(
            "Enter when MACD crosses above signal line, stop loss 1.5%"
        )

    def test_timeframe_plus_indicator_is_true(self):
        assert is_strategy_description(
            "Trade HDFCBANK on 15m chart with RSI oversold signal"
        )

    def test_greeting_is_false(self):
        assert not is_strategy_description("hi")

    def test_question_is_false(self):
        assert not is_strategy_description("What is the best timeframe for scalping?")

    def test_short_message_is_false(self):
        assert not is_strategy_description("yes intraday")

    def test_wizard_answer_is_false(self):
        assert not is_strategy_description("RELIANCE, 15m, intraday, bullish, intermediate")

    def test_one_pillar_only_is_false(self):
        # Only indicator mentioned — not enough
        assert not is_strategy_description("I want to use RSI for my trades")

    def test_ema_plus_stop_is_true(self):
        assert is_strategy_description(
            "Go long when EMA 9 crosses above EMA 21, stop at 1%"
        )

    def test_colon_structured_triggers_true(self):
        assert is_strategy_description(
            "Entry: RSI below 30. Exit: RSI above 70. Stop Loss: ATR × 1.5."
        )

    def test_direction_plus_indicator_is_true(self):
        assert is_strategy_description(
            "Short HDFCBANK when MACD shows bearish crossover on 15m chart"
        )


# ── artifact_to_signal_plan tests ─────────────────────────────────────────────

class TestArtifactToSignalPlan:
    def test_returns_dict(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert isinstance(plan, dict)

    def test_entry_signals_present(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert "entry" in plan
        assert len(plan["entry"]) >= 1
        assert plan["entry"][0]["name"] == "rsi_oversold"

    def test_exit_signals_present(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert "exit" in plan
        assert len(plan["exit"]) >= 1
        assert plan["exit"][0]["name"] == "rsi_overbought"

    def test_conditions_populated(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert plan.get("entry_condition")
        assert plan.get("exit_condition")
        assert "RSI" in plan["entry_condition"]

    def test_sl_tp_populated(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert plan["_sl_pct"] == pytest.approx(2.0)
        assert plan["_tp_pct"] == pytest.approx(4.0)  # 2.0 × 2.0

    def test_signals_used_nonempty(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert plan["signals_used"]
        assert "rsi_oversold" in plan["signals_used"]

    def test_signals_available_set(self):
        art = _eth_artifact()
        plan = artifact_to_signal_plan(art, _eth_sdl())
        assert plan["signals_available"] > 0

    def test_gate_fields_when_gates_set(self):
        sdl = _eth_sdl()
        # Patch gates onto the SDL
        sdl_data = sdl.model_dump(mode="python")
        sdl_data.pop("content_hash", None)
        sdl_data["gates"] = {"regime": {"allowed": ["trending"]}, "volatility": None,
                              "event": None, "session": None, "relative_strength": None, "volume_ratio": None}
        from app.planner.sdl import SDL as _SDL
        sdl_with_gates = _SDL(**sdl_data)
        art = compile_sdl(sdl_with_gates)
        plan = artifact_to_signal_plan(art, sdl_with_gates)
        assert plan["_regime_filter_allowed"] == ["trending"]

    def test_discovery_config_for_dynamic(self):
        from app.planner.sdl import DynamicRank, DynamicUniverse
        sdl = SDL(
            context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
            universe=DynamicUniverse(
                asset_class="equity_cash",
                screen=["CLOSE > VWAP"],
                rank=DynamicRank(by="rvol", order="desc"),
                tie_break="highest_relative_volume",
            ),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="ema_cross_up", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down", params={})]),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=1.5),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(field_sources={}),
        )
        art = compile_sdl(sdl)
        plan = artifact_to_signal_plan(art, sdl)
        assert plan["_discovery_config"] is not None
        assert plan["_discovery_config"]["type"] == "dynamic"


# ── populate_builder_from_artifact tests ──────────────────────────────────────

class TestPopulateBuilderFromArtifact:
    def test_entry_condition_set(self):
        builder = _MockBuilder()
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        populate_builder_from_artifact(builder, art, sdl, "readback")
        assert builder.entry_condition
        assert "RSI" in builder.entry_condition

    def test_stop_loss_set(self):
        builder = _MockBuilder()
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        populate_builder_from_artifact(builder, art, sdl, "readback")
        assert builder.stop_loss == pytest.approx(2.0)

    def test_take_profit_set(self):
        builder = _MockBuilder()
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        populate_builder_from_artifact(builder, art, sdl, "readback")
        assert builder.take_profit == pytest.approx(4.0)

    def test_sdl_stored_on_builder(self):
        builder = _MockBuilder()
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        populate_builder_from_artifact(builder, art, sdl, "my readback")
        assert builder._sdl is sdl
        assert builder._sdl_artifact is art
        assert builder._sdl_readback == "my readback"

    def test_stop_loss_spec_set(self):
        builder = _MockBuilder()
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        populate_builder_from_artifact(builder, art, sdl, "")
        assert builder.stop_loss_spec is not None
        assert builder.stop_loss_spec["type"] == "percent"

    def test_regime_gate_propagated(self):
        sdl = _eth_sdl()
        sdl_data = sdl.model_dump(mode="python")
        sdl_data.pop("content_hash", None)
        sdl_data["gates"] = {"regime": {"allowed": ["trending_up"]}, "volatility": None,
                              "event": None, "session": None, "relative_strength": None, "volume_ratio": None}
        from app.planner.sdl import SDL as _SDL
        sdl2 = _SDL(**sdl_data)
        art = compile_sdl(sdl2)
        builder = _MockBuilder()
        populate_builder_from_artifact(builder, art, sdl2, "")
        assert builder.regime_filter_allowed == ["trending_up"]


# ── try_sdl_plan integration tests ────────────────────────────────────────────

class TestTrySDLPlan:
    """Integration: try_sdl_plan() with mocked LLM."""

    def _golden_json(self):
        sdl = _eth_sdl()
        return sdl.model_dump_json()

    def test_strategy_description_uses_sdl(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan(
            "Buy ETH on 15m when RSI drops below 30. Stop loss 2%, TP 2:1.",
            builder,
        ))
        assert result.used_sdl is True

    def test_readback_nonempty(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan(
            "Buy ETH on 15m when RSI < 30. SL 2%, TP 2:1.",
            builder,
        ))
        assert result.readback_text

    def test_match_pct_positive(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan(
            "Buy ETH on 15m when RSI < 30. Stop loss 2%, take profit 2:1.",
            builder,
        ))
        assert result.match_pct > 0

    def test_signal_plan_has_entry_condition(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan(
            "Buy ETH on 15m when RSI drops below 30. Stop 2%, TP 2:1.",
            builder,
        ))
        assert result.signal_plan.get("entry_condition")

    def test_non_strategy_skips_sdl(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan("hello there", builder))
        assert result.used_sdl is False
        assert "not_strategy_description" in result.skip_reason

    def test_wizard_answer_skips_sdl(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan("yes intraday", builder))
        assert result.used_sdl is False

    def test_llm_error_degrades_gracefully(self):
        def bad_llm(msgs, tools):
            raise RuntimeError("LLM unavailable")
        set_llm_override(bad_llm)
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan(
            "Buy ETH on 15m when RSI < 30. Stop 2%, TP 2:1.",
            builder,
        ))
        # Must not raise; falls back
        assert result.used_sdl is False
        assert result.skip_reason

    def test_builder_populated_on_success(self):
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        run(try_sdl_plan(
            "Buy ETH on 15m when RSI drops below 30. Stop loss 2%, take profit 2:1.",
            builder,
        ))
        assert builder._sdl_artifact is not None
        assert builder.entry_condition
        assert builder.stop_loss == pytest.approx(2.0)

    def test_force_flag_bypasses_detection(self):
        """force=True runs SDL even for short/non-strategy messages."""
        set_llm_override(lambda msgs, tools: self._golden_json())
        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan("yes", builder, force=True))
        # With mocked LLM returning valid SDL, it should succeed
        assert result.used_sdl is True

    def test_validation_failure_sets_validation_ok_false(self):
        """When SDL has an unknown signal, validation fails, used_sdl=True but validation_ok=False."""
        bad_sdl = _eth_sdl()
        bad_data = bad_sdl.model_dump(mode="python")
        bad_data.pop("content_hash", None)
        # Plant unknown signal name — referential validation will fail
        bad_data["legs"][0]["entry"]["trigger"]["name"] = "TOTALLY_FAKE_SIGNAL_XYZ"
        from app.planner.sdl import SDL as _SDL
        bad_sdl_obj = _SDL(**bad_data)
        set_llm_override(lambda msgs, tools: bad_sdl_obj.model_dump_json())

        from app.planner.sdl_flow import try_sdl_plan
        builder = _MockBuilder()
        result = run(try_sdl_plan(
            "Buy ETH on 15m when RSI < 30. Stop 2%, TP 2:1.",
            builder,
        ))
        # used_sdl=True (we did run the SDL path), but validation failed
        assert result.used_sdl is True
        assert result.validation_ok is False
        assert len(result.validation_errors) > 0
