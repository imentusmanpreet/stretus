"""Strategy-preset registry and pipeline integration tests."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.kb import kb
from app.planner.intent_extractor import IntentExtractor
from app.planner.pipeline import Pipeline


class StubLLM:
    def __init__(self, payload: dict):
        self._payload = payload

    async def chat(self, messages):
        return json.dumps(self._payload)


_STUB_INTENT = {
    "hold_horizon": "minutes", "frequency": "high",
    "profit_size": "small", "style": "breakout",
    "risk_appetite": "moderate",
}


def _pipe() -> Pipeline:
    return Pipeline(intent_extractor=IntentExtractor(llm_service=StubLLM(_STUB_INTENT)))


def _builder(**overrides):
    base = {
        "symbol": "RELIANCE",
        "timeframe": "5m",
        "sentiment": "bullish",
        "experience": "intermediate",
        "objective": "intraday",
        "goal": "let me trade",
        "strategy_preset": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ── KB-level checks ──────────────────────────────────────────────────────────


def test_presets_load_and_signal_refs_resolve():
    assert len(kb.presets) >= 8
    for name, preset in kb.presets.items():
        assert preset.name == name
        for leg_name in ("bullish", "bearish"):
            leg = getattr(preset, leg_name)
            if leg is None:
                continue
            for sig in leg.all_signal_names():
                assert sig in kb.signals, f"{name}/{leg_name}: missing signal {sig}"


def test_lookup_preset_by_name_and_keyword():
    assert kb.lookup_preset("orb") is not None
    assert kb.lookup_preset("opening range breakout") is not None
    assert kb.lookup_preset("ema_pullback") is not None
    assert kb.lookup_preset("20 ema pullback") is not None
    assert kb.lookup_preset("nope-not-real") is None


def test_keyword_detection_in_text_picks_longest_match():
    """When a sentence contains multiple keywords, the longest wins so that
    'opening range breakout' beats the bare 'orb'."""
    preset = kb.detect_preset_in_text("Let's run an opening range breakout setup today")
    assert preset is not None
    assert preset.name == "orb"


def test_keyword_detection_returns_none_when_no_match():
    assert kb.detect_preset_in_text("just a vanilla mean-revision request") is None
    assert kb.detect_preset_in_text("") is None


# ── Pipeline integration ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_pins_orb_preset_signals_directly():
    plan = await _pipe().plan(_builder(strategy_preset="orb"))
    assert plan.entry_trigger.name == "opening_range_breakout"
    assert {f.name for f in plan.entry_filters} == {"vwap_bullish", "volume_spike"}
    assert plan.exit_trigger.name == "vwap_bearish"


@pytest.mark.asyncio
async def test_pipeline_picks_bearish_leg_when_sentiment_is_bearish():
    plan = await _pipe().plan(_builder(strategy_preset="orb", sentiment="bearish"))
    assert plan.entry_trigger.name == "opening_range_breakdown"
    assert {f.name for f in plan.entry_filters} == {"vwap_bearish", "volume_spike"}
    assert plan.exit_trigger.name == "vwap_bullish"


@pytest.mark.asyncio
async def test_pipeline_resolves_preset_from_goal_keyword():
    """The user types a goal mentioning 'VWAP reversal' — pipeline picks that
    preset without an explicit strategy_preset field."""
    plan = await _pipe().plan(_builder(
        strategy_preset=None,
        goal="I want a VWAP reversal trade with reclaim entry",
    ))
    assert plan.entry_trigger.name == "vwap_reclaim_bullish"


@pytest.mark.asyncio
async def test_unknown_preset_name_falls_back_to_ranker():
    """An explicit preset that doesn't exist should not crash — the planner
    just falls back to its normal ranker pick."""
    plan = await _pipe().plan(_builder(strategy_preset="totally-made-up"))
    assert plan.entry_trigger is not None
    assert plan.exit_trigger is not None


# ── Phase 8a: volume_breakout_52w preset ─────────────────────────────────────


@pytest.mark.asyncio
async def test_volume_breakout_52w_bullish_leg_pins_full_signal_stack():
    """Verifies the entire user-requested strategy lands intact: volume spike
    + VWAP + RSI>60 filters on top of an EMA pullback trigger, with the daily
    52-week-high HTF gate and an ATR(14)*1.5 stop + 0.5% trailing."""
    plan = await _pipe().plan(_builder(
        strategy_preset="volume_breakout_52w",
        timeframe="5m",
        sentiment="bullish",
    ))

    assert plan.entry_trigger.name == "ema_pullback_bullish"
    filter_names = {f.name for f in plan.entry_filters}
    assert {"volume_spike", "vwap_bullish", "rsi_above_60"} <= filter_names

    # Phase 5 — HTF rules came from the bullish leg
    assert len(plan.htf_rules) == 1
    assert plan.htf_rules[0]["timeframe"] == "1d"
    assert "MAX(HIGH, 252)" in plan.htf_rules[0]["condition"]

    # Phase 3 — ATR stop + percent trailing rode along on the preset
    assert plan.stop_loss_spec == {"type": "atr", "multiplier": 1.5, "window": 14}
    assert plan.trailing_stop_spec["type"] == "percent"
    assert plan.trailing_stop_spec["distance_pct"] == 0.5
    assert plan.trailing_stop_spec["activate_after_pct"] == 1.0


@pytest.mark.asyncio
async def test_volume_breakout_52w_bearish_leg_uses_inverse_signals_and_52w_low():
    plan = await _pipe().plan(_builder(
        strategy_preset="volume_breakout_52w",
        timeframe="5m",
        sentiment="bearish",
    ))

    assert plan.entry_trigger.name == "ema_pullback_bearish"
    filter_names = {f.name for f in plan.entry_filters}
    assert {"volume_spike", "vwap_bearish", "rsi_below_40"} <= filter_names

    assert plan.htf_rules[0]["timeframe"] == "1d"
    assert "MIN(LOW, 252)" in plan.htf_rules[0]["condition"]


@pytest.mark.asyncio
async def test_volume_breakout_52w_keyword_pulls_in_preset_from_goal_text():
    """The chat layer detects keywords in the goal text and pins the preset
    without an explicit strategy_preset field — verifies user can type
    "volume spike breakout near 52 week high" and get this strategy."""
    plan = await _pipe().plan(_builder(
        strategy_preset=None,
        goal="give me a volume breakout setup on 5m with relative volume confirmation",
    ))
    assert plan.entry_trigger.name == "ema_pullback_bullish"
    filter_names = {f.name for f in plan.entry_filters}
    assert "volume_spike" in filter_names


@pytest.mark.asyncio
async def test_volume_breakout_52w_preset_carries_time_exit_to_plan():
    """Phase 8b: the volume_breakout_52w preset declares time_exit 15:15 IST.
    It must ride into the StrategyPlan so the builder writes it to the YAML
    and the simulator picks it up."""
    plan = await _pipe().plan(_builder(
        strategy_preset="volume_breakout_52w",
        timeframe="5m",
    ))
    assert plan.time_exit is not None
    assert plan.time_exit["exit_time"] == "15:15"
    assert plan.time_exit["timezone"] == "Asia/Kolkata"


@pytest.mark.asyncio
async def test_preset_without_time_exit_leaves_plan_field_none():
    """Sanity check that legacy presets stay clean."""
    plan = await _pipe().plan(_builder(strategy_preset="orb"))
    assert plan.time_exit is None


@pytest.mark.asyncio
async def test_preset_with_structural_sl_propagates_specs_to_plan():
    """A preset that declares stop_loss / trailing_stop blocks should attach
    them to the StrategyPlan so they end up in the generated YAML."""
    plan = await _pipe().plan(_builder(strategy_preset="orb_structural"))
    assert plan.stop_loss_spec is not None
    assert plan.stop_loss_spec["type"] == "structural"
    assert plan.stop_loss_spec["anchor"] == "opening_range_low"
    assert plan.trailing_stop_spec is not None
    assert plan.trailing_stop_spec["type"] == "percent"


@pytest.mark.asyncio
async def test_preset_without_specs_leaves_them_none():
    """The vanilla ORB preset doesn't declare SL specs — plan stays clean."""
    plan = await _pipe().plan(_builder(strategy_preset="orb"))
    assert plan.stop_loss_spec is None
    assert plan.trailing_stop_spec is None


@pytest.mark.asyncio
async def test_smart_money_preset_pins_bos_and_fvg_signals():
    """Phase 6: smart-money preset wires BOS as the trigger and FVG +
    uptrend_structure as filters — all of which use the new IS_* identifiers."""
    plan = await _pipe().plan(_builder(
        strategy_preset="smart_money",
        timeframe="15m",
    ))
    assert plan.entry_trigger.name == "bullish_market_structure_break"
    filter_names = {f.name for f in plan.entry_filters}
    assert "bullish_fvg_signal" in filter_names
    assert "uptrend_structure" in filter_names


@pytest.mark.asyncio
async def test_smart_money_preset_bearish_leg_uses_inverse_signals():
    plan = await _pipe().plan(_builder(
        strategy_preset="smart_money",
        timeframe="15m",
        sentiment="bearish",
    ))
    assert plan.entry_trigger.name == "bearish_market_structure_break"
    filter_names = {f.name for f in plan.entry_filters}
    assert "bearish_fvg_signal" in filter_names
    assert "downtrend_structure" in filter_names


@pytest.mark.asyncio
async def test_preset_with_htf_propagates_leg_specific_rules():
    """Phase 5: HTF rules live on the leg, not the preset, because they are
    direction-dependent. Bullish leg should receive the bullish HTF list."""
    plan = await _pipe().plan(_builder(
        strategy_preset="mtf_confluence",
        timeframe="15m",
        sentiment="bullish",
    ))
    assert len(plan.htf_rules) == 2
    daily_rule = next(r for r in plan.htf_rules if r["timeframe"] == "1d")
    assert ">" in daily_rule["condition"]


@pytest.mark.asyncio
async def test_preset_with_htf_bearish_leg_uses_inverse_conditions():
    plan = await _pipe().plan(_builder(
        strategy_preset="mtf_confluence",
        timeframe="15m",
        sentiment="bearish",
    ))
    assert len(plan.htf_rules) == 2
    daily_rule = next(r for r in plan.htf_rules if r["timeframe"] == "1d")
    assert "<" in daily_rule["condition"]


@pytest.mark.asyncio
async def test_preset_without_htf_leaves_htf_rules_empty():
    plan = await _pipe().plan(_builder(strategy_preset="orb"))
    assert plan.htf_rules == []


@pytest.mark.asyncio
async def test_preset_with_reference_symbol_propagates_to_plan():
    """Phase 4: a preset that declares reference_symbol must surface it on
    the StrategyPlan so the API layer knows to fetch the benchmark series."""
    plan = await _pipe().plan(_builder(
        strategy_preset="relative_strength",
        timeframe="1d",
    ))
    assert plan.reference_symbol == "^NSEI"


@pytest.mark.asyncio
async def test_preset_without_reference_symbol_keeps_it_none():
    plan = await _pipe().plan(_builder(strategy_preset="orb"))
    assert plan.reference_symbol is None


@pytest.mark.asyncio
async def test_preset_rejects_unsupported_timeframe():
    """The opening_drive preset is intraday-only; asking for it on 1d must
    raise UnsupportedTimeframe rather than silently picking something else."""
    from app.planner.pipeline import UnsupportedTimeframe
    with pytest.raises(UnsupportedTimeframe):
        await _pipe().plan(_builder(strategy_preset="opening_drive", timeframe="1d"))
