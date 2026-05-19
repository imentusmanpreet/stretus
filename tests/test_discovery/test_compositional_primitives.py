"""Phase 9k — compositional discovery primitives.

The user complained: "Why is it talking about 52 weeks high and low
when I only mentioned volume spike?". Pre-9k the scanner ALWAYS ran
the preset's full hardcoded condition list (volume + pullback OR
52w-high OR 52w-low), even when the user only typed one of those.

Phase 9k flips the model: the chat layer parses the user's prose
into an ordered list of primitive `{name, params}` dicts; the
orchestrator runs the scanner with EXACTLY those constraints.

These tests pin:
  1. Each primitive's render produces a syntactically-correct AST
     condition string and accepts user-supplied params.
  2. The chat extractor (_extract_discovery_conditions) recognises
     the user's 5 example prompts and produces the right primitive
     list.
  3. When discovery_conditions is non-empty, the orchestrator IGNORES
     the preset's hardcoded conditions and uses the parsed list.
  4. When discovery_conditions is empty, the orchestrator falls back
     to the preset's defaults (backward compat with non-prose prompts).
  5. The no-match reply surfaces the primitives that actually ran.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")


# ── Primitive library ──────────────────────────────────────────────────────


from app.services.discovery.primitives import (
    PRIMITIVES,
    Primitive,
    primitive_descriptions,
    render_conditions,
    render_primitive,
)


def test_primitive_library_has_expected_keys():
    """Spot-check the primitive set so renames break loudly."""
    for key in [
        "volume_spike", "volume_above_avg",
        "near_52w_high", "near_52w_low",
        "above_52w_high", "below_52w_low",
        "shallow_pullback_long", "shallow_pullback_short",
        "rsi_above", "rsi_below",
        "above_vwap", "below_vwap",
        "above_ema", "below_ema",
        "near_day_high", "near_day_low",
    ]:
        assert key in PRIMITIVES, f"primitive missing: {key!r}"


def test_volume_spike_renders_with_user_multiplier():
    out = render_primitive({"name": "volume_spike", "params": {"multiplier": 1.2}})
    assert out == "VOL > AVG(VOL, 20) * 1.2"


def test_volume_spike_uses_default_multiplier_when_unspecified():
    out = render_primitive({"name": "volume_spike"})
    assert out == "VOL > AVG(VOL, 20) * 2"


def test_near_52w_high_renders_with_user_window_and_factor():
    out = render_primitive({
        "name": "near_52w_high",
        "params": {"window": 100, "factor": 0.95},
    })
    assert out == "CLOSE >= MAX(HIGH, 100) * 0.95"


def test_near_52w_low_renders_with_user_window_and_factor():
    out = render_primitive({
        "name": "near_52w_low",
        "params": {"window": 252, "factor": 1.03},
    })
    assert out == "CLOSE <= MIN(LOW, 252) * 1.03"


def test_above_52w_high_renders_breakout_condition():
    """For "breaking above 52-week high" — strict inequality, no factor."""
    out = render_primitive({"name": "above_52w_high"})
    assert out == "CLOSE > MAX(HIGH, 252)"


def test_shallow_pullback_long_renders_default_ema_and_window():
    out = render_primitive({"name": "shallow_pullback_long"})
    assert out == "MIN(LOW, 3) <= EMA(20) AND CLOSE > EMA(20)"


def test_rsi_above_renders_with_threshold():
    out = render_primitive({"name": "rsi_above", "params": {"threshold": 65}})
    assert out == "RSI(14) > 65"


def test_above_vwap_is_a_simple_comparison():
    assert render_primitive({"name": "above_vwap"}) == "CLOSE > VWAP"


def test_above_ema_renders_with_user_period():
    out = render_primitive({"name": "above_ema", "params": {"period": 50}})
    assert out == "CLOSE > EMA(50)"


def test_render_primitive_raises_on_unknown_name():
    with pytest.raises(ValueError) as exc:
        render_primitive({"name": "this_does_not_exist"})
    assert "this_does_not_exist" in str(exc.value)


def test_render_conditions_handles_a_full_user_prompt_combination():
    """Example 1 from the user's brief: volume 2x AND near 52w high
    AND shallow pullback. The renderer produces three concrete AST
    conditions the scanner can evaluate."""
    out = render_conditions([
        {"name": "volume_spike", "params": {"multiplier": 2.0}},
        {"name": "near_52w_high", "params": {"window": 252, "factor": 0.98}},
        {"name": "shallow_pullback_long", "params": {}},
    ])
    assert out == [
        "VOL > AVG(VOL, 20) * 2",
        "CLOSE >= MAX(HIGH, 252) * 0.98",
        "MIN(LOW, 3) <= EMA(20) AND CLOSE > EMA(20)",
    ]


def test_primitive_descriptions_translate_to_user_words():
    """The no-match reply uses these one-line descriptions to show
    EXACTLY what was checked."""
    descs = primitive_descriptions([
        {"name": "volume_spike", "params": {"multiplier": 1.5}},
        {"name": "near_52w_high", "params": {"window": 252, "factor": 0.98}},
        {"name": "rsi_above", "params": {"threshold": 60}},
        {"name": "above_vwap", "params": {}},
    ])
    assert descs == [
        "volume ≥ 1.5× the 20-day average",
        "within 2% of 1-year high",
        "RSI(14) > 60",
        "price above VWAP",
    ]


# ── Chat extractor: maps prose to primitive lists ──────────────────────────


_CHAT_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "services" / "chat" / "chat_service.py"
)


def _load_extractors():
    """Compile both _extract_discovery_parameter_overrides AND
    _extract_discovery_conditions out of chat_service.py without
    importing the full module (asyncpg gap). Plus the regex constants
    they share."""
    import re as re_module
    from typing import Any, Optional
    src = _CHAT_SERVICE_PATH.read_text(encoding="utf-8")

    # The two extractors live next to each other along with their
    # regex constants. Slice from the first param-override constant
    # through the function that follows _extract_discovery_conditions.
    helpers_idx = src.index("_VOLUME_MULTIPLIER_RE = re.compile")
    block_start = helpers_idx
    while block_start > 0:
        prev_nl = src.rfind("\n", 0, block_start - 1)
        line_start = prev_nl + 1 if prev_nl != -1 else 0
        line = src[line_start:block_start - 1].strip()
        if line.startswith("#"):
            block_start = line_start
        else:
            break
    fn_marker = "def _extract_discovery_conditions("
    fn_idx = src.index(fn_marker, helpers_idx)
    after_fn = re_module.search(
        r"\n(def |async def |class )", src[fn_idx + len(fn_marker):]
    )
    end_idx = (
        fn_idx + len(fn_marker) + after_fn.start() + 1
        if after_fn
        else len(src)
    )
    body = src[block_start:end_idx]
    namespace: dict = {"re": re_module, "Optional": Optional, "Any": Any}
    exec(body, namespace)
    return (
        namespace["_extract_discovery_parameter_overrides"],
        namespace["_extract_discovery_conditions"],
    )


def _extract_conditions(message: str) -> list[dict]:
    """Helper: run both extractors as the chat layer does."""
    extract_overrides, extract_conds = _load_extractors()
    overrides = extract_overrides(message)
    return extract_conds(message, overrides)


# ── User's stated bug: minimal volume-only prompt ─────────────────────────


def test_volume_only_prompt_yields_only_volume_primitive():
    """The actual user complaint — '1.2x volume' must NOT pull in the
    52-week or pullback conditions."""
    conds = _extract_conditions(
        "create intraday strategy on NSE stock whose volume spike up today 1.2x"
    )
    assert len(conds) == 1
    assert conds[0]["name"] == "volume_spike"
    assert conds[0]["params"]["multiplier"] == 1.2


def test_just_volume_spike_phrase_uses_default_multiplier():
    """User said "volume spike" without a number — default 2x."""
    conds = _extract_conditions("intraday strategy with volume spike")
    assert len(conds) == 1
    assert conds[0]["name"] == "volume_spike"
    assert conds[0]["params"]["multiplier"] == 2.0


# ── User's 5 examples ─────────────────────────────────────────────────────


def test_example_1_volume_2x_near_52w_high_pullback():
    """Example 1: today's volume is 2x higher, price is near 52-week
    high, and pullback is less than 1%."""
    conds = _extract_conditions(
        "Create an intraday NSE strategy for stocks where today's volume is "
        "2x higher than 20-day average volume, price is near 52-week high, "
        "and pullback is less than 1%"
    )
    names = [c["name"] for c in conds]
    assert "volume_spike" in names
    assert "near_52w_high" in names
    assert "shallow_pullback_long" in names
    # Volume multiplier honored
    vol = next(c for c in conds if c["name"] == "volume_spike")
    assert vol["params"]["multiplier"] == 2.0


def test_example_2_breaking_above_52w_high_with_volume_and_vwap():
    """Example 2: stocks breaking above 52-week high with strong volume
    and VWAP confirmation."""
    conds = _extract_conditions(
        "Find NSE stocks breaking above 52-week high with strong volume "
        "and create a breakout strategy with VWAP confirmation"
    )
    names = [c["name"] for c in conds]
    assert "above_52w_high" in names
    assert "volume_spike" in names
    assert "above_vwap" in names


def test_example_3_bearish_near_52w_low_with_volume_and_pullback():
    """Example 3: bearish near 52-week low with 2x volume and weak
    recovery after pullback."""
    conds = _extract_conditions(
        "Create bearish intraday strategy for stocks near 52-week low with "
        "2x volume spike and weak recovery after pullback"
    )
    names = [c["name"] for c in conds]
    assert "near_52w_low" in names
    assert "volume_spike" in names
    assert "shallow_pullback_short" in names
    vol = next(c for c in conds if c["name"] == "volume_spike")
    assert vol["params"]["multiplier"] == 2.0


def test_example_4_volume_3x_rsi_above_60_above_vwap_near_day_high():
    """Example 4: volume spike above 3x, RSI above 60, price above VWAP,
    stock close to day high."""
    conds = _extract_conditions(
        "Build a scanner for stocks where volume spike is above 3x, "
        "RSI is above 60, price is above VWAP, and stock is close to day high"
    )
    names = [c["name"] for c in conds]
    assert "volume_spike" in names
    assert "rsi_above" in names
    assert "above_vwap" in names
    assert "near_day_high" in names
    vol = next(c for c in conds if c["name"] == "volume_spike")
    assert vol["params"]["multiplier"] == 3.0
    rsi = next(c for c in conds if c["name"] == "rsi_above")
    assert rsi["params"]["threshold"] == 60.0


def test_example_5_high_volume_above_20_ema_pullback_above_vwap():
    """Example 5: high-volume stocks where price is above 20 EMA,
    pullback is shallow, and candle closes above VWAP."""
    conds = _extract_conditions(
        "Create a pullback strategy for high-volume stocks where price is "
        "above 20 EMA, pullback is shallow, and candle closes above VWAP"
    )
    names = [c["name"] for c in conds]
    assert "volume_spike" in names         # "high-volume" → default 2x
    assert "above_ema" in names
    assert "shallow_pullback_long" in names
    assert "above_vwap" in names
    ema = next(c for c in conds if c["name"] == "above_ema")
    assert ema["params"]["period"] == 20


def test_user_original_full_prompt_picks_pullback_and_extremes():
    """The user's original intent — volume + pullback + 52w extremes."""
    conds = _extract_conditions(
        "create intraday strategy on NSE stock whose volume spike up today "
        "2x and is having low pull back or is on breaking on verge of 52 "
        "weeks high or low"
    )
    names = [c["name"] for c in conds]
    assert "volume_spike" in names
    assert "shallow_pullback_long" in names
    # "verge of 52 weeks high or low" → both extremes
    assert "near_52w_high" in names
    assert "near_52w_low" in names


def test_extractor_returns_empty_for_unrelated_prose():
    """A message with no discovery-relevant content yields no primitives."""
    assert _extract_conditions("hi how are you") == []
    assert _extract_conditions("show me the tutorial") == []


# ── Orchestrator integration: parsed primitives override preset ───────────


from app.services.discovery.orchestrator import _preset_discovery_config
from app.services.strategy.builder import StrategyBuilder


def _builder_with_preset() -> StrategyBuilder:
    b = StrategyBuilder()
    b.strategy_preset = "volume_breakout_52w"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout"
    return b


def test_orchestrator_uses_user_primitives_when_present():
    """Volume-only primitive → scanner sees ONLY the volume condition.
    The preset's pullback / 52w-high / 52w-low conditions are gone."""
    b = _builder_with_preset()
    b.discovery_conditions = [
        {"name": "volume_spike", "params": {"multiplier": 1.2}},
    ]
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    assert cfg.conditions == ["VOL > AVG(VOL, 20) * 1.2"]
    # No pullback / 52-week clauses snuck in
    assert not any("EMA(20)" in c for c in cfg.conditions)
    assert not any("MAX(HIGH, 252)" in c for c in cfg.conditions)


def test_orchestrator_falls_back_to_preset_when_no_primitives():
    """Backward compat: builder without parsed primitives → preset
    defaults still drive the scanner."""
    b = _builder_with_preset()
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    # Preset's default 2.0× volume + the OR clause
    assert any("AVG(VOL, 20) * 2" in c for c in cfg.conditions)
    assert any("MAX(HIGH, 252)" in c for c in cfg.conditions)


def test_orchestrator_renders_a_compound_primitive_list():
    """Example 4 end-to-end: 4 primitives → 4 distinct scanner conditions."""
    b = _builder_with_preset()
    b.discovery_conditions = [
        {"name": "volume_spike", "params": {"multiplier": 3.0}},
        {"name": "rsi_above",    "params": {"threshold": 60}},
        {"name": "above_vwap",   "params": {}},
        {"name": "near_day_high","params": {}},
    ]
    cfg = _preset_discovery_config(b)
    assert len(cfg.conditions) == 4
    assert cfg.conditions[0] == "VOL > AVG(VOL, 20) * 3"
    assert cfg.conditions[1] == "RSI(14) > 60"
    assert cfg.conditions[2] == "CLOSE > VWAP"
    assert cfg.conditions[3] == "CLOSE >= HIGH * 0.99"


def test_orchestrator_falls_back_when_user_primitives_fail_to_render():
    """Defensive: a malformed primitive list (e.g. typoed name) must
    NOT crash the scan — fall back to preset defaults so the user
    still gets a result."""
    b = _builder_with_preset()
    b.discovery_conditions = [
        {"name": "this_primitive_does_not_exist", "params": {}},
    ]
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    # Fallback to preset defaults
    assert any("AVG(VOL, 20) * 2" in c for c in cfg.conditions)


# ── Persistence: discovery_conditions survives across turns ────────────────


def test_discovery_conditions_round_trip_through_draft():
    b = _builder_with_preset()
    b.discovery_conditions = [
        {"name": "volume_spike", "params": {"multiplier": 1.2}},
        {"name": "above_vwap",   "params": {}},
    ]
    draft = b.to_draft_json()
    assert "discovery_conditions" in draft

    fresh = StrategyBuilder()
    fresh.merge_preview(draft)
    assert fresh.discovery_conditions is not None
    assert len(fresh.discovery_conditions) == 2
    assert fresh.discovery_conditions[0]["name"] == "volume_spike"
    assert fresh.discovery_conditions[0]["params"]["multiplier"] == 1.2
    assert fresh.discovery_conditions[1]["name"] == "above_vwap"


def test_discovery_conditions_persistence_filters_malformed_entries():
    """A draft with garbage entries must hydrate cleanly (drop the
    bad entries, keep the good ones)."""
    b = StrategyBuilder()
    b.merge_preview({
        "discovery_conditions": [
            {"name": "volume_spike", "params": {"multiplier": 2.0}},
            {"params": {"oops": 1}},                # missing name
            "not even a dict",
            {"name": "", "params": {}},             # blank name
            {"name": "above_vwap", "params": {}},
        ],
    })
    assert b.discovery_conditions == [
        {"name": "volume_spike", "params": {"multiplier": 2.0}},
        {"name": "above_vwap",   "params": {}},
    ]


# ── No-match reply surfaces the parsed primitives ─────────────────────────


from app.services.chat.strategy_flow import build_discovery_no_match_reply


def test_no_match_reply_uses_constraints_when_conditions_present():
    msg = build_discovery_no_match_reply(
        conditions_used=[
            {"name": "volume_spike", "params": {"multiplier": 1.2}},
            {"name": "above_vwap",   "params": {}},
        ],
    )
    assert "Constraints applied:" in msg
    assert "1.2× the 20-day average" in msg
    assert "above VWAP" in msg


def test_no_match_reply_falls_back_to_thresholds_summary_without_conditions():
    """Backward compat: when only parameters_used is supplied, the
    pre-9k summary form is shown."""
    msg = build_discovery_no_match_reply(
        parameters_used={"volume_multiplier": 1.5},
    )
    assert "Thresholds used:" in msg
    assert "1.5×" in msg
