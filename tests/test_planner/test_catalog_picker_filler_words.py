"""
Phase 5 — catalog picker must tolerate natural filler words ("the"/"a"/"an").

Root cause of "user said Bollinger, got Keltner": the BB entry-trigger phrase
`...above (upper )?bollinger` could not match the way users actually write it —
"closes above THE upper Bollinger band" — because of the word "the". The picker
then found no explicit trigger and fell back to the preset's Keltner. With
filler-word tolerance the picker now recognises the Bollinger entry trigger.
"""
from __future__ import annotations

from app.kb import kb
from app.planner.catalog_signal_picker import (
    merge_picker_with_preset,
    pick_plan_from_catalog,
)

BOLLINGER_PROMPT = (
    "Trade ETHUSDC on the 15-minute timeframe. Use Bollinger Bands and ATR. "
    "Enter long when price closes above the upper Bollinger Band with volume "
    "1.5x the 20-period average."
)


def _trigger_names(plan: dict) -> list[str]:
    return [
        s["name"]
        for s in plan.get("entry", [])
        if str(s.get("signal_type", "")).upper() == "TRIGGER"
    ]


def test_picker_finds_bollinger_trigger_with_the():
    pick = pick_plan_from_catalog(
        BOLLINGER_PROMPT, kb=kb, timeframe="15m", sentiment="bullish"
    )
    assert "price_above_bb_upper" in _trigger_names(pick.signal_plan)


def test_bollinger_trigger_beats_keltner_preset():
    """End-to-end: a volatility_squeeze preset (Keltner trigger) must be
    overridden by the user's explicit Bollinger entry (merge Mode A)."""
    pick = pick_plan_from_catalog(
        BOLLINGER_PROMPT, kb=kb, timeframe="15m", sentiment="bullish"
    )
    preset_plan = {
        "entry": [
            {"name": "keltner_breakout_up", "signal_type": "TRIGGER"},
            {"name": "bb_squeeze", "signal_type": "FILTER"},
        ],
        "exit": [],
        "_sl_pct": 2,
    }
    merged, mode = merge_picker_with_preset(
        preset_plan, pick.signal_plan, picker_confidence=pick.confidence
    )
    triggers = _trigger_names(merged)
    assert "price_above_bb_upper" in triggers
    assert "keltner_breakout_up" not in triggers


def test_filler_tolerance_does_not_break_plain_phrasing():
    """Phrasing WITHOUT filler words must still match (no regression)."""
    pick = pick_plan_from_catalog(
        "Enter long when price closes above upper Bollinger band.",
        kb=kb, timeframe="15m", sentiment="bullish",
    )
    assert "price_above_bb_upper" in _trigger_names(pick.signal_plan)
