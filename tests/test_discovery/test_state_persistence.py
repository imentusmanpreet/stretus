"""Phase 9j — discovery state must persist across chat turns.

The chat service rehydrates a fresh StrategyBuilder on every turn by
calling `builder.merge_preview(last_draft)` where `last_draft` is the
`to_draft_json()` from the previous turn. Pre-9j, `to_draft_json` did
NOT serialize ANY discovery state (pending, candidates, tie-break
options, the resolved symbol, parameter_overrides) — so whatever the
chat layer learned mid-flow was wiped the moment the user replied.

That broke:
  1. The user-typed parameter overrides from Phases 9h/9i. Turn 1:
     user types "create intraday strategy on NSE stock whose volume
     spike up today 1.2x". Builder captures volume_multiplier=1.2.
     Turn 2: user types "1m" (the timeframe). Builder rehydrates
     from draft, overrides → None. Scanner runs at the default 2.0×.
  2. The tie-break flow from Phase 9b. Turn 1: scanner returns 4
     candidates, builder.discovery_pending=True. Turn 2: user types
     "1". Builder rehydrates without pending=True, so the reply
     gets routed to normal flow instead of handle_pending_tie_break.

These tests round-trip the builder through to_draft_json → fresh
builder + merge_preview and confirm every discovery field survives.
"""
from __future__ import annotations

import sys
import types

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import pytest

from app.services.strategy.builder import StrategyBuilder


def _builder_with_full_discovery_state() -> StrategyBuilder:
    """A builder with every discovery field populated. Drives the
    round-trip tests so we know every field is covered."""
    b = StrategyBuilder()
    b.strategy_preset = "volume_breakout_52w"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout"
    b.discovery_pending = True
    b.discovery_no_match = False
    b.discovery_candidates = [
        {"symbol": "HDFCBANK.NS", "display_name": "HDFC Bank",
         "sector": "banking", "metrics": {"relative_volume": 3.2}},
        {"symbol": "RELIANCE.NS", "display_name": "Reliance",
         "sector": "energy", "metrics": {"relative_volume": 2.1}},
    ]
    b.discovery_tie_break_options = [
        {"method": "highest_relative_volume",
         "label": "Highest relative volume", "description": "..."},
        {"method": "closest_to_52w_high",
         "label": "Closest to 52-week high", "description": "..."},
    ]
    b.discovery_chosen_method = None
    b.discovered_symbol = None
    b.discovery_parameter_overrides = {
        "volume_multiplier": 1.2,
        "lookback_window_bars": 260.0,
    }
    b.discovery_parameters_used = {
        "volume_multiplier": 1.2,
        "near_52w_high_factor": 0.98,
        "near_52w_low_factor": 1.02,
        "lookback_window_bars": 260.0,
    }
    return b


def _roundtrip(b: StrategyBuilder) -> StrategyBuilder:
    """Simulate one chat turn: serialise to draft, hydrate fresh."""
    draft = b.to_draft_json()
    fresh = StrategyBuilder()
    fresh.merge_preview(draft)
    return fresh


# ── Each field round-trips ──────────────────────────────────────────────────


def test_discovery_pending_persists_across_turns():
    b = _builder_with_full_discovery_state()
    fresh = _roundtrip(b)
    assert fresh.discovery_pending is True


def test_discovery_no_match_persists_across_turns():
    b = _builder_with_full_discovery_state()
    b.discovery_pending = False
    b.discovery_no_match = True
    fresh = _roundtrip(b)
    assert fresh.discovery_no_match is True


def test_discovery_candidates_persist_across_turns():
    b = _builder_with_full_discovery_state()
    fresh = _roundtrip(b)
    assert fresh.discovery_candidates is not None
    assert len(fresh.discovery_candidates) == 2
    assert fresh.discovery_candidates[0]["symbol"] == "HDFCBANK.NS"
    assert fresh.discovery_candidates[0]["metrics"]["relative_volume"] == 3.2


def test_discovery_tie_break_options_persist_across_turns():
    b = _builder_with_full_discovery_state()
    fresh = _roundtrip(b)
    assert fresh.discovery_tie_break_options is not None
    assert fresh.discovery_tie_break_options[0]["method"] == "highest_relative_volume"
    assert "Highest relative volume" in fresh.discovery_tie_break_options[0]["label"]


def test_discovery_chosen_method_persists_across_turns():
    b = _builder_with_full_discovery_state()
    b.discovery_pending = False
    b.discovery_chosen_method = "highest_relative_volume"
    fresh = _roundtrip(b)
    assert fresh.discovery_chosen_method == "highest_relative_volume"


def test_discovered_symbol_persists_across_turns():
    b = _builder_with_full_discovery_state()
    b.discovered_symbol = "HDFCBANK.NS"
    fresh = _roundtrip(b)
    assert fresh.discovered_symbol == "HDFCBANK.NS"


def test_discovery_parameter_overrides_persist_across_turns():
    """The core 9j regression — user types 1.2x on turn 1, the
    override must survive to turn 2."""
    b = _builder_with_full_discovery_state()
    fresh = _roundtrip(b)
    assert fresh.discovery_parameter_overrides == {
        "volume_multiplier": 1.2,
        "lookback_window_bars": 260.0,
    }


def test_discovery_parameters_used_persist_across_turns():
    b = _builder_with_full_discovery_state()
    fresh = _roundtrip(b)
    assert fresh.discovery_parameters_used == {
        "volume_multiplier": 1.2,
        "near_52w_high_factor": 0.98,
        "near_52w_low_factor": 1.02,
        "lookback_window_bars": 260.0,
    }


# ── Default / empty state also round-trips cleanly ──────────────────────────


def test_fresh_builder_serialises_without_discovery_state():
    """A brand-new builder must serialise without raising and must
    rehydrate to the same default discovery state."""
    b = StrategyBuilder()
    draft = b.to_draft_json()
    # Fields are present in the draft (may be None / False / empty).
    assert "discovery_pending" in draft
    assert "discovery_parameter_overrides" in draft

    fresh = StrategyBuilder()
    fresh.merge_preview(draft)
    assert fresh.discovery_pending is False
    assert fresh.discovery_no_match is False
    assert fresh.discovery_candidates is None
    assert fresh.discovery_tie_break_options is None
    assert fresh.discovery_chosen_method is None
    assert fresh.discovered_symbol is None
    assert fresh.discovery_parameter_overrides is None
    assert fresh.discovery_parameters_used is None


# ── Robustness: malformed data on the draft is silently ignored ─────────────


def test_merge_preview_rejects_unknown_override_value_types():
    """A non-numeric override should be filtered out rather than crash
    the rehydration."""
    b = StrategyBuilder()
    b.merge_preview({
        "discovery_parameter_overrides": {
            "volume_multiplier": 1.5,
            "garbage_value": "not-a-number",
        },
    })
    # Only the numeric value survives.
    assert b.discovery_parameter_overrides == {"volume_multiplier": 1.5}


def test_merge_preview_handles_missing_discovery_block():
    """A pre-9j draft (no discovery_* keys) must hydrate cleanly into a
    fresh-state builder — backward compat for sessions in flight when
    9j ships."""
    b = StrategyBuilder()
    # Manually craft a "legacy" draft missing all discovery_* keys.
    legacy_draft = {
        "mode": "collect_user_input",
        "market": "indian_stocks",
        "symbol": None,
        "timeframe": "5m",
    }
    b.merge_preview(legacy_draft)
    assert b.discovery_pending is False
    assert b.discovery_parameter_overrides is None


# ── End-to-end: the user's bug ──────────────────────────────────────────────


def test_user_typed_1_2x_survives_a_turn_of_timeframe_input():
    """Turn 1: user types the volume_multiplier=1.2 prompt. Turn 2:
    user types just "1m" (no volume context). The override from
    turn 1 must still be on the builder so the scanner uses 1.2×."""
    # Turn 1: builder captures the override (simulated)
    turn1 = StrategyBuilder()
    turn1.strategy_preset = "volume_breakout_52w"
    turn1.objective = "intraday"
    turn1.discovery_parameter_overrides = {"volume_multiplier": 1.2}
    draft_after_turn1 = turn1.to_draft_json()

    # Turn 2: chat service rehydrates a fresh builder from the
    # previous draft, then captures only the timeframe from "1m"
    # (no volume context → parser yields {} on turn 2's content).
    turn2 = StrategyBuilder()
    turn2.merge_preview(draft_after_turn1)
    turn2.timeframe = "1m"   # captured from user's "1m" message

    # Critical invariant: the 1.2× override survived.
    assert turn2.discovery_parameter_overrides == {"volume_multiplier": 1.2}
