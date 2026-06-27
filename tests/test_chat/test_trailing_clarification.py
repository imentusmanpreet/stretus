"""Trailing-knob clarification flow.

A bare "change trailing <profit|stop> percentage to N%" is ambiguous — it could
mean the give-back ``distance_pct`` or the ``activate_after_pct`` gate. The chat
flow must ASK which one instead of silently defaulting to distance, then apply
the user's choice on the follow-up turn. These tests cover the pure helpers that
implement that behaviour.
"""
from __future__ import annotations

from app.services.chat.chat_service import (
    _build_trailing_clarification_question,
    _build_risk_update_reply,
    _detect_ambiguous_trailing_update,
    _merge_agent_trailing_config,
    _resolve_trailing_choice,
    _trailing_cue,
    _TRAILING_PARAM_KEYS,
)
from app.services.strategy.builder import StrategyBuilder


# ── Ambiguity detection ────────────────────────────────────────────────────
def test_bare_trailing_profit_percentage_is_ambiguous():
    assert _detect_ambiguous_trailing_update(
        "change trailing profit percentage to 0.21",
        {"trailing_take_profit_distance_pct": 0.21},
    ) == ("trailing_take_profit", 0.21)


def test_bare_trailing_stop_percentage_is_ambiguous():
    assert _detect_ambiguous_trailing_update(
        "change trailing stop loss percentage to 0.6",
        {"trailing_stop_pct": 0.6},
    ) == ("trailing_stop", 0.6)


def test_ambiguous_regardless_of_which_param_the_llm_emits():
    # Even if the router emits the activate param, a bare phrasing is still ambiguous.
    assert _detect_ambiguous_trailing_update(
        "change trailing profit percentage to 5",
        {"trailing_take_profit_activate_after_pct": 5},
    ) == ("trailing_take_profit", 5.0)


def test_explicit_distance_cue_is_not_ambiguous():
    assert _detect_ambiguous_trailing_update(
        "change trailing profit distance to 3",
        {"trailing_take_profit_distance_pct": 3},
    ) is None


def test_explicit_activate_cue_is_not_ambiguous():
    assert _detect_ambiguous_trailing_update(
        "change trailing profit activate after to 3",
        {"trailing_take_profit_activate_after_pct": 3},
    ) is None


def test_multi_value_bulk_set_is_not_ambiguous():
    # Two values across legs → take at face value, never drop one behind a prompt.
    assert _detect_ambiguous_trailing_update(
        "set trailing stop 2% and trailing profit 3%",
        {"trailing_stop_pct": 2, "trailing_take_profit_distance_pct": 3},
    ) is None


def test_both_knobs_one_leg_is_not_ambiguous():
    assert _detect_ambiguous_trailing_update(
        "set trailing profit",
        {
            "trailing_take_profit_distance_pct": 3,
            "trailing_take_profit_activate_after_pct": 1,
        },
    ) is None


def test_no_trailing_param_is_not_ambiguous():
    assert _detect_ambiguous_trailing_update(
        "change stop loss to 2",
        {"stop_loss_pct": 2},
    ) is None


# ── Cue / choice parsing ───────────────────────────────────────────────────
def test_trailing_cue_classification():
    assert _trailing_cue("set the distance to 2") == "distance"
    assert _trailing_cue("activate after 1%") == "activate"
    assert _trailing_cue("trailing profit to 2") is None


def test_resolve_trailing_choice_words_and_indices():
    assert _resolve_trailing_choice("distance") == "distance"
    assert _resolve_trailing_choice("give back") == "distance"
    assert _resolve_trailing_choice("activate after") == "activate"
    assert _resolve_trailing_choice("kick in") == "activate"
    assert _resolve_trailing_choice("1") == "distance"
    assert _resolve_trailing_choice("2") == "activate"
    assert _resolve_trailing_choice("first") == "distance"
    assert _resolve_trailing_choice("second") == "activate"
    assert _resolve_trailing_choice("something else") is None


def test_param_key_map_is_complete():
    assert _TRAILING_PARAM_KEYS[("trailing_take_profit", "distance")] == "trailing_take_profit_distance_pct"
    assert _TRAILING_PARAM_KEYS[("trailing_take_profit", "activate")] == "trailing_take_profit_activate_after_pct"
    assert _TRAILING_PARAM_KEYS[("trailing_stop", "distance")] == "trailing_stop_pct"
    assert _TRAILING_PARAM_KEYS[("trailing_stop", "activate")] == "trailing_stop_activate_after_pct"


def test_clarification_question_names_the_value_and_leg():
    q = _build_trailing_clarification_question("trailing_take_profit", 0.21)
    assert "trailing take-profit" in q
    assert "0.21%" in q
    assert "distance" in q and "activate" in q


# ── Full resolution chain (detect → choose → apply to the chosen knob) ──────
def _fresh_builder():
    b = StrategyBuilder()
    b.trailing_take_profit_spec = {
        "type": "percent", "source": "user", "distance_pct": 0.5, "activate_after_pct": 0.53,
    }
    b.take_profit = 0.0
    return b


def test_resolution_applies_value_to_distance_knob():
    b = _fresh_builder()
    # User picked "distance" for a pending TTP value of 0.21.
    key = _TRAILING_PARAM_KEYS[("trailing_take_profit", _resolve_trailing_choice("distance"))]
    changed = _merge_agent_trailing_config(b, {key: 0.21}, {"rms_sources": {}})
    assert changed == ["trailing_take_profit"]
    assert b.trailing_take_profit_spec["distance_pct"] == 0.21
    # Activation preserved — only the chosen knob changed.
    assert b.trailing_take_profit_spec["activate_after_pct"] == 0.53


def test_resolution_applies_value_to_activate_knob():
    b = _fresh_builder()
    # User picked "activate after" for a pending TTP value of 0.21.
    key = _TRAILING_PARAM_KEYS[("trailing_take_profit", _resolve_trailing_choice("activate after"))]
    changed = _merge_agent_trailing_config(b, {key: 0.21}, {"rms_sources": {}})
    assert changed == ["trailing_take_profit"]
    assert b.trailing_take_profit_spec["activate_after_pct"] == 0.21
    # Distance preserved — only the chosen knob changed.
    assert b.trailing_take_profit_spec["distance_pct"] == 0.5


def test_resolution_index_choice_maps_to_activate():
    b = _fresh_builder()
    # Reply "2" → activate.
    key = _TRAILING_PARAM_KEYS[("trailing_take_profit", _resolve_trailing_choice("2"))]
    _merge_agent_trailing_config(b, {key: 1.5}, {"rms_sources": {}})
    assert b.trailing_take_profit_spec["activate_after_pct"] == 1.5
    assert b.trailing_take_profit_spec["distance_pct"] == 0.5
