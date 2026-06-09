"""Exit-filter placement: lenient (planner) mode must accept exit FILTERS.

Regression for the "doji_indecision … cannot be used as a standalone exit
signal" crash at assemble: the planner legitimately picks exit filters
alongside the exit trigger, but lenient validation was asymmetric — it allowed
entry filters in the entry slot yet rejected exit filters in the exit slot,
crashing yaml generation. Strict mode (a user naming a single signal as "the
exit") must still reject filter-only cards.
"""
from __future__ import annotations

from app.kb import kb
from app.kb.signal_validation import (
    assert_valid_signal_lists,
    validate_signal_placement,
)


def test_exit_filter_allowed_in_lenient_mode():
    # doji_indecision has the exit_filter role (no exit_trigger).
    assert "exit_filter" in kb.signals["doji_indecision"].roles
    assert validate_signal_placement("doji_indecision", "exit", mode="lenient") is None


def test_exit_filter_still_rejected_in_strict_mode():
    # Strict = user explicitly naming it as THE exit → still rejected.
    err = validate_signal_placement("doji_indecision", "exit", mode="strict")
    assert err is not None
    assert "filter" in err.message.lower()


def test_entry_only_card_still_rejected_in_exit_lenient():
    # The fix must NOT let a genuine entry-only card into the exit slot.
    assert "exit_filter" not in kb.signals["ema_cross_up"].roles
    assert validate_signal_placement("ema_cross_up", "exit", mode="lenient") is not None


def test_planner_exit_list_with_filter_does_not_crash():
    # A plan whose exit list = [trigger, exit-filter] must pass lenient batch
    # validation (this is exactly what builder.to_yaml_dict runs at assemble).
    assert_valid_signal_lists(
        ["ema_cross_up"],
        ["supertrend_bullish", "doji_indecision"],
        mode="lenient",
    )
