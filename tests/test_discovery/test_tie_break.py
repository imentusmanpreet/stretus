"""Phase 9 — tie-break framework: methods, parser, fallbacks."""
from __future__ import annotations

import sys
import types

# Stub asyncpg before any app.services.* import so DB layer loads cleanly
# (same workaround used elsewhere; pre-existing local-env gap).
if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import pytest

from app.services.discovery.tie_break import (
    TIE_BREAK_METHODS,
    apply_tie_break,
    available_tie_break_options,
    parse_user_tie_break_reply,
)
from app.services.discovery.types import Candidate, TieBreakOption


def _c(symbol: str, **metrics) -> Candidate:
    return Candidate(symbol=symbol, display_name=symbol.split(".")[0], sector="x", metrics=metrics)


# ── TIE_BREAK_METHODS rankers ────────────────────────────────────────────────


def test_highest_relative_volume_picks_highest_first():
    cs = [_c("A.NS", relative_volume=2.5), _c("B.NS", relative_volume=4.0), _c("C.NS", relative_volume=1.5)]
    out = apply_tie_break("highest_relative_volume", cs)
    assert [c.symbol for c in out] == ["B.NS", "A.NS", "C.NS"]


def test_closest_to_52w_high_picks_smallest_distance_first():
    cs = [_c("A.NS", distance_to_52w_high_pct=3.0),
          _c("B.NS", distance_to_52w_high_pct=0.5),
          _c("C.NS", distance_to_52w_high_pct=1.7)]
    out = apply_tie_break("closest_to_52w_high", cs)
    assert [c.symbol for c in out] == ["B.NS", "C.NS", "A.NS"]


def test_highest_rsi_picks_highest_first():
    cs = [_c("A.NS", rsi_14=55), _c("B.NS", rsi_14=72), _c("C.NS", rsi_14=40)]
    out = apply_tie_break("highest_rsi", cs)
    assert out[0].symbol == "B.NS"


def test_alphabetical_picks_first_symbol():
    cs = [_c("ZZZ.NS"), _c("AAA.NS"), _c("MMM.NS")]
    out = apply_tie_break("alphabetical", cs)
    assert out[0].symbol == "AAA.NS"


def test_apply_tie_break_with_missing_metric_sends_candidate_last():
    """A candidate missing the metric must not crash the sort, must go
    to the end of the ranking."""
    cs = [_c("A.NS"), _c("B.NS", relative_volume=3.0), _c("C.NS")]
    out = apply_tie_break("highest_relative_volume", cs)
    assert out[0].symbol == "B.NS"
    # A.NS and C.NS both have no metric — order between them is undefined
    # but they must come AFTER B.
    assert out[1].symbol in {"A.NS", "C.NS"}


def test_apply_tie_break_unknown_method_raises():
    with pytest.raises(KeyError, match="unknown tie_break method"):
        apply_tie_break("magic", [_c("A.NS")])


def test_apply_tie_break_empty_input_returns_empty():
    assert apply_tie_break("highest_relative_volume", []) == []


def test_all_methods_have_metadata():
    """available_tie_break_options() for the full known set returns one
    option per method, all with non-empty labels."""
    opts = available_tie_break_options()
    method_ids = {o.method for o in opts}
    assert method_ids == set(TIE_BREAK_METHODS.keys())
    for o in opts:
        assert o.label
        assert isinstance(o, TieBreakOption)


# ── parse_user_tie_break_reply ───────────────────────────────────────────────


@pytest.fixture
def options():
    return [
        TieBreakOption(method="highest_relative_volume", label="Highest relative volume"),
        TieBreakOption(method="closest_to_52w_high",     label="Closest to 52-week high"),
        TieBreakOption(method="highest_rsi",             label="Highest RSI(14)"),
    ]


def test_parse_user_reply_accepts_numeric_index(options):
    assert parse_user_tie_break_reply("1", options) == "highest_relative_volume"
    assert parse_user_tie_break_reply("2", options) == "closest_to_52w_high"
    assert parse_user_tie_break_reply("3.", options) == "highest_rsi"


def test_parse_user_reply_accepts_method_id(options):
    assert parse_user_tie_break_reply("highest_relative_volume", options) == "highest_relative_volume"
    assert parse_user_tie_break_reply("HIGHEST_RSI", options) == "highest_rsi"


def test_parse_user_reply_accepts_full_label_case_insensitive(options):
    assert parse_user_tie_break_reply("highest relative volume", options) == "highest_relative_volume"
    assert parse_user_tie_break_reply("Closest to 52-week high", options) == "closest_to_52w_high"


def test_parse_user_reply_accepts_unique_substring(options):
    assert parse_user_tie_break_reply("rsi", options) == "highest_rsi"
    assert parse_user_tie_break_reply("relative", options) == "highest_relative_volume"


def test_parse_user_reply_rejects_out_of_range_index(options):
    assert parse_user_tie_break_reply("0", options) is None     # 1-based
    assert parse_user_tie_break_reply("99", options) is None


def test_parse_user_reply_rejects_ambiguous_substring(options):
    # "h" matches all three labels — must be ambiguous, returns None
    assert parse_user_tie_break_reply("h", options) is None


def test_parse_user_reply_rejects_blank_and_unknown(options):
    assert parse_user_tie_break_reply("", options) is None
    assert parse_user_tie_break_reply("   ", options) is None
    assert parse_user_tie_break_reply("garbage", options) is None


def test_parse_user_reply_with_no_options_returns_none(options):
    assert parse_user_tie_break_reply("anything", []) is None
