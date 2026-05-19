"""Phase 9m — no-match diagnostics.

When the scanner returns zero candidates, the reply has to give the
user enough information to take an informed next step:

  1. The exact constraints applied (Phase 9k+)
  2. Universe size + asof date (so stale data / tiny universe is
     visible at a glance)
  3. Per-condition failure counts (so the user knows WHICH constraint
     was the bottleneck)
  4. Tailored suggestions based ONLY on active constraints — no more
     "relax 52-week proximity" when 52-week isn't even applied

This is the user's last line of defense against a silent-failure
"nothing matched, good luck" message. These tests pin each part.
"""
from __future__ import annotations

import sys
import types

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import pytest

from app.services.chat.strategy_flow import build_discovery_no_match_reply


# ── Active-constraint-aware suggestions ────────────────────────────────────


def test_volume_only_active_does_not_mention_52_week_proximity():
    """The exact bug the user reported: a volume-only prompt produced
    'relax the 52-week proximity' text even though no 52-week
    condition was applied."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
    )
    assert "52-week" not in msg, (
        "no-match reply must not mention 52-week proximity when only "
        "volume_spike is active"
    )
    assert "Lower the volume threshold" in msg or "volume" in msg.lower()


def test_volume_suggestion_proposes_lower_multiplier():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 2.0}}],
    )
    # Suggestion should propose dropping to a lower value (round to
    # 0.1, capped at 1.0).
    assert "from 2× to 1.7×" in msg or "from 2 to 1.7" in msg or "1.7" in msg


def test_volume_suggestion_caps_at_one():
    """Don't suggest a sub-1× volume threshold (it's meaningless —
    you'd match every stock that traded at all)."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.1}}],
    )
    # Either suggests 1.0× exactly, or doesn't suggest at all (since
    # 1.1 - 0.3 < 1, clamped to 1.0).
    if "Lower the volume threshold" in msg:
        # Make sure no number below 1 appears in the volume hint
        # The hint format is "from X× to Y×" — both should be ≥ 1.
        assert "to 0" not in msg


def test_rsi_active_suggests_lower_threshold():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "rsi_above", "params": {"threshold": 80}}],
    )
    assert "RSI threshold" in msg
    assert "70" in msg  # 80 - 10 = 70


def test_rsi_below_active_suggests_higher_threshold():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "rsi_below", "params": {"threshold": 20}}],
    )
    assert "RSI threshold" in msg
    assert "30" in msg  # 20 + 10 = 30


def test_pullback_active_suggests_dropping_pullback():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "shallow_pullback_long", "params": {}}],
    )
    assert "pullback constraint" in msg.lower() or "drop the pullback" in msg.lower()


def test_above_breakout_active_suggests_switching_to_proximity():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "above_52w_high", "params": {"window": 252}}],
    )
    # The breakout hint suggests switching to proximity
    assert "proximity" in msg.lower()


def test_near_52w_high_active_suggests_widening_proximity():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "near_52w_high", "params": {"window": 252, "factor": 0.98}}],
        parameters_used={"near_52w_high_factor": 0.98, "lookback_window_bars": 252},
    )
    assert "52-week proximity" in msg


def test_above_vwap_only_gets_generic_fallback_hint():
    """When the only active constraint has no tunable param, fall
    back to a generic message rather than inventing a suggestion."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "above_vwap", "params": {}}],
    )
    # The generic fallback fires when no primitive-specific hint
    # applies.
    assert "Loosen the constraints" in msg or "relaxed thresholds" in msg.lower()


# ── Diagnostics: asof + universe size + per-condition fails ────────────────


def test_diagnostics_shows_universe_size_and_asof_date():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T15:30:00Z",
            "scanned_count": 19,
            "failed_fetches": 0,
            "fail_counts": {"VOL > AVG(VOL, 20) * 1.2": 19},
        },
    )
    assert "Scanned 19 stocks" in msg
    assert "2026-05-13" in msg
    # Per-condition fail count surfaces in the diagnostic line.
    assert "19" in msg  # the fail count


def test_diagnostics_calls_out_excluded_fetches():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T15:30:00Z",
            "scanned_count": 19,
            "failed_fetches": 5,
            "fail_counts": {"VOL > AVG(VOL, 20) * 1.2": 14},
        },
    )
    # Renderer phrases this as "5 stocks excluded" when no per-symbol
    # list is supplied, or "Couldn't fetch data (5)" when it is.
    assert ("5 stocks excluded" in msg) or ("Couldn't fetch data" in msg)


def test_diagnostics_shows_worst_failing_condition_description():
    """The user should see WHICH condition was the bottleneck. When
    18/19 stocks failed at the volume condition and 0/19 failed at
    the VWAP condition, the diagnostic line must point at volume."""
    msg = build_discovery_no_match_reply(
        conditions_used=[
            {"name": "volume_spike", "params": {"multiplier": 3.0}},
            {"name": "above_vwap"},
        ],
        scan_diagnostics={
            "asof_iso": "2026-05-13T00:00:00Z",
            "scanned_count": 19,
            "failed_fetches": 0,
            "fail_counts": {
                "VOL > AVG(VOL, 20) * 3": 18,
                "CLOSE > VWAP": 0,
            },
        },
    )
    # The worst-failing condition gets called out — the user-friendly
    # description from primitives.py is what appears.
    assert "volume ≥ 3× the 20-day average" in msg or "volume" in msg.lower()
    assert "18" in msg


def test_diagnostics_stale_asof_surfaces_hint():
    """When the asof date is older than 3 days, surface a 'data may
    be stale' hint at the TOP of the suggestions."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2020-01-01T00:00:00Z",   # very stale
            "scanned_count": 19,
            "failed_fetches": 0,
            "fail_counts": {"VOL > AVG(VOL, 20) * 1.2": 19},
        },
    )
    assert "stale" in msg.lower() or "market is closed" in msg.lower()


def test_diagnostics_recent_asof_does_not_surface_stale_hint():
    """A 1-day-old asof is normal market behaviour — no stale hint."""
    from datetime import datetime, timezone, timedelta
    yesterday = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat().replace("+00:00", "Z")
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": yesterday,
            "scanned_count": 19,
            "failed_fetches": 0,
            "fail_counts": {},
        },
    )
    assert "stale" not in msg.lower()


# ── Backward compat ───────────────────────────────────────────────────────


def test_pre_9m_callers_without_diagnostics_still_get_reply():
    """Callers that don't pass scan_diagnostics get the same reply
    they had pre-9m (just no diagnostic line)."""
    msg = build_discovery_no_match_reply()
    assert "No stocks matched" in msg
    assert "You can either:" in msg


def test_diagnostics_missing_fields_degrades_gracefully():
    """A malformed diagnostics dict should not raise."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            # missing scanned_count and asof_iso
            "fail_counts": {},
        },
    )
    assert "No stocks matched" in msg


# ── Multi-constraint user gets all relevant hints ──────────────────────────


def test_multi_constraint_prompt_gets_all_relevant_hints():
    """User typed volume + RSI + near-high. Hints should cover ALL
    three tunables, not just one."""
    msg = build_discovery_no_match_reply(
        conditions_used=[
            {"name": "volume_spike", "params": {"multiplier": 2.5}},
            {"name": "rsi_above", "params": {"threshold": 70}},
            {"name": "near_52w_high", "params": {"window": 252, "factor": 0.98}},
        ],
        parameters_used={"lookback_window_bars": 252},
    )
    # All three families of hint should be present
    assert "volume" in msg.lower()
    assert "RSI" in msg
    assert "52-week" in msg


# ── Source-text invariants the scanner pipeline relies on ─────────────────


def test_scan_result_carries_fail_counts():
    """The ScanResult dataclass must have a fail_counts field so the
    orchestrator can stash diagnostics on the builder."""
    from app.services.discovery.types import ScanResult, TieBreakOption
    sr = ScanResult(
        status="none", candidates=[], tie_break_options=[],
    )
    assert hasattr(sr, "fail_counts")
    assert isinstance(sr.fail_counts, dict)


def test_scan_result_dict_serialises_fail_counts():
    from app.services.discovery.types import ScanResult
    sr = ScanResult(
        status="none", candidates=[], tie_break_options=[],
        fail_counts={"VOL > AVG(VOL, 20) * 1.2": 19},
    )
    out = sr.to_dict()
    assert out["fail_counts"] == {"VOL > AVG(VOL, 20) * 1.2": 19}
