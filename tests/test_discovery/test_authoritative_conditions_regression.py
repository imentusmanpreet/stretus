"""Post-PR#71 follow-up regression — pin the Phase 9k/9l design intent.

PR #71 introduced `_preset_carries_authoritative_conditions(...)`, a
guard that suppressed primitive extraction whenever the pinned preset
had a `discovery.conditions:` block. The stated rationale ("Bug D —
OR-disjunction wiped") was a misread of the orchestrator's behaviour
and the net effect was to UN-FIX Phases 9k+9l: for the
`volume_breakout_52w` preset, a user typing "volume spike 1.2x"
would get the preset's full 52-week conditions applied anyway,
exactly the symptom the user complained about ("Why is it talking
about 52-week when I asked about volume").

This test locks in the correct flow so that guard can't be
reintroduced silently:

  1. When the user supplies primitives, the orchestrator MUST
     replace the preset's `discovery.conditions:` block with the
     rendered primitive list — no implicit 52-week clauses.
  2. When the user supplies NOTHING, the preset's conditions still
     apply via the fall-through path (so legacy "no override"
     flows are preserved).
"""
from __future__ import annotations

import sys
import types

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import pytest

from app.services.discovery.orchestrator import _preset_discovery_config
from app.services.strategy.builder import StrategyBuilder


def _builder_for(preset_name: str = "volume_breakout_52w") -> StrategyBuilder:
    b = StrategyBuilder()
    b.strategy_preset = preset_name
    b.timeframe = "5m"
    return b


def test_volume_only_user_primitives_replace_preset_conditions():
    """User typed 'volume spike 1.2x' → ONLY volume condition runs.
    The preset's 52-week OR clause must NOT leak through."""
    b = _builder_for("volume_breakout_52w")
    b.discovery_conditions = [
        {"name": "volume_spike", "params": {"multiplier": 1.2}},
    ]
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    assert cfg.conditions == ["VOL > AVG(VOL, 20) * 1.2"], (
        f"user-supplied primitive list MUST replace the preset's full "
        f"conditions block, not augment it. Got: {cfg.conditions}"
    )
    # And specifically — no 52-week clause should appear.
    assert not any("MAX(HIGH, 252)" in c or "MIN(LOW, 252)" in c
                   for c in cfg.conditions), (
        "the preset's 52-week clauses must not appear when the user "
        "only typed about volume"
    )


def test_user_supplied_multi_primitive_list_replaces_preset_conditions():
    """User typed 'RSI > 70 and above VWAP' → exactly those two
    conditions, no preset extras."""
    b = _builder_for("volume_breakout_52w")
    b.discovery_conditions = [
        {"name": "rsi_above",  "params": {"threshold": 70}},
        {"name": "above_vwap"},
    ]
    cfg = _preset_discovery_config(b)
    assert cfg.conditions == ["RSI(14) > 70", "CLOSE > VWAP"]


def test_no_user_primitives_falls_through_to_preset_conditions():
    """When the user supplies NO primitives (legacy flow), the
    preset's full conditions block IS used. This is the fall-through
    path PR #71's guard was trying to protect — but it should kick
    in automatically based on `builder.discovery_conditions is None`,
    not require a separate guard."""
    b = _builder_for("volume_breakout_52w")
    b.discovery_conditions = None
    cfg = _preset_discovery_config(b)
    # The preset's conditions DO carry through. Both clauses present.
    joined = " ".join(cfg.conditions)
    assert "VOL > AVG(VOL, 20)" in joined
    assert "MAX(HIGH, 252)" in joined or "MIN(LOW, 252)" in joined


def test_empty_user_primitive_list_is_treated_as_no_override():
    """An empty list should fall through to the preset's conditions
    (defensive — chat layer shouldn't produce empty lists but if it
    does, don't silently strip all constraints)."""
    b = _builder_for("volume_breakout_52w")
    b.discovery_conditions = []
    cfg = _preset_discovery_config(b)
    joined = " ".join(cfg.conditions)
    assert "VOL" in joined


def test_no_authoritative_conditions_helper_remains_in_chat_service():
    """Source-text guard: if a future PR re-introduces the function
    definition, this test fails so the issue gets revisited in review
    rather than silently regressing the user's flow. (The function
    NAME may still appear in inline comments documenting why the
    guard was removed — we match the `def` line specifically.)"""
    from pathlib import Path
    chat_service_path = (
        Path(__file__).resolve().parents[2]
        / "app" / "services" / "chat" / "chat_service.py"
    )
    src = chat_service_path.read_text(encoding="utf-8")
    assert "def _preset_carries_authoritative_conditions" not in src, (
        "The `_preset_carries_authoritative_conditions` guard reverses "
        "Phase 9k/9l's design (the user gets the preset's full conditions "
        "applied even when they only typed about volume). Don't re-add it "
        "— the orchestrator's existing fall-through behaviour (use "
        "primitives when builder.discovery_conditions is set, otherwise "
        "use the preset) is already correct."
    )
