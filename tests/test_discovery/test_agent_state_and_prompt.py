"""Phase 9f — agent state + prompt expose discovery context.

The previous iterations (9b/9d/9e) all populated builder.strategy_preset
upstream of the agent router, but the agent's `build_agent_state`
serialization NEVER exposed strategy_preset, so the LLM kept asking the
user for a specific stock. Plus the agent's system prompt blacklisted
the Phase 1.5 universe additions (SBIN, ICICIBANK, etc.) — so even if
discovery picked one of those, the agent might refuse to use it.

These tests lock in the four invariants that finally close the loop:

  1. build_agent_state exposes builder.strategy_preset to the LLM
  2. build_agent_state exposes discovery_will_supply_symbol (computed
     from builder.requires_discovery())
  3. The agent system prompt has a DYNAMIC DISCOVERY section telling
     the LLM not to ask for a symbol when discovery is active
  4. The agent system prompt includes the Phase 1.5 universe additions
     (SBIN, ICICIBANK, AXISBANK, KOTAKBANK, BHARTIARTL, ITC, LT,
     MARUTI, SUNPHARMA, HCLTECH) and no longer explicitly blacklists them
"""
from __future__ import annotations

import sys
import types

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import pytest

from app.services.agent.prompt import MASTER_AGENTIC_SYSTEM_PROMPT
from app.services.agent.state import build_agent_state
from app.services.strategy.builder import StrategyBuilder


def _builder_with_discovery_preset() -> StrategyBuilder:
    """Builder mid-flow: preset detected (Phase 9e early-detect), user gave
    timeframe/sentiment/experience/objective/goal, no symbol."""
    b = StrategyBuilder()
    b.strategy_preset = "volume_breakout_52w"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout intraday"
    return b


# ── state.py: discovery fields are exposed ──────────────────────────────────


def test_agent_state_exposes_strategy_preset_field():
    """The LLM cannot reason about discovery if it can't see the preset name."""
    b = _builder_with_discovery_preset()
    state = build_agent_state(session_id="s", builder=b, previous_state="collect_user_input")
    assert state["strategy_preset"] == "volume_breakout_52w"


def test_agent_state_exposes_discovery_will_supply_symbol_flag():
    """The flag is the LLM's signal to NOT ask for a stock."""
    b = _builder_with_discovery_preset()
    state = build_agent_state(session_id="s", builder=b, previous_state="collect_user_input")
    assert state["discovery_will_supply_symbol"] is True


def test_agent_state_discovery_flag_false_when_user_pinned_symbol():
    """Once the user has supplied a symbol manually, discovery should NOT
    fire — and the agent should see that in the state so it doesn't try
    to override the user's choice."""
    b = _builder_with_discovery_preset()
    b.symbol = "RELIANCE.NS"
    state = build_agent_state(session_id="s", builder=b, previous_state="collect_user_input")
    assert state["discovery_will_supply_symbol"] is False


def test_agent_state_discovery_flag_false_for_classic_preset():
    """Classic presets (orb, ema_pullback, …) don't declare a discovery
    block. The flag must be False so the agent's normal "ask for symbol"
    behavior is preserved for those presets."""
    b = StrategyBuilder()
    b.strategy_preset = "orb"
    b.timeframe = "5m"
    state = build_agent_state(session_id="s", builder=b, previous_state="collect_user_input")
    assert state["discovery_will_supply_symbol"] is False


def test_agent_state_strategy_preset_is_null_when_unset():
    """Backward compat: builders that haven't pinned a preset should
    serialize strategy_preset as None (not raise, not omit the field)."""
    b = StrategyBuilder()
    state = build_agent_state(session_id="s", builder=b, previous_state="collect_user_input")
    assert "strategy_preset" in state
    assert state["strategy_preset"] is None
    assert state["discovery_will_supply_symbol"] is False


# ── prompt.py: discovery instructions + refreshed universe ──────────────────


def test_prompt_has_dynamic_discovery_section():
    """The instruction telling the LLM not to ask for a stock under
    discovery is the actual fix — without it, the LLM has the data
    (state.discovery_will_supply_symbol) but doesn't know what to do.

    The guidance now lives under "PRESET DETECTION DISCIPLINE" rather than a
    standalone "DYNAMIC DISCOVERY" header, so we assert the SEMANTICS (the LLM can
    read the flag and is told not to ask which stock) rather than a header string —
    that survives prompt reorganisations while still locking in the rule."""
    p = MASTER_AGENTIC_SYSTEM_PROMPT.lower()
    # The LLM must be able to read the discovery flag…
    assert "discovery_will_supply_symbol" in p
    # …and be told NOT to ask the user for a stock when discovery will supply it.
    assert "do not ask" in p
    assert "which stock" in p or "specific stock" in p


def test_prompt_does_not_blacklist_phase_1_5_universe_additions():
    """Pre-Phase 1.5 the prompt explicitly told the LLM to never use
    SBIN, ICICIBANK, AXISBANK, KOTAKBANK, BHARTIARTL — those tickers
    are now in the KB universe and discovery picks from them."""
    # The literal blacklist string must be gone
    assert "no SBIN, ICICIBANK, ITC, AXISBANK, KOTAKBANK, BHARTIARTL" \
        not in MASTER_AGENTIC_SYSTEM_PROMPT
    # Belt-and-suspenders: the prompt should no longer call out the
    # phrase "Never invent or suggest tickers outside this list" with
    # the stale list following it.
    assert "(no SBIN" not in MASTER_AGENTIC_SYSTEM_PROMPT


def test_prompt_defers_to_universe_csv_instead_of_enumerating():
    """The KB universe has since grown from a hand-picked 9 stocks to 100+ equities
    (plus crypto), so the agent prompt no longer ENUMERATES supported stocks — it
    DEFERS to universe.csv. That is the drift-proof design: the LLM cannot advertise a
    stock the backend can't scan, nor refuse one it can, because it doesn't carry its
    own stale copy of the list; the backend validates against the data file.

    The original test pinned an exact 9-stock enumeration in the prompt — an invariant
    that is obsolete now that the universe is large and CSV-driven. We instead lock in
    the current contract: the prompt points at universe.csv as the source of truth."""
    p = MASTER_AGENTIC_SYSTEM_PROMPT
    assert "universe.csv" in p, (
        "agent prompt must defer to universe.csv as the supported-stock source of "
        "truth rather than enumerating a (now 100+ symbol) list that would drift"
    )


def test_prompt_still_calls_out_timeframe_validation_path():
    """Regression: the SUPPORTED UNIVERSE section must keep the timeframe-validation
    guidance. Timeframes are now a CONTINUOUS 1m–1d range (the engine resamples any
    interval from 1-minute data — see app/strategy/enums.py), not a fixed preset list,
    so the prompt describes the range and the pass-raw-to-backend path rather than
    enumerating exact values."""
    p = MASTER_AGENTIC_SYSTEM_PROMPT
    lower = p.lower()
    # The supported timeframe range bounds must be stated…
    assert "1m" in p and "1d" in p
    assert "timeframe" in lower
    # …and the "don't substitute; let the backend validate" path must remain.
    assert "pass raw to backend" in lower or "pass the user's raw" in lower
