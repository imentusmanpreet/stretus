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
    (state.discovery_will_supply_symbol) but doesn't know what to do."""
    assert "DYNAMIC DISCOVERY" in MASTER_AGENTIC_SYSTEM_PROMPT
    # The critical guidance — phrased multiple ways so a small prompt
    # edit doesn't accidentally remove the rule.
    p = MASTER_AGENTIC_SYSTEM_PROMPT.lower()
    assert "discovery_will_supply_symbol" in p
    assert "do not ask" in p and "specific stock" in p


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


def test_prompt_supported_stocks_match_universe_csv():
    """Post-PR#71 the KB universe was trimmed to 9 stocks; the agent
    prompt must list exactly those — no more, no less. Drift between
    the prompt and the data file causes the LLM to either accept stocks
    the backend can't scan (advertise a non-existent capability) or
    refuse stocks the backend does support."""
    from app.kb import kb

    # Hardcoded mapping from canonical bare ticker to a string the
    # prompt is expected to contain. Keeps the test deterministic
    # while tolerating friendly renames (e.g. "Tata Consultancy
    # Services" → "TCS" in prose).
    expected_prompt_token = {
        "HDFCBANK":   "HDFC Bank",
        "RELIANCE":   "Reliance",
        "TCS":        "TCS",
        "INFY":       "Infosys",
        "ADANIENT":   "Adani Enterprises",
        "NHPC":       "NHPC",
        "SUZLON":     "Suzlon",
        "GMRAIRPORT": "GMR Airports",
        "IDEA":       "Vodafone Idea",
    }

    # Every enabled stock must have its expected token in the prompt.
    enabled_bare = {
        s.symbol.split(".", 1)[0].upper()
        for s in kb.stocks.values() if s.enabled
    }
    for bare in enabled_bare:
        token = expected_prompt_token.get(bare)
        assert token is not None, (
            f"universe.csv has new stock {bare!r} but this test doesn't "
            f"know what prompt token to expect — add it to "
            f"`expected_prompt_token`"
        )
        assert token in MASTER_AGENTIC_SYSTEM_PROMPT, (
            f"agent prompt missing universe stock {bare!r} "
            f"(searched for {token!r})"
        )

    # And in the OTHER direction: any bare ticker NOT in universe.csv
    # must NOT appear in the prompt as a supported stock. This catches
    # drift where the universe is shrunk but the prompt still
    # advertises the removed tickers.
    deprecated_tokens = [
        "ICICI Bank", "SBIN", "Axis Bank", "Kotak", "Bharti Airtel",
        "Maruti", "Sun Pharma", "HCL Technologies",
    ]
    for token in deprecated_tokens:
        if token in expected_prompt_token.values():
            # This token IS in the universe (e.g. a sub-string of a
            # currently-enabled stock); skip the deprecation check.
            continue
        assert token not in MASTER_AGENTIC_SYSTEM_PROMPT, (
            f"agent prompt still advertises {token!r} but it is not in "
            f"universe.csv — the LLM will accept user requests for it "
            f"that the backend can't fulfil"
        )


def test_prompt_still_calls_out_timeframe_validation_path():
    """Regression: I rewrote the SUPPORTED UNIVERSE section; the
    timeframe-validation guidance that lived in the same section must
    still be intact."""
    p = MASTER_AGENTIC_SYSTEM_PROMPT
    assert "Supported timeframes are exactly 1m, 5m, 10m, 15m, 30m, 1h, 1d" in p
    assert "pass the user's raw value to the validator" in p
