"""Phase 9b — chat_integration helpers.

Tests the wrappers that the chat_service.py call sites use. Both
`handle_pending_tie_break` and `maybe_dispatch_discovery` return a
DiscoveryStepResult so the integration in chat_service.py is just a
small if-block — these tests validate the result contract for every
status case.

The orchestrator is mocked so we don't hit OHLCV fetchers; the helpers
are pure wiring on top of the orchestrator's already-tested behavior.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

from app.services.discovery.chat_integration import (
    DiscoveryStepResult,
    handle_pending_tie_break,
    maybe_dispatch_discovery,
)
from app.services.discovery.types import Candidate, ScanResult, TieBreakOption
from app.services.strategy.builder import StrategyBuilder


def _builder_with_pending_tie_break() -> StrategyBuilder:
    """A builder that has already run discovery and got multiple candidates,
    awaiting the user's tie-break choice."""
    b = StrategyBuilder()
    b.strategy_preset = "volume_breakout_52w"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout"
    b.discovery_pending = True
    b.discovery_candidates = [
        {"symbol": "HDFCBANK.NS", "display_name": "HDFC Bank", "sector": "banking",
         "metrics": {"relative_volume": 3.2, "distance_to_52w_high_pct": 0.8, "rsi_14": 68.0}},
        {"symbol": "RELIANCE.NS", "display_name": "Reliance Industries", "sector": "energy",
         "metrics": {"relative_volume": 2.1, "distance_to_52w_high_pct": 1.5, "rsi_14": 61.0}},
    ]
    b.discovery_tie_break_options = [
        {"method": "highest_relative_volume", "label": "Highest relative volume", "description": ""},
        {"method": "closest_to_52w_high",     "label": "Closest to 52-week high", "description": ""},
    ]
    return b


def _builder_for_fresh_discovery() -> StrategyBuilder:
    b = StrategyBuilder()
    b.strategy_preset = "volume_breakout_52w"
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout"
    return b


# ── handle_pending_tie_break ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pending_tie_break_resolves_with_numeric_reply():
    b = _builder_with_pending_tie_break()
    step = await handle_pending_tie_break(b, "1")
    assert step.handled is True
    assert step.symbol_resolved is True
    assert b.symbol == "HDFCBANK.NS"
    assert b.discovery_pending is False
    assert "HDFC Bank" in step.assistant_text
    assert step.assistant_state == "collect_user_input"


@pytest.mark.asyncio
async def test_pending_tie_break_resolves_with_method_id():
    b = _builder_with_pending_tie_break()
    step = await handle_pending_tie_break(b, "highest_relative_volume")
    assert step.symbol_resolved is True
    assert b.symbol == "HDFCBANK.NS"   # highest rel-vol (3.2 > 2.1)


@pytest.mark.asyncio
async def test_pending_tie_break_with_unparseable_reply_re_prompts():
    """An unrecognised reply should re-emit the choice prompt and KEEP the
    builder in pending state so the next user message gets another chance."""
    b = _builder_with_pending_tie_break()
    step = await handle_pending_tie_break(b, "what is this")
    assert step.handled is True
    assert step.symbol_resolved is False
    assert b.discovery_pending is True       # still waiting
    assert b.symbol is None
    assert step.assistant_state == "discovery_pending"
    # Re-prompt should mention the parser error message
    assert "didn't recognise" in step.assistant_text or "didn't recognize" in step.assistant_text


@pytest.mark.asyncio
async def test_pending_tie_break_with_empty_reply_reshows_prompt():
    """Empty / whitespace user reply → re-show the same prompt."""
    b = _builder_with_pending_tie_break()
    step = await handle_pending_tie_break(b, "   ")
    assert step.handled is True
    assert step.symbol_resolved is False
    assert b.discovery_pending is True
    # Re-shows the candidates + options prompt
    assert "HDFC Bank" in step.assistant_text
    assert "Highest relative volume" in step.assistant_text


@pytest.mark.asyncio
async def test_pending_tie_break_returns_unhandled_when_not_pending():
    """If the builder isn't waiting for a tie-break, the helper does
    nothing — caller should fall through to normal flow."""
    b = StrategyBuilder()
    step = await handle_pending_tie_break(b, "1")
    assert step.handled is False
    assert step.symbol_resolved is False


# ── maybe_dispatch_discovery ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_discovery_skips_when_no_preset():
    b = StrategyBuilder()         # no strategy_preset
    step = await maybe_dispatch_discovery(b)
    assert step.handled is False


@pytest.mark.asyncio
async def test_dispatch_discovery_skips_when_user_set_a_symbol():
    b = _builder_for_fresh_discovery()
    b.symbol = "RELIANCE.NS"      # user pinned a stock manually
    step = await maybe_dispatch_discovery(b)
    assert step.handled is False
    # The helper should NOT have run a scan — symbol is already set.


@pytest.mark.asyncio
async def test_dispatch_discovery_skips_while_already_pending():
    b = _builder_with_pending_tie_break()
    step = await maybe_dispatch_discovery(b)
    assert step.handled is False    # tie-break handler owns this case


@pytest.mark.asyncio
async def test_dispatch_discovery_short_circuits_on_persistent_no_match():
    """If a previous scan already returned 'none' and the user hasn't
    changed the criteria, re-emit the no-match message without burning
    the universe."""
    b = _builder_for_fresh_discovery()
    b.discovery_no_match = True
    step = await maybe_dispatch_discovery(b)
    assert step.handled is True
    assert step.symbol_resolved is False
    assert step.assistant_state == "discovery_no_match"
    assert "loosen" in step.assistant_text.lower() or "different" in step.assistant_text.lower()


@pytest.mark.asyncio
async def test_dispatch_discovery_single_match_sets_symbol_and_emits_confirmation():
    b = _builder_for_fresh_discovery()

    fake_result = ScanResult(
        status="single",
        candidates=[Candidate(symbol="HDFCBANK.NS", display_name="HDFC Bank",
                              sector="banking", metrics={})],
        tie_break_options=[],
        asof_iso="2026-05-12T09:30Z",
    )

    async def fake_run_discovery(builder, fetch_ohlcv=None):
        builder.discovered_symbol = "HDFCBANK.NS"
        builder.symbol = "HDFCBANK.NS"
        return fake_result

    with patch("app.services.discovery.chat_integration.run_discovery",
               new=AsyncMock(side_effect=fake_run_discovery)):
        step = await maybe_dispatch_discovery(b)

    assert step.handled is True
    assert step.symbol_resolved is True
    assert b.symbol == "HDFCBANK.NS"
    assert "HDFC Bank" in step.assistant_text


@pytest.mark.asyncio
async def test_dispatch_discovery_multiple_match_emits_choice_prompt():
    b = _builder_for_fresh_discovery()

    fake_result = ScanResult(
        status="multiple",
        candidates=[
            Candidate(symbol="HDFCBANK.NS", display_name="HDFC Bank", sector="banking",
                      metrics={"relative_volume": 3.2}),
            Candidate(symbol="RELIANCE.NS", display_name="Reliance Industries", sector="energy",
                      metrics={"relative_volume": 2.1}),
        ],
        tie_break_options=[
            TieBreakOption(method="highest_relative_volume", label="Highest relative volume"),
            TieBreakOption(method="closest_to_52w_high",     label="Closest to 52-week high"),
        ],
    )

    async def fake_run_discovery(builder, fetch_ohlcv=None):
        builder.discovery_pending = True
        builder.discovery_candidates = [c.to_dict() for c in fake_result.candidates]
        builder.discovery_tie_break_options = [o.model_dump() for o in fake_result.tie_break_options]
        return fake_result

    with patch("app.services.discovery.chat_integration.run_discovery",
               new=AsyncMock(side_effect=fake_run_discovery)):
        step = await maybe_dispatch_discovery(b)

    assert step.handled is True
    assert step.symbol_resolved is False
    assert b.symbol is None
    assert b.discovery_pending is True
    assert step.assistant_state == "discovery_pending"
    assert "HDFC Bank" in step.assistant_text
    assert "Highest relative volume" in step.assistant_text


@pytest.mark.asyncio
async def test_dispatch_discovery_no_match_emits_relax_message():
    b = _builder_for_fresh_discovery()

    fake_result = ScanResult(
        status="none",
        candidates=[],
        tie_break_options=[],
    )

    async def fake_run_discovery(builder, fetch_ohlcv=None):
        builder.discovery_no_match = True
        return fake_result

    with patch("app.services.discovery.chat_integration.run_discovery",
               new=AsyncMock(side_effect=fake_run_discovery)):
        step = await maybe_dispatch_discovery(b)

    assert step.handled is True
    assert step.symbol_resolved is False
    assert step.assistant_state == "discovery_no_match"


@pytest.mark.asyncio
async def test_dispatch_discovery_propagates_orchestrator_skip_as_unhandled():
    """When run_discovery returns None (e.g. preset has no discovery block
    or timeframe isn't set), the helper should report unhandled so the
    caller continues with normal flow."""
    b = _builder_for_fresh_discovery()

    with patch("app.services.discovery.chat_integration.run_discovery",
               new=AsyncMock(return_value=None)):
        step = await maybe_dispatch_discovery(b)

    assert step.handled is False
    assert step.symbol_resolved is False
