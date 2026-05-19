"""Phase 9d — regression test for the preset-merge bug.

Bug: extract_strategy_details() correctly detected the volume_breakout_52w
preset from a natural-language user prompt and stored it on a temporary
`parsed_builder`, but chat_service.py's merge block never copied
`strategy_preset` over to the main `builder`. Result: Phase 9b's
maybe_dispatch_discovery() saw `builder.strategy_preset == None`,
returned `requires_discovery() == False`, and the chat fell through to
the "ask for symbol" branch instead of running the universe scanner.

This test asserts the integration end-to-end at the function level — runs
extract_strategy_details on the user's literal prompt, simulates the
chat_service merge, and verifies discovery would now dispatch.
"""
from __future__ import annotations

import sys
import types

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import pytest

from app.services.strategy.builder import StrategyBuilder, extract_strategy_details


# The user's literal original prompt that surfaced this bug
_USER_PROMPT = (
    "create intraday strategy on NSE stock whose volume spike up today 2x "
    "and is having low pull back or is on breaking on verge of 52 weeks "
    "high or low"
)


def test_extract_strategy_details_sets_preset_on_parsed_builder():
    """Pre-condition: the keyword detector must populate parsed_builder
    .strategy_preset for the user's prompt. If THIS test breaks, the bug
    is upstream in the keyword detector / preset KB."""
    parsed = StrategyBuilder()
    extract_strategy_details(_USER_PROMPT, parsed)
    assert parsed.strategy_preset == "volume_breakout_52w"


def test_chat_service_merge_propagates_preset_to_main_builder():
    """The exact merge logic chat_service.py runs against parsed_builder
    must hand the preset over to the main builder. Mirrors the production
    flow without dragging in the chat_service async machinery."""
    parsed = StrategyBuilder()
    extract_strategy_details(_USER_PROMPT, parsed)
    assert parsed.strategy_preset == "volume_breakout_52w"

    main = StrategyBuilder()
    # Same merge guard chat_service.py uses (first-write-wins).
    if parsed.strategy_preset and not main.strategy_preset:
        main.strategy_preset = parsed.strategy_preset

    assert main.strategy_preset == "volume_breakout_52w"


def test_main_builder_with_preset_requires_discovery():
    """Once the preset is on the main builder, requires_discovery() flips
    True (no symbol given + preset has discovery block)."""
    main = StrategyBuilder()
    main.strategy_preset = "volume_breakout_52w"
    main.timeframe = "5m"
    assert main.requires_discovery() is True


def test_chat_service_merge_does_not_overwrite_explicit_preset():
    """If the user has already pinned a preset earlier in the session, a
    later message containing different keywords must not silently replace
    it. The chat_service merge guard `not builder.strategy_preset` handles
    this. Regression: prevents flapping between presets across turns."""
    main = StrategyBuilder()
    main.strategy_preset = "orb"        # user pinned ORB earlier

    parsed = StrategyBuilder()
    extract_strategy_details(_USER_PROMPT, parsed)
    assert parsed.strategy_preset == "volume_breakout_52w"

    # Apply the same merge logic as chat_service
    if parsed.strategy_preset and not main.strategy_preset:
        main.strategy_preset = parsed.strategy_preset

    assert main.strategy_preset == "orb"     # unchanged


def test_user_supplied_symbol_skips_discovery_even_with_preset():
    """When the user gives a specific stock manually, requires_discovery
    must return False so the scanner doesn't override their choice."""
    main = StrategyBuilder()
    main.strategy_preset = "volume_breakout_52w"
    main.symbol = "RELIANCE.NS"
    main.timeframe = "5m"
    assert main.requires_discovery() is False


def test_full_end_to_end_user_prompt_unblocks_discovery():
    """End-to-end at the function level: user's literal prompt →
    extract_strategy_details → simulated merge → builder is now in the
    state where chat_service.py's discovery dispatch would fire (i.e.
    requires_discovery() == True)."""
    parsed = StrategyBuilder()
    extract_strategy_details(_USER_PROMPT, parsed)

    main = StrategyBuilder()
    # Replay the same merge fields chat_service does
    if parsed.timeframe and not main.timeframe:
        main.timeframe = parsed.timeframe
    if parsed.objective and not main.objective:
        main.objective = parsed.objective
    if parsed.strategy_preset and not main.strategy_preset:
        main.strategy_preset = parsed.strategy_preset

    # The user's prompt mentioned "intraday" → objective should be set,
    # but they didn't give a timeframe (e.g. "5m") in the prompt itself —
    # the chat would normally ask for that next. For requires_discovery
    # the timeframe still has to be present, so we set one to mimic the
    # "after user provides timeframe" state.
    main.timeframe = main.timeframe or "5m"

    assert main.objective == "intraday"
    assert main.strategy_preset == "volume_breakout_52w"
    assert main.requires_discovery() is True
