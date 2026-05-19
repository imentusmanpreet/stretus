"""Phase 9 — orchestrator + builder integration + reply-builders.

Verifies:
  - run_discovery writes the right state onto the builder for each status
  - resolve_tie_break parses + applies user replies
  - is_user_input_complete treats symbol as optional under discovery
  - reply-builders produce sensible user-facing text for each case
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

from app.services.chat.strategy_flow import (
    build_discovery_choice_prompt,
    build_discovery_no_match_reply,
    build_discovery_resolved_reply,
)
from app.services.discovery.orchestrator import resolve_tie_break, run_discovery
from app.services.strategy.builder import StrategyBuilder


def _bar(ts, c, v):
    return {"timestamp": ts.isoformat().replace("+00:00", "Z"),
            "open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": v}


def _bars(n=300, vol_mult_last=1.0):
    rows = []
    base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
    for i in range(n):
        ts = base - pd.Timedelta(days=(n - 1 - i))
        close = 100.0 + i * 0.05
        vol = 100_000.0 if i < n - 1 else 100_000.0 * vol_mult_last
        rows.append(_bar(ts, close, vol))
    return rows


def _builder_for_discovery(preset_name: str = "volume_breakout_52w") -> StrategyBuilder:
    b = StrategyBuilder()
    b.strategy_preset = preset_name
    b.timeframe = "1d"          # use 1d so we don't have to fabricate intraday bars
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout 52w"
    return b


# ── builder.is_user_input_complete with discovery ────────────────────────────


def test_builder_treats_symbol_as_optional_under_discovery():
    """When the preset declares discovery and no symbol is set, the input
    is considered complete (the scanner fills it in later)."""
    b = _builder_for_discovery()
    assert b.symbol is None
    assert b.requires_discovery() is True
    assert b.is_user_input_complete() is True
    assert "symbol" not in b.missing_user_input_fields()


def test_builder_requires_symbol_when_preset_has_no_discovery():
    """Regression: classic presets still require a symbol."""
    b = _builder_for_discovery(preset_name="orb")        # ORB has no discovery
    assert b.requires_discovery() is False
    assert b.is_user_input_complete() is False
    assert "symbol" in b.missing_user_input_fields()


def test_builder_treats_user_supplied_symbol_as_winning():
    b = _builder_for_discovery()
    b.symbol = "RELIANCE.NS"
    assert b.requires_discovery() is False
    assert b.is_user_input_complete() is True


# ── run_discovery state side effects ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_discovery_single_match_sets_builder_symbol():
    target = "HDFCBANK.NS"

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _bars(n=300, vol_mult_last=3.0 if symbol == target else 1.0)

    b = _builder_for_discovery()
    result = await run_discovery(b, fetch_ohlcv=stub_fetcher,
                                 asof_utc=datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc))
    assert result.status == "single"
    assert b.discovered_symbol == target
    assert b.symbol == target
    assert b.discovery_pending is False


@pytest.mark.asyncio
async def test_run_discovery_no_match_sets_flag():
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _bars(n=300, vol_mult_last=1.0)

    b = _builder_for_discovery()
    result = await run_discovery(b, fetch_ohlcv=stub_fetcher)
    assert result.status == "none"
    assert b.discovery_no_match is True
    assert b.symbol is None
    assert b.discovery_pending is False


@pytest.mark.asyncio
async def test_run_discovery_multiple_match_stores_candidates_and_options():
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _bars(n=300, vol_mult_last=3.0)

    b = _builder_for_discovery()
    result = await run_discovery(b, fetch_ohlcv=stub_fetcher)
    assert result.status == "multiple"
    assert b.discovery_pending is True
    assert b.discovery_candidates is not None
    assert len(b.discovery_candidates) > 1
    assert b.discovery_tie_break_options is not None
    assert b.symbol is None    # not picked yet — awaiting user choice
    # Tie-break options came from the preset's declaration.
    method_ids = {o["method"] for o in b.discovery_tie_break_options}
    assert "highest_relative_volume" in method_ids


@pytest.mark.asyncio
async def test_run_discovery_skips_when_preset_has_no_discovery():
    b = _builder_for_discovery(preset_name="orb")
    result = await run_discovery(b, fetch_ohlcv=None)
    assert result is None


@pytest.mark.asyncio
async def test_run_discovery_skips_when_timeframe_unset():
    b = _builder_for_discovery()
    b.timeframe = None
    result = await run_discovery(b, fetch_ohlcv=None)
    assert result is None


# ── resolve_tie_break ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_tie_break_with_numeric_reply_sets_symbol():
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _bars(n=300, vol_mult_last=3.0)

    b = _builder_for_discovery()
    await run_discovery(b, fetch_ohlcv=stub_fetcher)
    assert b.discovery_pending is True

    ok, msg = resolve_tie_break(b, "1")    # picks the first tie-break option
    assert ok is True
    assert b.symbol is not None
    assert b.discovery_pending is False
    assert b.discovery_chosen_method is not None
    assert b.discovery_candidates is None        # cleared after resolution


@pytest.mark.asyncio
async def test_resolve_tie_break_with_method_id_works():
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _bars(n=300, vol_mult_last=3.0)

    b = _builder_for_discovery()
    await run_discovery(b, fetch_ohlcv=stub_fetcher)
    ok, _ = resolve_tie_break(b, "highest_relative_volume")
    assert ok is True
    assert b.discovery_chosen_method == "highest_relative_volume"


@pytest.mark.asyncio
async def test_resolve_tie_break_with_unrecognised_reply_reprompts():
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _bars(n=300, vol_mult_last=3.0)

    b = _builder_for_discovery()
    await run_discovery(b, fetch_ohlcv=stub_fetcher)
    ok, msg = resolve_tie_break(b, "completely unrelated garbage")
    assert ok is False
    # Discovery state stays pending so the chat can re-ask.
    assert b.discovery_pending is True
    assert "didn't recognise" in msg or "didn't recognize" in msg


def test_resolve_tie_break_when_not_pending_is_a_noop():
    b = _builder_for_discovery()
    # Never ran discovery
    ok, msg = resolve_tie_break(b, "1")
    assert ok is False
    assert "no pending" in msg.lower()


# ── Reply-builders ───────────────────────────────────────────────────────────


def test_build_discovery_no_match_reply_suggests_relaxing_criteria():
    msg = build_discovery_no_match_reply()
    assert "No" in msg or "no" in msg
    # Should suggest at least one path forward
    assert any(hint in msg.lower() for hint in ("loosen", "different", "manual"))


def test_build_discovery_choice_prompt_lists_candidates_and_options():
    scan_result = {
        "status": "multiple",
        "candidates": [
            {"symbol": "HDFCBANK.NS", "display_name": "HDFC Bank", "sector": "banking",
             "metrics": {"relative_volume": 3.2, "distance_to_52w_high_pct": 0.8, "rsi_14": 68.0}},
            {"symbol": "RELIANCE.NS", "display_name": "Reliance Industries", "sector": "energy",
             "metrics": {"relative_volume": 2.1, "distance_to_52w_high_pct": 1.5, "rsi_14": 61.0}},
        ],
        "tie_break_options": [
            {"method": "highest_relative_volume", "label": "Highest relative volume",
             "description": "Pick the stock with the strongest volume vs 20-bar average"},
            {"method": "closest_to_52w_high", "label": "Closest to 52-week high",
             "description": "Pick the stock whose close is nearest its 52-week high"},
        ],
    }
    msg = build_discovery_choice_prompt(scan_result)
    # Both candidates listed
    assert "HDFC Bank" in msg
    assert "Reliance" in msg
    # Both options listed with numeric prefixes
    assert "1. Highest relative volume" in msg
    assert "2. Closest to 52-week high" in msg
    # In-line metric context surfaced
    assert "vol×3.2" in msg or "vol×3.20" in msg


def test_build_discovery_resolved_reply_includes_symbol_and_method():
    msg = build_discovery_resolved_reply("HDFCBANK.NS", "HDFC Bank", "highest_relative_volume")
    assert "HDFC Bank" in msg
    assert "HDFCBANK.NS" in msg
    assert "highest relative volume" in msg.lower()


def test_build_discovery_choice_prompt_with_no_options_falls_back():
    """Defensive: degrade gracefully if a preset somehow has zero tie-break options."""
    scan_result = {"candidates": [{"symbol": "A.NS", "display_name": "A", "sector": "x",
                                    "metrics": {}}],
                   "tie_break_options": []}
    msg = build_discovery_choice_prompt(scan_result)
    assert "manually" in msg.lower()
