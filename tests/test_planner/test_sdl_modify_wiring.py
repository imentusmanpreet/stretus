"""SDL modify wiring: a strategy built via the SDL flow can be UPDATED via the
SDL modify path (edit the ticket), and the ticket survives across chat turns.

Before this, try_sdl_modify was dead code (never called, and the builder never
stored the SDL), so every "change RSI to 20" went through the legacy modifier.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.planner.sdl_selector import set_llm_override, clear_cache
from app.planner.sdl_flow import try_sdl_plan, try_sdl_modify
from app.services.strategy.builder import StrategyBuilder


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _sdl_json(threshold: int) -> str:
    return json.dumps({
        "context": {"market": "crypto", "timeframe": "15m", "objective": "mean_reversion"},
        "universe": {"type": "static", "asset_class": "crypto_spot", "symbol": "ETH_USDC"},
        "legs": [{
            "direction": "long",
            "entry": {"trigger": {"name": "rsi_oversold", "params": {"window": 14, "threshold": threshold}}, "filters": []},
            "exit": {"triggers": [{"name": "rsi_overbought", "params": {}}], "filters": []},
        }],
        "risk": {"stop_loss": {"type": "percent", "value": 2.0}, "take_profit": {"type": "rr", "ratio": 2.0}},
    })


@pytest.fixture(autouse=True)
def _reset():
    clear_cache()
    set_llm_override(None)
    yield
    clear_cache()
    set_llm_override(None)


def _create(builder, threshold=30):
    set_llm_override(lambda m, t: _sdl_json(threshold))
    return _run(try_sdl_plan("Buy ETH 15m when RSI below 30, SL 2%, TP 2:1", builder, force=True))


class TestSdlModifyWiring:
    def test_create_sets_builder_sdl(self):
        b = StrategyBuilder()
        r = _create(b)
        assert r.used_sdl
        assert b._sdl is not None
        assert "RSI(14) < 30" in b.entry_condition

    def test_sdl_persists_across_a_turn(self):
        b = StrategyBuilder()
        _create(b)
        draft = b.to_draft_json()
        assert draft.get("sdl_json") is not None
        # New builder (next turn) rehydrates from the draft.
        b2 = StrategyBuilder()
        b2.merge_preview(draft)
        assert b2._sdl is not None
        assert b2._sdl.content_hash == b._sdl.content_hash

    def test_modify_edits_the_existing_ticket(self):
        b = StrategyBuilder()
        _create(b, threshold=30)
        clear_cache()
        set_llm_override(lambda m, t: _sdl_json(20))   # user: "change RSI to 20"
        r = _run(try_sdl_modify("change the RSI threshold to 20", b))
        assert r.used_sdl
        assert r.validation_ok
        assert "RSI(14) < 20" in b.entry_condition
        # Version bumped → went through modify_sdl (merge), NOT a fresh compile.
        assert b._sdl.version == 2
        # The rest of the strategy is preserved (exit still rsi_overbought).
        assert b._sdl.legs[0].exit.triggers[0].name == "rsi_overbought"

    def test_modify_without_sdl_falls_back(self):
        # A legacy-built strategy (no _sdl) must defer to the legacy modifier.
        b = StrategyBuilder()
        r = _run(try_sdl_modify("change RSI to 20", b))
        assert r.used_sdl is False
        assert r.skip_reason == "no_existing_sdl"

    def test_modify_updates_signal_plan(self):
        b = StrategyBuilder()
        _create(b, threshold=30)
        clear_cache()
        set_llm_override(lambda m, t: _sdl_json(20))
        r = _run(try_sdl_modify("change RSI to 20", b))
        # The returned plan reflects the edit (used by the chat route to refresh
        # builder.signal_plan).
        assert "rsi_oversold" in (r.signal_plan.get("signals_used") or [])
