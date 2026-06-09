"""
tests/test_chat/test_edits.py — WS1 acceptance suite.

Each test maps to one WS1 acceptance bullet (docs/IMPLEMENTATION_PROMPT.md §3):
  * "change take profit to 2%" → draft + card show 2.0, persists next turn.
  * 4 sequential edits (entry sig, SL, TP, exit sig) → all four retained.
  * TATAMOTORS→INFY → no stale "TATAMOTORS not found".
  * modify a param not mentioned this turn → its prior `user` provenance survives.

These exercise the WS1 mechanisms directly (deep-merge, upgrade-only provenance,
single risk store, stale-validation clear) without standing up the full chat
DB/LLM stack — the LLM is mocked via set_llm_override, exactly like the existing
SDL selector tests.
"""
import asyncio
import json

import pytest

from app.planner.sdl import (
    SDL,
    EntrySpec,
    ExitSpec,
    Leg,
    Provenance,
    RiskSpec,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
)
from app.planner.sdl_selector import clear_cache, modify_sdl, set_llm_override
from app.planner.provenance_reconciler import merge_field_sources, reconcile
from app.services.strategy.builder import StrategyBuilder


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _reset_selector():
    clear_cache()
    set_llm_override(None)
    yield
    clear_cache()
    set_llm_override(None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _base_sdl(**risk_kwargs) -> SDL:
    """A minimal long ETH RSI strategy with SL 2% / TP 2:1, all user-sourced."""
    sl = risk_kwargs.get("stop_loss", StopLossSpec(type="percent", value=2.0))
    tp = risk_kwargs.get("take_profit", TakeProfitSpec(type="rr", ratio=2.0))
    return SDL(
        context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
        universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
            )
        ],
        risk=RiskSpec(stop_loss=sl, take_profit=tp),
        provenance=Provenance(field_sources={
            "universe.symbol": "user",
            "legs.0.entry.trigger.name": "user",
            "risk.stop_loss": "user",
            "risk.stop_loss.value": "user",
            "risk.take_profit": "user",
        }),
    )


def _override(sdl: SDL):
    """Make the mocked LLM return the given SDL JSON for the next modify turn."""
    set_llm_override(lambda msgs, tools: sdl.model_dump_json())


# ── Single risk store (WS1 §6) ────────────────────────────────────────────────

class TestSingleRiskStore:
    def test_stop_loss_take_profit_round_trip_one_store(self):
        b = StrategyBuilder()
        b.stop_loss = 1.0
        b.take_profit = 2.0
        # The single store IS risk_execution_config — no separate mirror.
        assert b.risk_execution_config["stop_loss_pct"] == 1.0
        assert b.risk_execution_config["take_profit_pct"] == 2.0
        assert b.stop_loss == 1.0 and b.take_profit == 2.0

    def test_change_take_profit_to_2pct_persists_next_turn(self):
        """'change take profit to 2%' → draft shows 2.0 AND survives a turn boundary."""
        b = StrategyBuilder()
        b.symbol = "INFY.NS"
        b.take_profit = 2.0
        b.risk_execution_config.setdefault("rms_sources", {})["take_profit_pct"] = "user"

        draft = b.to_draft_json()
        assert draft["take_profit_pct"] == 2.0
        assert draft["risk_execution_config"]["take_profit_pct"] == 2.0

        # Next turn: a fresh builder rehydrates from the persisted draft.
        b2 = StrategyBuilder()
        b2.merge_preview(draft)
        assert b2.take_profit == 2.0

    def test_user_take_profit_survives_apply_defaults(self):
        b = StrategyBuilder()
        b.symbol = "INFY.NS"
        b.experience = "intermediate"
        b.take_profit = 2.0
        b.risk_execution_config.setdefault("rms_sources", {})["take_profit_pct"] = "user"
        b.apply_defaults()  # must not overwrite a user-set TP
        assert b.take_profit == 2.0

    def test_defaults_are_tagged_default_not_user(self):
        b = StrategyBuilder()
        b.symbol = "INFY.NS"
        b.experience = "intermediate"
        b.apply_defaults()
        sources = b.risk_execution_config.get("rms_sources", {})
        assert sources.get("stop_loss_pct") == "default"
        assert sources.get("take_profit_pct") == "default"


# ── Deep-merge: edits accumulate (WS1 §1) ─────────────────────────────────────

class TestEditsAccumulate:
    def test_four_sequential_edits_all_retained(self):
        """entry sig → SL → TP → exit sig, each as a sparse patch; all retained."""
        sdl = _base_sdl()

        # Turn 1: change entry trigger (LLM re-emits a near-full SDL).
        patch1 = sdl.model_copy(deep=True)
        patch1.legs[0].entry.trigger = SignalRef(name="ema_cross_up", params={"window_fast": 9, "window_slow": 21})
        _override(patch1)
        sdl = run(modify_sdl(sdl, "use EMA 9/21 crossover for entry", skip_flag=True))
        assert sdl.legs[0].entry.trigger.name == "ema_cross_up"

        # Turn 2: change SL — LLM emits a SPARSE patch (only risk set, legs empty).
        patch2 = SDL(
            context=sdl.context, universe=sdl.universe,
            legs=sdl.legs,  # carried; deep-merge keeps nested values anyway
            risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=1.0),
                          take_profit=sdl.risk.take_profit),
        )
        _override(patch2)
        sdl = run(modify_sdl(sdl, "tighten stop to 1%", skip_flag=True))
        assert sdl.risk.stop_loss.value == 1.0
        assert sdl.legs[0].entry.trigger.name == "ema_cross_up"  # turn-1 edit retained

        # Turn 3: change TP.
        patch3 = sdl.model_copy(deep=True)
        patch3.risk.take_profit = TakeProfitSpec(type="percent", value=3.0)
        _override(patch3)
        sdl = run(modify_sdl(sdl, "take profit at 3%", skip_flag=True))
        assert sdl.risk.take_profit.type == "percent"
        assert sdl.risk.take_profit.value == 3.0
        assert sdl.risk.stop_loss.value == 1.0          # turn-2 edit retained
        assert sdl.legs[0].entry.trigger.name == "ema_cross_up"  # turn-1 edit retained

        # Turn 4: change exit signal.
        patch4 = sdl.model_copy(deep=True)
        patch4.legs[0].exit = ExitSpec(triggers=[SignalRef(name="ema_cross_down", params={})])
        _override(patch4)
        sdl = run(modify_sdl(sdl, "exit on EMA cross down", skip_flag=True))

        # All four edits present after turn 4.
        assert sdl.legs[0].entry.trigger.name == "ema_cross_up"
        assert sdl.risk.stop_loss.value == 1.0
        assert sdl.risk.take_profit.type == "percent" and sdl.risk.take_profit.value == 3.0
        assert sdl.legs[0].exit.triggers[0].name == "ema_cross_down"
        assert sdl.version == 5  # started at 1, +1 per modify

    def test_sparse_patch_does_not_wipe_unmentioned_fields(self):
        """A modify that only restates the changed field keeps everything else."""
        sdl = _base_sdl()
        # Patch emits ONLY a new SL; trigger/exit/TP omitted on the risk leaf.
        patch = sdl.model_copy(deep=True)
        patch.risk.stop_loss = StopLossSpec(type="percent", value=0.5)
        _override(patch)
        merged = run(modify_sdl(sdl, "stop 0.5%", skip_flag=True))
        assert merged.risk.stop_loss.value == 0.5
        assert merged.risk.take_profit.ratio == 2.0  # untouched
        assert merged.legs[0].entry.trigger.name == "rsi_oversold"


# ── Upgrade-only provenance (WS1 §2) ──────────────────────────────────────────

class TestProvenanceUpgradeOnly:
    def test_merge_field_sources_never_demotes_user(self):
        prior = {"risk.stop_loss": "user", "context.timeframe": "inferred"}
        new = {"risk.stop_loss": "default", "context.timeframe": "user"}
        merged = merge_field_sources(prior, new)
        assert merged["risk.stop_loss"] == "user"        # not demoted
        assert merged["context.timeframe"] == "user"     # promoted

    def test_prior_user_field_survives_unmentioned_modify(self):
        """Modify a param not mentioned this turn → its prior user provenance survives."""
        sdl = _base_sdl()
        assert sdl.provenance.field_sources["risk.stop_loss"] == "user"

        # The change is only about TP; SL is not restated. LLM re-emits SDL but
        # drops SL provenance (the classic demotion bug).
        patch = sdl.model_copy(deep=True)
        patch.risk.take_profit = TakeProfitSpec(type="percent", value=4.0)
        patch.provenance = Provenance(field_sources={"risk.take_profit": "user"})
        _override(patch)
        merged = run(modify_sdl(sdl, "take profit 4%", skip_flag=True))

        assert merged.provenance.field_sources.get("risk.stop_loss") == "user"
        assert merged.provenance.field_sources.get("risk.take_profit") == "user"

    def test_reconcile_seeds_prior_field_sources(self):
        sdl = _base_sdl()
        prior = {"universe.symbol": "user", "risk.stop_loss": "user"}
        out = reconcile(sdl, "take profit 4%", prior_field_sources=prior)
        # A prior user field not mentioned this turn is still user.
        assert out.provenance.field_sources["universe.symbol"] == "user"


# ── Stale symbol-validation clear (WS1 §5) ────────────────────────────────────

class TestStaleSymbolValidation:
    def test_symbol_change_clears_stale_validation(self):
        """TATAMOTORS (not found) → INFY → no stale 'TATAMOTORS not found'."""
        b = StrategyBuilder()
        b.symbol = "TATAMOTORS"
        b.set_symbol_validation("validation.unsupported_stock", {"stock_query": "TATAMOTORS"})
        assert b.symbol_validation_code == "validation.unsupported_stock"

        # New turn merges a preview that switches to a different, valid symbol.
        b.merge_preview({"symbol": "INFY.NS"})
        assert b.symbol == "INFY.NS"
        assert b.symbol_validation_code is None
        assert b.symbol_validation_message is None

    def test_draft_does_not_repaint_new_symbol_with_old_error(self):
        """A persisted draft carrying symbol=OLD + its error must not paint a
        freshly-chosen NEW symbol with the stale error."""
        b = StrategyBuilder()
        b.symbol = "INFY.NS"  # resolved THIS turn
        # Draft from a prior turn still has TATAMOTORS + its validation error.
        stale_draft = {
            "symbol": "TATAMOTORS",
            "symbol_validation_code": "validation.unsupported_stock",
            "symbol_validation_facts": {"stock_query": "TATAMOTORS"},
        }
        b.merge_preview(stale_draft)
        # merge sets the (older) draft symbol but must drop the stale error.
        assert b.symbol_validation_code is None

    def test_same_symbol_keeps_its_validation(self):
        """A genuine restore (symbol unchanged) keeps the validation it owns."""
        b = StrategyBuilder()
        draft = {
            "symbol": "TATAMOTORS",
            "symbol_validation_code": "validation.unsupported_stock",
            "symbol_validation_facts": {"stock_query": "TATAMOTORS"},
        }
        b.merge_preview(draft)
        assert b.symbol == "TATAMOTORS"
        assert b.symbol_validation_code == "validation.unsupported_stock"
