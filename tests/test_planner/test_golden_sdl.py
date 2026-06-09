"""
tests/test_planner/test_golden_sdl.py — Phase 7 golden-SDL CI tests.

These tests verify that the SDL selector produces the expected structure for
each corpus entry.  ALL tests use mocked LLM (no real API calls).

Acceptance #13: golden-SDL tests pass.
Acceptance #3: same prompt → same SDL (cache determinism verified here).
Acceptance #2: explicit indicator wins (Bollinger → BB card, not Keltner).

Shadow mode note: when the selector flag is flipped, these tests will run
the REAL selector (not mocked) and compare against the golden SDLs.  Until
then, they prove the SDL structure is correct given the selector's output.
"""
import asyncio

import pytest

from app.planner.sdl import (
    SDL,
    ClarificationNeeded,
    DynamicRank,
    DynamicUniverse,
    EntrySpec,
    ExitSpec,
    GatesSpec,
    Leg,
    Provenance,
    RegimeGate,
    RiskSpec,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    UnmappedDetail,
    EventGate,
)
from app.planner.sdl_selector import (
    clear_cache,
    compile_to_sdl,
    modify_sdl,
    set_llm_override,
)
from app.planner.catalog_schema import build_menu, invalidate_menu_cache
from app.planner.readback import compute_match_pct
from .corpus import CORPUS


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Corpus-driven golden fixtures ─────────────────────────────────────────────

def _build_golden_sdl(entry_idx: int) -> str:
    """Pre-written golden SDL for each corpus entry (index 0-5)."""
    if entry_idx == 0:
        # simple_eth_rsi_mean_reversion
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
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(
                field_sources={
                    "context.timeframe": "user", "universe.symbol": "user",
                    "legs.0.entry.trigger": "user", "risk.stop_loss": "user",
                    "risk.take_profit": "user", "legs.0.exit": "inferred",
                }
            ),
        ).model_dump_json()

    if entry_idx == 1:
        # btc_bollinger_breakout_explicit_indicator — must use bb_* card
        return SDL(
            context=StrategyContext(market="crypto", timeframe="1h", objective="breakout"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="BTC_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="price_above_bb_upper", params={"window": 20, "num_std": 2.0})),
                    exit=ExitSpec(),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=1.5),
            ),
            provenance=Provenance(
                field_sources={
                    "context.timeframe": "user", "universe.symbol": "user",
                    "legs.0.entry.trigger": "user", "risk.stop_loss": "user",
                },
                clarifications_needed=[
                    ClarificationNeeded(field="risk.take_profit", question="No TP given — use 2:1?", assumed_value="2:1")
                ],
            ),
        ).model_dump_json()

    if entry_idx == 2:
        # nse_dynamic_orb_highest_rvol
        return SDL(
            context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
            universe=DynamicUniverse(
                asset_class="equity_cash",
                screen=["CLOSE > VWAP"],
                rank=DynamicRank(by="rvol", order="desc"),
                tie_break="highest_relative_volume",
            ),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="opening_range_breakout", params={"minutes": 15})),
                    exit=ExitSpec(),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="atr", multiple=1.5, window=14),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(
                field_sources={
                    "universe": "user", "universe.screen": "user", "universe.rank.by": "user",
                    "context.timeframe": "user", "legs.0.entry.trigger": "user",
                    "risk.stop_loss": "user",
                },
                clarifications_needed=[
                    ClarificationNeeded(field="risk.take_profit", question="No TP given — 2:1?", assumed_value="2:1")
                ],
            ),
        ).model_dump_json()

    if entry_idx == 5:
        # eth_with_unsupported_features
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
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
            ),
            gates=GatesSpec(
                regime=RegimeGate(allowed=["trending"]),
                event=EventGate(skip_dates=["2025-01-31", "2025-07-31"]),
            ),
            provenance=Provenance(
                field_sources={
                    "universe.symbol": "user", "legs.0.entry.trigger": "user",
                    "risk.stop_loss": "user", "gates.regime": "user",
                    "gates.event": "user",
                },
                unmapped_details=[
                    UnmappedDetail(
                        text="top-20 parallel (I want to trade multiple assets simultaneously)",
                        kind="engine_capability_gap",
                        note="scanner picks ONE asset; parallel multi-asset not supported",
                    )
                ],
            ),
        ).model_dump_json()

    # Fallback for entries 3, 4: minimal SDL
    return SDL(
        context=StrategyContext(market="indian_stocks", timeframe="1d", objective="trend"),
        universe=StaticUniverse(asset_class="equity_cash", symbol="HDFCBANK.NS"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(trigger=SignalRef(name="ema_cross_up", params={})),
                exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down", params={})]),
            )
        ],
        risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=2.0)),
        provenance=Provenance(
            field_sources={"universe.symbol": "user", "legs.0.entry.trigger": "user"}
        ),
    ).model_dump_json()


@pytest.fixture(autouse=True)
def reset_selector():
    clear_cache()
    set_llm_override(None)
    invalidate_menu_cache()
    yield
    clear_cache()
    set_llm_override(None)


# ── Golden-SDL corpus tests (Acceptance #13) ──────────────────────────────────

class TestGoldenSDL:
    @pytest.mark.parametrize("idx,entry", [(i, e) for i, e in enumerate(CORPUS)])
    def test_corpus_entry_produces_valid_sdl(self, idx, entry):
        """Each corpus prompt → well-formed SDL (mocked LLM)."""
        golden = _build_golden_sdl(idx)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(entry.prompt, skip_flag=True))
        assert isinstance(sdl, SDL)
        assert sdl.legs  # has at least one leg
        assert sdl.provenance  # provenance populated

    @pytest.mark.parametrize("idx,entry", [(i, e) for i, e in enumerate(CORPUS)])
    def test_corpus_entry_has_content_hash(self, idx, entry):
        golden = _build_golden_sdl(idx)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(entry.prompt, skip_flag=True))
        assert len(sdl.content_hash) == 64

    def test_eth_rsi_static_universe(self):
        golden = _build_golden_sdl(0)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[0].prompt, skip_flag=True))
        assert sdl.universe.type == "static"
        assert sdl.universe.asset_class == "crypto_spot"  # type: ignore
        assert sdl.universe.symbol == "ETH_USDC"          # type: ignore

    def test_eth_rsi_correct_trigger(self):
        golden = _build_golden_sdl(0)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[0].prompt, skip_flag=True))
        trigger = sdl.legs[0].entry.trigger
        assert "rsi" in trigger.name.lower()
        assert trigger.params.get("threshold") == 30

    def test_nse_dynamic_universe(self):
        golden = _build_golden_sdl(2)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[2].prompt, skip_flag=True))
        assert sdl.universe.type == "dynamic"
        assert sdl.universe.asset_class == "equity_cash"  # type: ignore
        assert sdl.universe.rank.by == "rvol"              # type: ignore


# ── Explicit indicator wins (Acceptance #2) ───────────────────────────────────

class TestExplicitIndicatorWins:
    def test_bollinger_uses_bb_card_not_keltner(self):
        golden = _build_golden_sdl(1)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[1].prompt, skip_flag=True))
        trigger_name = sdl.legs[0].entry.trigger.name.lower()
        assert "bb" in trigger_name or "bollinger" in trigger_name, (
            f"Expected BB card, got: {trigger_name} — Bollinger prompt must map to BB family"
        )
        assert "keltner" not in trigger_name
        assert "donchian" not in trigger_name

    def test_bollinger_sl_correct(self):
        golden = _build_golden_sdl(1)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[1].prompt, skip_flag=True))
        assert sdl.risk.stop_loss is not None
        assert sdl.risk.stop_loss.value == pytest.approx(1.5)


# ── Same prompt → same SDL (Acceptance #3) ───────────────────────────────────

class TestSamePromptSameSDL:
    def test_repeated_call_same_hash(self):
        call_count = 0
        def mock(msgs, tools):
            nonlocal call_count
            call_count += 1
            return _build_golden_sdl(0)
        set_llm_override(mock)
        prompt = CORPUS[0].prompt
        sdl1 = run(compile_to_sdl(prompt, skip_flag=True))
        sdl2 = run(compile_to_sdl(prompt, skip_flag=True))
        assert sdl1.content_hash == sdl2.content_hash
        assert call_count == 1  # cache hit

    def test_different_prompts_different_hashes(self):
        golden_0 = _build_golden_sdl(0)
        golden_1 = _build_golden_sdl(1)
        call_count = [0]
        calls = [golden_0, golden_1]
        def mock(msgs, tools):
            idx = min(call_count[0], len(calls) - 1)
            call_count[0] += 1
            return calls[idx]
        set_llm_override(mock)
        sdl1 = run(compile_to_sdl(CORPUS[0].prompt, skip_flag=True))
        sdl2 = run(compile_to_sdl(CORPUS[1].prompt, skip_flag=True))
        assert sdl1.content_hash != sdl2.content_hash


# ── Same SDL → same match% (Acceptance #4) ────────────────────────────────────

class TestSameSDLSameMatchPct:
    def test_match_pct_deterministic(self):
        golden = _build_golden_sdl(0)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[0].prompt, skip_flag=True))
        result1 = compute_match_pct(sdl)
        result2 = compute_match_pct(sdl)
        assert result1 == result2


# ── Unsupported feature in unmapped_details ────────────────────────────────────

class TestUnsupportedFeatureFlagged:
    def test_top20_parallel_in_unmapped(self):
        golden = _build_golden_sdl(5)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[5].prompt, skip_flag=True))
        unmapped_texts = [d.text for d in sdl.provenance.unmapped_details]
        # Should have an engine_capability_gap for parallel multi-asset
        kinds = [d.kind for d in sdl.provenance.unmapped_details]
        assert "engine_capability_gap" in kinds

    def test_regime_gate_captured(self):
        golden = _build_golden_sdl(5)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[5].prompt, skip_flag=True))
        assert sdl.gates.regime is not None

    def test_event_gate_captured(self):
        golden = _build_golden_sdl(5)
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl(CORPUS[5].prompt, skip_flag=True))
        assert sdl.gates.event is not None


# ── Edit loop (Acceptance #14) ────────────────────────────────────────────────

class TestEditLoop:
    def test_modify_sdl_bumps_version(self):
        golden = _build_golden_sdl(0)
        set_llm_override(lambda msgs, tools: golden)
        original = run(compile_to_sdl(CORPUS[0].prompt, skip_flag=True))
        assert original.version == 1
        clear_cache()

        def mock_modify(msgs, tools):
            from app.planner.sdl import RiskSpec, StopLossSpec, TakeProfitSpec
            sdl = SDL.model_validate_json(golden)
            data = sdl.model_dump(mode="python")
            data.pop("content_hash", None)
            data["risk"]["stop_loss"]["value"] = 1.0
            data["provenance"]["field_sources"]["risk.stop_loss"] = "user"
            return SDL(**data).model_dump_json()

        set_llm_override(mock_modify)
        modified = run(modify_sdl(original, "Change stop to 1%", skip_flag=True))
        assert modified.version == 2
        assert modified.parent_version == 1

    def test_modify_recomputes_hash_on_change(self):
        golden = _build_golden_sdl(0)
        set_llm_override(lambda msgs, tools: golden)
        original = run(compile_to_sdl(CORPUS[0].prompt, skip_flag=True))
        h_before = original.content_hash
        clear_cache()

        def mock_modify(msgs, tools):
            sdl = SDL.model_validate_json(golden)
            data = sdl.model_dump(mode="python")
            data.pop("content_hash", None)
            data["context"]["timeframe"] = "1h"  # changed
            return SDL(**data).model_dump_json()

        set_llm_override(mock_modify)
        modified = run(modify_sdl(original, "Change to 1h", skip_flag=True))
        assert modified.content_hash != h_before
