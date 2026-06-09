"""
tests/test_planner/test_gate.py — WS2 acceptance suite for the verification gate.

One assertion-group per tester (PDF) bug, each checking the COVERAGE buckets /
repaired list / clarification — never raw errors (design §5/§10).

The gate is pure (no LLM, no DB), so these build SDLs directly and call
strategy_gate.verify() with a SymbolContext.
"""
import pytest

from app.planner.sdl import (
    SDL,
    EntrySpec,
    ExitSpec,
    IntentComparison,
    IntentStub,
    Leg,
    Provenance,
    RiskSpec,
    SignalRef,
    SizingSpec,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    UnmappedDetail,
)
from app.planner.catalog_schema import build_menu
from app.planner.strategy_gate import SymbolContext, verify


@pytest.fixture(scope="module")
def menu():
    return build_menu()


INFY = SymbolContext(symbol="INFY.NS", asset_class="equity_cash", market="indian_stocks")


def _sdl(
    *,
    trigger: SignalRef,
    direction="long",
    filters=None,
    sl=None,
    tp=None,
    sizing=None,
    asset_class="crypto_spot",
    symbol="ETH_USDC",
    market="crypto",
    unmapped=None,
    field_sources=None,
):
    return SDL(
        context=StrategyContext(market=market, timeframe="15m", objective="mean_reversion"),
        universe=StaticUniverse(asset_class=asset_class, symbol=symbol),
        legs=[Leg(
            direction=direction,
            entry=EntrySpec(trigger=trigger, filters=filters or []),
            exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
        )],
        risk=RiskSpec(
            stop_loss=sl or StopLossSpec(type="percent", value=2.0),
            take_profit=tp or TakeProfitSpec(type="rr", ratio=2.0),
            sizing=sizing,
        ),
        provenance=Provenance(
            field_sources=field_sources or {"universe.symbol": "user"},
            unmapped_details=unmapped or [],
        ),
    )


# ── #8 — market/symbol reconciliation (wrapper≠code, crypto-for-INFY) ──────────

class TestMarketReconciliation:
    def test_symbol_overwrites_llm_asset_guess(self, menu):
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"), asset_class="crypto_spot", market="crypto")
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.sdl.universe.asset_class == "equity_cash"
        assert any(e["path"] == "universe.asset_class" and e["to"] == "equity_cash"
                   for e in r.coverage.repaired)

    def test_generic_prompt_market_symbol_asset_agree(self, menu):
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"), asset_class="crypto_spot", symbol="WRONG")
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.sdl.universe.symbol == "INFY.NS"
        assert r.sdl.universe.asset_class == "equity_cash"
        assert r.sdl.context.market == "indian_stocks"


# ── #13 — vwap_deviation stop never reaches the engine ─────────────────────────

class TestEngineContract:
    def test_vwap_deviation_stop_repaired_or_clarified(self, menu):
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"),
                   sl=StopLossSpec(type="vwap_deviation"))
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        # Never an unmappable stop downstream; either percent or a structural anchor.
        assert r.sdl.risk.stop_loss.type in ("percent", "atr")
        assert r.clarification is not None and r.clarification["field"] == "risk.stop_loss"
        assert any("vwap" in u["concept"] for u in r.coverage.unmapped)

    def test_structural_swing_low_mapped_to_engine_anchor(self, menu):
        from app.planner import engine_contract as ec
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"),
                   sl=StopLossSpec(type="swing_low"))
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.sdl.risk.stop_loss.anchor in ec.STOP_LOSS_ANCHORS


# ── #1 — contradictions: RSI>90∧<10, bullish∧bearish ───────────────────────────

class TestContradiction:
    def test_rsi_overbought_and_oversold_clarifies(self, menu):
        # "RSI>90 AND RSI<10" → overbought ∧ oversold (card-level contradiction).
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"),
                   filters=[SignalRef(name="rsi_overbought")])
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.clarification is not None
        assert "rsi_oversold" in r.clarification["question"] or "rsi_overbought" in r.clarification["question"]

    def test_bullish_signal_on_short_leg_clarifies_without_proof(self, menu):
        # Direction incoherence with no intent stub → ask, never silent flip.
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"), direction="short")  # oversold is bullish
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.clarification is not None


# ── #1/§5b.5 — intent-proof mapping repair (INFY "9 EMA above close") ──────────

class TestIntentRepair:
    def test_mismap_repaired_with_intent_proof(self, menu):
        # Proposer chose the bullish price_above_ema for a SHORT leg, but the
        # intent stub proves the user meant price BELOW ema → silent re-map.
        trig = SignalRef(
            name="price_above_ema",
            intent=IntentStub(user_span="9 EMA above close",
                              intended=IntentComparison(lhs="price", rhs="ema", op="lt")),
        )
        sdl = _sdl(trigger=trig, direction="short")
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.sdl.legs[0].entry.trigger.name == "price_below_ema"
        assert any(e.get("path") == "signal" and e.get("to") == "price_below_ema"
                   for e in r.coverage.repaired)
        assert r.clarification is None  # proof present → no question

    def test_mismap_without_proof_clarifies(self, menu):
        # Same incoherence, NO intent stub → must ask, never silently swap.
        sdl = _sdl(trigger=SignalRef(name="price_above_ema"), direction="short")
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.clarification is not None
        assert r.sdl.legs[0].entry.trigger.name == "price_above_ema"  # untouched


# ── #10/#11 — input sanity (100% risk, 0% drawdown) ────────────────────────────

class TestInputSanity:
    def test_impossible_risk_pct_capped_with_note(self, menu):
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"),
                   sizing=SizingSpec(mode="risk_based", risk_pct=100.0))
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.sdl.risk.sizing.risk_pct <= 10.0
        assert any(e["path"] == "risk.sizing.risk_pct" for e in r.coverage.repaired)
        assert any("risk" in n.lower() for n in r.notes)

    def test_zero_percent_stop_asks(self, menu):
        sdl = _sdl(trigger=SignalRef(name="rsi_oversold"),
                   sl=StopLossSpec(type="percent", value=0.0))
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.clarification is not None and r.clarification["field"] == "risk.stop_loss"


# ── param hygiene — LLM stuffs sibling params on a card that doesn't take them ─

class TestParamPrune:
    def test_unknown_params_pruned_so_validation_passes(self, menu):
        # Repro of a real run: price_above_ema only takes `window`, but the LLM
        # added window_fast/window_slow → validator rejected → strategy failed.
        from app.planner.sdl_validator import validate_sdl
        sdl = _sdl(
            trigger=SignalRef(name="macd_bullish_cross",
                              params={"window_fast": 12, "window_slow": 26, "window_sign": 9}),
            filters=[SignalRef(name="price_above_ema",
                               params={"window": 50, "window_fast": 9, "window_slow": 21})],
            symbol="INFY.NS", asset_class="equity_cash", market="indian_stocks",
        )
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert r.sdl.legs[0].entry.filters[0].params == {"window": 50}
        # MACD card genuinely has no params_by_timeframe→ its params are left as-is
        # only if unknown; either way the SDL must now validate cleanly.
        assert validate_sdl(r.sdl, menu).ok


# ── #2/#3 — unmapped honesty (order blocks / gamma / RS) ────────────────────────

class TestUnmappedHonesty:
    def test_unmapped_concept_surfaced_never_substituted(self, menu):
        sdl = _sdl(
            trigger=SignalRef(name="rsi_oversold"),
            unmapped=[UnmappedDetail(text="order blocks", kind="missing_card", note="")],
        )
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        assert any(u["concept"] == "order blocks" for u in r.coverage.unmapped)
        assert not r.coverage.fully_captured()


# ── #15 — provenance honesty (recomputed from the SDL, not LLM self-report) ─────

class TestProvenanceHonesty:
    def test_coverage_recomputed_from_field_sources(self, menu):
        sdl = _sdl(
            trigger=SignalRef(name="rsi_oversold"),
            field_sources={
                "universe.symbol": "user",
                "context.timeframe": "inferred",
                "risk.stop_loss": "default",
            },
        )
        r = verify(sdl, symbol_ctx=INFY, catalog=menu)
        captured = {e["path"] for e in r.coverage.captured}
        inferred = {e["path"] for e in r.coverage.inferred}
        defaults = {e["path"] for e in r.coverage.defaults}
        assert "universe.symbol" in captured
        assert "context.timeframe" in inferred
        assert "risk.stop_loss" in defaults


# ── §5b.6 — bounded self-heal (one re-propose, else clarify) ───────────────────

class TestSelfHeal:
    def test_self_heal_repropose_resolves_contradiction(self, menu):
        bad = _sdl(trigger=SignalRef(name="rsi_oversold"), direction="short")  # incoherent

        def repropose(_reason):
            # The re-proposal fixes the leg direction to match the bullish signal.
            return _sdl(trigger=SignalRef(name="rsi_oversold"), direction="long")

        r = verify(bad, symbol_ctx=INFY, catalog=menu, repropose=repropose)
        assert r.clarification is None  # healed, no question needed
        assert r.sdl.legs[0].direction == "long"

    def test_self_heal_still_bad_falls_back_to_one_clarification(self, menu):
        bad = _sdl(trigger=SignalRef(name="rsi_oversold"), direction="short")

        def repropose(_reason):
            return _sdl(trigger=SignalRef(name="rsi_oversold"), direction="short")  # still bad

        r = verify(bad, symbol_ctx=INFY, catalog=menu, repropose=repropose)
        assert r.clarification is not None
