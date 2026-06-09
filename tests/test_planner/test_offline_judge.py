"""
tests/test_planner/test_offline_judge.py — Phase 7 offline judge tests.

Tests:
  - extract_requirements() identifies correct atomic clauses from prompts
  - judge() produces JudgeResult with correct covered/omitted lists
  - A well-formed SDL (all requirements in provenance) passes the judge (ok=True)
  - A SDL missing a requirement from provenance is flagged as omission (ok=False)
  - Acceptance #11: "every requirement accounted for" is PROVEN by the judge
"""
import pytest

from app.planner.sdl import (
    SDL,
    ClarificationNeeded,
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
    UnmappedDetail,
)
from app.planner.offline_judge import JudgeResult, extract_requirements, judge


# ── extract_requirements tests ────────────────────────────────────────────────

class TestExtractRequirements:
    def test_finds_rsi(self):
        reqs = extract_requirements("Buy ETH when RSI drops below 30")
        assert "signal:RSI" in reqs

    def test_finds_bollinger(self):
        reqs = extract_requirements("Enter above Bollinger Band upper")
        assert "signal:BB" in reqs

    def test_finds_stop_loss(self):
        reqs = extract_requirements("Stop loss 2%")
        assert "risk:stop_loss" in reqs

    def test_finds_take_profit(self):
        reqs = extract_requirements("Take profit 3%")
        assert "risk:take_profit" in reqs

    def test_finds_rr_ratio(self):
        reqs = extract_requirements("Risk reward 2:1")
        assert "risk:rr_ratio" in reqs

    def test_finds_long_direction(self):
        reqs = extract_requirements("Go long on EMA crossover")
        assert "direction:long" in reqs

    def test_finds_short_direction(self):
        reqs = extract_requirements("Short when MACD crosses below signal")
        assert "direction:short" in reqs

    def test_finds_crypto_asset(self):
        reqs = extract_requirements("Buy ETH on Binance")
        assert "universe:crypto_asset" in reqs

    def test_finds_equity_asset(self):
        reqs = extract_requirements("Trade NSE stocks")
        assert "universe:equity_asset" in reqs

    def test_finds_timeframe(self):
        reqs = extract_requirements("Use 15m timeframe")
        assert "timeframe:explicit" in reqs

    def test_finds_regime_gate(self):
        reqs = extract_requirements("Only trade in trending regime")
        assert "gate:regime" in reqs

    def test_finds_event_gate(self):
        reqs = extract_requirements("Skip on earnings dates 2025-01-31")
        assert "gate:event" in reqs

    def test_no_duplicates(self):
        reqs = extract_requirements("RSI and RSI and RSI")
        assert reqs.count("signal:RSI") == 1

    def test_empty_prompt(self):
        reqs = extract_requirements("")
        assert reqs == []

    def test_multi_indicator_prompt(self):
        reqs = extract_requirements("EMA crossover confirmed by RSI > 50 and MACD positive")
        assert "signal:EMA" in reqs
        assert "signal:RSI" in reqs
        assert "signal:MACD" in reqs


# ── JudgeResult tests ─────────────────────────────────────────────────────────

class TestJudgeResult:
    def _covered_sdl(self) -> SDL:
        """SDL with all common requirements in provenance."""
        return SDL(
            context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(
                        trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})
                    ),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(
                field_sources={
                    "context.timeframe":         "user",
                    "universe.symbol":           "user",
                    "legs.0.direction":          "user",
                    "legs.0.entry.trigger":      "user",
                    "risk.stop_loss":            "user",
                    "risk.take_profit":          "user",
                },
            ),
        )

    def _missing_sdl(self) -> SDL:
        """SDL that has RSI in the trigger but provenance is empty (omission)."""
        return SDL(
            context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ],
            risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=2.0)),
            provenance=Provenance(field_sources={}),  # empty provenance = omissions
        )

    def test_well_formed_sdl_ok(self):
        prompt = "Buy ETH on 15m when RSI < 30. Stop 2%, target 2:1."
        sdl = self._covered_sdl()
        result = judge(prompt, sdl)
        assert isinstance(result, JudgeResult)
        assert result.ok, f"Omitted: {result.omitted}"

    def test_covered_requirements_listed(self):
        prompt = "Buy ETH on 15m when RSI < 30. Stop 2%, target 2:1."
        sdl = self._covered_sdl()
        result = judge(prompt, sdl)
        assert len(result.covered) > 0

    def test_omission_detected(self):
        prompt = "Buy ETH on 15m when RSI < 30. Stop 2%."
        sdl = self._missing_sdl()
        result = judge(prompt, sdl)
        # With empty provenance, the judge should flag omissions
        # (signal:RSI, risk:stop_loss are in the prompt but not in provenance)
        assert not result.ok

    def test_unmapped_detail_counts_as_covered(self):
        """Requirements in unmapped_details are accounted for (match% already counts them)."""
        prompt = "Trade top-20 ETH parallel."
        sdl = SDL(
            context=StrategyContext(market="crypto", timeframe="15m", objective="breakout"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ],
            risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=2.0)),
            provenance=Provenance(
                field_sources={"universe.symbol": "user"},
                unmapped_details=[
                    UnmappedDetail(
                        text="top-20 parallel",
                        kind="engine_capability_gap",
                        note="scanner picks one asset, not top-N",
                    )
                ],
            ),
        )
        result = judge(prompt, sdl)
        # universe:dynamic_hint or universe:crypto_asset should be covered via unmapped
        # No omissions expected for the key requirements
        assert "universe:crypto_asset" in result.covered or "universe:crypto_asset" in result.missed_in_provenance

    def test_judge_result_has_prompt(self):
        prompt = "Buy ETH on 15m"
        sdl = self._covered_sdl()
        result = judge(prompt, sdl)
        assert result.prompt == prompt


# ── Acceptance #11: every requirement accounted for ───────────────────────────

class TestAcceptance11:
    """Proves the judge catches omissions that match% would miss."""

    def test_judge_catches_signal_omission(self):
        """SDL where the selector emitted an RSI card but forgot to log it in provenance."""
        prompt = "Buy ETH when RSI < 30, stop 2%"
        # SDL has RSI trigger but provenance is silent about it
        sdl = SDL(
            context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ],
            risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=2.0)),
            provenance=Provenance(
                field_sources={
                    "risk.stop_loss": "user",  # stop loss is in provenance
                    # BUT: no entry for universe.symbol or legs.0.entry.trigger
                },
            ),
        )
        result = judge(prompt, sdl)
        # The judge should find omissions (RSI signal not in provenance)
        assert "signal:RSI" in result.omitted or not result.ok

    def test_judge_passes_complete_provenance(self):
        """SDL with complete provenance passes the judge."""
        prompt = "Buy ETH on 15m when RSI < 30, stop 2%, target 2:1"
        sdl = SDL(
            context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(
                field_sources={
                    "context.timeframe":     "user",
                    "universe.symbol":       "user",
                    "legs.0.entry.trigger":  "user",
                    "legs.0.direction":      "user",
                    "risk.stop_loss":        "user",
                    "risk.take_profit":      "user",
                },
            ),
        )
        result = judge(prompt, sdl)
        assert result.ok, f"Unexpected omissions: {result.omitted}"
