"""
tests/test_planner/test_evaluator.py — Phase 6 evaluation integration tests.

Tests the thin adapter layer (evaluator.py) without running the actual engine:
  - artifact_to_yaml() produces valid YAML the engine loader can parse
  - YAML contains correct fields (symbol, conditions, risk, gates, direction)
  - stamp_result() adds version metadata
  - resolve_dynamic_symbol() extracts symbol from candidates
  - run_evaluation() raises RuntimeError when engine not importable (stub test)

Acceptance #8: output is the existing backtest result, version-stamped.
"""
import yaml

import pytest

from app.planner.sdl import (
    SDL,
    DynamicRank,
    DynamicUniverse,
    EntrySpec,
    EventGate,
    ExitSpec,
    GatesSpec,
    HTFRule,
    Leg,
    Provenance,
    RegimeGate,
    RelativeStrengthGate,
    RiskSpec,
    SessionGate,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    TrailingSpec,
    VolatilityGate,
)
from app.planner.compiler import compile_sdl, StrategyArtifact
from app.planner.evaluator import artifact_to_yaml, resolve_dynamic_symbol, stamp_result


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _eth_artifact() -> StrategyArtifact:
    sdl = SDL(
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
        provenance=Provenance(field_sources={"universe.symbol": "user"}),
    )
    return compile_sdl(sdl)


def _nse_artifact_with_gates() -> StrategyArtifact:
    sdl = SDL(
        context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
        universe=StaticUniverse(asset_class="equity_cash", symbol="HDFCBANK.NS"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(trigger=SignalRef(name="ema_cross_up", params={})),
                exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down", params={})]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="atr", multiple=1.5, window=14),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        gates=GatesSpec(
            regime=RegimeGate(allowed=["trending_up"]),
            volatility=VolatilityGate(metric="atr", window=14, min=0.3, max=4.0),
            event=EventGate(skip_dates=["2025-01-26"]),
            session=SessionGate(start="09:15", end="15:00", timezone="IST"),
            relative_strength=RelativeStrengthGate(window=14, min_ratio=1.05, reference_symbol="NIFTY50"),
        ),
        htf_rules=[HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating")],
        provenance=Provenance(field_sources={}),
    )
    return compile_sdl(sdl)


def _dynamic_artifact() -> StrategyArtifact:
    sdl = SDL(
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
                entry=EntrySpec(trigger=SignalRef(name="ema_cross_up", params={})),
                exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down", params={})]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=1.5),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        provenance=Provenance(field_sources={}),
    )
    return compile_sdl(sdl)


# ── artifact_to_yaml tests ────────────────────────────────────────────────────

class TestArtifactToYAML:
    def test_produces_valid_yaml(self):
        art = _eth_artifact()
        raw = artifact_to_yaml(art)
        parsed = yaml.safe_load(raw)
        assert isinstance(parsed, dict)
        assert "strategy" in parsed

    def test_strategy_has_required_fields(self):
        art = _eth_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        strat = parsed["strategy"]
        for key in ("name", "symbol", "market", "timeframe", "entry_condition",
                    "exit_condition", "stop_loss_pct", "take_profit_pct"):
            assert key in strat, f"Missing key: {key}"

    def test_symbol_correct_for_static(self):
        art = _eth_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        assert parsed["strategy"]["symbol"] == "ETH_USDC"

    def test_symbol_override_for_dynamic(self):
        art = _dynamic_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art, symbol_override="TATASTEEL.NS"))
        assert parsed["strategy"]["symbol"] == "TATASTEEL.NS"

    def test_entry_condition_in_yaml(self):
        art = _eth_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        entry = parsed["strategy"]["entry_condition"]
        assert "RSI" in entry

    def test_exit_condition_in_yaml(self):
        art = _eth_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        exit_cond = parsed["strategy"]["exit_condition"]
        assert exit_cond  # not empty

    def test_direction_in_yaml(self):
        art = _eth_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        assert parsed["strategy"]["direction"] == "long_only"

    def test_stop_loss_spec_in_yaml(self):
        art = _eth_artifact()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        sl = parsed["strategy"].get("stop_loss")
        assert sl is not None
        assert sl["type"] == "percent"
        assert sl["pct"] == pytest.approx(2.0)

    def test_atr_stop_loss_in_yaml(self):
        art = _nse_artifact_with_gates()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        sl = parsed["strategy"].get("stop_loss")
        assert sl is not None
        assert sl["type"] == "atr"
        assert sl["multiplier"] == pytest.approx(1.5)
        assert sl["window"] == 14

    def test_regime_filter_in_yaml(self):
        art = _nse_artifact_with_gates()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        rf = parsed["strategy"].get("regime_filter")
        assert rf is not None
        assert "trending_up" in rf["allowed"]

    def test_volatility_filter_in_yaml(self):
        art = _nse_artifact_with_gates()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        vf = parsed["strategy"].get("volatility_filter")
        assert vf is not None
        assert vf["metric"] == "atr"
        assert vf["min"] == pytest.approx(0.3)

    def test_event_filter_in_yaml(self):
        art = _nse_artifact_with_gates()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        ef = parsed["strategy"].get("event_filter")
        assert ef is not None
        assert "2025-01-26" in ef["skip_dates"]

    def test_rs_filter_in_yaml(self):
        art = _nse_artifact_with_gates()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        rf = parsed["strategy"].get("relative_strength_filter")
        assert rf is not None
        assert rf["window"] == 14
        assert rf["min_ratio"] == pytest.approx(1.05)

    def test_htf_rules_in_yaml(self):
        art = _nse_artifact_with_gates()
        parsed = yaml.safe_load(artifact_to_yaml(art))
        htf = parsed["strategy"].get("htf")
        assert htf is not None
        assert len(htf) == 1
        assert htf[0]["timeframe"] == "1h"
        assert "EMA" in htf[0]["condition"]

    def test_trailing_stop_in_yaml(self):
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
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                trailing=TrailingSpec(enabled=True, type="atr_based", distance_atr=2.0),
            ),
            provenance=Provenance(field_sources={}),
        )
        art = compile_sdl(sdl)
        parsed = yaml.safe_load(artifact_to_yaml(art))
        ts = parsed["strategy"].get("trailing_stop")
        assert ts is not None
        assert ts["type"] == "atr"


# ── Loader compatibility test (Acceptance #7) ─────────────────────────────────

class TestLoaderCompatibility:
    """Acceptance #7: artifact IS the existing engine contract.

    Verify that the YAML produced by artifact_to_yaml() can be parsed by
    the engine's load_strategy_from_content() without errors.
    """
    def test_yaml_parseable_by_engine_loader(self):
        try:
            from engine.loader import load_strategy_from_content
        except ImportError:
            pytest.skip("engine not importable in this environment")

        art = _eth_artifact()
        yaml_str = artifact_to_yaml(art)
        cfg = load_strategy_from_content(yaml_str)
        assert cfg.symbol == "ETH_USDC"
        assert cfg.timeframe == "15m"
        assert "RSI" in cfg.entry_condition

    def test_yaml_parseable_with_gates(self):
        try:
            from engine.loader import load_strategy_from_content
        except ImportError:
            pytest.skip("engine not importable in this environment")

        art = _nse_artifact_with_gates()
        yaml_str = artifact_to_yaml(art)
        cfg = load_strategy_from_content(yaml_str)
        assert cfg.symbol == "HDFCBANK.NS"
        assert cfg.regime_filter_allowed is not None
        assert "trending_up" in cfg.regime_filter_allowed


# ── stamp_result tests ────────────────────────────────────────────────────────

class TestStampResult:
    def test_adds_sdl_artifact_key(self):
        art = _eth_artifact()
        result = {"trades": [], "metrics": {}}
        stamped = stamp_result(result, art)
        assert "_sdl_artifact" in stamped

    def test_artifact_version_in_stamp(self):
        art = _eth_artifact()
        result = {"trades": []}
        stamped = stamp_result(result, art)
        meta = stamped["_sdl_artifact"]
        assert "artifact_version" in meta
        assert meta["artifact_version"] == art.version

    def test_content_hash_in_stamp(self):
        art = _eth_artifact()
        stamped = stamp_result({}, art)
        assert stamped["_sdl_artifact"]["content_hash"] == art.content_hash

    def test_artifact_id_in_stamp(self):
        art = _eth_artifact()
        stamped = stamp_result({}, art)
        assert stamped["_sdl_artifact"]["artifact_id"] == art.artifact_id

    def test_original_result_preserved(self):
        art = _eth_artifact()
        result = {"trades": [1, 2, 3], "metrics": {"sharpe": 1.2}}
        stamped = stamp_result(result, art)
        assert stamped["trades"] == [1, 2, 3]
        assert stamped["metrics"]["sharpe"] == pytest.approx(1.2)


# ── resolve_dynamic_symbol tests ──────────────────────────────────────────────

class TestResolveDynamicSymbol:
    def test_returns_artifact_symbol_for_static(self):
        art = _eth_artifact()
        resolved = resolve_dynamic_symbol(art)
        assert resolved == "ETH_USDC"

    def test_returns_none_for_dynamic_no_candidates(self):
        art = _dynamic_artifact()
        resolved = resolve_dynamic_symbol(art, candidates=None)
        assert resolved is None

    def test_resolves_from_string_candidate(self):
        art = _dynamic_artifact()
        resolved = resolve_dynamic_symbol(art, candidates=["TATASTEEL.NS"])
        assert resolved == "TATASTEEL.NS"

    def test_resolves_from_dict_candidate(self):
        art = _dynamic_artifact()
        resolved = resolve_dynamic_symbol(art, candidates=[{"symbol": "HDFCBANK.NS"}])
        assert resolved == "HDFCBANK.NS"

    def test_resolves_from_object_candidate(self):
        class FakeCandidate:
            symbol = "INFY.NS"
        art = _dynamic_artifact()
        resolved = resolve_dynamic_symbol(art, candidates=[FakeCandidate()])
        assert resolved == "INFY.NS"


# ── run_evaluation stub test ──────────────────────────────────────────────────

class TestRunEvaluation:
    def test_raises_when_engine_not_importable(self, monkeypatch):
        import sys
        # Remove engine from sys.modules so import fails
        monkeypatch.setitem(sys.modules, "engine", None)
        monkeypatch.setitem(sys.modules, "engine.runner", None)

        from app.planner.evaluator import run_evaluation
        art = _eth_artifact()
        with pytest.raises(RuntimeError, match="quant_engine not importable"):
            run_evaluation(art, ohlcv_data=None)
