"""
tests/test_planner/test_sdl.py — Phase 0 SDL ticket unit tests.

Tests: construction, serialization, round-trip, content_hash stability.
"""
import json
from datetime import datetime, timezone

import pytest

from app.planner.sdl import (
    SDL,
    AssetClass,
    ClarificationNeeded,
    DynamicRank,
    DynamicUniverse,
    EntrySpec,
    ExitSpec,
    GatesSpec,
    HTFRule,
    Leg,
    Provenance,
    RegimeGate,
    RiskSpec,
    ScaleOutSpec,
    SignalRef,
    SizingSpec,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    TrailingSpec,
    UnmappedDetail,
    VolatilityGate,
    _hash_executable,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _eth_static_sdl(**overrides) -> SDL:
    """Minimal valid static-universe SDL (ETH mean-reversion example)."""
    kwargs = dict(
        context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
        universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=2.0),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        provenance=Provenance(
            field_sources={
                "universe.symbol": "user",
                "legs.0.entry.trigger": "user",
                "risk.stop_loss": "user",
                "risk.take_profit": "user",
                "legs.0.exit": "inferred",
            },
        ),
    )
    kwargs.update(overrides)
    return SDL(**kwargs)


def _nse_dynamic_sdl() -> SDL:
    """Minimal valid dynamic-universe SDL (NSE ORB example)."""
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
            field_sources={"universe": "user", "legs.0.entry.trigger": "user", "risk.stop_loss": "user"},
            clarifications_needed=[
                ClarificationNeeded(field="risk.take_profit", question="No target given — use 2:1?", assumed_value="2:1")
            ],
        ),
    )


# ── Construction tests ────────────────────────────────────────────────────────

class TestSDLConstruction:
    def test_static_universe_fields(self):
        sdl = _eth_static_sdl()
        assert sdl.universe.type == "static"
        assert sdl.universe.asset_class == "crypto_spot"
        assert sdl.universe.symbol == "ETH_USDC"

    def test_dynamic_universe_fields(self):
        sdl = _nse_dynamic_sdl()
        assert sdl.universe.type == "dynamic"
        assert sdl.universe.asset_class == "equity_cash"
        assert sdl.universe.screen == ["CLOSE > VWAP"]
        assert sdl.universe.rank.by == "rvol"
        assert sdl.universe.rank.order == "desc"
        assert sdl.universe.tie_break == "highest_relative_volume"

    def test_leg_direction_and_entry(self):
        sdl = _eth_static_sdl()
        leg = sdl.legs[0]
        assert leg.direction == "long"
        assert leg.entry.trigger.name == "rsi_oversold"
        assert leg.entry.trigger.params["window"] == 14
        assert leg.entry.trigger.params["threshold"] == 30

    def test_risk_stop_loss(self):
        sdl = _eth_static_sdl()
        assert sdl.risk.stop_loss is not None
        assert sdl.risk.stop_loss.type == "percent"
        assert sdl.risk.stop_loss.value == 2.0

    def test_risk_take_profit_rr(self):
        sdl = _eth_static_sdl()
        assert sdl.risk.take_profit is not None
        assert sdl.risk.take_profit.type == "rr"
        assert sdl.risk.take_profit.ratio == 2.0

    def test_version_defaults(self):
        sdl = _eth_static_sdl()
        assert sdl.version == 1
        assert sdl.parent_version is None

    def test_content_hash_auto_populated(self):
        sdl = _eth_static_sdl()
        assert isinstance(sdl.content_hash, str)
        assert len(sdl.content_hash) == 64  # SHA-256 hex

    def test_created_at_is_utc(self):
        sdl = _eth_static_sdl()
        assert sdl.created_at.tzinfo is not None

    def test_provenance_field_sources_set(self):
        sdl = _eth_static_sdl()
        assert sdl.provenance.field_sources["universe.symbol"] == "user"
        assert sdl.provenance.field_sources["legs.0.exit"] == "inferred"

    def test_provenance_clarification(self):
        sdl = _nse_dynamic_sdl()
        assert len(sdl.provenance.clarifications_needed) == 1
        c = sdl.provenance.clarifications_needed[0]
        assert c.field == "risk.take_profit"
        assert c.assumed_value == "2:1"

    def test_unmapped_detail_roundtrip(self):
        sdl = _eth_static_sdl()
        sdl.provenance.unmapped_details.append(
            UnmappedDetail(text="top-20 parallel", kind="engine_capability_gap", note="scanner picks one")
        )
        assert sdl.provenance.unmapped_details[0].kind == "engine_capability_gap"

    def test_gates_default_empty(self):
        sdl = _eth_static_sdl()
        assert sdl.gates.regime is None
        assert sdl.gates.volatility is None

    def test_gates_with_regime(self):
        sdl = _eth_static_sdl(gates=GatesSpec(regime=RegimeGate(allowed=["trending"])))
        assert sdl.gates.regime is not None
        assert sdl.gates.regime.allowed == ["trending"]

    def test_htf_rule(self):
        sdl = _eth_static_sdl(
            htf_rules=[HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating")]
        )
        assert len(sdl.htf_rules) == 1
        assert sdl.htf_rules[0].timeframe == "1h"
        assert sdl.htf_rules[0].role == "gating"

    def test_two_legs_long_and_short(self):
        sdl = _eth_static_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(),
                ),
                Leg(
                    direction="short",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_overbought", params={})),
                    exit=ExitSpec(),
                ),
            ]
        )
        assert len(sdl.legs) == 2
        assert sdl.legs[0].direction == "long"
        assert sdl.legs[1].direction == "short"

    def test_scale_outs(self):
        sdl = _eth_static_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=3.0),
                scale_outs=[ScaleOutSpec(at_rr=1.5, size_pct=50.0)],
            )
        )
        assert len(sdl.risk.scale_outs) == 1
        assert sdl.risk.scale_outs[0].at_rr == 1.5
        assert sdl.risk.scale_outs[0].size_pct == 50.0

    def test_trailing_stop(self):
        sdl = _eth_static_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=1.5),
                trailing=TrailingSpec(enabled=True, type="atr_based", distance_atr=2.0),
            )
        )
        assert sdl.risk.trailing is not None
        assert sdl.risk.trailing.enabled is True
        assert sdl.risk.trailing.type == "atr_based"


# ── Serialization tests ───────────────────────────────────────────────────────

class TestSDLSerialization:
    def test_model_dump_is_dict(self):
        sdl = _eth_static_sdl()
        d = sdl.model_dump(mode="json")
        assert isinstance(d, dict)

    def test_json_serializable(self):
        sdl = _eth_static_sdl()
        raw = sdl.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["context"]["timeframe"] == "15m"

    def test_static_universe_type_in_json(self):
        sdl = _eth_static_sdl()
        parsed = json.loads(sdl.model_dump_json())
        assert parsed["universe"]["type"] == "static"
        assert parsed["universe"]["symbol"] == "ETH_USDC"

    def test_dynamic_universe_type_in_json(self):
        sdl = _nse_dynamic_sdl()
        parsed = json.loads(sdl.model_dump_json())
        assert parsed["universe"]["type"] == "dynamic"
        assert "CLOSE > VWAP" in parsed["universe"]["screen"]

    def test_provenance_in_json(self):
        sdl = _eth_static_sdl()
        parsed = json.loads(sdl.model_dump_json())
        assert "provenance" in parsed
        assert parsed["provenance"]["field_sources"]["universe.symbol"] == "user"

    def test_content_hash_in_json(self):
        sdl = _eth_static_sdl()
        parsed = json.loads(sdl.model_dump_json())
        assert len(parsed["content_hash"]) == 64


# ── Round-trip tests ──────────────────────────────────────────────────────────

class TestSDLRoundTrip:
    def test_static_sdl_round_trips(self):
        sdl = _eth_static_sdl()
        raw_json = sdl.model_dump_json()
        restored = SDL.model_validate_json(raw_json)
        assert restored.universe.symbol == "ETH_USDC"  # type: ignore[union-attr]
        assert restored.legs[0].entry.trigger.name == "rsi_oversold"
        assert restored.version == 1

    def test_dynamic_sdl_round_trips(self):
        sdl = _nse_dynamic_sdl()
        raw_json = sdl.model_dump_json()
        restored = SDL.model_validate_json(raw_json)
        assert restored.universe.type == "dynamic"
        assert restored.universe.rank.by == "rvol"  # type: ignore[union-attr]

    def test_content_hash_preserved_after_round_trip(self):
        sdl = _eth_static_sdl()
        h1 = sdl.content_hash
        restored = SDL.model_validate_json(sdl.model_dump_json())
        assert restored.content_hash == h1

    def test_dict_round_trip(self):
        sdl = _eth_static_sdl()
        d = sdl.model_dump(mode="python")
        restored = SDL(**d)
        assert restored.legs[0].direction == "long"
        assert restored.provenance.field_sources["legs.0.exit"] == "inferred"


# ── Content-hash stability tests ──────────────────────────────────────────────

class TestContentHash:
    def test_same_strategy_same_hash(self):
        sdl1 = _eth_static_sdl()
        sdl2 = _eth_static_sdl()
        assert sdl1.content_hash == sdl2.content_hash

    def test_different_symbol_different_hash(self):
        sdl1 = _eth_static_sdl()
        sdl2 = _eth_static_sdl(
            universe=StaticUniverse(asset_class="crypto_spot", symbol="BTC_USDC")
        )
        assert sdl1.content_hash != sdl2.content_hash

    def test_different_timeframe_different_hash(self):
        sdl1 = _eth_static_sdl()
        sdl2 = _eth_static_sdl(
            context=StrategyContext(market="crypto", timeframe="5m", objective="mean_reversion")
        )
        assert sdl1.content_hash != sdl2.content_hash

    def test_different_sl_different_hash(self):
        sdl1 = _eth_static_sdl()
        sdl2 = _eth_static_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=3.0),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            )
        )
        assert sdl1.content_hash != sdl2.content_hash

    def test_provenance_change_does_not_change_hash(self):
        sdl1 = _eth_static_sdl()
        h1 = sdl1.content_hash
        sdl2 = _eth_static_sdl(
            provenance=Provenance(
                field_sources={"universe.symbol": "default"},  # changed source
                unmapped_details=[UnmappedDetail(text="extra", kind="missing_card")],
            )
        )
        assert sdl2.content_hash == h1

    def test_version_change_does_not_change_hash(self):
        sdl1 = _eth_static_sdl()
        h1 = sdl1.content_hash
        sdl2 = _eth_static_sdl(version=3, parent_version=2)
        assert sdl2.content_hash == h1

    def test_bump_version_preserves_logic_hash(self):
        sdl = _eth_static_sdl()
        h_before = sdl.content_hash
        bumped = sdl.bump_version()
        assert bumped.version == 2
        assert bumped.parent_version == 1
        assert bumped.content_hash == h_before

    def test_bump_version_after_field_change_differs(self):
        sdl = _eth_static_sdl()
        changed = _eth_static_sdl(
            universe=StaticUniverse(asset_class="crypto_spot", symbol="SOL_USDC")
        )
        bumped = changed.bump_version()
        assert bumped.content_hash != sdl.content_hash

    def test_hash_is_deterministic_across_instances(self):
        h_values = {_eth_static_sdl().content_hash for _ in range(5)}
        assert len(h_values) == 1

    def test_gate_change_changes_hash(self):
        sdl1 = _eth_static_sdl()
        sdl2 = _eth_static_sdl(
            gates=GatesSpec(volatility=VolatilityGate(metric="atr", window=14, min=0.5))
        )
        assert sdl1.content_hash != sdl2.content_hash

    def test_htf_rule_change_changes_hash(self):
        sdl1 = _eth_static_sdl()
        sdl2 = _eth_static_sdl(
            htf_rules=[HTFRule(timeframe="4h", condition="CLOSE > EMA(50)", role="gating")]
        )
        assert sdl1.content_hash != sdl2.content_hash
