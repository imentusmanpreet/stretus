"""
tests/test_planner/test_compiler.py — Phase 4 strategy compiler tests.

Validates:
  - compile_sdl returns StrategyArtifact
  - condition strings built from catalog card formulas + params
  - risk mapped correctly (percent SL, ATR SL, RR TP, trailing)
  - gates mapped correctly (regime, volatility, event, session, RS)
  - HTF rules mapped
  - static vs dynamic universe
  - artifact metadata (version, content_hash, artifact_id)
  - to_strategy_config_dict() produces valid dict
  - no provenance/LLM fields in artifact
  - Acceptance #7: artifact IS the existing engine contract (no second pipeline)
"""
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
    ScaleOutSpec,
    SessionGate,
    SignalRef,
    SizingSpec,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    TrailingSpec,
    VolatilityGate,
    VolumeRatioGate,
)
from app.planner.compiler import StrategyArtifact, compile_sdl


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _eth_sdl(**overrides) -> SDL:
    base = dict(
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
    base.update(overrides)
    return SDL(**base)


def _nse_sdl(**overrides) -> SDL:
    base = dict(
        context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
        universe=StaticUniverse(asset_class="equity_cash", symbol="HDFCBANK.NS"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(
                    trigger=SignalRef(name="ema_cross_up", params={"window_fast": 9, "window_slow": 21})
                ),
                exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down", params={"window_fast": 9, "window_slow": 21})]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=1.5),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        provenance=Provenance(field_sources={}),
    )
    base.update(overrides)
    return SDL(**base)


# ── Basic artifact tests ──────────────────────────────────────────────────────

class TestArtifactBasics:
    def test_returns_artifact(self):
        art = compile_sdl(_eth_sdl())
        assert isinstance(art, StrategyArtifact)

    def test_artifact_id_is_uuid(self):
        import uuid
        art = compile_sdl(_eth_sdl())
        uuid.UUID(art.artifact_id)  # should not raise

    def test_version_inherited_from_sdl(self):
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        assert art.version == sdl.version

    def test_content_hash_inherited_from_sdl(self):
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        assert art.content_hash == sdl.content_hash

    def test_no_provenance_fields(self):
        art = compile_sdl(_eth_sdl())
        d = art.model_dump()
        assert "provenance" not in d
        assert "field_sources" not in d
        assert "unmapped_details" not in d
        assert "clarifications_needed" not in d

    def test_symbol_set_for_static(self):
        art = compile_sdl(_eth_sdl())
        assert art.symbol == "ETH_USDC"

    def test_market_and_timeframe(self):
        art = compile_sdl(_eth_sdl())
        assert art.market == "crypto"
        assert art.timeframe == "15m"

    def test_direction_long_only(self):
        art = compile_sdl(_eth_sdl())
        assert art.direction == "long_only"

    def test_direction_short_only(self):
        sdl = _eth_sdl(
            legs=[
                Leg(
                    direction="short",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_overbought", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_oversold", params={})]),
                )
            ]
        )
        art = compile_sdl(sdl)
        assert art.direction == "short_only"

    def test_direction_both(self):
        sdl = _eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                ),
                Leg(
                    direction="short",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_overbought", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_oversold", params={})]),
                ),
            ]
        )
        art = compile_sdl(sdl)
        assert art.direction == "both"

    def test_artifact_is_json_serializable(self):
        import json
        art = compile_sdl(_eth_sdl())
        raw = art.model_dump_json()
        d = json.loads(raw)
        assert "entry_condition" in d
        assert "artifact_id" in d


# ── Condition string tests ────────────────────────────────────────────────────

class TestConditionStrings:
    def test_rsi_oversold_entry_condition(self):
        art = compile_sdl(_eth_sdl())
        # rsi_oversold formula: RSI({window}) < {threshold}
        assert "RSI(14)" in art.entry_condition
        assert "< 30" in art.entry_condition

    def test_rsi_overbought_exit_condition(self):
        art = compile_sdl(_eth_sdl())
        # rsi_overbought formula should contain RSI > something
        assert "RSI" in art.exit_condition

    def test_ema_cross_up_entry(self):
        art = compile_sdl(_nse_sdl())
        # ema_cross_up involves EMA crossover
        assert "EMA" in art.entry_condition

    def test_entry_condition_nonempty(self):
        art = compile_sdl(_eth_sdl())
        assert art.entry_condition.strip()

    def test_entry_with_filter_uses_and(self):
        sdl = _eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(
                        trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30}),
                        filters=[SignalRef(name="ema_above", params={"window_fast": 9, "window_slow": 21})],
                    ),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ]
        )
        art = compile_sdl(sdl)
        assert " AND " in art.entry_condition
        assert "RSI" in art.entry_condition
        assert "EMA" in art.entry_condition

    def test_multiple_exit_triggers_uses_or(self):
        sdl = _eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[
                        SignalRef(name="rsi_overbought", params={}),
                        SignalRef(name="ema_cross_down", params={"window_fast": 9, "window_slow": 21}),
                    ]),
                )
            ]
        )
        art = compile_sdl(sdl)
        assert " OR " in art.exit_condition


# ── Risk mapping tests ────────────────────────────────────────────────────────

class TestRiskMapping:
    def test_percent_sl(self):
        art = compile_sdl(_eth_sdl())
        assert art.stop_loss_pct == 2.0
        assert art.stop_loss_spec is not None
        assert art.stop_loss_spec["type"] == "percent"
        assert art.stop_loss_spec["pct"] == 2.0

    def test_atr_sl(self):
        sdl = _eth_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="atr", multiple=1.5, window=14),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            )
        )
        art = compile_sdl(sdl)
        assert art.stop_loss_spec is not None
        assert art.stop_loss_spec["type"] == "atr"
        assert art.stop_loss_spec["multiplier"] == 1.5
        assert art.stop_loss_spec["window"] == 14

    def test_rr_tp_multiplies_sl(self):
        art = compile_sdl(_eth_sdl())  # SL=2%, RR=2 → TP=4%
        assert art.take_profit_pct == pytest.approx(4.0)

    def test_percent_tp(self):
        sdl = _eth_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=1.5),
                take_profit=TakeProfitSpec(type="percent", value=3.0),
            )
        )
        art = compile_sdl(sdl)
        assert art.take_profit_pct == pytest.approx(3.0)

    def test_trailing_stop_mapped(self):
        sdl = _eth_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=3.0),
                trailing=TrailingSpec(enabled=True, type="atr_based", distance_atr=2.0),
            )
        )
        art = compile_sdl(sdl)
        assert art.trailing_stop_spec is not None
        assert art.trailing_stop_spec["type"] == "atr"

    def test_no_trailing_when_disabled(self):
        art = compile_sdl(_eth_sdl())
        assert art.trailing_stop_spec is None


# ── Gate mapping tests ────────────────────────────────────────────────────────

class TestGateMapping:
    def test_regime_gate(self):
        sdl = _eth_sdl(
            gates=GatesSpec(regime=RegimeGate(allowed=["trending_up", "trending_down"]))
        )
        art = compile_sdl(sdl)
        assert art.regime_filter_allowed == ["trending_up", "trending_down"]

    def test_volatility_gate(self):
        sdl = _eth_sdl(
            gates=GatesSpec(volatility=VolatilityGate(metric="atr", window=14, min=0.5, max=3.0))
        )
        art = compile_sdl(sdl)
        assert art.vol_filter_metric == "atr"
        assert art.vol_filter_window == 14
        assert art.vol_filter_min == 0.5
        assert art.vol_filter_max == 3.0

    def test_event_gate(self):
        sdl = _eth_sdl(
            gates=GatesSpec(event=EventGate(skip_dates=["2025-01-26", "2025-08-15"]))
        )
        art = compile_sdl(sdl)
        assert art.event_skip_dates == ["2025-01-26", "2025-08-15"]

    def test_session_gate_ist_to_utc(self):
        # 09:15 IST = 09*60+15 - 330 = 555 - 330 = 225 UTC minutes
        sdl = _eth_sdl(
            gates=GatesSpec(session=SessionGate(start="09:15", end="15:00", timezone="IST"))
        )
        art = compile_sdl(sdl)
        assert art.entry_window_start_utc == 225
        assert art.entry_window_end_utc == (15 * 60 - 330) % (24 * 60)

    def test_relative_strength_gate(self):
        sdl = _eth_sdl(
            gates=GatesSpec(
                relative_strength=RelativeStrengthGate(
                    window=14, min_ratio=1.05, reference_symbol="NIFTY50"
                )
            )
        )
        art = compile_sdl(sdl)
        assert art.rs_filter_window == 14
        assert art.rs_filter_min_ratio == pytest.approx(1.05)
        assert art.reference_symbol == "NIFTY50"

    def test_volume_ratio_gate(self):
        sdl = _eth_sdl(
            gates=GatesSpec(volume_ratio=VolumeRatioGate(window=20, min_ratio=1.5))
        )
        art = compile_sdl(sdl)
        assert art.volume_ratio_threshold == pytest.approx(1.5)

    def test_no_gates_all_none(self):
        art = compile_sdl(_eth_sdl())
        assert art.regime_filter_allowed is None
        assert art.vol_filter_metric is None
        assert art.event_skip_dates is None
        assert art.entry_window_start_utc is None


# ── HTF rules ─────────────────────────────────────────────────────────────────

class TestHTFRules:
    def test_htf_rule_mapped(self):
        sdl = _eth_sdl(
            htf_rules=[HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating")]
        )
        art = compile_sdl(sdl)
        assert len(art.htf_rules) == 1
        assert art.htf_rules[0]["timeframe"] == "1h"
        assert "EMA" in art.htf_rules[0]["condition"]

    def test_only_first_htf_used(self):
        # Engine supports only 1 HTF (gap surfaced by validator)
        sdl = _eth_sdl(
            htf_rules=[
                HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating"),
                HTFRule(timeframe="4h", condition="CLOSE > EMA(200)", role="gating"),
            ]
        )
        art = compile_sdl(sdl)
        assert len(art.htf_rules) == 1
        assert art.htf_rules[0]["timeframe"] == "1h"

    def test_no_htf_rules_empty_list(self):
        art = compile_sdl(_eth_sdl())
        assert art.htf_rules == []


# ── Dynamic universe ──────────────────────────────────────────────────────────

class TestDynamicUniverse:
    def test_symbol_empty_for_dynamic(self):
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
        art = compile_sdl(sdl)
        assert art.symbol == ""

    def test_discovery_config_set_for_dynamic(self):
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
        art = compile_sdl(sdl)
        assert art.discovery_config is not None
        dc = art.discovery_config
        assert dc["type"] == "dynamic"
        assert dc["asset_class"] == "equity_cash"
        assert "CLOSE > VWAP" in dc["screen"]
        assert dc["rank"]["by"] == "rvol"
        assert dc["tie_break"] == "highest_relative_volume"

    def test_discovery_config_none_for_static(self):
        art = compile_sdl(_eth_sdl())
        assert art.discovery_config is None


# ── to_strategy_config_dict ───────────────────────────────────────────────────

class TestToStrategyConfigDict:
    def test_produces_valid_dict(self):
        art = compile_sdl(_eth_sdl())
        d = art.to_strategy_config_dict()
        assert isinstance(d, dict)
        assert "entry_condition" in d
        assert "exit_condition" in d
        assert "stop_loss" in d
        assert "take_profit" in d

    def test_required_fields_present(self):
        art = compile_sdl(_eth_sdl())
        d = art.to_strategy_config_dict()
        for key in ("name", "symbol", "market", "timeframe", "objective",
                    "entry_condition", "exit_condition", "stop_loss", "take_profit",
                    "indicators", "daily_loss_cap_pct", "max_trades_per_day"):
            assert key in d, f"Missing key: {key}"

    def test_stop_loss_spec_present_when_set(self):
        art = compile_sdl(_eth_sdl())
        d = art.to_strategy_config_dict()
        assert "stop_loss_spec" in d
        assert d["stop_loss_spec"]["type"] == "percent"

    def test_no_provenance_in_config_dict(self):
        art = compile_sdl(_eth_sdl())
        d = art.to_strategy_config_dict()
        for bad_key in ("provenance", "field_sources", "unmapped_details",
                         "clarifications_needed", "content_hash", "artifact_id"):
            assert bad_key not in d, f"Unexpected key: {bad_key}"


# ── Phase 12: two-sided (long + short) compilation ─────────────────────────────

def _both_sdl() -> SDL:
    return _eth_sdl(
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
            ),
            Leg(
                direction="short",
                entry=EntrySpec(trigger=SignalRef(name="rsi_overbought", params={"window": 14, "threshold": 70})),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_oversold", params={})]),
            ),
        ]
    )


class TestTwoSidedCompilation:
    def test_both_emits_long_in_entry_and_short_in_short_entry(self):
        art = compile_sdl(_both_sdl())
        assert art.direction == "both"
        # Long leg drives the primary entry; short leg goes to short_entry.
        assert "RSI(14) < 30" in art.entry_condition
        assert art.short_entry_condition.strip()
        assert "RSI(14) > 70" in art.short_entry_condition
        assert art.short_exit_condition.strip()
        # The two legs must NOT collapse into one another.
        assert art.entry_condition != art.short_entry_condition

    def test_long_only_has_no_short_conditions(self):
        art = compile_sdl(_eth_sdl())   # single long leg
        assert art.direction == "long_only"
        assert art.short_entry_condition == ""
        assert art.short_exit_condition == ""

    def test_short_only_puts_leg_in_primary_entry(self):
        sdl = _eth_sdl(
            legs=[
                Leg(
                    direction="short",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_overbought", params={"window": 14, "threshold": 70})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_oversold", params={})]),
                )
            ]
        )
        art = compile_sdl(sdl)
        assert art.direction == "short_only"
        # The single short leg lives in the primary entry (direction tells the
        # engine to short); short_entry stays empty to avoid a double pass.
        assert "RSI(14) > 70" in art.entry_condition
        assert art.short_entry_condition == ""

    def test_short_conditions_flow_into_config_dict(self):
        d = compile_sdl(_both_sdl()).to_strategy_config_dict()
        assert "RSI(14) > 70" in d["short_entry_condition"]
        assert d["short_exit_condition"].strip()
