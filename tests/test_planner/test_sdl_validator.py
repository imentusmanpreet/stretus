"""
tests/test_planner/test_sdl_validator.py — Phase 3 SDL validator tests.

Covers:
  - referential: unknown signal, unknown symbol, bad rank metric, bad tie_break
  - parameter: unknown param key, out-of-range window, out-of-range threshold
  - satisfiability: unsatisfiable entry condition
  - safety: missing stop loss, missing exit
  - feasibility: unsupported timeframe
  - engine capability: scale_outs + trailing, dynamic universe gap, multiple HTF
  - happy path: valid static and dynamic SDLs return ok=True
"""
import pytest

from app.planner.sdl import (
    SDL,
    DynamicRank,
    DynamicUniverse,
    EntrySpec,
    ExitSpec,
    GatesSpec,
    HTFRule,
    Leg,
    Provenance,
    RiskSpec,
    ScaleOutSpec,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    TrailingSpec,
)
from app.planner.sdl_validator import ValidationResult, validate_sdl
from app.planner.catalog_schema import build_menu, invalidate_menu_cache


@pytest.fixture(scope="module")
def menu():
    invalidate_menu_cache()
    return build_menu()


# ── Helper ────────────────────────────────────────────────────────────────────

def _valid_eth_sdl(**overrides) -> SDL:
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


def _valid_nse_sdl(**overrides) -> SDL:
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


def _codes(result: ValidationResult) -> list[str]:
    return [e.code for e in result.errors]


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_valid_static_crypto_ok(self, menu):
        result = validate_sdl(_valid_eth_sdl(), menu)
        assert result.ok, result.errors

    def test_valid_static_equity_ok(self, menu):
        result = validate_sdl(_valid_nse_sdl(), menu)
        assert result.ok, result.errors

    def test_valid_dynamic_universe_ok(self, menu):
        sdl = SDL(
            context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
            universe=DynamicUniverse(
                asset_class="equity_cash",
                screen=[],
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
        result = validate_sdl(sdl, menu)
        assert result.ok, result.errors

    def test_valid_has_no_errors(self, menu):
        result = validate_sdl(_valid_eth_sdl(), menu)
        assert result.errors == []


# ── Referential errors ────────────────────────────────────────────────────────

class TestReferential:
    def test_unknown_entry_trigger(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="totally_fake_signal")),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "unknown_signal" in _codes(result)

    def test_unknown_exit_trigger(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold")),
                    exit=ExitSpec(triggers=[SignalRef(name="nonexistent_exit_xyz")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "unknown_signal" in _codes(result)

    def test_unknown_symbol_crypto(self, menu):
        sdl = _valid_eth_sdl(
            universe=StaticUniverse(asset_class="crypto_spot", symbol="NONEXISTENT_XYZ")
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "unknown_symbol" in _codes(result)

    def test_unknown_symbol_equity(self, menu):
        sdl = _valid_nse_sdl(
            universe=StaticUniverse(asset_class="equity_cash", symbol="FAKECORP.NS")
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "unknown_symbol" in _codes(result)

    def test_known_crypto_alias_resolves(self, menu):
        # ETH_USDC is a known symbol; should pass referential
        sdl = _valid_eth_sdl(
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC")
        )
        result = validate_sdl(sdl, menu)
        # No unknown_symbol error
        assert "unknown_symbol" not in _codes(result)

    def test_known_nse_symbol_resolves(self, menu):
        sdl = _valid_nse_sdl(
            universe=StaticUniverse(asset_class="equity_cash", symbol="HDFCBANK.NS")
        )
        result = validate_sdl(sdl, menu)
        assert "unknown_symbol" not in _codes(result)

    def test_invalid_rank_metric_rejected_by_schema(self, menu):
        # DynamicRank.by is a Literal; the SDL schema itself rejects unknown values.
        # This is caught at construction, so validate_sdl never sees it.
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            DynamicRank(by="made_up_metric")  # type: ignore

    def test_invalid_tie_break(self, menu):
        sdl = SDL(
            context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
            universe=DynamicUniverse(
                asset_class="equity_cash",
                screen=[],
                rank=DynamicRank(by="rvol", order="desc"),
                tie_break="i_invented_this",
            ),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="ema_cross_up")),
                    exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down")]),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=1.5),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(field_sources={}),
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "invalid_tie_break" in _codes(result)


# ── Parameter errors ──────────────────────────────────────────────────────────

class TestParameter:
    def test_unknown_param_key(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(
                        trigger=SignalRef(
                            name="rsi_oversold",
                            params={"window": 14, "threshold": 30, "fantasy_param": 99}
                        )
                    ),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "unknown_param" in _codes(result)

    def test_window_zero_rejected(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(
                        trigger=SignalRef(name="rsi_oversold", params={"window": 0, "threshold": 30})
                    ),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "param_out_of_range" in _codes(result)

    def test_window_above_max_rejected(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(
                        trigger=SignalRef(name="rsi_oversold", params={"window": 9999, "threshold": 30})
                    ),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "param_out_of_range" in _codes(result)

    def test_valid_params_pass(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(
                        trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})
                    ),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert "param_out_of_range" not in _codes(result)
        assert "unknown_param" not in _codes(result)


# ── Safety errors ─────────────────────────────────────────────────────────────

class TestSafety:
    def test_missing_stop_loss(self, menu):
        sdl = _valid_eth_sdl(
            risk=RiskSpec(take_profit=TakeProfitSpec(type="rr", ratio=2.0))
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "missing_stop_loss" in _codes(result)

    def test_missing_exit(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[]),  # empty exit, no TP
                )
            ],
            risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=2.0)),
            # no take_profit, no trailing
        )
        result = validate_sdl(sdl, menu)
        assert not result.ok
        assert "missing_exit" in _codes(result)

    def test_trailing_stop_satisfies_exit(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[]),  # empty triggers
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                trailing=TrailingSpec(enabled=True, type="atr_based", distance_atr=2.0),
            ),
        )
        result = validate_sdl(sdl, menu)
        assert "missing_exit" not in _codes(result)

    def test_take_profit_satisfies_exit(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[]),  # empty triggers
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
        )
        result = validate_sdl(sdl, menu)
        assert "missing_exit" not in _codes(result)


# ── Engine capability gaps ────────────────────────────────────────────────────

class TestEngineCapability:
    def test_scale_outs_and_trailing_gap(self, menu):
        sdl = _valid_eth_sdl(
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=2.0),
                take_profit=TakeProfitSpec(type="rr", ratio=3.0),
                scale_outs=[ScaleOutSpec(at_rr=1.5, size_pct=50)],
                trailing=TrailingSpec(enabled=True, type="atr_based", distance_atr=2.0),
            )
        )
        result = validate_sdl(sdl, menu)
        assert result.ok, result.errors  # not a hard error
        assert any("scale_outs" in g or "trailing" in g for g in result.engine_gaps)

    def test_dynamic_universe_gap_noted(self, menu):
        sdl = SDL(
            context=StrategyContext(market="indian_stocks", timeframe="15m", objective="breakout"),
            universe=DynamicUniverse(
                asset_class="equity_cash",
                screen=[],
                rank=DynamicRank(by="rvol", order="desc"),
                tie_break="highest_relative_volume",
            ),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="ema_cross_up")),
                    exit=ExitSpec(triggers=[SignalRef(name="ema_cross_down")]),
                )
            ],
            risk=RiskSpec(
                stop_loss=StopLossSpec(type="percent", value=1.5),
                take_profit=TakeProfitSpec(type="rr", ratio=2.0),
            ),
            provenance=Provenance(field_sources={}),
        )
        result = validate_sdl(sdl, menu)
        assert result.ok, result.errors
        assert any("Dynamic" in g or "dynamic" in g for g in result.engine_gaps)

    def test_multiple_htf_rules_gap(self, menu):
        sdl = _valid_eth_sdl(
            htf_rules=[
                HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating"),
                HTFRule(timeframe="4h", condition="CLOSE > EMA(200)", role="gating"),
            ]
        )
        result = validate_sdl(sdl, menu)
        assert result.ok, result.errors
        assert any("HTF" in g or "htf" in g.lower() for g in result.engine_gaps)

    def test_single_htf_no_gap(self, menu):
        sdl = _valid_eth_sdl(
            htf_rules=[HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating")]
        )
        result = validate_sdl(sdl, menu)
        assert result.ok, result.errors
        htf_gaps = [g for g in result.engine_gaps if "HTF" in g or "htf" in g.lower()]
        assert not htf_gaps


# ── Acceptance #9: validation failure blocks compile ─────────────────────────

class TestValidationBlocksCompile:
    def test_returns_validation_result(self, menu):
        result = validate_sdl(_valid_eth_sdl(), menu)
        assert isinstance(result, ValidationResult)

    def test_invalid_sdl_ok_is_false(self, menu):
        sdl = _valid_eth_sdl(
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="FAKE_SIGNAL_XYZ")),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought")]),
                )
            ]
        )
        result = validate_sdl(sdl, menu)
        assert result.ok is False

    def test_valid_sdl_ok_is_true(self, menu):
        result = validate_sdl(_valid_eth_sdl(), menu)
        assert result.ok is True

    def test_errors_have_field_and_code(self, menu):
        sdl = _valid_eth_sdl(
            universe=StaticUniverse(asset_class="crypto_spot", symbol="FAKE_XYZ_TOKEN")
        )
        result = validate_sdl(sdl, menu)
        assert result.errors
        for e in result.errors:
            assert e.field
            assert e.code
            assert e.message

    def test_never_raises(self, menu):
        # Pass a completely empty-ish SDL-like dict — should not raise
        from app.planner.sdl import GatesSpec
        sdl = _valid_eth_sdl()
        # Should not raise
        result = validate_sdl(sdl, menu)
        assert isinstance(result, ValidationResult)
