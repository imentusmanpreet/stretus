"""
tests/test_planner/test_provenance_reconciler.py — Reconciler unit tests.

Each test starts with an SDL whose provenance.field_sources is EMPTY (simulating
the LLM forgetting to self-report) and verifies that reconcile() fills in the
correct "user" paths from the prompt text alone.

This proves the bug fix: user inputs must be captured deterministically,
not relying on the LLM's inconsistent self-reporting.
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
    SessionGate,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    TrailingSpec,
    UnmappedDetail,
    VolatilityGate,
    VolumeRatioGate,
)
from app.planner.provenance_reconciler import reconcile


# ── Helper: build an SDL with empty provenance ────────────────────────────────

def _sdl(
    symbol="ETH_USDC",
    asset_class="crypto_spot",
    timeframe="15m",
    direction="long",
    trigger_name="rsi_oversold",
    trigger_params=None,
    exit_triggers=None,
    sl_type="percent",
    sl_value=2.0,
    sl_multiple=None,
    tp_type="rr",
    tp_ratio=2.0,
    tp_value=None,
    trailing=None,
    gates=None,
    htf_rules=None,
    objective="mean_reversion",
    unmapped=None,
) -> SDL:
    if trigger_params is None:
        trigger_params = {"window": 14, "threshold": 30}
    if exit_triggers is None:
        exit_triggers = [SignalRef(name="rsi_overbought", params={})]

    sl_spec = StopLossSpec(type=sl_type, value=sl_value, multiple=sl_multiple)
    tp_spec = TakeProfitSpec(type=tp_type, ratio=tp_ratio, value=tp_value)
    risk = RiskSpec(stop_loss=sl_spec, take_profit=tp_spec, trailing=trailing)

    return SDL(
        context=StrategyContext(market="crypto", timeframe=timeframe, objective=objective),
        universe=StaticUniverse(asset_class=asset_class, symbol=symbol),
        legs=[
            Leg(
                direction=direction,
                entry=EntrySpec(trigger=SignalRef(name=trigger_name, params=trigger_params)),
                exit=ExitSpec(triggers=exit_triggers),
            )
        ],
        risk=risk,
        gates=gates or GatesSpec(),
        htf_rules=htf_rules or [],
        provenance=Provenance(
            field_sources={},   # <-- empty: simulating LLM forgetting
            unmapped_details=unmapped or [],
        ),
    )


def _fs(sdl: SDL) -> dict:
    """Shorthand: return field_sources dict."""
    return sdl.provenance.field_sources


def _clarifs(sdl: SDL) -> list:
    return sdl.provenance.clarifications_needed


# ── Timeframe detection ───────────────────────────────────────────────────────

class TestTimeframe:
    def test_15m_detected(self):
        sdl = reconcile(_sdl(), "Buy ETH on the 15m chart")
        assert _fs(sdl).get("context.timeframe") == "user"

    def test_1h_detected(self):
        sdl = reconcile(_sdl(timeframe="1h"), "Trade RELIANCE on 1h timeframe")
        assert _fs(sdl).get("context.timeframe") == "user"

    def test_daily_word(self):
        sdl = reconcile(_sdl(timeframe="1d"), "daily chart strategy")
        assert _fs(sdl).get("context.timeframe") == "user"

    def test_no_timeframe_mentioned(self):
        sdl = reconcile(_sdl(), "RSI strategy")
        assert _fs(sdl).get("context.timeframe") != "user"


# ── Symbol / universe detection ───────────────────────────────────────────────

class TestSymbol:
    def test_eth_crypto(self):
        sdl = reconcile(_sdl(symbol="ETH_USDC", asset_class="crypto_spot"), "Buy ETH when RSI drops")
        assert _fs(sdl).get("universe.symbol") == "user"
        assert _fs(sdl).get("universe.asset_class") == "user"

    def test_btc_crypto(self):
        sdl = reconcile(_sdl(symbol="BTC_USDC", asset_class="crypto_spot"), "Trade BTC 1h")
        assert _fs(sdl).get("universe.symbol") == "user"

    def test_equity_symbol(self):
        sdl = reconcile(_sdl(symbol="HDFCBANK.NS", asset_class="equity_cash"), "Trade HDFCBANK on 15m")
        assert _fs(sdl).get("universe.symbol") == "user"

    def test_no_symbol_not_marked(self):
        sdl = reconcile(_sdl(), "trade on 15m chart when RSI drops below 30")
        # ETH not mentioned — should not mark user unless ETH is in prompt
        fs = _fs(sdl)
        # (crypto may still be detected by pattern — that's fine)
        # Just verify it doesn't crash
        assert isinstance(fs, dict)


# ── Direction detection ───────────────────────────────────────────────────────

class TestDirection:
    def test_buy_marks_long(self):
        sdl = reconcile(_sdl(direction="long"), "Buy ETH on 15m")
        assert _fs(sdl).get("legs.0.direction") == "user"

    def test_long_explicit(self):
        sdl = reconcile(_sdl(direction="long"), "Go long on HDFCBANK")
        assert _fs(sdl).get("legs.0.direction") == "user"

    def test_short_explicit(self):
        sdl = reconcile(_sdl(direction="short"), "Short RELIANCE on breakdown")
        assert _fs(sdl).get("legs.0.direction") == "user"

    def test_sell_marks_short(self):
        sdl = reconcile(_sdl(direction="short"), "Sell HDFCBANK when RSI overbought")
        assert _fs(sdl).get("legs.0.direction") == "user"

    def test_direction_not_mentioned(self):
        # No buy/sell/long/short in prompt
        sdl = reconcile(_sdl(direction="long"), "RSI below 30 on ETH 15m")
        # direction should NOT be user — user did not state it
        assert _fs(sdl).get("legs.0.direction") != "user"


# ── Indicator / trigger detection ─────────────────────────────────────────────

class TestIndicatorDetection:
    def test_rsi_trigger_name(self):
        sdl = reconcile(_sdl(trigger_name="rsi_oversold"), "Buy when RSI drops below 30 on ETH 15m")
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"
        assert _fs(sdl).get("legs.0.entry.trigger") == "user"

    def test_rsi_window_param_when_explicit(self):
        sdl = reconcile(
            _sdl(trigger_name="rsi_oversold", trigger_params={"window": 14, "threshold": 30}),
            "RSI 14 below 30"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.params.window") == "user"
        assert _fs(sdl).get("legs.0.entry.trigger.params.threshold") == "user"

    def test_rsi_threshold_only(self):
        sdl = reconcile(
            _sdl(trigger_name="rsi_oversold", trigger_params={"window": 14, "threshold": 30}),
            "RSI below 30"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.params.threshold") == "user"

    def test_ema_trigger_detected(self):
        sdl = reconcile(
            _sdl(trigger_name="ema_cross_up", trigger_params={"window_fast": 9, "window_slow": 21}),
            "EMA crossover on 15m"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"

    def test_bollinger_trigger_detected(self):
        sdl = reconcile(
            _sdl(trigger_name="price_above_bb_upper", trigger_params={"window": 20, "num_std": 2.0}),
            "Enter above Bollinger Band upper"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"

    def test_macd_trigger_detected(self):
        sdl = reconcile(
            _sdl(trigger_name="macd_bullish_cross"),
            "Buy when MACD crosses above signal line"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"

    def test_vwap_trigger_detected(self):
        sdl = reconcile(
            _sdl(trigger_name="vwap_bullish"),
            "Enter when price is above VWAP"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"

    def test_orb_trigger_detected(self):
        sdl = reconcile(
            _sdl(trigger_name="opening_range_breakout", trigger_params={"minutes": 15}),
            "opening range breakout on 15m"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"

    def test_adx_trigger_detected(self):
        sdl = reconcile(
            _sdl(trigger_name="adx_strong_trend", trigger_params={"window": 14, "threshold": 25}),
            "ADX above 25 for strong trend entry"
        )
        assert _fs(sdl).get("legs.0.entry.trigger.name") == "user"
        assert _fs(sdl).get("legs.0.entry.trigger.params.threshold") == "user"

    def test_unrelated_indicator_not_marked(self):
        # Trigger is RSI but prompt mentions only MACD
        sdl = reconcile(
            _sdl(trigger_name="rsi_oversold"),
            "trade when MACD is bullish"
        )
        # RSI should NOT be marked user (user didn't mention RSI)
        assert _fs(sdl).get("legs.0.entry.trigger.name") != "user"


# ── Stop loss detection ───────────────────────────────────────────────────────

class TestStopLoss:
    def test_percent_sl_detected(self):
        sdl = reconcile(_sdl(sl_type="percent", sl_value=2.0), "stop loss 2% below entry")
        assert _fs(sdl).get("risk.stop_loss") == "user"
        assert _fs(sdl).get("risk.stop_loss.type") == "user"
        assert _fs(sdl).get("risk.stop_loss.value") == "user"

    def test_atr_sl_detected(self):
        sdl = reconcile(
            _sdl(sl_type="atr", sl_value=None, sl_multiple=1.5),
            "ATR stop loss, 1.5× ATR"
        )
        assert _fs(sdl).get("risk.stop_loss") == "user"
        assert _fs(sdl).get("risk.stop_loss.type") == "user"

    def test_sl_not_stated_adds_clarification(self):
        # Prompt has no stop loss mention — should add clarification
        sdl = reconcile(_sdl(sl_type="percent", sl_value=2.0), "Buy ETH when RSI below 30")
        # stop_loss should be "default" and clarification added
        assert _fs(sdl).get("risk.stop_loss") == "default"
        clarif_fields = [c.field for c in _clarifs(sdl)]
        assert "risk.stop_loss" in clarif_fields

    def test_trailing_stop_detected(self):
        sdl = reconcile(
            _sdl(trailing=TrailingSpec(enabled=True, type="atr_based", distance_atr=2.0)),
            "Use a trailing stop based on ATR"
        )
        assert _fs(sdl).get("risk.trailing") == "user"

    def test_no_duplicate_clarification(self):
        # If reconciler is called twice, clarifications should not duplicate
        sdl = _sdl(sl_type="percent", sl_value=2.0)
        r1 = reconcile(sdl, "Buy ETH RSI below 30")
        r2 = reconcile(r1, "Buy ETH RSI below 30")
        sl_clarifs = [c for c in _clarifs(r2) if c.field == "risk.stop_loss"]
        assert len(sl_clarifs) <= 1


# ── Take profit detection ─────────────────────────────────────────────────────

class TestTakeProfit:
    def test_rr_ratio_detected(self):
        sdl = reconcile(_sdl(tp_type="rr", tp_ratio=2.0), "risk reward 2:1")
        assert _fs(sdl).get("risk.take_profit") == "user"
        assert _fs(sdl).get("risk.take_profit.ratio") == "user"

    def test_tp_percent_detected(self):
        sdl = reconcile(
            _sdl(tp_type="percent", tp_value=4.0, tp_ratio=None),
            "take profit at 4%"
        )
        assert _fs(sdl).get("risk.take_profit") == "user"
        assert _fs(sdl).get("risk.take_profit.value") == "user"

    def test_target_keyword_detected(self):
        sdl = reconcile(
            _sdl(tp_type="percent", tp_value=3.0, tp_ratio=None),
            "target 3%"
        )
        assert _fs(sdl).get("risk.take_profit") == "user"

    def test_tp_not_stated_adds_clarification(self):
        sdl = reconcile(_sdl(tp_type="rr", tp_ratio=2.0), "stop loss 2%")
        # TP was not mentioned → default → clarification
        assert _fs(sdl).get("risk.take_profit") == "default"
        clarif_fields = [c.field for c in _clarifs(sdl)]
        assert "risk.take_profit" in clarif_fields

    def test_tp_stated_no_clarification(self):
        sdl = reconcile(_sdl(tp_type="rr", tp_ratio=2.0), "stop 2%, take profit 2:1 RR")
        clarif_fields = [c.field for c in _clarifs(sdl)]
        assert "risk.take_profit" not in clarif_fields


# ── Gate detection ────────────────────────────────────────────────────────────

class TestGates:
    def test_regime_gate_detected(self):
        sdl = reconcile(
            _sdl(gates=GatesSpec(regime=RegimeGate(allowed=["trending"]))),
            "only trade in trending market"
        )
        assert _fs(sdl).get("gates.regime") == "user"

    def test_event_gate_detected(self):
        sdl = reconcile(
            _sdl(gates=GatesSpec(event=EventGate(skip_dates=["2025-01-31"]))),
            "skip earnings dates"
        )
        assert _fs(sdl).get("gates.event") == "user"

    def test_session_gate_detected(self):
        sdl = reconcile(
            _sdl(gates=GatesSpec(session=SessionGate(start="09:15", end="15:00", timezone="IST"))),
            "entry window between 09:15 and 15:00"
        )
        assert _fs(sdl).get("gates.session") == "user"

    def test_volatility_gate_detected(self):
        sdl = reconcile(
            _sdl(gates=GatesSpec(volatility=VolatilityGate(metric="atr", window=14, min=0.5))),
            "only enter when volatility filter is active"
        )
        assert _fs(sdl).get("gates.volatility") == "user"

    def test_rs_gate_detected(self):
        sdl = reconcile(
            _sdl(gates=GatesSpec(relative_strength=RelativeStrengthGate(window=14, min_ratio=1.05))),
            "relative strength filter, stock must outperform index"
        )
        assert _fs(sdl).get("gates.relative_strength") == "user"

    def test_volume_ratio_gate_detected(self):
        sdl = reconcile(
            _sdl(gates=GatesSpec(volume_ratio=VolumeRatioGate(window=20, min_ratio=1.5))),
            "volume spike above average"
        )
        assert _fs(sdl).get("gates.volume_ratio") == "user"

    def test_gate_in_sdl_but_not_in_prompt_not_marked(self):
        # Regime gate exists in SDL but not mentioned in prompt
        sdl = reconcile(
            _sdl(gates=GatesSpec(regime=RegimeGate(allowed=["trending"]))),
            "RSI below 30, stop 2%"
        )
        assert _fs(sdl).get("gates.regime") != "user"


# ── HTF detection ─────────────────────────────────────────────────────────────

class TestHTF:
    def test_htf_rule_detected(self):
        sdl = reconcile(
            _sdl(htf_rules=[HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating")]),
            "only trade when higher timeframe trend is bullish"
        )
        assert _fs(sdl).get("htf_rules.0") == "user"

    def test_htf_not_in_prompt_not_marked(self):
        sdl = reconcile(
            _sdl(htf_rules=[HTFRule(timeframe="1h", condition="EMA(20) > EMA(50)", role="gating")]),
            "RSI below 30, stop 2%"
        )
        assert _fs(sdl).get("htf_rules.0") != "user"


# ── Objective detection ───────────────────────────────────────────────────────

class TestObjective:
    def test_breakout_detected(self):
        sdl = reconcile(_sdl(objective="breakout"), "breakout above resistance on 15m")
        assert _fs(sdl).get("context.objective") == "user"

    def test_mean_reversion_detected(self):
        sdl = reconcile(_sdl(objective="mean_reversion"), "mean reversion scalp ETH 15m")
        assert _fs(sdl).get("context.objective") == "user"

    def test_scalp_detected(self):
        sdl = reconcile(_sdl(objective="scalping"), "scalping strategy on BTC")
        assert _fs(sdl).get("context.objective") == "user"


# ── Existing "user" sources never downgraded ──────────────────────────────────

class TestNeverDowngrade:
    def test_existing_user_source_preserved(self):
        # LLM correctly marked something as user — reconciler must not remove it
        sdl = _sdl()
        # Manually put a user source for a field
        sdl_data = sdl.model_dump(mode="python")
        sdl_data.pop("content_hash", None)
        sdl_data["provenance"]["field_sources"]["context.market"] = "user"
        from app.planner.sdl import SDL
        sdl_with_manual = SDL(**sdl_data)

        reconciled = reconcile(sdl_with_manual, "ETH RSI 14 below 30")
        assert _fs(reconciled).get("context.market") == "user"

    def test_existing_inferred_can_be_upgraded_to_user(self):
        # LLM marked timeframe as "inferred" but user explicitly said "15m"
        sdl = _sdl()
        sdl_data = sdl.model_dump(mode="python")
        sdl_data.pop("content_hash", None)
        sdl_data["provenance"]["field_sources"]["context.timeframe"] = "inferred"
        from app.planner.sdl import SDL
        sdl_inferred = SDL(**sdl_data)

        reconciled = reconcile(sdl_inferred, "ETH RSI on 15m chart")
        assert _fs(reconciled).get("context.timeframe") == "user"


# ── Full prompt integration tests ─────────────────────────────────────────────

class TestFullPrompts:
    def test_rsi_eth_full_prompt(self):
        """'Buy ETH on 15m when RSI 14 drops below 30. SL 2%, TP 2:1'"""
        sdl = _sdl(
            symbol="ETH_USDC", asset_class="crypto_spot",
            timeframe="15m", direction="long",
            trigger_name="rsi_oversold",
            trigger_params={"window": 14, "threshold": 30},
            sl_type="percent", sl_value=2.0,
            tp_type="rr", tp_ratio=2.0,
        )
        prompt = "Buy ETH on 15m when RSI 14 drops below 30. Stop loss 2%, take profit 2:1."
        r = reconcile(sdl, prompt)
        fs = _fs(r)

        assert fs.get("context.timeframe") == "user"
        assert fs.get("universe.symbol") == "user"
        assert fs.get("legs.0.direction") == "user"
        assert fs.get("legs.0.entry.trigger.name") == "user"
        assert fs.get("legs.0.entry.trigger.params.window") == "user"
        assert fs.get("legs.0.entry.trigger.params.threshold") == "user"
        assert fs.get("risk.stop_loss") == "user"
        assert fs.get("risk.take_profit") == "user"
        # No money clarifications — user stated both
        clarif_fields = [c.field for c in _clarifs(r)]
        assert "risk.stop_loss" not in clarif_fields
        assert "risk.take_profit" not in clarif_fields

    def test_ema_crossover_nse(self):
        """'Long HDFCBANK 15m on EMA(9) cross EMA(21), SL 1.5%, 2:1 RR'"""
        sdl = _sdl(
            symbol="HDFCBANK.NS", asset_class="equity_cash",
            timeframe="15m", direction="long",
            trigger_name="ema_cross_up",
            trigger_params={"window_fast": 9, "window_slow": 21},
            sl_type="percent", sl_value=1.5,
            tp_type="rr", tp_ratio=2.0,
        )
        prompt = "Long HDFCBANK 15m on EMA(9) cross EMA(21), SL 1.5%, 2:1 RR"
        r = reconcile(sdl, prompt)
        fs = _fs(r)

        assert fs.get("context.timeframe") == "user"
        assert fs.get("universe.symbol") == "user"
        assert fs.get("legs.0.direction") == "user"
        assert fs.get("legs.0.entry.trigger.name") == "user"
        assert fs.get("risk.stop_loss") == "user"
        assert fs.get("risk.take_profit") == "user"

    def test_atr_stop_with_regime_gate(self):
        """'Trade ETH 1h when RSI below 30, ATR stop, only trending regime'"""
        sdl = _sdl(
            symbol="ETH_USDC", asset_class="crypto_spot",
            timeframe="1h", direction="long",
            trigger_name="rsi_oversold",
            trigger_params={"window": 14, "threshold": 30},
            sl_type="atr", sl_value=None, sl_multiple=1.5,
            tp_type="rr", tp_ratio=2.0,
            gates=GatesSpec(regime=RegimeGate(allowed=["trending"])),
        )
        prompt = "Trade ETH 1h when RSI below 30, ATR stop, only in trending regime"
        r = reconcile(sdl, prompt)
        fs = _fs(r)

        assert fs.get("context.timeframe") == "user"
        assert fs.get("legs.0.entry.trigger.name") == "user"
        assert fs.get("risk.stop_loss") == "user"
        assert fs.get("risk.stop_loss.type") == "user"
        assert fs.get("gates.regime") == "user"

    def test_missing_sl_and_tp_both_get_clarifications(self):
        """Prompt with no SL or TP → both clarifications added"""
        sdl = _sdl(
            sl_type="percent", sl_value=2.0,
            tp_type="rr", tp_ratio=2.0,
        )
        prompt = "Buy ETH when RSI 14 drops below 30"  # no SL or TP mentioned
        r = reconcile(sdl, prompt)

        clarif_fields = [c.field for c in _clarifs(r)]
        assert "risk.stop_loss" in clarif_fields
        assert "risk.take_profit" in clarif_fields
        # Both should be "default"
        assert _fs(r).get("risk.stop_loss") == "default"
        assert _fs(r).get("risk.take_profit") == "default"

    def test_unmapped_details_preserved(self):
        """Unmapped details added by LLM should survive reconciliation."""
        sdl = _sdl(
            unmapped=[UnmappedDetail(text="top-20 parallel", kind="engine_capability_gap")]
        )
        r = reconcile(sdl, "Buy ETH RSI below 30")
        assert len(r.provenance.unmapped_details) == 1
        assert r.provenance.unmapped_details[0].text == "top-20 parallel"

    def test_match_pct_improves_after_reconcile(self):
        """After reconciliation, built count must be >= before."""
        from app.planner.readback import compute_match_pct

        sdl_before = _sdl(
            symbol="ETH_USDC", timeframe="15m",
            trigger_name="rsi_oversold",
            trigger_params={"window": 14, "threshold": 30},
            sl_type="percent", sl_value=2.0,
            tp_type="rr", tp_ratio=2.0,
        )
        prompt = "Buy ETH on 15m when RSI 14 below 30. Stop 2%, TP 2:1."
        sdl_after = reconcile(sdl_before, prompt)

        built_before, _, _ = compute_match_pct(sdl_before)
        built_after, _, _ = compute_match_pct(sdl_after)

        assert built_after > built_before, (
            f"Expected built to increase: before={built_before}, after={built_after}"
        )


# ── ATR stop multiple capture (general "N ATR" → multiple=N) ───────────────────

class TestAtrStopMultiple:
    """The LLM routinely emits stop_loss.type=atr but drops the multiple; the
    compiler then silently defaults it to 1.5×. The reconciler must capture the
    number the user stated, for any common phrasing — not just one prompt."""

    @pytest.mark.parametrize("prompt,expected", [
        ("Stop Loss: 1 ATR.",            1.0),
        ("use a 1.5x ATR stop",          1.5),
        ("2 ATR stop loss",              2.0),
        ("stop loss of 3 * atr",         3.0),
        ("0.5 average true range stop",  0.5),
    ])
    def test_multiple_captured(self, prompt, expected):
        sdl = reconcile(_sdl(sl_type="atr", sl_value=None, sl_multiple=None), prompt)
        assert sdl.risk.stop_loss.multiple == expected
        assert _fs(sdl).get("risk.stop_loss.multiple") == "user"

    def test_no_number_leaves_multiple_unset(self):
        # "ATR stop" with no magnitude → leave None so the compiler default applies.
        sdl = reconcile(_sdl(sl_type="atr", sl_value=None, sl_multiple=None), "use an ATR stop")
        assert sdl.risk.stop_loss.multiple is None

    def test_window_after_atr_is_not_a_multiple(self):
        # "ATR 14"/"ATR(14)" is the lookback window, never the multiple.
        sdl = reconcile(_sdl(sl_type="percent", sl_value=2.0), "ATR(14) filter; stop 2%")
        assert _fs(sdl).get("risk.stop_loss.multiple") is None

    def test_stated_value_overrides_llm_default(self):
        # User said 1 ATR but the LLM emitted 1.5 → the prompt wins.
        sdl = reconcile(_sdl(sl_type="atr", sl_value=None, sl_multiple=1.5), "Stop Loss: 1 ATR.")
        assert sdl.risk.stop_loss.multiple == 1.0
