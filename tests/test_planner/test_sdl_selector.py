"""
tests/test_planner/test_sdl_selector.py — Phase 2 SDL selector tests.

All tests use set_llm_override() so no real LLM is called.
The override returns pre-written golden SDL JSON strings, ensuring:
  - same prompt → same SDL (via cache)
  - modify_sdl merges fields correctly (version++, parent_version)
  - provenance is threaded through correctly
  - explicit indicator wins rule is verifiable via golden fixtures
"""
import json
import asyncio

import pytest

from app.planner.sdl import (
    SDL,
    DynamicRank,
    DynamicUniverse,
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
    ClarificationNeeded,
)
from app.planner.sdl_selector import (
    clear_cache,
    compile_to_sdl,
    modify_sdl,
    set_llm_override,
)
from app.planner.catalog_schema import build_menu, invalidate_menu_cache


# ── Golden SDL fixtures ───────────────────────────────────────────────────────

def _eth_rsi_sdl_json() -> str:
    """Golden SDL: 'Buy ETH on 15m when RSI < 30, SL 2%, TP 2:1'"""
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
        provenance=Provenance(
            field_sources={
                "universe.symbol": "user",
                "legs.0.entry.trigger": "user",
                "risk.stop_loss": "user",
                "risk.take_profit": "user",
                "legs.0.exit": "inferred",
                "context.timeframe": "user",
            }
        ),
    )
    return sdl.model_dump_json()


def _bollinger_sdl_json() -> str:
    """Golden SDL: 'Enter above Bollinger Band upper, SL 1.5%'
    (Bollinger → bb_* card, NOT Keltner — explicit indicator wins)"""
    sdl = SDL(
        context=StrategyContext(market="crypto", timeframe="1h", objective="breakout"),
        universe=StaticUniverse(asset_class="crypto_spot", symbol="BTC_USDC"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(
                    trigger=SignalRef(name="price_above_bb_upper", params={"window": 20, "num_std": 2.0})
                ),
                exit=ExitSpec(),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=1.5),
        ),
        provenance=Provenance(
            field_sources={
                "universe.symbol": "user",
                "legs.0.entry.trigger": "user",
                "risk.stop_loss": "user",
            },
            clarifications_needed=[
                ClarificationNeeded(
                    field="risk.take_profit",
                    question="No take-profit given — use 2:1 RR?",
                    assumed_value="2:1",
                )
            ],
        ),
    )
    return sdl.model_dump_json()


def _nse_dynamic_sdl_json() -> str:
    """Golden SDL: dynamic NSE universe with ORB entry"""
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
                entry=EntrySpec(
                    trigger=SignalRef(name="opening_range_breakout", params={"minutes": 15})
                ),
                exit=ExitSpec(),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="atr", multiple=1.5, window=14),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        provenance=Provenance(
            field_sources={
                "universe": "user",
                "legs.0.entry.trigger": "user",
                "risk.stop_loss": "user",
                "risk.take_profit": "default",
            },
            clarifications_needed=[
                ClarificationNeeded(
                    field="risk.take_profit",
                    question="No target given — I used 2:1. OK?",
                    assumed_value="2:1",
                )
            ],
        ),
    )
    return sdl.model_dump_json()


def _modified_eth_sdl_json(original: SDL) -> str:
    """Modified SDL: change stop to 1% and add EMA filter."""
    from app.planner.sdl import GatesSpec, RegimeGate
    sdl = SDL(
        context=original.context,
        universe=original.universe,
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(
                    trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30}),
                    filters=[SignalRef(name="ema_above", params={"window_fast": 9, "window_slow": 21})],
                ),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=1.0),  # changed
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        gates=GatesSpec(regime=RegimeGate(allowed=["trending"])),
        provenance=Provenance(
            field_sources={
                "universe.symbol": "user",
                "legs.0.entry.trigger": "user",
                "risk.stop_loss": "user",
                "risk.take_profit": "user",
                "legs.0.exit": "inferred",
                "context.timeframe": "user",
                "legs.0.entry.filters.0": "user",
                "gates.regime": "user",
            }
        ),
        version=original.version,
        parent_version=original.parent_version,
    )
    return sdl.model_dump_json()


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Fixture setup ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_selector():
    """Clear cache and LLM override before each test."""
    clear_cache()
    set_llm_override(None)
    invalidate_menu_cache()
    yield
    clear_cache()
    set_llm_override(None)


# ── compile_to_sdl tests ──────────────────────────────────────────────────────

class TestCompileToSDL:
    def test_returns_sdl_object(self):
        golden = _eth_rsi_sdl_json()
        set_llm_override(lambda msgs, tools: golden)
        sdl = run(compile_to_sdl("Buy ETH on 15m when RSI < 30, SL 2%, TP 2:1", skip_flag=True))
        assert isinstance(sdl, SDL)

    def test_static_universe_resolved(self):
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH on 15m when RSI < 30", skip_flag=True))
        assert sdl.universe.type == "static"
        assert sdl.universe.symbol == "ETH_USDC"  # type: ignore[union-attr]

    def test_entry_trigger_preserved(self):
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH on 15m when RSI < 30", skip_flag=True))
        assert sdl.legs[0].entry.trigger.name == "rsi_oversold"
        assert sdl.legs[0].entry.trigger.params["window"] == 14
        assert sdl.legs[0].entry.trigger.params["threshold"] == 30

    def test_risk_preserved(self):
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH SL 2% TP 2:1", skip_flag=True))
        assert sdl.risk.stop_loss is not None
        assert sdl.risk.stop_loss.value == 2.0
        assert sdl.risk.take_profit is not None
        assert sdl.risk.take_profit.ratio == 2.0

    def test_provenance_user_fields(self):
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH SL 2%", skip_flag=True))
        fs = sdl.provenance.field_sources
        assert fs.get("universe.symbol") == "user"
        assert fs.get("risk.stop_loss") == "user"

    def test_provenance_inferred_exit(self):
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH SL 2%", skip_flag=True))
        assert sdl.provenance.field_sources.get("legs.0.exit") == "inferred"

    def test_content_hash_populated(self):
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH SL 2%", skip_flag=True))
        assert len(sdl.content_hash) == 64

    def test_dynamic_universe_preserved(self):
        set_llm_override(lambda msgs, tools: _nse_dynamic_sdl_json())
        sdl = run(compile_to_sdl("Trade highest rvol NSE stock above VWAP, ORB 15m", skip_flag=True))
        assert sdl.universe.type == "dynamic"
        u = sdl.universe
        assert u.asset_class == "equity_cash"      # type: ignore[union-attr]
        assert u.rank.by == "rvol"                  # type: ignore[union-attr]
        assert u.tie_break == "highest_relative_volume"  # type: ignore[union-attr]

    def test_clarification_threaded_through(self):
        set_llm_override(lambda msgs, tools: _nse_dynamic_sdl_json())
        # Prompt has no SL or TP → reconciler correctly adds clarifications for both
        sdl = run(compile_to_sdl("Trade highest rvol NSE stock", skip_flag=True))
        clarif_fields = [c.field for c in sdl.provenance.clarifications_needed]
        assert "risk.take_profit" in clarif_fields


class TestExplicitIndicatorWins:
    """Acceptance #2: Bollinger → Bollinger card, never Keltner/Donchian."""

    def test_bollinger_entry_uses_bb_card(self):
        set_llm_override(lambda msgs, tools: _bollinger_sdl_json())
        sdl = run(compile_to_sdl("Enter above Bollinger Band upper on BTC 1h", skip_flag=True))
        trigger_name = sdl.legs[0].entry.trigger.name
        assert "bb" in trigger_name.lower() or "bollinger" in trigger_name.lower(), (
            f"Expected BB card, got: {trigger_name}"
        )
        assert "keltner" not in trigger_name.lower()
        assert "donchian" not in trigger_name.lower()

    def test_bollinger_clarification_for_missing_tp(self):
        set_llm_override(lambda msgs, tools: _bollinger_sdl_json())
        sdl = run(compile_to_sdl("Enter above BB upper, SL 1.5%", skip_flag=True))
        assert len(sdl.provenance.clarifications_needed) >= 1


class TestSDLCache:
    """Acceptance #3: same prompt → same SDL (via cache)."""

    def test_same_prompt_returns_same_hash(self):
        call_count = 0

        def mock_llm(msgs, tools):
            nonlocal call_count
            call_count += 1
            return _eth_rsi_sdl_json()

        set_llm_override(mock_llm)
        prompt = "Buy ETH on 15m when RSI < 30, SL 2%, TP 2:1"
        sdl1 = run(compile_to_sdl(prompt, skip_flag=True))
        sdl2 = run(compile_to_sdl(prompt, skip_flag=True))

        assert sdl1.content_hash == sdl2.content_hash
        assert call_count == 1, "LLM should only be called once for the same prompt"

    def test_different_prompts_both_hit_llm(self):
        call_count = 0

        def mock_llm(msgs, tools):
            nonlocal call_count
            call_count += 1
            return _eth_rsi_sdl_json()

        set_llm_override(mock_llm)
        run(compile_to_sdl("Buy ETH on 15m when RSI < 30", skip_flag=True))
        run(compile_to_sdl("Buy BTC on 1h when RSI < 25", skip_flag=True))
        assert call_count == 2

    def test_cache_cleared_between_tests(self):
        call_count = 0

        def mock_llm(msgs, tools):
            nonlocal call_count
            call_count += 1
            return _eth_rsi_sdl_json()

        set_llm_override(mock_llm)
        run(compile_to_sdl("Buy ETH on 15m", skip_flag=True))
        assert call_count == 1  # fresh cache


# ── modify_sdl tests ──────────────────────────────────────────────────────────

class TestModifySDL:
    def _original_sdl(self) -> SDL:
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl("Buy ETH on 15m SL 2%", skip_flag=True))
        clear_cache()
        return sdl

    def test_version_incremented(self):
        original = self._original_sdl()
        assert original.version == 1

        set_llm_override(lambda msgs, tools: _modified_eth_sdl_json(original))
        modified = run(modify_sdl(original, "Change stop to 1% and add EMA filter", skip_flag=True))
        assert modified.version == 2
        assert modified.parent_version == 1

    def test_parent_version_set(self):
        original = self._original_sdl()
        set_llm_override(lambda msgs, tools: _modified_eth_sdl_json(original))
        modified = run(modify_sdl(original, "Change stop to 1%", skip_flag=True))
        assert modified.parent_version == original.version

    def test_changed_fields_have_user_source(self):
        original = self._original_sdl()
        set_llm_override(lambda msgs, tools: _modified_eth_sdl_json(original))
        modified = run(modify_sdl(original, "Change stop to 1%", skip_flag=True))
        assert modified.provenance.field_sources.get("risk.stop_loss") == "user"

    def test_modified_sl_value(self):
        original = self._original_sdl()
        set_llm_override(lambda msgs, tools: _modified_eth_sdl_json(original))
        modified = run(modify_sdl(original, "Change stop to 1%", skip_flag=True))
        assert modified.risk.stop_loss is not None
        assert modified.risk.stop_loss.value == 1.0

    def test_new_filter_added(self):
        original = self._original_sdl()
        set_llm_override(lambda msgs, tools: _modified_eth_sdl_json(original))
        modified = run(modify_sdl(original, "Add EMA filter", skip_flag=True))
        filters = modified.legs[0].entry.filters
        assert len(filters) == 1
        assert "ema" in filters[0].name.lower()

    def test_content_hash_changes_after_modify(self):
        original = self._original_sdl()
        h_before = original.content_hash
        set_llm_override(lambda msgs, tools: _modified_eth_sdl_json(original))
        modified = run(modify_sdl(original, "Change stop to 1%", skip_flag=True))
        assert modified.content_hash != h_before

    def test_modify_cache_works(self):
        original = self._original_sdl()
        call_count = 0

        def mock_llm(msgs, tools):
            nonlocal call_count
            call_count += 1
            return _modified_eth_sdl_json(original)

        set_llm_override(mock_llm)
        change = "Change stop to 1%"
        run(modify_sdl(original, change, skip_flag=True))
        run(modify_sdl(original, change, skip_flag=True))
        assert call_count == 1, "Second call should hit the modify cache"


# ── Flag gate tests ───────────────────────────────────────────────────────────

class TestFlagGate:
    def test_compile_raises_without_flag(self, monkeypatch):
        monkeypatch.delenv("SDL_SELECTOR_ENABLED", raising=False)
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        with pytest.raises(RuntimeError, match="SDL_SELECTOR_ENABLED"):
            run(compile_to_sdl("Buy ETH", skip_flag=False))

    def test_modify_raises_without_flag(self, monkeypatch):
        monkeypatch.delenv("SDL_SELECTOR_ENABLED", raising=False)
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        original = run(compile_to_sdl("Buy ETH", skip_flag=True))
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        with pytest.raises(RuntimeError, match="SDL_SELECTOR_ENABLED"):
            run(modify_sdl(original, "change stop", skip_flag=False))


# ── asset_class normalization (regression: LLM emitted 'crypto' not 'crypto_spot') ──

import pytest as _pytest
from app.planner.sdl_selector import _normalize_sdl_data, _parse_sdl_json


@_pytest.mark.parametrize("given,expected", [
    ("crypto", "crypto_spot"),
    ("spot", "crypto_spot"),
    ("perp", "crypto_spot"),
    ("NSE", "equity_cash"),
    ("equity", "equity_cash"),
    ("indian_stocks", "equity_cash"),
])
def test_asset_class_alias_normalized(given, expected):
    data = {"context": {"market": "crypto"},
            "universe": {"type": "static", "asset_class": given, "symbol": "X"}}
    assert _normalize_sdl_data(data)["universe"]["asset_class"] == expected


def test_asset_class_derived_from_market_when_missing():
    data = {"context": {"market": "crypto"}, "universe": {"type": "static", "symbol": "BTC_USDT"}}
    assert _normalize_sdl_data(data)["universe"]["asset_class"] == "crypto_spot"
    data2 = {"context": {"market": "indian_stocks"}, "universe": {"type": "static", "symbol": "RELIANCE.NS"}}
    assert _normalize_sdl_data(data2)["universe"]["asset_class"] == "equity_cash"


def test_parse_sdl_json_accepts_crypto_alias():
    """The exact bug: a full SDL whose universe.asset_class='crypto' must parse,
    not raise a literal_error (which previously forced the legacy fallback)."""
    raw = json.dumps({
        "context": {"market": "crypto", "timeframe": "1h", "objective": "intraday"},
        "universe": {"type": "static", "asset_class": "crypto", "symbol": "ETH_USDC"},
        "legs": [{
            "direction": "long",
            "entry": {"trigger": {"name": "macd_bullish_cross", "params": {}}, "filters": []},
            "exit": {"triggers": []},
        }],
        "risk": {}, "gates": {}, "htf_rules": [],
        "provenance": {"field_sources": {}, "unmapped_details": [], "clarifications_needed": []},
        "version": 1,
    })
    sdl = _parse_sdl_json(raw)
    assert sdl.universe.asset_class == "crypto_spot"


# ── Signal-name repair (symmetric naming the catalog is inconsistent about) ────

from app.planner.sdl_selector import (
    _canonical_card_name,
    _normalize_sdl_data,
)
from app.kb import kb as _kb


class TestSignalNameRepair:
    _VALID = set(_kb.signals.keys())

    def test_price_above_sma_resolves_to_is_above_sma(self):
        # The exact hallucination from the Supertrend prompt: `price_above_sma`
        # doesn't exist (the card is named `is_above_sma`).
        assert _canonical_card_name("price_above_sma", self._VALID) == "is_above_sma"

    def test_is_below_sma_resolves_to_price_below_sma(self):
        assert _canonical_card_name("is_below_sma", self._VALID) == "price_below_sma"

    def test_real_name_needs_no_repair(self):
        # A name that exists is never rewritten (function only fires on misses).
        assert _canonical_card_name("rsi_oversold", self._VALID) is None

    def test_unknown_name_left_for_validator(self):
        # No symmetric variant exists → return None so the validator flags it.
        assert _canonical_card_name("price_above_unicorn", self._VALID) is None

    def test_never_flips_direction(self):
        # above must never be repaired to a below card (would invert intent).
        out = _canonical_card_name("price_above_sma", self._VALID)
        assert out is None or "below" not in out

    def test_normalize_repairs_filter_name_in_place(self):
        data = {
            "context": {"market": "crypto", "timeframe": "15m", "objective": "trend_following"},
            "universe": {"type": "static", "asset_class": "crypto_spot", "symbol": "ETH_USDC"},
            "legs": [{
                "direction": "long",
                "entry": {
                    "trigger": {"name": "supertrend_bullish", "params": {}},
                    "filters": [{"name": "price_above_sma", "params": {"window": 10}}],
                },
                "exit": {"triggers": [], "filters": []},
            }],
        }
        out = _normalize_sdl_data(data)
        assert out["legs"][0]["entry"]["filters"][0]["name"] == "is_above_sma"


class TestMissingBelowMaCardsAdded:
    @pytest.mark.parametrize("name,formula", [
        ("price_below_dema", "CLOSE < DEMA"),
        ("price_below_tema", "CLOSE < TEMA"),
        ("price_below_wma",  "CLOSE < WMA"),
    ])
    def test_below_ma_card_present(self, name, formula):
        card = _kb.signals.get(name)
        assert card is not None, f"{name} not loaded"
        assert formula in card.formula


# ── Selector reliability: retry + no cache poisoning (root cause of legacy fallback) ──

from app.planner.sdl_selector import _cache, _cache_key
from app.planner.catalog_schema import build_menu as _build_menu


class TestSelectorReliability:
    """A transient LLM blip (rate-limit / truncated JSON / empty response) used to
    kill the SDL attempt and silently drop chat to the legacy pipeline — and a
    truncated response got cached, replaying the failure forever. These pin the
    retry + cache-after-parse fixes."""

    _PROMPT = "Buy ETH on 15m when RSI < 30, SL 2%, TP 2:1"

    def test_retry_recovers_from_transient_failure(self):
        calls = {"n": 0}
        def flaky(msgs, tools):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient blip")
            return _eth_rsi_sdl_json()
        set_llm_override(flaky)
        sdl = run(compile_to_sdl(self._PROMPT, skip_flag=True))
        assert sdl.legs[0].entry.trigger.name == "rsi_oversold"
        assert calls["n"] == 3   # retried twice, succeeded on the third

    def test_empty_response_raises_not_silent(self):
        # An empty "{}" must surface as an error, never a silently-broken SDL.
        set_llm_override(lambda msgs, tools: "{}")
        with pytest.raises(Exception):
            run(compile_to_sdl(self._PROMPT, skip_flag=True))

    def test_broken_response_is_not_cached(self):
        set_llm_override(lambda msgs, tools: "{}")
        try:
            run(compile_to_sdl(self._PROMPT, skip_flag=True))
        except Exception:
            pass
        key = _cache_key(self._PROMPT, _build_menu().catalog_version)
        assert key not in _cache   # cache must stay clean → no poisoning

    def test_truncated_json_retried_then_raised(self):
        calls = {"n": 0}
        def truncated(msgs, tools):
            calls["n"] += 1
            return '{"context":{"market":"crypto"'   # cut off mid-object
        set_llm_override(truncated)
        with pytest.raises(Exception):
            run(compile_to_sdl(self._PROMPT, skip_flag=True))
        assert calls["n"] == 3   # all retries exhausted before giving up

    def test_good_call_succeeds_normally(self):
        # Sanity: the happy path still works (no regression from the retry wrapper).
        set_llm_override(lambda msgs, tools: _eth_rsi_sdl_json())
        sdl = run(compile_to_sdl(self._PROMPT, skip_flag=True))
        assert len(sdl.legs) == 1
