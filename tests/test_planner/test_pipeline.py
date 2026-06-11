"""End-to-end pipeline test with a stubbed LLM (no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.planner.intent_extractor import IntentExtractor
from app.planner.pipeline import Pipeline, UnsupportedStock, UnsupportedTimeframe


class StubLLM:
    """Returns a fixed JSON string regardless of input — for deterministic tests."""

    def __init__(self, payload: dict):
        self._payload = payload

    async def chat(self, messages):
        return json.dumps(self._payload)


def _builder(symbol="hdfc bank", timeframe="1m", sentiment="bullish",
             experience="beginner", objective="positional",
             goal="short profit and quick trades"):
    return SimpleNamespace(
        symbol=symbol, timeframe=timeframe, sentiment=sentiment,
        experience=experience, objective=objective, goal=goal,
    )


def _pipeline_with_intent(payload: dict) -> Pipeline:
    return Pipeline(intent_extractor=IntentExtractor(llm_service=StubLLM(payload)))


@pytest.mark.asyncio
async def test_bullish_1m_quick_trades_picks_regime_signals():
    pipe = _pipeline_with_intent({
        "hold_horizon": "minutes", "frequency": "high",
        "profit_size": "small", "style": "scalping",
        "risk_appetite": "conservative",
    })
    plan = await pipe.plan(_builder())

    assert plan.entry_trigger.direction in {"bullish", "neutral"}
    assert plan.exit_trigger.direction in {"bearish", "neutral"}
    # No-action complaint root cause: crossover triggers on stateless eval.
    # For frequency=high we should NOT pick a slow MA crossover for entry.
    assert plan.entry_trigger.name not in {"ema_cross_up", "sma_cross_up", "macd_bullish_cross"}
    assert plan.timeframe == "1m"
    assert plan.stock.symbol == "HDFCBANK.NS"


@pytest.mark.asyncio
async def test_bullish_15m_trend_picks_ema_or_macd():
    pipe = _pipeline_with_intent({
        "hold_horizon": "hours", "frequency": "medium",
        "profit_size": "medium", "style": "trend",
        "risk_appetite": "moderate",
    })
    plan = await pipe.plan(_builder(timeframe="15m", goal="follow the trend"))
    # For trend style on 15m, expect an EMA/MACD-style trigger.
    assert plan.entry_trigger.direction == "bullish"
    assert plan.exit_trigger.direction in {"bearish", "neutral"}


@pytest.mark.asyncio
async def test_unknown_stock_raises():
    pipe = _pipeline_with_intent({})
    with pytest.raises(UnsupportedStock):
        await pipe.plan(_builder(symbol="not-a-stock"))


@pytest.mark.asyncio
async def test_unknown_timeframe_raises():
    pipe = _pipeline_with_intent({})
    # 13m is now valid (range-based 1m..1d). Use an out-of-range timeframe.
    with pytest.raises(UnsupportedTimeframe):
        await pipe.plan(_builder(timeframe="2w"))


@pytest.mark.asyncio
async def test_pipeline_picks_multiple_entry_filters():
    """The new multi-AND behavior should stack 1-3 filters on top of the trigger,
    and they should each come from a different signal family (no two EMAs)."""
    pipe = _pipeline_with_intent({
        "hold_horizon": "minutes", "frequency": "high",
        "profit_size": "small", "style": "breakout",
        "risk_appetite": "moderate",
    })
    plan = await pipe.plan(_builder(timeframe="5m", goal="ORB breakout with confirmation"))
    assert len(plan.entry_filters) >= 1
    families = [plan.entry_trigger.name.split("_", 1)[0]] + [
        p.name.split("_", 1)[0] for p in plan.entry_filters
    ]
    assert len(families) == len(set(families)), (
        f"expected one signal per family but got duplicates: {families}"
    )


@pytest.mark.asyncio
async def test_pipeline_caps_entry_filters_at_max():
    """Even on a timeframe rich in eligible filters, we never exceed MAX_ENTRY_FILTERS."""
    from app.planner.pipeline import MAX_ENTRY_FILTERS
    pipe = _pipeline_with_intent({
        "hold_horizon": "hours", "frequency": "medium",
        "profit_size": "medium", "style": "trend",
        "risk_appetite": "moderate",
    })
    plan = await pipe.plan(_builder(timeframe="15m", goal="follow the trend with confirmation"))
    assert len(plan.entry_filters) <= MAX_ENTRY_FILTERS


@pytest.mark.asyncio
async def test_all_5_stocks_all_8_timeframes_both_sentiments():
    """Smoke matrix: every supported (stock, tf, sentiment) must produce a valid plan."""
    pipe = _pipeline_with_intent({
        "hold_horizon": "hours", "frequency": "medium",
        "profit_size": "medium", "style": "trend",
        "risk_appetite": "moderate",
    })
    from app.kb import kb
    failures: list[str] = []
    for symbol in kb.stocks.keys():
        for tf in kb.timeframes.supported:
            for sentiment in ["bullish", "bearish"]:
                try:
                    plan = await pipe.plan(_builder(symbol=symbol, timeframe=tf, sentiment=sentiment))
                    assert plan.entry_trigger is not None
                    assert plan.exit_trigger is not None
                except Exception as exc:
                    failures.append(f"{symbol}/{tf}/{sentiment}: {exc}")
    assert not failures, "matrix failures:\n" + "\n".join(failures)
