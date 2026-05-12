"""Hard-filter property tests."""
from __future__ import annotations

import pytest

from app.kb import kb
from app.kb.schemas import Intent
from app.planner.filters import filter_candidates, opposite_direction


@pytest.fixture
def default_intent() -> Intent:
    return Intent(**kb.intent_taxonomy.defaults)


@pytest.fixture
def all_cards():
    return list(kb.signals.values())


@pytest.mark.parametrize("sentiment", ["bullish", "bearish"])
@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "1h", "1d"])
def test_entry_trigger_pool_has_no_opposite_direction(sentiment, timeframe, default_intent, all_cards):
    survivors, _ = filter_candidates(
        all_cards, role="entry_trigger", sentiment=sentiment,
        timeframe=timeframe, intent=default_intent,
    )
    forbidden = opposite_direction(sentiment)
    for card in survivors:
        assert card.direction != forbidden, (
            f"entry_trigger pool for sentiment={sentiment} contains "
            f"{card.name} with direction={card.direction}"
        )


@pytest.mark.parametrize("sentiment", ["bullish", "bearish"])
def test_exit_trigger_pool_is_opposite_or_neutral(sentiment, default_intent, all_cards):
    survivors, _ = filter_candidates(
        all_cards, role="exit_trigger", sentiment=sentiment,
        timeframe="15m", intent=default_intent,
    )
    expected = opposite_direction(sentiment)
    for card in survivors:
        assert card.direction in {expected, "neutral"}, (
            f"exit_trigger pool for sentiment={sentiment} contains "
            f"{card.name} with wrong direction={card.direction}"
        )


def test_unsupported_timeframe_excluded(default_intent, all_cards):
    """vwap_bullish has unsupported_on=[1d]. It must be eliminated for 1d."""
    survivors, eliminations = filter_candidates(
        all_cards, role="entry_trigger", sentiment="bullish",
        timeframe="1d", intent=default_intent,
    )
    assert all(c.name != "vwap_bullish" for c in survivors)
    eliminated_names = [s for s, _ in eliminations]
    assert "vwap_bullish" in eliminated_names


def test_high_frequency_eliminates_crossover_entries(all_cards):
    intent = Intent(
        hold_horizon="seconds", frequency="very_high", profit_size="small",
        style="scalping", risk_appetite="conservative",
    )
    survivors, _ = filter_candidates(
        all_cards, role="entry_trigger", sentiment="bullish",
        timeframe="1m", intent=intent,
    )
    survivor_names = {c.name for c in survivors}
    assert "ema_cross_up" not in survivor_names
    assert "macd_bullish_cross" not in survivor_names
    # Regime signals should still be present.
    assert "ema_above" in survivor_names or "rsi_above_50" in survivor_names
