"""Soft-rank scoring tests."""
from __future__ import annotations

from app.kb import kb
from app.kb.schemas import Intent
from app.planner import ranker


def _intent(style: str = "trend") -> Intent:
    base = dict(kb.intent_taxonomy.defaults)
    base["style"] = style
    return Intent(**base)


def test_intent_style_dominates_ranking_for_pure_trend():
    cards = [c for c in kb.signals.values() if "entry_trigger" in c.roles and c.direction == "bullish"]
    ranking = ranker.rank(cards, intent=_intent("trend"), experience="intermediate", timeframe="15m")
    top3 = [r.signal for r in ranking[:3]]
    # Trend-aligned signals should dominate the top of the ranking.
    assert any(s in {"ema_cross_up", "ema_above", "macd_bullish_cross", "macd_positive"} for s in top3)


def test_avoid_family_pushes_same_family_down():
    cards = [c for c in kb.signals.values() if "entry_filter" in c.roles and c.direction in {"bullish", "neutral"}]
    no_avoid = ranker.rank(cards, intent=_intent("trend"), experience="intermediate", timeframe="15m")
    with_avoid = ranker.rank(cards, intent=_intent("trend"), experience="intermediate", timeframe="15m", avoid_family="ema")
    # With avoid_family="ema", any ema_* card must score lower than without.
    no_avoid_scores = {r.signal: r.score for r in no_avoid}
    with_avoid_scores = {r.signal: r.score for r in with_avoid}
    for name, score in with_avoid_scores.items():
        if name.startswith("ema"):
            assert score < no_avoid_scores[name]


def test_pick_returns_a_card_from_input_list():
    cards = [c for c in kb.signals.values() if "entry_trigger" in c.roles and c.direction == "bullish"]
    winner, ranking = ranker.pick(cards, intent=_intent("trend"), experience="beginner", timeframe="15m")
    names = {c.name for c in cards}
    assert winner.name in names
    assert ranking[0].signal == winner.name
