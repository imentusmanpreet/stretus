"""
app/planner/ranker.py — Soft ranking within hard-filtered pools.

Score factors (all multiplicative or additive, no magic constants buried in code):
    base                  1.0
  × experience_fit        card.experience_fit[experience]            (default 1.0)
  + intent_fit_bonus      card.intent_fit[intent.style]              (default 0.0)
  + tf_affinity           +0.5 if tf in works_best_on,
                          -0.3 if tf in weak_on
  + pair_bonus            +0.2 if pair_with.name in card.pairs_well_with
  + neutral_bonus         +0.1 if card.direction == 'neutral'
  - family_penalty        -1.0 if avoid_family == card.family
  - contradicts_penalty   -1.0 if pair_with.name in card.contradicts
"""
from __future__ import annotations

from app.kb.schemas import Intent, SignalCard
from app.planner.trace import ScoreRecord


def _score(
    card: SignalCard,
    *,
    intent: Intent,
    experience: str,
    timeframe: str,
    avoid_family: str | None,
    pair_with: SignalCard | None,
) -> ScoreRecord:
    components: dict[str, float] = {}

    base = 1.0
    components["base"] = base

    exp_fit = card.experience_fit.get(experience, 1.0)
    components["experience_fit"] = exp_fit

    intent_fit = card.intent_fit.get(intent.style, 0.0)
    components["intent_fit"] = intent_fit

    tf_affinity = 0.0
    if timeframe in card.works_best_on:
        tf_affinity += 0.5
    if timeframe in card.weak_on:
        tf_affinity -= 0.3
    components["tf_affinity"] = tf_affinity

    pair_bonus = 0.0
    contradicts_penalty = 0.0
    if pair_with is not None:
        if pair_with.name in card.pairs_well_with:
            pair_bonus = 0.2
        if pair_with.name in card.contradicts:
            contradicts_penalty = -1.0
    components["pair_bonus"] = pair_bonus
    components["contradicts_penalty"] = contradicts_penalty

    neutral_bonus = 0.1 if card.direction == "neutral" else 0.0
    components["neutral_bonus"] = neutral_bonus

    family_penalty = -1.0 if avoid_family and card.family == avoid_family else 0.0
    components["family_penalty"] = family_penalty

    score = (base * exp_fit) + intent_fit + tf_affinity + pair_bonus + contradicts_penalty + neutral_bonus + family_penalty
    return ScoreRecord(signal=card.name, score=round(score, 4), components=components)


def rank(
    cards: list[SignalCard],
    *,
    intent: Intent,
    experience: str,
    timeframe: str,
    avoid_family: str | None = None,
    pair_with: SignalCard | None = None,
) -> list[ScoreRecord]:
    """Score every card, return ranked list (highest first)."""
    scored = [
        _score(c, intent=intent, experience=experience, timeframe=timeframe,
               avoid_family=avoid_family, pair_with=pair_with)
        for c in cards
    ]
    return sorted(scored, key=lambda r: r.score, reverse=True)


def pick(
    cards: list[SignalCard],
    *,
    intent: Intent,
    experience: str,
    timeframe: str,
    avoid_family: str | None = None,
    pair_with: SignalCard | None = None,
) -> tuple[SignalCard, list[ScoreRecord]]:
    """Pick the highest-scoring card. Returns (winner, full ranking).

    Caller MUST guarantee `cards` is non-empty (use filter_candidates first
    and check before calling).
    """
    if not cards:
        raise ValueError("ranker.pick called with empty candidate list")
    by_name = {c.name: c for c in cards}
    ranking = rank(cards, intent=intent, experience=experience, timeframe=timeframe,
                   avoid_family=avoid_family, pair_with=pair_with)
    winner_name = ranking[0].signal
    return by_name[winner_name], ranking
