"""
app/planner/filters.py — Hard filters (deterministic, pure functions).

A signal that fails any hard filter is ELIMINATED from the candidate pool.
No score can recover it. This is what stops bullish strategies from ever
selecting bearish-direction entry triggers.
"""
from __future__ import annotations

from app.kb.schemas import Intent, SignalCard, SignalDirection, SignalRole


def opposite_direction(direction: SignalDirection) -> SignalDirection:
    if direction == "bullish":
        return "bearish"
    if direction == "bearish":
        return "bullish"
    return "neutral"


def required_direction_for_role(
    role: SignalRole,
    sentiment: SignalDirection,
) -> set[SignalDirection]:
    """The direction(s) a signal must have to be valid in this role.

    - entry_trigger / entry_filter: must align with sentiment.
      "neutral" signals (e.g. inside_bar, volume_spike) are accepted as filters
      but only as a triggers if the user wants confirmation-style intent;
      we permit them universally and let the ranker down-weight as needed.
    - exit_trigger / exit_filter: must be opposite-direction to entry sentiment, OR neutral.
    """
    if role in {"entry_trigger", "entry_filter"}:
        return {sentiment, "neutral"}
    if role in {"exit_trigger", "exit_filter"}:
        return {opposite_direction(sentiment), "neutral"}
    return {"bullish", "bearish", "neutral"}


def _matches_contraindication(
    rule: dict,
    *,
    role: SignalRole,
    sentiment: SignalDirection,
    intent: Intent,
) -> bool:
    """A contraindication rule is a dict of field-value pairs that must ALL match."""
    facts = {
        "role": role,
        "sentiment": sentiment,
        "hold_horizon": intent.hold_horizon,
        "frequency": intent.frequency,
        "profit_size": intent.profit_size,
        "style": intent.style,
        "risk_appetite": intent.risk_appetite,
    }
    for key, expected in rule.items():
        if facts.get(key) != expected:
            return False
    return True


def filter_candidates(
    cards: list[SignalCard],
    *,
    role: SignalRole,
    sentiment: SignalDirection,
    timeframe: str,
    intent: Intent,
) -> tuple[list[SignalCard], list[tuple[str, str]]]:
    """Apply hard filters. Returns (survivors, eliminations) where each
    elimination is (signal_name, reason)."""
    valid_directions = required_direction_for_role(role, sentiment)
    survivors: list[SignalCard] = []
    eliminations: list[tuple[str, str]] = []

    for card in cards:
        # 1. Role eligibility
        if role not in card.roles:
            eliminations.append((card.name, f"role_not_supported({role})"))
            continue

        # 2. Direction must satisfy the role's requirement
        if card.direction not in valid_directions:
            eliminations.append(
                (card.name, f"direction_mismatch(card={card.direction},need={sorted(valid_directions)})")
            )
            continue

        # 3. Timeframe must not be explicitly unsupported
        if timeframe in card.unsupported_on:
            eliminations.append((card.name, f"timeframe_unsupported({timeframe})"))
            continue

        # 4. Contraindications (sentiment/intent rules)
        contra_hit = False
        for rule in card.contraindicated_when:
            if _matches_contraindication(rule, role=role, sentiment=sentiment, intent=intent):
                eliminations.append((card.name, f"contraindicated({rule})"))
                contra_hit = True
                break
        if contra_hit:
            continue

        survivors.append(card)

    return survivors, eliminations
