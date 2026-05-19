"""
app/planner/semantic_signal_composer.py — Inject KB signals derived from
CanonicalSemanticIntent into the planner's entry filter list.

No hardcoded signal-name maps.  Every signal lookup goes through the KB:
  - Bearish variants are found via SignalCard.contradicts rather than a
    static dictionary.
  - HTF and filter signal resolution searches the KB by keyword scoring
    against SignalCard.name / description / roles.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.kb.canonical_intent import CanonicalFilter, CanonicalHTFRule, CanonicalSemanticIntent
from app.kb.loader import KB
from app.kb.schemas import SignalCard

logger = logging.getLogger(__name__)

# Hard cap: how many semantic-derived filters we add on top of the
# preset/ranker-selected list.  Keeps strategies from becoming over-constrained.
_MAX_SEMANTIC_INJECT = 2


# ── Direction helpers (KB-driven) ─────────────────────────────────────────────

def _bearish_variant(signal_name: str, kb: KB) -> str:
    """Return the bearish counterpart of *signal_name* using KB contradicts.

    Falls back to the original name when no dedicated bearish signal exists
    (e.g. volume_spike, adx_strong_trend are direction-neutral).
    """
    card = kb.signals.get(signal_name)
    if not card:
        return signal_name
    for contradicted in (card.contradicts or []):
        other = kb.signals.get(contradicted)
        if other is not None:
            other_dir = (other.direction or "").lower()
            if other_dir == "bearish":
                return contradicted
    return signal_name


def _direction_signal(name: str, sentiment: str, kb: KB) -> str:
    """Return the direction-correct variant of *name* for the given sentiment."""
    if sentiment == "bearish":
        return _bearish_variant(name, kb)
    return name


# ── KB keyword search ─────────────────────────────────────────────────────────

def _score_signal(signal: SignalCard, keywords: list[str], bias: str) -> float:
    """Score a SignalCard against keywords and directional bias."""
    name = signal.name.lower()
    desc = (signal.description or "").lower()
    sig_dir = (signal.direction or "neutral").lower()

    if bias == "bearish" and sig_dir == "bullish":
        return 0.0
    if bias == "bullish" and sig_dir == "bearish":
        return 0.0

    score = 0.0
    for kw in keywords:
        kw = kw.lower()
        if name.startswith(kw):
            score += 10.0
        elif kw in name:
            score += 6.0
        elif kw in desc:
            score += 2.0
    if sig_dir != "neutral":
        score += 0.5
    return score


def _find_entry_filter(kb: KB, keywords: list[str], bias: str) -> SignalCard | None:
    """Return the best entry_filter SignalCard matching keywords + bias, or None."""
    best: SignalCard | None = None
    best_score = 0.0
    for signal in kb.signals.values():
        if "entry_filter" not in (signal.roles or []):
            continue
        score = _score_signal(signal, keywords, bias)
        if score > best_score:
            best_score = score
            best = signal
    if best:
        logger.debug(
            "semantic_composer|kb_search|kws=%s|bias=%r|found=%r|score=%.1f",
            keywords, bias, best.name, best_score,
        )
    return best


class SemanticSignalComposer:
    """Inject entry filter signals from CanonicalSemanticIntent into the plan."""

    def __init__(self, kb: KB):
        self.kb = kb

    def inject_entry_filters(
        self,
        intent_dict: dict[str, Any] | None,
        *,
        sentiment: str,
        entry_trigger_card: SignalCard,
        existing_filter_cards: list[SignalCard],
        timeframe: str,
        max_total_filters: int = 3,
    ) -> list[SignalCard]:
        """Return the (possibly extended) entry_filter_cards list.

        Looks up semantic HTF confluence and secondary filters from the KB,
        appending only cards that:
          - exist in the KB
          - are not already present (by name)
          - don't share a family with existing selections (dedup)
          - don't exceed max_total_filters

        Args:
            intent_dict: builder.semantic_intent as a plain dict or None.
            sentiment: "bullish" | "bearish"
            entry_trigger_card: the already-selected entry trigger
            existing_filter_cards: the already-selected entry filters
            timeframe: execution timeframe for compatibility check
            max_total_filters: hard cap on total entry filters (incl. existing)

        Returns:
            Updated (possibly same) list of entry filter cards.
        """
        if not intent_dict:
            return list(existing_filter_cards)

        try:
            intent = CanonicalSemanticIntent(**intent_dict)
        except Exception as exc:
            logger.warning("semantic_composer|bad_intent_dict|err=%s", exc)
            return list(existing_filter_cards)

        slots = max_total_filters - len(existing_filter_cards)
        if slots <= 0:
            return list(existing_filter_cards)

        injected: list[SignalCard] = []
        used_families: set[str] = {entry_trigger_card.family} | {
            c.family for c in existing_filter_cards
        }
        used_names: set[str] = {entry_trigger_card.name} | {
            c.name for c in existing_filter_cards
        }

        # 1. HTF confluence (gating rules first)
        for rule in sorted(intent.htf_confluence,
                           key=lambda r: 0 if r.role == "gating" else 1):
            if len(injected) >= min(slots, _MAX_SEMANTIC_INJECT):
                break
            card = self._resolve_htf_card(rule, sentiment, timeframe)
            if card and self._can_inject(card, used_families, used_names):
                injected.append(card)
                used_families.add(card.family)
                used_names.add(card.name)
                logger.info(
                    "semantic_composer|inject_htf|signal=%r|tf=%r|role=%r",
                    card.name, rule.timeframe, rule.role,
                )

        # 2. Secondary filters (RS, volume, ADX, …)
        for filt in intent.filters:
            if len(injected) >= min(slots, _MAX_SEMANTIC_INJECT):
                break
            card = self._resolve_filter_card(filt, sentiment, timeframe)
            if card and self._can_inject(card, used_families, used_names):
                injected.append(card)
                used_families.add(card.family)
                used_names.add(card.name)
                logger.info(
                    "semantic_composer|inject_filter|type=%r|signal=%r",
                    filt.type, card.name,
                )

        if injected:
            logger.info(
                "semantic_composer|injected_total=%d|signals=%s",
                len(injected), [c.name for c in injected],
            )

        return list(existing_filter_cards) + injected

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_htf_card(
        self,
        rule: CanonicalHTFRule,
        sentiment: str,
        timeframe: str,
    ) -> SignalCard | None:
        """Look up the KB signal card for an HTF confluence rule."""
        signal_name = rule.signal
        if signal_name:
            # The normalizer already picked a bullish/neutral signal name;
            # swap to bearish variant via KB contradicts if needed.
            signal_name = _direction_signal(signal_name, sentiment, self.kb)
        else:
            # No pre-resolved name — search by HTF condition keywords
            keywords = _condition_to_keywords(rule.condition or rule.indicator or "trend")
            card = _find_entry_filter(self.kb, keywords, sentiment)
            if card is None:
                return None
            signal_name = card.name

        return self._lookup_and_validate(signal_name, timeframe)

    def _resolve_filter_card(
        self,
        filt: CanonicalFilter,
        sentiment: str,
        timeframe: str,
    ) -> SignalCard | None:
        """Look up the KB signal card for a secondary filter."""
        signal_name = filt.signal
        if signal_name:
            signal_name = _direction_signal(signal_name, sentiment, self.kb)
        else:
            # Search by filter type keywords
            keywords = _condition_to_keywords(filt.type)
            card = _find_entry_filter(self.kb, keywords, sentiment)
            if card is None:
                return None
            signal_name = card.name

        return self._lookup_and_validate(signal_name, timeframe)

    def _lookup_and_validate(
        self, signal_name: str, timeframe: str
    ) -> SignalCard | None:
        """Fetch a SignalCard by name and check role + timeframe support."""
        card = self.kb.signals.get(signal_name)
        if card is None:
            logger.debug("semantic_composer|signal_not_found|name=%r", signal_name)
            return None
        if "entry_filter" not in (card.roles or []):
            return None
        if timeframe in (card.unsupported_on or []):
            return None
        return card

    def _can_inject(
        self,
        card: SignalCard,
        used_families: set[str],
        used_names: set[str],
    ) -> bool:
        if card.name in used_names:
            return False
        # RS-style signals are additive reference conditions — allow even when
        # family overlaps with an existing trend/momentum signal.
        # Check the signal's tags (KB-declared) rather than guessing from name.
        if "relative_strength" in card.tags:
            return True
        if card.family in used_families:
            return False
        return True


# ── Utility ────────────────────────────────────────────────────────────────────

def _condition_to_keywords(text: str) -> list[str]:
    """Extract meaningful search keywords from a condition/type string."""
    text_lower = text.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", text_lower) if len(t) >= 2]
    stop = {"the", "a", "an", "is", "be", "in", "on", "at", "to", "of",
            "and", "or", "should", "must", "will", "above", "below", "up",
            "down", "with", "for", "if", "are", "was", "that", "this"}
    result = [t for t in tokens if t not in stop]
    return result or ["trend"]
