"""
app/kb/signal_resolver.py — resolve partial / fuzzy signal names to KB entries.

When the trader writes a signal name that is incomplete or close to several
known names ("rsi_over" → ``rsi_oversold``, ``rsi_overbought``), this resolver
returns the candidate set so the caller (chat / API / agent tool) can either
apply the only match directly or ask the user to pick one of 2–4 candidates.

The trader's query is first **normalized** so natural-language phrasing
(spaces, dashes, capitalization) lines up with the KB's underscored signal
names: ``"rsi above 60"`` → ``"rsi_above_60"`` before matching, so the
trader does not need to type the canonical signal name verbatim.

Matching strategy (in order, dedup-preserving), all against the normalized
query:
  1. Exact match on the full signal name → ``state = "exact"``.
  2. Prefix match — names that start with the query string.
  3. Family match — names whose leading word equals the query's leading word.
  4. Substring match — query appears anywhere in the name.
  5. Token-set match — every alphanumeric token of the query appears somewhere
     in the signal name (catches reorderings like "above 60 rsi").

When ``slot`` is provided, candidates are restricted to signals legal in that
slot (using the same role rules as :mod:`app.kb.signal_validation`):

    - slot ``"entry"`` → roles include ``entry_trigger`` or ``entry_filter``
    - slot ``"exit"``  → roles include ``exit_trigger``
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.kb import kb
from app.kb.signal_validation import (
    Mode,
    Slot,
    SignalPlacementError,
    _ENTRY_OK_ROLES,
    _ENTRY_TRIGGER_ONLY,
    _EXIT_OK_ROLES,
    validate_signal_placement,
)


_NORMALIZE_SEP_RE = re.compile(r"[\s\-/.]+")
_NORMALIZE_DROP_RE = re.compile(r"[^a-z0-9_]")
_NORMALIZE_COLLAPSE_RE = re.compile(r"_+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_signal_query(query: str) -> str:
    """Fold natural phrasing into the KB's underscored signal-name shape.

    ``"rsi above 60"`` → ``"rsi_above_60"``,
    ``"MACD Negative"`` → ``"macd_negative"``,
    ``"ema-cross-up"`` → ``"ema_cross_up"``. Leaves already-canonical names
    (``"rsi_above_60"``) unchanged.
    """
    clean = (query or "").strip().lower()
    if not clean:
        return ""
    clean = _NORMALIZE_SEP_RE.sub("_", clean)
    clean = _NORMALIZE_DROP_RE.sub("", clean)
    clean = _NORMALIZE_COLLAPSE_RE.sub("_", clean).strip("_")
    return clean


def _query_tokens(query: str) -> list[str]:
    return _TOKEN_RE.findall((query or "").lower())


ResolutionState = Literal["exact", "ambiguous", "none"]


@dataclass(frozen=True)
class SignalResolution:
    """Outcome of resolving a query string against the KB signal catalog."""

    query: str
    slot: Slot | None
    state: ResolutionState
    exact_match: str | None
    candidates: tuple[str, ...]
    message: str

    @property
    def is_exact(self) -> bool:
        return self.state == "exact"

    @property
    def is_ambiguous(self) -> bool:
        return self.state == "ambiguous"


def _slot_ok_roles(slot: Slot | None, mode: Mode) -> frozenset[str] | None:
    if slot is None:
        return None
    if slot == "entry":
        # Strict (trader UX) → triggers only; lenient (planner) → triggers + filter
        return _ENTRY_TRIGGER_ONLY if mode == "strict" else _ENTRY_OK_ROLES
    if slot == "exit":
        return _EXIT_OK_ROLES
    raise ValueError(f"Unknown slot {slot!r}; expected 'entry' or 'exit'.")


def _eligible_names(slot: Slot | None, mode: Mode) -> list[str]:
    ok_roles = _slot_ok_roles(slot, mode)
    if ok_roles is None:
        return sorted(kb.signals.keys())
    return sorted(
        name
        for name, card in kb.signals.items()
        if any(role in ok_roles for role in card.roles)
    )


def _rank_candidates(query: str, eligible: list[str]) -> list[str]:
    """Return eligible names ordered by match strength for the query.

    The query is normalized so trader-friendly phrasing like
    ``"rsi above 60"`` or ``"MACD Negative"`` lines up with the KB's
    underscored signal names before any of the four match passes run.
    """
    q = normalize_signal_query(query)
    tokens = _query_tokens(query)
    if not q and not tokens:
        return []

    head = q.split("_", 1)[0] if q else (tokens[0] if tokens else "")
    seen: set[str] = set()
    ordered: list[str] = []

    def _push(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        ordered.append(name)

    # 1. exact
    if q:
        for name in eligible:
            if name.lower() == q:
                _push(name)

    # 2. prefix
    if q:
        for name in eligible:
            if name.lower().startswith(q):
                _push(name)

    # 3. same leading-word family (e.g. "rsi" → rsi_*)
    if head:
        for name in eligible:
            if name.split("_", 1)[0].lower() == head:
                _push(name)

    # 4. substring
    if q:
        for name in eligible:
            if q in name.lower():
                _push(name)

    # 5. token-set match — covers reorderings and stray filler words
    # ("above 60 rsi", "rsi crossing above the 60 level"). Only kicks in for
    # multi-token queries so single-word queries don't over-match.
    if len(tokens) > 1:
        for name in eligible:
            name_lower = name.lower()
            if all(t in name_lower for t in tokens):
                _push(name)

    return ordered


def resolve_signal_name(
    query: str,
    slot: Slot | None = None,
    *,
    max_candidates: int = 4,
    mode: Mode = "strict",
) -> SignalResolution:
    """Map ``query`` to a single KB signal or a short candidate list.

    Args:
        query: trader-supplied partial / fuzzy name.
        slot: ``"entry"`` / ``"exit"`` to restrict candidates by role, or
            ``None`` to search across all signals.
        max_candidates: cap on the ambiguous-case candidate list (default 4
            — the upper bound the trader UX is designed for).

    Returns:
        :class:`SignalResolution` with ``state``:
            - ``"exact"``     — exactly one match (in ``exact_match``)
            - ``"ambiguous"`` — 2..N candidates
            - ``"none"``      — no match
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    clean = str(query or "").strip()
    if not clean:
        return SignalResolution(
            query="",
            slot=slot,
            state="none",
            exact_match=None,
            candidates=(),
            message="Please enter a signal name (or partial name) to look up.",
        )

    eligible = _eligible_names(slot, mode)
    ranked = _rank_candidates(clean, eligible)
    normalized = normalize_signal_query(clean)

    # Exact hit on a fully-typed name (case-insensitive) wins outright.
    # Compared against the normalized form so "rsi above 60" matches
    # the KB name "rsi_above_60" without forcing the trader to use
    # underscores.
    if normalized:
        for name in ranked:
            if name.lower() == normalized:
                return SignalResolution(
                    query=clean,
                    slot=slot,
                    state="exact",
                    exact_match=name,
                    candidates=(name,),
                    message=f"Resolved to '{name}'.",
                )

    if not ranked:
        slot_label = f" {slot}" if slot else ""
        return SignalResolution(
            query=clean,
            slot=slot,
            state="none",
            exact_match=None,
            candidates=(),
            message=(
                f"No{slot_label} signal matches '{clean}'. "
                f"Try a different keyword or check the signal catalog."
            ),
        )

    # Single-match → treat as exact for caller ergonomics.
    if len(ranked) == 1:
        return SignalResolution(
            query=clean,
            slot=slot,
            state="exact",
            exact_match=ranked[0],
            candidates=(ranked[0],),
            message=f"Resolved '{clean}' → '{ranked[0]}'.",
        )

    # Ambiguous: return top-N for the user to choose.
    top = tuple(ranked[:max_candidates])
    slot_label = f" {slot}" if slot else ""
    listed = ", ".join(top)
    message = (
        f"Multiple{slot_label} signals match '{clean}'. "
        f"Please choose one of: {listed}."
    )
    return SignalResolution(
        query=clean,
        slot=slot,
        state="ambiguous",
        exact_match=None,
        candidates=top,
        message=message,
    )


def resolve_or_validate(
    query: str,
    slot: Slot,
    *,
    max_candidates: int = 4,
    mode: Mode = "strict",
) -> tuple[SignalResolution, SignalPlacementError | None]:
    """Combine resolution + role-placement validation.

    Use when the caller knows the slot (e.g. trader said "use X for exit").
    Returns the resolution plus a placement error if the exact match (or
    only candidate) lands in the wrong slot.
    """
    resolution = resolve_signal_name(
        query, slot=slot, max_candidates=max_candidates, mode=mode,
    )
    if resolution.state != "exact":
        return resolution, None
    placement_err = validate_signal_placement(
        resolution.exact_match or "", slot, mode=mode,
    )
    return resolution, placement_err
