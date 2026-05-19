"""
app/services/discovery/tie_break.py
───────────────────────────────────
Phase 9 — declarative tie-break methods.

When the scanner returns more than one candidate, the chat layer asks the
user which method to use. Each method is a pure function over a list of
Candidate objects, returning the SAME list ordered best→worst by that
method's criterion. Picking the best candidate is then `result[0]`.

Methods are read off Candidate.metrics, which the scanner pre-populates so
this stays an O(N) sort with no extra I/O.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from app.services.discovery.types import Candidate, TieBreakOption

logger = logging.getLogger(__name__)


def _by_metric_desc(metric: str) -> Callable[[list[Candidate]], list[Candidate]]:
    """Sort candidates highest-metric first. Missing metric → sorted last."""
    def _ranker(candidates: list[Candidate]) -> list[Candidate]:
        return sorted(
            candidates,
            key=lambda c: c.metrics.get(metric, float("-inf")),
            reverse=True,
        )
    return _ranker


def _by_metric_asc(metric: str) -> Callable[[list[Candidate]], list[Candidate]]:
    """Sort candidates lowest-metric first. Missing metric → sorted last."""
    def _ranker(candidates: list[Candidate]) -> list[Candidate]:
        return sorted(
            candidates,
            key=lambda c: c.metrics.get(metric, float("inf")),
        )
    return _ranker


# Method id → ranker function. Method ids are stable; chat layer stores
# the user's pick by id. Adding a new method here automatically makes it
# available to any preset that lists it in tie_break_options.
TIE_BREAK_METHODS: dict[str, Callable[[list[Candidate]], list[Candidate]]] = {
    "highest_relative_volume":  _by_metric_desc("relative_volume"),
    "closest_to_52w_high":      _by_metric_asc("distance_to_52w_high_pct"),
    "closest_to_52w_low":       _by_metric_asc("distance_to_52w_low_pct"),
    "highest_rsi":              _by_metric_desc("rsi_14"),
    "lowest_rsi":               _by_metric_asc("rsi_14"),
    "highest_close":            _by_metric_desc("close"),
    "lowest_close":             _by_metric_asc("close"),
    "highest_volatility":       _by_metric_desc("atr_14_pct"),
    "lowest_volatility":        _by_metric_asc("atr_14_pct"),
    "alphabetical":             lambda cs: sorted(cs, key=lambda c: c.symbol),
}


# Human-readable metadata for each method. The chat layer uses the label
# when presenting choices and the description when the user asks for help.
_METHOD_LABELS = {
    "highest_relative_volume":  ("Highest relative volume",
                                 "Pick the stock with the strongest volume vs its 20-bar average"),
    "closest_to_52w_high":      ("Closest to 52-week high",
                                 "Pick the stock whose close is nearest its 52-week high"),
    "closest_to_52w_low":       ("Closest to 52-week low",
                                 "Pick the stock whose close is nearest its 52-week low"),
    "highest_rsi":              ("Highest RSI(14)",
                                 "Pick the strongest momentum reading"),
    "lowest_rsi":               ("Lowest RSI(14)",
                                 "Pick the most-oversold reading"),
    "highest_close":            ("Highest closing price",
                                 "Pick the highest-priced stock"),
    "lowest_close":              ("Lowest closing price",
                                 "Pick the lowest-priced stock"),
    "highest_volatility":       ("Highest volatility (ATR%)",
                                 "Pick the stock with the largest ATR relative to its price"),
    "lowest_volatility":        ("Lowest volatility (ATR%)",
                                 "Pick the calmest stock"),
    "alphabetical":             ("Alphabetical (A-Z)",
                                 "Deterministic fallback — pick the first symbol alphabetically"),
}


def available_tie_break_options(method_ids: list[str] | None = None) -> list[TieBreakOption]:
    """Return TieBreakOption objects for the given method ids (or all known
    methods when None). Used to materialise a preset's declared
    tie_break_options when only ids were supplied."""
    ids = method_ids or list(TIE_BREAK_METHODS.keys())
    out: list[TieBreakOption] = []
    for mid in ids:
        if mid not in TIE_BREAK_METHODS:
            logger.warning("tie_break|unknown_method=%s — skipping", mid)
            continue
        label, desc = _METHOD_LABELS.get(mid, (mid, ""))
        out.append(TieBreakOption(method=mid, label=label, description=desc))
    return out


def apply_tie_break(method: str, candidates: list[Candidate]) -> list[Candidate]:
    """Apply a tie-break method to candidates. Returns the SAME list ordered
    best→worst. Caller picks `result[0]` as the chosen symbol.

    Raises KeyError on unknown method id so the chat layer surfaces a clear
    error instead of silently picking 'alphabetical' or whatever.
    """
    if method not in TIE_BREAK_METHODS:
        raise KeyError(
            f"unknown tie_break method {method!r}. Known: {sorted(TIE_BREAK_METHODS)}"
        )
    if not candidates:
        return []
    return TIE_BREAK_METHODS[method](list(candidates))


# ── Parsing user replies (chat-layer helper) ─────────────────────────────────


_NUMERIC_RE = re.compile(r"^\s*(\d+)\s*\.?\s*$")


def parse_user_tie_break_reply(
    reply: str,
    options: list[TieBreakOption],
) -> str | None:
    """Try to interpret a free-text user reply as a tie-break choice.

    Accepts:
      - A 1-based index ("1", "2", "3", "1.")
      - The method id ("highest_relative_volume")
      - The label, case-insensitive ("highest relative volume")
      - A unique substring of the label ("relative volume")

    Returns the chosen method id, or None if the reply is ambiguous /
    unparseable. Caller is responsible for re-prompting on None.
    """
    if not reply or not options:
        return None
    text = reply.strip().lower()
    if not text:
        return None

    # 1-based numeric index
    m = _NUMERIC_RE.match(text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx].method
        return None

    # Exact id match (case-insensitive)
    for opt in options:
        if opt.method.lower() == text:
            return opt.method

    # Exact label match (case-insensitive)
    for opt in options:
        if opt.label.lower() == text:
            return opt.method

    # Unique substring match against label or id
    matches = [
        opt for opt in options
        if text in opt.label.lower() or text in opt.method.lower()
    ]
    if len(matches) == 1:
        return matches[0].method
    return None
