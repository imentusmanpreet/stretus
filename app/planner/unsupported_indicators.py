"""
app/planner/unsupported_indicators.py — Detect requested-but-unsupported indicators.

Users frequently ask for indicators or concepts the KB does not cover yet
(Ichimoku, Heikin-Ashi, Renko, Fibonacci, Wyckoff, full ICT/SMC vocabulary,
options Greeks, statistical thresholds). When that happens the rest of the
planner silently drops the request and the user gets a strategy that ignores
their stated intent.

This module gives the chat layer a way to spot those mentions early so it can
either (a) tell the user we don't support that yet and offer the closest
mapped alternative, or (b) capture the mention in the audit trail for later
review.

It is intentionally a thin keyword detector — paraphrase coverage lives in the
LLM slot extractor. The goal here is to catch the obvious cases that would
otherwise vanish.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnsupportedMention:
    """One instance of a requested-but-unsupported indicator/concept."""

    canonical_name: str
    matched_phrase: str
    category: str
    suggested_alternative: str | None = None


# Each entry: (canonical, [regex patterns], category, suggested mapped alternative)
_UNSUPPORTED: tuple[tuple[str, tuple[str, ...], str, str | None], ...] = (
    (
        "ichimoku",
        (
            r"\bichimoku\b",
            r"\bkumo\b",
            r"\btenkan\b",
            r"\bkijun\b",
            r"\bsenkou\b",
            r"\bchikou\b",
            r"\bcloud\s+(?:cross|breakout|alignment|support)\b",
        ),
        "trend",
        "EMA cross + ADX trend filter",
    ),
    (
        "heikin_ashi",
        (r"\bheik[ei]n[\s\-]*ashi\b", r"\bha\s+candles?\b"),
        "candles",
        "rejection-candle / market-structure-break signals",
    ),
    (
        "renko",
        (r"\brenko\b", r"\brenko\s+bricks?\b"),
        "candles",
        "ATR-based breakout signal",
    ),
    (
        "pivot_points",
        (
            r"\bpivot\s+points?\b",
            r"\bclassic\s+pivot\b",
            r"\bcamarilla\b",
            r"\bwoodie\s+pivot\b",
            r"\bfloor\s+pivot\b",
        ),
        "levels",
        "prev-day high/low breakout",
    ),
    (
        "fibonacci",
        (
            r"\bfib(?:onacci)?\b",
            r"\b0?\.(?:236|382|500|618|786)\s+(?:retrace|extension|level)\w*\b",
        ),
        "levels",
        "swing-high/swing-low structural anchor",
    ),
    (
        "wyckoff",
        (
            r"\bwyckoff\b",
            r"\baccumulation\s+phase\b",
            r"\bdistribution\s+phase\b",
            r"\bspring\s+pattern\b",
        ),
        "structure",
        "market-structure-break + volume confirmation",
    ),
    (
        "ict_order_block",
        (r"\border\s+block\b", r"\bob\s+(?:bullish|bearish)\b", r"\bmitigation\s+block\b"),
        "ict_smc",
        "market-structure-break (closest current approximation)",
    ),
    (
        "ict_liquidity_sweep",
        (
            r"\bliquidity\s+(?:sweep|grab|run|pool)\b",
            r"\bstop\s+hunt\b",
            r"\bsweep\s+of\s+(?:lows|highs|liquidity)\b",
        ),
        "ict_smc",
        "rejection-candle at prev-day high/low",
    ),
    (
        "ict_breaker_block",
        (r"\bbreaker\s+block\b", r"\bbreaker\s+(?:bullish|bearish)\b"),
        "ict_smc",
        None,
    ),
    (
        "ict_inducement",
        (r"\binducement\b", r"\bidm\b"),
        "ict_smc",
        None,
    ),
    (
        "options_greeks",
        (
            r"\b(?:theta|delta|gamma|vega|rho)\s+decay\b",
            r"\b(?:theta|delta|gamma|vega|rho)\s+(?:neutral|hedge|exposure)\b",
            r"\bimplied\s+volatility\b",
            r"\biv\s+(?:crush|rank|percentile)\b",
        ),
        "options",
        "cash-segment equivalent (no options layer yet)",
    ),
    (
        "stat_threshold",
        (
            r"\b\d+(?:\.\d+)?\s*(?:standard\s+deviation|σ|sigma)s?\b",
            r"\bz[\s\-]*score\b",
            r"\bbollinger\s+\d+(?:\.\d+)?[\s\-]*(?:std|sigma)\b",
        ),
        "statistics",
        "Bollinger-band breakout signal",
    ),
    (
        "vix_filter",
        (r"\bindia\s+vix\b", r"\bvix\s+(?:above|below|under|over)\s+\d+", r"\bvix\s+filter\b"),
        "cross_asset",
        "no cross-asset gate yet",
    ),
    (
        "pair_trading",
        (r"\bpair\s+trading?\b", r"\bspread\s+trade\b", r"\bcointegration\b"),
        "cross_asset",
        "no multi-symbol strategy support yet",
    ),
    (
        "expiry_logic",
        (r"\bexpiry\s+day\b", r"\bweekly\s+expiry\b", r"\bmonthly\s+expiry\b"),
        "options",
        "cash-segment only, no expiry calendar",
    ),
    (
        "schedule_calendar",
        (
            r"\bevery\s+(?:monday|tuesday|wednesday|thursday|friday)\b",
            r"\bonce\s+a\s+week\b",
            r"\bweekly\s+sip\b",
            r"\bmonthly\s+sip\b",
        ),
        "scheduling",
        "no calendar/cron triggers yet — use intraday trigger instead",
    ),
)


def detect_unsupported(text: str) -> list[UnsupportedMention]:
    """Return all unsupported indicator/concept mentions in `text`."""
    if not text:
        return []

    mentions: list[UnsupportedMention] = []
    seen: set[str] = set()

    for canonical, patterns, category, alt in _UNSUPPORTED:
        if canonical in seen:
            continue
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                mentions.append(
                    UnsupportedMention(
                        canonical_name=canonical,
                        matched_phrase=m.group(0),
                        category=category,
                        suggested_alternative=alt,
                    )
                )
                seen.add(canonical)
                logger.info(
                    "unsupported_indicator|detected|name=%s|category=%s"
                    "|matched=%r|alternative=%s",
                    canonical, category, m.group(0), alt,
                )
                break

    return mentions


def format_user_message(mentions: list[UnsupportedMention]) -> str | None:
    """Build a single user-facing message naming each unsupported mention.

    Returns None if the list is empty.
    """
    if not mentions:
        return None

    lines = ["I noticed a few things I can't model yet:"]
    for m in mentions:
        line = f"  • **{m.matched_phrase}** ({m.category})"
        if m.suggested_alternative:
            line += f" — closest available: _{m.suggested_alternative}_"
        lines.append(line)
    lines.append(
        "Would you like to swap to the suggested alternative, drop the "
        "constraint, or keep it on your wishlist for now?"
    )
    return "\n".join(lines)
