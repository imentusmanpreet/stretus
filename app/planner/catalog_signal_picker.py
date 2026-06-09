"""
app/planner/catalog_signal_picker.py — Catalog-driven signal selection.

This is the Phase 14 architectural rewrite of how signals are chosen.

The traditional pipeline picks a strategy preset, hydrates its hardcoded
template, then runs `constraint_compiler` to patch in or strip out signals
based on what the user actually said. That approach required Python code for
every new indicator or phrasing.

The catalog picker takes a different approach:

  1. Every signal YAML in `app/kb/signals/` may declare a `match_when` block
     listing the natural-language phrases that should select it.
  2. The picker walks the entire catalog, runs each signal's phrases against
     the user's prompt, and collects every match with its role + confidence.
  3. Conflicts (e.g. multiple TRIGGER candidates) are resolved by confidence
     score; the loser becomes a FILTER or is dropped.
  4. The output is a complete signal_plan, ready for SL/TP/RR overlay.

No signal-specific Python code lives here. Adding a new indicator means
authoring its YAML and giving it a `match_when` block. The picker reads it on
KB hot-reload and starts selecting it.

Public API:

    pick_plan_from_catalog(prompt, *, kb, timeframe, sentiment,
                           exit_on_opposite=False) → CatalogPick

`CatalogPick` carries the final plan plus a `confidence` score the chat
layer uses to decide whether to bypass the legacy preset → constraint
pipeline (high confidence) or fall back to it (low confidence).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.kb.loader import KB
from app.kb.schemas import MatchParamSpec, SignalCard

logger = logging.getLogger(__name__)


# Connective "glue" words users write between an indicator and its condition
# that rigid phrase regexes don't anticipate. Stripping a copy of the prompt of
# these (and matching against both the literal and stripped text) lets natural
# phrasing match the catalog phrases. Pure connectives only — articles, copulas,
# and movement verbs that resolve to a comparator — so no signal meaning is lost.
_FILLER_RE = re.compile(
    r"\b(?:the|a|an|is|are|was|were|be|been|being|has|have|had|its|it|that|"
    r"drops?|dropping|falls?|falling|dips?|rises?|rising|climbs?|moves?|moving|"
    r"goes|going|trading|currently|now|reaches?|reaching|"
    r"turns?|turning|flips?|flipping|gets?|getting|becomes?|becoming)\s+",
    re.IGNORECASE,
)


# ── Public dataclasses ────────────────────────────────────────────────────────


@dataclass
class CatalogMatch:
    """One signal that matched at least one of its phrases."""

    signal_name: str
    role: str                              # "entry_trigger" | "entry_filter" | "exit_trigger" | "exit_filter"
    confidence: float                      # 0.0 – 1.0
    params: dict[str, Any] = field(default_factory=dict)
    matched_phrase: str = ""
    matched_text: str = ""
    family: str = ""


@dataclass
class CatalogPick:
    """Final picker output."""

    signal_plan: dict[str, Any]
    matches: list[CatalogMatch] = field(default_factory=list)
    confidence: float = 0.0                # overall confidence — used for fallback
    used_preset_fallback: bool = False
    audit: list[str] = field(default_factory=list)


# ── Match-time helpers ────────────────────────────────────────────────────────


def _coerce(value: str, kind: str) -> Any:
    try:
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
    except (TypeError, ValueError):
        return None
    return value


def _extract_params(
    match: re.Match[str],
    spec: dict[str, MatchParamSpec] | None,
) -> dict[str, Any]:
    """Pull params from regex captures per the signal's extract_params spec.

    Named-capture groups are looked up by name. Numbered groups by index.
    Missing groups fall back to the spec's `default`.
    """
    out: dict[str, Any] = {}
    if not spec:
        return out
    for param_name, ps in spec.items():
        raw = None
        try:
            if ps.from_group is not None:
                raw = match.group(ps.from_group)
        except (IndexError, KeyError):
            raw = None
        if raw is None:
            if ps.default is not None:
                out[param_name] = ps.default
            continue
        coerced = _coerce(raw, ps.kind)
        if coerced is not None:
            out[param_name] = coerced
        elif ps.default is not None:
            out[param_name] = ps.default
    return out


def _phrase_confidence(phrase: str, matched_text: str, base: float) -> float:
    """Bias scoring toward more specific (longer) phrase matches.

    A 6-word match beats a 2-word match for the same base confidence so that
    "macd line crosses above signal" wins over a generic "macd" phrase.
    """
    span_len = max(1, len(matched_text))
    phrase_len = max(1, len(phrase))
    specificity = min(1.0, (span_len + phrase_len) / 200.0)
    return round(min(1.0, base + 0.2 * specificity), 4)


def _passes_keywords(text: str, requires: list[str], forbids: list[str]) -> bool:
    for pat in requires or ():
        if not re.search(pat, text, re.IGNORECASE):
            return False
    for pat in forbids or ():
        if re.search(pat, text, re.IGNORECASE):
            return False
    return True


def _family_of(name: str) -> str:
    """Coarse family classifier — return the canonical family tag for any
    signal name.

    Order matters: we check by substring (more selective first), then fall
    through to a prefix-based map. The goal is for `price_above_ema`,
    `ema_cross_up`, and `ema_above` to ALL classify as 'ema' family — so
    the auto-fill backstop sees them together.
    """
    n = (name or "").lower()
    # Substring family detectors — order matters; longer / more specific first.
    family_substrings: tuple[tuple[str, str], ...] = (
        ("ichimoku",       "ichimoku"),
        ("supertrend",     "supertrend"),
        ("keltner",        "keltner"),
        ("donchian",       "donchian"),
        ("stoch",          "stoch"),
        ("zscore",         "zscore"),
        ("z_score",        "zscore"),
        ("vwap_reclaim",   "vwap"),
        ("vwap_zscore",    "vwap"),
        ("vwap",           "vwap"),
        ("macd",           "macd"),
        ("fvg",            "fvg"),
        ("opening_range",  "breakout"),
        ("n_bar_high",     "breakout"),
        ("n_bar_low",      "breakout"),
        ("breakout",       "breakout"),
        ("market_structure", "structure"),
        ("uptrend",        "structure"),
        ("downtrend",      "structure"),
        ("rejection_candle", "candle"),
        ("inside_bar",     "candle"),
        ("bullish_candle", "candle"),
        ("bearish_candle", "candle"),
        ("hammer",         "candle"),
        ("doji",           "candle"),
        ("reference_above_sma", "reference"),
        ("reference_below_sma", "reference"),
        ("rs_outperforms", "reference"),
        ("rs_underperforms", "reference"),
        ("gap",            "gap"),
        ("volume",         "volume"),
        ("delivery_volume", "volume"),
        ("chaikin",        "volume"),
        ("near_52",        "level"),
        ("adx",            "adx"),
        ("atr",            "atr"),
        ("rsi",            "rsi"),
        ("ema",            "ema"),
        ("sma",            "sma"),
        ("mfi",            "mfi"),
        ("cci",            "cci"),
        ("obv",            "obv"),
        ("bb_",            "bb"),
        ("bollinger",      "bb"),
        ("is_above_sma",   "sma"),
        ("is_below_sma",   "sma"),
    )
    for substr, family in family_substrings:
        if substr in n:
            return family
    return n.split("_", 1)[0]


# ── The picker ────────────────────────────────────────────────────────────────


# Negation keywords — when one of these appears within 25 chars BEFORE the
# matched phrase, the match is suppressed. Prevents "avoid gap-down" from
# selecting the gap_down_open trigger.
_NEGATION_RE = re.compile(
    r"\b(?:avoid|ignore|skip|don'?t|do\s+not|never|no\s+trade(?:s)?|exclude|"
    r"filter\s+out|without|except|outside\s+of|prohibit)\b",
    re.IGNORECASE,
)


def _is_negated(prompt: str, match_start: int) -> bool:
    """True when a negation keyword sits within ~60 chars before the match
    AND no clause separator (period / semicolon) lies between them.

    The clause-separator check prevents leakage like
        "Avoid X. Use Y." → negation of "Use Y" by "Avoid"
    Punctuation marks the end of the negated clause.
    """
    window_start = max(0, match_start - 80)
    window = prompt[window_start:match_start]
    neg = _NEGATION_RE.search(window)
    if not neg:
        return False
    # Reject if a clause separator sits BETWEEN the negation keyword and the
    # match (e.g. "Avoid X. Trade Y" — "Trade Y" is NOT negated).
    between = window[neg.end():]
    if re.search(r"[.;]\s|\bbut\b|\bhowever\b|\binstead\b", between, re.IGNORECASE):
        return False
    return True


def _find_all_matches(prompt: str, kb: KB) -> list[CatalogMatch]:
    """Scan every signal's phrases against the prompt; collect raw matches.

    A phrase match is dropped when a negation keyword ("avoid", "ignore",
    "skip", "don't", "no trade", …) appears within 25 chars before the
    matched span. This prevents "avoid gap-down" from being read as a
    gap-down entry trigger.
    """
    if not prompt:
        return []
    matches: list[CatalogMatch] = []
    normalized = re.sub(r"\s+", " ", prompt).strip()
    # Filler-tolerant variant: drop the connective "glue" words users naturally
    # write between an indicator and its condition, which rigid phrase regexes
    # don't anticipate. Examples this fixes (ALL signals, not one):
    #   "stochastic IS oversold"      → "stochastic oversold"
    #   "ADX IS above 25"             → "ADX above 25"
    #   "RSI DROPS below 30"          → "RSI below 30"
    #   "price closes above THE upper bollinger" → "...above upper bollinger"
    # We try the literal text first and fall back to this variant, so phrases
    # that DO contain such a word still match against `normalized` — this can
    # only ADD matches, never remove them. The words below are pure connectives
    # (articles, copulas, and movement verbs that resolve to a comparator); none
    # carry signal meaning, so stripping them is safe.
    normalized_nofill = _FILLER_RE.sub("", normalized)

    for name, card in kb.signals.items():
        mw = card.match_when
        if mw is None or not mw.phrases:
            continue
        if not (
            _passes_keywords(normalized, mw.requires_keyword, mw.forbids_keyword)
            or _passes_keywords(normalized_nofill, mw.requires_keyword, mw.forbids_keyword)
        ):
            continue

        # Take the FIRST phrase that hits; record the best by specificity.
        best: CatalogMatch | None = None
        for phrase in mw.phrases:
            try:
                m = re.search(phrase, normalized, re.IGNORECASE)
                match_text = normalized
                if not m and normalized_nofill != normalized:
                    m = re.search(phrase, normalized_nofill, re.IGNORECASE)
                    match_text = normalized_nofill
            except re.error as exc:
                logger.warning(
                    "catalog_picker|bad_regex|signal=%s|phrase=%r|err=%s",
                    name, phrase, exc,
                )
                continue
            if not m:
                continue
            # Negation guard — skip if "avoid|ignore|skip|…" precedes the match
            if _is_negated(match_text, m.start()):
                logger.info(
                    "catalog_picker|negated_match|signal=%s|matched=%r|context=%r",
                    name, m.group(0),
                    normalized[max(0, m.start() - 30): m.end() + 5],
                )
                continue
            conf = _phrase_confidence(phrase, m.group(0), mw.confidence)
            params = _extract_params(m, mw.extract_params)
            cand = CatalogMatch(
                signal_name=name,
                role=mw.acts_as,
                confidence=conf,
                params=params,
                matched_phrase=phrase,
                matched_text=m.group(0),
                family=_family_of(name),
            )
            if best is None or cand.confidence > best.confidence:
                best = cand
        if best is not None:
            matches.append(best)

    matches.sort(key=lambda x: x.confidence, reverse=True)
    return matches


def _resolve_role_conflicts(matches: list[CatalogMatch]) -> dict[str, list[CatalogMatch]]:
    """Group matches into roles with conflict resolution.

    Rules:
      * Exactly ONE entry_trigger per family — the highest confidence wins;
        losers are demoted to entry_filter.
      * Multiple entry_filters across DIFFERENT families are allowed.
      * Exactly ONE exit_trigger per family — the highest confidence wins;
        losers are demoted to exit_filter.
      * Multiple exit_filters across DIFFERENT families are allowed.
    """
    by_role: dict[str, list[CatalogMatch]] = {
        "entry_trigger": [],
        "entry_filter":  [],
        "exit_trigger":  [],
        "exit_filter":   [],
    }

    # Pass 1: collect candidates for entry_trigger, resolve conflicts.
    trigger_candidates = [m for m in matches if m.role == "entry_trigger"]
    seen_families: set[str] = set()
    primary_trigger: CatalogMatch | None = None
    demoted: list[CatalogMatch] = []
    for m in trigger_candidates:
        if primary_trigger is None:
            primary_trigger = m
            seen_families.add(m.family)
            by_role["entry_trigger"].append(m)
            continue
        # Multiple triggers of the SAME family beyond the leader → demote.
        if m.family in seen_families:
            demoted_match = CatalogMatch(
                signal_name=m.signal_name, role="entry_filter",
                confidence=m.confidence, params=m.params,
                matched_phrase=m.matched_phrase, matched_text=m.matched_text,
                family=m.family,
            )
            demoted.append(demoted_match)
            continue
        # Different family → also demote (only one leading trigger total).
        # User can express compound entries via filters.
        demoted_match = CatalogMatch(
            signal_name=m.signal_name, role="entry_filter",
            confidence=m.confidence, params=m.params,
            matched_phrase=m.matched_phrase, matched_text=m.matched_text,
            family=m.family,
        )
        demoted.append(demoted_match)
        seen_families.add(m.family)

    # Pass 2: collect explicit entry filter matches; merge demoted triggers.
    for m in matches:
        if m.role == "entry_filter":
            by_role["entry_filter"].append(m)
    by_role["entry_filter"].extend(demoted)

    # Deduplicate entry filters by signal_name; keep highest confidence.
    seen_filters: dict[str, CatalogMatch] = {}
    for m in by_role["entry_filter"]:
        existing = seen_filters.get(m.signal_name)
        if existing is None or m.confidence > existing.confidence:
            seen_filters[m.signal_name] = m
    by_role["entry_filter"] = list(seen_filters.values())

    # Pass 3: exit_trigger — one primary per family, extras demoted to exit_filter.
    exit_trigger_candidates = [m for m in matches if m.role == "exit_trigger"]
    seen_exit_families: set[str] = set()
    primary_exit: CatalogMatch | None = None
    demoted_exits: list[CatalogMatch] = []
    for m in exit_trigger_candidates:
        if primary_exit is None:
            primary_exit = m
            seen_exit_families.add(m.family)
            by_role["exit_trigger"].append(m)
            continue
        if m.family in seen_exit_families:
            demoted_match = CatalogMatch(
                signal_name=m.signal_name, role="exit_filter",
                confidence=m.confidence, params=m.params,
                matched_phrase=m.matched_phrase, matched_text=m.matched_text,
                family=m.family,
            )
            demoted_exits.append(demoted_match)
            continue
        demoted_match = CatalogMatch(
            signal_name=m.signal_name, role="exit_filter",
            confidence=m.confidence, params=m.params,
            matched_phrase=m.matched_phrase, matched_text=m.matched_text,
            family=m.family,
        )
        demoted_exits.append(demoted_match)
        seen_exit_families.add(m.family)

    # Pass 4: collect explicit exit_filter matches; merge demoted exit_triggers.
    for m in matches:
        if m.role == "exit_filter":
            by_role["exit_filter"].append(m)
    by_role["exit_filter"].extend(demoted_exits)

    # Deduplicate exit filters by signal_name; keep highest confidence.
    seen_exit_filters: dict[str, CatalogMatch] = {}
    for m in by_role["exit_filter"]:
        existing = seen_exit_filters.get(m.signal_name)
        if existing is None or m.confidence > existing.confidence:
            seen_exit_filters[m.signal_name] = m
    by_role["exit_filter"] = list(seen_exit_filters.values())

    return by_role


def _build_signal(match: CatalogMatch, timeframe: str | None) -> dict[str, Any]:
    """Build the dict the strategy_assembler expects."""
    role_map = {
        "entry_trigger": "TRIGGER",
        "entry_filter":  "FILTER",
        "exit_trigger":  "TRIGGER",
        "exit_filter":   "FILTER",
    }
    return {
        "name":        match.signal_name,
        "params":      dict(match.params or {}),
        "timeframe":   timeframe,
        "signal_type": role_map.get(match.role, "FILTER"),
    }


def _apply_exit_on_opposite(
    by_role: dict[str, list[CatalogMatch]],
    kb: KB,
    timeframe: str | None,
) -> tuple[dict[str, list[CatalogMatch]], list[str]]:
    """If the entry trigger has `match_when.mirrors_to`, add that as an exit
    trigger when the user wrote 'exit on opposite'."""
    audit: list[str] = []
    triggers = by_role.get("entry_trigger") or []
    if not triggers:
        return by_role, audit
    primary = triggers[0]
    card = kb.signals.get(primary.signal_name)
    if not card or not card.match_when or not card.match_when.mirrors_to:
        return by_role, audit
    mirror = card.match_when.mirrors_to
    mirror_card = kb.signals.get(mirror)
    if not mirror_card:
        logger.info(
            "catalog_picker|mirror_target_unknown|trigger=%s|mirror=%s",
            primary.signal_name, mirror,
        )
        return by_role, audit
    # Don't double-add.
    existing_exits = {m.signal_name for m in by_role.get("exit_trigger", [])}
    if mirror in existing_exits:
        return by_role, audit
    mirror_match = CatalogMatch(
        signal_name=mirror, role="exit_trigger",
        confidence=primary.confidence, params={},
        matched_phrase="(mirror)", matched_text="(opposite)",
        family=_family_of(mirror),
    )
    by_role.setdefault("exit_trigger", []).append(mirror_match)
    audit.append(f"exit_on_opposite:{primary.signal_name}->{mirror}")
    return by_role, audit


# ── Public entry point ────────────────────────────────────────────────────────


# Minimum overall confidence at which the picker's plan is preferred over a
# preset. Below this, the caller should fall back. Tunable.
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


# ── Generic auto-fill backstop ────────────────────────────────────────────────
#
# The backstop detects indicator families the user clearly named in their
# prompt and ensures at least one signal of that family is in the final plan.
# This works for EVERY family because it's driven by the catalog YAMLs — we
# look up the catalog's `family_default_filter` mapping per family and use the
# matching signal whose `acts_as` is `entry_filter` (regime check).
#
# Adding a new indicator just means writing its YAML with the right
# `match_when.acts_as` — no Python edits needed.

# Indicator family keyword detection. The keys here are the family tags; the
# values are the regex patterns that prove the user named that family in
# their prompt. We extract an optional `window` integer from the named group.
_FAMILY_PROMPT_PATTERNS: dict[str, list[str]] = {
    "sma":  [r"\b(?P<window>\d+)[\s\-]*(?:day|period|bar)?\s*sma\b", r"\bsma\s*\(?\s*(?P<window>\d+)\s*\)?\b", r"\bsimple\s+moving\s+average\b"],
    "ema":  [r"\b(?P<window>\d+)[\s\-]*(?:day|period|bar)?\s*ema\b", r"\bema\s*\(?\s*(?P<window>\d+)\s*\)?\b", r"\bexponential\s+moving\s+average\b"],
    "vwap": [r"\bvwap\b", r"volume[\s\-]+weighted\s+average\s+price"],
    "rsi":  [r"\brsi\s*\(?\s*(?P<window>\d+)\s*\)?\b", r"\brelative\s+strength\s+index\b", r"\brsi\b"],
    "macd": [r"\bmacd\b"],
    "adx":  [r"\badx\s*\(?\s*(?P<window>\d+)\s*\)?\b", r"\badx\b"],
    "atr":  [r"\batr\s*\(?\s*(?P<window>\d+)\s*\)?\b", r"\batr\b", r"\baverage\s+true\s+range\b"],
    "bb":   [r"\bbollinger(?:\s+bands?)?\b", r"\bbb\s+bands?\b"],
    "stoch": [r"\bstoch(?:astic)?\b"],
    "supertrend": [r"\bsuper\s*trend\b"],
    "keltner": [r"\bkeltner\b"],
    "obv":  [r"\bobv\b", r"on[\s\-]+balance\s+volume"],
    "cci":  [r"\bcci\b"],
    "mfi":  [r"\bmfi\b", r"money[\s\-]+flow"],
    "ichimoku": [r"\bichimoku\b"],
}


def _detect_family_mentions(prompt: str) -> dict[str, dict[str, Any]]:
    """For each indicator family, did the user mention it? With what period?"""
    out: dict[str, dict[str, Any]] = {}
    if not prompt:
        return out
    for family, patterns in _FAMILY_PROMPT_PATTERNS.items():
        period: int | None = None
        matched = False
        for pat in patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                matched = True
                try:
                    raw = m.groupdict().get("window")
                except AttributeError:
                    raw = None
                if raw:
                    try:
                        period = int(raw)
                    except (TypeError, ValueError):
                        pass
                if period is not None:
                    break
            if matched and period is not None:
                break
        if matched:
            out[family] = {"window": period}
    return out


def _pick_family_default_signal(
    family: str,
    direction: str,
    kb: KB,
    *,
    prefers_single_window: bool = True,
) -> SignalCard | None:
    """Pick the best 'regime' signal for the given family.

    Selection strategy (priority order):
      1. Direction-neutral entry_filter (e.g. adx_strong_trend works for any
         sentiment — pick it first when the user just said "ADX filter").
      2. entry_filter, matching direction, single window — the cleanest case.
      3. entry_filter, matching direction (any param shape).
      4. ANY role with matching direction + single window (e.g. price_below_sma
         is tagged exit_trigger but is the natural bearish-SMA filter too).
      5. entry_filter, any direction.
      6. Anything in this family.

    `prefers_single_window` should be True when the user gave one period.
    """
    candidates: list[SignalCard] = []
    for card in kb.signals.values():
        if _family_of(card.name) != family:
            continue
        if card.match_when is None:
            continue
        candidates.append(card)
    if not candidates:
        return None

    def _takes_single_window(c: SignalCard) -> bool:
        if c.match_when and "window" in (c.match_when.extract_params or {}):
            return True
        n = c.name.lower()
        return n.startswith(("price_above_", "price_below_", "is_above_", "is_below_"))

    # Priority 1: neutral-direction filter (most generic — "ADX filter" etc.)
    for c in candidates:
        if (
            c.match_when.acts_as == "entry_filter"
            and c.direction == "neutral"
        ):
            return c
    # Priority 2: filter, matching direction, single window
    if prefers_single_window:
        for c in candidates:
            if (
                c.match_when.acts_as == "entry_filter"
                and c.direction == direction
                and _takes_single_window(c)
            ):
                return c
    # Priority 3: filter, matching direction (any shape)
    for c in candidates:
        if (
            c.match_when.acts_as == "entry_filter"
            and c.direction == direction
        ):
            return c
    # Priority 4: ANY role, matching direction, single window — covers bearish
    # signals tagged exit_trigger that the user is reusing as a filter
    # ("trade below 200 SMA" for a short setup).
    if prefers_single_window:
        for c in candidates:
            if (
                c.direction == direction
                and _takes_single_window(c)
            ):
                return c
    # Priority 5: filter, any direction
    for c in candidates:
        if c.match_when.acts_as == "entry_filter":
            return c
    # Priority 6: anything in this family
    return candidates[0]


def auto_fill_missing_families(
    plan: dict[str, Any],
    prompt: str,
    *,
    kb: KB,
    timeframe: str | None,
    sentiment: str = "bullish",
) -> list[str]:
    """Inspect the prompt for indicator families the user named but the plan
    doesn't include. For each, append the family's default regime signal
    with the period the user specified (or the signal's default for that
    timeframe).

    Returns a list of audit entries describing each fill. Modifies `plan`
    in place. This is the backstop that prevents 'user said SMA but plan
    has no SMA signal' situations — works for every family without
    Python changes (just edit the catalog or _FAMILY_PROMPT_PATTERNS).
    """
    audit: list[str] = []
    if not prompt:
        return audit
    mentions = _detect_family_mentions(prompt)
    if not mentions:
        return audit

    # What families are already represented in the plan?
    present: set[str] = set()
    for bucket in ("entry", "exit"):
        for s in plan.get(bucket, []) or []:
            if not isinstance(s, dict):
                continue
            fam = _family_of(s.get("name", ""))
            if fam:
                present.add(fam)

    direction = "bullish" if sentiment != "bearish" else "bearish"

    for family, info in mentions.items():
        if family in present:
            continue
        prefers_single = info.get("window") is not None
        card = _pick_family_default_signal(
            family, direction, kb, prefers_single_window=prefers_single,
        )
        if card is None:
            continue
        # Build signal dict
        params: dict[str, Any] = {}
        if card.match_when and card.match_when.extract_params:
            # Pull the timeframe default if any
            for pname, pspec in card.match_when.extract_params.items():
                if pspec.default is not None:
                    params[pname] = pspec.default
        # Override window with user-specified period when present
        if info.get("window") is not None:
            params["window"] = info["window"]
        role = "FILTER" if card.match_when.acts_as == "entry_filter" else "TRIGGER"
        sig = {
            "name": card.name,
            "params": params,
            "timeframe": timeframe,
            "signal_type": role,
        }
        bucket = "exit" if card.match_when.acts_as == "exit_trigger" else "entry"
        plan.setdefault(bucket, []).append(sig)
        # Keep signals_used in sync.
        used = plan.setdefault("signals_used", [])
        if card.name not in used:
            used.append(card.name)
        audit.append(
            f"auto_fill:{family}->{card.name}({role.lower()},window={params.get('window')})"
        )
        logger.info(
            "catalog_picker|auto_fill_missing_family|family=%s|signal=%s"
            "|role=%s|window=%s|direction=%s",
            family, card.name, role.lower(), params.get("window"), direction,
        )

    return audit


def merge_picker_with_preset(
    preset_plan: dict[str, Any],
    picker_plan: dict[str, Any],
    *,
    picker_confidence: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[dict[str, Any], str]:
    """Combine the legacy preset plan with the catalog picker output.

    Mode A — picker has a TRIGGER above threshold: replace preset entirely
        with picker output (preserve preset's SL/TP underscore keys).
    Mode B — picker has filters/exits but no high-confidence trigger:
        keep preset's TRIGGER, replace preset's FILTERS with picker's filters,
        merge exits.
    Mode C — picker confidence below threshold: return preset unchanged.

    Returns (merged_plan, mode_audit_str).
    """
    picker_entry = picker_plan.get("entry") or []
    picker_triggers = [s for s in picker_entry if (s.get("signal_type") or "").upper() == "TRIGGER"]
    picker_filters  = [s for s in picker_entry if (s.get("signal_type") or "").upper() == "FILTER"]
    picker_exits    = picker_plan.get("exit") or []

    # Mode C: low confidence → preset wins outright
    if picker_confidence < confidence_threshold:
        return preset_plan, "mode=C(preset_only)"

    # Mode A: picker has a trigger → take picker's plan
    if picker_triggers:
        merged = dict(picker_plan)
        # Preserve any underscore-prefixed keys (SL/TP scaffolding) from preset
        for k, v in (preset_plan or {}).items():
            if k.startswith("_") and k not in merged:
                merged[k] = v
        return merged, f"mode=A(picker_full|trigger={picker_triggers[0].get('name')})"

    # Mode B: picker has only filters/exits → keep preset's TRIGGER, replace filters
    if picker_filters or picker_exits:
        merged = dict(preset_plan or {})
        preset_entry = list(merged.get("entry") or [])
        preset_triggers = [s for s in preset_entry if (s.get("signal_type") or "").upper() == "TRIGGER"]
        # New entry = preset's TRIGGERs + picker's FILTERS (dedup by name)
        new_entry: list[dict] = list(preset_triggers)
        seen: set[str] = {(s.get("name") or "").lower() for s in new_entry}
        for f in picker_filters:
            n = (f.get("name") or "").lower()
            if n and n not in seen:
                new_entry.append(f)
                seen.add(n)
        merged["entry"] = new_entry
        # Exits: replace if picker has any, else keep preset's
        if picker_exits:
            merged["exit"] = list(picker_exits)
        # Sync signals_used
        names = [
            (s.get("name") or "")
            for s in (merged.get("entry") or []) + (merged.get("exit") or [])
            if isinstance(s, dict) and s.get("name")
        ]
        merged["signals_used"] = names
        return merged, f"mode=B(preset_trigger+picker_filters|filters={len(picker_filters)}|exits={len(picker_exits)})"

    # Picker matched nothing meaningful → preset wins
    return preset_plan, "mode=C(preset_only)"


def pick_plan_from_catalog(
    prompt: str,
    *,
    kb: KB,
    timeframe: str | None,
    sentiment: str = "bullish",
    exit_on_opposite: bool = False,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> CatalogPick:
    """Build a signal_plan directly from the user's prompt via the catalog.

    Returns a CatalogPick. Caller decides whether to use the plan based on
    `confidence` — when high enough it bypasses the legacy preset pipeline.
    """
    matches = _find_all_matches(prompt, kb)
    if not matches:
        return CatalogPick(
            signal_plan={"entry": [], "exit": [], "signals_used": [], "signals_available": len(kb.signals)},
            matches=[], confidence=0.0,
            audit=["catalog_picker:no_matches"],
        )

    by_role = _resolve_role_conflicts(matches)

    # Apply "exit on opposite" if the user asked for it.
    audit: list[str] = []
    if exit_on_opposite:
        by_role, opp_audit = _apply_exit_on_opposite(by_role, kb, timeframe)
        audit.extend(opp_audit)

    entry_signals = [_build_signal(m, timeframe) for m in by_role.get("entry_trigger", [])]
    entry_signals.extend(
        _build_signal(m, timeframe) for m in by_role.get("entry_filter", [])
    )
    exit_signals = [_build_signal(m, timeframe) for m in by_role.get("exit_trigger", [])]
    exit_signals.extend(
        _build_signal(m, timeframe) for m in by_role.get("exit_filter", [])
    )

    signals_used = [
        s["name"] for s in entry_signals + exit_signals if s.get("name")
    ]

    plan: dict[str, Any] = {
        "entry":             entry_signals,
        "exit":              exit_signals,
        "signals_used":      signals_used,
        "signals_available": len(kb.signals),
    }

    # Overall confidence: average of the best trigger + the best filter.
    confs: list[float] = []
    if by_role.get("entry_trigger"):
        confs.append(by_role["entry_trigger"][0].confidence)
    if by_role.get("entry_filter"):
        confs.append(max(m.confidence for m in by_role["entry_filter"]))
    overall = round(sum(confs) / len(confs), 4) if confs else 0.0

    audit.append(
        f"catalog_picker:matches={len(matches)} triggers={len(by_role.get('entry_trigger', []))}"
        f" filters={len(by_role.get('entry_filter', []))}"
        f" exits={len(by_role.get('exit_trigger', []))}"
        f" exit_filters={len(by_role.get('exit_filter', []))}"
        f" confidence={overall}"
    )

    logger.info(
        "catalog_picker|pick|prompt_len=%d|matches=%d|trigger=%s|filters=%d"
        "|exits=%d|exit_filters=%d|confidence=%.3f|threshold=%.3f|use_picker=%s",
        len(prompt or ""), len(matches),
        (by_role.get("entry_trigger") or [None])[0].signal_name if by_role.get("entry_trigger") else None,
        len(by_role.get("entry_filter", [])),
        len(by_role.get("exit_trigger", [])),
        len(by_role.get("exit_filter", [])),
        overall, confidence_threshold,
        overall >= confidence_threshold,
    )

    return CatalogPick(
        signal_plan=plan,
        matches=matches,
        confidence=overall,
        used_preset_fallback=False,
        audit=audit,
    )
