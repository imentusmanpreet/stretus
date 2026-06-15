"""
app/planner/constraint_compiler.py — Apply user/semantic constraints to executable signal plans.

Runs AFTER preset hydration and ExecutionOrchestrator gates, BEFORE builder.apply_signal_plan.
Mutates entry/exit signal lists and rebuilds entry_condition / exit_condition strings so
metadata (RSI band, EMA windows, breakout rules) becomes executable logic.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any

from app.kb.execution_schemas import SemanticInstructions
from app.planner.formulas import render_formula

logger = logging.getLogger(__name__)

# Below this score, auto-confirming a signal plan is unsafe.
SEMANTIC_QUALITY_REVIEW_THRESHOLD = 0.35

_RSI_BAND_RE = re.compile(
    r"rsi\s+(?:between|from)\s+(\d+\.?\d*)\s+(?:and|to|-)\s+(\d+\.?\d*)",
    re.IGNORECASE,
)
_EMA_CROSS_RE = re.compile(
    r"(\d+)\s*ema\s*(?:>|above|over|≥|>=)\s*(\d+)\s*ema",
    re.IGNORECASE,
)
_PREV_TF_HIGH_RE = re.compile(
    r"(?:break(?:out|s)?\s+(?:above|over)|above)\s+(?:the\s+)?(?:previous|prior|last)\s+"
    r"(\d+)[\s-]*(?:min(?:ute)?s?|m)\s+(?:high|candle\s+high|bar\s+high)",
    re.IGNORECASE,
)
_EMA_SLOPE_BULL_RE = re.compile(
    r"ema\s*(\d+)\s+(?:slope|trend)\s+(?:positive|rising|bullish|upward)",
    re.IGNORECASE,
)
_RISING_EMA_RE = re.compile(
    r"(?:rising|positive|bullish|upward)\s+(\d+)\s*ema"
    r"|(\d+)\s*ema\s+(?:is\s+)?(?:rising|positive|bullish|(?:slope\s+)?positive)",
    re.IGNORECASE,
)
_ATR_STOP_RE = re.compile(
    r"(\d+\.?\d*)\s*(?:x\s*)?atr\s+(?:stop|sl)|atr\s+stop(?:\s+loss)?(?:\s+at)?\s*(\d+\.?\d*)?",
    re.IGNORECASE,
)

_EMA_CROSS_SIGNALS = frozenset({
    "ema_cross_up", "ema_cross_down", "ema_above", "ema_below",
    "price_above_ema", "price_below_ema",
})
_RSI_GENERIC_SIGNALS = frozenset({
    "rsi_above_50", "rsi_above_60", "rsi_below_50", "rsi_below_40",
    "rsi_above", "rsi_below",
})
_VOLUME_SIGNALS = frozenset({"volume_spike", "relative_volume_spike"})

# Maps for new-field consumers
_VWAP_RELATION_TO_SIGNAL = {
    "price_above_vwap": "vwap_bullish",
    "price_below_vwap": "vwap_bearish",
    "vwap_reclaim":     "vwap_reclaim_bullish",
    "vwap_bounce":      "vwap_reclaim_bullish",
}
_MACD_STATE_TO_SIGNAL = {
    "histogram_positive": "macd_positive",
    "histogram_negative": "macd_negative",
    "bullish_cross":      "macd_bullish_cross",
    "bearish_cross":      "macd_bearish_cross",
    # divergence: no canonical KB signal yet — caller will log unsupported.
}
_VWAP_FAMILY = frozenset({
    "vwap_bullish", "vwap_bearish",
    "vwap_reclaim_bullish", "vwap_reclaim_bearish",
    "vwap_zscore_overbought", "vwap_zscore_oversold",
})
_MACD_FAMILY = frozenset({
    "macd_positive", "macd_negative",
    "macd_bullish_cross", "macd_bearish_cross",
})
# Bullish-direction RSI threshold signals (entry filter), keyed by threshold
_RSI_BULLISH_THRESHOLD_SIGNALS = {
    50: "rsi_above_50",
    60: "rsi_above_60",
}
_RSI_BEARISH_THRESHOLD_SIGNALS = {
    50: "rsi_below_50",
    40: "rsi_below_40",
}


def needs_manual_review(quality_score: float) -> bool:
    """True when semantic extraction ran but confidence is too low to auto-finalize."""
    return 0.0 < quality_score < SEMANTIC_QUALITY_REVIEW_THRESHOLD


def apply_semantic_constraints(
    signal_plan: dict[str, Any],
    builder: Any,
    semantic_instructions: SemanticInstructions | None = None,
    source_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Deep-copy *signal_plan*, apply builder + semantic overrides, rebuild conditions.

    Order (user wins over preset):
      preset plan → semantic orchestrator → this compiler → apply_signal_plan
    """
    if not signal_plan:
        return signal_plan

    plan = copy.deepcopy(signal_plan)
    audit: list[str] = list(plan.get("_constraint_compiler_applied") or [])
    prompt = (source_prompt or getattr(builder, "goal", None) or "").strip()

    ema_windows = _resolve_ema_windows(builder, semantic_instructions, prompt)
    if ema_windows:
        fast, slow = ema_windows
        if _ensure_ema_cross_signal(plan, fast, slow, builder):
            audit.append(f"ema_cross:{fast}/{slow}")
        _patch_ema_signals(plan, fast, slow)

    if _apply_rsi_band(plan, builder):
        audit.append(
            f"rsi_band:{getattr(builder, 'rsi_entry_band_min', None)}"
            f"-{getattr(builder, 'rsi_entry_band_max', None)}"
        )

    if _apply_volume_ratio(plan, builder, semantic_instructions, prompt):
        audit.append(f"volume_ratio:{getattr(builder, 'volume_ratio_threshold', None)}")

    if _apply_breakout_rule(plan, builder, prompt):
        audit.append("breakout:prev_tf_high")

    # Apply ORB duration when user explicitly stated "first N-hour range" etc.
    orb_minutes = getattr(semantic_instructions, "orb_opening_bars_minutes", None)
    if orb_minutes and orb_minutes > 0:
        if _apply_orb_duration(plan, builder, orb_minutes):
            audit.append(f"orb_duration:{orb_minutes}min")

    if _apply_ema_slope_filter(plan, prompt):
        audit.append("ema_slope:positive")

    # Phase 12 — consume fine-grained semantic fields. These produce the
    # actual KB signal selections that match the user's literal prompt, e.g.
    # "price closes above VWAP" → vwap_bullish in entry, "MACD histogram
    # turns positive" → macd_positive, "RSI above 60" → rsi_above_60.
    tf = getattr(builder, "timeframe", None)
    audit.extend(_apply_ma_relations(plan, semantic_instructions, tf))
    audit.extend(_apply_vwap_relations(plan, semantic_instructions, tf))
    audit.extend(_apply_rsi_thresholds(plan, builder, semantic_instructions))
    audit.extend(_apply_macd_states(plan, semantic_instructions, tf))
    # Run "exit on opposite" AFTER all entry-trigger consumers so we have the
    # final entry trigger to mirror as the exit.
    audit.extend(_apply_exit_on_opposite(plan, semantic_instructions, tf))
    audit.extend(_apply_regime_preference(plan, semantic_instructions, tf))
    audit.extend(_apply_sl_type_hint(plan, builder, semantic_instructions))
    audit.extend(_apply_partial_exits(builder, semantic_instructions))

    # Phase 13 — subtractive pass: remove filter / exit signals whose
    # indicator family the user never mentioned. This is the gate that
    # prevents preset templates from auto-adding rsi_oversold, bb_squeeze,
    # supertrend_bullish, reference_above_sma, etc. on prompts that don't
    # ask for them. Runs LAST so we can see which families the additive
    # passes pinned in as triggers.
    original_user_prompt = (
        getattr(builder, "original_user_prompt", None) or prompt or ""
    )
    audit.extend(
        _subtract_unrequested_signals(plan, builder, original_user_prompt)
    )

    _apply_sl_tp_from_plan(plan, builder, semantic_instructions)
    _apply_time_exit(plan, builder, semantic_instructions, prompt)

    _hydrate_builder_metadata(builder, semantic_instructions, plan)
    _rebuild_conditions(plan)

    if _compile_exit_rules(plan, builder, semantic_instructions, prompt):
        audit.append("exit:compiled")

    _recompute_signals_used(plan)

    if audit:
        plan["_constraint_compiler_applied"] = audit
        logger.info(
            "constraint_compiler|applied|count=%d|changes=%s",
            len(audit),
            ", ".join(audit),
        )
    return plan


# ── EMA ───────────────────────────────────────────────────────────────────────


def _resolve_ema_windows(
    builder: Any,
    semantic: SemanticInstructions | None,
    prompt: str,
) -> tuple[int, int] | None:
    """Return (fast, slow) EMA periods from explicit user prose."""
    # 1. "9 EMA > 21 EMA" cross pattern (highest precision)
    m = _EMA_CROSS_RE.search(prompt)
    if m:
        fast, slow = int(m.group(1)), int(m.group(2))
        if fast < slow:
            return fast, slow
        return min(fast, slow), max(fast, slow)

    # 2. SemanticInstructions.indicators — require both windows; never infer.
    if semantic and semantic.indicators.get("EMA"):
        windows = sorted({int(w) for w in semantic.indicators["EMA"] if w})
        if len(windows) >= 2:
            return windows[0], windows[-1]
        if len(windows) == 1:
            logger.info(
                "constraint_compiler|ema_single_window|window=%d"
                "|action=skip_cross|reason=user_did_not_specify_second_window",
                windows[0],
            )

    # 3. Canonical semantic_intent on builder
    intent = getattr(builder, "semantic_intent", None) or {}
    if isinstance(intent, dict):
        ind = intent.get("indicators") or {}
        ema = ind.get("EMA") if isinstance(ind, dict) else None
        if isinstance(ema, list) and len(ema) >= 2:
            windows = sorted(int(x) for x in ema)
            return windows[0], windows[-1]

    return None


def _ensure_ema_cross_signal(
    plan: dict[str, Any],
    fast: int,
    slow: int,
    builder: Any,
) -> bool:
    """Insert ema_cross_up when user specified EMA windows but preset omitted EMA."""
    entry = plan.setdefault("entry", [])
    names = {(s.get("name") or "").lower() for s in entry if isinstance(s, dict)}
    if any(n in _EMA_CROSS_SIGNALS for n in names):
        return False
    entry.append({
        "name": "ema_cross_up",
        "params": {"window_fast": fast, "window_slow": slow},
        "signal_type": "FILTER",
        "source": "SEMANTIC",
    })
    # Persist on builder semantic intent indicators for downstream YAML
    intent = getattr(builder, "semantic_intent", None) or {}
    if isinstance(intent, dict):
        ind = dict(intent.get("indicators") or {})
        ind["EMA"] = [fast, slow]
        intent["indicators"] = ind
        builder.semantic_intent = intent
    return True


def _patch_ema_signals(plan: dict[str, Any], fast: int, slow: int) -> bool:
    changed = False
    for bucket in ("entry", "entry_filters"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict):
                continue
            name = (sig.get("name") or "").lower()
            if name not in _EMA_CROSS_SIGNALS:
                continue
            params = dict(sig.get("params") or {})
            if "window_fast" in params or name.startswith("ema_cross") or name in ("ema_above", "ema_below"):
                params["window_fast"] = fast
                params["window_slow"] = slow
                changed = True
            elif "period" in params or name.startswith("price_"):
                params["period"] = fast
                changed = True
            else:
                params["window_fast"] = fast
                params["window_slow"] = slow
                changed = True
            sig["params"] = params
    return changed


# ── RSI band ──────────────────────────────────────────────────────────────────


def _apply_rsi_band(plan: dict[str, Any], builder: Any) -> bool:
    rmin = getattr(builder, "rsi_entry_band_min", None)
    rmax = getattr(builder, "rsi_entry_band_max", None)
    if rmin is None and rmax is None:
        return False

    lo = float(rmin) if rmin is not None else 0.0
    hi = float(rmax) if rmax is not None else 100.0
    window = 14

    # Drop generic RSI threshold signals that contradict the band
    for bucket in ("entry", "entry_filters"):
        if bucket not in plan:
            continue
        plan[bucket] = [
            s for s in plan[bucket]
            if isinstance(s, dict)
            and (s.get("name") or "").lower() not in _RSI_GENERIC_SIGNALS
        ]

    band_clause = f"RSI({int(window)}) >= {lo:g} AND RSI({int(window)}) <= {hi:g}"
    extra = plan.setdefault("_entry_constraint_clauses", [])
    if band_clause not in extra:
        extra.append(band_clause)
    return True


# ── Volume ratio ──────────────────────────────────────────────────────────────


def _apply_volume_ratio(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None = None,
    prompt: str = "",
) -> bool:
    ratio = getattr(builder, "volume_ratio_threshold", None)
    if ratio is None and semantic and semantic.volume_ratio_threshold is not None:
        ratio = semantic.volume_ratio_threshold
        builder.volume_ratio_threshold = float(ratio)
    if ratio is None:
        m = re.search(
            r"volume\s+(?:at\s+least\s+|≥|>|above\s+)?(\d+\.?\d*)(?:x|×)",
            prompt,
            re.IGNORECASE,
        )
        if m:
            ratio = float(m.group(1))
            builder.volume_ratio_threshold = ratio
    if ratio is None:
        return False
    changed = False
    for bucket in ("entry", "entry_filters"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict):
                continue
            if (sig.get("name") or "").lower() not in _VOLUME_SIGNALS:
                continue
            params = dict(sig.get("params") or {})
            params["multiplier"] = float(ratio)
            sig["params"] = params
            changed = True
    return changed


# ── Breakout ──────────────────────────────────────────────────────────────────


def _apply_breakout_rule(
    plan: dict[str, Any],
    builder: Any,
    prompt: str,
) -> bool:
    m = _PREV_TF_HIGH_RE.search(prompt)
    if not m and not re.search(
        r"break(?:out|s)?\s+(?:above|over)\s+(?:previous|prior|last)\s+(?:high|bar)",
        prompt,
        re.IGNORECASE,
    ):
        return False

    tf_minutes = int(m.group(1)) if m else 15
    chart_tf = (getattr(builder, "timeframe", None) or "15m").lower().strip()
    window = _bars_for_timeframe(tf_minutes, chart_tf)

    entry = plan.setdefault("entry", [])
    names = {(s.get("name") or "").lower() for s in entry if isinstance(s, dict)}

    # Prefer Donchian-style breakout; demote generic ORB if present
    if "opening_range_breakout" in names and "n_bar_high_breakout" not in names:
        entry[:] = [
            s for s in entry
            if (s.get("name") or "").lower() != "opening_range_breakout"
        ]
        names.discard("opening_range_breakout")

    if "n_bar_high_breakout" not in names:
        entry.insert(0, {
            "name": "n_bar_high_breakout",
            "params": {"window": max(1, window)},
            "signal_type": "TRIGGER",
            "source": "SEMANTIC",
        })
        return True
    # Update window on existing breakout signal
    for sig in entry:
        if (sig.get("name") or "").lower() == "n_bar_high_breakout":
            sig.setdefault("params", {})["window"] = max(1, window)
            return True
    return False


def _apply_orb_duration(
    plan: dict[str, Any],
    builder: Any,
    duration_minutes: int,
) -> bool:
    """Replace generic n_bar_high_breakout with opening_range_breakout using the
    correct opening_bars computed from the user's explicitly stated range duration.

    "First 1-hour range on 15m chart" → opening_bars = 60/15 = 4 bars.

    This fixes the critical bug where:
      - n_bar_high_breakout(window=20) was used instead of opening_range_breakout
      - The formula CLOSE > MAX(HIGH, 20) can never fire (current HIGH ≥ CLOSE)
      - The correct formula CLOSE > OPENING_RANGE_HIGH(4) is computable and correct
    """
    chart_tf = (getattr(builder, "timeframe", None) or "15m").lower().strip()
    opening_bars = max(1, _bars_for_timeframe(duration_minutes, chart_tf))

    entry = plan.setdefault("entry", [])
    changed = False

    # Replace n_bar_high_breakout → opening_range_breakout with correct window
    for i, sig in enumerate(entry):
        name = (sig.get("name") or "").lower()
        if name == "n_bar_high_breakout":
            entry[i] = {
                "name": "opening_range_breakout",
                "params": {"opening_bars": opening_bars},
                "signal_type": sig.get("signal_type", "TRIGGER"),
                "timeframe": sig.get("timeframe", chart_tf),
                "source": "SEMANTIC_ORB_DURATION",
            }
            changed = True
            logger.info(
                "constraint_compiler|orb_duration|replaced n_bar_high_breakout"
                "|opening_bars=%d|duration_min=%d|chart_tf=%s",
                opening_bars, duration_minutes, chart_tf,
            )

    # Also replace n_bar_low_breakdown → opening_range_breakdown for short legs
    exit_and_filter = plan.get("exit", []) + plan.get("entry_filters", [])
    for i, sig in enumerate(plan.get("entry", [])):
        name = (sig.get("name") or "").lower()
        if name == "n_bar_low_breakdown":
            plan["entry"][i] = {
                "name": "opening_range_breakdown",
                "params": {"opening_bars": opening_bars},
                "signal_type": sig.get("signal_type", "TRIGGER"),
                "timeframe": sig.get("timeframe", chart_tf),
                "source": "SEMANTIC_ORB_DURATION",
            }
            changed = True

    return changed


def _apply_ema_slope_filter(plan: dict[str, Any], prompt: str) -> bool:
    """Add ema_sloping_up when user asks for positive EMA slope."""
    window: int | None = None
    m = _EMA_SLOPE_BULL_RE.search(prompt)
    if m:
        window = int(m.group(1))
    else:
        rm = _RISING_EMA_RE.search(prompt)
        if rm:
            window = int(rm.group(1) or rm.group(2) or 21)
    if window is None:
        return False
    entry = plan.setdefault("entry", [])
    names = {(s.get("name") or "").lower() for s in entry if isinstance(s, dict)}
    if "ema_sloping_up" in names:
        return False
    entry.append({
        "name": "ema_sloping_up",
        "params": {"window": window, "lookback": 3},
        "signal_type": "FILTER",
        "source": "SEMANTIC",
    })
    return True


def _bars_for_timeframe(target_minutes: int, chart_tf: str) -> int:
    # Parse the chart timeframe generically so any interval (2m, 7m, 45m, 4h, …)
    # scales correctly, instead of a fixed lookup that fell back to 15.
    from app.services.strategy.builder import _timeframe_to_minutes

    chart_m = _timeframe_to_minutes(chart_tf) or 0
    if chart_m <= 0:
        return max(1, target_minutes // 15)
    return max(1, round(target_minutes / chart_m))


# ── Subtractive filter (Phase 13) ─────────────────────────────────────────────

# Map signal-name substrings → indicator family they belong to. Used to
# decide whether a filter/trigger references a family the user actually
# mentioned in their original message.
#
# Order matters: longer / more specific prefixes are checked FIRST so a
# signal like "reference_above_sma" maps to REFERENCE (not SMA via the
# trailing "sma" substring), and "n_bar_high_breakout" maps to BREAKOUT
# (not VOLUME or anything else).
_SIGNAL_TO_FAMILY: tuple[tuple[str, str], ...] = (
    # Compound / prefix-style names — must come BEFORE the short keywords.
    ("reference_",       "REFERENCE"),
    ("market_structure", "STRUCTURE"),
    ("rejection_candle", "CANDLE"),
    ("inside_bar",       "CANDLE"),
    ("opening_range",    "BREAKOUT"),
    ("n_bar_high",       "BREAKOUT"),
    ("n_bar_low",        "BREAKOUT"),
    ("breakout",         "BREAKOUT"),
    ("gap_",             "GAP"),
    ("gap_up",           "GAP"),
    ("gap_down",         "GAP"),
    ("bb_",              "BB"),
    ("bollinger",        "BB"),
    # Short, unambiguous family substrings — order within is by selectivity.
    ("supertrend",  "SUPERTREND"),
    ("ichimoku",    "ICHIMOKU"),
    ("keltner",     "KELTNER"),
    ("donchian",    "DONCHIAN"),
    ("stoch",       "STOCHASTIC"),
    ("zscore",      "ZSCORE"),
    ("vwap",        "VWAP"),
    ("macd",        "MACD"),
    ("fvg",         "FVG"),
    ("adx",         "ADX"),
    ("atr",         "ATR"),
    ("obv",         "OBV"),
    ("cci",         "CCI"),
    ("mfi",         "MFI"),
    ("rsi",         "RSI"),
    ("sma",         "SMA"),
    ("ema",         "EMA"),
    ("volume",      "VOLUME"),
    ("delivery_volume", "VOLUME"),
)

# Keywords that imply each family was MENTIONED in the user message.
_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "SMA":        (r"\bsma\b", r"\bsimple\s+moving\s+average\b"),
    "EMA":        (r"\bema\b", r"\bexponential\s+(?:moving\s+)?average\b"),
    "VWAP":       (r"\bvwap\b", r"volume[\s\-]+weighted\s+average\s+price"),
    "RSI":        (r"\brsi\b", r"\brelative\s+strength\b"),
    "MACD":       (r"\bmacd\b",),
    "BB":         (r"\bbollinger\b", r"\bbb\s+bands?\b"),
    "ADX":        (r"\badx\b", r"\bdirectional\s+index\b"),
    "ATR":        (r"\batr\b", r"average\s+true\s+range"),
    "SUPERTREND": (r"\bsuper\s*trend\b",),
    "STOCHASTIC": (r"\bstoch(?:astic)?\b",),
    "ICHIMOKU":   (r"\bichimoku\b",),
    "KELTNER":    (r"\bkeltner\b",),
    "DONCHIAN":   (r"\bdonchian\b",),
    "OBV":        (r"\bobv\b", r"on[\s\-]+balance\s+volume"),
    "CCI":        (r"\bcci\b",),
    "MFI":        (r"\bmfi\b", r"money[\s\-]+flow"),
    "ZSCORE":     (r"\bz[\s\-]*score\b", r"standard\s+deviation"),
    "REFERENCE":  (r"\bnifty\b", r"\bbanknifty\b", r"\bsensex\b",
                   r"reference\s+symbol", r"outperform", r"relative\s+strength\s+(?:to|vs)"),
    "VOLUME":     (r"\bvolume\b",),
    "GAP":        (r"\bgap\s+(?:up|down|open)", r"opening\s+gap"),
    "BREAKOUT":   (r"\bbreak(?:out|s)?\b", r"\bopening\s+range\b",
                   r"prev(?:ious)?\s+day\s+(?:high|low)", r"\bn[\s\-]+bar\b"),
    "STRUCTURE":  (r"market\s+structure", r"swing\s+(?:high|low)", r"higher\s+(?:high|low)"),
    "FVG":        (r"\bfvg\b", r"fair\s+value\s+gap"),
    "CANDLE":     (r"bullish\s+candle", r"bearish\s+candle",
                   r"candle\s+closes?\s+(?:bullish|bearish)",
                   r"closes?\s+bullish", r"closes?\s+bearish",
                   r"rejection\s+candle", r"inside\s+bar", r"hammer", r"doji",
                   r"strong\s+bullish", r"strong\s+bearish"),
}


def _signal_family_of(name: str) -> str | None:
    n = (name or "").lower()
    for substr, family in _SIGNAL_TO_FAMILY:
        if substr in n:
            return family
    return None


def _families_mentioned_in_prompt(text: str) -> set[str]:
    """Set of indicator/concept families the user explicitly named."""
    if not text:
        return set()
    out: set[str] = set()
    for family, patterns in _FAMILY_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                out.add(family)
                break
    return out


def _subtract_unrequested_signals(
    plan: dict[str, Any],
    builder: Any,
    source_prompt: str,
) -> list[str]:
    """Drop signals whose indicator family the user never named.

    Rules:
      * Filters / exit triggers in a non-mentioned family are removed.
      * An entry TRIGGER in a non-mentioned family is removed ONLY if at
        least one other entry TRIGGER (of a mentioned family) survives —
        this prevents a preset's stale leading TRIGGER from drowning out
        the Phase-12 trigger our consumers added (e.g. preset says
        ``zscore_oversold`` but user asked for ``RSI > 30`` →
        ``rsi_cross_up`` was added → ``zscore_oversold`` should now go).
      * Signals whose family cannot be classified (``_signal_family_of``
        returns ``None``) are kept — better to leave it in than lose info.
    """
    if not source_prompt:
        return []

    requested = _families_mentioned_in_prompt(source_prompt)
    audit: list[str] = []

    # Classify every entry signal up front so we know which families are
    # represented and whether removing a TRIGGER would empty the plan.
    entry_sigs = [s for s in (plan.get("entry") or []) if isinstance(s, dict)]
    classified = [
        (s, (s.get("signal_type") or "").upper(), _signal_family_of(s.get("name", "")))
        for s in entry_sigs
    ]
    requested_triggers = [
        s for s, role, fam in classified
        if role == "TRIGGER" and fam and fam in requested
    ]

    # Pass 1 — entry signals (filters + triggers).
    new_entry: list[dict] = []
    for sig, role, fam in classified:
        if role == "TRIGGER":
            if fam and fam in requested:
                new_entry.append(sig)
                continue
            # Remove only when a user-aligned trigger exists to take over.
            if requested_triggers:
                audit.append(f"removed_trigger:{sig.get('name')}({fam}|not_in_prompt)")
                logger.info(
                    "constraint_compiler|subtract_unrequested|name=%s|family=%s"
                    "|role=trigger|reason=user_did_not_mention_family"
                    "|replacement_present=true",
                    sig.get("name"), fam,
                )
                continue
            new_entry.append(sig)
            continue
        # Filters: keep if family in requested OR family is None (unclassified)
        if fam is None or fam in requested:
            new_entry.append(sig)
            continue
        audit.append(f"removed_filter:{sig.get('name')}({fam}|not_in_prompt)")
        logger.info(
            "constraint_compiler|subtract_unrequested|name=%s|family=%s"
            "|role=filter|reason=user_did_not_mention_family",
            sig.get("name"), fam,
        )
    plan["entry"] = new_entry

    # Pass 2 — exits. Same rule: keep when family matches user text OR
    # when the entry triggers share the family (e.g. EMA entry → EMA exit).
    surviving_entry_trigger_families: set[str] = {
        _signal_family_of(s.get("name", ""))
        for s in new_entry
        if isinstance(s, dict) and (s.get("signal_type") or "").upper() == "TRIGGER"
        and _signal_family_of(s.get("name", ""))
    }
    new_exit: list[dict] = []
    for sig in plan.get("exit") or []:
        if not isinstance(sig, dict):
            new_exit.append(sig)
            continue
        fam = _signal_family_of(sig.get("name", ""))
        if fam is None or fam in requested or fam in surviving_entry_trigger_families:
            new_exit.append(sig)
            continue
        audit.append(f"removed_exit:{sig.get('name')}({fam}|not_in_prompt)")
        logger.info(
            "constraint_compiler|subtract_unrequested|name=%s|family=%s"
            "|role=exit|reason=user_did_not_mention_family",
            sig.get("name"), fam,
        )
    plan["exit"] = new_exit

    return audit


# ── New-field consumers (Phase 12) ────────────────────────────────────────────


def _direction_for_signal(signal_name: str) -> str:
    """Return 'bullish' or 'bearish' based on a KB signal name suffix."""
    n = signal_name.lower()
    if any(k in n for k in ("bullish", "_up", "_above", "positive", "oversold", "overbought_exit")):
        return "bullish"
    if any(k in n for k in ("bearish", "_down", "_below", "negative", "overbought", "oversold_exit")):
        return "bearish"
    return "bullish"  # safe default; never blocks


def _ensure_signal(
    plan: dict[str, Any],
    *,
    name: str,
    role: str,
    timeframe: str | None,
    params: dict | None = None,
) -> bool:
    """Insert a signal into plan['entry'] (TRIGGER or FILTER) if absent.

    Returns True when the signal was added.
    """
    bucket = plan.setdefault("entry", [])
    existing = {(s.get("name") or "").lower() for s in bucket if isinstance(s, dict)}
    if name.lower() in existing:
        return False
    sig: dict[str, Any] = {
        "name": name,
        "signal_type": role.upper(),
        "params": params or {},
    }
    if timeframe:
        sig["timeframe"] = timeframe
    bucket.append(sig)
    return True


def _apply_ma_relations(
    plan: dict[str, Any],
    semantic: SemanticInstructions | None,
    timeframe: str | None,
) -> list[str]:
    """Translate ma_relations into the precise price-vs-SMA/EMA KB signal.

    Behaviour mirrors VWAP: the first relation drives the TRIGGER role (and
    replaces any conflicting MA TRIGGER the preset injected, e.g. an
    ``ema_above`` cross when the user asked for "price crosses above 20
    SMA"). Subsequent relations become FILTERs / exits as appropriate.
    """
    audit: list[str] = []
    relations = getattr(semantic, "ma_relations", None) if semantic else None
    if not relations:
        return audit

    # Map (family, side, kind) → KB signal name.
    sig_for = {
        ("sma", "above", "cross"):  ("price_cross_above_sma", "TRIGGER"),
        ("sma", "below", "cross"):  ("price_cross_below_sma", "TRIGGER"),
        ("sma", "above", "regime"): ("is_above_sma",          "FILTER"),
        ("sma", "below", "regime"): ("price_below_sma",       "TRIGGER"),
        ("ema", "above", "cross"):  ("ema_above",             "TRIGGER"),
        ("ema", "below", "cross"):  ("ema_below",             "TRIGGER"),
        ("ema", "above", "regime"): ("price_above_ema",       "FILTER"),
        ("ema", "below", "regime"): ("price_below_ema",       "TRIGGER"),
    }

    for idx, rel in enumerate(relations):
        key = (rel.get("family"), rel.get("side"), rel.get("kind"))
        mapping = sig_for.get(key)
        if not mapping:
            logger.info("constraint_compiler|ma_relation_unmapped|relation=%s", rel)
            continue
        sig_name, default_role = mapping
        window = int(rel.get("window", 20))
        params = {"window": window}

        # First relation → drive the entry TRIGGER if the mapping says so.
        # Otherwise add as FILTER or as the exit when it's a price-below
        # regime in a long-only plan.
        if idx == 0 and default_role == "TRIGGER":
            # Replace any existing MA-family trigger.
            new_entry: list[dict] = []
            replaced = False
            for sig in plan.get("entry", []) or []:
                if not isinstance(sig, dict):
                    new_entry.append(sig)
                    continue
                name = (sig.get("name") or "").lower()
                fam = _signal_family_of(name)
                if (
                    sig.get("signal_type", "").upper() == "TRIGGER"
                    and fam in ("SMA", "EMA")
                    and name != sig_name
                ):
                    audit.append(
                        f"ma:replaced_trigger:{sig.get('name')}->{sig_name}({window})"
                    )
                    logger.info(
                        "constraint_compiler|ma_replace_trigger|old=%s|new=%s"
                        "|window=%d|family=%s|side=%s|kind=%s",
                        sig.get("name"), sig_name, window,
                        rel.get("family"), rel.get("side"), rel.get("kind"),
                    )
                    replaced = True
                    continue
                new_entry.append(sig)
            if replaced:
                plan["entry"] = new_entry
            if _ensure_signal(
                plan, name=sig_name, role="TRIGGER", timeframe=timeframe, params=params,
            ):
                if not replaced:
                    audit.append(f"ma:{key}->{sig_name}({window})(trigger)")
            else:
                # Already present — patch its params to match the user's window.
                for sig in plan.get("entry", []) or []:
                    if isinstance(sig, dict) and (sig.get("name") or "").lower() == sig_name:
                        sig_params = dict(sig.get("params") or {})
                        sig_params["window"] = window
                        sig["params"] = sig_params
        else:
            # Sell-side / regime relations: route to exit if it's a sell-side
            # cross trigger, otherwise add (or patch) as a filter.
            target_role = default_role
            target_bucket = "exit" if target_role == "TRIGGER" and "below" in sig_name else "entry"
            if target_bucket == "exit":
                exit_bucket = plan.setdefault("exit", [])
                existing = next(
                    (s for s in exit_bucket
                     if isinstance(s, dict) and (s.get("name") or "").lower() == sig_name),
                    None,
                )
                if existing is None:
                    exit_bucket.append({
                        "name": sig_name, "signal_type": "TRIGGER",
                        "timeframe": timeframe, "params": params,
                    })
                    audit.append(f"ma:{key}->{sig_name}({window})(exit)")
                elif (existing.get("params") or {}).get("window") != window:
                    existing_params = dict(existing.get("params") or {})
                    old_window = existing_params.get("window")
                    existing_params["window"] = window
                    existing["params"] = existing_params
                    audit.append(
                        f"ma:{key}->{sig_name}.window({old_window}->{window})(exit)"
                    )
            else:
                # FILTER (or existing same-name signal): patch window to the
                # user-specified value rather than silently keeping the
                # preset's default (e.g. user said EMA(50) but preset filter
                # has window=20).
                existing = next(
                    (s for s in plan.get("entry") or []
                     if isinstance(s, dict) and (s.get("name") or "").lower() == sig_name),
                    None,
                )
                if existing is None:
                    if _ensure_signal(
                        plan, name=sig_name, role=target_role,
                        timeframe=timeframe, params=params,
                    ):
                        audit.append(f"ma:{key}->{sig_name}({window})({target_role.lower()})")
                else:
                    existing_params = dict(existing.get("params") or {})
                    old_window = existing_params.get("window")
                    if old_window != window:
                        existing_params["window"] = window
                        existing["params"] = existing_params
                        audit.append(
                            f"ma:{key}->{sig_name}.window({old_window}->{window})"
                            f"({target_role.lower()})"
                        )

    return audit


def _apply_vwap_relations(
    plan: dict[str, Any],
    semantic: SemanticInstructions | None,
    timeframe: str | None,
) -> list[str]:
    """Translate the user's VWAP relations into the specific KB signal.

    Key behaviour: when the user states a VWAP relation, the chosen signal
    becomes the canonical VWAP signal in the plan. Any **other** VWAP
    variant the preset injected (e.g. ``vwap_reclaim_bullish`` when the
    user said "stays above VWAP" → ``vwap_bullish``) is **replaced**, not
    layered on top. This prevents semantic drift like "stays above" being
    silently mapped to "reclaims".

      * First relation → becomes / replaces the TRIGGER role.
      * Additional relations → added as FILTERs.
    """
    audit: list[str] = []
    if not (semantic and semantic.vwap_relations):
        return audit

    for idx, rel in enumerate(semantic.vwap_relations):
        sig_name = _VWAP_RELATION_TO_SIGNAL.get(rel)
        if not sig_name:
            logger.info("constraint_compiler|vwap_relation_unmapped|relation=%s", rel)
            continue

        if idx == 0:
            # First relation drives the trigger choice. If a different VWAP
            # variant is already in the entry as a TRIGGER, replace it.
            new_entry: list[dict] = []
            replaced = False
            for sig in plan.get("entry", []) or []:
                if not isinstance(sig, dict):
                    new_entry.append(sig)
                    continue
                name = (sig.get("name") or "").lower()
                if (
                    sig.get("signal_type", "").upper() == "TRIGGER"
                    and name in _VWAP_FAMILY
                    and name != sig_name
                ):
                    audit.append(
                        f"vwap:replaced_trigger:{sig.get('name')}->{sig_name}"
                    )
                    logger.info(
                        "constraint_compiler|vwap_replace_trigger|old=%s|new=%s"
                        "|relation=%s",
                        sig.get("name"), sig_name, rel,
                    )
                    replaced = True
                    continue
                new_entry.append(sig)
            if replaced:
                plan["entry"] = new_entry
            if _ensure_signal(plan, name=sig_name, role="TRIGGER", timeframe=timeframe):
                if not replaced:
                    audit.append(f"vwap:{rel}->{sig_name}(trigger)")
        else:
            if _ensure_signal(plan, name=sig_name, role="FILTER", timeframe=timeframe):
                audit.append(f"vwap:{rel}->{sig_name}(filter)")
    return audit


def _apply_rsi_thresholds(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
) -> list[str]:
    """Translate user RSI thresholds to signals/clauses.

    Rules:
      - "RSI above 60" / "RSI > 50" → entry filter (rsi_above_60 if exact, else
        a custom clause `RSI(14) >= N`).
      - "RSI crosses above 30" / "RSI from oversold" → entry trigger
        `rsi_cross_up` with threshold 30 / 40.
      - "RSI reaches above 65" → exit trigger `rsi_overbought` at 65.
      - "RSI below 30" → entry trigger `rsi_oversold` (long context) or filter.
    The function never invents thresholds the user didn't state.
    """
    audit: list[str] = []
    if not (semantic and semantic.rsi_thresholds):
        return audit

    # Drop generic RSI signals first so the planner's defaults don't clash.
    for bucket in ("entry", "entry_filters", "exit"):
        if bucket in plan:
            plan[bucket] = [
                s for s in plan[bucket]
                if not (
                    isinstance(s, dict)
                    and (s.get("name") or "").lower() in _RSI_GENERIC_SIGNALS
                )
            ]

    # Map each threshold to one of four roles based on value range:
    #   * "cross-above from low" (val ≤ 40, long): entry trigger rsi_cross_up
    #   * "trend filter"          (40 < val < 65, long): entry filter
    #   * "overbought exit"       (val ≥ 65, long): exit trigger rsi_overbought
    #   * "oversold entry"        (op=below, val ≤ 35, long): rsi_oversold
    tf = getattr(builder, "timeframe", None)
    sentiment = (getattr(builder, "sentiment", "") or "bullish").lower()
    is_long = sentiment == "bullish"

    for thr in semantic.rsi_thresholds:
        op = thr["op"]
        val = float(thr["value"])

        # ── Overbought exit (≥ 65) ──────────────────────────────────────────
        if op == "above" and is_long and val >= 65:
            exit_bucket = plan.setdefault("exit", [])
            existing = next(
                (s for s in exit_bucket
                 if isinstance(s, dict) and (s.get("name") or "").lower() == "rsi_overbought"),
                None,
            )
            if existing is None:
                exit_bucket.append({
                    "name": "rsi_overbought",
                    "signal_type": "TRIGGER",
                    "timeframe": tf,
                    "params": {"window": 14, "threshold": val},
                })
                audit.append(f"rsi_threshold:above_{val:g}->rsi_overbought(exit)")
            else:
                # Patch threshold to user's value (preset default is usually 70).
                params = dict(existing.get("params") or {})
                old_threshold = params.get("threshold")
                if old_threshold != val:
                    params["threshold"] = val
                    params.setdefault("window", 14)
                    existing["params"] = params
                    audit.append(
                        f"rsi_threshold:above_{val:g}->rsi_overbought.threshold"
                        f"({old_threshold}->{val:g})"
                    )
            continue

        # ── Cross-above from low / oversold-exit entry (≤ 40) ───────────────
        if op == "above" and is_long and val <= 40:
            if _ensure_signal(
                plan, name="rsi_cross_up", role="TRIGGER",
                timeframe=tf,
                params={"window": 14, "threshold": val},
            ):
                audit.append(f"rsi_threshold:above_{val:g}->rsi_cross_up(trigger)")
            continue

        # ── Trend filter (40 < val < 65) ────────────────────────────────────
        if op == "above" and is_long:
            target = _RSI_BULLISH_THRESHOLD_SIGNALS.get(int(val))
            if target and _ensure_signal(
                plan, name=target, role="FILTER", timeframe=tf,
                params={"window": 14, "threshold": val},
            ):
                audit.append(f"rsi_threshold:above_{val:g}->{target}(filter)")
            else:
                extra = plan.setdefault("_entry_constraint_clauses", [])
                clause = f"RSI(14) > {val:g}"
                if clause not in extra:
                    extra.append(clause)
                    audit.append(f"rsi_threshold:above_{val:g}->clause")
            continue

        # ── Oversold entry / bearish filter (below) ─────────────────────────
        if op == "below" and is_long and val <= 35:
            if _ensure_signal(
                plan, name="rsi_oversold", role="TRIGGER",
                timeframe=tf,
                params={"window": 14, "threshold": val},
            ):
                audit.append(f"rsi_threshold:below_{val:g}->rsi_oversold(trigger)")

    return audit


_MIRROR_EXIT_MAP = {
    "macd_bullish_cross":     "macd_bearish_cross",
    "macd_bearish_cross":     "macd_bullish_cross",
    "ema_above":              "ema_below",
    "ema_below":              "ema_above",
    "ema_cross_up":           "ema_cross_down",
    "ema_cross_down":         "ema_cross_up",
    "sma_cross_up":           "sma_cross_down",
    "sma_cross_down":         "sma_cross_up",
    "price_cross_above_sma":  "price_cross_below_sma",
    "price_cross_below_sma":  "price_cross_above_sma",
    "vwap_bullish":           "vwap_bearish",
    "vwap_bearish":           "vwap_bullish",
    "vwap_reclaim_bullish":   "vwap_reclaim_bearish",
    "vwap_reclaim_bearish":   "vwap_reclaim_bullish",
    "rsi_oversold":           "rsi_overbought",
    "rsi_cross_up":           "rsi_overbought",
    "bullish_market_structure_break": "bearish_market_structure_break",
    "bullish_fvg_signal":     "bearish_fvg_signal",
    "keltner_breakout_up":    "keltner_breakout_down",
}


def _apply_exit_on_opposite(
    plan: dict[str, Any],
    semantic: SemanticInstructions | None,
    timeframe: str | None,
) -> list[str]:
    """When the user said "exit on opposite crossover" (or "exit when it
    reverses"), mirror the entry trigger as the exit trigger and drop any
    unrelated exit signal the preset put in.

    Runs AFTER the additive consumers so the entry trigger is already
    finalised (e.g. macd_bullish_cross), giving us the correct seed for
    mirroring (→ macd_bearish_cross).
    """
    audit: list[str] = []
    if not (semantic and getattr(semantic, "exit_on_opposite", False)):
        return audit

    # Find the entry TRIGGER. Prefer a Phase-12 family the user named
    # explicitly (MACD / VWAP / SMA / EMA) over any preset leftover.
    entry_trigger_name: str | None = None
    for sig in plan.get("entry", []) or []:
        if not isinstance(sig, dict):
            continue
        if sig.get("signal_type", "").upper() != "TRIGGER":
            continue
        name = (sig.get("name") or "").lower()
        if name in _MIRROR_EXIT_MAP:
            entry_trigger_name = name
            break
    if entry_trigger_name is None:
        # Fall back to the first trigger we can find
        for sig in plan.get("entry", []) or []:
            if isinstance(sig, dict) and sig.get("signal_type", "").upper() == "TRIGGER":
                entry_trigger_name = (sig.get("name") or "").lower() or None
                break

    if not entry_trigger_name:
        return audit

    mirror_name = _MIRROR_EXIT_MAP.get(entry_trigger_name)
    if not mirror_name:
        logger.info(
            "constraint_compiler|exit_on_opposite_no_mirror|trigger=%s"
            "|action=keep_existing_exit",
            entry_trigger_name,
        )
        return audit

    # Replace the exit triggers with the mirror.
    exit_bucket = plan.get("exit") or []
    existing_names = {(s.get("name") or "").lower() for s in exit_bucket if isinstance(s, dict)}
    if mirror_name in existing_names:
        return audit

    # Drop unrelated exit triggers that don't match the user's mirror intent.
    new_exit: list[dict] = []
    dropped: list[str] = []
    for sig in exit_bucket:
        if isinstance(sig, dict) and sig.get("signal_type", "").upper() == "TRIGGER":
            sig_name = (sig.get("name") or "").lower()
            if sig_name == mirror_name:
                new_exit.append(sig)
                continue
            dropped.append(sig.get("name") or "?")
            continue
        new_exit.append(sig)
    new_exit.append({
        "name": mirror_name,
        "signal_type": "TRIGGER",
        "timeframe": timeframe,
        "params": {},
    })
    plan["exit"] = new_exit

    audit.append(
        f"exit_on_opposite:{entry_trigger_name}->{mirror_name}"
        + (f"|dropped={','.join(dropped)}" if dropped else "")
    )
    logger.info(
        "constraint_compiler|exit_on_opposite|trigger=%s|mirror=%s|dropped=%s",
        entry_trigger_name, mirror_name, dropped,
    )
    return audit


def _apply_macd_states(
    plan: dict[str, Any],
    semantic: SemanticInstructions | None,
    timeframe: str | None,
) -> list[str]:
    """Insert MACD signals matching the user's stated state(s).

    When the first MACD state is a TRIGGER (bullish_cross / bearish_cross)
    and the plan already contains a different-family TRIGGER from the
    preset, the preset's trigger is demoted to FILTER. This prevents
    plans like "MACD entry trigger ... AND EMA-cross entry trigger" when
    the user only asked for MACD.
    """
    audit: list[str] = []
    if not (semantic and semantic.macd_states):
        return audit

    has_macd = any(
        (s.get("name") or "").lower() in _MACD_FAMILY
        for s in plan.get("entry", []) or []
    )

    for idx, state in enumerate(semantic.macd_states):
        sig_name = _MACD_STATE_TO_SIGNAL.get(state)
        if not sig_name:
            logger.info("constraint_compiler|macd_state_unmapped|state=%s", state)
            continue

        # First MACD trigger drives the entry: demote competing triggers.
        if (
            idx == 0
            and not has_macd
            and state in ("bullish_cross", "bearish_cross")
        ):
            for sig in plan.get("entry", []) or []:
                if not isinstance(sig, dict):
                    continue
                if sig.get("signal_type", "").upper() != "TRIGGER":
                    continue
                fam = _signal_family_of(sig.get("name", ""))
                if fam and fam != "MACD":
                    sig["signal_type"] = "FILTER"
                    audit.append(
                        f"macd:demoted_trigger:{sig.get('name')}({fam})->filter"
                    )
                    logger.info(
                        "constraint_compiler|macd_demote_competing_trigger"
                        "|name=%s|family=%s|new_role=filter",
                        sig.get("name"), fam,
                    )

        role = "FILTER" if (has_macd or idx > 0) else "TRIGGER"
        if _ensure_signal(plan, name=sig_name, role=role, timeframe=timeframe):
            audit.append(f"macd:{state}->{sig_name}({role.lower()})")
            has_macd = True
    return audit


def _apply_regime_preference(
    plan: dict[str, Any],
    semantic: SemanticInstructions | None,
    timeframe: str | None,
) -> list[str]:
    """Add an ADX-based filter when the user asked to avoid sideways markets.

    For 'trending' preference we add `adx_strong_trend` (ADX > 25).
    """
    audit: list[str] = []
    if not (semantic and semantic.regime_preference):
        return audit

    pref = semantic.regime_preference
    if pref == "trending":
        if _ensure_signal(
            plan, name="adx_strong_trend", role="FILTER", timeframe=timeframe,
            params={"window": 14, "threshold": 25},
        ):
            audit.append("regime:trending->adx_strong_trend(filter)")
    elif pref == "ranging":
        # Keep the slot open for future: a low-ADX signal would go here.
        logger.info(
            "constraint_compiler|regime_ranging_not_implemented"
            "|no_kb_signal_for_adx_below_threshold_yet"
        )
    elif pref == "volatile":
        if _ensure_signal(
            plan, name="atr_high_volatility", role="FILTER", timeframe=timeframe,
            params={"window": 14},
        ):
            audit.append("regime:volatile->atr_high_volatility(filter)")
    return audit


def _apply_sl_type_hint(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
) -> list[str]:
    """When user specified an SL family (atr/structural/percent), set the spec
    on the builder so apply_signal_plan picks the right SL loader.

    Does NOT invent any value (no ATR multiplier, no SL %). It only sets the
    `type` so the downstream loader knows which family to use. The actual
    multiplier comes from the existing semantic_extractor flow (or from a
    further chat-layer clarification).
    """
    audit: list[str] = []
    if not (semantic and semantic.sl_type_hint):
        return audit

    hint = semantic.sl_type_hint
    existing = getattr(builder, "stop_loss_spec", None) or {}
    # Don't overwrite a more specific structural anchor already set.
    if existing.get("type") and existing.get("anchor") and hint == "structural":
        return audit

    if hint == "atr":
        # Only set if user didn't already give an ATR multiplier.
        if not (existing.get("type") == "atr" and existing.get("multiplier")):
            builder.stop_loss_spec = dict(existing, **{
                "type": "atr",
                "source": "semantic",
                # Multiplier remains None → chat layer will ask for it.
            })
            plan["_stop_loss_spec"] = dict(builder.stop_loss_spec)
            audit.append("sl_hint:atr->spec_set")
    elif hint == "percent":
        if not existing.get("type") == "percent":
            builder.stop_loss_spec = dict(existing, **{
                "type": "percent",
                "source": "semantic",
            })
            plan["_stop_loss_spec"] = dict(builder.stop_loss_spec)
            audit.append("sl_hint:percent->spec_set")
    # "structural" already handled by _apply_sl_tp_from_plan flow.
    return audit


def _apply_partial_exits(builder: Any, semantic: SemanticInstructions | None) -> list[str]:
    """Persist partial exit definitions onto the builder so the execution
    layer (or backtester) can act on them. We use builder.execution_layers
    keyed under 'partial_exits' for downstream consumers.
    """
    audit: list[str] = []
    if not (semantic and semantic.partial_exits):
        return audit

    layers = getattr(builder, "execution_layers", None) or {}
    if not isinstance(layers, dict):
        layers = {}
    existing = layers.get("partial_exits") or []
    merged = list(existing)
    for pe in semantic.partial_exits:
        if pe not in merged:
            merged.append(pe)
            audit.append(
                f"partial_exit:{pe.get('trigger')}={pe.get('value')}"
                + (f"@{pe.get('size_pct')}%" if pe.get("size_pct") else "")
            )
    if merged:
        layers["partial_exits"] = merged
        builder.execution_layers = layers
    return audit


# ── SL / TP / time exit ───────────────────────────────────────────────────────


def _apply_sl_tp_from_plan(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
) -> None:
    """Sync planner SL/TP onto builder; honor semantic RR when present."""
    sl = plan.get("_sl_pct")
    tp = plan.get("_tp_pct")
    if sl is not None and getattr(builder, "stop_loss", None) is None:
        builder.stop_loss = float(sl)
    if tp is not None and getattr(builder, "take_profit", None) is None:
        builder.take_profit = float(tp)

    rr_ratio = None
    if semantic and semantic.risk_reward and semantic.risk_reward.ratio:
        rr_ratio = float(semantic.risk_reward.ratio)
    elif plan.get("_semantic_risk_reward", {}).get("ratio"):
        rr_ratio = float(plan["_semantic_risk_reward"]["ratio"])

    if rr_ratio and rr_ratio > 0 and getattr(builder, "stop_loss", None):
        # Honor a user-given TP. Only overwrite when the user did NOT
        # provide an explicit TP — otherwise the user's intent wins, and we
        # leave the RR for downstream auditing without recomputing TP.
        # If user gave BOTH and they disagree, log a warning so the fidelity
        # validator can surface it as a clarification question.
        rms = dict(getattr(builder, "risk_execution_config", None) or {})
        sources = dict(rms.get("rms_sources", {}))
        user_gave_tp = sources.get("take_profit_pct") == "user"
        derived_tp = round(float(builder.stop_loss) * rr_ratio, 4)
        if user_gave_tp:
            # Compare; if mismatched > 0.05% absolute, log inconsistency for
            # the fidelity validator to flag.
            existing_tp = float(builder.take_profit or 0.0)
            if abs(existing_tp - derived_tp) > 0.05:
                logger.warning(
                    "constraint_compiler|tp_rr_inconsistent|user_tp=%.3f|rr=%.3f|sl=%.3f"
                    "|derived_tp=%.3f|action=keep_user_tp",
                    existing_tp, rr_ratio, float(builder.stop_loss), derived_tp,
                )
                # Record both so the fidelity validator can ask the user.
                rms["risk_reward"] = rr_ratio
                rms["_tp_rr_inconsistent"] = {
                    "user_tp": existing_tp,
                    "rr_ratio": rr_ratio,
                    "derived_tp": derived_tp,
                }
                sources.setdefault("risk_reward", "user")
                rms["rms_sources"] = sources
                builder.risk_execution_config = rms
        else:
            builder.take_profit = derived_tp
            if sources.get("take_profit_pct") != "user":
                rms["take_profit_pct"] = builder.take_profit
                rms["risk_reward"] = rr_ratio
                sources["take_profit_pct"] = "semantic"
                sources["risk_reward"] = "semantic"
                rms["rms_sources"] = sources
                builder.risk_execution_config = rms


def _apply_time_exit(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
    prompt: str,
) -> None:
    """Square-off / mandatory exit time from prose or entry window end."""
    exit_time = None
    if re.search(
        r"(?:square[\s-]?off|mandatory\s+exit|force\s+exit|exit\s+all)\s+(?:at|by)\s+(\d{1,2}:\d{2})",
        prompt,
        re.IGNORECASE,
    ):
        m = re.search(
            r"(?:square[\s-]?off|mandatory\s+exit|force\s+exit|exit\s+all)\s+(?:at|by)\s+(\d{1,2}:\d{2})",
            prompt,
            re.IGNORECASE,
        )
        if m:
            exit_time = m.group(1)
    if not exit_time:
        exit_time = getattr(builder, "entry_window_end", None)

    if exit_time and not plan.get("_time_exit"):
        plan["_time_exit"] = {
            "exit_time": str(exit_time).strip(),
            "timezone": "Asia/Kolkata",
        }
        builder.time_exit = dict(plan["_time_exit"])


# ── Rebuild condition strings ─────────────────────────────────────────────────


def _rebuild_conditions(plan: dict[str, Any]) -> None:
    """Recompute entry_condition / exit_condition from (possibly mutated) signals."""
    entry_parts: list[str] = []
    for bucket in ("entry", "entry_filters"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict) or not sig.get("name"):
                continue
            part = render_formula(sig["name"], dict(sig.get("params") or {}))
            if part:
                entry_parts.append(part)

    for clause in plan.get("_entry_constraint_clauses") or []:
        if clause and clause not in entry_parts:
            entry_parts.append(clause)

    if entry_parts:
        plan["entry_condition"] = " AND ".join(entry_parts)

    exit_parts: list[str] = []
    for sig in plan.get("exit") or []:
        if not isinstance(sig, dict) or not sig.get("name"):
            continue
        part = render_formula(sig["name"], dict(sig.get("params") or {}))
        if part:
            exit_parts.append(part)

    if exit_parts:
        plan["exit_condition"] = " OR ".join(exit_parts)


def _recompute_signals_used(plan: dict[str, Any]) -> None:
    """Refresh signals_used after semantic rewrites (ORB → n_bar_high_breakout, etc.)."""
    names: list[str] = []
    seen: set[str] = set()
    for bucket in ("entry", "entry_filters", "exit"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict):
                continue
            name = (sig.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    if names:
        plan["signals_used"] = names


def _hydrate_builder_metadata(
    builder: Any,
    semantic: SemanticInstructions | None,
    plan: dict[str, Any],
) -> None:
    """Sync compiled values back onto builder so root/draft JSON stops showing nulls."""
    if getattr(builder, "volume_ratio_threshold", None) is None:
        for bucket in ("entry", "entry_filters"):
            for sig in plan.get(bucket) or []:
                if not isinstance(sig, dict):
                    continue
                if (sig.get("name") or "").lower() not in _VOLUME_SIGNALS:
                    continue
                mult = (sig.get("params") or {}).get("multiplier")
                if mult is not None:
                    builder.volume_ratio_threshold = float(mult)
                    break
        if builder.volume_ratio_threshold is None and semantic and semantic.volume_ratio_threshold:
            builder.volume_ratio_threshold = float(semantic.volume_ratio_threshold)

    if plan.get("_stop_loss_spec") and not getattr(builder, "stop_loss_spec", None):
        builder.stop_loss_spec = dict(plan["_stop_loss_spec"])
    if plan.get("_trailing_stop_spec") and not getattr(builder, "trailing_stop_spec", None):
        builder.trailing_stop_spec = dict(plan["_trailing_stop_spec"])
    if plan.get("_time_exit") and not getattr(builder, "time_exit", None):
        builder.time_exit = dict(plan["_time_exit"])


def _semantic_sl_to_loader_spec(sl: Any) -> dict[str, Any] | None:
    """Map SemanticInstructions.stop_loss to quant_engine loader shape.

    Returns None when the semantic stop carries no recognisable BASIS — i.e. the
    user merely mentioned "stop loss" without naming a structure (swing/ORB low) or
    ATR. In that case we do NOT fabricate a typed (ATR) stop; returning None lets
    the caller fall back to the plain percent `stop_loss_pct` (the tier/objective
    default). A vague mention should become a simple percent stop, not an invented
    ATR one.
    """
    padding = sl.padding
    if padding and padding.method == "atr":
        return {
            "type": "atr",
            "multiplier": float(padding.atr_multiple or 1.5),
            "window": 14,
        }
    anchor = (sl.anchor or "").strip().lower()
    if anchor in {
        "swing_low_recent", "opening_range_low", "opening_range_high",
        "candle_low", "reclaim_candle", "vwap_deviation", "opposite_side",
        "prev_n_bar_low", "prev_n_bar_high",
    }:
        spec: dict[str, Any] = {"type": "structural", "anchor": anchor}
        if "opening_range" in anchor:
            spec["opening_bars"] = 3
        else:
            spec["window"] = 5
        if padding and padding.method == "percent" and padding.percent:
            spec["padding_pct"] = float(padding.percent)
        return spec
    # Plain ATR stop phrasing without structural anchor.
    if re.search(r"atr", (sl.description or sl.type or ""), re.IGNORECASE):
        mult = 1.5
        if padding and padding.atr_multiple:
            mult = float(padding.atr_multiple)
        return {"type": "atr", "multiplier": mult, "window": 14}
    # No structure, no ATR → default to a plain percent stop (caller uses
    # stop_loss_pct), instead of inventing an ATR stop the user never asked for.
    return None


def _semantic_ts_to_loader_spec(ts: Any) -> dict[str, Any]:
    """Map TrailingStopConfig to quant_engine loader shape."""
    raw_type = (ts.type or "").lower()
    spec: dict[str, Any] = {
        "activate_after_pct": float(ts.activate_after_pct or 0.0),
    }
    if raw_type in ("ema_based", "ema"):
        return {
            **spec,
            "type": "ema",
            "window": int(ts.ema_period or 9),
        }
    if raw_type in ("atr_based", "atr", "chandelier"):
        return {
            **spec,
            "type": "atr",
            "multiplier": float(ts.trail_distance_atr_multiple or 2.0),
            "window": 14,
        }
    if raw_type == "percent_based":
        return {
            **spec,
            "type": "percent",
            "distance_pct": float(ts.trail_distance_percent or 1.0),
        }
    return {**spec, "type": "ema", "window": int(ts.ema_period or 9)}


def _compile_exit_rules(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
    prompt: str,
) -> bool:
    """Materialize ATR/structural SL, trailing stop, RR, and time exit on plan + builder."""
    changed = False
    sl_spec: dict[str, Any] | None = None
    ts_spec: dict[str, Any] | None = None

    if getattr(builder, "stop_loss_spec", None):
        sl_spec = dict(builder.stop_loss_spec)
    elif semantic and semantic.stop_loss:
        sl_spec = _semantic_sl_to_loader_spec(semantic.stop_loss)
    elif _ATR_STOP_RE.search(prompt):
        m = _ATR_STOP_RE.search(prompt)
        raw_mult = m.group(1) or m.group(2) if m else None
        if raw_mult:
            mult = float(raw_mult)
            sl_spec = {"type": "atr", "multiplier": mult, "window": 14}
        else:
            logger.warning(
                "constraint_compiler|atr_stop_missing_multiplier"
                "|action=skip_atr_sl|reason=user_said_atr_stop_without_multiplier"
                "|prompt_snippet=%s",
                prompt[:120],
            )

    if sl_spec:
        sl_spec.setdefault("source", "semantic")
        plan["_stop_loss_spec"] = sl_spec
        builder.stop_loss_spec = dict(sl_spec)
        changed = True

    if getattr(builder, "trailing_stop_spec", None):
        ts_spec = dict(builder.trailing_stop_spec)
    elif semantic and semantic.trailing_stop and semantic.trailing_stop.enabled:
        ts_spec = _semantic_ts_to_loader_spec(semantic.trailing_stop)
    elif re.search(r"trailing\s+stop|trail\s+(?:sl|stop)|ema\s+trail", prompt, re.IGNORECASE):
        ema_m = re.search(r"ema\s*(\d+)", prompt, re.IGNORECASE)
        ts_spec = {
            "type": "ema",
            "window": int(ema_m.group(1)) if ema_m else 9,
            "activate_after_pct": 0.0,
            "source": "semantic",
        }

    if ts_spec:
        ts_spec.setdefault("source", "semantic")
        plan["_trailing_stop_spec"] = ts_spec
        builder.trailing_stop_spec = dict(ts_spec)
        changed = True

    # Rich exit: percent SL/TP engine path + optional signal exits (not VWAP-only)
    exit_parts: list[str] = []
    if sl_spec or ts_spec or plan.get("_time_exit"):
        exit_parts.append(
            "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
        )
    for sig in plan.get("exit") or []:
        if not isinstance(sig, dict):
            continue
        name = (sig.get("name") or "").lower()
        # VWAP flip alone is too weak when user asked for ATR/trailing/time exits
        if sl_spec and name in ("vwap_bearish", "price_below_vwap", "below_vwap"):
            continue
        part = render_formula(sig.get("name", ""), dict(sig.get("params") or {}))
        if part and part not in exit_parts:
            exit_parts.append(part)

    if exit_parts:
        plan["exit_condition"] = " OR ".join(exit_parts)
        changed = True

    plan["risk_model"] = {
        "stop_loss": sl_spec,
        "trailing_stop": ts_spec,
        "risk_reward": (
            semantic.risk_reward.ratio
            if semantic and semantic.risk_reward and semantic.risk_reward.ratio
            else (plan.get("_semantic_risk_reward") or {}).get("ratio")
        ),
        "time_exit": plan.get("_time_exit"),
    }
    return changed
