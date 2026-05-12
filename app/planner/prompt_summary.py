"""
app/planner/prompt_summary.py — Build a comprehensive user-facing summary of
everything captured from a strategy prompt, detect critical gaps, and enforce
zero-loss capture of user-stated conditions in the planner output.

Three public helpers:

1. build_prompt_summary(builder, instructions, extra_prompt_text=None)
   Returns a dict with:
     - text: human-readable confirmation message body listing every captured
       detail (core fields + semantic extraction layers).
     - snapshot: structured dict suitable for persisting on the builder so it
       survives across turns and can be rehydrated.
     - missing_critical: list of {field, label} entries the user never
       supplied and the system refuses to default.

2. find_missing_critical_inputs(builder, instructions)
   Pure detection — returns the same missing_critical list. The user is
   blocked at the confirmation step until each entry is resolved.

3. enforce_zero_loss_capture(plan, instructions)
   Mutates `plan` so its entry_condition / exit_condition strings reflect
   every semantic instruction extracted from the prompt. Returns a dict
   describing what was augmented (for logging + telemetry).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.kb.execution_schemas import SemanticInstructions

logger = logging.getLogger(__name__)


_INDICATOR_DEFAULT_REPRS: dict[str, str] = {
    "EMA":   "EMA({n})",
    "SMA":   "SMA({n})",
    "RSI":   "RSI({n})",
    "ATR":   "ATR({n})",
    "ADX":   "ADX({n})",
    "MACD":  "MACD()",
    "VWAP":  "VWAP()",
}

# Vocabulary the threshold extractor recognises on the left-hand side of
# comparison phrases. Each entry's "lhs_groups" tells the resolver how many
# of the match's leading groups belong to the LHS (so threshold groups added
# by the comparison clause aren't misread as the indicator's window length).
_THRESHOLD_LHS_TOKENS: list[dict[str, Any]] = [
    {"pattern": r"\brsi\s*\(\s*(\d+)\s*\)",  "template": "RSI({n})", "lhs_groups": 1},
    {"pattern": r"\brsi\b",                  "template": "RSI({n})", "lhs_groups": 0},
    {"pattern": r"\badx\s*\(\s*(\d+)\s*\)",  "template": "ADX({n})", "lhs_groups": 1},
    {"pattern": r"\badx\b",                  "template": "ADX({n})", "lhs_groups": 0},
    {"pattern": r"\batr\s*\(\s*(\d+)\s*\)",  "template": "ATR({n})", "lhs_groups": 1},
    {"pattern": r"\batr\b",                  "template": "ATR({n})", "lhs_groups": 0},
    {"pattern": r"\bmfi\s*\(\s*(\d+)\s*\)",  "template": "MFI({n})", "lhs_groups": 1},
    {"pattern": r"\bmfi\b",                  "template": "MFI({n})", "lhs_groups": 0},
    {"pattern": r"\b(\d+)\s*ema\b",          "template": "EMA({n})", "lhs_groups": 1},
    {"pattern": r"\bema\s*\(\s*(\d+)\s*\)",  "template": "EMA({n})", "lhs_groups": 1},
    {"pattern": r"\b(\d+)\s*sma\b",          "template": "SMA({n})", "lhs_groups": 1},
    {"pattern": r"\bsma\s*\(\s*(\d+)\s*\)",  "template": "SMA({n})", "lhs_groups": 1},
    {"pattern": r"\b(?:close|price)\b",      "template": "CLOSE",    "lhs_groups": 0},
    {"pattern": r"\bvolume\b",               "template": "VOLUME",   "lhs_groups": 0},
    {"pattern": r"\bvwap\b",                 "template": "VWAP()",   "lhs_groups": 0},
    {"pattern": r"\bmacd\b",                 "template": "MACD",     "lhs_groups": 0},
]

_DEFAULT_WINDOWS: dict[str, int] = {
    "RSI": 14, "ADX": 14, "ATR": 14, "MFI": 14,
    "STOCH_K": 14, "STOCH_D": 3, "WILLR": 14, "CCI": 20, "CMF": 20,
    "ROC": 10, "CMO": 9, "MOMENTUM": 10, "TRIX": 15, "DISPARITY": 14,
    "STDEV": 20, "HV": 20, "CHOPPINESS": 14,
    "BB_WIDTH": 20, "BB_PCT_B": 20,
    "VOLUME_SMA": 20, "VROC": 12,
    "AROON_UP": 14, "AROON_DOWN": 14, "AROON_OSC": 14,
    "HHV": 20, "LLV": 20, "DON_UPPER": 20, "DON_LOWER": 20, "DON_MID": 20,
}


def _catalog_lhs_tokens() -> list[dict]:
    """Build the LHS-pattern table dynamically from the indicator catalog.

    Each catalog entry that's marked engine_status="implemented" and is
    a periodic indicator (single int period) becomes a recognisable LHS
    token in threshold expressions like 'STOCH_K above 80'. The static
    _THRESHOLD_LHS_TOKENS list still wins for hand-tuned forms like
    'close', 'price', '20 EMA' that benefit from explicit regexes.
    """
    from app.kb.indicator_catalog import CATALOG

    extra: list[dict] = []
    seen: set[str] = {t["template"].split("(")[0].split("{")[0].rstrip("(") for t in _THRESHOLD_LHS_TOKENS}
    for spec in CATALOG.values():
        if spec.engine_status != "implemented":
            continue
        if spec.name in seen:
            continue
        # Only handle scalar (no-param) and single-int-param indicators
        # automatically. Multi-param ones (Supertrend) are matched via
        # their default token in the static table when needed.
        if not spec.params:
            for alias in (spec.name.lower(), *spec.aliases):
                extra.append({
                    "pattern":    r"\b" + re.escape(alias) + r"\b",
                    "template":   spec.token_template,
                    "lhs_groups": 0,
                })
            seen.add(spec.name)
            continue
        if len(spec.params) == 1 and spec.params[0].kind == "int":
            default_window = int(spec.params[0].default or _DEFAULT_WINDOWS.get(spec.name, 14))
            template = spec.token_template.format(**{spec.params[0].name: "{n}"})
            # Also write the indicator's default window into _DEFAULT_WINDOWS
            # so _resolve_lhs_token picks it up when the user didn't specify.
            _DEFAULT_WINDOWS.setdefault(spec.name, default_window)
            for alias in (spec.name.lower(), *spec.aliases):
                extra.append({
                    "pattern":    r"\b" + re.escape(alias) + r"\s*\(\s*(\d+)\s*\)",
                    "template":   template,
                    "lhs_groups": 1,
                })
                extra.append({
                    "pattern":    r"\b" + re.escape(alias) + r"\b",
                    "template":   template,
                    "lhs_groups": 0,
                })
            seen.add(spec.name)
    return extra


# ── 0. Threshold-condition extraction ─────────────────────────────────────────
#
# The SemanticExtractor catches structured semantic layers (HTF rules, SL,
# trailing stop, RR, session) but does NOT capture inline comparative
# conditions like "RSI between 40 and 65", "price above 9 EMA",
# "volume > 1.5x average". These are the literal AND-clauses Feature 2
# refuses to lose. This pass walks the prompt for those phrases and
# emits machine-readable expressions the engine can evaluate alongside
# whatever the KB planner picks.


def extract_threshold_conditions(prompt_text: str) -> list[dict[str, Any]]:
    """Return a list of {phrase, expression, kind} entries for every
    comparative condition recognised in `prompt_text`. The phrase is the
    user's original wording (kept for the summary); the expression is the
    machine-readable form (used by enforce_zero_loss_capture).
    """
    if not prompt_text:
        return []

    text = re.sub(r"\s+", " ", prompt_text).strip()
    out: list[dict[str, Any]] = []
    seen_expressions: set[str] = set()

    def _record(phrase: str, expression: str, kind: str) -> None:
        expr = expression.strip()
        if not expr or expr in seen_expressions:
            return
        seen_expressions.add(expr)
        out.append({
            "phrase":     phrase.strip(),
            "expression": expr,
            "kind":       kind,
        })

    # RHS atom: either a (possibly negative) number with optional %, or an
    # indicator literal like "9 EMA" / "EMA(9)" / "VWAP" / "ATR(14)". The
    # more specific alternatives are listed first so they win the
    # alternation.
    rhs_atom = (
        r"(?:"
        r"\d+\s*(?:ema|sma|atr|rsi|adx)\s*\(\s*\d+\s*\)"
        r"|(?:ema|sma|atr|rsi|adx|mfi)\s*\(\s*\d+\s*\)"
        r"|\d+\s*(?:ema|sma|atr)\b"
        r"|vwap\s*\(\s*\)?"
        r"|vwap\b"
        r"|-?\d+(?:\.\d+)?\s*%?"
        r")"
    )

    # Static hand-tuned tokens win first, then catalog-derived tokens so any
    # engine-implemented indicator is recognisable on the LHS of a threshold
    # expression even if it isn't enumerated above.
    lhs_specs = list(_THRESHOLD_LHS_TOKENS) + _catalog_lhs_tokens()
    for spec in lhs_specs:
        lhs_pattern = spec["pattern"]
        lhs_template = spec["template"]
        lhs_groups   = spec["lhs_groups"]

        # "X between A and B" → A <= X <= B (negatives allowed)
        between_re = re.compile(
            lhs_pattern + r"\s+between\s+(-?\d+(?:\.\d+)?)\s+(?:and|to|\-)\s+(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        for m in between_re.finditer(text):
            lhs_token = _resolve_lhs_token(lhs_template, m, lhs_groups)
            lo = m.group(lhs_groups + 1)
            hi = m.group(lhs_groups + 2)
            _record(m.group(0), f"{lhs_token} >= {lo} AND {lhs_token} <= {hi}", "between")

        # "X above Y" / ">", "greater than", "crosses above", "breaks above"
        above_re = re.compile(
            lhs_pattern
            + r"\s+(?:is\s+)?(?:crosses?\s+above|breaks?\s+above|above|over|greater\s+than|>\s*=?)\s+"
            + r"(" + rhs_atom + r")",
            re.IGNORECASE,
        )
        for m in above_re.finditer(text):
            lhs_token = _resolve_lhs_token(lhs_template, m, lhs_groups)
            rhs = _normalise_rhs(m.group(lhs_groups + 1))
            _record(m.group(0), f"{lhs_token} > {rhs}", "above")

        # "X below Y" / "<", "less than", "crosses below", "breaks below"
        below_re = re.compile(
            lhs_pattern
            + r"\s+(?:is\s+)?(?:crosses?\s+below|breaks?\s+below|below|under|less\s+than|<\s*=?)\s+"
            + r"(" + rhs_atom + r")",
            re.IGNORECASE,
        )
        for m in below_re.finditer(text):
            lhs_token = _resolve_lhs_token(lhs_template, m, lhs_groups)
            rhs = _normalise_rhs(m.group(lhs_groups + 1))
            _record(m.group(0), f"{lhs_token} < {rhs}", "below")

    return out


def _resolve_lhs_token(template: str, match: re.Match[str], lhs_groups: int) -> str:
    if "{n}" not in template:
        return template
    window: int | None = None
    for i in range(1, lhs_groups + 1):
        grp = match.group(i)
        if grp and str(grp).isdigit():
            window = int(grp)
            break
    if window is None:
        indicator_match = re.match(r"([A-Z]+)", template)
        indicator = indicator_match.group(1) if indicator_match else ""
        window = _DEFAULT_WINDOWS.get(indicator, 14)
    return template.replace("{n}", str(window))


def _normalise_rhs(raw: str) -> str:
    rhs = re.sub(r"\s+", " ", raw or "").strip()
    # Strip trailing percent signs: "2%" → "2"
    rhs = re.sub(r"\s*%$", "", rhs)
    # "9 EMA"  →  "EMA(9)"
    m = re.match(r"^(\d+)\s*(ema|sma|atr)\b\s*(?:\(\s*\d+\s*\))?$", rhs, re.IGNORECASE)
    if m:
        return f"{m.group(2).upper()}({m.group(1)})"
    # "ema(20)" → "EMA(20)"
    m = re.match(r"^(ema|sma|atr|rsi|adx|mfi)\s*\(\s*(\d+)\s*\)$", rhs, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}({m.group(2)})"
    # bare indicator tokens
    if rhs.lower() in {"vwap", "vwap()"}:
        return "VWAP()"
    return rhs


# ── 1. User-facing prompt summary ─────────────────────────────────────────────


def build_prompt_summary(
    builder: Any,
    instructions: SemanticInstructions | None,
    extra_prompt_text: str | None = None,
) -> dict[str, Any]:
    """Produce a comprehensive snapshot + confirmation text for the user.

    The text mirrors every dimension the user can have mentioned: strategy
    type, instruments, timeframes, indicators, entry confirmations, exit
    rules, risk parameters, sessions, reference symbols. Missing dimensions
    are listed as 'not specified' so the user can correct them in one turn.
    """
    snapshot = _snapshot(builder, instructions, extra_prompt_text)
    text = _render_summary_text(snapshot)
    missing_critical = find_missing_critical_inputs(builder, instructions)
    return {
        "text": text,
        "snapshot": snapshot,
        "missing_critical": missing_critical,
    }


def _prompt_text_for_extraction(
    builder: Any,
    instructions: SemanticInstructions | None,
    extra_prompt_text: str | None,
) -> str:
    parts = [
        extra_prompt_text or "",
        (instructions.original_prompt if instructions else "") or "",
        getattr(builder, "goal", None) or "",
    ]
    return "\n".join(p for p in parts if p)


def _snapshot(
    builder: Any,
    instructions: SemanticInstructions | None,
    extra_prompt_text: str | None,
) -> dict[str, Any]:
    asset = ""
    if hasattr(builder, "format_symbol"):
        asset = builder.format_symbol() or ""
    asset = asset or getattr(builder, "symbol", "") or ""

    core: dict[str, Any] = {
        "asset":      asset,
        "timeframe":  getattr(builder, "timeframe", None),
        "objective":  getattr(builder, "objective", None),
        "sentiment":  getattr(builder, "sentiment", None),
        "experience": getattr(builder, "experience", None),
        "goal":       getattr(builder, "goal", None),
    }

    extracted: dict[str, Any] = {}
    if instructions is not None:
        extracted = {
            "strategy_family":      instructions.strategy_family,
            "indicators":           dict(instructions.indicators or {}),
            "htf_rules": [
                {
                    "timeframe":   r.timeframe,
                    "condition":   r.condition,
                    "role":        r.role,
                    "description": r.description,
                }
                for r in (instructions.htf_rules or [])
            ],
            "reference_symbols": [
                {
                    "reference_symbol": r.reference_symbol,
                    "relation":         r.relation,
                    "condition":        r.condition,
                }
                for r in (instructions.reference_symbols or [])
            ],
            "stop_loss":          _stop_loss_dict(instructions.stop_loss),
            "trailing_stop":      _trailing_stop_dict(instructions.trailing_stop),
            "risk_reward":        _risk_reward_dict(instructions.risk_reward),
            "session_filters":    _session_filters_dict(instructions.session_filters),
            "volume_confirmation":   _volume_dict(instructions.volume_momentum),
            "momentum_confirmation": _momentum_dict(instructions.volume_momentum),
            "candle_confirmation":   _candle_dict(instructions.candle_confirmation),
            "extraction_quality":    instructions.extraction_quality_score,
        }

    raw_prompt = _prompt_text_for_extraction(builder, instructions, extra_prompt_text)
    threshold_conditions = extract_threshold_conditions(raw_prompt)
    extracted["threshold_conditions"] = threshold_conditions

    return {
        "core":      core,
        "extracted": extracted,
        "raw_prompt": (extra_prompt_text or (instructions.original_prompt if instructions else None)),
    }


def _stop_loss_dict(sl: Any) -> dict[str, Any] | None:
    if sl is None:
        return None
    padding = getattr(sl, "padding", None)
    return {
        "type":     getattr(sl, "type", None),
        "anchor":   getattr(sl, "anchor", None),
        "padding": (
            {
                "method":       getattr(padding, "method", None),
                "atr_multiple": getattr(padding, "atr_multiple", None),
                "percent":      getattr(padding, "percent", None),
                "points":       getattr(padding, "points", None),
            }
            if padding else None
        ),
        "description": getattr(sl, "description", None),
    }


def _trailing_stop_dict(ts: Any) -> dict[str, Any] | None:
    if ts is None or not getattr(ts, "enabled", False):
        return None
    return {
        "type":               getattr(ts, "type", None),
        "ema_period":         getattr(ts, "ema_period", None),
        "atr_multiple":       getattr(ts, "atr_multiple", None),
        "activate_after_pct": getattr(ts, "activate_after_pct", None),
        "description":        getattr(ts, "description", None),
    }


def _risk_reward_dict(rr: Any) -> dict[str, Any] | None:
    if rr is None:
        return None
    return {
        "type":        getattr(rr, "type", None),
        "ratio":       getattr(rr, "ratio", None),
        "description": getattr(rr, "description", None),
    }


def _session_filters_dict(sf: Any) -> dict[str, Any] | None:
    if sf is None or not getattr(sf, "enabled", False):
        return None
    return {
        "session":          getattr(sf, "session", None),
        "valid_windows":    [_window_dict(w) for w in getattr(sf, "valid_windows", []) or []],
        "blackout_windows": [_window_dict(w) for w in getattr(sf, "blackout_windows", []) or []],
    }


def _window_dict(w: Any) -> dict[str, Any]:
    return {
        "start_time":       getattr(w, "start_time", None),
        "end_time":         getattr(w, "end_time", None),
        "duration_minutes": getattr(w, "duration_minutes", None),
        "from_open":        getattr(w, "from_open", None),
    }


def _volume_dict(vm: Any) -> dict[str, Any] | None:
    if vm is None or vm.volume is None:
        return None
    return {
        "filter_type":   vm.volume.filter_type,
        "sma_window":    getattr(vm.volume, "sma_window", None),
        "spike_multiple": getattr(vm.volume, "spike_multiple", None),
    }


def _momentum_dict(vm: Any) -> dict[str, Any] | None:
    if vm is None or vm.momentum is None:
        return None
    return {
        "filter_type":   vm.momentum.filter_type,
        "adx_threshold": getattr(vm.momentum, "adx_threshold", None),
    }


def _candle_dict(cc: Any) -> dict[str, Any] | None:
    if cc is None:
        return None
    return {
        "filter_type":  cc.filter_type,
        "description":  getattr(cc, "description", None),
    }


def _render_summary_text(snapshot: dict[str, Any]) -> str:
    core = snapshot.get("core", {})
    ex   = snapshot.get("extracted", {}) or {}

    asset = core.get("asset") or "this asset"
    timeframe = core.get("timeframe") or "not specified"

    lines: list[str] = [
        "Here is a summary of everything I captured from your strategy prompt. "
        "Please review carefully — I will not start building until you confirm.",
        "",
        "── Core inputs ──",
        f"Asset: {asset}",
        f"Timeframe: {timeframe}",
        f"Trade type / objective: {core.get('objective') or 'not specified'}",
        f"Market view / sentiment: {core.get('sentiment') or 'not specified'}",
        f"Trading experience: {core.get('experience') or 'not specified'}",
        f"Goal: {core.get('goal') or 'not specified'}",
    ]

    lines += ["", "── Strategy details extracted from your prompt ──"]
    lines.append(f"Strategy type: {ex.get('strategy_family') or 'not detected'}")

    indicators = ex.get("indicators") or {}
    if indicators:
        rendered = ", ".join(
            f"{name}({','.join(str(n) for n in windows)})" if windows else name
            for name, windows in indicators.items()
        )
        lines.append(f"Indicators referenced: {rendered}")
    else:
        lines.append("Indicators referenced: none mentioned")

    thresholds = ex.get("threshold_conditions") or []
    if thresholds:
        lines.append("Conditions you specified (will be enforced verbatim):")
        for t in thresholds:
            lines.append(f"  - {t['phrase']}  →  {t['expression']}")
    else:
        lines.append("Conditions you specified: none detected as numeric thresholds")

    htf_rules = ex.get("htf_rules") or []
    if htf_rules:
        for r in htf_rules:
            tf  = r.get("timeframe") or "?"
            cond = r.get("condition") or r.get("description") or ""
            lines.append(f"Higher-timeframe rule ({tf}): {cond}".rstrip())
    else:
        lines.append("Higher-timeframe rules: none mentioned")

    vol = ex.get("volume_confirmation")
    if vol:
        lines.append(f"Volume confirmation: {vol.get('filter_type')}")
    else:
        lines.append("Volume confirmation: not specified")

    mom = ex.get("momentum_confirmation")
    if mom:
        thr = mom.get("adx_threshold")
        if thr is not None:
            lines.append(f"Momentum confirmation: {mom.get('filter_type')} (ADX > {thr})")
        else:
            lines.append(f"Momentum confirmation: {mom.get('filter_type')}")
    else:
        lines.append("Momentum confirmation: not specified")

    candle = ex.get("candle_confirmation")
    if candle:
        lines.append(f"Candle confirmation: {candle.get('filter_type')}")
    else:
        lines.append("Candle confirmation: not specified")

    refs = ex.get("reference_symbols") or []
    if refs:
        for r in refs:
            sym = r.get("reference_symbol") or "?"
            rel = r.get("relation") or "reference"
            cond = r.get("condition") or ""
            lines.append(f"Reference symbol: {sym} ({rel}) {cond}".rstrip())
    else:
        lines.append("Reference symbols / relative strength: none mentioned")

    sl = ex.get("stop_loss")
    if sl:
        anchor = sl.get("anchor") or sl.get("type") or "structural"
        pad = sl.get("padding") or {}
        pad_repr = ""
        if pad and pad.get("method"):
            if pad.get("method") == "atr" and pad.get("atr_multiple"):
                pad_repr = f", padded by {pad['atr_multiple']} x ATR"
            elif pad.get("method") == "percent" and pad.get("percent"):
                pad_repr = f", padded by {pad['percent']}%"
            elif pad.get("method") == "points" and pad.get("points"):
                pad_repr = f", padded by {pad['points']} points"
        lines.append(f"Stop loss anchor: {anchor}{pad_repr}")
    else:
        lines.append("Stop loss anchor: not specified")

    ts = ex.get("trailing_stop")
    if ts:
        bits = [ts.get("type") or "trailing"]
        if ts.get("ema_period"):
            bits.append(f"EMA({ts['ema_period']})")
        if ts.get("atr_multiple"):
            bits.append(f"{ts['atr_multiple']} x ATR")
        if ts.get("activate_after_pct"):
            bits.append(f"activate after {ts['activate_after_pct']}%")
        lines.append(f"Trailing stop: {' / '.join(b for b in bits if b)}")
    else:
        lines.append("Trailing stop: not specified")

    rr = ex.get("risk_reward")
    if rr and rr.get("ratio"):
        kind = rr.get("type") or "fixed"
        lines.append(f"Risk-reward: 1:{rr['ratio']} ({kind})")
    else:
        lines.append("Risk-reward: not specified")

    sf = ex.get("session_filters")
    if sf:
        bits = []
        if sf.get("session"):
            bits.append(sf["session"])
        for win in (sf.get("valid_windows") or []):
            if win.get("start_time"):
                bits.append(f"from {win['start_time']}")
            if win.get("end_time"):
                bits.append(f"until {win['end_time']}")
            if win.get("duration_minutes") and win.get("from_open"):
                bits.append(f"first {win['duration_minutes']}min of session")
        for win in (sf.get("blackout_windows") or []):
            if win.get("duration_minutes") and win.get("from_open"):
                bits.append(f"avoid first {win['duration_minutes']}min")
        lines.append(f"Session / time-window filters: {', '.join(bits) if bits else 'enabled'}")
    else:
        lines.append("Session / time-window filters: none mentioned")

    lines += [
        "",
        "Reply 'confirm' to proceed with building the entry and exit signals. "
        "If anything above is missing or wrong, tell me what to add or change "
        "and I will update the summary before building.",
    ]
    return "\n".join(lines)


# ── 2. Missing-critical-input detection ──────────────────────────────────────


_EXIT_HINT_PATTERNS = [
    r"\bstop[- ]?loss\b",
    r"\bsl\b",
    r"\btarget\b",
    r"\btake[- ]?profit\b",
    r"\btp\b",
    r"\bexit\b",
    r"\btrail(?:ing)?\b",
    r"\brisk[: ]+reward\b",
    r"\brr\b",
]


def find_missing_critical_inputs(
    builder: Any,
    instructions: SemanticInstructions | None,
) -> list[dict[str, str]]:
    """Return the list of exit-side details that the user did not specify in
    the prompt. Critical inputs are those the system refuses to default —
    today that is the exit-side block (stop loss + take profit / exit
    condition). If both are absent, the flow must ask the user.
    """
    missing: list[dict[str, str]] = []

    has_sl_from_extraction  = bool(instructions and instructions.stop_loss is not None)
    has_rr_from_extraction  = bool(
        instructions
        and instructions.risk_reward is not None
        and getattr(instructions.risk_reward, "ratio", None)
    )
    has_trailing_stop = bool(
        instructions
        and instructions.trailing_stop is not None
        and getattr(instructions.trailing_stop, "enabled", False)
    )

    # Free-text hint check: did the goal / prompt at least mention an exit
    # concept? If yes, the user has *some* exit idea even if the regex
    # extractor didn't structure it, so don't block.
    haystack = " ".join(
        str(p) for p in [
            getattr(builder, "goal", "") or "",
            (instructions.original_prompt if instructions else "") or "",
        ]
    ).lower()
    text_mentions_exit = any(re.search(p, haystack) for p in _EXIT_HINT_PATTERNS)

    has_any_exit_signal = (
        has_sl_from_extraction
        or has_rr_from_extraction
        or has_trailing_stop
        or text_mentions_exit
        or getattr(builder, "stop_loss", None) is not None
        or getattr(builder, "take_profit", None) is not None
        or bool(getattr(builder, "exit_condition", None))
    )

    if not has_any_exit_signal:
        missing.append({
            "field": "exit_block",
            "label": "exit rules (stop loss and target / exit condition)",
            "question": (
                "What should trigger the exit? For example: a percentage stop loss "
                "and target, a swing-low stop with a 1:2 risk-reward target, an EMA "
                "trailing exit, or an opposite signal."
            ),
        })

    return missing


# ── 3. Zero-loss capture enforcement ──────────────────────────────────────────


def enforce_zero_loss_capture(
    plan: dict[str, Any],
    instructions: SemanticInstructions | None,
    *,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    """Mutate `plan` so its entry_condition / exit_condition strings include
    every condition the user mentioned. Conditions that don't fit neatly as
    AND-clauses are attached to the plan's underscore-prefixed slots
    (htf_rules, stop_loss_spec, trailing_stop_spec, reference_symbol, …) so
    the assembler and engine still see them.

    Returns a dict describing what was added — caller logs it as a
    'condition_augmented' event.
    """
    augmented: dict[str, Any] = {"entry_added": [], "exit_added": [], "side_channels": []}
    if not isinstance(plan, dict) or instructions is None:
        return augmented

    entry_condition = str(plan.get("entry_condition") or "").strip()
    exit_condition  = str(plan.get("exit_condition")  or "").strip()

    new_entry_clauses: list[str] = []
    new_exit_clauses: list[str] = []

    # ── Threshold conditions from the prompt (e.g. "RSI between 40-65") ──
    # These are the user's verbatim comparisons — top priority for Feature
    # 2's zero-loss guarantee. Each one is appended as a real AND-clause.
    raw_prompt = prompt_text or (instructions.original_prompt if instructions else "")
    threshold_conditions = extract_threshold_conditions(raw_prompt or "")
    haystack = entry_condition + " " + exit_condition
    for cond in threshold_conditions:
        expr = cond["expression"]
        if expr in haystack:
            continue
        new_entry_clauses.append(expr)
        augmented["entry_added"].append(expr)
        haystack += " " + expr

    # ── Indicators the user mentioned that didn't show up at all ─────────
    # These are recorded in plan["_required_indicators"] so the engine
    # computes them and surfaces them in the strategy YAML — without
    # adding a tautological clause to the condition string.
    required_indicators: dict[str, list[int]] = dict(plan.get("_required_indicators") or {})
    for name, windows in (instructions.indicators or {}).items():
        upper = name.upper()
        if windows:
            for window in windows:
                token = _INDICATOR_DEFAULT_REPRS.get(upper, f"{upper}({{n}})").format(n=window)
                if _condition_contains_token(haystack, token):
                    continue
                required_indicators.setdefault(upper, [])
                if window not in required_indicators[upper]:
                    required_indicators[upper].append(window)
                    augmented["side_channels"].append({"required_indicator": token})
        else:
            if not _condition_contains_token(haystack, upper):
                required_indicators.setdefault(upper, [])
                augmented["side_channels"].append({"required_indicator": upper})
    if required_indicators:
        plan["_required_indicators"] = required_indicators

    # ── Volume confirmation ───────────────────────────────────────────────
    if instructions.volume_momentum and instructions.volume_momentum.volume:
        vol_window = getattr(instructions.volume_momentum.volume, "sma_window", None) or 20
        clause = f"VOLUME > VOLUME_SMA({vol_window})"
        if clause not in haystack and "VOLUME_SMA" not in haystack:
            new_entry_clauses.append(clause)
            augmented["entry_added"].append(clause)
            haystack += " " + clause

    # ── Momentum (ADX threshold) ──────────────────────────────────────────
    # Skip when a threshold-extracted ADX clause is already in haystack so
    # we don't duplicate the user's "ADX above 25" condition.
    if instructions.volume_momentum and instructions.volume_momentum.momentum:
        mom = instructions.volume_momentum.momentum
        threshold = getattr(mom, "adx_threshold", None)
        if threshold is not None:
            clause = f"ADX(14) > {threshold}"
            if not _condition_contains_token(haystack, "ADX("):
                new_entry_clauses.append(clause)
                augmented["entry_added"].append(clause)
                haystack += " " + clause
        elif mom.filter_type == "adx_strong" and not _condition_contains_token(haystack, "ADX("):
            new_entry_clauses.append("ADX(14) > 25")
            augmented["entry_added"].append("ADX(14) > 25")
            haystack += " ADX(14) > 25"

    # ── Candle confirmation ───────────────────────────────────────────────
    if instructions.candle_confirmation:
        kind = instructions.candle_confirmation.filter_type
        if kind == "bullish_confirmation" and "CLOSE > OPEN" not in haystack:
            new_entry_clauses.append("CLOSE > OPEN")
            augmented["entry_added"].append("CLOSE > OPEN")
            haystack += " CLOSE > OPEN"
        elif kind == "bearish_confirmation" and "CLOSE < OPEN" not in haystack:
            new_entry_clauses.append("CLOSE < OPEN")
            augmented["entry_added"].append("CLOSE < OPEN")
            haystack += " CLOSE < OPEN"

    # ── Risk:Reward → exit-side target enforcement ────────────────────────
    if instructions.risk_reward and getattr(instructions.risk_reward, "ratio", None):
        ratio = float(instructions.risk_reward.ratio)
        plan.setdefault("_rr_ratio", ratio)
        clause = f"PROFIT >= STOP_LOSS_TARGET * {ratio}"
        if "PROFIT" not in exit_condition:
            new_exit_clauses.append(clause)
            augmented["exit_added"].append(clause)
        augmented["side_channels"].append({"rr_ratio": ratio})

    # ── Side-channel specs (engine consumes these directly) ───────────────
    if instructions.stop_loss is not None and not plan.get("_stop_loss_spec"):
        plan["_stop_loss_spec"] = {
            "type":   instructions.stop_loss.type,
            "anchor": instructions.stop_loss.anchor,
            "padding": _stop_loss_dict(instructions.stop_loss).get("padding"),
        }
        augmented["side_channels"].append({"stop_loss_spec": plan["_stop_loss_spec"]})

    if instructions.trailing_stop and getattr(instructions.trailing_stop, "enabled", False):
        if not plan.get("_trailing_stop_spec"):
            ts = instructions.trailing_stop
            plan["_trailing_stop_spec"] = {
                "type":               ts.type,
                "ema_period":         getattr(ts, "ema_period", None),
                "atr_multiple":       getattr(ts, "atr_multiple", None),
                "activate_after_pct": getattr(ts, "activate_after_pct", None),
            }
            augmented["side_channels"].append({"trailing_stop_spec": plan["_trailing_stop_spec"]})

    if instructions.htf_rules:
        existing = list(plan.get("_htf_rules") or [])
        existing_keys = {(str(r.get("timeframe")), str(r.get("condition"))) for r in existing if isinstance(r, dict)}
        for rule in instructions.htf_rules:
            key = (str(rule.timeframe), str(rule.condition or ""))
            if rule.condition and key not in existing_keys:
                existing.append({"timeframe": rule.timeframe, "condition": rule.condition})
                existing_keys.add(key)
                augmented["side_channels"].append({"htf_rule": {"timeframe": rule.timeframe, "condition": rule.condition}})
        if existing:
            plan["_htf_rules"] = existing

    if instructions.reference_symbols and not plan.get("_reference_symbol"):
        first = instructions.reference_symbols[0]
        if getattr(first, "reference_symbol", None):
            plan["_reference_symbol"] = first.reference_symbol
            augmented["side_channels"].append({"reference_symbol": first.reference_symbol})

    if instructions.session_filters and getattr(instructions.session_filters, "enabled", False):
        if not plan.get("_session_filters"):
            plan["_session_filters"] = _session_filters_dict(instructions.session_filters)
            augmented["side_channels"].append({"session_filters": plan["_session_filters"]})

    # ── Stitch new clauses back into the condition strings ────────────────
    if new_entry_clauses:
        plan["entry_condition"] = _join_clauses(entry_condition, new_entry_clauses)
    if new_exit_clauses:
        plan["exit_condition"] = _join_clauses(exit_condition, new_exit_clauses)

    if augmented["entry_added"] or augmented["exit_added"] or augmented["side_channels"]:
        logger.info(
            "prompt_summary|event=condition_augmented|entry_added=%s|exit_added=%s|side=%s",
            augmented["entry_added"],
            augmented["exit_added"],
            augmented["side_channels"],
        )

    return augmented


def _condition_contains_token(haystack: str, token: str) -> bool:
    if not haystack or not token:
        return False
    return token.upper() in haystack.upper()


def _join_clauses(existing: str, new_clauses: list[str]) -> str:
    new_clauses = [c for c in (clause.strip() for clause in new_clauses) if c]
    if not new_clauses:
        return existing
    joined = " AND ".join(new_clauses)
    if not existing:
        return joined
    # Avoid double-parenthesising obviously simple expressions.
    if " AND " in existing or " OR " in existing:
        return f"({existing}) AND {joined}"
    return f"{existing} AND {joined}"


__all__ = [
    "build_prompt_summary",
    "find_missing_critical_inputs",
    "enforce_zero_loss_capture",
]
