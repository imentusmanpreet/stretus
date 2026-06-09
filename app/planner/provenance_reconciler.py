"""
app/planner/provenance_reconciler.py — Deterministic provenance derivation.

THE PROBLEM
──────────────────────────────────────────────────────────────────────────────
The SDL selector asks the LLM to self-report which fields came from the user
(field_sources source="user"). LLMs do this inconsistently:
  • They use coarse paths ("legs.0.entry.trigger") and miss granular ones
    ("legs.0.entry.trigger.params.threshold").
  • They silently default money fields without adding clarifications.
  • They omit paths entirely for clearly stated values (timeframe, direction).

THE FIX
──────────────────────────────────────────────────────────────────────────────
After the LLM fills in an SDL, run this reconciler:
  1. Parse the raw prompt with pattern matching (no LLM).
  2. For each detected element, check whether the corresponding SDL field is
     actually populated in the LLM's response.
  3. If it is, mark it "user" in field_sources (overriding a missing/wrong value).
  4. If money fields (SL, TP, RR) are populated but the user never said them,
     add a clarification question and mark them "default".

This is the sole authoritative source of "user" provenance.  The LLM's own
field_sources output is used as a SUPPLEMENT (it may add things we miss) but
is never trusted to be complete on its own.

IMPORTANT: this module mainly adds / upgrades provenance entries and never
removes a "user" tag the LLM set.  The one exception is value backfill for a
clearly-stated numeric the LLM dropped — currently the ATR stop multiple
("1 ATR" → multiple=1.0) — because the compiler would otherwise silently
default it (to 1.5×) and misrepresent the user's risk.
"""
from __future__ import annotations

import re
from typing import Any

# ── Provenance source ranking (upgrade-only) ──────────────────────────────────
# default < inferred < user. A field's source may only ever be PROMOTED across
# turns — never demoted. The classic bug this kills: a modify turn that doesn't
# restate an earlier user value would otherwise drop its provenance back to
# default/inferred (and the gate would then "clarify" something the user already
# told us). carry_forward + merge_field_sources enforce monotonic promotion.
_SOURCE_RANK: dict[str, int] = {"default": 0, "inferred": 1, "user": 2}


def merge_field_sources(
    prior: dict[str, str] | None, new: dict[str, str] | None
) -> dict[str, str]:
    """Union of two field_sources maps, keeping the STRONGER source per path.

    Never demotes: a path that was "user" in `prior` stays "user" even if `new`
    reports it as "inferred"/"default" (or omits it).
    """
    out: dict[str, str] = dict(prior or {})
    for path, src in (new or {}).items():
        cur = out.get(path)
        if cur is None or _SOURCE_RANK.get(src, 0) > _SOURCE_RANK.get(cur, 0):
            out[path] = src
    return out


# ── Pattern library ───────────────────────────────────────────────────────────

_TF_PAT = re.compile(
    r"\b(\d+)\s*(m(?:in(?:ute)?s?)?|h(?:our)?s?|d(?:ay)?s?|w(?:eek)?s?)\b",
    re.IGNORECASE,
)
_TF_WORDS = re.compile(
    r"\b(daily|weekly|hourly|intraday|scalp(?:ing)?|swing)\b", re.IGNORECASE,
)
_DIR_LONG  = re.compile(r"\b(?:buy|long|go long|long only)\b", re.IGNORECASE)
_DIR_SHORT = re.compile(r"\b(?:sell|short|go short|short only|short sell)\b", re.IGNORECASE)

# Symbol patterns: equity (word.NS, word.BO) or crypto (ETH, BTC, SOL…)
_EQUITY_SYM_PAT = re.compile(r"\b([A-Z]{2,12})(?:\.NS|\.BO)?\b")
_CRYPTO_SYM_PAT = re.compile(
    r"\b(ETH|BTC|SOL|ADA|MATIC|AVAX|ATOM|LINK|DOT|XRP|DOGE|AAVE|UNI|CRV|LTC|NEAR|FIL|SAND|MANA)\b",
    re.IGNORECASE,
)

# Risk patterns
_SL_PCT_PAT  = re.compile(r"\bstop[\s\-]*(?:loss)?\s*(?:at\s*)?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_SL_ATR_PAT  = re.compile(r"\b(?:atr|average\s+true\s+range)\s*stop", re.IGNORECASE)
# ATR multiple: a number IMMEDIATELY BEFORE "ATR" is the multiplier — "1 ATR",
# "1.5x ATR", "2 * atr", "1.5 average true range". (A number AFTER ATR, e.g.
# "ATR 14" or "ATR(14)", is the lookback window, not the multiple — so we only
# capture the leading number here.) General across phrasings; not prompt-specific.
_ATR_MULT_PAT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:x|\*|times)?\s*(?:atr|average\s+true\s+range)\b",
    re.IGNORECASE,
)
_TP_PCT_PAT  = re.compile(r"\btake[\s\-]*profit\s*(?:at\s*)?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_RR_PAT      = re.compile(r"\b(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*(?:r(?:isk)?(?:\s*[\-:]\s*)?r(?:eward)?)?", re.IGNORECASE)
_RR_WORD_PAT = re.compile(r"\b(?:rr|risk[\s\-]*reward|reward[\s\-]*ratio)\b", re.IGNORECASE)
_TARGET_PAT  = re.compile(r"\btarget\s*(?:at\s*)?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_TRAIL_PAT   = re.compile(r"\btrail(?:ing)?\s*stop\b", re.IGNORECASE)

# Gate patterns
_REGIME_PAT = re.compile(
    r"\b(?:trending|only\s+(?:in\s+)?trend(?:ing)?|regime|market\s+regime|avoid\s+(?:ranging|sideways))\b",
    re.IGNORECASE,
)
_EVENT_PAT  = re.compile(
    r"\b(?:earnings|expiry|skip\s+dates?|event\s+(?:dates?|filter)|avoid\s+(?:dates?|days?))\b",
    re.IGNORECASE,
)
_SESSION_PAT = re.compile(
    r"\b(?:session|entry\s+window|between\s+\d{1,2}:\d{2}|after\s+\d{1,2}:\d{2}|before\s+market\s+close)\b",
    re.IGNORECASE,
)
_VOL_GATE_PAT = re.compile(
    r"\b(?:volatility\s+(?:filter|gate|band)|only\s+(?:if|when)\s+(?:vol|volatility|atr)\s+(?:is\s+)?(?:above|below|between)|vol\s+(?:filter|gate))\b",
    re.IGNORECASE,
)
_RS_GATE_PAT = re.compile(
    r"\b(?:relative\s+strength|outperform(?:ing)?|stronger\s+than|rs\s*(?:filter|gate|>|>=))\b",
    re.IGNORECASE,
)
_VOL_RATIO_PAT = re.compile(
    r"\b(?:volume\s+(?:ratio|spike|filter|confirmation)|above[\s\-]+average\s+volume|vol(?:ume)?\s+(?:>|>=|above)\s+\d)\b",
    re.IGNORECASE,
)

# HTF patterns
_HTF_PAT = re.compile(
    r"\b(?:higher[\s\-]*timeframe|htf|(?:above|below)\s+(?:the\s+)?\d+[mh]\s+(?:ema|sma)|daily\s+trend|1h\s+trend)\b",
    re.IGNORECASE,
)

# Indicator-family → catalog field patterns
# Each tuple: (family_id, detection_regex, param_patterns)
# Threshold pattern for RSI: "RSI 14 below 30" or "RSI below 30" or "RSI(14) < 30"
_RSI_THRESHOLD_PAT = re.compile(
    r"\brsi[\s(]?\s*\d*[\s)]*\s*(?:below|under|drops?\s+below|<|<=|above|over|crosses?\s+above|>|>=)\s*(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

# SL abbreviation patterns (in addition to the full "stop loss")
_SL_ABBREV_PAT = re.compile(
    r"\bsl\s*(?:at\s*)?(\d+(?:\.\d+)?)\s*%|\bstop\s*(?:loss)?\s*(?:at\s*)?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _trigger_matches_family(trigger_name: str, family: str) -> bool:
    """Check if the card's trigger name belongs to the indicator family."""
    tn = trigger_name.lower()
    fam = family.lower()

    # Direct family substring
    if fam in tn:
        return True

    # Special multi-word families
    if family == "ORB" and ("opening" in tn or "orb" in tn):
        return True
    if family == "BB" and ("bb_" in tn or "bollinger" in tn or "band" in tn):
        return True
    if family == "MACD" and "macd" in tn:
        return True
    if family == "SUPERTREND" and "supertrend" in tn:
        return True
    if family == "STOCH" and "stoch" in tn:
        return True

    # Single underscore-free part matching (e.g. "EMA" → "ema_cross_up")
    for part in fam.split("_"):
        if len(part) >= 3 and part in tn:
            return True

    return False


_INDICATOR_FAMILIES: list[tuple[str, re.Pattern, dict[str, re.Pattern]]] = [
    ("RSI", re.compile(r"\brsi\b", re.IGNORECASE), {
        # "RSI 14" or "RSI(14)"
        "window":    re.compile(r"\brsi[\s(]?\s*(\d+)\b", re.IGNORECASE),
        # "RSI [14] below 30" or "RSI below 30" or "RSI < 30"
        "threshold": _RSI_THRESHOLD_PAT,
    }),
    ("EMA", re.compile(r"\bema\b|\bexponential\s+(?:moving\s+)?average\b", re.IGNORECASE), {
        "window_fast": re.compile(r"\bema[\s(]?\s*(\d+)\b", re.IGNORECASE),
    }),
    ("SMA", re.compile(r"\bsma\b|\bsimple\s+(?:moving\s+)?average\b", re.IGNORECASE), {
        "window": re.compile(r"\bsma[\s(]?\s*(\d+)\b", re.IGNORECASE),
    }),
    ("MACD", re.compile(r"\bmacd\b", re.IGNORECASE), {}),
    ("BB", re.compile(r"\bbollinger\b|\bbb\s+bands?\b", re.IGNORECASE), {
        "window":  re.compile(r"\bbollinger[\s(]?\s*(\d+)\b", re.IGNORECASE),
        "num_std": re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:standard\s+deviations?|std)\b", re.IGNORECASE),
    }),
    ("ATR", re.compile(r"\batr\b|\baverage\s+true\s+range\b", re.IGNORECASE), {
        "window": re.compile(r"\batr[\s(]?\s*(\d+)\b", re.IGNORECASE),
    }),
    ("ADX", re.compile(r"\badx\b", re.IGNORECASE), {
        "window":    re.compile(r"\badx[\s(]?\s*(\d+)\b", re.IGNORECASE),
        "threshold": re.compile(r"\badx\s*(?:above|>|>=)\s*(\d+)\b", re.IGNORECASE),
    }),
    ("VWAP", re.compile(r"\bvwap\b", re.IGNORECASE), {}),
    ("STOCH", re.compile(r"\bstoch(?:astic)?\b", re.IGNORECASE), {}),
    ("CCI",   re.compile(r"\bcci\b", re.IGNORECASE), {}),
    ("MFI",   re.compile(r"\bmfi\b|\bmoney\s+flow\b", re.IGNORECASE), {}),
    ("OBV",   re.compile(r"\bobv\b|\bon[\s\-]*balance\s+volume\b", re.IGNORECASE), {}),
    ("ORB",   re.compile(r"\bopening[\s\-]*range\b|\borb\b", re.IGNORECASE), {
        "minutes": re.compile(r"\bopening[\s\-]*range[\s\-]*(\d+)[\s\-]*(?:minutes?|m)\b", re.IGNORECASE),
    }),
    ("DONCHIAN",   re.compile(r"\bdonchian\b", re.IGNORECASE), {}),
    ("KELTNER",    re.compile(r"\bkeltner\b", re.IGNORECASE), {}),
    ("SUPERTREND", re.compile(r"\bsuper\s*trend\b", re.IGNORECASE), {}),
]


# ── Main reconciler ───────────────────────────────────────────────────────────

def reconcile(sdl, prompt: str, prior_field_sources: dict[str, str] | None = None) -> "SDL":
    """Return a new SDL with provenance.field_sources fully reconciled.

    The reconciler:
    1. Detects what the user explicitly stated in the prompt.
    2. For each detected item, if the SDL has a corresponding non-empty field,
       marks it "user" in field_sources (never downgrades an existing "user").
    3. Detects defaulted money fields (SL/TP/RR) with no user input and adds
       clarification_needed entries if missing.

    `prior_field_sources` carries the PREVIOUS turn's provenance into a modify
    turn. It is the floor: a path the user established earlier stays at its
    earlier (stronger) source even if this turn's message doesn't restate it.
    Promotion is monotonic (default→inferred→user); demotion never happens.

    Returns a new SDL (the input SDL is not mutated).
    """
    # Work on mutable copies
    data = sdl.model_dump(mode="python")
    data.pop("content_hash", None)

    prov: dict = data.get("provenance") or {}
    # Seed with the prior turn's provenance so unmentioned-but-previously-user
    # fields survive; upgrade-only merge guarantees no demotion.
    fs: dict[str, str] = merge_field_sources(prior_field_sources, prov.get("field_sources"))
    clarifs: list[dict] = list(prov.get("clarifications_needed") or [])
    unmapped: list[dict] = list(prov.get("unmapped_details") or [])

    prompt_l = str(prompt or "").lower()

    # ── 1. Timeframe ──────────────────────────────────────────────────────────
    if _TF_PAT.search(prompt) or _TF_WORDS.search(prompt):
        _mark(fs, "context.timeframe", "user")

    # ── 2. Symbol / universe ──────────────────────────────────────────────────
    uni = data.get("universe") or {}
    uni_type = uni.get("type", "static")
    if uni_type == "static":
        if _CRYPTO_SYM_PAT.search(prompt) or _EQUITY_SYM_PAT.search(prompt):
            _mark(fs, "universe.symbol", "user")
            _mark(fs, "universe.asset_class", "user")
    else:
        _mark(fs, "universe", "user")
        if uni.get("screen"):
            _mark(fs, "universe.screen", "user")
        if uni.get("rank"):
            _mark(fs, "universe.rank.by", "user")
        if uni.get("tie_break"):
            _mark(fs, "universe.tie_break", "user")

    # ── 3. Direction ──────────────────────────────────────────────────────────
    for leg_i, leg in enumerate(data.get("legs") or []):
        d = leg.get("direction", "")
        if d == "long" and _DIR_LONG.search(prompt):
            _mark(fs, f"legs.{leg_i}.direction", "user")
        elif d == "short" and _DIR_SHORT.search(prompt):
            _mark(fs, f"legs.{leg_i}.direction", "user")

    # ── 4. Indicators (entry trigger + filters) ───────────────────────────────
    for leg_i, leg in enumerate(data.get("legs") or []):
        entry = leg.get("entry") or {}
        trigger = entry.get("trigger") or {}
        trigger_name = (trigger.get("name") or "").lower()

        # Detect which indicator family the user mentioned
        for family, fam_pat, param_pats in _INDICATOR_FAMILIES:
            if not fam_pat.search(prompt):
                continue
            # This family was mentioned — check if the trigger card uses it
            if _trigger_matches_family(trigger_name, family):
                _mark(fs, f"legs.{leg_i}.entry.trigger.name", "user")
                _mark(fs, f"legs.{leg_i}.entry.trigger", "user")
                # Mark specific params that were explicitly stated in the prompt
                trigger_params = trigger.get("params") or {}
                for pname, ppat in param_pats.items():
                    m = ppat.search(prompt)
                    if m and pname in trigger_params:
                        _mark(fs, f"legs.{leg_i}.entry.trigger.params.{pname}", "user")

        # Filters: check if any filter uses a mentioned indicator
        for fi, flt in enumerate(entry.get("filters") or []):
            flt_name = (flt.get("name") or "").lower()
            for family, fam_pat, _ in _INDICATOR_FAMILIES:
                if fam_pat.search(prompt) and family.lower() in flt_name:
                    _mark(fs, f"legs.{leg_i}.entry.filters.{fi}", "user")

        # Exit triggers
        for ti, trig in enumerate(leg.get("exit", {}).get("triggers") or []):
            trig_name = (trig.get("name") or "").lower()
            # If user explicitly mentioned the exit (e.g. "exit on RSI overbought")
            exit_explicit = re.search(
                r"\b(?:exit|close\s+position|take\s+profit|sell)\s+(?:when|on|at|if)\b",
                prompt, re.IGNORECASE,
            )
            if exit_explicit:
                _mark(fs, f"legs.{leg_i}.exit.triggers.{ti}", "user")

    # ── 5. Stop loss ──────────────────────────────────────────────────────────
    risk = data.get("risk") or {}
    sl = risk.get("stop_loss")
    if sl:
        sl_pct_stated = _SL_PCT_PAT.search(prompt) or _SL_ABBREV_PAT.search(prompt)
        if sl_pct_stated and sl.get("type") == "percent":
            _mark(fs, "risk.stop_loss", "user")
            _mark(fs, "risk.stop_loss.type", "user")
            _mark(fs, "risk.stop_loss.value", "user")
        elif sl.get("type") == "atr" and (_SL_ATR_PAT.search(prompt) or _ATR_MULT_PAT.search(prompt)):
            _mark(fs, "risk.stop_loss", "user")
            _mark(fs, "risk.stop_loss.type", "user")
            # Capture the ATR multiple the user actually stated ("1 ATR" → 1.0,
            # "1.5x ATR" → 1.5). LLMs routinely emit type=atr but drop the
            # number, which the compiler then silently defaults to 1.5×. The
            # prompt is authoritative, so backfill the stated value here.
            m = _ATR_MULT_PAT.search(prompt)
            if m:
                sl["multiple"] = float(m.group(1))
                _mark(fs, "risk.stop_loss.multiple", "user")
        elif re.search(r"\b(?:stop|sl)\b", prompt, re.IGNORECASE) and sl.get("type"):
            # generic stop mention ("stop at swing low", "structural stop")
            _mark(fs, "risk.stop_loss", "user")
        else:
            # Stop was defaulted but not stated → clarification needed
            _ensure_clarification(clarifs, "risk.stop_loss",
                                  "No stop loss specified — what stop loss should I use? (e.g. 2% or ATR×1.5)",
                                  sl.get("value") or sl.get("multiple"))
            fs.setdefault("risk.stop_loss", "default")

    # ── 6. Take profit / RR ───────────────────────────────────────────────────
    tp = risk.get("take_profit")
    if tp:
        # Accept "2:1 RR", "RR 2:1", "2:1", "risk reward 2:1", "TP 4%", "target 4%"
        user_stated_tp = (
            _TP_PCT_PAT.search(prompt)
            or _TARGET_PAT.search(prompt)
            or _RR_WORD_PAT.search(prompt)   # "rr", "risk reward", etc. alone is enough
            or _RR_PAT.search(prompt)         # any "N:M" near-RR pattern
        )
        if user_stated_tp:
            _mark(fs, "risk.take_profit", "user")
            _mark(fs, "risk.take_profit.type", "user")
            if tp.get("ratio"):
                _mark(fs, "risk.take_profit.ratio", "user")
            if tp.get("value"):
                _mark(fs, "risk.take_profit.value", "user")
        else:
            # TP was defaulted — add clarification if not already there
            _ensure_clarification(clarifs, "risk.take_profit",
                                  "No take-profit specified — I used 2:1 RR. OK?",
                                  tp.get("ratio") or tp.get("value"))
            fs.setdefault("risk.take_profit", "default")

    # ── 7. Trailing stop ──────────────────────────────────────────────────────
    trailing = risk.get("trailing")
    if trailing and trailing.get("enabled") and _TRAIL_PAT.search(prompt):
        _mark(fs, "risk.trailing", "user")

    # ── 8. Gates ─────────────────────────────────────────────────────────────
    gates = data.get("gates") or {}
    if gates.get("regime") and _REGIME_PAT.search(prompt):
        _mark(fs, "gates.regime", "user")
    if gates.get("event") and _EVENT_PAT.search(prompt):
        _mark(fs, "gates.event", "user")
    if gates.get("session") and _SESSION_PAT.search(prompt):
        _mark(fs, "gates.session", "user")
    if gates.get("volatility") and _VOL_GATE_PAT.search(prompt):
        _mark(fs, "gates.volatility", "user")
    if gates.get("relative_strength") and _RS_GATE_PAT.search(prompt):
        _mark(fs, "gates.relative_strength", "user")
    if gates.get("volume_ratio") and _VOL_RATIO_PAT.search(prompt):
        _mark(fs, "gates.volume_ratio", "user")

    # ── 9. HTF rules ──────────────────────────────────────────────────────────
    if data.get("htf_rules") and _HTF_PAT.search(prompt):
        for ri in range(len(data["htf_rules"])):
            _mark(fs, f"htf_rules.{ri}", "user")

    # ── 10. Objective (context) ───────────────────────────────────────────────
    obj_words = re.compile(
        r"\b(?:breakout|scalp(?:ing)?|mean[\s\-]*reversion|momentum|trend\s+follow(?:ing)?|swing|positional|intraday|day\s+trad(?:e|ing))\b",
        re.IGNORECASE,
    )
    if obj_words.search(prompt):
        _mark(fs, "context.objective", "user")

    # ── Write back ────────────────────────────────────────────────────────────
    prov["field_sources"] = fs
    prov["clarifications_needed"] = _dedup_clarifications(clarifs)
    prov["unmapped_details"] = unmapped
    data["provenance"] = prov

    from app.planner.sdl import SDL
    return SDL(**data)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mark(fs: dict, path: str, source: str) -> None:
    """Promote path → source (upgrade-only; never downgrades a stronger source)."""
    cur = fs.get(path)
    if cur is None or _SOURCE_RANK.get(source, 0) > _SOURCE_RANK.get(cur, 0):
        fs[path] = source



def _ensure_clarification(
    clarifs: list[dict], field: str, question: str, assumed_value: Any
) -> None:
    """Add a clarification entry if one for this field doesn't exist yet."""
    if any(c.get("field") == field for c in clarifs):
        return
    clarifs.append({
        "field": field,
        "question": question,
        "assumed_value": assumed_value,
    })


def _dedup_clarifications(clarifs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in clarifs:
        k = c.get("field", "")
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out
