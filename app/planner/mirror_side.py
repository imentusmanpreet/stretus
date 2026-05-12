"""
app/planner/mirror_side.py — Auto-generate the opposite-direction (SELL/Short)
setup for a strategy whose primary direction is BUY/Long (or vice versa).

The rule from Phase 3 spec #8 — "for every BUY setup you describe, also
define the mirror SELL setup" — means every entry condition, exit
condition, stop loss anchor, and target rule needs a flipped twin. This
module produces that twin from the primary plan + the catalog metadata
that says how each indicator inverts.

Mirror policies per indicator class:

  • directional       — Flip the comparison operator
                        (CLOSE > EMA(20)  ↔  CLOSE < EMA(20))
  • bounded_0_100     — Reflect the threshold around 50
                        (RSI > 65 ↔ RSI < 35;  STOCH_K < 80 ↔ STOCH_K > 20)
  • bounded_neg100_100 — Reflect the threshold around 0
                        (CCI > 100 ↔ CCI < -100;  WILLR < -20 ↔ WILLR > -80)
  • direction_agnostic — Stay the same
                        (ADX, ATR, CHOPPINESS, VOLUME, …)

mirror_plan() consumes a "primary" plan dict (same shape as
plan_signals_v2's output) and returns a "mirror" plan dict with flipped
entry_condition, exit_condition, side-channel SL/trailing specs, and a
direction tag. The chat layer surfaces both in the summary so the user
can see the opposite-side setup before committing.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Per-indicator mirror policies ────────────────────────────────────────────


MirrorPolicy = str  # "directional" | "bounded_0_100" | "bounded_neg100_100" | "direction_agnostic"


_DIRECTION_AGNOSTIC = {
    "ADX", "ATR", "CHOPPINESS", "HV", "STDEV",
    "VOLUME", "VOLUME_SMA", "BB_WIDTH", "VROC", "VROC_12",
    "TRUE_RANGE", "HIGH_LOW", "DI_PLUS", "DI_MINUS",
    "CHAIKIN_VOL", "ULCER",
}

_BOUNDED_0_100 = {
    "RSI", "STOCH_K", "STOCH_D", "MFI", "BB_PCT_B",
    "AROON_UP", "AROON_DOWN", "AROON_OSC", "STC",
    "IMI", "RVI",
}

_BOUNDED_NEG100_100 = {
    "CCI", "CMO", "WILLR", "ROC", "MOMENTUM", "PMO",
    "CMF", "TRIX", "DISPARITY", "AO",
}

_DIRECTIONAL_DEFAULT = {
    # Everything that isn't in the lists above gets flipped operator-wise.
    "CLOSE", "OPEN", "HIGH", "LOW",
    "EMA", "SMA", "VWAP", "MACD", "MACD_SIGNAL", "MACD_HIST",
    "OBV", "ACCDIST",
    "BB_UPPER", "BB_LOWER", "BB_MID",
    "DON_UPPER", "DON_LOWER", "DON_MID",
    "KC_UPPER", "KC_LOWER", "KC_MID",
    "ATR_UPPER", "ATR_LOWER",
    "SUPERTREND", "SUPERTREND_DIR", "PSAR",
    "HHV", "LLV", "PIVOT", "R1", "R2", "S1", "S2",
    "MEDIAN_PRICE", "TYPICAL_PRICE", "WEIGHTED_CLOSE",
}


def _indicator_policy(name: str) -> MirrorPolicy:
    upper = name.upper().strip()
    if upper in _DIRECTION_AGNOSTIC:
        return "direction_agnostic"
    if upper in _BOUNDED_0_100:
        return "bounded_0_100"
    if upper in _BOUNDED_NEG100_100:
        return "bounded_neg100_100"
    return "directional"


_OP_FLIP = {
    ">":  "<",  "<":  ">",
    ">=": "<=", "<=": ">=",
    "==": "==", "!=": "!=",
}


# ── Mirror a single AND-clause ───────────────────────────────────────────────


_LHS_TOKEN_RE = re.compile(
    r"""
    (
        (?:[A-Z_]+\s*\([^)]*\))     # FUNCTION_NAME(args), e.g. RSI(14), STOCH_K(14), SUPERTREND(10,3)
        |
        (?:[A-Z_]+)                 # bare identifier, e.g. CLOSE, VWAP, OBV
    )
    """,
    re.VERBOSE,
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_lhs_name(token: str) -> str:
    m = re.match(r"([A-Z_]+)", token.strip())
    return m.group(1) if m else token.strip()


def _mirror_threshold(value: float, policy: MirrorPolicy) -> float:
    if policy == "bounded_0_100":
        return round(100.0 - value, 4)
    if policy == "bounded_neg100_100":
        return round(-value, 4)
    return value


def mirror_clause(clause: str) -> str:
    """Mirror a single comparison clause. Operators flip; bounded-oscillator
    thresholds reflect; direction-agnostic clauses pass through unchanged.
    Compound boolean strings should be handled by mirror_condition_string
    (which splits on AND/OR first)."""
    s = clause.strip()
    if not s:
        return s

    # Find the comparison operator (first match). Order matters — try the
    # two-character operators before single-char.
    op_match = re.search(r"(>=|<=|==|!=|>|<)", s)
    if not op_match:
        return s
    op = op_match.group(1)
    lhs_raw = s[:op_match.start()].strip()
    rhs_raw = s[op_match.end():].strip()

    # Identify the dominant LHS indicator. If the LHS itself contains an
    # indicator token, classify by that. Otherwise fall back to directional.
    lhs_token_match = _LHS_TOKEN_RE.search(lhs_raw)
    if not lhs_token_match:
        return s
    lhs_name = _extract_lhs_name(lhs_token_match.group(1))
    policy = _indicator_policy(lhs_name)

    if policy == "direction_agnostic":
        return s

    flipped_op = _OP_FLIP.get(op, op)

    # Mirror RHS numeric literal when bounded-oscillator policy applies.
    if policy in ("bounded_0_100", "bounded_neg100_100"):
        num_match = _NUMBER_RE.search(rhs_raw)
        if num_match:
            try:
                value = float(num_match.group(0))
                mirrored = _mirror_threshold(value, policy)
                rhs_raw = (
                    rhs_raw[:num_match.start()]
                    + (str(int(mirrored)) if mirrored == int(mirrored) else str(mirrored))
                    + rhs_raw[num_match.end():]
                )
            except ValueError:
                pass

    return f"{lhs_raw} {flipped_op} {rhs_raw}"


_CONNECTOR_RE = re.compile(r"^\s*(AND|OR)\s+", re.IGNORECASE)


def mirror_condition_string(cond: str) -> str:
    """Mirror every comparison atom in a boolean condition. AND / OR
    connectors are preserved; parenthesised sub-groups recurse.

    The split is paren-aware — AND / OR tokens inside a function call like
    `PREV(EMA(20), 3)` or a grouping like `(A AND B)` are left alone; only
    top-level connectors at depth 0 are used to delimit atoms.
    """
    if not cond or not cond.strip():
        return ""
    return _mirror_expr(cond.strip())


def _mirror_expr(expr: str) -> str:
    atoms = _split_top_level(expr)
    if len(atoms) == 1:
        return _mirror_atom(atoms[0])
    pieces: list[str] = []
    for atom in atoms:
        upper = atom.strip().upper()
        if upper in ("AND", "OR"):
            pieces.append(upper)
        else:
            pieces.append(_mirror_atom(atom))
    return " ".join(p for p in pieces if p)


def _mirror_atom(atom: str) -> str:
    atom = atom.strip()
    if not atom:
        return ""
    # Parenthesised sub-group: recurse on the inner expression.
    if atom.startswith("(") and atom.endswith(")") and _outer_paren_balanced(atom):
        return "(" + _mirror_expr(atom[1:-1]) + ")"
    return mirror_clause(atom)


def _outer_paren_balanced(s: str) -> bool:
    """True when the leading `(` matches the trailing `)` (i.e. the whole
    string is a single parenthesised group, not 'foo(x) AND bar(y)')."""
    if not s.startswith("(") or not s.endswith(")"):
        return False
    depth = 0
    for idx, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx != len(s) - 1:
                return False
    return depth == 0


def _split_top_level(expr: str) -> list[str]:
    """Split `expr` on AND / OR tokens that appear at paren-depth 0. Returns
    a flat list of [atom, connector, atom, connector, …].
    """
    atoms: list[str] = []
    depth = 0
    i, n = 0, len(expr)
    last = 0
    while i < n:
        ch = expr[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if depth == 0 and (ch == " " or ch == "\t"):
            # Check if the next token is a top-level connector.
            tail = expr[i:]
            m = _CONNECTOR_RE.match(tail)
            if m:
                atoms.append(expr[last:i].strip())
                atoms.append(m.group(1).upper())
                i += m.end()
                last = i
                continue
        i += 1
    atoms.append(expr[last:].strip())
    return [a for a in atoms if a]


# ── Plan-level mirror ────────────────────────────────────────────────────────


def mirror_plan(
    plan: dict[str, Any],
    *,
    primary_direction: str = "long",
) -> dict[str, Any]:
    """Build the opposite-side plan. The result has the same shape as the
    input plan; entry_condition / exit_condition / signal-name lists are
    flipped where it matters. Side-channel specs (stop loss anchor, trailing
    stop, HTF rules, reference symbol) are deep-copied and their directional
    fields are flipped where present.

    Set `primary_direction` to "long" if the input plan is the long setup;
    the result is the short setup. Set it to "short" to invert.
    """
    if not isinstance(plan, dict):
        return {}

    mirror_direction = "short" if primary_direction == "long" else "long"
    mirrored = copy.deepcopy(plan)

    # entry/exit condition strings.
    entry = str(plan.get("entry_condition") or "").strip()
    exit_ = str(plan.get("exit_condition")  or "").strip()
    mirrored["entry_condition"] = mirror_condition_string(entry)
    mirrored["exit_condition"]  = mirror_condition_string(exit_)

    # Stop loss anchor — bullish swing-low becomes bearish swing-high, etc.
    sl = mirrored.get("_stop_loss_spec")
    if isinstance(sl, dict):
        if sl.get("anchor") == "swing_low_recent":
            sl["anchor"] = "swing_high_recent"
        elif sl.get("anchor") == "swing_high_recent":
            sl["anchor"] = "swing_low_recent"
        if sl.get("type") == "swing_low":
            sl["type"] = "swing_high"
        elif sl.get("type") == "swing_high":
            sl["type"] = "swing_low"

    # HTF rules: condition strings flip per the same logic.
    htf = mirrored.get("_htf_rules")
    if isinstance(htf, list):
        for rule in htf:
            if isinstance(rule, dict) and rule.get("condition"):
                rule["condition"] = mirror_condition_string(rule["condition"])

    mirrored["_direction"] = mirror_direction
    mirrored["_mirrors_primary"] = True
    return mirrored


# ── Trading-window stability warning (rule #7) ───────────────────────────────


# Indian-market unstable periods, expressed as (start_minutes_from_open,
# end_minutes_from_open). 09:15 → 0min. 15:30 → 375min. Reference: pre-open
# auction at 09:00 leaks volatility into the first 15 minutes; the closing
# call auction at 15:30 sometimes spikes the last 15 minutes.
UNSTABLE_WINDOWS_INDIA = [
    {"name": "opening choppiness",   "from_open_min": 0,   "to_open_min": 15,
     "description": "First 15 minutes after open — bid/ask spreads wide, gap fills, low-quality fills."},
    {"name": "pre-close manipulation", "from_open_min": 360, "to_open_min": 375,
     "description": "Final 15 minutes — price discovery hands over to closing call auction; "
                    "unreliable signals from EMAs / momentum oscillators."},
]


def assess_trading_window(session_filters_dict: dict | None) -> list[dict[str, Any]]:
    """Inspect the SessionFilter snapshot the user / extractor produced and
    return any warnings about overlap with known unstable windows. Each
    warning is a dict {window, why, suggestion}. The chat layer surfaces
    these in the comprehensive summary so the user can decide whether to
    tighten the trading window before signal planning."""
    if not isinstance(session_filters_dict, dict):
        return _warnings_for_default_window()

    valid = session_filters_dict.get("valid_windows") or []
    blackout = session_filters_dict.get("blackout_windows") or []

    # The user has explicit blackout windows already — skip warnings for any
    # unstable period they've covered.
    covered_starts: set[int] = set()
    for win in blackout:
        if isinstance(win, dict) and win.get("from_open") and win.get("duration_minutes"):
            covered_starts.add(0)
        if isinstance(win, dict) and win.get("duration_minutes"):
            covered_starts.add(int(win["duration_minutes"]))

    warnings: list[dict[str, Any]] = []
    for unstable in UNSTABLE_WINDOWS_INDIA:
        if unstable["from_open_min"] == 0 and 0 in covered_starts:
            continue
        # Default warning if the user has any valid window that overlaps.
        overlaps = _overlaps_unstable(valid, unstable) if valid else True
        if overlaps:
            warnings.append({
                "window":     unstable["name"],
                "why":        unstable["description"],
                "suggestion": _suggestion_for_unstable(unstable),
            })
    return warnings


def _warnings_for_default_window() -> list[dict[str, Any]]:
    """When the user hasn't restricted the session at all, surface every
    unstable window as a heads-up."""
    return [
        {
            "window":     u["name"],
            "why":        u["description"],
            "suggestion": _suggestion_for_unstable(u),
        }
        for u in UNSTABLE_WINDOWS_INDIA
    ]


def _overlaps_unstable(valid_windows: list[dict], unstable: dict) -> bool:
    """Crude overlap check between any valid window and the unstable range,
    expressed in minutes-from-open."""
    u_lo, u_hi = unstable["from_open_min"], unstable["to_open_min"]
    if not valid_windows:
        return True
    for win in valid_windows:
        if not isinstance(win, dict):
            continue
        # from_open windows describe minutes from open
        if win.get("from_open") and win.get("duration_minutes"):
            v_lo, v_hi = 0, int(win["duration_minutes"])
        elif win.get("start_time") or win.get("end_time"):
            v_lo = _hhmm_to_open_min(win.get("start_time")) or 0
            v_hi = _hhmm_to_open_min(win.get("end_time")) or 375
        else:
            v_lo, v_hi = 0, 375
        if u_lo < v_hi and u_hi > v_lo:
            return True
    return False


def _hhmm_to_open_min(hhmm: str | None) -> int | None:
    if not hhmm:
        return None
    try:
        h, m = hhmm.split(":")
        return max(0, (int(h) * 60 + int(m)) - (9 * 60 + 15))
    except (ValueError, AttributeError):
        return None


def _suggestion_for_unstable(unstable: dict) -> str:
    if unstable["name"] == "opening choppiness":
        return "Consider 'avoid first 15 minutes from open' or 'trade only after 09:30'."
    if unstable["name"] == "pre-close manipulation":
        return "Consider 'trade before 15:15' or 'avoid last 15 minutes'."
    return f"Consider excluding {unstable['name']} from the session window."


__all__ = [
    "UNSTABLE_WINDOWS_INDIA",
    "assess_trading_window",
    "mirror_clause",
    "mirror_condition_string",
    "mirror_plan",
]
