"""
app/planner/fidelity_validator.py — Compare assembled strategy to user's prompt.

After the planner picks signals and the assembler emits a strategy_draft, this
validator checks that **every concrete thing the user said is reflected in the
output**. It catches the most damaging silent failures:

  * User asked for SMA → strategy uses EMA (indicator family swap)
  * User specified period N → strategy uses a different period
  * User said long-only → strategy has short signals
  * User specified an RR → strategy ignores it or uses a different one
  * User specified an SL anchor → strategy uses a different one
  * Strategy adds entry filters the user never asked for, on signals
    the user did not mention (filter fabrication)

Each finding is a `FidelityFinding` with severity. The chat layer should
surface CRITICAL findings as clarification questions instead of confirming
the strategy.

This is intentionally cheap (regex + dict lookups) so it can run on every
turn without an extra LLM call.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


_INDICATOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "SMA": (r"\bsma\b", r"\bsimple\s+moving\s+average\b"),
    "EMA": (r"\bema\b", r"\bexponential\s+(?:moving\s+)?average\b"),
    "VWAP": (r"\bvwap\b",),
    "RSI": (r"\brsi\b", r"\brelative\s+strength\b"),
    "MACD": (r"\bmacd\b",),
    "BB":  (r"\bbollinger\b", r"\bbb\s+bands?\b"),
    "ADX": (r"\badx\b",),
    "ATR": (r"\batr\b",),
    "SUPERTREND": (r"\bsuper\s*trend\b",),
    "STOCHASTIC": (r"\bstoch(?:astic)?\b",),
    "ICHIMOKU": (r"\bichimoku\b", r"\bkumo\b"),
    "KELTNER": (r"\bkeltner\b",),
    "DONCHIAN": (r"\bdonchian\b",),
    "OBV": (r"\bobv\b", r"\bon[\s\-]*balance\s+volume\b"),
    "CCI": (r"\bcci\b",),
    "MFI": (r"\bmfi\b", r"\bmoney\s+flow\b"),
    "PIVOT": (r"\bpivot\b",),
    "FIBONACCI": (r"\bfib(?:onacci)?\b",),
}

# Map signal-name substrings → indicator family the signal actually uses.
_SIGNAL_FAMILY: tuple[tuple[str, str], ...] = (
    ("sma",        "SMA"),
    ("ema",        "EMA"),
    ("vwap",       "VWAP"),
    ("rsi",        "RSI"),
    ("macd",       "MACD"),
    ("bb_",        "BB"),
    ("bollinger",  "BB"),
    ("adx",        "ADX"),
    ("atr",        "ATR"),
    ("supertrend", "SUPERTREND"),
    ("stoch",      "STOCHASTIC"),
    ("ichimoku",   "ICHIMOKU"),
    ("keltner",    "KELTNER"),
    ("donchian",   "DONCHIAN"),
    ("obv",        "OBV"),
    ("cci",        "CCI"),
    ("mfi",        "MFI"),
)

# Indicator families the planner commonly confuses because they play the same
# role. If the user named one member and the ENTRY TRIGGER uses a different
# member of the same group, that's a silent swap (e.g. "enter above the
# Bollinger Band" → a Keltner-channel breakout). (members, human label).
_SIBLING_GROUPS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"BB", "KELTNER", "DONCHIAN"}), "volatility channel"),
    (frozenset({"SMA", "EMA"}), "moving average"),
)


@dataclass
class FidelityFinding:
    severity: str          # "critical" | "warning" | "info"
    code: str
    message: str
    field: str | None = None
    user_value: Any = None
    assembled_value: Any = None


def _indicators_in_text(text: str) -> set[str]:
    found: set[str] = set()
    if not text:
        return found
    for family, patterns in _INDICATOR_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.add(family)
                break
    return found


def _signal_family(name: str) -> str | None:
    n = (name or "").lower()
    for substr, family in _SIGNAL_FAMILY:
        if substr in n:
            return family
    return None


def _families_in_plan(plan: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    for bucket in ("entry", "exit"):
        for sig in (plan or {}).get(bucket, []) or []:
            fam = _signal_family(sig.get("name", ""))
            if fam:
                families.add(fam)
    return families


def _entry_trigger_family(plan: dict[str, Any]) -> str | None:
    """Indicator family of the ENTRY TRIGGER signal (the primary entry), or None."""
    entries = (plan or {}).get("entry", []) or []
    trigger = next(
        (s for s in entries if str(s.get("signal_type", "")).upper() == "TRIGGER"),
        None,
    )
    if trigger is None and entries:
        trigger = entries[0]
    return _signal_family(trigger.get("name", "")) if trigger else None


def _direction_from_text(text: str) -> str | None:
    if not text:
        return None
    has_long = bool(re.search(
        r"\b(?:buy|long|go\s+long|long[\s\-]*only|bullish\s+only|enter\s+long)\b",
        text, re.IGNORECASE,
    ))
    has_short = bool(re.search(
        r"\b(?:short|sell\s+short|go\s+short|short[\s\-]*only|bearish\s+only|enter\s+short)\b",
        text, re.IGNORECASE,
    ))
    if has_long and not has_short:
        return "long"
    if has_short and not has_long:
        return "short"
    if has_long and has_short:
        return "both"
    return None


def _explicit_periods(text: str, family: str) -> set[int]:
    """Return any explicit integer periods the user wrote next to `family`."""
    if not text:
        return set()
    patterns = _INDICATOR_KEYWORDS.get(family, ())
    out: set[int] = set()
    for kw_pat in patterns:
        # "20 SMA", "SMA(20)", "SMA 20"
        for m in re.finditer(rf"(\d+)\s*{kw_pat}", text, re.IGNORECASE):
            out.add(int(m.group(1)))
        for m in re.finditer(rf"{kw_pat}\s*\(?\s*(\d+)", text, re.IGNORECASE):
            out.add(int(m.group(1)))
    return out


def _periods_in_plan_for_family(plan: dict[str, Any], family: str) -> set[int]:
    out: set[int] = set()
    target = family.lower()
    for bucket in ("entry", "exit"):
        for sig in (plan or {}).get(bucket, []) or []:
            if _signal_family(sig.get("name", "")) != family:
                continue
            params = sig.get("params") or {}
            for k, v in params.items():
                if isinstance(v, (int, float)) and k.lower() in (
                    "window", "period", "length", "fast", "slow",
                    "window_fast", "window_slow", "n",
                ):
                    out.add(int(v))
    return out


def validate_strategy_fidelity(
    user_prompt: str,
    *,
    signal_plan: dict[str, Any] | None,
    risk_execution_config: dict[str, Any] | None,
    stop_loss_spec: dict[str, Any] | None = None,
    sdl_field_sources: dict[str, str] | None = None,
) -> list[FidelityFinding]:
    """Compare a fully-assembled strategy to the user's literal prompt.

    sdl_field_sources: when the strategy was built via the SDL flow, this is the
    SDL's provenance map (dotted field path → "user"/"default"/"inferred"). It is
    the AUTHORITATIVE source of "did the user state this?" — the legacy
    rms_sources is re-derived and unreliable. When the SDL says a risk field came
    from the user, we trust it and suppress the "system default" clarifications.
    """
    findings: list[FidelityFinding] = []
    plan = signal_plan or {}
    rms = risk_execution_config or {}
    rms_sources = rms.get("rms_sources") or {}

    # SDL provenance trumps the legacy rms_sources for risk-field origin.
    _fs = sdl_field_sources or {}

    def _sdl_user(prefix: str) -> bool:
        """True if the SDL marked `prefix` (or any sub-path) as user-stated."""
        return any(
            v == "user" and (k == prefix or k.startswith(prefix + "."))
            for k, v in _fs.items()
        )

    sl_is_user = _sdl_user("risk.stop_loss")
    tp_is_user = _sdl_user("risk.take_profit")
    # The RR is user-given if the SDL says so OR the user gave both SL and TP
    # (the ratio follows from them — no separate confirmation needed).
    rr_is_user = _fs.get("risk.take_profit.ratio") == "user" or (sl_is_user and tp_is_user)

    # 0. Satisfiability — does the assembled entry rule EVER fire? A condition
    # that can never be true (e.g. a collapsed RSI band ``RSI >= 0 AND RSI <= 0``),
    # or one that won't even compile (an unknown indicator like KC_UPPER),
    # backtests to zero trades — the single most wasteful failure mode.
    entry_condition = plan.get("entry_condition")
    if isinstance(entry_condition, str) and entry_condition.strip():
        from app.planner.condition_satisfiability import check_entry_satisfiable
        sat = check_entry_satisfiable(entry_condition)
        if sat:
            findings.append(FidelityFinding(**sat))

    requested = _indicators_in_text(user_prompt)
    used = _families_in_plan(plan)

    # 1. Family swap — user asked for X but strategy has only Y
    for family in requested:
        if family not in used:
            # Did the planner pick a *substitute* family? Common confusion:
            # SMA ↔ EMA. Flag CRITICAL on family swaps in the same group.
            sub = None
            if family == "SMA" and "EMA" in used:
                sub = "EMA"
            elif family == "EMA" and "SMA" in used:
                sub = "SMA"
            if sub:
                findings.append(FidelityFinding(
                    severity="critical",
                    code="indicator_family_swap",
                    message=(
                        f"You asked for **{family}** but the strategy uses **{sub}** instead. "
                        f"Want me to switch back to {family}?"
                    ),
                    field=f"indicator.{family}",
                    user_value=family,
                    assembled_value=sub,
                ))
            else:
                findings.append(FidelityFinding(
                    severity="critical",
                    code="indicator_missing",
                    message=(
                        f"You mentioned **{family}** but the strategy does not include it. "
                        f"Should I add a {family}-based signal?"
                    ),
                    field=f"indicator.{family}",
                    user_value=family,
                    assembled_value=None,
                ))

    # 1b. Entry-trigger swap within a sibling group — user asked to ENTER on X
    # but the entry trigger uses sibling Y. Check #1 can miss this when X's family
    # appears elsewhere (e.g. a bb_squeeze filter) so `used` looks satisfied. This
    # is a backstop; the catalog picker should now pick the right trigger.
    trigger_family = _entry_trigger_family(plan)
    if trigger_family:
        for members, label in _SIBLING_GROUPS:
            requested_in_group = requested & members
            if (
                requested_in_group
                and trigger_family in members
                and trigger_family not in requested_in_group
            ):
                asked = sorted(requested_in_group)[0]
                findings.append(FidelityFinding(
                    severity="critical",
                    code="entry_trigger_swap",
                    message=(
                        f"You asked to enter using **{asked}** but the entry trigger "
                        f"uses **{trigger_family}** instead (a different {label}). "
                        f"Want me to switch the entry to {asked}?"
                    ),
                    field="entry_trigger",
                    user_value=asked,
                    assembled_value=trigger_family,
                ))
                break

    # 2. Period mismatch — user said "20 SMA" but plan uses SMA(50)
    for family in requested:
        wanted = _explicit_periods(user_prompt, family)
        if not wanted:
            continue
        got = _periods_in_plan_for_family(plan, family)
        if not got:
            continue
        if not (wanted & got):
            findings.append(FidelityFinding(
                severity="critical",
                code="indicator_period_mismatch",
                message=(
                    f"You specified {family} period(s) {sorted(wanted)} but the strategy "
                    f"uses {sorted(got)}. Want me to use what you asked for?"
                ),
                field=f"indicator.{family}.period",
                user_value=sorted(wanted),
                assembled_value=sorted(got),
            ))

    # 3. Fabricated filters — user didn't mention X but plan added it as filter
    USER_FAMILIES = requested | {None}
    fabricated: list[str] = []
    for sig in (plan.get("entry") or []):
        if sig.get("signal_type", "").upper() != "FILTER":
            continue
        fam = _signal_family(sig.get("name", ""))
        if fam and fam not in USER_FAMILIES:
            fabricated.append(f"{sig.get('name')} ({fam})")
    if fabricated:
        findings.append(FidelityFinding(
            severity="warning",
            code="fabricated_filters",
            message=(
                "I added filters you didn't ask for: " + ", ".join(fabricated) + ". "
                "Keep them, drop them, or pick which to keep?"
            ),
            field="entry_filters",
            user_value=None,
            assembled_value=fabricated,
        ))

    # 4. Direction mismatch
    user_dir = _direction_from_text(user_prompt)
    if user_dir and user_dir != "both":
        # Inspect exit signals — for long-only, exits should be bearish
        # signals applied to longs, not entry shorts.
        plan_has_short_entry = any(
            "bearish" in (s.get("name") or "").lower()
            or "_down" in (s.get("name") or "").lower()
            or "_below" in (s.get("name") or "").lower()
            for s in (plan.get("entry") or [])
        )
        if user_dir == "long" and plan_has_short_entry:
            findings.append(FidelityFinding(
                severity="critical",
                code="direction_mismatch",
                message=(
                    "You asked for **long-only** entries but the strategy includes "
                    "bearish/short entry signals."
                ),
                field="direction",
                user_value="long",
                assembled_value="both",
            ))

    # 5. RR mismatch — user gave RR in prompt; strategy uses something else.
    # Capture the N:M nearest a reward/risk keyword (within ±25 chars) so we
    # don't mis-grab a time like "09:30".
    rr_user_str: str | None = None
    rr_phrase_re = re.compile(
        r"\b(r:?r|risk[\s:]*(?:to[\s:]*)?reward|reward[\s:]*(?:to[\s:]*)?risk"
        r"|reward[\s\-]*ratio|risk[\s\-]*ratio|risk[\s\-]*reward)\b",
        re.IGNORECASE,
    )
    for digit_match in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\b", user_prompt or "",
    ):
        d_start, d_end = digit_match.span()
        # Skip time-of-day matches (HH:MM format with both sides in 0-59 / 0-23)
        a, b = digit_match.group(1), digit_match.group(2)
        try:
            if (
                "." not in a and "." not in b
                and 0 <= int(a) <= 23 and 0 <= int(b) <= 59
                and ":" in digit_match.group(0)
            ):
                # Look for "AM/PM" or "to HH:MM" nearby — strong time hint
                nearby = (user_prompt or "")[max(0, d_start - 20): min(len(user_prompt or ""), d_end + 20)]
                if re.search(r"\b(am|pm|hours?|hrs?|hh:mm|to\s+\d{1,2}:\d{2})\b", nearby, re.IGNORECASE):
                    continue
        except ValueError:
            pass
        # Require a reward/risk keyword within ±25 chars
        window = (user_prompt or "")[max(0, d_start - 25): min(len(user_prompt or ""), d_end + 25)]
        if rr_phrase_re.search(window):
            rr_user_str = digit_match.group(0)
            break

    if rr_user_str and not rr_is_user:
        rr_source = rms_sources.get("risk_reward")
        if rr_source not in ("user", "semantic") and rms.get("risk_reward") is not None:
            findings.append(FidelityFinding(
                severity="critical",
                code="rr_not_from_user",
                message=(
                    "You wrote a reward:risk ratio in your message but the strategy "
                    f"is using **{rms.get('risk_reward')}** (source: {rr_source or 'unknown'}). "
                    "Want me to use your stated ratio instead?"
                ),
                field="risk_reward",
                user_value=rr_user_str,
                assembled_value=rms.get("risk_reward"),
            ))

    # 5b. TP-RR inconsistency — user provided BOTH a TP% and a different RR ratio.
    # The planner now stashes _tp_rr_inconsistent into rms when this happens.
    tp_rr = rms.get("_tp_rr_inconsistent")
    if isinstance(tp_rr, dict):
        findings.append(FidelityFinding(
            severity="critical",
            code="tp_rr_inconsistent",
            message=(
                f"You gave both a take-profit ({tp_rr.get('user_tp')}%) and a "
                f"risk-reward ratio ({tp_rr.get('rr_ratio')}). With your SL, those "
                f"two values disagree — RR would imply TP={tp_rr.get('derived_tp')}%. "
                "Which one should I use?"
            ),
            field="take_profit_pct",
            user_value=tp_rr.get("user_tp"),
            assembled_value=tp_rr.get("derived_tp"),
        ))

    # 6. SL/TP sourced from system defaults (violates RMS-user-given rule).
    # Skipped when the SDL provenance says the user stated them (authoritative).
    if not sl_is_user and rms_sources.get("stop_loss_pct") == "system_default" and rms.get("stop_loss_pct"):
        findings.append(FidelityFinding(
            severity="critical",
            code="sl_system_default",
            message=(
                f"Stop loss **{rms.get('stop_loss_pct')}%** is a system default, not "
                "your input. Please confirm the SL you want before I assemble the strategy."
            ),
            field="stop_loss_pct",
            user_value=None,
            assembled_value=rms.get("stop_loss_pct"),
        ))
    if not tp_is_user and rms_sources.get("take_profit_pct") == "system_default" and rms.get("take_profit_pct"):
        findings.append(FidelityFinding(
            severity="critical",
            code="tp_system_default",
            message=(
                f"Take profit **{rms.get('take_profit_pct')}%** is a system default, not "
                "your input. Please confirm the TP you want."
            ),
            field="take_profit_pct",
            user_value=None,
            assembled_value=rms.get("take_profit_pct"),
        ))

    if findings:
        for f in findings:
            logger.warning(
                "fidelity_validator|%s|code=%s|field=%s|user=%r|assembled=%r",
                f.severity, f.code, f.field, f.user_value, f.assembled_value,
            )

    return findings


def format_user_message(findings: list[FidelityFinding]) -> str | None:
    """Build a single user-facing message listing all critical/warning findings."""
    if not findings:
        return None
    criticals = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    lines: list[str] = []
    if criticals:
        lines.append("Before I confirm this strategy, please clarify:")
        for f in criticals:
            lines.append(f"  • {f.message}")
    if warnings:
        lines.append("")
        lines.append("Heads up — a few extras the planner added:")
        for f in warnings:
            lines.append(f"  • {f.message}")
    return "\n".join(lines)
