"""
app/planner/offline_judge.py — Dev-time faithfulness judge.

NOT in the request path.  Run in CI on the measurement corpus to verify
that the selector did not silently omit requirements from provenance.

Design (Acceptance #11)
───────────────────────
The judge independently re-derives atomic requirements from the prompt text
using deterministic regex + heuristics (no LLM).  It then checks each
requirement against the SDL's provenance:

  Found in field_sources (source=user) → captured ✓
  Found in unmapped_details            → captured (as miss) ✓
  Not found in either                  → OMISSION ✗  ← this is what match% can't catch

This is the one thing match% cannot self-report: a requirement the selector
simply forgot to put anywhere.

The offline judge never constrains prompt logic — it only measures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeResult:
    """Result of one judge run (one prompt + SDL)."""
    prompt: str
    covered: list[str] = field(default_factory=list)       # requirements captured
    missed_in_provenance: list[str] = field(default_factory=list)  # match% already counts these
    omitted: list[str] = field(default_factory=list)        # NOT in provenance at all — CRITICAL

    @property
    def ok(self) -> bool:
        return not self.omitted


# ── Requirement extractor (deterministic, no LLM) ────────────────────────────

_INDICATORS = [
    ("RSI",         r"\brsi\b"),
    ("EMA",         r"\bema\b|\bexponential\s+(?:moving\s+)?average\b"),
    ("SMA",         r"\bsma\b|\bsimple\s+(?:moving\s+)?average\b"),
    ("MACD",        r"\bmacd\b"),
    ("BB",          r"\bbollinger\b|\bbb\s+band\b"),
    ("ATR",         r"\batr\b|\baverage\s+true\s+range\b"),
    ("ADX",         r"\badx\b"),
    ("VWAP",        r"\bvwap\b"),
    ("SUPERTREND",  r"\bsuper\s*trend\b"),
    ("STOCH",       r"\bstoch(?:astic)?\b"),
    ("OBV",         r"\bobv\b|\bon[\s\-]*balance\s+volume\b"),
    ("ORB",         r"\bopening[\s\-]*range\b|\borb\b"),
    ("DONCHIAN",    r"\bdonchian\b"),
    ("KELTNER",     r"\bkeltner\b"),
]

_SL_RE = re.compile(r"\bstop[\s\-]*(?:loss)?\b", re.IGNORECASE)
_TP_RE = re.compile(r"\btake[\s\-]*profit\b|\btp\b|\btarget\b", re.IGNORECASE)
_RR_RE = re.compile(r"\b\d+\s*(?::|to)\s*\d+\b|\brisk[\s\-]*reward\b|\brr\b", re.IGNORECASE)

_DIRECTION_RE = re.compile(r"\b(?:buy|long|short|sell)\b", re.IGNORECASE)
_DYNAMIC_RE   = re.compile(r"\b(?:highest|top\s*\d+|stock\s+with|universe|scanner)\b", re.IGNORECASE)

_GATE_REGIME_RE   = re.compile(r"\btrend(?:ing)?\s*(?:regime|market|only)\b|\bregime\b", re.IGNORECASE)
_GATE_EVENT_RE    = re.compile(r"\bearning[s]?\s*date|\bskip\s+(?:trading\s+)?on\b|\bevent\b", re.IGNORECASE)
_GATE_SESSION_RE  = re.compile(r"\b(?:only|avoid)\s*(?:trade|entry)\s+(?:between|from|after)\b", re.IGNORECASE)
_GATE_VOLATILITY_RE = re.compile(r"\bvolatility\s*(?:filter|gate|band)\b", re.IGNORECASE)


def extract_requirements(prompt: str) -> list[str]:
    """Return a list of atomic requirement strings from the prompt.

    These are coarser than field_sources keys — they describe WHAT the user
    mentioned, not where it should go in the SDL.
    """
    found: list[str] = []
    p = str(prompt or "")

    # Signals / indicators
    for fam, pat in _INDICATORS:
        if re.search(pat, p, re.IGNORECASE):
            found.append(f"signal:{fam}")

    # Risk
    if _SL_RE.search(p):
        found.append("risk:stop_loss")
    if _TP_RE.search(p):
        found.append("risk:take_profit")
    if _RR_RE.search(p):
        found.append("risk:rr_ratio")

    # Direction
    if re.search(r"\blong\b|\bbuy\b", p, re.IGNORECASE):
        found.append("direction:long")
    if re.search(r"\bshort\b", p, re.IGNORECASE):
        found.append("direction:short")

    # Universe
    if _DYNAMIC_RE.search(p):
        found.append("universe:dynamic_hint")
    if re.search(r"\b(?:crypto|eth|btc|sol|matic|avax|ada)\b", p, re.IGNORECASE):
        found.append("universe:crypto_asset")
    if re.search(r"\b(?:nse|bse|india|nifty|sensex|\.ns)\b", p, re.IGNORECASE):
        found.append("universe:equity_asset")

    # Timeframe
    if re.search(r"\b\d+\s*(?:m|min|minute|h|hour|d|day)\b", p, re.IGNORECASE):
        found.append("timeframe:explicit")

    # Gates
    if _GATE_REGIME_RE.search(p):
        found.append("gate:regime")
    if _GATE_EVENT_RE.search(p):
        found.append("gate:event")
    if _GATE_SESSION_RE.search(p):
        found.append("gate:session")
    if _GATE_VOLATILITY_RE.search(p):
        found.append("gate:volatility")

    return list(dict.fromkeys(found))  # deduplicate preserving order


def _provenance_covers(req_key: str, sdl) -> bool:
    """True if the requirement is reflected in the SDL's provenance.

    Checks:
      - field_sources has at least one user-sourced entry that relates to req_key
      - OR unmapped_details has an entry that mentions the req_key category
    """
    prov = sdl.provenance
    fs = prov.field_sources or {}
    ud = prov.unmapped_details or []

    category, _, topic = req_key.partition(":")

    # Check field_sources for user-sourced entries in the right category
    for path, source in fs.items():
        if source != "user":
            continue
        if category == "signal" and "trigger" in path or "filter" in path or "exit" in path:
            return True
        if category == "risk" and "risk" in path:
            return True
        if category == "direction" and "direction" in path or "leg" in path:
            return True
        if category == "universe" and ("universe" in path or "symbol" in path):
            return True
        if category == "timeframe" and "timeframe" in path or "context" in path:
            return True
        if category == "gate" and topic in path:
            return True

    # Check unmapped_details
    for detail in ud:
        text_lower = detail.text.lower()
        kind = detail.kind
        if category == "signal" and (topic.lower() in text_lower or "signal" in text_lower):
            return True
        if category == "gate" and topic in text_lower:
            return True
        if kind == "engine_capability_gap" and category in ("universe", "signal"):
            return True
        if kind == "unsupported_universe" and category == "universe":
            return True

    return False


def judge(prompt: str, sdl) -> JudgeResult:
    """Run the offline judge on one (prompt, SDL) pair.

    Returns a JudgeResult flagging any requirements not found in provenance.
    Omissions (not in field_sources AND not in unmapped_details) are the
    critical findings — match% cannot catch these.
    """
    requirements = extract_requirements(prompt)
    result = JudgeResult(prompt=prompt)
    prov = sdl.provenance

    for req in requirements:
        if _provenance_covers(req, sdl):
            result.covered.append(req)
        else:
            # Is it in unmapped (selector saw it but couldn't map)?
            ud_texts = [(d.text or "").lower() for d in (prov.unmapped_details or [])]
            category, _, topic = req.partition(":")
            if any(topic.lower() in t or category in t for t in ud_texts):
                result.missed_in_provenance.append(req)
            else:
                result.omitted.append(req)

    return result
