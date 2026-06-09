"""
app/planner/catalog_schema.py — Compact, token-budget-aware catalog menu.

Converts app/kb/signals/*.yaml + execution_schemas.py + discovery scanner
options into a deterministic machine-readable menu that the SDL selector
can use without inventing names or formulas.

Two public surfaces
───────────────────
build_menu()           → CatalogMenu pydantic model (for validation + caching)
format_menu_for_llm()  → compact JSON string for the LLM system prompt

Design rules
────────────
* CATALOG-ONLY: the selector picks NAMES from this menu; it never writes formulas.
* Referential validation uses menu.signal_names (O(1) set lookup).
* catalog_version = SHA-256 of sorted signal names — changes when cards are
  added/removed but not on description edits (intentional: stable for caching).
* Token budget: the full menu is ~120 signal entries; the compact text
  representation uses one line per signal to stay within ~4 K tokens.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field


# ── Indicator family lookup (mirrors fidelity_validator._SIGNAL_FAMILY) ───────

_FAMILY_SUBSTRINGS: tuple[tuple[str, str], ...] = (
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
    ("aroon",      "AROON"),
    ("cmo",        "CMO"),
    ("pivot",      "PIVOT"),
    ("fib",        "FIBONACCI"),
    ("opening_range", "ORB"),
    ("orb",        "ORB"),
    ("vwap",       "VWAP"),
    ("regime",     "REGIME"),
    ("volume",     "VOLUME"),
    ("candle",     "CANDLE"),
    ("engulfing",  "CANDLE"),
    ("doji",       "CANDLE"),
    ("hammer",     "CANDLE"),
    ("rejection",  "CANDLE"),
    ("fvg",        "SMC"),
    ("market_structure", "SMC"),
    ("cmf",        "CMF"),
    ("gap",        "GAP"),
)


def _infer_family(name: str) -> str:
    n = name.lower()
    for substr, family in _FAMILY_SUBSTRINGS:
        if substr in n:
            return family
    return "OTHER"


# ── Per-param descriptor ──────────────────────────────────────────────────────

class ParamDescriptor(BaseModel):
    type: str        # "int" | "float" | "str"
    default: Any     # the catalog default (from 15m or first available tf)
    description: str = ""


# ── Signal menu entry (one per catalog card) ──────────────────────────────────

class SignalEntry(BaseModel):
    name: str
    description: str
    roles: list[str]           # entry_trigger | entry_filter | exit_trigger | exit_filter
    family: str                # SMA | EMA | RSI | … | OTHER
    direction: str             # bullish | bearish | neutral
    params: dict[str, ParamDescriptor] = Field(default_factory=dict)
    timeframes: list[str]      # works_best_on (supported)
    # WS3 — data-driven card metadata surfaced to the LLM proposer + the gate.
    comparison: dict | None = None       # {lhs, rhs, op} when the card declares it
    contradicts: list[str] = Field(default_factory=list)


# ── Discovery scanner metadata ────────────────────────────────────────────────

class ScannerMeta(BaseModel):
    """What the discovery scanner can express in dynamic universes."""

    rank_metrics: list[str]       # valid DynamicRank.by values
    tie_break_ids: list[str]      # valid DynamicUniverse.tie_break values
    screen_condition_notes: str   # prose note for the LLM


# ── Full catalog menu ─────────────────────────────────────────────────────────

class CatalogMenu(BaseModel):
    signals: list[SignalEntry]
    scanner: ScannerMeta
    catalog_version: str          # SHA-256 of sorted signal names

    @property
    def signal_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.signals)

    def get(self, name: str) -> SignalEntry | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None


# ── Builders ──────────────────────────────────────────────────────────────────

def _param_type(value: Any) -> str:
    if isinstance(value, float):
        return "float"
    if isinstance(value, int):
        return "int"
    return "str"


def _representative_params(card) -> dict[str, ParamDescriptor]:
    """Extract default params from the card (prefer 15m; fall back to first)."""
    by_tf = card.params_by_timeframe or {}
    defaults: dict[str, Any] = {}
    if "15m" in by_tf:
        defaults = dict(by_tf["15m"])
    elif by_tf:
        defaults = dict(next(iter(by_tf.values())))

    result: dict[str, ParamDescriptor] = {}
    for k, v in defaults.items():
        result[k] = ParamDescriptor(type=_param_type(v), default=v)
    return result


def _build_signal_entry(card) -> SignalEntry:
    roles = [str(r) for r in (card.roles or [])]
    timeframes = list(card.works_best_on or [])
    if not timeframes and card.params_by_timeframe:
        timeframes = list(card.params_by_timeframe.keys())
    comparison = None
    if getattr(card, "comparison", None) is not None:
        comparison = {"lhs": card.comparison.lhs, "rhs": card.comparison.rhs, "op": card.comparison.op}
    return SignalEntry(
        name=card.name,
        description=str(card.description or ""),
        roles=roles,
        family=_infer_family(card.name),
        direction=str(card.direction or "neutral"),
        params=_representative_params(card),
        timeframes=timeframes,
        comparison=comparison,
        contradicts=list(getattr(card, "contradicts", []) or []),
    )


def _scanner_meta() -> ScannerMeta:
    from app.services.discovery.tie_break import TIE_BREAK_METHODS
    return ScannerMeta(
        rank_metrics=["rvol", "rsi", "atr_pct", "distance_52w"],
        tie_break_ids=list(TIE_BREAK_METHODS.keys()),
        screen_condition_notes=(
            "Screen conditions are condition strings the engine can evaluate "
            "(e.g. 'CLOSE > VWAP', 'RSI(14) < 40'). "
            "Only use catalog-supported functions. "
            "rank.by must be one of rank_metrics. "
            "tie_break must be one of tie_break_ids."
        ),
    )


def _catalog_version(names: list[str]) -> str:
    payload = json.dumps(sorted(names), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def build_menu() -> CatalogMenu:
    """Build (and cache) the catalog menu from the live KB."""
    import logging as _logging
    _log = _logging.getLogger(__name__)

    from app.kb import kb

    entries = [_build_signal_entry(card) for card in kb.signals.values()]
    entries.sort(key=lambda e: e.name)

    version = _catalog_version([e.name for e in entries])
    menu = CatalogMenu(
        signals=entries,
        scanner=_scanner_meta(),
        catalog_version=version,
    )
    families = sorted({e.family for e in entries if e.family != "OTHER"})
    _log.info(
        "📋 catalog | MENU BUILT | signals=%d | families=%d [%s] | "
        "rank_metrics=%d | tie_breaks=%d | version=%s",
        len(entries), len(families), ",".join(families[:6]) + ("…" if len(families) > 6 else ""),
        len(menu.scanner.rank_metrics),
        len(menu.scanner.tie_break_ids),
        version,
    )
    return menu


def invalidate_menu_cache() -> None:
    """Force rebuild on next call (use in tests or after KB reload)."""
    build_menu.cache_clear()


# ── Compact LLM text format ───────────────────────────────────────────────────

def format_menu_for_llm(menu: CatalogMenu | None = None) -> str:
    """Return a compact JSON-like string for the LLM system prompt.

    Format (one entry per line):
      {"name":"ema_above","roles":["entry_filter"],"dir":"bullish","family":"EMA",
       "params":{"window_fast":9,"window_slow":21},"tf":["15m","1h"]}

    Plus a scanner section listing rank metrics and tie-break ids.

    Token budget: ~120 signals × ~120 chars ≈ ~14 400 chars ≈ ~3 600 tokens.
    """
    if menu is None:
        menu = build_menu()

    lines: list[str] = ["## SIGNAL CATALOG (choose names only from this list)"]
    for s in menu.signals:
        params_compact = {k: v.default for k, v in s.params.items()}
        entry = {
            "name": s.name,
            "roles": s.roles,
            "dir": s.direction,
            "family": s.family,
            "desc": s.description[:80] if s.description else "",
            "params": params_compact,
            "tf": s.timeframes,
        }
        lines.append(json.dumps(entry, separators=(",", ":")))

    lines.append("")
    lines.append("## DYNAMIC UNIVERSE — DISCOVERY SCANNER")
    lines.append(json.dumps({
        "rank_by": menu.scanner.rank_metrics,
        "tie_break": menu.scanner.tie_break_ids,
        "screen_note": menu.scanner.screen_condition_notes,
    }, separators=(",", ":")))

    lines.append("")
    lines.append(f"## catalog_version: {menu.catalog_version}")
    return "\n".join(lines)
