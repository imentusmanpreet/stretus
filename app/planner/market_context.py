"""
app/planner/market_context.py — Phase 4 market-context filter pack.

Six independent filters that operate at the broader-market / event /
timeframe level, complementing the Phase-2 signal filters (which run on
the single trading instrument). Each filter has the same shape used
elsewhere in the planner (enabled / params / source / description), so
the chat layer renders them in the comprehensive summary and the
plan integration can splice them as AND-clauses or side-channel specs.

  18. broader_market_direction        — compare against NIFTY (or user-named
                                         reference symbol). Suppress signals
                                         that fight the broader trend.
  19. higher_timeframe_confirmation   — require an HTF trend in the trade
                                         direction. Defaults to the next
                                         coarser timeframe.
  20. gap_handling                    — detect gap-up / gap-down opens and
                                         apply the user-selected policy
                                         (skip / wait_for_fill / adjust).
  21. news_event_filter               — suppress signals within N bars of
                                         a known event date (earnings,
                                         policy, macro release).
  22. candlestick_pattern_confirmation
                                      — require one of HAMMER / ENGULFING /
                                         PIN_BAR / DOJI / MORNING_STAR /
                                         EVENING_STAR at the entry bar.
  23. sector_strength                 — compare against the stock's sector
                                         ETF / index. Skip when sector is
                                         weaker than the broader market.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.kb.execution_schemas import SemanticInstructions

logger = logging.getLogger(__name__)


# ── Filter spec ───────────────────────────────────────────────────────────────


@dataclass
class MarketContextFilterSpec:
    name:        str
    enabled:     bool = False
    params:      dict[str, Any] = field(default_factory=dict)
    source:      str = "default"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":        self.name,
            "enabled":     self.enabled,
            "params":      dict(self.params),
            "source":      self.source,
            "description": self.description,
        }


@dataclass
class MarketContextConfig:
    broader_market_direction:       MarketContextFilterSpec
    higher_timeframe_confirmation:  MarketContextFilterSpec
    gap_handling:                   MarketContextFilterSpec
    news_event_filter:              MarketContextFilterSpec
    candlestick_pattern:            MarketContextFilterSpec
    sector_strength:                MarketContextFilterSpec

    def all(self) -> list[MarketContextFilterSpec]:
        return [
            self.broader_market_direction,
            self.higher_timeframe_confirmation,
            self.gap_handling,
            self.news_event_filter,
            self.candlestick_pattern,
            self.sector_strength,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {f.name: f.to_dict() for f in self.all()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MarketContextConfig":
        data = data or {}

        def _load(key: str) -> MarketContextFilterSpec:
            raw = data.get(key) or {}
            return MarketContextFilterSpec(
                name=key,
                enabled=bool(raw.get("enabled", False)),
                params=dict(raw.get("params") or {}),
                source=str(raw.get("source") or "default"),
                description=str(raw.get("description") or ""),
            )

        return cls(
            broader_market_direction      = _load("broader_market_direction"),
            higher_timeframe_confirmation = _load("higher_timeframe_confirmation"),
            gap_handling                  = _load("gap_handling"),
            news_event_filter             = _load("news_event_filter"),
            candlestick_pattern           = _load("candlestick_pattern"),
            sector_strength               = _load("sector_strength"),
        )


# ── Detection ─────────────────────────────────────────────────────────────────


_PROMPT_TRIGGERS: dict[str, list[tuple[str, Callable[[re.Match], dict] | None]]] = {
    "broader_market_direction": [
        (r"(?:nifty|broader\s+market|index)\s+(?:trend|direction)", None),
        (r"(?:in\s+line\s+with|same\s+direction\s+as)\s+(?P<sym>nifty|sensex|bank\s*nifty)",
            lambda m: {"reference_symbol": _clean_ref(m.group("sym"))}),
        (r"only\s+(?:buy|long).*?if\s+(?:nifty|index|broader\s+market)\s+(?:is\s+)?(?:bullish|up|rising)",
            lambda m: {"reference_symbol": "^NSEI", "direction_rule": "match"}),
        (r"(?:relative\s+strength|outperform).*?(?P<sym>nifty|sensex|bank\s*nifty)",
            lambda m: {"reference_symbol": _clean_ref(m.group("sym")), "direction_rule": "rs"}),
    ],
    "higher_timeframe_confirmation": [
        (r"(?:higher\s+timeframe|htf)\s+(?:confirm|trend|direction)", None),
        (r"(?:check|confirm)\s+(?:on\s+)?(?P<tf>1h|4h|daily|weekly)\s+(?:trend|chart)",
            lambda m: {"htf": _norm_tf(m.group("tf"))}),
        (r"(?P<tf>1h|4h|daily|weekly)\s+trend\s+must\s+(?:be|align|match)",
            lambda m: {"htf": _norm_tf(m.group("tf"))}),
    ],
    "gap_handling": [
        (r"gap\s+up\s+(?:rule|policy|handling)", None),
        (r"skip\s+(?:trade|signal|entry)\s+(?:if|on)\s+gap",
            lambda m: {"policy": "skip"}),
        (r"wait\s+(?:for|until)\s+gap\s+(?:fill|fills)",
            lambda m: {"policy": "wait_for_fill"}),
        (r"adjust\s+entry\s+(?:on|for)\s+gap",
            lambda m: {"policy": "adjust_entry"}),
        (r"gap\s+(?:greater|more)\s+than\s+(?P<pct>\d+(?:\.\d+)?)\s*%",
            lambda m: {"gap_threshold_pct": float(m.group("pct"))}),
    ],
    "news_event_filter": [
        (r"(?:skip|avoid).*?(?:earnings|results)\s+(?:day|date|announcement)",
            lambda m: {"event_types": ["earnings"]}),
        (r"(?:no|avoid)\s+(?:trades?|signals?).*?(?:fed|rbi|policy|fomc).*?(?:meeting|decision|announcement)",
            lambda m: {"event_types": ["policy"]}),
        (r"avoid\s+(?:trades?|signals?)\s+(?:on|around)\s+(?:event|news)\s+days?", None),
        (r"news\s+(?:filter|blackout)", None),
        (r"(?:suppress|skip)\s+signals?\s+(?P<n>\d+)\s+(?:days?|bars?)\s+(?:before|around)\s+(?:earnings|results)",
            lambda m: {"proximity_days": int(m.group("n")), "event_types": ["earnings"]}),
    ],
    "candlestick_pattern": [
        (r"(?:hammer|hanging\s+man)\s+(?:pattern|candle)?",
            lambda m: {"patterns": ["HAMMER"]}),
        (r"(?:engulfing)\s+(?:pattern|candle)?",
            lambda m: {"patterns": ["ENGULFING"]}),
        (r"(?:pin\s+bar)",
            lambda m: {"patterns": ["PIN_BAR"]}),
        (r"(?:doji)",
            lambda m: {"patterns": ["DOJI"]}),
        (r"(?:morning\s+star)",
            lambda m: {"patterns": ["MORNING_STAR"]}),
        (r"(?:evening\s+star)",
            lambda m: {"patterns": ["EVENING_STAR"]}),
        (r"candle(?:stick)?\s+pattern\s+confirmation", None),
        (r"(?:require|need)\s+(?:specific\s+)?candle(?:stick)?\s+pattern", None),
    ],
    "sector_strength": [
        (r"sector\s+(?:strength|performance|trend)", None),
        (r"(?:check|verify)\s+sector\s+(?:is\s+)?(?:performing|strong)", None),
        (r"sector\s+(?:must|should)\s+(?:be\s+)?(?:bullish|outperforming|strong)",
            lambda m: {"require": "bullish"}),
        (r"avoid\s+(?:stocks?\s+in\s+)?weak\s+sectors?",
            lambda m: {"reject_weak_sector": True}),
    ],
}


def _clean_ref(s: str) -> str:
    s = (s or "").lower().replace(" ", "")
    if "banknifty" in s:
        return "^NSEBANK"
    if "sensex" in s:
        return "^BSESN"
    return "^NSEI"


def _norm_tf(s: str) -> str:
    s = (s or "").lower().strip()
    if s in ("daily", "1d", "d"):
        return "1d"
    if s in ("weekly", "1w", "w"):
        return "1w"
    if s in ("hourly", "1h", "h"):
        return "1h"
    if s in ("4h",):
        return "4h"
    return s or "1h"


# Per-symbol sector mapping for the supported universe. Used by the
# sector-strength filter to know which sector index to compare against.
SECTOR_MAP: dict[str, str] = {
    "TCS":               "^CNXIT",
    "TCS.NS":            "^CNXIT",
    "INFY":              "^CNXIT",
    "INFY.NS":           "^CNXIT",
    "INFOSYS":           "^CNXIT",
    "RELIANCE":          "^CNXENERGY",
    "RELIANCE.NS":       "^CNXENERGY",
    "ADANI":             "^CNXENERGY",
    "ADANI.NS":          "^CNXENERGY",
    "ADANIENT":          "^CNXENERGY",
    "ADANIENT.NS":       "^CNXENERGY",
    "HDFCBANK":          "^NSEBANK",
    "HDFCBANK.NS":       "^NSEBANK",
    "HDFC BANK":         "^NSEBANK",
    "NHPC":              "^CNXENERGY",
    "NHPC.NS":           "^CNXENERGY",
    "SUZLON":            "^CNXENERGY",
    "SUZLON.NS":         "^CNXENERGY",
    "GMRAIRPORTS":       "^CNXINFRA",
    "GMRAIRPORTS.NS":    "^CNXINFRA",
    "GMR AIRPORTS":      "^CNXINFRA",
    "VODAFONE":          "^CNXIT",        # Telecom not on the universe; closest proxy.
    "VODAFONE IDEA":     "^CNXIT",
    "IDEA":              "^CNXIT",
    "IDEA.NS":           "^CNXIT",
}


def build_market_context_config(
    prompt_text: str,
    instructions: SemanticInstructions | None,
    builder: Any,
    *,
    existing: MarketContextConfig | None = None,
) -> MarketContextConfig:
    cfg = existing or _default_config()
    haystack = re.sub(r"\s+", " ", (prompt_text or "")).lower()

    # Filters whose extractors return list-typed params (patterns, event
    # types) accumulate across triggers — the user may mention several
    # candlestick patterns or several event categories in the same prompt.
    LIST_MERGE_FILTERS = {"candlestick_pattern", "news_event_filter"}
    for fname, triggers in _PROMPT_TRIGGERS.items():
        spec = getattr(cfg, fname)
        list_merge = fname in LIST_MERGE_FILTERS
        for pattern, extractor in triggers:
            match = re.search(pattern, haystack)
            if not match:
                continue
            spec.enabled = True
            spec.source  = "prompt"
            if extractor:
                try:
                    new_params = extractor(match) or {}
                except Exception:
                    logger.debug(
                        "market_context|param_extractor_failed|filter=%s",
                        fname,
                        exc_info=True,
                    )
                    new_params = {}
                if list_merge:
                    for k, v in new_params.items():
                        if isinstance(v, list):
                            existing = list(spec.params.get(k) or [])
                            for item in v:
                                if item not in existing:
                                    existing.append(item)
                            spec.params[k] = existing
                        else:
                            spec.params[k] = v
                else:
                    spec.params.update(new_params)
            if not list_merge:
                break

    # Cross-reference semantic extraction.
    if instructions is not None:
        if instructions.reference_symbols:
            cfg.broader_market_direction.enabled = True
            cfg.broader_market_direction.source  = cfg.broader_market_direction.source or "prompt"
            first = instructions.reference_symbols[0]
            if getattr(first, "reference_symbol", None):
                cfg.broader_market_direction.params.setdefault(
                    "reference_symbol", first.reference_symbol,
                )
        if instructions.htf_rules:
            cfg.higher_timeframe_confirmation.enabled = True
            cfg.higher_timeframe_confirmation.source  = cfg.higher_timeframe_confirmation.source or "prompt"
            # First HTF rule's timeframe wins as default.
            first = instructions.htf_rules[0]
            if getattr(first, "timeframe", None):
                cfg.higher_timeframe_confirmation.params.setdefault(
                    "htf", first.timeframe,
                )

    _apply_defaults(cfg, builder)
    return cfg


def _default_config() -> MarketContextConfig:
    return MarketContextConfig(
        broader_market_direction      = MarketContextFilterSpec(name="broader_market_direction"),
        higher_timeframe_confirmation = MarketContextFilterSpec(name="higher_timeframe_confirmation"),
        gap_handling                  = MarketContextFilterSpec(name="gap_handling"),
        news_event_filter             = MarketContextFilterSpec(name="news_event_filter"),
        candlestick_pattern           = MarketContextFilterSpec(name="candlestick_pattern"),
        sector_strength               = MarketContextFilterSpec(name="sector_strength"),
    )


def _apply_defaults(cfg: MarketContextConfig, builder: Any) -> None:
    sentiment = (getattr(builder, "sentiment", "") or "").lower()
    direction = "bullish" if sentiment in {"bullish", "bull", "long"} else "bearish"

    bmd = cfg.broader_market_direction
    bmd.params.setdefault("reference_symbol", "^NSEI")
    bmd.params.setdefault("ema_period", 20)
    bmd.params.setdefault("direction_rule", "match")
    bmd.description = (
        f"Require {bmd.params['reference_symbol']} CLOSE "
        f"{'>' if direction == 'bullish' else '<'} EMA({bmd.params['ema_period']}) "
        f"before {'long' if direction == 'bullish' else 'short'} entries."
    )

    htf = cfg.higher_timeframe_confirmation
    htf.params.setdefault("htf", _next_coarser_tf(getattr(builder, "timeframe", None)))
    htf.params.setdefault("ema_period", 20)
    htf.params.setdefault("rule", f"CLOSE {'>' if direction == 'bullish' else '<'} EMA({htf.params['ema_period']})")
    htf.description = (
        f"On the {htf.params['htf']} timeframe, require {htf.params['rule']} "
        f"on the most recent closed bar."
    )

    gap = cfg.gap_handling
    gap.params.setdefault("policy", "skip")           # skip | wait_for_fill | adjust_entry
    gap.params.setdefault("gap_threshold_pct", 1.0)
    gap.description = (
        f"On gaps >= {gap.params['gap_threshold_pct']}% from prior close, "
        f"{gap.params['policy'].replace('_', ' ')}."
    )

    news = cfg.news_event_filter
    news.params.setdefault("proximity_days", 1)
    news.params.setdefault("event_types", ["earnings", "policy"])
    news.params.setdefault("event_calendar", [])      # populated externally
    news.description = (
        f"Suppress signals within {news.params['proximity_days']} day(s) of "
        f"{', '.join(news.params['event_types'])} events."
    )

    candle = cfg.candlestick_pattern
    candle.params.setdefault("patterns", ["HAMMER", "ENGULFING", "PIN_BAR", "DOJI"])
    candle.params.setdefault("require_any", True)
    candle.description = (
        "Require at least one of "
        + ", ".join(candle.params["patterns"])
        + " on the entry bar."
    )

    sec = cfg.sector_strength
    sym = (getattr(builder, "symbol", "") or "").upper()
    sec_idx = SECTOR_MAP.get(sym, "^NSEI")
    sec.params.setdefault("sector_symbol", sec_idx)
    sec.params.setdefault("rule", "outperform")        # outperform | bullish_only
    sec.description = (
        f"Require {sym or 'this stock'} to outperform {sec_idx} over the last "
        f"20 bars before entering."
    )


def _next_coarser_tf(tf: str | None) -> str:
    mapping = {
        "1m": "5m", "3m": "15m", "5m": "15m", "15m": "1h",
        "30m": "1h", "1h": "4h", "4h": "1d", "1d": "1w", "1w": "1mo",
    }
    if not tf:
        return "1h"
    return mapping.get(tf, "1d")


# ── Plan integration ──────────────────────────────────────────────────────────


def apply_market_context_to_plan(
    plan: dict[str, Any],
    cfg: MarketContextConfig,
) -> dict[str, Any]:
    """Translate market-context filters into AND-clauses where the engine
    already understands the tokens, and attach side-channel specs for the
    rest. Returns an audit dict describing what was added.
    """
    audit: dict[str, Any] = {"entry_added": [], "side_channels": []}
    entry = str(plan.get("entry_condition") or "").strip()
    haystack = entry

    new_clauses: list[str] = []

    bmd = cfg.broader_market_direction
    if bmd.enabled:
        ema_n = int(bmd.params.get("ema_period", 20))
        rule  = bmd.params.get("direction_rule", "match")
        if rule == "rs":
            clause = "RS(20) > 1.0"
        else:
            # REF_CLOSE token is a runtime-resolved scalar (set by the runner
            # when a reference symbol is attached). We compare against REF
            # SMA via the inline AVG() form so the engine doesn't need a
            # dedicated REF_EMA token.
            clause = f"REF_CLOSE > AVG(REF_CLOSE, {ema_n})"
        if clause not in haystack:
            new_clauses.append(clause)
            audit["entry_added"].append(clause)
            haystack += " " + clause
        # Tell the runner which reference symbol to attach.
        plan.setdefault("_reference_symbol", bmd.params.get("reference_symbol", "^NSEI"))
        audit["side_channels"].append({"broader_market_direction": bmd.to_dict()})

    htf = cfg.higher_timeframe_confirmation
    if htf.enabled:
        # Re-use the existing HTF rules side-channel so the simulator's
        # higher-timeframe entry gate consumes this filter.
        htf_rules = list(plan.get("_htf_rules") or [])
        rule = {"timeframe": htf.params.get("htf", "1h"), "condition": htf.params.get("rule")}
        if rule["condition"] and rule not in htf_rules:
            htf_rules.append(rule)
        plan["_htf_rules"] = htf_rules
        audit["side_channels"].append({"higher_timeframe_confirmation": htf.to_dict()})

    if cfg.gap_handling.enabled:
        plan["_gap_handling"] = cfg.gap_handling.to_dict()
        audit["side_channels"].append({"gap_handling": cfg.gap_handling.to_dict()})

    if cfg.news_event_filter.enabled:
        plan["_news_event_filter"] = cfg.news_event_filter.to_dict()
        audit["side_channels"].append({"news_event_filter": cfg.news_event_filter.to_dict()})

    candle = cfg.candlestick_pattern
    if candle.enabled:
        patterns = [p.upper() for p in (candle.params.get("patterns") or [])]
        # Bullish patterns by default (HAMMER, BULLISH_ENGULFING, MORNING_STAR).
        # The engine exposes them as IS_<PATTERN> 0/1 columns once the
        # candlestick-pattern precomputer runs.
        if patterns and candle.params.get("require_any", True):
            sub = " OR ".join(f"IS_{p} == 1" for p in patterns)
            clause = f"({sub})"
            if clause not in haystack:
                new_clauses.append(clause)
                audit["entry_added"].append(clause)
                haystack += " " + clause
        plan["_candlestick_patterns"] = candle.to_dict()
        audit["side_channels"].append({"candlestick_pattern": candle.to_dict()})

    sec = cfg.sector_strength
    if sec.enabled:
        plan["_sector_strength"] = sec.to_dict()
        audit["side_channels"].append({"sector_strength": sec.to_dict()})

    if new_clauses:
        plan["entry_condition"] = _join_clauses(entry, new_clauses)

    plan["_market_context"] = cfg.to_dict()
    return audit


def _join_clauses(existing: str, new_clauses: list[str]) -> str:
    new_clauses = [c.strip() for c in new_clauses if c.strip()]
    if not new_clauses:
        return existing
    joined = " AND ".join(new_clauses)
    if not existing:
        return joined
    if " AND " in existing or " OR " in existing:
        return f"({existing}) AND {joined}"
    return f"{existing} AND {joined}"


# ── Missing-critical ──────────────────────────────────────────────────────────


def find_missing_market_context(
    cfg: MarketContextConfig,
) -> list[dict[str, str]]:
    """Surface the toggles we want the user to make a deliberate decision on
    rather than silently default. Per the Phase-4 spec the user should
    confirm the gap-handling policy (most consequential, no safe default)
    and the news-event filter preference (event calendar can be empty)."""
    missing: list[dict[str, str]] = []
    if not cfg.gap_handling.enabled:
        missing.append({
            "field":    "mc.gap_handling",
            "label":    "gap-handling policy",
            "question": "What should the system do when the stock opens with "
                        "a significant gap? (skip the trade, wait for the gap "
                        "to fill, or adjust the entry)",
        })
    if not cfg.news_event_filter.enabled:
        missing.append({
            "field":    "mc.news_event_filter",
            "label":    "news / event filter",
            "question": "Do you want to suppress signals near earnings or "
                        "policy events? If yes, tell me how many days before "
                        "and after to blackout.",
        })
    return missing


# ── Summary rendering ────────────────────────────────────────────────────────


_LABELS = {
    "broader_market_direction":      "Broader market direction (rule 18)",
    "higher_timeframe_confirmation": "Higher-timeframe confirmation (rule 19)",
    "gap_handling":                  "Gap handling (rule 20)",
    "news_event_filter":             "News / event filter (rule 21)",
    "candlestick_pattern":           "Candlestick pattern confirmation (rule 22)",
    "sector_strength":               "Sector strength check (rule 23)",
}


def render_market_context_for_summary(cfg: MarketContextConfig) -> list[str]:
    lines: list[str] = ["── Market-context filters ──"]
    for spec in cfg.all():
        status = "ON (from your prompt)" if (spec.enabled and spec.source == "prompt") \
            else ("ON" if spec.enabled else "off")
        lines.append(f"{_LABELS.get(spec.name, spec.name)}: {status}")
        if spec.enabled and spec.description:
            lines.append(f"    └ {spec.description}")
    return lines


__all__ = [
    "MarketContextConfig",
    "MarketContextFilterSpec",
    "SECTOR_MAP",
    "apply_market_context_to_plan",
    "build_market_context_config",
    "find_missing_market_context",
    "render_market_context_for_summary",
]
