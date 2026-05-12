"""
app/planner/signal_filters.py — Toggleable behavioural filters layered on top
of any strategy's entry signal logic.

There are six filters in this pack. Each is a separate spec with its own
`enabled` flag and parameters. Each spec carries:

  • emit_clauses(builder, instructions)
        Returns the list of AND-clauses to splice into the strategy's
        entry_condition. May be empty if the filter is expressed only as a
        side-channel spec the engine reads directly.

  • side_channel(builder, instructions)
        Returns an optional dict the planner attaches to the plan under
        `_signal_filters[name]` so the simulator can act on it later (e.g.
        leg-aware volume confirmation, multi-bar consecutive confirmation).

  • describe_for_summary(builder, instructions)
        One-line trader-facing string used in the prompt-summary so the
        user can see at a glance which filter is on and what threshold it's
        using.

Filters adapt to whatever indicators the user's prompt mentions — they do
not hard-code numeric thresholds. If the user said `ADX above 28` we use
28; if they said nothing we use the recommended catalog default (25 for
ADX). Each filter records whether the chosen threshold came from the
prompt or from defaults so the summary can surface that to the user.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.kb.execution_schemas import SemanticInstructions

logger = logging.getLogger(__name__)


# ── Filter spec data class ────────────────────────────────────────────────────


@dataclass
class SignalFilterSpec:
    """Single filter's state. enabled + params are user-facing toggles; the
    other fields are filled in by build_signal_filters() during summary
    generation."""
    name:        str
    enabled:     bool = False
    params:      dict[str, Any] = field(default_factory=dict)
    source:      str = "default"        # "prompt" | "default"
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
class SignalFilterConfig:
    volume_confirmation: SignalFilterSpec
    volatility_cap:      SignalFilterSpec
    market_regime:       SignalFilterSpec
    trend_direction:     SignalFilterSpec
    multi_candle:        SignalFilterSpec
    momentum_filter:     SignalFilterSpec

    def all(self) -> list[SignalFilterSpec]:
        return [
            self.volume_confirmation,
            self.volatility_cap,
            self.market_regime,
            self.trend_direction,
            self.multi_candle,
            self.momentum_filter,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {f.name: f.to_dict() for f in self.all()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SignalFilterConfig":
        data = data or {}
        return cls(
            volume_confirmation=_load(data, "volume_confirmation"),
            volatility_cap=_load(data, "volatility_cap"),
            market_regime=_load(data, "market_regime"),
            trend_direction=_load(data, "trend_direction"),
            multi_candle=_load(data, "multi_candle"),
            momentum_filter=_load(data, "momentum_filter"),
        )


def _load(data: dict, key: str) -> SignalFilterSpec:
    raw = data.get(key) or {}
    return SignalFilterSpec(
        name=key,
        enabled=bool(raw.get("enabled", False)),
        params=dict(raw.get("params") or {}),
        source=str(raw.get("source") or "default"),
        description=str(raw.get("description") or ""),
    )


# ── Builder: derive the filter config from prompt + extraction ────────────────


# Phrases that turn each filter ON. Each tuple is (regex, optional param-extractor
# returning a partial dict that overrides defaults).
_FILTER_PROMPT_TRIGGERS: dict[str, list[tuple[str, Callable[[re.Match], dict] | None]]] = {
    "volume_confirmation": [
        (r"volume\s+(?:on\s+)?(?:pull[- ]?back|decline|drop)", None),
        (r"volume\s+(?:on\s+)?(?:bounce|reversal|reclaim|rise)", None),
        (r"decreas(?:ing|e)\s+volume.*?(?:then|followed|with).*?increas(?:ing|e)\s+volume", None),
        (r"volume\s+confirm(?:ation)?", None),
        (r"strong\s+(?:volume|participation)", None),
    ],
    "volatility_cap": [
        (r"only\s+trade.*?(?:low|stable)\s+volatility", None),
        (r"skip\s+(?:high\s+)?volatil", None),
        (r"volatility\s+(?:filter|cap|gate)", None),
        (r"choppy\s+market", None),
        (r"avoid\s+(?:high\s+)?volatil", None),
    ],
    "market_regime": [
        (r"trending\s+market", None),
        (r"only\s+(?:in\s+a\s+)?trending", None),
        (r"avoid\s+(?:choppy|sideways|range[- ]?bound)", None),
        (r"regime\s+filter", None),
        (r"adx\s+(?:above|>)\s*(?P<adx>\d+(?:\.\d+)?)",
            lambda m: {"adx_threshold": float(m.group("adx"))}),
    ],
    "trend_direction": [
        (r"flat\s+(?:ema|sma|indicator)", None),
        (r"(?:ema|sma|moving\s+average)\s+(?:should\s+be\s+)?(?:rising|sloping\s+up)", None),
        (r"trend\s+direction\s+(?:check|filter)", None),
        (r"slope\s+(?:check|filter)", None),
    ],
    "multi_candle": [
        # "2 consecutive candles", "3 confirmation candles",
        # "2 consecutive confirmation candles" (two qualifying words between
        # the number and 'candles' is allowed).
        (r"(?P<bars>\d+)\s+(?:consecutive|confirming|confirmation)(?:\s+(?:confirmation|closing|green|red|bullish|bearish))?\s+candles?",
            lambda m: {"bars": int(m.group("bars"))}),
        (r"(?:multi[- ]?candle|two[- ]?bar|three[- ]?bar)\s+confirmation", None),
        (r"require\s+(?P<bars>\d+)\s+bars?\s+confirmation",
            lambda m: {"bars": int(m.group("bars"))}),
    ],
    "momentum_filter": [
        (r"overbought", None),
        (r"oversold", None),
        (r"avoid\s+(?:late|chasing)\s+entries?", None),
        (r"momentum\s+(?:filter|check)", None),
    ],
}


def build_signal_filters(
    prompt_text: str,
    instructions: SemanticInstructions | None,
    builder: Any,
    *,
    existing: SignalFilterConfig | None = None,
) -> SignalFilterConfig:
    """Detect which filters the user enabled in their prompt, extract any
    explicit parameters, and fall back to safe defaults otherwise. Returns
    a complete SignalFilterConfig — every filter is present, only `enabled`
    differs.

    `existing` (if provided) preserves user toggles set in earlier turns.
    """
    haystack = re.sub(r"\s+", " ", (prompt_text or "")).lower()
    cfg = existing or _default_config()

    for fname, triggers in _FILTER_PROMPT_TRIGGERS.items():
        spec = getattr(cfg, fname)
        for pattern, extractor in triggers:
            match = re.search(pattern, haystack)
            if not match:
                continue
            spec.enabled = True
            spec.source  = "prompt"
            if extractor:
                try:
                    spec.params.update(extractor(match) or {})
                except Exception:
                    logger.debug("signal_filters|param_extractor_failed|name=%s", fname, exc_info=True)
            break

    # Cross-fertilise from already-extracted semantic instructions.
    if instructions is not None:
        if instructions.volume_momentum and instructions.volume_momentum.volume:
            cfg.volume_confirmation.enabled = True
            cfg.volume_confirmation.source  = cfg.volume_confirmation.source or "prompt"
        if instructions.volume_momentum and instructions.volume_momentum.momentum:
            mom = instructions.volume_momentum.momentum
            if mom.adx_threshold is not None:
                cfg.market_regime.enabled = True
                cfg.market_regime.source  = "prompt"
                cfg.market_regime.params["adx_threshold"] = float(mom.adx_threshold)
            elif mom.filter_type == "adx_strong":
                cfg.market_regime.enabled = True
                cfg.market_regime.source  = "prompt"

    _apply_defaults(cfg, builder, instructions)
    return cfg


def _default_config() -> SignalFilterConfig:
    return SignalFilterConfig(
        volume_confirmation=SignalFilterSpec(name="volume_confirmation"),
        volatility_cap     =SignalFilterSpec(name="volatility_cap"),
        market_regime      =SignalFilterSpec(name="market_regime"),
        trend_direction    =SignalFilterSpec(name="trend_direction"),
        multi_candle       =SignalFilterSpec(name="multi_candle"),
        momentum_filter    =SignalFilterSpec(name="momentum_filter"),
    )


def _apply_defaults(
    cfg: SignalFilterConfig,
    builder: Any,
    instructions: SemanticInstructions | None,
) -> None:
    # Volume confirmation default — uses the volume SMA window the user gave
    # (or 20), with a tighter pullback/bounce check expressed as a
    # side-channel spec.
    vol = cfg.volume_confirmation
    vol.params.setdefault("sma_window",       20)
    vol.params.setdefault("pullback_lookback", 3)
    vol.params.setdefault("bounce_lookback",   2)
    vol.description = (
        "Entry requires pullback volume to decline and bounce volume to rise "
        f"versus the {vol.params['sma_window']}-bar volume average."
    )

    # Volatility cap default — ATR/HV/Choppiness. Prefer Choppiness because
    # the user usually asks about "stable conditions"; cap at 60 by default.
    vc = cfg.volatility_cap
    vc.params.setdefault("indicator", "CHOPPINESS")
    vc.params.setdefault("period", 14)
    vc.params.setdefault("max_value", 60.0)
    vc.description = (
        f"Skip entries when {vc.params['indicator']}({vc.params['period']}) "
        f"> {vc.params['max_value']}."
    )

    # Market regime default — ADX threshold from prompt, else 25.
    mr = cfg.market_regime
    mr.params.setdefault("indicator", "ADX")
    mr.params.setdefault("period", 14)
    mr.params.setdefault("adx_threshold", 25.0)
    mr.description = (
        f"Only enter when {mr.params['indicator']}({mr.params['period']}) "
        f"> {mr.params['adx_threshold']} (market is trending)."
    )

    # Trend direction default — derived from the user's EMA/SMA usage. If
    # they mentioned EMA(20), check EMA(20) > PREV(EMA(20), 3).
    td = cfg.trend_direction
    chosen_ema = _pick_primary_ma(instructions)
    if chosen_ema:
        td.params.setdefault("indicator", chosen_ema[0])
        td.params.setdefault("period",    chosen_ema[1])
    else:
        td.params.setdefault("indicator", "EMA")
        td.params.setdefault("period",    20)
    td.params.setdefault("lookback", 3)
    td.description = (
        f"Reject entries when {td.params['indicator']}({td.params['period']}) "
        f"is flat (≤ its value {td.params['lookback']} bars ago)."
    )

    # Multi-candle confirmation default — 1 bar means filter is a no-op even
    # if enabled. Most users who mention "consecutive candles" want 2 or 3.
    mc = cfg.multi_candle
    mc.params.setdefault("bars", 1)
    mc.description = (
        f"Require {mc.params['bars']} consecutive bars where the entry "
        "condition stays true before firing."
    )

    # Momentum overbought / oversold filter default — uses RSI(14) with a
    # band tied to the user's stated sentiment.
    mf = cfg.momentum_filter
    sentiment = (getattr(builder, "sentiment", "") or "").lower()
    if sentiment in {"bullish", "bull", "long"}:
        mf.params.setdefault("indicator", "RSI")
        mf.params.setdefault("period", 14)
        mf.params.setdefault("max_value", 70.0)
        mf.description = f"Reject longs when RSI({mf.params['period']}) > {mf.params['max_value']} (overbought)."
    elif sentiment in {"bearish", "bear", "short"}:
        mf.params.setdefault("indicator", "RSI")
        mf.params.setdefault("period", 14)
        mf.params.setdefault("min_value", 30.0)
        mf.description = f"Reject shorts when RSI({mf.params['period']}) < {mf.params['min_value']} (oversold)."
    else:
        mf.params.setdefault("indicator", "RSI")
        mf.params.setdefault("period", 14)
        mf.params.setdefault("min_value", 30.0)
        mf.params.setdefault("max_value", 70.0)
        mf.description = (
            f"Reject when RSI({mf.params['period']}) is outside "
            f"[{mf.params['min_value']}, {mf.params['max_value']}]."
        )


def _pick_primary_ma(instructions: SemanticInstructions | None) -> tuple[str, int] | None:
    if instructions is None:
        return None
    for name in ("EMA", "SMA"):
        windows = instructions.indicators.get(name) or []
        if not windows:
            continue
        # Pick the smallest (typically the user's signal MA, not the trend MA).
        try:
            return name, int(min(windows))
        except (TypeError, ValueError):
            continue
    return None


# ── Filter → plan integration ────────────────────────────────────────────────


def apply_filters_to_plan(
    plan: dict[str, Any],
    cfg: SignalFilterConfig,
) -> dict[str, Any]:
    """Mutate `plan` so its entry_condition reflects every enabled filter,
    and stash the full filter snapshot under `_signal_filters` so the
    engine / assembler can read it. Returns an audit dict listing every
    clause/side-channel that was added."""
    audit: dict[str, Any] = {"entry_added": [], "side_channels": []}
    entry_condition = str(plan.get("entry_condition") or "").strip()
    haystack = entry_condition

    new_clauses: list[str] = []

    if cfg.volume_confirmation.enabled:
        # Leg-aware volume comparison cannot be expressed as a single-bar
        # AND clause — it needs decreasing-then-increasing volume legs over
        # a window. We attach it as a side-channel spec the simulator can
        # honour. We also add a softer single-bar fallback so even
        # engines that ignore the side-channel get the basic filter.
        sw = int(cfg.volume_confirmation.params["sma_window"])
        soft_clause = f"VOLUME > VOLUME_SMA({sw})"
        if soft_clause not in haystack:
            new_clauses.append(soft_clause)
            audit["entry_added"].append(soft_clause)
            haystack += " " + soft_clause
        audit["side_channels"].append({"volume_confirmation": cfg.volume_confirmation.to_dict()})

    if cfg.volatility_cap.enabled:
        ind = cfg.volatility_cap.params.get("indicator", "CHOPPINESS")
        period = int(cfg.volatility_cap.params.get("period", 14))
        max_v = float(cfg.volatility_cap.params.get("max_value", 60.0))
        clause = f"{ind}({period}) < {max_v}"
        if clause not in haystack:
            new_clauses.append(clause)
            audit["entry_added"].append(clause)
            haystack += " " + clause

    if cfg.market_regime.enabled:
        ind = cfg.market_regime.params.get("indicator", "ADX")
        period = int(cfg.market_regime.params.get("period", 14))
        thr = float(cfg.market_regime.params.get("adx_threshold", 25.0))
        clause = f"{ind}({period}) > {thr}"
        if not re.search(rf"\b{re.escape(ind)}\s*\(\s*\d+\s*\)\s*>", haystack):
            new_clauses.append(clause)
            audit["entry_added"].append(clause)
            haystack += " " + clause

    if cfg.trend_direction.enabled:
        ind = cfg.trend_direction.params.get("indicator", "EMA")
        period = int(cfg.trend_direction.params.get("period", 20))
        lookback = int(cfg.trend_direction.params.get("lookback", 3))
        clause = f"{ind}({period}) > PREV({ind}({period}), {lookback})"
        if clause not in haystack:
            new_clauses.append(clause)
            audit["entry_added"].append(clause)
            haystack += " " + clause

    if cfg.multi_candle.enabled and int(cfg.multi_candle.params.get("bars", 1)) > 1:
        # Multi-bar consecutive confirmation cannot be expressed as a
        # simple AND clause on the same bar; the simulator needs to require
        # the entry condition to remain true for N consecutive bars before
        # firing. Side-channel spec carries the bar count.
        audit["side_channels"].append({"multi_candle": cfg.multi_candle.to_dict()})

    if cfg.momentum_filter.enabled:
        ind = cfg.momentum_filter.params.get("indicator", "RSI")
        period = int(cfg.momentum_filter.params.get("period", 14))
        min_v = cfg.momentum_filter.params.get("min_value")
        max_v = cfg.momentum_filter.params.get("max_value")
        clauses_to_add: list[str] = []
        if min_v is not None:
            clauses_to_add.append(f"{ind}({period}) > {float(min_v)}")
        if max_v is not None:
            clauses_to_add.append(f"{ind}({period}) < {float(max_v)}")
        for clause in clauses_to_add:
            if clause not in haystack:
                new_clauses.append(clause)
                audit["entry_added"].append(clause)
                haystack += " " + clause

    if new_clauses:
        plan["entry_condition"] = _join_clauses(entry_condition, new_clauses)

    plan["_signal_filters"] = cfg.to_dict()
    if audit["entry_added"] or audit["side_channels"]:
        logger.info(
            "signal_filters|event=applied|entry_added=%s|side_channels=%s",
            audit["entry_added"],
            [list(s.keys())[0] for s in audit["side_channels"]],
        )
    return audit


def _join_clauses(existing: str, new_clauses: list[str]) -> str:
    new_clauses = [c for c in (c.strip() for c in new_clauses) if c]
    if not new_clauses:
        return existing
    joined = " AND ".join(new_clauses)
    if not existing:
        return joined
    if " AND " in existing or " OR " in existing:
        return f"({existing}) AND {joined}"
    return f"{existing} AND {joined}"


# ── Summary rendering ────────────────────────────────────────────────────────


def render_filters_for_summary(cfg: SignalFilterConfig) -> list[str]:
    """Lines listing each filter's state for the prompt-summary."""
    lines = ["── Signal filters ──"]
    for spec in cfg.all():
        status = "ON" if spec.enabled else "off"
        src = f" (from your prompt)" if spec.enabled and spec.source == "prompt" else ""
        lines.append(f"{_label(spec.name)}: {status}{src}")
        if spec.enabled and spec.description:
            lines.append(f"    └ {spec.description}")
    lines.append(
        "Reply 'enable <filter>' or 'disable <filter>' to toggle any of these. "
        "Filters not toggled stay at their defaults."
    )
    return lines


_LABELS = {
    "volume_confirmation": "Volume confirmation",
    "volatility_cap":      "Volatility cap",
    "market_regime":       "Market regime (trending)",
    "trend_direction":     "Trend / momentum direction",
    "multi_candle":        "Multi-candle confirmation",
    "momentum_filter":     "Momentum overbought / oversold",
}


def _label(name: str) -> str:
    return _LABELS.get(name, name.replace("_", " ").title())


__all__ = [
    "SignalFilterConfig",
    "SignalFilterSpec",
    "apply_filters_to_plan",
    "build_signal_filters",
    "render_filters_for_summary",
]
