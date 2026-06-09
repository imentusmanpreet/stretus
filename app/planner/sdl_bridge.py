"""
app/planner/sdl_bridge.py — Format bridge between the SDL pipeline and the
existing builder / signal_plan legacy shapes.

Three responsibilities:
  1. Detect whether a user prompt is a strategy description (not a wizard
     answer or a general question) — gates the SDL path.
  2. Convert a StrategyArtifact → legacy signal_plan dict so everything
     downstream of plan_signals (response composers, backtest, fidelity
     validator) keeps working unchanged.
  3. Populate a StrategyBuilder from an SDL + artifact so the rest of
     chat_service.py sees familiar fields (entry_condition, stop_loss, etc.).

Nothing here modifies the SDL schema or the engine contract.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.planner.compiler import StrategyArtifact
    from app.planner.sdl import SDL

logger = logging.getLogger(__name__)

# ── Strategy-description detector ────────────────────────────────────────────
# Heuristic: a message is a strategy description when it contains at least 2
# of the 4 "pillars" (indicator, risk, timeframe, direction).  This is fast,
# no LLM needed, and has very low false-positive rate because answering a
# wizard question like "yes, intraday" rarely mentions both a specific
# indicator AND a stop level.

_INDICATOR_RE = re.compile(
    r"\b(?:rsi|ema|sma|macd|vwap|atr|bollinger|adx|supertrend|ichimoku|"
    r"stoch(?:astic)?|cci|mfi|obv|keltner|donchian|aroon|cmo|pivot|"
    r"opening[\s\-]+range|orb|breakout|breakdown|crossover|cross[\s\-]+above|"
    r"cross[\s\-]+below|above.*band|below.*band)\b",
    re.IGNORECASE,
)
_RISK_RE = re.compile(
    r"\b(?:stop[\s\-]*loss|sl\b|take[\s\-]*profit|tp\b|risk[\s\-]*reward|"
    r"\brr\b|target|atr.*stop|stop.*atr|trailing.*stop|structural.*stop|"
    r"\d+%.*(?:stop|target))\b",
    re.IGNORECASE,
)
_TIMEFRAME_RE = re.compile(
    r"\b(?:\d+\s*(?:min(?:ute)?s?|m)\b|\d+\s*h(?:our)?s?\b|daily|weekly|"
    r"15m|5m|1h|4h|1d|intraday|scalp(?:ing)?)\b",
    re.IGNORECASE,
)
_DIRECTION_RE = re.compile(
    r"\b(?:buy|sell|long|short|enter|entry|exit|breakout|breakdown|"
    r"go long|go short|long entry|short entry)\b",
    re.IGNORECASE,
)
# Extra signal that the message contains a FULL strategy spec (not just one field)
_STRATEGY_STRUCTURE_RE = re.compile(
    r"\b(?:entry\s*:|exit\s*:|stop\s*loss\s*:|take\s*profit\s*:|sl\s*:|tp\s*:|"
    r"define\s*:|strategy\s*:|when\s+(?:rsi|ema|price|close|volume|macd|vwap|atr))\b",
    re.IGNORECASE,
)


def is_strategy_description(prompt: str) -> bool:
    """True when the prompt looks like a full strategy description.

    Requires ≥ 2 of the 4 pillars (indicator, risk, timeframe, direction)
    OR the presence of structural keywords ("Entry:", "Stop Loss:", etc.).
    Short messages (< 15 words) are never treated as strategy descriptions
    even if they match — they're likely wizard answers.
    """
    if not prompt or not prompt.strip():
        return False

    word_count = len(prompt.split())
    if word_count < 7:
        return False

    # Structural marker alone is strong evidence
    if _STRATEGY_STRUCTURE_RE.search(prompt):
        pillars = sum([
            bool(_INDICATOR_RE.search(prompt)),
            bool(_RISK_RE.search(prompt)),
            bool(_TIMEFRAME_RE.search(prompt)),
            bool(_DIRECTION_RE.search(prompt)),
        ])
        if pillars >= 1:
            return True

    # Generic threshold: ≥ 2 pillars
    pillars = sum([
        bool(_INDICATOR_RE.search(prompt)),
        bool(_RISK_RE.search(prompt)),
        bool(_TIMEFRAME_RE.search(prompt)),
        bool(_DIRECTION_RE.search(prompt)),
    ])
    return pillars >= 2


# ── Artifact → legacy signal_plan shape ──────────────────────────────────────

def artifact_to_signal_plan(artifact: "StrategyArtifact", sdl: "SDL") -> dict[str, Any]:
    """Convert a StrategyArtifact to the legacy signal_plan dict.

    The legacy shape consumed by builder.apply_signal_plan and the response
    composers is:
      {
        entry: [{name, params, signal_type, timeframe}],
        exit:  [{name, params, signal_type, timeframe}],
        entry_condition: str,
        exit_condition:  str,
        signals_used:    [str],
        signals_available: int,
        _sl_pct: float,
        _tp_pct: float,
        _sdl_artifact: {...},   ← new: bridges to the artifact for backtest
      }
    """
    from app.kb import kb

    # Build entry signal list from SDL legs
    entry_signals: list[dict] = []
    exit_signals: list[dict] = []

    for leg in sdl.legs or []:
        tf = artifact.timeframe
        trigger = leg.entry.trigger
        entry_signals.append({
            "name":        trigger.name,
            "params":      dict(trigger.params or {}),
            "signal_type": "TRIGGER",
            "timeframe":   tf,
            "source":      "SDL",
        })
        for flt in leg.entry.filters or []:
            entry_signals.append({
                "name":        flt.name,
                "params":      dict(flt.params or {}),
                "signal_type": "FILTER",
                "timeframe":   tf,
                "source":      "SDL",
            })
        for trig in leg.exit.triggers or []:
            exit_signals.append({
                "name":        trig.name,
                "params":      dict(trig.params or {}),
                "signal_type": "TRIGGER",
                "timeframe":   tf,
                "source":      "SDL",
            })

    signals_used = [s["name"] for s in entry_signals + exit_signals if s.get("name")]

    return {
        "entry":             entry_signals,
        "exit":              exit_signals,
        "entry_condition":   artifact.entry_condition,
        "exit_condition":    artifact.exit_condition or (
            "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
        ),
        # Phase 12 — short leg (present only for two-sided "both" strategies).
        "short_entry_condition": artifact.short_entry_condition or "",
        "short_exit_condition":  artifact.short_exit_condition or "",
        "direction":             artifact.direction,
        "signals_used":      signals_used,
        "signals_available": len(kb.signals),
        "_sl_pct":           artifact.stop_loss_pct,
        "_tp_pct":           artifact.take_profit_pct,
        # Bridge: the artifact carries the compiled strategy for backtest
        "_sdl_artifact_id":  artifact.artifact_id,
        "_sdl_version":      artifact.version,
        "_sdl_content_hash": artifact.content_hash,
        # Gate fields for apply_signal_plan
        "_stop_loss_spec":         artifact.stop_loss_spec,
        "_trailing_stop_spec":     artifact.trailing_stop_spec,
        "_htf_rules":              artifact.htf_rules,
        "_regime_filter_allowed":  artifact.regime_filter_allowed,
        "_vol_filter_metric":      artifact.vol_filter_metric,
        "_vol_filter_window":      artifact.vol_filter_window,
        "_vol_filter_min":         artifact.vol_filter_min,
        "_vol_filter_max":         artifact.vol_filter_max,
        "_event_skip_dates":       artifact.event_skip_dates,
        "_entry_window_start_utc": artifact.entry_window_start_utc,
        "_entry_window_end_utc":   artifact.entry_window_end_utc,
        "_volume_ratio_threshold": artifact.volume_ratio_threshold,
        "_reference_symbol":       artifact.reference_symbol,
        "_discovery_config":       artifact.discovery_config,
    }


# ── Artifact → builder fields ─────────────────────────────────────────────────

def populate_builder_from_artifact(
    builder: Any,
    artifact: "StrategyArtifact",
    sdl: "SDL",
    readback_text: str,
) -> None:
    """Populate a StrategyBuilder from an SDL artifact.

    Writes the minimal set of builder fields that the rest of the chat flow
    (response composers, backtest, fidelity validator) reads from.
    """
    builder.entry_condition  = artifact.entry_condition
    builder.exit_condition   = artifact.exit_condition or (
        "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
    )
    # Phase 12 — short leg + direction so the backtest runs both sides.
    builder.direction = artifact.direction
    if artifact.short_entry_condition:
        builder.short_entry_condition = artifact.short_entry_condition
        builder.short_exit_condition = artifact.short_exit_condition or (
            "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
        )
    builder.stop_loss        = artifact.stop_loss_pct
    builder.take_profit      = artifact.take_profit_pct
    builder.stop_loss_spec   = artifact.stop_loss_spec
    builder.trailing_stop_spec = artifact.trailing_stop_spec

    if artifact.htf_rules:
        builder.htf_rules = artifact.htf_rules
    if artifact.reference_symbol:
        builder.reference_symbol = artifact.reference_symbol
    if artifact.regime_filter_allowed is not None:
        builder.regime_filter_allowed = artifact.regime_filter_allowed
    if artifact.entry_window_start_utc is not None:
        builder.entry_window_start_utc = artifact.entry_window_start_utc
    if artifact.entry_window_end_utc is not None:
        builder.entry_window_end_utc = artifact.entry_window_end_utc
    if artifact.volume_ratio_threshold is not None:
        builder.volume_ratio_threshold = artifact.volume_ratio_threshold
    if artifact.vol_filter_metric:
        builder.vol_filter_metric = artifact.vol_filter_metric
        builder.vol_filter_window = artifact.vol_filter_window
        builder.vol_filter_min    = artifact.vol_filter_min
        builder.vol_filter_max    = artifact.vol_filter_max

    # Store SDL artefacts on the builder so backtest path can use them
    builder._sdl            = sdl
    builder._sdl_artifact   = artifact
    builder._sdl_readback   = readback_text   # shown to user in plan_signals reply

    logger.info(
        "🔗 sdl_bridge | POPULATED | sl=%.2f%% | tp=%.2f%% | "
        "entry=%.80s | gates=[regime=%s, vol=%s, session=%s]",
        artifact.stop_loss_pct, artifact.take_profit_pct,
        artifact.entry_condition,
        artifact.regime_filter_allowed is not None,
        artifact.vol_filter_metric is not None,
        artifact.entry_window_start_utc is not None,
    )
