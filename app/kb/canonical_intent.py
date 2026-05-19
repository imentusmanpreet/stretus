"""
app/kb/canonical_intent.py — Single canonical normalized strategy intent.

This is the *single source of truth* that bridges:
  - raw user prompt (via SemanticExtractor)
  - strategy builder state (preset, timeframe, sentiment, builder specs)
  - planner signal selection (via SemanticSignalComposer)
  - execution layer (stop_loss_spec, trailing_stop_spec, htf_rules)

Design principle: every downstream system (planner, assembler, backtest
executor, UI) should read from this object rather than re-deriving meaning
from the raw text or from scattered builder fields.

Shape mirrors the target architecture in the product spec:
  {
    "base_framework": "orb_structural",
    "execution_timeframe": "5m",
    "market_bias": "bullish",
    "filters": [{"type": "relative_strength", "benchmark": "NIFTY50", ...}],
    "htf_confluence": [{"signal": "ema_above", "timeframe": "1h", ...}],
    "risk_model": {
      "stop_loss": {"type": "structural", "anchor": "opening_range_low"},
      "risk_reward": {"ratio": 2.0},
      "trailing_stop": {"type": "ema_trailing"}
    }
  }
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Filter layer ──────────────────────────────────────────────────────────────

class CanonicalFilter(BaseModel):
    """A secondary execution filter (RS, volume, ADX, momentum, …).

    `signal` is the KB signal name that implements this filter for the given
    sentiment; the SemanticSignalComposer resolves it at plan time.
    """
    type: str
    """Semantic type: "relative_strength" | "volume" | "adx_strong" |
    "adx_threshold" | "momentum" | "macd" | "obv"."""

    benchmark: str | None = None
    """For relative-strength filters: reference symbol, e.g. "NIFTY50"."""

    lookback: int | None = None
    """Lookback window when applicable (e.g. RS lookback in bars)."""

    timeframe: str | None = None
    """Filter-specific timeframe when different from execution_timeframe."""

    signal: str | None = None
    """Pre-resolved KB signal name (filled by normalizer; used by composer)."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Extra params passed through to the signal (e.g. adx_threshold=25)."""

    source: str = "semantic"
    """"user" | "semantic" | "preset"."""


# ── HTF confluence layer ───────────────────────────────────────────────────────

class CanonicalHTFRule(BaseModel):
    """A higher-timeframe entry gate or confirmation rule."""

    timeframe: str
    """e.g. "1h", "1d", "4h"."""

    condition: str | None = None
    """Raw condition text extracted from user prompt."""

    indicator: str | None = None
    """Indicator keyword: "ema", "supertrend", "vwap", "sma", "trend"."""

    signal: str | None = None
    """Pre-resolved KB signal name (e.g. "ema_above")."""

    params: dict[str, Any] = Field(default_factory=dict)
    """e.g. {"period": 20} for EMA-based HTF."""

    role: str = "gating"
    """"gating" (hard gate) | "confirmation" (soft filter)."""

    source: str = "semantic"


# ── Risk model layer ──────────────────────────────────────────────────────────

class CanonicalStopLoss(BaseModel):
    """Structural or percentage-based stop-loss specification."""

    type: str
    """"structural" | "percent" | "atr_multiple" | "points"."""

    anchor: str | None = None
    """Structural anchor: "opening_range_low", "swing_low_recent",
    "candle_low", "orb_candle", "vwap_deviation", "opposite_side"."""

    percent: float | None = None
    """For type="percent"."""

    atr_multiple: float | None = None
    """For type="atr_multiple" or ATR padding on structural SL."""

    source: str = "semantic"
    """"user" | "semantic" | "preset" | "default"."""


class CanonicalRiskReward(BaseModel):
    """Risk:Reward ratio specification."""

    ratio: float
    """Reward multiple, e.g. 2.0 means 1:2 (risk 1, target 2)."""

    type: str = "minimum"
    """"minimum" (actual RR must be >= ratio) | "fixed"."""

    source: str = "semantic"


class CanonicalTrailingStop(BaseModel):
    """Trailing stop specification."""

    type: str
    """"ema_trailing" | "atr_based" | "percent_based" | "chandelier"."""

    ema_period: int | None = None
    activate_after_pct: float | None = None

    source: str = "semantic"


class CanonicalRiskModel(BaseModel):
    """Complete risk management model for the strategy."""

    stop_loss: CanonicalStopLoss | None = None
    risk_reward: CanonicalRiskReward | None = None
    trailing_stop: CanonicalTrailingStop | None = None


# ── Master canonical intent ────────────────────────────────────────────────────

class CanonicalSemanticIntent(BaseModel):
    """Single canonical normalized representation of the user's strategy intent.

    Lifecycle:
      1. SemanticExtractor parses the raw user prompt → SemanticInstructions.
      2. SemanticIntentNormalizer merges SemanticInstructions + builder state
         (preset, timeframe, sentiment, stop_loss_spec, …) → this object.
      3. This object is stored as builder.semantic_intent (plain dict for
         JSON round-trip across turns).
      4. SemanticSignalComposer reads this object inside Pipeline.plan() to
         inject entry filter signals for HTF confluence and RS conditions.
      5. ExecutionOrchestrator reads this to bridge structural SL/trailing/
         reference_symbol into plan underscore keys.
    """

    # ── Core strategy identity ────────────────────────────────────────────────
    base_framework: str | None = None
    """KB preset name that best captures the user's primary framework,
    e.g. "orb_structural", "trend_following", "vwap_reversal"."""

    execution_timeframe: str | None = None
    """Canonical timeframe string, e.g. "5m", "1h"."""

    market_bias: str | None = None
    """"bullish" | "bearish"."""

    # ── Filter layer ──────────────────────────────────────────────────────────
    filters: list[CanonicalFilter] = Field(default_factory=list)
    """Secondary execution filters extracted from user intent
    (relative strength, volume, ADX, momentum, …)."""

    # ── HTF confluence ────────────────────────────────────────────────────────
    htf_confluence: list[CanonicalHTFRule] = Field(default_factory=list)
    """Higher-timeframe entry gates / confirmations."""

    # ── Risk model ────────────────────────────────────────────────────────────
    risk_model: CanonicalRiskModel = Field(default_factory=CanonicalRiskModel)

    # ── Additional metadata ───────────────────────────────────────────────────
    indicators: dict[str, list[int]] = Field(default_factory=dict)
    """Explicitly mentioned indicators with periods, e.g. {"EMA": [20, 50]}."""

    quality_score: float = 0.0
    """Completeness/confidence score 0.0–1.0."""

    source_prompt: str | None = None
    """Truncated source prompt for debugging (≤ 200 chars)."""
