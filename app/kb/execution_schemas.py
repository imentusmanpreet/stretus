"""
app/kb/execution_schemas.py — Execution-layer data models for advanced strategy features.

These models represent the execution configuration that must be extracted from user prompts:
- Structural stop-loss anchoring
- Trailing stops
- Multi-timeframe entry gates
- Reference symbol conditions
- Risk:Reward derivation
- Session/time-window filters
- Volume and momentum confirmation specs
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ── Stop-Loss Models ──────────────────────────────────────────────────────────

StopLossType = Literal[
    "percent",
    "points",
    "atr_multiple",
    "swing_low",
    "swing_high",
    "candle_low",
    "candle_high",
    "orb_low",
    "orb_high",
    "vwap_deviation",
    "opposite_side",
]

PaddingMethod = Literal["atr", "percent", "points", "none"]


class StopLossPadding(BaseModel):
    """Optional padding applied to a structural SL anchor."""
    method: PaddingMethod = "none"
    atr_multiple: float | None = None  # e.g., 1.5 × ATR(14)
    percent: float | None = None       # e.g., 0.5%
    points: float | None = None        # e.g., 5 points


class StructuralStopLoss(BaseModel):
    """Structural SL specification — anchored to candles/swings, not just %."""
    type: StopLossType
    anchor: str | None = None          # e.g., "reclaim_candle", "orb_candle", "swing_low_recent"
    padding: StopLossPadding = Field(default_factory=lambda: StopLossPadding())
    description: str | None = None     # e.g., "Below VWAP reclaim candle low"


# ── Trailing Stop Models ──────────────────────────────────────────────────────

TrailingStopType = Literal[
    "ema_based",
    "atr_based",
    "chandelier",
    "percent_based",
    "dynamic",
]


class TrailingStopConfig(BaseModel):
    """Trailing stop specification."""
    enabled: bool = False
    type: TrailingStopType | None = None
    activate_after_pct: float | None = None  # e.g., 1.0 = activate after 1% profit
    trail_distance_atr_multiple: float | None = None  # For ATR-based
    trail_distance_percent: float | None = None  # For percent-based
    ema_period: int | None = None  # For EMA-based trailing
    description: str | None = None


# ── Multi-Timeframe Rules ──────────────────────────────────────────────────────

HTFRoleType = Literal["gating", "confirmation", "optional_filter"]


class HTFCondition(BaseModel):
    """Higher-timeframe rule for entry gating or confirmation."""
    timeframe: str                          # "1h", "daily", "4h", etc.
    condition: str | None = None            # e.g., "EMA(20) > EMA(50)", "CLOSE > OPEN"
    indicator: str | None = None            # e.g., "EMA", "RSI", "MACD"
    role: HTFRoleType = "gating"
    description: str | None = None


# ── Reference Symbol / Relative Strength Models ──────────────────────────────

RelationshipType = Literal["rs", "relative_strength", "outperformance", "index_confirmation", "benchmark"]


class ReferenceSymbolCondition(BaseModel):
    """Cross-symbol or benchmark condition."""
    reference_symbol: str                   # e.g., "NIFTY50", "BANKNIFTY"
    relation: RelationshipType              # How the main symbol relates
    condition: str | None = None            # e.g., "RS(14) > 0", "stronger_than_nifty"
    description: str | None = None


# ── Risk:Reward Models ────────────────────────────────────────────────────────

RiskRewardType = Literal["fixed", "derived", "minimum"]


class RiskRewardSpec(BaseModel):
    """Risk:Reward specification and derivation."""
    type: RiskRewardType = "fixed"
    ratio: float | None = None              # e.g., 2.0 for 1:2 RR
    tp_formula: str | None = None           # Mathematical formula for TP derivation
    sl_formula: str | None = None           # Mathematical formula for SL derivation
    minimum_ratio: float | None = None      # Minimum acceptable RR
    description: str | None = None


# ── Session & Time-Window Filters ────────────────────────────────────────────

class TimeWindow(BaseModel):
    """A time window for entry/exit restrictions."""
    start_time: str | None = None           # HH:MM format, e.g., "10:00"
    end_time: str | None = None             # HH:MM format, e.g., "15:30"
    duration_minutes: int | None = None     # For windows like "first 15 minutes"
    from_open: bool = False                 # If True, duration_minutes from market open
    from_close: bool = False                # If True, duration_minutes before market close
    timezone: str = "IST"


class SessionFilter(BaseModel):
    """Session and time-window execution filters."""
    enabled: bool = False
    session: Literal["morning", "afternoon", "full_day"] | None = None
    valid_windows: list[TimeWindow] = Field(default_factory=list)
    blackout_windows: list[TimeWindow] = Field(default_factory=list)
    description: str | None = None


# ── Volume & Momentum Confirmation ────────────────────────────────────────────

VolumeFilterType = Literal["spike", "above_average", "obv_rising", "chaikin_positive"]
MomentumFilterType = Literal["adx_strong", "adx_threshold", "mfi_rising", "rsi_extreme", "macd_bullish"]


class VolumeConfirmation(BaseModel):
    """Volume filter specification."""
    filter_type: VolumeFilterType | None = None
    description: str | None = None


class CandleConfirmation(BaseModel):
    """Candle pattern confirmation filter."""
    filter_type: Literal["bullish_confirmation", "bearish_confirmation"] | None = None
    description: str | None = None


class MomentumConfirmation(BaseModel):
    """Momentum filter specification."""
    filter_type: MomentumFilterType | None = None
    adx_threshold: float | None = None       # e.g., 25 for "ADX > 25"
    description: str | None = None


class VolumeAndMomentumFilters(BaseModel):
    """Combined volume and momentum confirmation specs."""
    volume: VolumeConfirmation | None = None
    momentum: MomentumConfirmation | None = None


# ── Full Semantic Instructions (Extracted from Prompt) ────────────────────────

class SemanticInstructions(BaseModel):
    """Complete semantic extraction from a user's strategy prompt.

    This is the intermediate representation that bridges user intent to
    execution-ready configuration. Created by SemanticExtractor, consumed
    by StrategyFamilyPreserver and ExecutionConfigurator.
    """

    # Strategy family detection
    strategy_family: str | None = None      # "ORB", "VWAP", "ICT", "EMA_PULLBACK", etc.
    family_confidence: float = 0.0          # 0.0 to 1.0

    # Execution layers
    entry_rules: dict = Field(default_factory=dict)
    exit_rules: dict = Field(default_factory=dict)

    # Structural SL extraction
    stop_loss: StructuralStopLoss | None = None

    # TP extraction (may be derived from RR)
    take_profit: dict | None = None

    # Trailing stop
    trailing_stop: TrailingStopConfig | None = None

    # Multi-timeframe gating
    htf_rules: list[HTFCondition] = Field(default_factory=list)

    # Reference symbols (relative strength, benchmark comparison)
    reference_symbols: list[ReferenceSymbolCondition] = Field(default_factory=list)

    # Risk:Reward specification
    risk_reward: RiskRewardSpec | None = None

    # Session timing filters
    session_filters: SessionFilter | None = None

    # Volume and momentum
    volume_momentum: VolumeAndMomentumFilters | None = None

    # Candle pattern confirmation
    candle_confirmation: CandleConfirmation | None = None

    # Indicators explicitly mentioned in prompt (e.g., EMA(20), ADX(14))
    indicators: dict[str, list[int]] = Field(default_factory=dict)

    # Raw prompt for debugging
    original_prompt: str | None = None

    # Extraction quality metrics
    extraction_quality_score: float = 0.0   # 0.0 to 1.0, indicates completeness


# ── Execution Configuration (Assembled from SemanticInstructions + Signals) ────

class ExecutionConfig(BaseModel):
    """Complete execution configuration ready for backtesting/live trading.

    Combines signal logic from StrategyPlan with execution-layer specs from
    SemanticInstructions. This is the final output before sending to engine.
    """

    # Signal logic (from StrategyPlan)
    signal_logic: dict

    # Structural stop-loss (from SemanticInstructions)
    structural_sl: StructuralStopLoss | None = None

    # Trailing stop (from SemanticInstructions)
    trailing_stop: TrailingStopConfig | None = None

    # Risk:Reward enforcement (from SemanticInstructions)
    risk_reward: RiskRewardSpec | None = None

    # Multi-timeframe entry gates (from SemanticInstructions)
    htf_rules: list[HTFCondition] = Field(default_factory=list)

    # Reference symbols (from SemanticInstructions)
    reference_symbols: list[ReferenceSymbolCondition] = Field(default_factory=list)

    # Session filters (from SemanticInstructions)
    session_filters: SessionFilter | None = None

    # Volume and momentum (from SemanticInstructions)
    volume_momentum: VolumeAndMomentumFilters | None = None

    # Strategy family preservation tag
    strategy_family: str | None = None

    # Extraction trace
    semantic_extraction_trace: dict = Field(default_factory=dict)


# ── Strategy Family Intent Preservation ──────────────────────────────────────

StrategyFamilyType = Literal[
    "ORB",
    "VWAP_RECLAIM",
    "EMA_PULLBACK",
    "MOMENTUM",
    "REVERSAL",
    "ICT_BOS_FVG",
    "MEAN_REVERSION",
    "BREAKOUT",
    "SCALPING",
    "OTHER",
]


class CanonicalSignalSpec(BaseModel):
    """Canonical signals for a strategy family."""
    signal_name: str
    role: Literal["entry_trigger", "entry_filter", "exit_trigger"]
    mandatory: bool = False                 # If True, family isn't valid without it
    description: str | None = None


class StrategyFamilySpec(BaseModel):
    """Family-specific execution requirements and signal preferences."""
    family: StrategyFamilyType
    canonical_signals: list[CanonicalSignalSpec] = Field(default_factory=list)

    # Structural requirements
    sl_anchor_type: StopLossType | None = None  # e.g., "orb_low" for ORB

    # Key invariants
    preserved_invariants: list[str] = Field(default_factory=list)

    # Signals to avoid (contraindicated)
    contraindicated_signals: list[str] = Field(default_factory=list)

    description: str | None = None
