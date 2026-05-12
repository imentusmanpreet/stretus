"""
app/kb/schemas.py — Pydantic models for the structured knowledge base.

These models are the contract between YAML/CSV files and the planner.
Validation happens at load time so any malformed file fails loudly on
startup instead of corrupting a runtime plan.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

SignalCategory = Literal["trend", "momentum", "volume", "volatility", "pattern", "india"]
SignalDirection = Literal["bullish", "bearish", "neutral"]
SignalKind = Literal["crossover", "regime", "threshold", "pattern"]
SignalRole = Literal["entry_trigger", "entry_filter", "exit_trigger"]

HoldHorizon = Literal["seconds", "minutes", "hours", "days", "weeks"]
Frequency = Literal["very_low", "low", "medium", "high", "very_high"]
ProfitSize = Literal["small", "medium", "large"]
IntentStyle = Literal[
    "trend", "momentum", "breakout", "reversal", "scalping",
    "mean_reversion", "confirmation",
]
RiskAppetite = Literal["conservative", "moderate", "aggressive"]
Experience = Literal["beginner", "intermediate", "expert", "default"]


# ── Signal cards ──────────────────────────────────────────────────────────────


class ContraindicationRule(BaseModel):
    """A single contraindication: an exact-match dict like {sentiment: bearish}."""

    model_config = ConfigDict(extra="allow")


class SignalCard(BaseModel):
    """One structured card per signal — replaces free-text KB scoring."""

    name: str
    category: SignalCategory
    direction: SignalDirection
    kind: SignalKind
    roles: list[SignalRole]
    formula: str | None = None
    description: str = ""

    works_best_on: list[str] = Field(default_factory=list)
    weak_on: list[str] = Field(default_factory=list)
    unsupported_on: list[str] = Field(default_factory=list)

    contraindicated_when: list[dict] = Field(default_factory=list)
    pairs_well_with: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)

    params_by_timeframe: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    experience_fit: dict[str, float] = Field(default_factory=dict)
    intent_fit: dict[str, float] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def _roles_not_empty(cls, value: list[SignalRole]) -> list[SignalRole]:
        if not value:
            raise ValueError("SignalCard.roles must contain at least one role")
        return value

    @property
    def family(self) -> str:
        """Coarse family grouping used to avoid redundant picks (e.g. ema/sma/rsi)."""
        prefix = self.name.split("_", 1)[0]
        return prefix


# ── Strategy presets ──────────────────────────────────────────────────────────


class PresetLeg(BaseModel):
    """One sentiment leg of a preset (bullish or bearish)."""

    entry_trigger: str
    entry_filters: list[str] = Field(default_factory=list)
    exit_trigger: str
    # Phase 5 — leg-scoped HTF entry gates. Direction-dependent: a bullish
    # leg's HTF rules ("1h trend up") differ from a bearish leg's ("1h trend
    # down"), so they belong on the leg, not the preset.
    htf: list[dict] = Field(default_factory=list)

    def all_signal_names(self) -> list[str]:
        return [self.entry_trigger, *self.entry_filters, self.exit_trigger]


class StrategyPreset(BaseModel):
    """A named strategy template that pins a specific signal combination.

    Presets short-circuit the planner's filter+rank loop. When the user types
    a known strategy name (or any of its keywords), the pipeline skips ranking
    and uses the leg matching the requested sentiment.

    `stop_loss` and `trailing_stop` are passed straight through to the strategy
    YAML so the simulator can use structural SL (e.g. anchor at the opening
    range low) or trailing exits — both Phase 3 capabilities.
    """

    name: str
    display_name: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    supported_timeframes: list[str] = Field(default_factory=list)
    bullish: PresetLeg | None = None
    bearish: PresetLeg | None = None
    stop_loss: dict | None = None
    trailing_stop: dict | None = None
    # Phase 4 — when set, this benchmark's OHLCV is fetched alongside the
    # trade symbol and joined into the df with REF_ columns. Strategies that
    # use REF_CLOSE/RS(n)/reference_above_sma must declare this.
    reference_symbol: str | None = None

    def leg(self, sentiment: SignalDirection) -> PresetLeg | None:
        if sentiment == "bullish":
            return self.bullish
        if sentiment == "bearish":
            return self.bearish
        return None


# ── Stock universe ────────────────────────────────────────────────────────────


class Stock(BaseModel):
    """One row of the stock universe (CSV)."""

    symbol: str
    display_name: str
    exchange: str
    upstox_key: str
    sector: str
    enabled: bool = True


# ── Timeframes / markets / risk tiers ─────────────────────────────────────────


class TimeframeConfig(BaseModel):
    supported: list[str]
    buckets: dict[str, list[str]] = Field(default_factory=dict)
    synonyms: dict[str, str] = Field(default_factory=dict)

    def normalize(self, raw: str) -> str | None:
        """Map a free-text timeframe to a canonical supported one. Returns None if unknown."""
        if not raw:
            return None
        canonical = self.synonyms.get(raw.strip().lower(), raw.strip().lower())
        return canonical if canonical in self.supported else None

    def bucket_of(self, timeframe: str) -> str | None:
        for name, members in self.buckets.items():
            if timeframe in members:
                return name
        return None


class MarketHours(BaseModel):
    open: str
    close: str
    tz: str


class MarketConfig(BaseModel):
    suffix: str
    hours: MarketHours
    default_tick_size: float = 0.05
    default_lot_size: int = 1


class RiskTier(BaseModel):
    per_trade_risk_pct: float
    daily_loss_cap_pct: float
    max_open_positions: int
    base_sl_pct: float = 1.5
    base_tp_pct: float = 3.0


# ── Intent taxonomy ───────────────────────────────────────────────────────────


class IntentTaxonomy(BaseModel):
    fields: dict[str, list[str]]
    defaults: dict[str, str]


class Intent(BaseModel):
    """Structured representation of the user's trading goal."""

    hold_horizon: HoldHorizon
    frequency: Frequency
    profit_size: ProfitSize
    style: IntentStyle
    risk_appetite: RiskAppetite


# ── Planner output ────────────────────────────────────────────────────────────


class PickedSignal(BaseModel):
    """One signal that has been filtered, ranked, and parameterised."""

    name: str
    role: SignalRole
    direction: SignalDirection
    formula: str | None
    params: dict[str, float | int]
    timeframe: str

    @classmethod
    def from_card(
        cls,
        card: SignalCard,
        role: SignalRole,
        params: dict[str, float | int],
        timeframe: str,
    ) -> "PickedSignal":
        return cls(
            name=card.name,
            role=role,
            direction=card.direction,
            formula=card.formula,
            params=params,
            timeframe=timeframe,
        )


class ScoredCard(BaseModel):
    """Used in trace logs — what each candidate scored and why."""

    name: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)


class StrategyPlan(BaseModel):
    """The end-product of the planner pipeline.

    `entry_filters` is a list because real-world strategies (ORB+volume+VWAP,
    20-EMA pullback with EMA-stack confirmation, etc.) require multiple
    AND-ed conditions on top of the trigger. The pipeline picks up to
    MAX_ENTRY_FILTERS filters; the list may be empty when no compatible
    filter survives the hard-filter pass.
    """

    stock: Stock
    timeframe: str
    sentiment: SignalDirection
    intent: Intent
    entry_trigger: PickedSignal
    entry_filters: list[PickedSignal] = Field(default_factory=list)
    exit_trigger: PickedSignal
    sl_pct: float
    tp_pct: float
    risk: dict
    # Phase 3 — optional structural / trailing SL specs, propagated into the
    # strategy YAML when set. Pinned by presets like `orb_structural` so the
    # SL anchors at the opening-range low instead of a flat percent.
    stop_loss_spec: dict | None = None
    trailing_stop_spec: dict | None = None
    # Phase 4 — optional reference symbol for cross-symbol comparisons (e.g.
    # Nifty 50 for relative-strength filters). Propagated into the strategy
    # YAML so the API layer fetches the benchmark series before backtesting.
    reference_symbol: str | None = None
    # Phase 5 — optional higher-timeframe entry gates carried from a preset
    # into the strategy YAML. Each item: {timeframe: "1d", condition: "..."}.
    htf_rules: list[dict] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)

    @property
    def entry_filter(self) -> PickedSignal | None:
        """Backward-compat shim for callers that only knew the singular field."""
        return self.entry_filters[0] if self.entry_filters else None
