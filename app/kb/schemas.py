"""
app/kb/schemas.py — Pydantic models for the structured knowledge base.

These models are the contract between YAML/CSV files and the planner.
Validation happens at load time so any malformed file fails loudly on
startup instead of corrupting a runtime plan.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

SignalCategory = Literal["trend", "momentum", "volume", "volatility", "pattern", "india"]
SignalDirection = Literal["bullish", "bearish", "neutral"]
SignalKind = Literal["crossover", "regime", "threshold", "pattern"]
SignalRole = Literal["entry_trigger", "entry_filter", "exit_trigger", "exit_filter"]

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


# ── Catalog-driven phrase matching ────────────────────────────────────────────
#
# Each signal card can declare HOW the user might refer to it in natural
# language via a `match_when` block. The catalog signal picker uses these
# declarations to map user prompts directly to KB signals — replacing the
# hardcoded mapping dictionaries that used to live in constraint_compiler.py.
#
# A YAML example for macd_bullish_cross.yaml:
#
#     match_when:
#       phrases:
#         - "macd (line\\s+)?(?:crosses?|crossed)\\s+above\\s+signal"
#         - "macd bullish cross(over)?"
#         - "macd positive cross(over)?"
#       acts_as: entry_trigger      # role this signal plays when matched
#       mirrors_to: macd_bearish_cross
#       extract_params:
#         window: { from_group: 1, default: null, kind: int }
#
# Each `phrase` is a Python regex. Named-capture groups (e.g. `(?P<window>\\d+)`)
# can populate the signal's params when the picker assembles the plan.

SignalMatchRole = Literal["entry_trigger", "entry_filter", "exit_trigger", "exit_filter"]


class MatchParamSpec(BaseModel):
    """How to extract a single param value from the regex match."""

    from_group: str | int | None = None
    """Named capture group (str) or numbered group (int). None → no extraction."""

    default: float | int | str | None = None
    """Value to use when the group is absent. None means leave the param unset."""

    kind: Literal["int", "float", "str"] = "int"
    """How to coerce the captured string."""

    model_config = ConfigDict(extra="forbid")


class MatchWhen(BaseModel):
    """Catalog-driven recognition block for a signal card."""

    phrases: list[str] = Field(default_factory=list)
    """User-text regex patterns. ANY match is sufficient."""

    acts_as: SignalMatchRole = "entry_trigger"
    """The role this signal plays in the assembled plan when matched."""

    confidence: float = 1.0
    """Default confidence (0–1) — higher beats lower when two signals compete."""

    mirrors_to: str | None = None
    """KB signal name to use as the mirror exit (for 'exit on opposite' phrases)."""

    extract_params: dict[str, MatchParamSpec] = Field(default_factory=dict)
    """Param name → how to pull its value from the matched regex."""

    requires_keyword: list[str] = Field(default_factory=list)
    """Additional keywords that MUST appear in the prompt for this match to
    fire. Used to disambiguate (e.g. an RSI signal that should only fire when
    the user explicitly mentioned 'oversold' AND a value <= 35). Each entry
    is a regex; ALL must match."""

    forbids_keyword: list[str] = Field(default_factory=list)
    """Keywords that PREVENT this signal from matching. Inverse of
    requires_keyword. Useful for "vwap_bullish" to NOT fire when user said
    'vwap reclaim' (the reclaim signal should win)."""

    model_config = ConfigDict(extra="forbid")


# ── Extended card metadata (WS3) ──────────────────────────────────────────────
#
# Three additive blocks make card behaviour data-driven so the gate/compiler stop
# guessing (design §4):
#   * comparison{lhs,rhs,op}  → lets the gate adjudicate "EMA above price" vs
#     "price above EMA" deterministically (no LLM guesswork, precise repair).
#   * param_specs{name:{type,default,min,max,violation_action}} → param ranges
#     so out-of-range values become a note/clarify instead of a silent snap.
#   * engine{kind,anchors_supported} → the exact engine anchor ids a risk card
#     maps to, so the planner can never emit an anchor the engine rejects.
# All optional → existing cards keep validating; the backfill script + the
# card-validation test drive full population.

ComparisonSide = Literal[
    "price", "close", "open", "high", "low", "volume",
    "ema", "sma", "wma", "dema", "tema", "vwap", "rsi", "macd", "macd_signal",
    "adx", "atr", "bb_upper", "bb_lower", "bb_mid", "supertrend", "donchian",
    "stoch", "cci", "mfi", "obv", "zero", "level", "prev_value", "indicator",
]
ComparisonOp = Literal["gt", "lt", "ge", "le", "eq", "cross_up", "cross_down"]


class Comparison(BaseModel):
    """Structured reading of what a signal compares (design §4)."""

    lhs: ComparisonSide
    rhs: ComparisonSide
    op: ComparisonOp

    model_config = ConfigDict(extra="forbid")


ParamViolationAction = Literal["silent", "note", "clarify"]


class ParamSpec(BaseModel):
    """Per-param type + range + what to do when a value is out of range (§5d)."""

    type: Literal["int", "float", "str"] = "int"
    default: float | int | str | None = None
    min: float | int | None = None
    max: float | int | None = None
    violation_action: ParamViolationAction = "note"

    model_config = ConfigDict(extra="forbid")


class EngineMeta(BaseModel):
    """How a card maps onto the backtest engine contract."""

    kind: Literal["trigger", "filter", "exit"] | None = None
    anchors_supported: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SignalCard(BaseModel):
    """One structured card per signal — replaces free-text KB scoring."""

    name: str
    category: SignalCategory
    direction: SignalDirection
    kind: SignalKind
    roles: list[SignalRole]
    formula: str | None = None
    description: str = ""

    # WS3 — data-driven metadata (all optional; see block above).
    comparison: Comparison | None = None
    param_specs: dict[str, ParamSpec] = Field(default_factory=dict)
    engine: EngineMeta | None = None

    works_best_on: list[str] = Field(default_factory=list)
    weak_on: list[str] = Field(default_factory=list)
    unsupported_on: list[str] = Field(default_factory=list)

    contraindicated_when: list[dict] = Field(default_factory=list)
    pairs_well_with: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)

    # Free-form semantic tags for downstream filtering without hard-coded name
    # matching.  Examples: ["relative_strength", "benchmark_relative"],
    # ["ema_slope"], ["structural_sl"].  Any consumer can check
    # ``"relative_strength" in signal.tags`` instead of guessing from name.
    tags: list[str] = Field(default_factory=list)

    params_by_timeframe: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    experience_fit: dict[str, float] = Field(default_factory=dict)
    intent_fit: dict[str, float] = Field(default_factory=dict)

    # Catalog-driven recognition (Phase 14). When present, the
    # CatalogSignalPicker uses these patterns to map user prompts to this
    # signal directly — bypassing the preset → constraint-compiler pipeline.
    match_when: MatchWhen | None = None

    @field_validator("roles")
    @classmethod
    def _roles_not_empty(cls, value: list[SignalRole]) -> list[SignalRole]:
        if not value:
            raise ValueError("SignalCard.roles must contain at least one role")
        return value

    @model_validator(mode="after")
    def _derive_param_specs(self) -> "SignalCard":
        """Auto-derive a ParamSpec for every param the card uses (from
        params_by_timeframe) so EVERY card carries param metadata without
        hand-editing 124 YAMLs. An explicit `param_specs` entry in the YAML wins
        (it can add real min/max/violation_action); this only fills the gaps."""
        by_tf = self.params_by_timeframe or {}
        # Representative defaults: prefer 15m, else the first timeframe block.
        defaults: dict = {}
        if "15m" in by_tf:
            defaults = dict(by_tf["15m"])
        elif by_tf:
            defaults = dict(next(iter(by_tf.values())))
        for pname, pval in defaults.items():
            if pname in self.param_specs:
                continue
            self.param_specs[pname] = ParamSpec(
                type=("float" if isinstance(pval, float) else "int" if isinstance(pval, int) else "str"),
                default=pval,
            )
        return self

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
    exit_filters: list[str] = Field(default_factory=list)
    # Phase 5 — leg-scoped HTF entry gates. Direction-dependent: a bullish
    # leg's HTF rules ("1h trend up") differ from a bearish leg's ("1h trend
    # down"), so they belong on the leg, not the preset.
    htf: list[dict] = Field(default_factory=list)

    def all_signal_names(self) -> list[str]:
        return [self.entry_trigger, *self.entry_filters, self.exit_trigger, *self.exit_filters]


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
    # Phase 8b — wall-clock intraday cutoff carried from preset into the
    # strategy YAML. Shape: {exit_time: "15:15", timezone: "Asia/Kolkata"}.
    # Loader converts this to UTC minutes-of-day for the simulator.
    time_exit: dict | None = None
    # Phase 9 — when set, the preset is "dynamic": no symbol is required
    # from the user; instead the runtime universe scanner picks one from
    # the KB stock universe based on these conditions. The chat layer asks
    # the user to confirm a tie-break method when more than one stock
    # matches. Stored as a dict here (not the strongly-typed DiscoveryConfig)
    # to avoid an import cycle between the KB schema and the discovery
    # service; the loader/scanner cast as needed.
    discovery: dict | None = None

    # When True this preset is the PRIMARY strategy framework (e.g. ORB,
    # VWAP reclaim, EMA pullback).  When False it is a modifier/filter
    # preset (e.g. relative_strength) that should never be promoted to the
    # primary framework role by the KB keyword scorer.  Consumers that want
    # only framework presets should filter ``is_primary_framework=True``.
    is_primary_framework: bool = True

    def leg(self, sentiment: SignalDirection) -> PresetLeg | None:
        if sentiment == "bullish":
            return self.bullish
        if sentiment == "bearish":
            return self.bearish
        return None


# ── Stock universe ────────────────────────────────────────────────────────────


class Stock(BaseModel):
    """
    One row of the universe CSV — a tradeable instrument.

    KB holds canonical metadata only. Broker-specific identifiers (Upstox
    instrument key, Binance trading pair, etc.) live in
    ref_data.instrument_adapter_mappings and are resolved at runtime.
    """

    symbol: str                       # canonical identifier (RELIANCE.NS, BTC_USDT)
    display_name: str
    exchange: str                     # NSE | BSE | BINANCE_SPOT | ...
    asset_class: Literal["equity_cash", "crypto_spot"] = "equity_cash"
    quote_asset: str = "INR"          # INR for NSE/BSE equities; USDT / USDC for crypto
    sector: str
    enabled: bool = True


# ── Timeframes / markets / risk tiers ─────────────────────────────────────────

# Range-based timeframe acceptance: backtests fetch 1-minute data and the engine
# resamples up to any timeframe, so any well-formed 1m..1d interval is valid.
_TF_RANGE_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_TF_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}
_TF_MIN_MINUTES = 1
_TF_MAX_MINUTES = 1440


class TimeframeConfig(BaseModel):
    supported: list[str]
    buckets: dict[str, list[str]] = Field(default_factory=dict)
    synonyms: dict[str, str] = Field(default_factory=dict)

    def normalize(self, raw: str) -> str | None:
        """Map a free-text timeframe to its canonical form.

        Returns the canonical timeframe for any well-formed interval within the
        accepted range (1m..1d), not just the curated ``supported`` presets —
        the presets win as exact matches, everything else is range-checked.
        Returns None for malformed or out-of-range input.
        """
        if not raw:
            return None
        canonical = self.synonyms.get(raw.strip().lower(), raw.strip().lower())
        if canonical in self.supported:
            return canonical
        match = _TF_RANGE_RE.match(canonical)
        if not match:
            return None
        minutes = int(match.group(1)) * _TF_UNIT_MINUTES[match.group(2).lower()]
        if _TF_MIN_MINUTES <= minutes <= _TF_MAX_MINUTES:
            return canonical
        return None

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
    model_config = ConfigDict(extra="ignore")

    per_trade_risk_pct: float
    daily_loss_cap_pct: float
    max_open_positions: int
    base_sl_pct: float = 1.5
    base_tp_pct: float = 3.0
    # Risk circuit breakers
    max_consecutive_losses: int = 0       # 0 = disabled
    cooldown_bars_after_loss: int = 0     # 0 = disabled
    cooldown_bars_after_profit: int = 0   # 0 = disabled
    # Entry gate controls
    max_spread_bps: float = 0.0           # 0 = disabled (no spread gate)
    entry_confirmation_bars: int = 1      # bars signal must hold before entry
    gap_filter: str = "none"              # "none" | "ignore_gap_up" | "ignore_gap_down" | "ignore_both"
    # Position sizing
    max_capital_allocation_pct: float = 100.0  # 100 = no cap per trade
    position_sizing_mode: str = "fixed_fractional"
    min_trade_value: float = 1000.0


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
    exit_filters: list[PickedSignal] = Field(default_factory=list)
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
    # Phase 8b — optional wall-clock intraday cutoff carried from preset into
    # the strategy YAML. {exit_time: "15:15", timezone: "Asia/Kolkata"}.
    time_exit: dict | None = None
    trace: dict = Field(default_factory=dict)

    @property
    def entry_filter(self) -> PickedSignal | None:
        """Backward-compat shim for callers that only knew the singular field."""
        return self.entry_filters[0] if self.entry_filters else None
