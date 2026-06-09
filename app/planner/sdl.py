"""
app/planner/sdl.py — Strategy Definition Language (SDL) ticket schema.

The SDL is the single source of truth for a strategy created or modified by
the selector.  Everything downstream (validator, compiler, read-back) reads
ONLY from this typed document.  Nothing is silently dropped or invented.

Design rules
────────────
* Every prompt clause maps to a field OR lands in provenance.unmapped_details.
* Every field carries a provenance source: user | default | inferred.
* content_hash is computed from the executable fields (context + universe +
  legs + risk + gates + htf_rules) — provenance, timestamps, and version
  counters are excluded so two semantically-identical tickets hash identically.
* Universe is a discriminated union: static (one named symbol) or dynamic
  (discovery-scanner rule).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Field-source provenance ───────────────────────────────────────────────────

FieldSource = Literal["user", "default", "inferred"]

UnmappedKind = Literal[
    "missing_card",
    "missing_operator",
    "missing_param",
    "engine_capability_gap",
    "unsupported_universe",
]


class UnmappedDetail(BaseModel):
    text: str
    kind: UnmappedKind
    note: str = ""


class ClarificationNeeded(BaseModel):
    field: str
    question: str
    assumed_value: Any = None


class Provenance(BaseModel):
    """Records WHERE every value in the SDL came from.

    field_sources keys are dotted paths into the SDL, including list indices
    (e.g. "legs.0.entry.trigger.name", "risk.stop_loss.type").

    unmapped_details captures every user requirement that could not be mapped
    to an SDL field or catalog card.  It is NEVER empty for missed requirements
    — silently dropping is forbidden.

    clarifications_needed holds questions for the user about assumed defaults
    for money-affecting fields (stop, target, risk:reward).
    """

    field_sources: dict[str, FieldSource] = Field(default_factory=dict)
    unmapped_details: list[UnmappedDetail] = Field(default_factory=list)
    clarifications_needed: list[ClarificationNeeded] = Field(default_factory=list)


# ── Universe ──────────────────────────────────────────────────────────────────

AssetClass = Literal["equity_cash", "crypto_spot"]
RankMetric = Literal["rvol", "rsi", "atr_pct", "distance_52w"]
RankOrder = Literal["asc", "desc"]


class StaticUniverse(BaseModel):
    """A single, explicitly named asset."""

    type: Literal["static"] = "static"
    asset_class: AssetClass
    symbol: str  # normalised: e.g. HDFCBANK.NS or ETH_USDC


class DynamicRank(BaseModel):
    by: RankMetric
    order: RankOrder = "desc"


class DynamicUniverse(BaseModel):
    """A rule that the discovery scanner resolves to ONE asset at backtest time."""

    type: Literal["dynamic"] = "dynamic"
    asset_class: AssetClass
    screen: list[str] = Field(default_factory=list)  # catalog-condition expressions
    rank: DynamicRank
    tie_break: str  # one of the scanner's available tie_break method ids


Universe = StaticUniverse | DynamicUniverse


# ── Signal reference (catalog card + params) ──────────────────────────────────

class IntentComparison(BaseModel):
    """What the user meant the signal to compare — the proposer's reading.

    The gate adjudicates a mapping error (e.g. user said "EMA above price" but
    the card means "price below EMA") ONLY from this structured stub — never by
    re-reading the prose. lhs/rhs are coarse tokens (price | close | ema | rsi |
    macd | …); op is gt | lt | cross_up | cross_down | eq.
    """

    lhs: str | None = None
    rhs: str | None = None
    op: str | None = None

    @field_validator("lhs", "rhs", "op", mode="before")
    @classmethod
    def _coerce_scalar_to_str(cls, v: Any) -> str | None:
        # The intent stub is ADVISORY metadata for the gate — it must never crash
        # the SDL. The LLM naturally writes a numeric rhs for "RSI < 50"
        # (rhs=50); accept numbers/bools by stringifying, and drop anything
        # exotic rather than raising.
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)):
            return str(v)
        return None


class IntentStub(BaseModel):
    """Per-signal evidence of user intent (design §5b.2/.5).

    `user_span` is the verbatim slice of the prompt this signal came from;
    `intended` is the proposer's structured reading of the comparison. This is
    the ONLY evidence the gate uses to decide whether a mis-mapped card may be
    silently re-mapped (proof present) or must become a note/clarification.
    """

    user_span: str = ""
    intended: IntentComparison = Field(default_factory=IntentComparison)


class SignalRef(BaseModel):
    """Reference to a catalog signal card by name, with resolved params."""

    name: str                              # must exist in the catalog
    params: dict[str, Any] = Field(default_factory=dict)
    intent: IntentStub | None = None       # proposer's per-signal intent (optional)


# ── Legs ──────────────────────────────────────────────────────────────────────

class EntrySpec(BaseModel):
    trigger: SignalRef
    filters: list[SignalRef] = Field(default_factory=list)


class ExitSpec(BaseModel):
    triggers: list[SignalRef] = Field(default_factory=list)
    filters: list[SignalRef] = Field(default_factory=list)


class Leg(BaseModel):
    direction: Literal["long", "short"]
    entry: EntrySpec
    exit: ExitSpec


# ── Risk ──────────────────────────────────────────────────────────────────────

StopLossType = Literal["percent", "atr", "points", "swing_low", "swing_high",
                        "candle_low", "candle_high", "orb_low", "orb_high",
                        "vwap_deviation", "opposite_side"]

TakeProfitType = Literal["percent", "rr", "atr", "points"]

TrailingType = Literal["atr_based", "percent_based", "ema_based", "chandelier"]

SizingMode = Literal["risk_based", "fixed_fractional", "fixed_units"]


class StopLossSpec(BaseModel):
    type: StopLossType
    value: float | None = None        # for percent / points
    multiple: float | None = None     # for atr multiple
    window: int | None = None         # for atr window
    anchor: str | None = None         # for structural types
    padding_atr: float | None = None  # optional padding on structural SL


class TakeProfitSpec(BaseModel):
    type: TakeProfitType
    value: float | None = None        # for percent / points
    ratio: float | None = None        # for rr ratio (e.g. 2.0 = 2:1)
    multiple: float | None = None     # for atr multiple
    window: int | None = None         # for atr window


class TrailingSpec(BaseModel):
    enabled: bool = False
    type: TrailingType | None = None
    activate_after_pct: float | None = None
    distance_atr: float | None = None
    distance_pct: float | None = None
    ema_period: int | None = None


class ScaleOutSpec(BaseModel):
    at_rr: float       # e.g. 1.5 means "at 1.5R profit"
    size_pct: float    # percentage of the position to exit


class SizingSpec(BaseModel):
    mode: SizingMode = "risk_based"
    risk_pct: float | None = None    # % of capital to risk per trade
    fixed_pct: float | None = None   # % of capital per trade (fixed_fractional)
    fixed_units: int | None = None   # absolute unit count (fixed_units)


class RiskSpec(BaseModel):
    stop_loss: StopLossSpec | None = None
    take_profit: TakeProfitSpec | None = None
    trailing: TrailingSpec | None = None
    scale_outs: list[ScaleOutSpec] = Field(default_factory=list)
    sizing: SizingSpec | None = None


# ── Gates (entry filters not expressed as signal cards) ───────────────────────

class RegimeGate(BaseModel):
    allowed: list[str] = Field(default_factory=list)  # e.g. ["trending"]


class VolatilityGate(BaseModel):
    metric: Literal["atr", "natr"] = "atr"
    window: int = 14
    min: float | None = None
    max: float | None = None


class EventGate(BaseModel):
    skip_dates: list[str] = Field(default_factory=list)  # YYYY-MM-DD


class SessionGate(BaseModel):
    start: str | None = None   # HH:MM in local market time
    end: str | None = None
    timezone: str = "IST"


class RelativeStrengthGate(BaseModel):
    window: int = 14
    min_ratio: float | None = None
    reference_symbol: str | None = None


class VolumeRatioGate(BaseModel):
    window: int = 20
    min_ratio: float | None = None


class GatesSpec(BaseModel):
    regime: RegimeGate | None = None
    volatility: VolatilityGate | None = None
    event: EventGate | None = None
    session: SessionGate | None = None
    relative_strength: RelativeStrengthGate | None = None
    volume_ratio: VolumeRatioGate | None = None


# ── Higher-timeframe rules ────────────────────────────────────────────────────

HTFRole = Literal["gating", "confirmation", "optional_filter"]


class HTFRule(BaseModel):
    timeframe: str                          # e.g. "1h", "4h", "1d"
    condition: str | None = None            # condition string (engine-evaluable)
    signal: SignalRef | None = None         # OR a catalog card reference
    role: HTFRole = "gating"


# ── SDL ticket ────────────────────────────────────────────────────────────────

class StrategyContext(BaseModel):
    market: str                             # e.g. "indian_stocks", "crypto"
    timeframe: str                          # e.g. "15m", "1h"
    objective: str                          # e.g. "mean_reversion", "breakout"


class SDL(BaseModel):
    """Strategy Definition Language ticket — the single source of truth.

    All fields that drive execution (context, universe, legs, risk, gates,
    htf_rules) feed the content_hash.  Provenance, version counters, and
    timestamps are excluded from the hash so two SDL tickets with identical
    strategy logic but different provenance notes hash the same.
    """

    context: StrategyContext
    universe: Universe = Field(discriminator="type")
    legs: list[Leg]
    risk: RiskSpec = Field(default_factory=RiskSpec)
    gates: GatesSpec = Field(default_factory=GatesSpec)
    htf_rules: list[HTFRule] = Field(default_factory=list)

    # Provenance — mandatory
    provenance: Provenance = Field(default_factory=Provenance)

    # Versioning
    version: int = 1
    parent_version: int | None = None

    # Timestamps (set on creation/modification)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Content hash (computed from executable fields)
    content_hash: str = ""

    @model_validator(mode="after")
    def _compute_hash(self) -> "SDL":
        self.content_hash = _hash_executable(self)
        return self

    def bump_version(self) -> "SDL":
        """Return a new SDL with version incremented and parent_version set."""
        data = self.model_dump(mode="python")
        data["parent_version"] = self.version
        data["version"] = self.version + 1
        data["created_at"] = datetime.now(timezone.utc)
        data.pop("content_hash", None)
        return SDL(**data)


# ── Content hash ──────────────────────────────────────────────────────────────

def _hash_executable(sdl: SDL) -> str:
    """SHA-256 of the JSON-serialised executable fields (sorted keys).

    Excludes: provenance, version, parent_version, created_at, content_hash.
    """
    payload = {
        "context": sdl.context.model_dump(mode="json"),
        "universe": sdl.universe.model_dump(mode="json"),
        "legs": [_leg_dict(leg) for leg in sdl.legs],
        "risk": sdl.risk.model_dump(mode="json", exclude_none=True),
        "gates": sdl.gates.model_dump(mode="json", exclude_none=True),
        "htf_rules": [r.model_dump(mode="json", exclude_none=True) for r in sdl.htf_rules],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _leg_dict(leg: Leg) -> dict:
    return {
        "direction": leg.direction,
        "entry": {
            "trigger": leg.entry.trigger.model_dump(mode="json"),
            "filters": [f.model_dump(mode="json") for f in leg.entry.filters],
        },
        "exit": {
            "triggers": [t.model_dump(mode="json") for t in leg.exit.triggers],
            "filters": [f.model_dump(mode="json") for f in leg.exit.filters],
        },
    }
