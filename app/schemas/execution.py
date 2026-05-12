"""
app/schemas/execution.py
─────────────────────────
Pydantic models for POST /api/v1/strategy/evaluate/execute.

Two request modes:
  Mode 1 (production)  — strategy_id + mode; all data fetched from DB/OMS.
  Mode 2 (direct)      — strategy_config + execution_state supplied inline.

One unified response shape returned in both cases.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class EvaluationMode(str, Enum):
    paper = "paper"
    live  = "live"


class ActionType(str, Enum):
    no_action       = "no_action"
    entry_created   = "entry_created"
    exit_triggered  = "exit_triggered"


class ExchangeOrderType(str, Enum):
    """
    NSE cash-segment / Indian retail broker *order type* (same family as
    Kite, Upstox, etc. `order_type` / `variety` payloads). Tells the exchange/OMS
    *how* the order works (price vs market vs stop).

    This is not the same as `order_role` (which leg in the bracket: entry, SL, TP).
    """

    market = "MARKET"
    limit  = "LIMIT"
    sl     = "SL"
    sl_m   = "SL-M"


class ProductType(str, Enum):
    mis = "MIS"   # intraday
    cnc = "CNC"   # delivery / positional


class OrderValidity(str, Enum):
    day = "DAY"
    ioc = "IOC"


class BracketOrderLegRole(str, Enum):
    """
    Which leg in the *bracket* (workflow). Distinct from `order_type` (ExchangeOrderType)
    and from `product_type` (MIS / CNC).
    """

    entry = "entry"
    stop_loss_exit = "stop_loss_exit"
    take_profit_exit = "take_profit_exit"


# ── Sub-schemas: Strategy Config ───────────────────────────────────────────────

class SignalRule(BaseModel):
    """A single trigger or filter rule referencing a stretus_kb signal."""

    type: str = Field(..., description="Signal name as registered in stretus_kb RuleRegistry")
    params: Dict[str, Any] = Field(default_factory=dict)


class EntryExitBlock(BaseModel):
    trigger: SignalRule
    filters: List[SignalRule] = Field(default_factory=list)


class SlTpConfig(BaseModel):
    type: Literal["percent"] = "percent"
    stop_loss_pct: float = Field(..., gt=0, le=50)
    take_profit_pct: float = Field(..., gt=0, le=100)


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = Field(2.0, ge=0.1, le=10.0)
    max_open_positions: int = Field(3, ge=1, le=20)
    cash_reserve_pct: float = Field(0.10, ge=0.0, le=0.5)
    cooldown_bars: int = Field(5, ge=0)
    min_trade_value: float = Field(500.0, ge=0)


class StrategyConfigPayload(BaseModel):
    """Inline strategy configuration for Mode 2."""

    strategy_id: Optional[str] = None
    symbol: str = Field(..., description="e.g. RELIANCE.NS")
    timeframe: str = Field(..., description="e.g. 5m, 15m, 1h")
    strategy_type: Literal["intraday", "positional"] = "intraday"
    entry: EntryExitBlock
    exit: EntryExitBlock
    sl_tp: SlTpConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)


# ── Sub-schemas: Execution State ───────────────────────────────────────────────

class OpenPosition(BaseModel):
    position_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    entry_price: float
    entry_time: Optional[str] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    entry_bar_index: Optional[int] = None


class ExecutionStatePayload(BaseModel):
    """Inline execution state for Mode 2."""

    available_margin: float = Field(..., ge=0)
    open_positions: List[OpenPosition] = Field(default_factory=list)
    bars_since_last_trade: int = Field(0, ge=0)
    capital: float = Field(100_000.0, ge=0)


# ── Request Models ─────────────────────────────────────────────────────────────

class EvaluateExecuteRequest(BaseModel):
    """
    Unified request for POST /strategy/evaluate/execute.

    Exactly one of (strategy_id) OR (strategy_config + execution_state)
    must be provided.
    """

    # Mode 1 fields
    strategy_id: Optional[str] = Field(
        None,
        description="UUID of a confirmed strategy. Triggers Mode 1 (DB-fetched config).",
    )
    mode: Optional[EvaluationMode] = Field(
        EvaluationMode.paper,
        description="paper = simulate decisions only; live = decisions sent to OMS",
    )

    # Mode 2 fields
    strategy_config: Optional[StrategyConfigPayload] = Field(
        None,
        description="Inline strategy config. Required for Mode 2.",
    )
    execution_state: Optional[ExecutionStatePayload] = Field(
        None,
        description="Inline execution state. Required for Mode 2.",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "EvaluateExecuteRequest":
        has_id = self.strategy_id is not None
        has_config = self.strategy_config is not None
        has_state  = self.execution_state is not None

        if not has_id and not has_config:
            raise ValueError(
                "Provide either 'strategy_id' (Mode 1) or "
                "'strategy_config' + 'execution_state' (Mode 2)."
            )
        if has_id and has_config:
            raise ValueError(
                "Provide either 'strategy_id' OR 'strategy_config', not both."
            )
        if has_config and not has_state:
            raise ValueError(
                "'execution_state' is required when 'strategy_config' is provided (Mode 2)."
            )
        return self


# ── Response Models ────────────────────────────────────────────────────────────

class OrderLeg(BaseModel):
    """One leg of a bracket; exit legs reference the entry via parent_order_id."""

    order_id: str = Field(..., description="UUID for this leg (client- or server-generated instruction id).")
    parent_order_id: Optional[str] = Field(
        None,
        description="Entry leg order_id; set on stop-loss and take-profit legs only.",
    )
    order_role: BracketOrderLegRole = Field(
        ...,
        description="Bracket leg: entry, stop_loss_exit, or take_profit_exit. Not the broker `order_type`.",
    )
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    order_type: ExchangeOrderType = Field(
        ...,
        description=(
            "Broker / exchange *order type* (LIMIT, MARKET, SL, SL-M) — not bracket leg. "
            "Use this when mapping to NSE cash-style or broker place-order APIs."
        ),
    )
    product_type: ProductType = Field(
        ...,
        description="Margin product: MIS (intraday) or CNC (delivery) per Indian broker norms.",
    )
    validity: OrderValidity = Field(
        default=OrderValidity.day,
        description="Time in force of this *order* (e.g. DAY = rest of session). See API docs; "
        "distinct from how long a CNC *position* may be held after fill.",
    )


class BracketOrder(BaseModel):
    """
    OCO / bracket set: one entry and two protective exits, plus idempotency for the OMS.
    """

    idempotency_key: str = Field(
        ...,
        description=(
            "Stable id for this decision (same strategy+symbol+bar -> same key). Use as REST "
            "`Idempotency-Key` (or OMS equivalent) so retries do not create duplicate child orders. "
            "The hash is a compact fingerprint, not a broker exchange order id."
        ),
    )
    entry_order: OrderLeg
    stop_loss_order: OrderLeg
    take_profit_order: OrderLeg
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extensibility. Prefer idempotency_key and leg fields for stable integrations.",
    )


class ExitInstruction(BaseModel):
    position_id: str
    symbol: str
    reason: Literal["stop_loss", "take_profit", "exit_signal", "time_based"]
    exit_price: float
    quantity: int
    product_type: ProductType


class RiskSnapshot(BaseModel):
    """Risk parameters used for this evaluation (for auditability)."""

    stop_loss_pct: float
    take_profit_pct: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    position_size: Optional[int] = None
    principal_amount: Optional[float] = None
    daily_loss_cap_pct: float
    per_trade_risk_pct: float


class EvaluateExecuteResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    action: ActionType = ActionType.no_action
    symbol: str
    ltp: Optional[float] = None
    mode: EvaluationMode = EvaluationMode.paper
    bracket_order: Optional[BracketOrder] = None
    exits: List[ExitInstruction] = Field(default_factory=list)
    risk_snapshot: Optional[RiskSnapshot] = None
    messages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
