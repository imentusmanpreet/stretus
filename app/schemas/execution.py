"""
app/schemas/execution.py
─────────────────────────
Pydantic models for POST /api/v1/strategy/evaluate/execute.

Two request modes:
  Mode 1 (production)  — strategy_id + mode; all data fetched from DB/OMS.
  Mode 2 (direct)      — strategy_config + execution_state supplied inline.

One unified response shape returned in both cases.

Multi-asset notes
─────────────────
A strategy carries an explicit `asset_class` ("equity_cash" for NSE/BSE via
Upstox; "crypto_spot" for crypto pairs via Binance). This selects (a) the
canonical-symbol form used to resolve ref_data mappings, (b) the market-data
client, (c) the broker order-type / product-type / validity mapping, and
(d) the trading-window/circuit-limit semantics. Legacy clients that omit
`asset_class` keep the equity behaviour (default "equity_cash").
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from app.schemas.backtest import TradeEntryReason, TradeExitReason
from app.strategy.spec import UniverseSpec


# ── Enums ──────────────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    """
    Asset class for a strategy / instrument. The same canonical id is used by
    `ref_data.instruments.asset_class_id`, `app/kb/schemas.py` and the chat
    capabilities block (`ChatCapabilities.asset_classes`).
    """

    equity_cash = "equity_cash"
    crypto_spot = "crypto_spot"


class EvaluationMode(str, Enum):
    paper = "paper"
    live  = "live"


class ActionType(str, Enum):
    no_action       = "no_action"
    entry_created   = "entry_created"
    exit_triggered  = "exit_triggered"


class ExchangeOrderType(str, Enum):
    """
    Broker / venue *order type*. The set is intentionally the union of all
    supported venues — pickers (e.g. ``TradeManager._order_types_for``) decide
    which subset is valid for a given asset class.

    Indian retail broker family (Upstox, Kite, etc.):
      MARKET, LIMIT, SL, SL-M

    Binance Spot:
      MARKET, LIMIT, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT

    Note: this is not the same as `order_role` (which leg in the bracket).
    """

    # Common across venues
    market = "MARKET"
    limit  = "LIMIT"

    # Indian retail broker (Upstox / Kite style)
    sl     = "SL"
    sl_m   = "SL-M"

    # Binance Spot
    stop_loss          = "STOP_LOSS"
    stop_loss_limit    = "STOP_LOSS_LIMIT"
    take_profit        = "TAKE_PROFIT"
    take_profit_limit  = "TAKE_PROFIT_LIMIT"


class ProductType(str, Enum):
    """
    Margin / settlement product. Equity uses MIS/CNC (Indian broker norms);
    crypto spot uses SPOT (no margin product on a cash spot order).
    """

    mis  = "MIS"   # intraday (equity)
    cnc  = "CNC"   # delivery / positional (equity)
    spot = "SPOT"  # crypto spot — non-margin
    margin = "MARGIN"  # crypto margin — reserved for future use


class OrderValidity(str, Enum):
    """
    Time-in-force. Indian broker family supports DAY/IOC; Binance Spot supports
    GTC/IOC/FOK. The enum is the union; pickers reject unsupported combos.
    """

    day = "DAY"
    ioc = "IOC"
    gtc = "GTC"
    fok = "FOK"


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


class ExitBlock(BaseModel):
    """
    Exit-phase signals. `trigger` is optional because many strategies exit
    purely on SL/TP and never define a signal-based exit. When omitted, the
    evaluator skips the exit-signal check and relies on price-based exits only.
    """

    trigger: Optional[SignalRule] = None
    filters: List[SignalRule] = Field(default_factory=list)


class TrailingOrderSpec(BaseModel):
    """OMS contract for a trailing exit leg (see docs/TRAILING_TAKE_PROFIT_DESIGN.md §4).

    The eval engine only EMITS this on a bracket leg; the external OMS monitors
    price, ratchets the line, and fires the exit. Phase 1 is percent-only so the
    OMS can honour it identically to the backtest (parity). `distance_pct` is the
    give-back behind the running peak (long) / trough (short); `activate_after_pct`
    is the profit the trade must reach before trailing engages (0 = immediate).
    """

    enabled: bool = True
    basis: Literal["percent"] = "percent"
    distance_pct: float = Field(..., gt=0, description="Give-back % behind the running extreme.")
    activate_after_pct: float = Field(
        0.0, ge=0, description="Profit % before trailing engages (0 = immediate)."
    )
    # OMS semantics: trail behind the running tick high (long) / low (short), and
    # fire a MARKET exit when price reverses to the line. Fixed for Phase 1.
    anchor: Literal["running_extreme"] = "running_extreme"
    fire: Literal["market"] = "market"


class TpRungState(BaseModel):
    """Runtime state for one rung of a multi-TP ladder on an open position."""

    rung_index: int
    price: float = Field(..., description="Absolute target price for this rung.")
    exit_fraction: float = Field(..., gt=0, le=1, description="Fraction of original position to exit.")
    exit_qty: Quantity = Field(..., description="Absolute quantity to exit at this rung.")
    hit: bool = Field(False, description="True once this rung has been executed.")
    risk_action: str = Field("none", description="SL mutation after firing: none|move_sl_to_breakeven|move_sl_to_previous_tp")


class TpLadder(BaseModel):
    """Full take-profit ladder attached to an open position."""

    rungs: List[TpRungState]
    total_quantity: Quantity = Field(..., description="Full position size at entry.")


class SlTpConfig(BaseModel):
    type: Literal["percent"] = "percent"
    stop_loss_pct: float = Field(..., gt=0, le=50)
    # take_profit_pct may be 0 when a trailing_take_profit or multi_take_profit
    # supplies the profit exit. The validator enforces at least one profit exit.
    take_profit_pct: float = Field(0.0, ge=0, le=100)
    trailing_take_profit: Optional[TrailingOrderSpec] = None
    trailing_stop: Optional[TrailingOrderSpec] = None
    # Multi-TP ladder: list of rung specs. When set, take_profit_pct must be 0.
    multi_take_profit: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Multi-TP ladder rungs. Each: {type, value, exit_fraction, risk_action}. "
            "When set, take_profit_pct must be 0 and trailing_take_profit must be None."
        ),
    )

    @model_validator(mode="after")
    def _require_profit_exit_and_one_trail(self) -> "SlTpConfig":
        has_multi_tp = bool(self.multi_take_profit)
        if self.take_profit_pct <= 0 and self.trailing_take_profit is None and not has_multi_tp:
            raise ValueError(
                "take_profit_pct must be > 0 unless a trailing_take_profit or multi_take_profit is set."
            )
        if self.trailing_stop is not None and self.trailing_take_profit is not None:
            raise ValueError(
                "trailing_stop and trailing_take_profit are mutually exclusive."
            )
        if has_multi_tp and self.trailing_take_profit is not None:
            raise ValueError(
                "multi_take_profit and trailing_take_profit are mutually exclusive."
            )
        return self


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = Field(2.0, ge=0.1, le=10.0)
    max_open_positions: int = Field(3, ge=1, le=20)
    cash_reserve_pct: float = Field(0.10, ge=0.0, le=0.5)
    cooldown_bars: int = Field(5, ge=0)
    # Quote-currency amount: INR for equity_cash, USDT/USDC for crypto_spot.
    # Default mirrors the equity_cash baseline; asset-class-aware lookups in
    # `risk_execution_config_service` may override it at runtime.
    min_trade_value: float = Field(10.0, ge=0)


class GatesConfig(BaseModel):
    """
    Phase 10 entry-gate constraints — parity with the backtest simulator's
    entry gates so a strategy behaves identically in backtest and live eval.

    Every gate defaults to *disabled*, so existing strategies and requests
    that omit this block behave exactly as before.
    """

    direction: Literal["long_only", "short_only", "both"] = "both"
    entry_window_start: Optional[str] = Field(
        None, description="HH:MM in IST — no entries before this time."
    )
    entry_window_end: Optional[str] = Field(
        None, description="HH:MM in IST — no fresh entries after this time."
    )
    max_consecutive_losses: int = Field(
        0, ge=0, description="Block entries after N consecutive losing trades. 0 = disabled."
    )
    cooldown_bars_after_loss: int = Field(
        0, ge=0, description="Bars to wait after a losing trade before re-entry. 0 = disabled."
    )
    cooldown_bars_after_profit: int = Field(
        0, ge=0, description="Bars to wait after a winning trade before re-entry. 0 = disabled."
    )
    max_spread_bps: float = Field(
        0.0, ge=0.0, description="Reject entry if estimated spread exceeds this. 0 = disabled."
    )
    gap_filter: Literal["none", "ignore_gap_up", "ignore_gap_down", "ignore_both"] = "none"
    gap_threshold_pct: float = Field(
        0.5, ge=0.0, description="Gap size (% vs prev close) that triggers the gap_filter."
    )
    entry_confirmation_bars: int = Field(
        1, ge=1, description="Entry signal must hold True for N consecutive closed bars."
    )
    rsi_entry_band_min: Optional[float] = Field(None, ge=0, le=100)
    rsi_entry_band_max: Optional[float] = Field(None, ge=0, le=100)
    volume_ratio_threshold: Optional[float] = Field(
        None, gt=0, description="Require current volume >= N x 20-bar average."
    )
    # Phase 2 — volatility-band gate. Block entries when ATR/NATR is outside the
    # band (too low = dead market, too high = chaotic). Must match the backtest
    # simulator's vol_filter_* gate exactly.
    vol_filter_metric: Optional[Literal["atr", "natr"]] = Field(
        None, description="Volatility metric for the band gate. None = disabled."
    )
    vol_filter_window: int = Field(14, ge=1, description="Look-back for the ATR/NATR band gate.")
    vol_filter_min: Optional[float] = Field(None, ge=0, description="Block if metric < this.")
    vol_filter_max: Optional[float] = Field(None, ge=0, description="Block if metric > this.")
    # Phase 2 — regime gate. Only enter when the detected regime is in this list.
    regime_filter_allowed: Optional[List[str]] = Field(
        None, description="Allowed regimes: trending_up|trending_down|ranging|volatile. None=off."
    )
    # Phase 2 — relative-strength gate (needs REF_close in the df).
    rs_filter_window: Optional[int] = Field(None, ge=1, description="RS look-back in bars. None=off.")
    rs_filter_min_ratio: float = Field(1.0, description="Block if RS ratio < this (1.0 = must match ref).")
    # Phase 2 — event filter: skip new entries on these YYYY-MM-DD dates.
    event_skip_dates: Optional[List[str]] = Field(None, description="Blackout dates. None=off.")
    # Phase 2 — lunch-lull skip window (UTC minutes-of-day, inclusive). Both
    # required to activate. Matches the simulator's lunch_lull gate.
    lunch_lull_start_utc: Optional[int] = Field(None, ge=0, le=1439)
    lunch_lull_end_utc: Optional[int] = Field(None, ge=0, le=1439)


class StrategyConfigPayload(BaseModel):
    """Inline strategy configuration for Mode 2."""

    strategy_id: Optional[str] = None
    symbol: str = Field(
        ...,
        description=(
            "Strategy ticker. Equity: 'RELIANCE.NS' or 'RELIANCE'. "
            "Crypto: 'BTC_USDT' (KB canonical BASE_QUOTE form)."
        ),
    )
    timeframe: str = Field(..., description="e.g. 5m, 15m, 1h")
    strategy_type: Literal["intraday", "positional"] = "intraday"
    asset_class: AssetClass = Field(
        default=AssetClass.equity_cash,
        description=(
            "Asset class for this strategy. Selects ref_data canonical-symbol "
            "form, market-data adapter (Upstox vs Binance), and product/order "
            "type mappings. Defaults to equity_cash for legacy clients."
        ),
    )
    entry: EntryExitBlock
    exit: ExitBlock
    sl_tp: SlTpConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)


# ── Quantity type (asset-class-aware) ──────────────────────────────────────────
#
# Equity exchanges (NSE/BSE) require integer quantities. Crypto exchanges
# (Binance Spot) require fractional quantities (e.g. 0.00125 BTC). We model
# the field as a ``Decimal`` so it round-trips losslessly for crypto, while
# still accepting integer JSON input so existing equity payloads are unchanged.
#
# - Inbound: Pydantic accepts ``5`` (int), ``5.0`` (float, only for crypto
#   asset-classes via downstream validation), and ``"0.00125"`` (string).
# - Outbound: ``Decimal`` is serialised as a JSON *string* by Pydantic v2 to
#   preserve precision. Equity quantities will appear as ``"5"`` instead of
#   ``5``; downstream consumers that parse with a JSON-number-aware deserialiser
#   should still tolerate this, but breaking changes are documented here.

Quantity = Decimal


# ── Sub-schemas: Execution State ───────────────────────────────────────────────

class OpenPosition(BaseModel):
    position_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: Quantity = Field(
        ...,
        description=(
            "Order quantity. Equity = integer share count (e.g. 5). "
            "Crypto spot = fractional base-asset amount (e.g. 0.00125)."
        ),
    )
    entry_price: float
    entry_time: Optional[str] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    entry_bar_index: Optional[int] = None
    # Multi-TP ladder state. None = single-TP mode (legacy behaviour).
    # Rungs with hit=True have already been executed and are skipped by the evaluator.
    take_profit_ladder: Optional[TpLadder] = None


class ExecutionStatePayload(BaseModel):
    """Inline execution state for Mode 2."""

    available_margin: float = Field(..., ge=0)
    open_positions: List[OpenPosition] = Field(default_factory=list)
    bars_since_last_trade: int = Field(0, ge=0)
    capital: float = Field(100_000.0, ge=0)
    # Phase 10 — gate state. Needed by the consecutive-loss and cooldown gates,
    # which (unlike the other gates) cannot be derived from the candle history.
    consecutive_losses: int = Field(
        0, ge=0, description="Count of consecutive losing trades for this strategy."
    )
    last_trade_was_loss: Optional[bool] = Field(
        None,
        description="True if the most recent closed trade was a loss, False if a win, "
        "None if no trades yet. Drives the cooldown_bars_after_loss/profit gates.",
    )


# ── Request Models ─────────────────────────────────────────────────────────────

class UniverseBlock(BaseModel):
    """Dynamic-universe extension of the evaluate request (presence ⇒ portfolio evaluation).

    The per-member strategy BODY is the request's own ``strategy_config`` (its ``symbol`` is a
    placeholder, stamped per member). This block adds the PORTFOLIO-level inputs that have no
    meaning for a single symbol: the member set (omit to load the deployment's active members),
    the shared-capital pool, the concurrent-position cap (clamped to the platform ceiling), and
    every member's open positions. When present, the endpoint returns a
    :class:`~app.schemas.universe_execution.UniverseEvaluateResponse` instead of the per-symbol
    response — the static single-symbol path is unaffected.
    """

    deployment_id: str = Field(..., description="Identifier for this dynamic deployment (audit/logging).")
    # Member-resolution precedence: explicit `members` (override) → live `rule` (resolve now) →
    # the deployment's active members in the DB (cadence-driven).
    rule: Optional[UniverseSpec] = Field(
        None,
        description="A dynamic-universe selection RULE (source/rank/take). When set (and `members` "
        "is omitted), the endpoint resolves the universe LIVE at request time (top-N) and trades "
        "the result — a self-contained, stateless dynamic call (no table, no pre-seeding).",
    )
    members: Optional[List[str]] = Field(
        None,
        description="Explicit member symbols (override, for testing). Omit to resolve the `rule` "
        "live, or — if no rule — to load the deployment's active members from the DB.",
    )
    # OMS-owned state. Optional in the schema so we can FAIL CLOSED in live mode when the OMS
    # omits them (an evaluator must never assume capital/flat-positions for real money). Paper
    # may default. None ⇒ "not supplied" (distinct from an explicit empty list = "flat").
    total_capital: Optional[float] = Field(
        None, gt=0, description="Shared capital pool (from the OMS). Required for mode=live."
    )
    max_positions: int = Field(2, gt=0, description="Concurrent-position cap (clamped to the platform ceiling).")
    sizing_mode: Literal["equal_weight", "risk_based"] = "equal_weight"
    risk_per_trade_pct: float = Field(1.0, gt=0)
    open_positions: Optional[List[OpenPosition]] = Field(
        None,
        description="The deployment's open positions across ALL members (from the OMS). "
        "Required for mode=live; an explicit [] means flat.",
    )
    bars_since_last_trade: int = Field(
        0, ge=0,
        description="Bars elapsed since the deployment's last trade, used by the cooldown account "
        "gate (an entry is blocked while this is below the strategy's cooldown_bars, default 5). "
        "Production: supplied by the OMS. Testing: set high (e.g. 999) to bypass the cooldown.",
    )


class EvaluateExecuteRequest(BaseModel):
    """
    Unified request for POST /strategy/evaluate/execute.

    Static (single symbol): exactly one of (strategy_id) OR (strategy_config + execution_state).
    Dynamic (universe): strategy_config (the per-member body) + a ``universe`` block. The
    presence of ``universe`` selects the portfolio-evaluation path; everything else is unchanged.
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

    # Dynamic-universe extension (presence ⇒ portfolio evaluation; see UniverseBlock).
    universe: Optional[UniverseBlock] = Field(
        None,
        description="Dynamic-universe portfolio inputs. When set, the per-member body is "
        "'strategy_config' and the response is a portfolio-level UniverseEvaluateResponse.",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "EvaluateExecuteRequest":
        has_id = self.strategy_id is not None
        has_config = self.strategy_config is not None
        has_state  = self.execution_state is not None

        # ── Dynamic path: a universe block needs the per-member body in strategy_config.
        # execution_state is NOT required (open positions live on the universe block).
        if self.universe is not None:
            if not has_config:
                raise ValueError(
                    "'strategy_config' (the per-member strategy body) is required when a "
                    "'universe' block is provided."
                )
            if has_id:
                raise ValueError("Provide either 'strategy_id' OR a 'universe' block, not both.")
            return self

        # ── Static path (unchanged) ───────────────────────────────────────────
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
    quantity: Quantity = Field(
        ...,
        description=(
            "Order quantity. Equity = integer share count; crypto spot = "
            "fractional base-asset amount (e.g. 0.00125 BTC)."
        ),
    )
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    order_type: ExchangeOrderType = Field(
        ...,
        description=(
            "Broker / venue *order type*. The set is the union of all supported "
            "venues — equity legs use LIMIT/SL-M, crypto spot legs use "
            "LIMIT/STOP_LOSS/TAKE_PROFIT (or the *_LIMIT variants). Not the "
            "bracket leg role."
        ),
    )
    product_type: ProductType = Field(
        ...,
        description=(
            "Margin / settlement product: MIS or CNC for Indian equity per "
            "broker norms; SPOT for crypto spot (non-margin)."
        ),
    )
    validity: OrderValidity = Field(
        default=OrderValidity.day,
        description=(
            "Time-in-force. Equity orders use DAY/IOC; crypto spot orders use "
            "GTC/IOC/FOK. Distinct from how long a CNC *position* may be held."
        ),
    )
    trailing: Optional[TrailingOrderSpec] = Field(
        default=None,
        description=(
            "Present only on a trailing exit leg (stop or take-profit). The OMS "
            "owns this leg: it monitors price, ratchets the line, and fires the "
            "exit. When set on a take-profit leg the `order_type` is a trigger "
            "type (TAKE_PROFIT / SL-M), never a resting LIMIT."
        ),
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
    reason: Literal["stop_loss", "take_profit", "exit_signal", "time_based", "partial_take_profit"]
    exit_price: float
    quantity: Quantity
    product_type: ProductType
    exit_detail: Optional[TradeExitReason] = None


class RiskSnapshot(BaseModel):
    """Risk parameters used for this evaluation (for auditability)."""

    stop_loss_pct: float
    take_profit_pct: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    position_size: Optional[Quantity] = Field(
        None,
        description=(
            "Computed entry quantity. Decimal so it round-trips losslessly for "
            "fractional crypto sizes (e.g. 0.00125 BTC) and integer equity "
            "share counts alike."
        ),
    )
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
    multi_tp_bracket: Optional[dict] = Field(
        None,
        description=(
            "Populated instead of bracket_order when the strategy uses a multi-TP ladder. "
            "Contains entry_order, stop_loss_order, and take_profit_orders (list)."
        ),
    )
    exits: List[ExitInstruction] = Field(default_factory=list)
    risk_snapshot: Optional[RiskSnapshot] = None
    entry_detail: Optional[TradeEntryReason] = None
    messages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
