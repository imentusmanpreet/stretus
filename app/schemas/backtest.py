"""
Pydantic models for backtest trigger and result APIs.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Trigger request ───────────────────────────────────────────────────────────

class BacktestTriggerRequest(BaseModel):
    strategy_id: str
    from_utc: Optional[str] = None
    to_utc: Optional[str] = None
    interval: Optional[str] = None
    symbol: Optional[str] = None
    # Multi-asset: explicit list of symbols to backtest sequentially. When
    # non-empty, one backtest runs per symbol (each overriding the YAML symbol)
    # and the response is the MultiAssetBacktestResult wrapper. When empty/None,
    # falls back to [symbol or YAML symbol] — still wrapped as a length-1 result.
    symbols: Optional[list[str]] = None
    starting_balance: float = 10_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 2.0
    max_holding_candles: Optional[int] = None
    objective: str = "positional"       # 'intraday' | 'positional'
    daily_loss_cap_pct: float = 0.0     # 0 = disabled
    max_trades_per_day: int = 0         # 0 = unlimited
    # Indian market: Securities Transaction Tax
    stt_intraday_sell_pct: float = 0.025
    stt_delivery_pct: float = 0.1
    # 1-minute execution (Phase 11). Tri-state:
    #   None  → AUTO: on for any strategy timeframe coarser than 1m (default).
    #   True  → force on.
    #   False → force off (e.g. a quick/cheap run).
    # When on, the main (and reference) OHLCV is fetched at 1-minute resolution
    # and the quant engine resamples it up to the strategy timeframe for signals
    # while resolving fills / stop-loss / take-profit on the underlying minute
    # bars for accuracy. A 1m strategy resolves to off (1m IS its execution TF).
    intrabar_execution: Optional[bool] = None

    @field_validator("starting_balance")
    @classmethod
    def validate_starting_balance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("starting_balance must be greater than zero.")
        return value

    @field_validator("slippage_bps", "commission_bps")
    @classmethod
    def validate_costs(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Execution cost inputs cannot be negative.")
        return value

    @field_validator("max_holding_candles")
    @classmethod
    def validate_max_holding_candles(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("max_holding_candles must be greater than zero.")
        return value


class BacktestTriggerResponse(BaseModel):
    backtest_id: str
    strategy_id: str
    status: str
    message: str


# ── Trade entry / exit rationale ──────────────────────────────────────────────

class TradeEntryReason(BaseModel):
    """Structured explanation of *why* a trade opened.

    Captured by the simulator at fill time: the bar whose data matched the entry
    signal, the indicator values that made it match, the fill price + cost
    breakdown, and the gates the bar cleared. `summary` is a one-line narrative
    of all of the above; the remaining fields expose the same detail in a
    machine-readable form. extra="allow" so the engine can add detail without a
    breaking schema change.
    """
    model_config = ConfigDict(extra="allow")

    summary: str
    evaluation_mode: str = "formula"               # formula | registry
    condition: str = ""                            # the entry formula that fired
    signal_bar: Optional[dict[str, Any]] = None    # OHLCV + index/timestamp of matched bar
    entry_bar: Optional[dict[str, Any]] = None     # index/timestamp of the fill bar
    indicators: dict[str, Optional[float]] = Field(default_factory=dict)  # matched values at signal bar
    confirmation_bars_required: int = 1
    confirmation_bars_observed: int = 1
    fill: dict[str, Any] = Field(default_factory=dict)   # raw/effective price + cost breakdown
    initial_stop_price: Optional[float] = None
    gates_passed: list[str] = Field(default_factory=list)


class TradeExitReason(BaseModel):
    """Structured explanation of *why* a trade closed.

    `code` mirrors the short `exit_reason` enum (STOP_LOSS / TAKE_PROFIT / …);
    `trigger` holds the price/condition that fired the exit; `fill` holds the
    raw/effective price and cost breakdown. `summary` narrates the complete
    reason in one line.
    """
    model_config = ConfigDict(extra="allow")

    summary: str
    code: str                                      # matches BacktestTrade.exit_reason
    trigger: dict[str, Any] = Field(default_factory=dict)
    exit_bar: Optional[dict[str, Any]] = None
    fill: dict[str, Any] = Field(default_factory=dict)
    holding_candles: int = 0
    pnl_pct: float = 0.0
    pnl_inr: float = 0.0
    indicators: dict[str, Optional[float]] = Field(default_factory=dict)


# ── Individual trade ──────────────────────────────────────────────────────────

class BacktestTrade(BaseModel):
    """One simulated trade with full entry/exit detail and risk excursion data."""
    instrument: str
    entry_date: str
    exit_date: str
    side: str
    entry_price: float
    exit_price: float
    # pnl_pct and outcome_pct are the same value — both names preserved for compatibility
    outcome_pct: float
    pnl_pct: float
    pnl_abs: float          # fractional return (used for compounding)
    pnl_inr: float = 0.0   # absolute rupee P&L per unit
    exit_reason: str        # short machine code, e.g. STOP_LOSS / TAKE_PROFIT
    holding_candles: int
    holding_duration_days: float = 0.0
    entry_market_condition: str = "Unknown"       # Bull / Bear / Sideways
    max_adverse_excursion_pct: float = 0.0        # worst unrealised loss during trade
    max_favorable_excursion_pct: float = 0.0      # best unrealised gain during trade
    # Full "why" behind the trade — see TradeEntryReason / TradeExitReason.
    entry_reason: Optional[TradeEntryReason] = None
    exit_reason_detail: Optional[TradeExitReason] = None


# ── Core performance metrics ──────────────────────────────────────────────────

class BacktestMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Calendar / balance
    num_days: int
    starting_balance: float
    ending_balance: float

    # Return metrics
    total_return_pct: float = 0.0
    net_return_pct: float = 0.0
    gross_return_pct: float = 0.0
    annual_return: float = 0.0
    avg_daily_return: float = 0.0
    average_outcome_per_trade: float = 0.0
    irr_daily: float = 0.0
    irr_annualized: float = 0.0

    # Risk-adjusted return
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0

    # Trade activity
    total_trades: int = 0
    win_rate: float = 0.0
    average_holding_duration: float = 0.0
    trades_per_month: float = 0.0
    longest_losing_streak: int = 0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility_pct: float = 0.0
    downside_deviation_pct: float = 0.0
    var_95_pct: float = 0.0
    expected_shortfall_95_pct: float = 0.0

    # Worst drawdown period detail
    worst_drawdown_start_date: Optional[str] = None
    worst_drawdown_end_date: Optional[str] = None
    drawdown_duration_days: int = 0
    recovery_date: Optional[str] = None
    recovery_time_days: Optional[int] = None   # None = never recovered within the window

    # Monthly performance (embedded in metrics for convenience)
    monthly_performance: list[dict[str, Any]] = Field(default_factory=list)
    monthly_statistics: dict[str, Any] = Field(default_factory=dict)

    # Trade list
    backtest_trades: Optional[list[BacktestTrade]] = None


# ── Assessment ────────────────────────────────────────────────────────────────

class BacktestAssessment(BaseModel):
    overall_grade: str
    return_potential: str
    risk_profile: str
    drawdown_tolerance_required: str
    recommended_for: str
    notes: str = ""


# ── Monthly performance entry ─────────────────────────────────────────────────

class MonthlyPerformanceEntry(BaseModel):
    month: str                      # YYYY-MM
    strategy_return_pct: float
    trades_count: int


# ── Monthly statistics ────────────────────────────────────────────────────────

class MonthlyStatistics(BaseModel):
    model_config = ConfigDict(extra="allow")

    highest_monthly_gain_pct: float = 0.0
    lowest_monthly_gain_pct: float = 0.0
    monthly_performance_range_pct: float = 0.0
    return_vs_drawdown_efficiency: float = 0.0


# ── Market phase analysis entry ───────────────────────────────────────────────

class MarketPhaseEntry(BaseModel):
    time_period: str                # e.g. "2026 Q1"
    market_type: str                # Bull / Bear / Sideways
    market_phase: str               # Uptrend / Downtrend / Range-bound
    observed_alignment: str         # Strong / Moderate / Weak
    price_change_pct: float
    strategy_trades: int
    strategy_win_rate_pct: float
    strategy_return_pct: float
    description: str


# ── Backtest date range ───────────────────────────────────────────────────────

class BacktestDateRange(BaseModel):
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None
    num_days: int = 0

    model_config = ConfigDict(populate_by_name=True)


# ── Run config (stored with result for transparency) ─────────────────────────

class BacktestRunConfig(BaseModel):
    symbol: str
    interval: str
    from_utc: str
    to_utc: str
    starting_balance: float
    slippage_bps: float
    commission_bps: float
    max_holding_candles: Optional[int] = None
    objective: str = "positional"
    daily_loss_cap_pct: float = 0.0
    max_trades_per_day: int = 0
    stt_intraday_sell_pct: float = 0.025
    stt_delivery_pct: float = 0.1
    warm_up_candles: int = 0


# ── Full result payload ───────────────────────────────────────────────────────

class BacktestResultPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    backtest_ref_id: str
    strategy_name: Optional[str] = None
    backtest_date_range: Optional[dict[str, Any]] = None

    # Core metrics
    metrics: BacktestMetrics

    # Assessment with notes
    assessment: Optional[BacktestAssessment] = None

    # Monthly breakdown (also embedded in metrics for convenience)
    monthly_performance: list[dict[str, Any]] = Field(default_factory=list)
    monthly_statistics: dict[str, Any] = Field(default_factory=dict)

    # Quarterly market phase analysis
    market_phase_analysis: list[dict[str, Any]] = Field(default_factory=list)

    # Pass/fail
    failure_reason: str = ""
    pass_: bool = Field(alias="pass")

    # Config transparency
    config: Optional[BacktestRunConfig] = None
    strategy_type: str = "positional"

    # Diagnostic internals (engine debugging)
    condition_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    diagnostic_summary: dict[str, Any] = Field(default_factory=dict)


# ── Multi-asset summary + wrapper ─────────────────────────────────────────────

class AssetBacktestSummary(BaseModel):
    """One comparison row per asset for the multi-asset summary.

    The metrics are lifted verbatim from each asset's BacktestMetrics and the
    grade/labels from its BacktestAssessment; no new computation happens here.
    """
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    backtest_ref_id: str
    total_return_pct: float = 0.0   # metrics.net_return_pct (a.k.a. total_return_pct)
    annual_return: float = 0.0      # metrics.annual_return
    volatility_pct: float = 0.0     # metrics.volatility_pct
    max_drawdown: float = 0.0       # metrics.max_drawdown
    # Additional per-asset metrics for the comparison table (lifted from metrics)
    sharpe_ratio: float = 0.0       # metrics.sharpe_ratio
    win_rate: float = 0.0           # metrics.win_rate
    total_trades: int = 0           # metrics.total_trades
    # Assessment labels (lifted from assessment; "" when no assessment ran)
    overall_grade: str = ""         # assessment.overall_grade
    return_potential: str = ""      # assessment.return_potential
    risk_profile: str = ""          # assessment.risk_profile
    recommendation: str = ""        # assessment.recommended_for
    pass_: bool = Field(default=False, alias="pass")
    failure_reason: str = ""


class MultiAssetBacktestResult(BaseModel):
    """Always-returned wrapper around one-or-more single-asset backtests.

    `results[i]` is the unchanged single-asset BacktestResultPayload, so a single
    element renders exactly like today's standalone result. `summary` holds one
    AssetBacktestSummary row per asset for side-by-side comparison.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    strategy_name: Optional[str] = None
    backtest_date_range: Optional[dict[str, Any]] = None
    num_assets: int = 0
    results: list[BacktestResultPayload] = Field(default_factory=list)
    summary: list[AssetBacktestSummary] = Field(default_factory=list)


# ── API response wrapper ──────────────────────────────────────────────────────

class BacktestResultResponse(BaseModel):
    backtest_id: str
    strategy_id: str
    status: str
    backtest_result: Optional[BacktestResultPayload] = None
    error_message: Optional[str] = None
    error: Optional[dict] = None
    created_at: str
    completed_at: Optional[str] = None
