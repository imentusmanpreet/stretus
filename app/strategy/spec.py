"""
app/strategy/spec.py — ``StrategySpec``, the strict strategy contract.

This Pydantic model is the single shape the LLM emits and the validator checks. It
captures the user's full intent — including rich, explainable risk-management
(every stop/target carries its ``type``, ``value``, ``source`` user|assumed,
``logic`` and ``reason``) — and renders to the YAML the quant engine loads, in
**formula mode only** (no KB signal registry).

Two layers, kept separate on purpose:
  * The rich ``stop_loss`` / ``take_profit`` / ``risk_management`` blocks preserve
    intent + reasoning for display, transparency, and future engine features.
  * :meth:`StrategySpec.to_engine_yaml_dict` REDUCES them to what the backtester
    runs today (a stop of type percent/atr/structural + a take-profit percent).
    Any reduction is recorded by flipping ``source`` to ``assumed`` / adding a note,
    never done silently.

``model_config = extra="forbid"`` everywhere — the model can never invent a field.
Engine-grammar and vocabulary checks (condition compiles, anchor is engine-legal,
timeframe is supported) live in :mod:`app.strategy.validator`, which reads the
engine live.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["user", "assumed"]


def _as_float(value: Any) -> float | None:
    """Coerce a number-or-string field to float, else None (e.g. '1.5' → 1.5)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%").strip())
    except (TypeError, ValueError):
        return None


# ── Rich risk-management blocks ───────────────────────────────────────────────

class StopLoss(BaseModel):
    """Stop-loss intent. ``value`` may be a number or a phrase (e.g. 'swing low')."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["percent", "atr", "structure", "fixed_points", "indicator_based"]
    value: float | str | None = None
    source: Source = "assumed"
    logic: str = ""
    reason: str = ""
    # Optional engine hints (used only when present; otherwise sensible defaults).
    anchor: str | None = None          # for structure (engine-legal anchor)
    window: int | None = Field(default=None, gt=0)
    padding_pct: float | None = Field(default=None, ge=0)

    def numeric(self) -> float | None:
        return _as_float(self.value)


class TakeProfit(BaseModel):
    """Take-profit intent. The engine consumes a percent; richer types reduce to it."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["percent", "risk_reward", "atr", "fixed_points", "indicator_based"]
    value: float | str | None = None
    source: Source = "assumed"
    logic: str = ""
    reason: str = ""

    def numeric(self) -> float | None:
        return _as_float(self.value)


class RiskManagement(BaseModel):
    """Portfolio-level risk controls. Fields are number-or-string for LLM tolerance."""

    model_config = ConfigDict(extra="forbid")

    risk_per_trade_pct: float | str | None = None
    daily_loss_cap_pct: float | str | None = None
    min_trade_value: float | str | None = None
    max_trades_per_day: int | str | None = None
    risk_reward: str | None = None
    notes: str = ""


class TrailingStop(BaseModel):
    """Optional trailing stop. ``type`` validated against the engine in the validator."""

    model_config = ConfigDict(extra="forbid")

    type: str  # "percent" | "atr" | "ema" | "chandelier"
    distance_pct: float | None = Field(default=None, gt=0)
    multiplier: float | None = Field(default=None, gt=0)
    window: int | None = Field(default=None, gt=0)
    activate_after_pct: float | None = Field(default=None, ge=0)

    def to_engine_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class IndicatorSpec(BaseModel):
    """A declared indicator + the EXACT parameters the user asked for (transparency).

    The engine derives periodic indicators (EMA/RSI/…) straight from the formula, and
    multi-parameter studies (MACD/STOCH/SUPERTREND/BB/PPO) run on its standard settings.
    This block records what was requested so the readback can show it — e.g.
    ``{"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class HtfRule(BaseModel):
    """One higher-timeframe gate: condition must hold on the last closed HTF bar."""

    model_config = ConfigDict(extra="forbid")

    timeframe: str
    condition: str


class Gates(BaseModel):
    """Optional Phase-10 entry gates. All fields optional; only set ones are emitted."""

    model_config = ConfigDict(extra="forbid")

    entry_window_start: str | None = None
    entry_window_end: str | None = None
    entry_window_tz: str | None = None
    cooldown_bars_after_loss: int | None = Field(default=None, ge=0)
    cooldown_bars_after_profit: int | None = Field(default=None, ge=0)
    max_consecutive_losses: int | None = Field(default=None, ge=0)
    gap_filter: str | None = None  # none|ignore_gap_up|ignore_gap_down|ignore_both
    gap_threshold_pct: float | None = Field(default=None, ge=0)
    entry_confirmation_bars: int | None = Field(default=None, ge=1)
    volume_ratio_threshold: float | None = Field(default=None, gt=0)
    rsi_entry_band_min: float | None = Field(default=None, ge=0, le=100)
    rsi_entry_band_max: float | None = Field(default=None, ge=0, le=100)
    vol_filter_metric: str | None = None  # atr|natr
    vol_filter_window: int | None = Field(default=None, gt=0)
    vol_filter_min: float | None = Field(default=None, ge=0)
    vol_filter_max: float | None = Field(default=None, ge=0)
    regime_filter_allowed: list[str] | None = None
    rs_filter_window: int | None = Field(default=None, gt=0)
    rs_filter_min_ratio: float | None = Field(default=None, gt=0)
    event_skip_dates: list[str] | None = None

    def to_engine_dict(self) -> dict[str, Any]:
        """Map set gate fields to the keys engine.loader reads (flat + nested)."""
        out: dict[str, Any] = {}
        if self.entry_window_start or self.entry_window_end:
            out["entry_window"] = {
                "start": self.entry_window_start,
                "end": self.entry_window_end,
                "timezone": self.entry_window_tz or "Asia/Kolkata",
            }
        for flat in (
            "max_consecutive_losses",
            "cooldown_bars_after_loss",
            "cooldown_bars_after_profit",
            "gap_filter",
            "gap_threshold_pct",
            "entry_confirmation_bars",
            "volume_ratio_threshold",
        ):
            val = getattr(self, flat)
            if val is not None:
                out[flat] = val
        if self.rsi_entry_band_min is not None or self.rsi_entry_band_max is not None:
            out["rsi_entry_band"] = {
                "min": self.rsi_entry_band_min,
                "max": self.rsi_entry_band_max,
            }
        if self.vol_filter_min is not None or self.vol_filter_max is not None:
            out["volatility_filter"] = {
                "metric": self.vol_filter_metric or "natr",
                "window": self.vol_filter_window or 14,
                "min": self.vol_filter_min,
                "max": self.vol_filter_max,
            }
        if self.regime_filter_allowed:
            out["regime_filter"] = {"allowed": list(self.regime_filter_allowed)}
        if self.rs_filter_window is not None:
            out["relative_strength_filter"] = {
                "window": self.rs_filter_window,
                "min_ratio": self.rs_filter_min_ratio if self.rs_filter_min_ratio is not None else 1.0,
            }
        if self.event_skip_dates:
            out["event_filter"] = {"skip_dates": list(self.event_skip_dates)}
        return out


# Engine fallbacks when the rich SL/TP type can't be expressed natively (recorded,
# never silent — the reduction is surfaced via the validator's notes).
_DEFAULT_SL_PCT = 2.0
_DEFAULT_TP_PCT = 4.0


class StrategySpec(BaseModel):
    """The complete strategy the LLM must emit. Renders to formula-mode engine YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str
    symbol: str
    market: str
    timeframe: str
    objective: Literal["intraday", "positional"]
    direction: Literal["long_only", "short_only", "both"]

    intent_summary: str = ""  # LLM's plain-words understanding of the user's intent

    # Conditions in the engine grammar (validated by app.strategy.validator).
    entry_condition: str
    exit_condition: str = ""
    short_entry_condition: str = ""
    short_exit_condition: str = ""

    # Rich, explainable risk management.
    stop_loss: StopLoss
    take_profit: TakeProfit
    risk_management: RiskManagement = Field(default_factory=RiskManagement)
    trailing_stop: TrailingStop | None = None

    indicators: list[IndicatorSpec] = Field(default_factory=list)
    reference_symbol: str | None = None
    htf_rules: list[HtfRule] = Field(default_factory=list)
    gates: Gates | None = None
    max_holding_candles: int | None = Field(default=None, gt=0)

    position_sizing_mode: Literal["fixed_fractional", "risk_based", "fixed_units"] = "fixed_fractional"
    max_capital_allocation_pct: float = Field(default=100.0, gt=0, le=100)

    # ── Risk reduction to the engine's runnable subset ────────────────────────

    def resolved_stop_loss_pct(self) -> float:
        """The legacy percent the engine always needs (>0), derived from stop_loss."""
        if self.stop_loss.type == "percent":
            v = self.stop_loss.numeric()
            if v and v > 0:
                return v
        return _DEFAULT_SL_PCT

    def resolved_take_profit_pct(self) -> float:
        """The percent the engine consumes, derived from take_profit (RR-aware)."""
        tp = self.take_profit
        if tp.type == "percent":
            v = tp.numeric()
            if v and v > 0:
                return v
        if tp.type == "risk_reward":
            rr = tp.numeric()
            if rr and rr > 0:
                return round(self.resolved_stop_loss_pct() * rr, 4)
        v = tp.numeric()
        if v and v > 0:
            return v
        return _DEFAULT_TP_PCT

    def stop_loss_engine_spec(self) -> dict[str, Any] | None:
        """The engine ``stop_loss`` spec dict, or None to use the plain percent."""
        sl = self.stop_loss
        if sl.type == "percent":
            return {"type": "percent", "pct": self.resolved_stop_loss_pct()}
        if sl.type == "atr":
            mult = sl.numeric() or 2.0
            spec = {"type": "atr", "multiplier": mult, "window": sl.window or 14}
            return spec
        if sl.type == "structure":
            anchor = sl.anchor or (
                "prev_n_bar_low" if self.direction != "short_only" else "prev_n_bar_high"
            )
            spec = {"type": "structural", "anchor": anchor, "window": sl.window or 5}
            if sl.padding_pct is not None:
                spec["padding_pct"] = sl.padding_pct
            return spec
        # fixed_points / indicator_based: engine has no native stop; fall back to the
        # plain percent (resolved_stop_loss_pct) — the validator flags the reduction.
        return None

    # ── Rendering to engine YAML ──────────────────────────────────────────────

    def to_engine_yaml_dict(self) -> dict[str, Any]:
        """The inner ``strategy:`` mapping engine.loader._build_strategy_config reads.

        Always formula mode: no ``entry_signals``/``exit_signals``/registry keys.
        """
        rm = self.risk_management
        profile: dict[str, Any] = {
            "objective": self.objective,
            "max_trades_per_day": int(_as_float(rm.max_trades_per_day) or 0),
        }
        daily_cap = _as_float(rm.daily_loss_cap_pct)
        if daily_cap is not None:
            profile["daily_loss_cap_pct"] = daily_cap

        strat: dict[str, Any] = {
            "name": self.name,
            "symbol": self.symbol,
            "market": self.market,
            "timeframe": self.timeframe,
            "objective": self.objective,
            "direction": self.direction,
            "profile": profile,
            "variables": {
                "TAKE_PROFIT_TARGET": self.resolved_take_profit_pct(),
                "STOP_LOSS_TARGET": self.resolved_stop_loss_pct(),
            },


            "risk_management": {
                "stop_loss_percent": self.resolved_stop_loss_pct(),
                "take_profit_percent": self.resolved_take_profit_pct(),
            },
            "entry_evaluation_mode": "formula",
            "exit_evaluation_mode": "formula",
            "position_sizing_mode": self.position_sizing_mode,
            "max_capital_allocation_pct": self.max_capital_allocation_pct,
            "per_trade_risk_pct": _as_float(rm.risk_per_trade_pct) or 2.0,
        }
        if self.exit_condition:
            strat["exit"] = {"condition": self.exit_condition}
        if self.short_entry_condition:
            strat["short_entry_condition"] = self.short_entry_condition
        if self.short_exit_condition:
            strat["short_exit_condition"] = self.short_exit_condition
        if self.max_holding_candles is not None:
            strat["max_holding_candles"] = self.max_holding_candles
        if self.reference_symbol:
            strat["reference_symbol"] = self.reference_symbol

        sl_spec = self.stop_loss_engine_spec()
        if sl_spec is not None:
            strat["stop_loss"] = sl_spec
        if self.trailing_stop is not None:
            strat["trailing_stop"] = self.trailing_stop.to_engine_dict()
        if self.htf_rules:
            strat["htf"] = [
                {"timeframe": r.timeframe, "condition": r.condition} for r in self.htf_rules
            ]
        if self.gates is not None:
            strat.update(self.gates.to_engine_dict())
        return strat

    def _indicators_engine_config(self) -> dict[str, Any]:
        """Engine ``indicators`` block for multi-parameter studies (MACD/STOCH/…).

        These carry their params dict so the engine runs them on the requested
        settings (e.g. ``MACD: {fast:12, slow:26, signal:9}``). Periodic indicators
        (EMA/RSI/ATR/…) are skipped — the engine derives those straight from the
        formula — so this block is additive and safe to merge.
        """
        out: dict[str, Any] = {}
        for ind in self.indicators:
            name = (ind.name or "").upper().strip()
            params = ind.params or {}
            if name and params and "period" not in params:
                out[name] = dict(params)
        return out

    @classmethod
    def json_schema_for_llm(cls) -> dict[str, Any]:
        """JSON Schema for the prompt / schema-constrained output."""
        return cls.model_json_schema()
