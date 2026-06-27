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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Source = Literal["user", "assumed"]


def _none_to_empty_list(value: Any) -> Any:
    """Coerce ``None`` → ``[]`` for list fields. LLMs routinely emit ``"field": null`` for
    an empty list; without this the strict schema rejects the whole spec and burns a (slow)
    repair round. Non-None values pass through untouched for normal validation."""
    return [] if value is None else value


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


class RiskAction(BaseModel):
    """SL mutation to apply immediately after a partial-exit rung fires."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["move_sl_to_breakeven", "move_sl_to_previous_tp", "none"] = "none"


class TakeProfitTarget(BaseModel):
    """One rung of a scale-out take-profit ladder.

    ``type`` is how the LEVEL is expressed; ``value`` the magnitude (an R-multiple,
    a percent, or an ATR multiple). ``exit_fraction`` is the portion of the ORIGINAL
    position to close at this rung — ``None`` means "share the remainder equally"
    (resolved at engine/spec time). The engine computes an R-multiple level against
    the ACTUAL stop distance, so it works with ATR/structural/percent stops alike.
    ``risk_action`` optionally mutates the stop-loss when this rung fires.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["risk_reward", "percent", "atr"]
    value: float | str
    exit_fraction: float | None = Field(default=None, gt=0, le=1)
    window: int | None = Field(default=None, gt=0)
    source: Source = "assumed"
    reason: str = ""
    risk_action: RiskAction | None = None

    def numeric(self) -> float | None:
        return _as_float(self.value)


class TakeProfit(BaseModel):
    """Take-profit intent. The engine consumes a percent; richer types reduce to it.

    A single target is expressed via ``type``/``value``. A scale-out ladder (TP1, TP2,
    …) is expressed via ``targets``; when non-empty it supersedes ``type``/``value``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["percent", "risk_reward", "atr", "fixed_points", "indicator_based"]
    value: float | str | None = None
    source: Source = "assumed"
    logic: str = ""
    reason: str = ""
    # Scale-out ladder. Empty ⇒ single-target (type/value) path. When set, the
    # furthest rung also feeds the legacy percent fallback for old engine paths.
    targets: list[TakeProfitTarget] = Field(default_factory=list)
    # 1-based index of the rung after which the remainder's stop moves to break-even.
    # 0/None ⇒ OFF (default) — never inject a behaviour the user didn't ask for.
    move_stop_to_breakeven_after: int | None = Field(default=None, ge=0)

    _coerce_targets = field_validator("targets", mode="before")(_none_to_empty_list)

    def numeric(self) -> float | None:
        return _as_float(self.value)

    def resolved_targets(self) -> list[TakeProfitTarget]:
        """The ladder with ``exit_fraction`` filled in (equal split of the remainder).

        Any rung that left ``exit_fraction`` unset shares whatever fraction is left
        after the explicitly-sized rungs, split equally. The final total is clamped
        to ≤ 1.0; a leftover (<1.0) is closed later by the stop / exit-signal / EOD.
        """
        if not self.targets:
            return []
        explicit = sum(t.exit_fraction for t in self.targets if t.exit_fraction is not None)
        unset = [t for t in self.targets if t.exit_fraction is None]
        remaining = max(0.0, 1.0 - explicit)
        share = (remaining / len(unset)) if unset else 0.0
        out: list[TakeProfitTarget] = []
        for t in self.targets:
            frac = t.exit_fraction if t.exit_fraction is not None else share
            out.append(t.model_copy(update={"exit_fraction": round(frac, 6)}))
        return out


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

    @model_validator(mode="after")
    def _require_distance_for_percent(self) -> "TrailingStop":
        # distance_pct is the trail distance itself — without it a percent trailing
        # stop is undefined ("trail by how much?"). activate_after_pct only sets WHEN
        # trailing engages, not how far it trails. The engine loader hard-rejects this
        # later (`distance_pct must be > 0 for type=percent`); enforce it here at the
        # contract so the chat repair loop fixes it instead of crashing the backtest.
        if self.type == "percent" and self.distance_pct is None:
            raise ValueError(
                "trailing_stop with type='percent' requires distance_pct (how far behind "
                "the running peak/trough the stop trails). activate_after_pct only sets WHEN "
                "trailing starts, not how far it trails."
            )
        return self

    def to_engine_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class TrailingTakeProfit(BaseModel):
    """Optional trailing take-profit — a ratcheting give-back line on the PROFIT
    side. Once the trade is ``activate_after_pct`` in profit, the exit trails
    ``distance_pct`` behind the running peak (long) / trough (short) and fires on
    the pull-back. Phase 1 is percent-only (the validator enforces the engine
    type set). It may coexist with ``trailing_stop`` — the engine runs both lines
    and exits on whichever a reversal hits first (a staged trail); the validator
    only WARNS when one line is dominated and can never fire."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="percent",
        description="Phase 1 supports 'percent' only (validated against the engine).",
    )
    distance_pct: float | None = Field(
        default=None, gt=0,
        description="Give-back distance: how far (percent) behind the running peak (long) / "
        "trough (short) the take-profit trails. e.g. 2.0 = exit 2% off the peak.",
    )
    activate_after_pct: float | None = Field(
        default=None, ge=0,
        description="Profit percent the trade must reach before trailing engages. "
        "Omit for immediate. e.g. 5.0 = only start trailing once up 5%.",
    )

    @model_validator(mode="after")
    def _require_distance_for_percent(self) -> "TrailingTakeProfit":
        # distance_pct is the give-back distance itself — without it a percent trailing
        # take-profit is undefined. activate_after_pct only sets WHEN trailing engages.
        # The engine loader hard-rejects a missing distance later; enforce it here at
        # the contract so the chat repair loop fixes it instead of crashing the backtest.
        if self.type == "percent" and self.distance_pct is None:
            raise ValueError(
                "trailing_take_profit with type='percent' requires distance_pct (how far behind "
                "the running peak/trough the take-profit trails). activate_after_pct only sets WHEN "
                "trailing starts, not how far it trails."
            )
        return self

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


# ── Dynamic-universe blocks (KB-free; direct StrategySpec path) ───────────────
# A dynamic-universe strategy names a RULE for selecting instruments instead of a
# single ``symbol``. The rule is re-resolved on a cadence into a time-varying set of
# members traded under one shared capital pool. These models are LLM-emitted (never
# regex-extracted — honors the LLM-only rule) and are validated/clamped downstream:
#   * conditions in ``screen`` compile against the SAME engine grammar as
#     ``entry_condition`` (app.strategy.validator);
#   * ``take`` / ``max_positions`` express USER INTENT — the platform clamps the
#     *effect* at runtime in the resolver (DYNAMIC_UNIVERSE_MAX_ASSETS, §7.1).
# Nothing here imports the KB / planner (Invariant 11).

# Ranking metrics the resolver knows how to compute. A trailing ``*`` in the spec
# marks metrics that need feeds not yet wired (delivery/oi) — accepted by the model
# (intent preserved) and surfaced by the validator as an informational note.
UniverseRankBy = Literal[
    "rvol",            # relative volume — "most active"
    "volume",          # raw traded volume
    "pct_change",      # period return — "biggest movers" (gainers/losers via order)
    "rel_strength",    # strength vs a reference index
    "momentum",        # price momentum over a window
    "atr_pct",         # ATR as % of price — "most volatile"
    "distance_52w",    # distance to the 52-week extreme — "near highs"
    "rsi",             # RSI level — "most oversold/overbought"
    "delivery_volume", # delivery quantity (needs a delivery feed)
    "oi_change",       # open-interest change (needs an OI feed)
]


class UniverseSource(BaseModel):
    """Where the candidate pool comes from — the KB-free universe-source registry.

    ``kind`` selects the registry resolver (app.services.universe.sources):
      * ``index`` / ``sector`` / ``f_and_o`` → point-in-time constituents
        (``name`` identifies the index/sector, e.g. ``"NIFTY500"`` / ``"banking"``);
      * ``crypto_all`` → the exchange's tradable-pairs catalog;
      * ``watchlist`` → the explicit ``symbols`` list.
    No field here references the knowledge base.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["index", "sector", "f_and_o", "crypto_all", "watchlist"]
    name: str | None = None          # index/sector identifier (e.g. "NIFTY500")
    symbols: list[str] = Field(default_factory=list)  # watchlist members

    _coerce_symbols = field_validator("symbols", mode="before")(_none_to_empty_list)


class UniverseRank(BaseModel):
    """How candidates that pass the screen are ordered before take-N.

    ``by`` is the ranking metric; ``order`` is ``desc`` (top, the usual case) or
    ``asc`` (bottom — e.g. biggest losers, most oversold). ``window`` is the lookback
    in bars for window-based metrics (momentum/atr_pct/distance_52w/rel_strength).
    """

    model_config = ConfigDict(extra="forbid")

    by: UniverseRankBy
    order: Literal["desc", "asc"] = "desc"
    window: int | None = Field(default=None, gt=0)


class UniverseEligibility(BaseModel):
    """Fail-closed tradability filter applied BEFORE ranking (Invariant 5).

    A candidate that cannot be confidently shown to be tradable is EXCLUDED — stale
    or missing input never admits by default. ``min_adv`` is an average-daily-VALUE
    floor (price×volume) computed from OHLCV (no vendor feed); ``adv_window`` its
    lookback in bars (defaults to the platform setting when unset).
    """

    model_config = ConfigDict(extra="forbid")

    min_adv: float | None = Field(default=None, ge=0)
    adv_window: int | None = Field(default=None, gt=0)
    require_tradable: bool = True
    exclude_in_circuit: bool = True


class UniverseBreadthGate(BaseModel):
    """Optional market-context gate evaluated ONCE per refresh over the Tier-A pool.

    When the gate fails (or its input is stale) the refresh admits NO new members —
    fail-closed (Invariant 5). ``metric`` selects the breadth measure; ``min_ratio``
    is the threshold the measure must clear for the gate to pass.
    """

    model_config = ConfigDict(extra="forbid")

    metric: Literal[
        "advance_decline", "new_high_new_low", "sector_strength", "market_trend"
    ]
    min_ratio: float | None = Field(default=None, ge=0)


class UniverseRefresh(BaseModel):
    """How often membership is re-resolved.

    ``cadence`` is the coarse rhythm; ``at`` is an optional wall-clock time
    (``"HH:MM"``, exchange tz) for the daily/intraday refresh (e.g. the canonical
    "one hour after open" → ``"10:15"``).
    """

    model_config = ConfigDict(extra="forbid")

    cadence: Literal["daily", "weekly", "intraday"] = "daily"
    at: str | None = None


class UniverseSpec(BaseModel):
    """The full dynamic-universe rule: source → screen → eligibility → rank → take-N.

    Mutually exclusive with ``StrategySpec.symbol`` (the one-of validator enforces it).
    The strategy body (``to_engine_template_dict``) is symbol-agnostic and stamped onto
    each resolved member; this block is the selection rule that produces those members.

    ``take`` / ``max_positions`` are USER INTENT — the resolver clamps the realised
    count to the platform cap at runtime (§7.1), so saved strategies need no re-validation
    when the cap changes.
    """

    model_config = ConfigDict(extra="forbid")

    source: UniverseSource
    # Boolean screen conditions in the engine grammar (same compiler as entry_condition),
    # evaluated on daily (or scan_timeframe) bars up to `asof`. A candidate must pass ALL.
    screen: list[str] = Field(default_factory=list)
    rank: UniverseRank
    take: int = Field(default=10, gt=0)        # how many members the USER wants
    _coerce_screen = field_validator("screen", mode="before")(_none_to_empty_list)
    max_positions: int | None = Field(default=None, gt=0)  # concurrent cap; None ⇒ take
    eligibility: UniverseEligibility = Field(default_factory=UniverseEligibility)
    breadth: UniverseBreadthGate | None = None
    refresh: UniverseRefresh = Field(default_factory=UniverseRefresh)
    # Timeframe the screen/rank metrics evaluate on (defaults to daily). The strategy's
    # own `timeframe` still governs execution (Tier B); this is the Tier-A screen TF.
    scan_timeframe: str | None = None
    # What happens to a member that drops out of the universe at a refresh: flatten
    # immediately (default, intraday-style) or hold the open position until its own exit.
    removal_policy: Literal["flatten", "hold_until_exit"] = "flatten"

    def effective_max_positions(self, *, cap: int | None = None) -> int:
        """Concurrent-position cap honoring intent + platform ceiling.

        = min(max_positions or take, take[, cap]). The platform cap is applied by the
        resolver (§7.1); pass it here only when reducing in one place.
        """
        base = self.max_positions if self.max_positions is not None else self.take
        candidates = [base, self.take]
        if cap is not None:
            candidates.append(cap)
        return max(1, min(candidates))


# Engine fallbacks when the rich SL/TP type can't be expressed natively (recorded,
# never silent — the reduction is surfaced via the validator's notes).
_DEFAULT_SL_PCT = 2.0
_DEFAULT_TP_PCT = 4.0


class StrategySpec(BaseModel):
    """The complete strategy the LLM must emit. Renders to formula-mode engine YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Exactly ONE of `symbol` / `universe` is set (the one-of validator enforces it).
    # `symbol` → the static single-instrument path (unchanged). `universe` → the
    # dynamic-universe path: a selection RULE re-resolved on a cadence (§3, §7).
    symbol: str | None = None
    universe: UniverseSpec | None = None
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
    # Optional: omit (None) when a trailing_take_profit provides the profit exit
    # instead. The validator requires at least one of the two. A static target set
    # at/below a trailing activation level would fire first and starve the trail —
    # see the strategy prompt's HARD RULE 5.
    take_profit: TakeProfit | None = None
    risk_management: RiskManagement = Field(default_factory=RiskManagement)
    trailing_stop: TrailingStop | None = None
    trailing_take_profit: TrailingTakeProfit | None = None

    indicators: list[IndicatorSpec] = Field(default_factory=list)
    reference_symbol: str | None = None
    htf_rules: list[HtfRule] = Field(default_factory=list)
    gates: Gates | None = None
    max_holding_candles: int | None = Field(default=None, gt=0)

    position_sizing_mode: Literal["fixed_fractional", "risk_based", "fixed_units"] = "fixed_fractional"
    max_capital_allocation_pct: float = Field(default=100.0, gt=0, le=100)

    # ── Instrument selection: static symbol XOR dynamic universe ──────────────

    _coerce_lists = field_validator(
        "indicators", "htf_rules", mode="before"
    )(_none_to_empty_list)

    @model_validator(mode="after")
    def _exactly_one_instrument_source(self) -> "StrategySpec":
        """Enforce the one-of contract: a spec names a single ``symbol`` OR a
        ``universe`` rule, never both and never neither.

        This keeps the static path identical (every existing spec sets ``symbol``
        and no ``universe``) while making the dynamic path explicit and unambiguous.
        Pydantic ``extra='forbid'`` plus this check is the whole gate on which path
        a spec takes.
        """
        has_symbol = bool(self.symbol and self.symbol.strip())
        has_universe = self.universe is not None
        if has_symbol and has_universe:
            raise ValueError(
                "A strategy names EITHER a single `symbol` OR a `universe` rule, "
                "not both. Drop one: set `symbol` for a single instrument, or "
                "`universe` for a dynamic, rule-selected set."
            )
        if not has_symbol and not has_universe:
            raise ValueError(
                "A strategy must name an instrument: set `symbol` for a single "
                "instrument, or a `universe` rule for a dynamic, rule-selected set."
            )
        return self

    @property
    def is_dynamic(self) -> bool:
        """True when this spec uses the dynamic-universe path (no fixed symbol)."""
        return self.universe is not None

    # ── Risk reduction to the engine's runnable subset ────────────────────────

    def resolved_stop_loss_pct(self) -> float:
        """The legacy percent the engine always needs (>0), derived from stop_loss."""
        if self.stop_loss.type == "percent":
            v = self.stop_loss.numeric()
            if v and v > 0:
                return v
        return _DEFAULT_SL_PCT

    def resolved_take_profit_pct(self) -> float:
        """The static-target percent the engine consumes, derived from take_profit
        (RR-aware). Returns 0.0 when there is no static target (take_profit omitted) —
        the engine then relies on the trailing_take_profit / exit conditions for the
        profit exit, and emits NO static take-profit."""
        tp = self.take_profit
        if tp is None:
            return 0.0
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

        Static path: renders against ``self.symbol`` (byte-for-byte unchanged). For a
        dynamic-universe spec (no fixed symbol) the per-member body is produced by
        :meth:`to_engine_template_dict` instead — calling this without a symbol is a
        programming error and raises.
        """
        if not (self.symbol and self.symbol.strip()):
            raise ValueError(
                "to_engine_yaml_dict() needs a fixed `symbol`; this is a dynamic-"
                "universe spec — render each member via to_engine_template_dict(symbol)."
            )
        return self._render_engine_strategy(self.symbol)

    def to_engine_template_dict(self, member_symbol: str) -> dict[str, Any]:
        """The symbol-agnostic strategy body stamped onto one resolved universe member.

        Identical to :meth:`to_engine_yaml_dict` except the instrument is supplied by
        the caller (the resolved member) rather than ``self.symbol``. This is the
        "strategy template" insight (§3.1): only the ``symbol`` line differs per member;
        the rest of the body is shared. Reuses the SAME renderer so backtest/live and
        static/dynamic can never drift.
        """
        return self._render_engine_strategy(member_symbol)

    def _render_engine_strategy(self, symbol: str) -> dict[str, Any]:
        """Render the inner ``strategy:`` mapping for a given instrument symbol.

        The single source of truth for the engine body, shared by the static path
        (``to_engine_yaml_dict``) and the dynamic per-member path
        (``to_engine_template_dict``) so they cannot diverge (Invariant 1).
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
            "symbol": symbol,
            "market": self.market,
            "timeframe": self.timeframe,
            "objective": self.objective,
            "direction": self.direction,
            "profile": profile,
            "variables": {
                "TAKE_PROFIT_TARGET": self.resolved_take_profit_pct(),
                "STOP_LOSS_TARGET": self.resolved_stop_loss_pct(),
            },
            "entry": {"condition": self.entry_condition},
            # Multi-param studies (MACD/STOCH/…) carry their params here; periodic
            # indicators are auto-derived from the formula by the engine.
            "indicators": self._indicators_engine_config(),
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
        # Multi-TP ladder — emitted when the spec carries a scale-out targets list.
        # Zeroing the scalar take_profit_percent prevents the engine from firing a
        # static target before any rung exits occur.
        if self.take_profit is not None and self.take_profit.targets:
            resolved = self.take_profit.resolved_targets()
            strat["risk_management"]["multi_take_profit"] = [
                {
                    "type": t.type,
                    "value": float(t.numeric() or 0),
                    "exit_fraction": float(t.exit_fraction or 0),
                    **({"risk_action": t.risk_action.type} if t.risk_action else {}),
                }
                for t in resolved
            ]
            strat["risk_management"]["take_profit_percent"] = 0.0
            strat["variables"]["TAKE_PROFIT_TARGET"] = 0.0
        if self.trailing_stop is not None:
            strat["trailing_stop"] = self.trailing_stop.to_engine_dict()
        if self.trailing_take_profit is not None:
            strat["trailing_take_profit"] = self.trailing_take_profit.to_engine_dict()
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
