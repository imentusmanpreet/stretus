"""
app/services/execution/strategy_evaluator.py
──────────────────────────────────────────────
Main orchestrator for POST /strategy/evaluate/execute.

Strict execution order (never skip a step):
  1.  Load strategy config      (from DB or request body)
  2.  Resolve Upstox key        (ref_data.equities → equity_data_source_mappings)
  3.  Fetch market data         (candles + LTP + circuit limits via Upstox API)
  4a. Build InstrumentDefaults  (tick/lot from ref_data.system_configs; circuits from live or pct default)
  4b. Fetch risk config + execution state from DB
  5.  EXIT PHASE  — for each open position check SL → TP → exit signal → time
  6.  ENTRY PHASE — evaluate entry signal → risk → account → bracket order

The evaluator is a STATELESS DECISION ENGINE.
It returns instructions to the OMS; it never submits orders itself.
"""
from __future__ import annotations

import logging
import types
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.execution import ExecutionState
from app.db.models.strategy import Strategy
from app.schemas.execution import (
    ActionType,
    EvaluateExecuteRequest,
    EvaluateExecuteResponse,
    EvaluationMode,
    ExecutionStatePayload,
    ExitInstruction,
    OpenPosition,
    RiskSnapshot,
    StrategyConfigPayload,
)
from app.services.execution.account_manager import AccountCheckInput, AccountManager
from app.services.execution.indicator_engine import compute_lookback
from app.services.execution.market_data_service import MarketDataService
from app.services.execution.ref_data_service import (
    InstrumentDefaults,
    lookup_instrument_defaults,
    lookup_upstox_instrument_key,
)
from app.services.execution.risk_execution_config_service import (
    RiskExecutionConfigSnapshot,
    resolve_active_risk_execution_config,
)
from app.services.execution.risk_manager import RiskInput, RiskManager
from app.services.execution.rule_engine import RuleEngine
from app.services.execution.trade_manager import TradeManager

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """
    Async orchestrator.  One instance per request (not a singleton).

    Usage:
        evaluator = StrategyEvaluator(db)
        response  = await evaluator.evaluate(request)
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._market = MarketDataService()
        self._rule   = RuleEngine()
        self._risk   = RiskManager()
        self._acct   = AccountManager()
        self._trade  = TradeManager()

    # ── Logging helper ────────────────────────────────────────────────────────

    @staticmethod
    def _emit(messages: List[str], msg: str, level: str = "info") -> None:
        """
        Append msg to the response messages list AND write it to the server log.
        Use this for every important decision step so the audit trail appears
        in both the API response body and the uvicorn/Docker stdout.
        """
        messages.append(msg)
        getattr(logger, level)(msg)

    # ══════════════════════════════════════════════════════════════════════════
    # Public entry point
    # ══════════════════════════════════════════════════════════════════════════

    async def evaluate(self, req: EvaluateExecuteRequest) -> EvaluateExecuteResponse:
        messages: List[str] = []

        try:
            # ── Step 1: Load strategy config ──────────────────────────────────
            strategy_cfg, exec_state_payload, strategy_id_str, mode, runtime_risk_config = (
                await self._load_strategy(req, messages)
            )

            symbol        = strategy_cfg.symbol
            timeframe     = strategy_cfg.timeframe
            strategy_type = strategy_cfg.strategy_type

            # ── Step 2: Resolve Upstox instrument key from ref_data ───────────
            upstox_key = await lookup_upstox_instrument_key(symbol, self._db)
            if upstox_key:
                self._emit(messages, f"🔑 Upstox key resolved from ref_data: {upstox_key}")
            else:
                self._emit(messages,
                    f"⚠️  No Upstox key in ref_data for {symbol} — falling back to NSE_EQ|TICKER.",
                    level="warning")

            # ── Step 3: Fetch market data (candles + LTP + circuit limits) ────
            all_rules = self._all_signal_rules(strategy_cfg)
            lookback  = compute_lookback(all_rules)
            self._emit(messages,
                f"📊 Fetching {lookback} candles for {symbol} ({timeframe}) "
                f"| signals: {len(all_rules)} rules, max_window={lookback}"
            )

            df             = await self._market.fetch_candles(symbol, timeframe, lookback, db_instrument_key=upstox_key)
            ltp            = await self._market.fetch_ltp(symbol, db_instrument_key=upstox_key)
            circuit_limits = await self._market.fetch_circuit_limits(symbol, db_instrument_key=upstox_key)

            self._emit(messages,
                f"📈 Market data | LTP=₹{ltp:.2f}  candles={len(df)}"
                + (f"  upper_circuit=₹{circuit_limits['upper_circuit']:.2f}"
                   f"  lower_circuit=₹{circuit_limits['lower_circuit']:.2f}"
                   if circuit_limits else "  circuit=system_defaults")
            )

            # ── Step 4a: Build instrument defaults from ref_data.system_configs ─
            instrument: InstrumentDefaults = await lookup_instrument_defaults(
                ltp=ltp,
                db=self._db,
                live_circuit_limits=circuit_limits,
            )
            self._emit(messages,
                f"⚙️  Instrument defaults | tick=₹{instrument.tick_size}  lot={instrument.lot_size}"
                f"  upper_circuit=₹{instrument.upper_circuit:.2f}"
                f"  lower_circuit=₹{instrument.lower_circuit:.2f}"
            )

            # ── Step 4b: Fetch execution state and log resolved config ────────
            db_exec_state = (
                await self._fetch_exec_state_row(strategy_id_str)
                if strategy_id_str
                else None
            )
            self._emit(
                messages,
                f"🗄️  Risk config (ref_data) | scope={runtime_risk_config.config_scope}"
                f"  SL={runtime_risk_config.stop_loss_pct}%"
                f"  TP={runtime_risk_config.take_profit_pct}%"
                f"  daily_cap={runtime_risk_config.daily_loss_cap}%"
                f"  per_trade_risk={runtime_risk_config.per_trade_risk}%",
            )
            if db_exec_state:
                self._emit(messages,
                    f"🗄️  Exec state (DB) | capital=₹{float(db_exec_state.capital):,.0f}"
                    f"  max_risk={db_exec_state.max_risk_per_trade_pct}%"
                    f"  max_open={db_exec_state.max_open_positions}"
                )
            else:
                self._emit(messages, "⚠️  No execution state row in DB — will use inline execution_state.", level="warning")

            # ── Build effective risk_config and exec_state ────────────────────
            # Mode 1 uses the resolved ref_data config directly.
            # Mode 2 uses the latest ref_data config as the baseline and overlays
            # any inline request values so runtime behavior never depends on
            # hardcoded fallbacks.
            effective_risk_config = (
                runtime_risk_config
                if req.strategy_id
                else _synthetic_risk_config(strategy_cfg, runtime_risk_config)
            )
            effective_exec_state  = db_exec_state  or _synthetic_exec_state(
                exec_state_payload, strategy_cfg, runtime_risk_config
            )

            if not req.strategy_id:
                self._emit(messages,
                    f"🗄️  Risk config | using inline values over ref_data baseline — "
                    f"SL={strategy_cfg.sl_tp.stop_loss_pct}%  "
                    f"TP={strategy_cfg.sl_tp.take_profit_pct}%"
                )
            if db_exec_state is None:
                self._emit(messages,
                    f"🗄️  Exec state | using inline values — "
                    f"capital=₹{exec_state_payload.capital:,.0f}  "
                    f"max_risk={strategy_cfg.risk.max_risk_per_trade_pct}%  "
                    f"min_trade=₹{strategy_cfg.risk.min_trade_value:,.0f}"
                )

            # Resolve account-level values (always from inline payload)
            available_margin      = exec_state_payload.available_margin
            open_positions        = exec_state_payload.open_positions
            bars_since_last_trade = exec_state_payload.bars_since_last_trade
            capital               = exec_state_payload.capital

            # ── Step 5: EXIT PHASE ────────────────────────────────────────────
            self._emit(messages,
                f"🚪 EXIT PHASE | checking {len(open_positions)} open position(s)"
            )
            exits = self._run_exit_phase(
                open_positions=open_positions,
                ltp=ltp,
                df=df,
                strategy_cfg=strategy_cfg,
                strategy_type=strategy_type,
                messages=messages,
            )

            if exits:
                self._emit(messages,
                    f"🚪 Exit triggered for {len(exits)} position(s) — returning exit_triggered."
                )
                return EvaluateExecuteResponse(
                    status="success",
                    action=ActionType.exit_triggered,
                    symbol=symbol,
                    ltp=ltp,
                    mode=mode,
                    exits=exits,
                    messages=messages,
                )

            # ── Step 6: ENTRY PHASE ───────────────────────────────────────────
            self._emit(messages,
                f"🚀 ENTRY PHASE | evaluating signal for {symbol} @ ₹{ltp:.2f}"
            )
            entry_signal, entry_msgs = self._rule.evaluate_entry(
                df, {"trigger": strategy_cfg.entry.trigger.model_dump(),
                     "filters": [f.model_dump() for f in strategy_cfg.entry.filters]}
            )
            for m in entry_msgs:
                self._emit(messages, m)

            if not entry_signal:
                self._emit(messages, "⛔ No entry signal fired — returning no_action.")
                return EvaluateExecuteResponse(
                    status="success",
                    action=ActionType.no_action,
                    symbol=symbol,
                    ltp=ltp,
                    mode=mode,
                    messages=messages,
                )

            # ── Risk calculation ───────────────────────────────────────────────
            self._emit(messages, "💰 RISK CALCULATION")
            risk_out = self._risk.calculate(
                RiskInput(
                    entry_price=ltp,
                    risk_config=effective_risk_config,
                    exec_state=effective_exec_state,
                    instrument=instrument,
                )
            )
            for m in risk_out.messages:
                self._emit(messages, m)

            if not risk_out.ok:
                self._emit(messages, "❌ Risk checks failed — no entry generated.", level="warning")
                return EvaluateExecuteResponse(
                    status="success",
                    action=ActionType.no_action,
                    symbol=symbol,
                    ltp=ltp,
                    mode=mode,
                    messages=messages,
                )

            # ── Account affordability check ────────────────────────────────────
            self._emit(messages, "🧾 ACCOUNT CHECK")
            max_open = int(getattr(effective_exec_state, "max_open_positions", strategy_cfg.risk.max_open_positions) or 3)
            cooldown = int(getattr(effective_exec_state, "cooldown_bars", strategy_cfg.risk.cooldown_bars) or 5)
            cash_res = float(getattr(effective_exec_state, "cash_reserve_pct", strategy_cfg.risk.cash_reserve_pct) or 0.10)

            acct_out = self._acct.validate(
                AccountCheckInput(
                    trade_value=risk_out.principal_amount,
                    available_margin=available_margin,
                    current_open_positions=len(open_positions),
                    bars_since_last_trade=bars_since_last_trade,
                    max_open_positions=max_open,
                    cash_reserve_pct=cash_res,
                    cooldown_bars=cooldown,
                    capital=capital,
                )
            )
            for m in acct_out.messages:
                self._emit(messages, m)

            if not acct_out.ok:
                self._emit(messages, "❌ Account checks failed — no entry generated.", level="warning")
                return EvaluateExecuteResponse(
                    status="success",
                    action=ActionType.no_action,
                    symbol=symbol,
                    ltp=ltp,
                    mode=mode,
                    messages=messages,
                )

            # ── Build bracket order ────────────────────────────────────────────
            last_bar_dt = str(df.index[-1]) if hasattr(df.index, "__iter__") else None
            bracket = self._trade.build_bracket_order(
                symbol=symbol,
                quantity=risk_out.position_size,
                entry_price=ltp,
                stop_loss_price=risk_out.stop_loss_price,
                take_profit_price=risk_out.take_profit_price,
                strategy_type=strategy_type,
                strategy_id=strategy_id_str,
                mode=mode.value,
                bar_datetime=last_bar_dt,
            )

            sl_pct  = float(getattr(effective_risk_config, "stop_loss_pct", strategy_cfg.sl_tp.stop_loss_pct))
            tp_pct  = float(getattr(effective_risk_config, "take_profit_pct", strategy_cfg.sl_tp.take_profit_pct))
            dlc_pct = float(getattr(effective_risk_config, "daily_loss_cap_pct", 3.0) or 3.0)
            ptr_pct = float(getattr(effective_risk_config, "per_trade_risk_pct", strategy_cfg.risk.max_risk_per_trade_pct))

            risk_snapshot = RiskSnapshot(
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                stop_loss_price=risk_out.stop_loss_price,
                take_profit_price=risk_out.take_profit_price,
                position_size=risk_out.position_size,
                principal_amount=risk_out.principal_amount,
                daily_loss_cap_pct=dlc_pct,
                per_trade_risk_pct=ptr_pct,
            )

            self._emit(messages,
                f"📦 BRACKET ORDER GENERATED | qty={risk_out.position_size}"
                f"  entry=₹{ltp:.2f}"
                f"  SL=₹{risk_out.stop_loss_price:.2f} (-{sl_pct}%)"
                f"  TP=₹{risk_out.take_profit_price:.2f} (+{tp_pct}%)"
                f"  principal=₹{risk_out.principal_amount:,.2f}"
            )

            return EvaluateExecuteResponse(
                status="success",
                action=ActionType.entry_created,
                symbol=symbol,
                ltp=ltp,
                mode=mode,
                bracket_order=bracket,
                risk_snapshot=risk_snapshot,
                messages=messages,
            )

        except Exception as exc:
            logger.error("💥 StrategyEvaluator error: %s", exc, exc_info=True)
            return EvaluateExecuteResponse(
                status="error",
                action=ActionType.no_action,
                symbol=getattr(req.strategy_config, "symbol", "unknown"),
                mode=req.mode or EvaluationMode.paper,
                messages=messages,
                error=str(exc),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Load strategy
    # ══════════════════════════════════════════════════════════════════════════

    async def _load_strategy(
        self,
        req: EvaluateExecuteRequest,
        messages: List[str],
    ) -> tuple[
        StrategyConfigPayload,
        ExecutionStatePayload,
        Optional[str],
        EvaluationMode,
        RiskExecutionConfigSnapshot,
    ]:
        """
        Returns (strategy_config, execution_state, strategy_id_str, mode, runtime_risk_config).

        Mode 1: fetches from DB; builds an inline ExecutionStatePayload from DB
                execution_state row + assumes no open positions (OMS not integrated yet).
        Mode 2: uses request body directly.
        """
        mode = req.mode or EvaluationMode.paper

        if req.strategy_id:
            # ── Mode 1: DB fetch ───────────────────────────────────────────────
            self._emit(messages, f"📋 Mode 1 | Loading strategy {req.strategy_id} from DB.")
            strategy_id = uuid.UUID(req.strategy_id)

            strategy_row = await self._db.get(Strategy, strategy_id)
            if strategy_row is None:
                raise ValueError(f"Strategy {req.strategy_id} not found in DB.")

            runtime_risk_config = await resolve_active_risk_execution_config(
                self._db,
                strategy_id=req.strategy_id,
            )
            raw_cfg = strategy_row.strategy_config or {}
            strategy_cfg = _strategy_config_from_db(
                raw_cfg,
                strategy_row,
                req.strategy_id,
                runtime_risk_config,
            )

            exec_row = await self._fetch_exec_state_row(req.strategy_id)
            exec_state = _exec_state_from_row(exec_row)
            self._emit(messages,
                f"✅ Strategy loaded | '{strategy_row.name}' ({strategy_row.symbol})"
                f"  timeframe={strategy_row.timeframe}  mode={mode.value}"
            )
            return strategy_cfg, exec_state, req.strategy_id, mode, runtime_risk_config

        else:
            # ── Mode 2: Inline ────────────────────────────────────────────────
            sym = getattr(req.strategy_config, "symbol", "?")
            tf  = getattr(req.strategy_config, "timeframe", "?")
            runtime_risk_config = await resolve_active_risk_execution_config(self._db)
            self._emit(messages,
                f"📋 Mode 2 | Inline strategy_config — symbol={sym}"
                f"  timeframe={tf}  mode={mode.value}"
            )
            return (
                req.strategy_config,  # type: ignore[return-value]
                req.execution_state,  # type: ignore[return-value]
                req.strategy_config.strategy_id,  # type: ignore[union-attr]
                mode,
                runtime_risk_config,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: EXIT PHASE
    # ══════════════════════════════════════════════════════════════════════════

    def _run_exit_phase(
        self,
        open_positions: List[OpenPosition],
        ltp: float,
        df,
        strategy_cfg: StrategyConfigPayload,
        strategy_type: str,
        messages: List[str],
    ) -> List[ExitInstruction]:
        """
        For each open position, check (in order):
          1. Stop loss hit
          2. Take profit hit
          3. Exit signal fires
          4. Time-based exit (max holding candles)

        Returns exit instructions for any that trigger.
        """
        exits: List[ExitInstruction] = []
        if not open_positions:
            self._emit(messages, "🚪 No open positions — exit phase skipped.")
            return exits

        for pos in open_positions:
            reason: Optional[str] = None
            entry_px = getattr(pos, "entry_price", None)
            pnl_str  = (
                f"  unrealised_PnL=₹{(ltp - entry_px) * getattr(pos, 'quantity', 1):+,.2f}"
                if entry_px else ""
            )
            self._emit(messages,
                f"🚪 Checking position {pos.position_id}"
                f"  qty={getattr(pos, 'quantity', '?')}"
                f"  entry=₹{entry_px or '?'}"
                f"  SL=₹{pos.stop_loss_price or '—'}"
                f"  TP=₹{pos.take_profit_price or '—'}"
                f"{pnl_str}"
            )

            # 1. Stop loss
            if pos.stop_loss_price is not None and self._rule.check_stop_loss(ltp, pos.stop_loss_price):
                reason = "stop_loss"
                self._emit(messages,
                    f"  🔴 STOP LOSS HIT | LTP=₹{ltp:.2f} ≤ SL=₹{pos.stop_loss_price:.2f}"
                )

            # 2. Take profit
            elif pos.take_profit_price is not None and self._rule.check_take_profit(ltp, pos.take_profit_price):
                reason = "take_profit"
                self._emit(messages,
                    f"  🟢 TAKE PROFIT HIT | LTP=₹{ltp:.2f} ≥ TP=₹{pos.take_profit_price:.2f}"
                )

            # 3. Exit signal
            else:
                exit_fired, exit_msgs = self._rule.evaluate_exit(
                    df,
                    {"trigger": strategy_cfg.exit.trigger.model_dump(),
                     "filters": [f.model_dump() for f in strategy_cfg.exit.filters]},
                )
                for m in exit_msgs:
                    self._emit(messages, m)
                if exit_fired:
                    reason = "exit_signal"

            if reason:
                exits.append(
                    self._trade.build_exit_instruction(
                        position=pos,
                        reason=reason,
                        exit_price=ltp,
                        strategy_type=strategy_type,
                    )
                )
            else:
                self._emit(messages, f"  ✅ No exit condition met for {pos.position_id} — hold.")

        return exits

    # ══════════════════════════════════════════════════════════════════════════
    # DB helpers
    # ══════════════════════════════════════════════════════════════════════════

    async def _fetch_exec_state_row(self, strategy_id_str: str) -> Optional[ExecutionState]:
        sid = uuid.UUID(strategy_id_str)
        result = await self._db.execute(
            select(ExecutionState).where(ExecutionState.strategy_id == sid)
        )
        return result.scalar_one_or_none()

    # ══════════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _all_signal_rules(cfg: StrategyConfigPayload) -> List[Dict[str, Any]]:
        """Collect all signal rule dicts for lookback calculation."""
        rules: List[Dict[str, Any]] = []
        rules.append(cfg.entry.trigger.model_dump())
        rules.extend(f.model_dump() for f in cfg.entry.filters)
        rules.append(cfg.exit.trigger.model_dump())
        rules.extend(f.model_dump() for f in cfg.exit.filters)
        return rules


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2 helpers: synthetic risk_config and exec_state from inline request
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_risk_config(
    cfg: StrategyConfigPayload,
    base_cfg: RiskExecutionConfigSnapshot,
) -> Any:
    """
    Build a SimpleNamespace that mimics a StrategyRiskConfig ORM row
    using the sl_tp and risk values from the inline strategy_config.

    Used in Mode 2 when no strategy_id is provided, so the risk manager
    receives the actual request values instead of hardcoded defaults.
    """
    return types.SimpleNamespace(
        stop_loss_pct=cfg.sl_tp.stop_loss_pct,
        take_profit_pct=cfg.sl_tp.take_profit_pct,
        daily_loss_cap_pct=base_cfg.daily_loss_cap,
        per_trade_risk_pct=cfg.risk.max_risk_per_trade_pct,
    )


def _synthetic_exec_state(
    exec_payload: ExecutionStatePayload,
    cfg: StrategyConfigPayload,
    base_cfg: RiskExecutionConfigSnapshot,
) -> Any:
    """
    Build a SimpleNamespace that mimics an ExecutionState ORM row
    using capital from the inline execution_state and limits from strategy_config.risk.

    Used in Mode 2 so the risk manager uses the request's capital (not ₹1,00,000 default)
    and the account manager uses the request's position/cooldown limits.
    """
    return types.SimpleNamespace(
        capital=exec_payload.capital if exec_payload.capital is not None else 100_000.0,
        max_risk_per_trade_pct=cfg.risk.max_risk_per_trade_pct,
        min_trade_value=(
            cfg.risk.min_trade_value
            if cfg.risk.min_trade_value is not None
            else base_cfg.minimum_trade_value
        ),
        max_open_positions=cfg.risk.max_open_positions,
        cash_reserve_pct=cfg.risk.cash_reserve_pct,
        cooldown_bars=cfg.risk.cooldown_bars,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build StrategyConfigPayload from DB row (Mode 1)
# ══════════════════════════════════════════════════════════════════════════════

def _strategy_config_from_db(
    raw: Dict[str, Any],
    strategy_row: Strategy,
    strategy_id: str,
    runtime_risk_config: RiskExecutionConfigSnapshot,
) -> StrategyConfigPayload:
    """
    Convert the strategy.strategy_config JSONB blob into a StrategyConfigPayload.

    The JSONB was written by the chat flow in a free-form shape.  We handle the
    common keys gracefully, falling back to defaults so Mode 1 never hard-fails
    on schema mismatches.
    """
    from app.schemas.execution import EntryExitBlock, RiskConfig, SignalRule, SlTpConfig

    def _rule(d: Any) -> SignalRule:
        if isinstance(d, dict):
            return SignalRule(type=d.get("type", ""), params=d.get("params", {}))
        return SignalRule(type=str(d), params={})

    def _block(b: Any) -> EntryExitBlock:
        if not isinstance(b, dict):
            return EntryExitBlock(trigger=SignalRule(type="", params={}))
        return EntryExitBlock(
            trigger=_rule(b.get("trigger", {})),
            filters=[_rule(f) for f in b.get("filters", [])],
        )

    entry_raw = raw.get("entry", {})
    exit_raw  = raw.get("exit", {})
    sl_tp_raw = raw.get("sl_tp", {})
    risk_raw  = raw.get("risk", {})

    sl_pct = float(
        sl_tp_raw.get("stop_loss_pct")
        or raw.get("stop_loss_pct")
        or runtime_risk_config.stop_loss_pct
    )
    tp_pct = float(
        sl_tp_raw.get("take_profit_pct")
        or raw.get("take_profit_pct")
        or runtime_risk_config.take_profit_pct
    )

    return StrategyConfigPayload(
        strategy_id=strategy_id,
        symbol=strategy_row.symbol,
        timeframe=strategy_row.timeframe,
        strategy_type=raw.get("strategy_type", "intraday"),
        entry=_block(entry_raw),
        exit=_block(exit_raw),
        sl_tp=SlTpConfig(stop_loss_pct=sl_pct, take_profit_pct=tp_pct),
        risk=RiskConfig(
            max_risk_per_trade_pct=float(
                risk_raw.get("max_risk_per_trade_pct", runtime_risk_config.per_trade_risk)
            ),
            max_open_positions=int(risk_raw.get("max_open_positions", 3)),
            cash_reserve_pct=float(risk_raw.get("cash_reserve_pct", 0.10)),
            cooldown_bars=int(risk_raw.get("cooldown_bars", 5)),
            min_trade_value=float(
                risk_raw.get("min_trade_value", runtime_risk_config.minimum_trade_value)
            ),
        ),
    )


def _exec_state_from_row(row: Optional[ExecutionState]) -> ExecutionStatePayload:
    """Build a baseline ExecutionStatePayload from a DB ExecutionState row."""
    if row is None:
        return ExecutionStatePayload(available_margin=100_000.0)

    snapshot = row.state_json or {}
    raw_positions = snapshot.get("open_positions", [])

    positions: List[OpenPosition] = []
    for p in raw_positions:
        try:
            positions.append(OpenPosition(**p))
        except Exception:
            pass

    return ExecutionStatePayload(
        available_margin=float(row.capital or 100_000.0),
        open_positions=positions,
        bars_since_last_trade=int(row.bars_since_last_trade or 0),
        capital=float(row.capital or 100_000.0),
    )
