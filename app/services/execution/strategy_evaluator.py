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

from app.core.tenant_context import get_current_tenant
from app.db.models.execution import ExecutionState
from app.db.models.strategy import Strategy
from app.schemas.backtest import TradeEntryReason, TradeExitReason
from app.schemas.execution import (
    ActionType,
    AssetClass,
    EvaluateExecuteRequest,
    EvaluateExecuteResponse,
    EvaluationMode,
    ExecutionStatePayload,
    ExitInstruction,
    GatesConfig,
    OpenPosition,
    RiskSnapshot,
    StrategyConfigPayload,
    TrailingOrderSpec,
)
from app.services.execution.account_manager import AccountCheckInput, AccountManager
from app.services.execution.entry_gates import evaluate_entry_gates
from app.services.execution.indicator_engine import compute_lookback
from app.services.execution.market_data_service import MarketDataService
from app.services.execution.ref_data_service import (
    InstrumentDefaults,
    lookup_adapter_symbol,
    lookup_instrument_defaults,
)
from app.services.execution.risk_execution_config_service import (
    RiskExecutionConfigSnapshot,
    resolve_active_risk_execution_config,
)
from app.services.execution.risk_manager import RiskInput, RiskManager
from app.services.execution.rule_engine import RuleEngine
from app.services.execution.trade_manager import TradeManager

logger = logging.getLogger(__name__)

_OHLCV_KEYS = {"open", "high", "low", "close", "volume",
               "Open", "High", "Low", "Close", "Volume"}


def _extract_last_bar(df) -> tuple[dict, dict]:
    """Return (signal_bar_ohlcv, indicators) from the last candle of df."""
    last_row = df.iloc[-1]
    signal_bar = {
        "datetime": str(df.index[-1]),
        "open":   float(last_row.get("open",  last_row.get("Open",  0)) or 0),
        "high":   float(last_row.get("high",  last_row.get("High",  0)) or 0),
        "low":    float(last_row.get("low",   last_row.get("Low",   0)) or 0),
        "close":  float(last_row.get("close", last_row.get("Close", 0)) or 0),
        "volume": float(last_row.get("volume",last_row.get("Volume",0)) or 0),
        "index":  len(df) - 1,
    }
    indicators: dict = {}
    for col, val in last_row.items():
        if col in _OHLCV_KEYS:
            continue
        try:
            fval = float(val)
            indicators[col] = None if (fval != fval) else fval  # NaN → None
        except (TypeError, ValueError):
            pass
    return signal_bar, indicators


def _parse_trailing_order_spec(raw: Any) -> Optional[TrailingOrderSpec]:
    """Map a persisted trailing spec ({type, distance_pct, activate_after_pct}) to
    the OMS-facing TrailingOrderSpec. Percent-only (Phase 1); returns None when
    absent or unparseable so trailing stays an additive, fail-safe feature."""
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        distance = float(raw.get("distance_pct"))
    except (TypeError, ValueError):
        return None
    if distance <= 0:
        return None
    try:
        activate = float(raw.get("activate_after_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        activate = 0.0
    return TrailingOrderSpec(distance_pct=distance, activate_after_pct=max(0.0, activate))


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
        tenant = get_current_tenant()
        if tenant is not None:
            self._emit(
                messages,
                f"🏢 tenant | id={tenant.tenant_id} code={tenant.tenant_code or '-'} "
                f"legacy={tenant.is_legacy} adapters={sorted(tenant.adapters.keys())}",
            )
            logger.info("tenant_config|event=evaluate|tenant_id=%s|legacy=%s", tenant.tenant_id, tenant.is_legacy)

        try:
            # ── Step 1: Load strategy config ──────────────────────────────────
            strategy_cfg, exec_state_payload, strategy_id_str, mode, runtime_risk_config = (
                await self._load_strategy(req, messages)
            )

            symbol        = strategy_cfg.symbol
            timeframe     = strategy_cfg.timeframe
            strategy_type = strategy_cfg.strategy_type
            asset_class   = strategy_cfg.asset_class
            currency_sym  = "$" if asset_class == AssetClass.crypto_spot else "₹"

            # ── Step 2: Resolve venue-native adapter symbol from ref_data ─────
            adapter_symbol = await lookup_adapter_symbol(
                symbol, asset_class, self._db,
            )
            if adapter_symbol:
                self._emit(
                    messages,
                    f"🔑 ref_data adapter symbol | asset_class={asset_class.value} "
                    f"symbol={symbol} → {adapter_symbol}",
                )
            else:
                self._emit(
                    messages,
                    f"⚠️  No ref_data mapping for asset_class={asset_class.value} "
                    f"symbol={symbol} — market-data client will fall back to "
                    f"best-effort derivation.",
                    level="warning",
                )

            # ── Step 3: Fetch market data (candles + LTP + circuit limits) ────
            all_rules = self._all_signal_rules(strategy_cfg)
            lookback  = compute_lookback(all_rules)
            self._emit(
                messages,
                f"📊 Fetching {lookback} candles for {symbol} ({timeframe}) "
                f"asset_class={asset_class.value} | signals={len(all_rules)} "
                f"rules={[r.get('type') for r in all_rules]} "
                f"params={[r.get('params') for r in all_rules]}",
            )

            df             = await self._market.fetch_candles(
                symbol, timeframe, lookback,
                db_instrument_key=adapter_symbol,
                asset_class=asset_class,
            )
            ltp            = await self._market.fetch_ltp(
                symbol,
                db_instrument_key=adapter_symbol,
                asset_class=asset_class,
            )
            circuit_limits = await self._market.fetch_circuit_limits(
                symbol,
                db_instrument_key=adapter_symbol,
                asset_class=asset_class,
            )

            self._emit(
                messages,
                f"📈 Market data | LTP={currency_sym}{ltp:.4f}  candles={len(df)}"
                + (
                    f"  upper_band={currency_sym}{circuit_limits['upper_circuit']:.4f}"
                    f"  lower_band={currency_sym}{circuit_limits['lower_circuit']:.4f}"
                    if circuit_limits
                    else "  band=system_defaults"
                ),
            )

            # ── Step 4a: Build instrument defaults from ref_data.system_configs ─
            instrument: InstrumentDefaults = await lookup_instrument_defaults(
                ltp=ltp,
                db=self._db,
                live_circuit_limits=circuit_limits,
                asset_class=asset_class,
                symbol=symbol,
            )
            self._emit(
                messages,
                f"⚙️  Instrument defaults | tick={currency_sym}{instrument.tick_size}"
                f"  lot={instrument.lot_size}"
                f"  qty_step={instrument.qty_step_size}"
                f"  min_notional={instrument.min_notional}"
                f"  upper_band={currency_sym}{(instrument.upper_circuit or 0):.4f}"
                f"  lower_band={currency_sym}{(instrument.lower_circuit or 0):.4f}",
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
                asset_class=asset_class,
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
            entry_block = {
                "trigger": strategy_cfg.entry.trigger.model_dump(),
                "filters": [f.model_dump() for f in strategy_cfg.entry.filters],
            }
            entry_signal, entry_msgs = self._rule.evaluate_entry(df, entry_block)
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

            # ── Phase 10: entry gates (parity with backtest simulator) ─────────
            # The signal fired — now apply the same entry gates the quant-engine
            # simulator enforces (direction, entry window, consecutive-loss
            # circuit breaker, cooldowns, spread, gap, confirmation bars, RSI
            # band, volume ratio). Without this a strategy would behave
            # differently in backtest vs live. Live eval generates long entries.
            self._emit(messages, "🚧 ENTRY GATES | applying Phase 10 gate checks")
            gate_result = evaluate_entry_gates(
                df=df,
                gates=getattr(strategy_cfg, "gates", None) or GatesConfig(),
                exec_state=exec_state_payload,
                side="BUY",
                rule_engine=self._rule,
                entry_block=entry_block,
                asset_class=asset_class,
            )
            for m in gate_result.messages:
                self._emit(messages, m)

            if not gate_result.passed:
                self._emit(
                    messages,
                    f"⛔ Entry blocked by gate '{gate_result.blocked_by}' — returning no_action.",
                    level="warning",
                )
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
                    asset_class=asset_class,
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
            sl_pct  = float(getattr(effective_risk_config, "stop_loss_pct", strategy_cfg.sl_tp.stop_loss_pct))
            tp_pct  = float(getattr(effective_risk_config, "take_profit_pct", strategy_cfg.sl_tp.take_profit_pct))
            dlc_pct = float(getattr(effective_risk_config, "daily_loss_cap_pct", 3.0) or 3.0)
            ptr_pct = float(getattr(effective_risk_config, "per_trade_risk_pct", strategy_cfg.risk.max_risk_per_trade_pct))

            _multi_tp_targets = getattr(strategy_cfg.sl_tp, "multi_take_profit", None) or []
            _use_multi_tp = bool(_multi_tp_targets)

            if _use_multi_tp:
                tp_ladder = self._risk.compute_tp_ladder(
                    entry_price=ltp,
                    stop_loss_price=risk_out.stop_loss_price,
                    total_quantity=risk_out.position_size,
                    tp_targets=_multi_tp_targets,
                    asset_class=asset_class,
                )
                multi_tp_bracket = self._trade.build_multi_tp_bracket_order(
                    symbol=symbol,
                    entry_price=ltp,
                    stop_loss_price=risk_out.stop_loss_price,
                    tp_ladder=tp_ladder,
                    strategy_type=strategy_type,
                    strategy_id=strategy_id_str,
                    mode=mode.value,
                    bar_datetime=last_bar_dt,
                    asset_class=asset_class,
                )
                bracket = None
                _rung_summary = ", ".join(
                    f"TP{r.rung_index + 1}@{currency_sym}{r.price:.2f}×{r.exit_qty}"
                    for r in tp_ladder.rungs
                )
                self._emit(messages,
                    f"📦 MULTI-TP BRACKET GENERATED | asset_class={asset_class.value}"
                    f"  qty={risk_out.position_size}"
                    f"  entry={currency_sym}{ltp:.4f}"
                    f"  SL={currency_sym}{risk_out.stop_loss_price:.4f} (-{sl_pct}%)"
                    f"  rungs=[{_rung_summary}]"
                    f"  principal={currency_sym}{risk_out.principal_amount:,.4f}"
                )
            else:
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
                    asset_class=asset_class,
                    trailing_take_profit=strategy_cfg.sl_tp.trailing_take_profit,
                    trailing_stop=strategy_cfg.sl_tp.trailing_stop,
                )
                multi_tp_bracket = None
                self._emit(messages,
                    f"📦 BRACKET ORDER GENERATED | asset_class={asset_class.value}"
                    f"  qty={risk_out.position_size}"
                    f"  entry={currency_sym}{ltp:.4f}"
                    f"  SL={currency_sym}{risk_out.stop_loss_price:.4f} (-{sl_pct}%)"
                    f"  TP={currency_sym}{risk_out.take_profit_price:.4f} (+{tp_pct}%)"
                    f"  principal={currency_sym}{risk_out.principal_amount:,.4f}"
                )

            risk_snapshot = RiskSnapshot(
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                stop_loss_price=risk_out.stop_loss_price,
                take_profit_price=risk_out.take_profit_price if not _use_multi_tp else None,
                position_size=risk_out.position_size,
                principal_amount=risk_out.principal_amount,
                daily_loss_cap_pct=dlc_pct,
                per_trade_risk_pct=ptr_pct,
            )

            signal_bar, indicators = _extract_last_bar(df)
            entry_detail = TradeEntryReason(
                summary=(
                    f"Entry signal fired: {strategy_cfg.entry.trigger.type} "
                    f"for {symbol} @ {currency_sym}{ltp:.4f}"
                ),
                evaluation_mode="registry",
                condition=f"{strategy_cfg.entry.trigger.type} {strategy_cfg.entry.trigger.params}",
                signal_bar=signal_bar,
                entry_bar={"datetime": str(df.index[-1]), "index": len(df) - 1},
                indicators=indicators,
                confirmation_bars_required=strategy_cfg.gates.entry_confirmation_bars,
                confirmation_bars_observed=strategy_cfg.gates.entry_confirmation_bars,
                fill={"price": ltp, "note": "estimated at LTP; actual fill from broker"},
                initial_stop_price=risk_out.stop_loss_price,
                gates_passed=gate_result.gates_passed,
            )

            return EvaluateExecuteResponse(
                status="success",
                action=ActionType.entry_created,
                symbol=symbol,
                ltp=ltp,
                mode=mode,
                bracket_order=bracket,
                multi_tp_bracket=multi_tp_bracket,
                risk_snapshot=risk_snapshot,
                entry_detail=entry_detail,
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
        asset_class: AssetClass = AssetClass.equity_cash,
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
                f"  unrealised_PnL=₹{(ltp - entry_px) * float(getattr(pos, 'quantity', 1)):+,.2f}"
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

            # 0. Multi-TP ladder — check rungs before the standard SL/TP path.
            #    Each un-hit rung whose price is reached triggers a partial exit.
            #    risk_action updates stop_loss_price in-place on the position so the
            #    SL check below uses the updated value (e.g. moved to breakeven).
            if pos.take_profit_ladder is not None:
                is_long = pos.side == "BUY"
                for rung in pos.take_profit_ladder.rungs:
                    if rung.hit:
                        continue
                    rung_hit = (
                        (is_long and ltp >= rung.price) or
                        (not is_long and ltp <= rung.price)
                    )
                    if not rung_hit:
                        continue
                    rung.hit = True
                    # Apply risk_action
                    if rung.risk_action == "move_sl_to_breakeven":
                        if is_long:
                            pos.stop_loss_price = max(pos.stop_loss_price or 0.0, pos.entry_price)
                        else:
                            pos.stop_loss_price = min(pos.stop_loss_price or float("inf"), pos.entry_price)
                    elif rung.risk_action == "move_sl_to_previous_tp":
                        fired = [r for r in pos.take_profit_ladder.rungs if r.hit and r.rung_index < rung.rung_index]
                        if fired:
                            prev_price = fired[-1].price
                            if is_long:
                                pos.stop_loss_price = max(pos.stop_loss_price or 0.0, prev_price)
                            else:
                                pos.stop_loss_price = min(pos.stop_loss_price or float("inf"), prev_price)
                    exits.append(self._trade.build_exit_instruction(
                        position=pos,
                        reason="partial_take_profit",
                        exit_price=ltp,
                        strategy_type=strategy_type,
                        asset_class=asset_class,
                    ))
                    exits[-1].quantity = rung.exit_qty
                    self._emit(messages,
                        f"  🟡 PARTIAL TP RUNG {rung.rung_index} HIT | "
                        f"LTP={ltp:.4f} ≥ {rung.price:.4f} | "
                        f"exit_qty={rung.exit_qty} | risk_action={rung.risk_action}"
                    )
                # If all rungs hit → full position exited; skip remaining checks.
                if all(r.hit for r in pos.take_profit_ladder.rungs):
                    continue

            # 1. Stop loss
            if pos.stop_loss_price is not None and self._rule.check_stop_loss(ltp, pos.stop_loss_price):
                reason = "stop_loss"
                self._emit(messages,
                    f"  🔴 STOP LOSS HIT | LTP=₹{ltp:.2f} ≤ SL=₹{pos.stop_loss_price:.2f}"
                )

            # 2. Take profit (single-TP or remainder when multi-TP is not active)
            elif pos.take_profit_price is not None and pos.take_profit_ladder is None and self._rule.check_take_profit(ltp, pos.take_profit_price):
                reason = "take_profit"
                self._emit(messages,
                    f"  🟢 TAKE PROFIT HIT | LTP=₹{ltp:.2f} ≥ TP=₹{pos.take_profit_price:.2f}"
                )

            # 3. Exit signal (skip if strategy has no signal-based exit)
            elif strategy_cfg.exit.trigger is None:
                self._emit(messages, "  ↳ No exit signal configured — SL/TP only.")
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
                exit_instr = self._trade.build_exit_instruction(
                    position=pos,
                    reason=reason,
                    exit_price=ltp,
                    strategy_type=strategy_type,
                    asset_class=asset_class,
                )
                exit_bar, exit_indicators = _extract_last_bar(df)
                currency_sym = "$" if asset_class == AssetClass.crypto_spot else "₹"
                exit_instr.exit_detail = TradeExitReason(
                    summary=(
                        f"Exit triggered ({reason}) for {pos.position_id} "
                        f"| LTP={currency_sym}{ltp:.4f}"
                    ),
                    code=reason.upper(),
                    trigger={"price": ltp, "type": reason},
                    exit_bar=exit_bar,
                    indicators=exit_indicators,
                )
                exits.append(exit_instr)
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
        if cfg.exit.trigger is not None:
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
        """Build a SignalRule from a signal dict. The chat flow tags the KB
        signal under "name"; the inline/legacy shape uses "type"."""
        if isinstance(d, dict):
            return SignalRule(
                type=str(d.get("type") or d.get("name") or ""),
                params=d.get("params", {}) or {},
            )
        return SignalRule(type=str(d), params={})

    def _block(b: Any) -> EntryExitBlock:
        # Shape A — chat flow (strategy_assembler.build_strategy_config): a flat
        # list of signal dicts, each tagged signal_type = "TRIGGER" | "FILTER".
        if isinstance(b, list):
            signals = [s for s in b if isinstance(s, dict)]
            triggers = [s for s in signals
                        if str(s.get("signal_type", "")).upper() == "TRIGGER"]
            others   = [s for s in signals
                        if str(s.get("signal_type", "")).upper() != "TRIGGER"]
            if not triggers and signals:
                # No explicit TRIGGER tag — treat the first signal as the trigger.
                triggers, others = [signals[0]], signals[1:]
            if not triggers:
                return EntryExitBlock(trigger=SignalRule(type="", params={}))
            # Extra triggers (if any) fold into filters so no signal is dropped.
            filter_signals = triggers[1:] + others
            return EntryExitBlock(
                trigger=_rule(triggers[0]),
                filters=[_rule(f) for f in filter_signals],
            )
        # Shape B — explicit {trigger, filters} dict (Mode 2 inline / legacy).
        if isinstance(b, dict):
            return EntryExitBlock(
                trigger=_rule(b.get("trigger", {})),
                filters=[_rule(f) for f in b.get("filters", [])],
            )
        return EntryExitBlock(trigger=SignalRule(type="", params={}))

    entry_raw = raw.get("entry", {})
    exit_raw  = raw.get("exit", {})
    sl_tp_raw = raw.get("sl_tp", {})
    risk_raw  = raw.get("risk", {})

    sl_pct = float(
        sl_tp_raw.get("stop_loss_pct")
        or raw.get("stop_loss_pct")
        or runtime_risk_config.stop_loss_pct
    )

    # Phase 3 — trailing exits. A trailing take-profit / stop can be persisted at
    # the strategy top level (engine-YAML shape) or under sl_tp. When a trailing
    # take-profit is present the static target is OPTIONAL: a take_profit_pct of 0
    # means "no static cap" and must NOT fall back to a default (which would fire
    # before the trail engages). The OMS owns these legs at run time.
    trailing_tp_spec = _parse_trailing_order_spec(
        raw.get("trailing_take_profit") or sl_tp_raw.get("trailing_take_profit")
    )
    trailing_st_spec = _parse_trailing_order_spec(
        raw.get("trailing_stop") or sl_tp_raw.get("trailing_stop")
    )
    if trailing_tp_spec is not None:
        _raw_tp = sl_tp_raw.get("take_profit_pct", raw.get("take_profit_pct"))
        tp_pct = float(_raw_tp) if _raw_tp is not None else 0.0
    else:
        tp_pct = float(
            sl_tp_raw.get("take_profit_pct")
            or raw.get("take_profit_pct")
            or runtime_risk_config.take_profit_pct
        )

    # Multi-TP ladder — written by build_strategy_config (strategy_assembler.py)
    # into sl_tp.multi_take_profit (engine format) for new strategies, or into
    # execution_layers.partial_exits (chat format) for older ones.
    _multi_tp: Optional[List[Dict[str, Any]]] = None
    _raw_mtp = sl_tp_raw.get("multi_take_profit")
    if not _raw_mtp:
        # Fallback: execution_layers.partial_exits uses chat format
        # {trigger, value, size_pct, risk_action} → convert to engine format.
        _chat_rungs = (raw.get("execution_layers") or {}).get("partial_exits") or []
        if _chat_rungs and isinstance(_chat_rungs, list):
            def _chat_to_engine(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                if not isinstance(r, dict):
                    return None
                trigger = r.get("trigger", r.get("type", "risk_reward"))
                tp_type = "risk_reward" if trigger == "rr_multiple" else trigger
                size_pct = r.get("size_pct")
                exit_frac = r.get("exit_fraction")
                if size_pct is not None:
                    exit_frac = round(float(size_pct) / 100.0, 4)
                elif exit_frac is not None:
                    exit_frac = round(float(exit_frac), 4)
                else:
                    return None
                return {
                    "type": tp_type,
                    "value": float(r.get("value", 0)),
                    "exit_fraction": exit_frac,
                    "risk_action": r.get("risk_action", "none"),
                }
            _converted = [_chat_to_engine(r) for r in _chat_rungs]
            _raw_mtp = [r for r in _converted if r is not None] or None
    if _raw_mtp and isinstance(_raw_mtp, list) and len(_raw_mtp) >= 2:
        _multi_tp = _raw_mtp
        tp_pct = 0.0  # ladder overrides scalar target

    # Phase 10 — entry gates. The chat flow persists these under a "gates" key
    # in strategy_config (see app/planner/strategy_assembler.py). Strategies
    # created before Phase 10 have no "gates" key → GatesConfig() defaults all
    # gates to disabled, so behaviour is unchanged for them.
    gates_raw = raw.get("gates") or {}
    try:
        gates_cfg = GatesConfig(**gates_raw) if isinstance(gates_raw, dict) else GatesConfig()
    except Exception as exc:  # noqa: BLE001 — never hard-fail Mode 1 on gate schema drift
        logger.warning("Could not parse strategy_config.gates (%s) — gates disabled.", exc)
        gates_cfg = GatesConfig()

    # Asset class resolution order (most-specific → fallback):
    #   1. strategy_config["asset_class"] (JSONB, written by the chat flow)
    #   2. strategy.market  ('crypto_spot' | 'indian_stocks')
    #   3. Default to equity_cash (legacy strategies)
    asset_class_value = (
        raw.get("asset_class")
        or _market_to_asset_class(strategy_row.market)
        or AssetClass.equity_cash.value
    )
    try:
        asset_class = AssetClass(str(asset_class_value))
    except ValueError:
        logger.warning(
            "Strategy %s has unknown asset_class=%r — defaulting to equity_cash.",
            strategy_id, asset_class_value,
        )
        asset_class = AssetClass.equity_cash

    return StrategyConfigPayload(
        strategy_id=strategy_id,
        symbol=strategy_row.symbol,
        timeframe=strategy_row.timeframe,
        strategy_type=raw.get("strategy_type", "intraday"),
        asset_class=asset_class,
        entry=_block(entry_raw),
        exit=_block(exit_raw),
        sl_tp=SlTpConfig(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            trailing_take_profit=trailing_tp_spec,
            trailing_stop=trailing_st_spec,
            multi_take_profit=_multi_tp,
        ),
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
        gates=gates_cfg,
    )


def _market_to_asset_class(market: Optional[str]) -> Optional[str]:
    """Translate the legacy ``strategy.market`` text column to an asset_class id.

    The chat flow writes free-text values like ``"indian_stocks"`` and
    ``"crypto"``; the asset-class taxonomy uses ``equity_cash`` / ``crypto_spot``.
    Returns None when no mapping is known so the caller can fall through to the
    safe equity_cash default.
    """
    if not market:
        return None
    m = str(market).strip().lower()
    if m in {"crypto", "crypto_spot", "binance", "binance_spot"}:
        return AssetClass.crypto_spot.value
    if m in {"indian_stocks", "equity", "equity_cash", "nse", "bse", "us_stocks"}:
        return AssetClass.equity_cash.value
    return None


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

    # Phase 10 — gate state. consecutive_losses / last_trade_was_loss are not
    # dedicated columns; they live in the execution_state.state_json blob,
    # written by the OMS/trade-tracking layer. Absent → gates that need them
    # (consecutive-loss, cooldown) stay inert.
    raw_consec = snapshot.get("consecutive_losses", 0)
    raw_last_loss = snapshot.get("last_trade_was_loss", None)

    return ExecutionStatePayload(
        available_margin=float(row.capital or 100_000.0),
        open_positions=positions,
        bars_since_last_trade=int(row.bars_since_last_trade or 0),
        capital=float(row.capital or 100_000.0),
        consecutive_losses=int(raw_consec or 0),
        last_trade_was_loss=(bool(raw_last_loss) if raw_last_loss is not None else None),
    )
