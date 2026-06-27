"""
app/services/execution/universe_evaluator.py — per-member evaluation fan-out + aggregation (§9).

A static deployment evaluates ONE symbol through :class:`StrategyEvaluator`. A dynamic
deployment evaluates its CURRENT member set on each tick and answers as one shared-capital
portfolio. This module does exactly that, reusing the existing per-symbol decision engine
unchanged (parity, Invariant 1):

  1. For each active member, stamp the member symbol onto the deployment's symbol-agnostic
     strategy body (the same "strategy template" insight backtest uses — only the symbol line
     differs) and run the EXISTING :meth:`StrategyEvaluator.evaluate`.
  2. Aggregate the per-member responses through :class:`LivePortfolioManager`:
       * EXITS are always allowed (risk-reducing) and free their capital first — an exit on one
         member can open a slot for another member's entry on the same tick.
       * ENTRIES are gated: ``max_positions``, available capital, and sector cap. A skipped
         entry is dropped with an explicit reason (never silently lost, §12.1).
  Every instruction keeps its owning symbol so the OMS can attribute each order.

The core :func:`aggregate_member_evaluations` is injectable (per-member ``evaluate`` callable +
exec-state factory + Portfolio Manager) so it is unit-testable with no DB, broker, or market
data. :func:`evaluate_universe_deployment` is the thin DB/StrategyEvaluator wiring on top.
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Awaitable, Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.execution import (
    ActionType,
    EvaluateExecuteRequest,
    EvaluateExecuteResponse,
    EvaluationMode,
    ExecutionStatePayload,
    StrategyConfigPayload,
)
from app.schemas.universe_execution import MemberOutcome, UniverseEvaluateResponse
from app.services.execution.portfolio_manager import LivePortfolioManager

logger = logging.getLogger(__name__)

# Type aliases for the injected per-member callables.
EvaluateMember = Callable[[EvaluateExecuteRequest], Awaitable[EvaluateExecuteResponse]]
ExecStateFor = Callable[[str], ExecutionStatePayload]


def _stamp_symbol(base: StrategyConfigPayload, symbol: str) -> StrategyConfigPayload:
    """The deployment's strategy body with ``symbol`` swapped to this member (template stamp).

    Mirrors :meth:`StrategySpec.to_engine_template_dict` at the execution-payload layer: every
    field except the instrument is shared across members, so a deep ``model_copy`` with only
    ``symbol`` replaced cannot let per-member bodies drift from the deployment's intent.
    """
    return base.model_copy(update={"symbol": symbol}, deep=True)


def _stop_distance_frac(resp: EvaluateExecuteResponse) -> Optional[float]:
    """Stop distance as a fraction of entry, for risk-based sizing — from the member's own
    risk snapshot (SL %) when present, else from the SL/entry prices, else None (→ equal-weight)."""
    rs = resp.risk_snapshot
    if rs is None:
        return None
    if rs.stop_loss_pct and rs.stop_loss_pct > 0:
        return float(rs.stop_loss_pct) / 100.0
    if rs.stop_loss_price and resp.ltp:
        dist = abs(float(resp.ltp) - float(rs.stop_loss_price)) / float(resp.ltp)
        return dist if dist > 0 else None
    return None


async def aggregate_member_evaluations(
    *,
    deployment_id: str,
    member_symbols: list[str],
    base_config: StrategyConfigPayload,
    evaluate_member: EvaluateMember,
    exec_state_for: ExecStateFor,
    portfolio: LivePortfolioManager,
    mode: EvaluationMode = EvaluationMode.paper,
    sector_for: Optional[Callable[[str], Optional[str]]] = None,
    on_admit: Optional[Callable[[str, float], Awaitable[None]]] = None,
    on_release: Optional[Callable[[str], Awaitable[None]]] = None,
) -> UniverseEvaluateResponse:
    """Fan out per-member evaluation and aggregate through the shared-capital Portfolio Manager.

    Injectable core — ``evaluate_member`` runs one member's per-symbol decision; ``exec_state_for``
    supplies that member's open positions; ``portfolio`` is the live shared-capital authority
    (already rebuilt to the deployment's open positions). ``on_admit``/``on_release`` persist the
    capital change at the edge (e.g. ``universe_members.allocated_capital``).
    """
    messages: list[str] = []
    results: list[MemberOutcome] = []
    t_begin = perf_counter()
    logger.info(
        "\n┌─🌌 UNIVERSE EVALUATION | deployment=%s | mode=%s ──────────────────────\n"
        "│  members   : %d  →  %s\n"
        "│  portfolio : open=%d/%d  free_cash=%.2f  sizing=%s  risk/trade=%.2f%%\n"
        "│  plan      : ① fan-out per member  ② exits free capital  ③ entries compete for slots\n"
        "└──────────────────────────────────────────────────────────────────────",
        deployment_id, getattr(mode, "value", mode), len(member_symbols), member_symbols,
        portfolio.open_count, portfolio.max_positions, portfolio.free_cash,
        portfolio.sizing_mode, portfolio.risk_per_trade_pct,
    )

    # ── Pass 1: run every member through the existing per-symbol engine ────────
    logger.info("🌌 [1/3] FAN-OUT | evaluating %d member(s) through the per-symbol engine", len(member_symbols))
    raw: list[tuple[str, EvaluateExecuteResponse]] = []
    t0 = perf_counter()
    for i, symbol in enumerate(member_symbols, start=1):
        logger.info("🌌 ╭── member %d/%d | %s ── evaluating ──────────────────", i, len(member_symbols), symbol)
        req = EvaluateExecuteRequest(
            strategy_config=_stamp_symbol(base_config, symbol),
            execution_state=exec_state_for(symbol),
            mode=mode,
        )
        resp = await evaluate_member(req)
        raw.append((symbol, resp))
        logger.info(
            "🌌 ╰── member %d/%d | %s → action=%s ltp=%s%s",
            i, len(member_symbols), symbol, getattr(resp.action, "value", resp.action),
            resp.ltp, f" err={resp.error}" if resp.error else "",
        )
    logger.info("⏱️ TIMING|step=universe_fanout|deployment=%s|duration=%.4fs|members=%d",
                deployment_id, perf_counter() - t0, len(member_symbols))

    # ── Pass 2a: EXITS first (risk-reducing; always allowed; free capital) ─────
    logger.info("🌌 [2/3] EXITS | risk-reducing, always allowed — released BEFORE entries compete")
    all_exits = []
    for symbol, resp in raw:
        if resp.action == ActionType.exit_triggered and resp.exits:
            reasons = ", ".join(e.reason for e in resp.exits)
            for ex in resp.exits:
                all_exits.append(ex)
            # Release the member's capital so a slot/cash frees for entries below.
            cash_before = portfolio.free_cash
            portfolio.release(symbol)
            if on_release is not None:
                await on_release(symbol)
            outcome = MemberOutcome(symbol=symbol, action=resp.action, response=resp)
            results.append(outcome)
            logger.info(
                "🌌   🚪 %s | exit_triggered | legs=%d (%s) | freed capital → cash %.2f→%.2f | open=%d/%d",
                symbol, len(resp.exits), reasons, cash_before, portfolio.free_cash,
                portfolio.open_count, portfolio.max_positions,
            )
            messages.append(f"🚪 {symbol} | exit_triggered ({len(resp.exits)} leg(s): {reasons}) — capital released")
    if not all_exits:
        logger.info("🌌   — no exits this tick")

    exited = {o.symbol for o in results}

    # ── Pass 2b: ENTRIES, gated by the portfolio layer (cap/capital/sector) ────
    logger.info(
        "🌌 [3/3] ENTRIES | %d candidate(s) compete for %d/%d free slot(s), cash=%.2f",
        sum(1 for s, r in raw if s not in exited and r.action == ActionType.entry_created),
        max(0, portfolio.max_positions - portfolio.open_count), portfolio.max_positions,
        portfolio.free_cash,
    )
    entries: list[MemberOutcome] = []
    skipped: list[MemberOutcome] = []
    for symbol, resp in raw:
        if symbol in exited:
            continue
        if resp.action != ActionType.entry_created:
            # no_action / error — record for the audit, no portfolio interaction.
            results.append(MemberOutcome(symbol=symbol, action=resp.action, response=resp))
            logger.info("🌌   ➖ %s | %s — no entry to arbitrate", symbol,
                        getattr(resp.action, "value", resp.action))
            continue

        sector = sector_for(symbol) if sector_for else None
        sdf = _stop_distance_frac(resp)
        decision = portfolio.evaluate_entry(symbol, stop_distance_frac=sdf, sector=sector)
        if decision.admit:
            portfolio.admit(symbol, decision.allocated_capital, sector=sector)
            if on_admit is not None:
                await on_admit(symbol, decision.allocated_capital)
            outcome = MemberOutcome(
                symbol=symbol, action=resp.action, admitted=True,
                allocated_capital=decision.allocated_capital, response=resp,
            )
            entries.append(outcome)
            results.append(outcome)
            logger.info(
                "🌌   🚀 %s | ADMITTED | allocated=%.2f (stop_dist=%s sector=%s) | open=%d/%d cash=%.2f",
                symbol, decision.allocated_capital,
                f"{sdf:.4f}" if sdf is not None else "—", sector or "—",
                portfolio.open_count, portfolio.max_positions, portfolio.free_cash,
            )
            messages.append(f"🚀 {symbol} | entry admitted | allocated={decision.allocated_capital:.2f}")
        else:
            outcome = MemberOutcome(
                symbol=symbol, action=resp.action, admitted=False,
                skip_reason=decision.reason, response=resp,
            )
            skipped.append(outcome)
            results.append(outcome)
            logger.info(
                "🌌   ⛔ %s | SKIPPED | reason=%s | open=%d/%d cash=%.2f",
                symbol, decision.reason, portfolio.open_count, portfolio.max_positions, portfolio.free_cash,
            )
            messages.append(f"⛔ {symbol} | entry skipped | reason={decision.reason}")

    logger.info(
        "\n└─🌌 UNIVERSE EVALUATION DONE | deployment=%s ──────────────────────────\n"
        "   entries=%d  skipped=%d  exits=%d  |  open_after=%d/%d  cash=%.2f  |  %.4fs\n"
        "   admitted : %s\n"
        "   skipped  : %s",
        deployment_id, len(entries), len(skipped), len(all_exits),
        portfolio.open_count, portfolio.max_positions, portfolio.free_cash, perf_counter() - t_begin,
        [e.symbol for e in entries] or "—",
        [f"{s.symbol}({s.skip_reason})" for s in skipped] or "—",
    )
    return UniverseEvaluateResponse(
        status="success",
        deployment_id=deployment_id,
        mode=mode,
        max_positions=portfolio.max_positions,
        members_evaluated=len(member_symbols),
        open_positions_after=portfolio.open_count,
        entries=entries,
        skipped=skipped,
        exits=all_exits,
        results=results,
        messages=messages,
    )


# ── DB / StrategyEvaluator wiring ─────────────────────────────────────────────
async def evaluate_universe_deployment(
    db: AsyncSession,
    *,
    deployment_id: str,
    base_config: StrategyConfigPayload,
    member_symbols: list[str],
    total_capital: float,
    max_positions: int,
    sizing_mode: str = "equal_weight",
    risk_per_trade_pct: float = 1.0,
    open_positions=None,
    bars_since_last_trade: int = 0,
    mode: EvaluationMode = EvaluationMode.paper,
) -> UniverseEvaluateResponse:
    """Wire the injectable core to the real :class:`StrategyEvaluator` + a fresh Portfolio
    Manager rebuilt to the deployment's current open positions.

    ``open_positions`` is the full list of the deployment's open :class:`OpenPosition`s across
    all members; each member is evaluated against only its own open positions, while the
    Portfolio Manager is seeded with all of them so the cap/capital math is portfolio-wide.
    """
    from app.services.execution.portfolio_manager import LivePosition
    from app.services.execution.strategy_evaluator import StrategyEvaluator

    open_positions = list(open_positions or [])
    by_symbol: dict[str, list] = {}
    for pos in open_positions:
        by_symbol.setdefault(pos.symbol, []).append(pos)

    # ── Held ∪ resolved ───────────────────────────────────────────────────────
    # The resolved set may have dropped a symbol we still HOLD (it left "top-N" this window).
    # Such a position must still be exit-managed, so evaluate the union: resolved members first
    # (eligible for fresh entries), then any held symbol not in the resolved set (exit-only).
    resolved_set = set(member_symbols)
    held_not_resolved = [s for s in by_symbol if s not in resolved_set]
    eval_symbols = list(member_symbols) + held_not_resolved

    logger.info(
        "🌌 PORTFOLIO SETUP | deployment=%s | total_capital=%.2f max_positions=%d sizing=%s\n"
        "   resolved members : %d → %s\n"
        "   held (OMS)       : %d across %s\n"
        "   held-but-dropped : %s  ← evaluated for EXIT only (not in current universe)\n"
        "   → evaluating     : %d symbol(s)",
        deployment_id, total_capital, max_positions, sizing_mode,
        len(member_symbols), list(member_symbols) or "—",
        len(open_positions), sorted(by_symbol) or "—",
        held_not_resolved or "—", len(eval_symbols),
    )
    portfolio = LivePortfolioManager(
        total_capital=total_capital,
        max_positions=max_positions,
        sizing_mode=sizing_mode,  # type: ignore[arg-type]
        risk_per_trade_pct=risk_per_trade_pct,
    )
    # Rebuild the shared pool from the deployment's currently-open positions (§9 step 7), so the
    # cap and free-cash math continue from the true live state rather than a fresh pool.
    if open_positions:
        portfolio.rebuild([
            LivePosition(symbol=p.symbol, allocated_capital=float(getattr(p, "entry_price", 0.0))
                         * float(getattr(p, "quantity", 0.0)))
            for p in open_positions
        ])
        logger.info("🌌 PORTFOLIO REBUILT | open=%d/%d | free_cash=%.2f (locked in live positions)",
                    portfolio.open_count, portfolio.max_positions, portfolio.free_cash)

    evaluator = StrategyEvaluator(db)

    def exec_state_for(symbol: str) -> ExecutionStatePayload:
        return ExecutionStatePayload(
            available_margin=portfolio.free_cash,
            open_positions=by_symbol.get(symbol, []),
            capital=total_capital,
            bars_since_last_trade=bars_since_last_trade,
        )

    return await aggregate_member_evaluations(
        deployment_id=deployment_id,
        member_symbols=eval_symbols,
        base_config=base_config,
        evaluate_member=evaluator.evaluate,
        exec_state_for=exec_state_for,
        portfolio=portfolio,
        mode=mode,
    )
