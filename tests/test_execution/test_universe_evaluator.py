"""Phase D — per-member fan-out + portfolio aggregation (§9).

Injectable core (no DB / market data): a fake per-member evaluator drives
``aggregate_member_evaluations`` so we can assert the portfolio-level contract directly —
entries capped at ``max_positions`` with per-symbol skip reasons, exits always allowed and
freeing a slot, and symbol attribution preserved on every instruction.
"""
from __future__ import annotations

from decimal import Decimal

from app.schemas.execution import (
    ActionType,
    EntryExitBlock,
    EvaluateExecuteResponse,
    EvaluationMode,
    ExitBlock,
    ExitInstruction,
    ProductType,
    RiskConfig,
    RiskSnapshot,
    SignalRule,
    SlTpConfig,
    StrategyConfigPayload,
)
from app.services.execution.portfolio_manager import LivePortfolioManager, LivePosition
from app.services.execution.universe_evaluator import aggregate_member_evaluations


def _base_config() -> StrategyConfigPayload:
    return StrategyConfigPayload(
        symbol="PLACEHOLDER",
        timeframe="5m",
        entry=EntryExitBlock(trigger=SignalRule(type="ema_crossover")),
        exit=ExitBlock(),
        sl_tp=SlTpConfig(stop_loss_pct=2.0, take_profit_pct=4.0),
        risk=RiskConfig(),
    )


def _entry_resp(symbol: str) -> EvaluateExecuteResponse:
    return EvaluateExecuteResponse(
        status="success", action=ActionType.entry_created, symbol=symbol, ltp=100.0,
        risk_snapshot=RiskSnapshot(
            stop_loss_pct=2.0, take_profit_pct=4.0, daily_loss_cap_pct=3.0, per_trade_risk_pct=2.0,
        ),
    )


def _no_action_resp(symbol: str) -> EvaluateExecuteResponse:
    return EvaluateExecuteResponse(
        status="success", action=ActionType.no_action, symbol=symbol, ltp=100.0,
    )


def _exit_resp(symbol: str) -> EvaluateExecuteResponse:
    return EvaluateExecuteResponse(
        status="success", action=ActionType.exit_triggered, symbol=symbol, ltp=100.0,
        exits=[ExitInstruction(
            position_id=f"pos-{symbol}", symbol=symbol, reason="stop_loss",
            exit_price=100.0, quantity=Decimal("1"), product_type=ProductType.cnc,
        )],
    )


def _evaluator_from_map(mapping: dict[str, EvaluateExecuteResponse]):
    async def _evaluate(req):
        return mapping[req.strategy_config.symbol]
    return _evaluate


def _exec_state_factory(_symbol: str):
    from app.schemas.execution import ExecutionStatePayload
    return ExecutionStatePayload(available_margin=100_000.0, capital=100_000.0)


async def test_entries_capped_at_max_positions_with_skip_reasons():
    members = ["AAA", "BBB", "CCC"]
    responses = {s: _entry_resp(s) for s in members}
    portfolio = LivePortfolioManager(total_capital=100_000.0, max_positions=2)

    out = await aggregate_member_evaluations(
        deployment_id="dep1", member_symbols=members, base_config=_base_config(),
        evaluate_member=_evaluator_from_map(responses), exec_state_for=_exec_state_factory,
        portfolio=portfolio, mode=EvaluationMode.paper,
    )

    assert len(out.entries) == 2                       # cap honoured
    assert {e.symbol for e in out.entries} == {"AAA", "BBB"}
    assert all(e.admitted is True and e.allocated_capital > 0 for e in out.entries)
    assert len(out.skipped) == 1
    assert out.skipped[0].symbol == "CCC"
    assert out.skipped[0].skip_reason == "position_cap"
    assert out.open_positions_after == 2
    assert out.members_evaluated == 3


async def test_exit_frees_a_slot_for_an_entry_same_tick():
    # CCC is already open (fills the cap together with an implicit second slot); it exits this
    # tick, which must free a slot so a fresh entry (BBB) can be admitted.
    members = ["BBB", "CCC"]
    responses = {"BBB": _entry_resp("BBB"), "CCC": _exit_resp("CCC")}
    portfolio = LivePortfolioManager(total_capital=100_000.0, max_positions=1)
    portfolio.rebuild([LivePosition(symbol="CCC", allocated_capital=50_000.0)])
    assert portfolio.open_count == 1                   # cap is full before the tick

    out = await aggregate_member_evaluations(
        deployment_id="dep1", member_symbols=members, base_config=_base_config(),
        evaluate_member=_evaluator_from_map(responses), exec_state_for=_exec_state_factory,
        portfolio=portfolio, mode=EvaluationMode.paper,
    )

    assert [e.symbol for e in out.entries] == ["BBB"]  # slot freed by CCC's exit
    assert len(out.exits) == 1 and out.exits[0].symbol == "CCC"
    assert out.open_positions_after == 1


async def test_no_action_members_recorded_not_admitted():
    members = ["AAA", "BBB"]
    responses = {"AAA": _entry_resp("AAA"), "BBB": _no_action_resp("BBB")}
    portfolio = LivePortfolioManager(total_capital=100_000.0, max_positions=5)

    out = await aggregate_member_evaluations(
        deployment_id="dep1", member_symbols=members, base_config=_base_config(),
        evaluate_member=_evaluator_from_map(responses), exec_state_for=_exec_state_factory,
        portfolio=portfolio, mode=EvaluationMode.paper,
    )

    assert [e.symbol for e in out.entries] == ["AAA"]
    assert not out.skipped
    # BBB still appears in the full results audit as a no_action outcome.
    bbb = next(r for r in out.results if r.symbol == "BBB")
    assert bbb.action == ActionType.no_action and bbb.admitted is None


async def test_held_but_dropped_symbol_still_evaluated_for_exit(monkeypatch):
    # A symbol we HOLD but which fell out of the resolved universe this window must still be
    # exit-managed (evaluate held ∪ resolved). Stub StrategyEvaluator so no DB/market data.
    import app.services.execution.strategy_evaluator as se
    from app.schemas.execution import OpenPosition

    class _FakeEvaluator:
        def __init__(self, db):
            pass

        async def evaluate(self, req):
            sym = req.strategy_config.symbol
            return _exit_resp("OLD_USDT") if sym == "OLD_USDT" else _no_action_resp(sym)

    monkeypatch.setattr(se, "StrategyEvaluator", _FakeEvaluator)

    from app.services.execution.universe_evaluator import evaluate_universe_deployment

    held = [OpenPosition(position_id="p1", symbol="OLD_USDT", side="BUY",
                         quantity=Decimal("1"), entry_price=100.0)]
    out = await evaluate_universe_deployment(
        db=None, deployment_id="dep1", base_config=_base_config(),
        member_symbols=["BTC_USDT", "ETH_USDT"],   # resolved set — does NOT include OLD_USDT
        total_capital=100_000.0, max_positions=2,
        open_positions=held, mode=EvaluationMode.paper,
    )
    assert any(e.symbol == "OLD_USDT" for e in out.exits), \
        "a held-but-dropped symbol must still be evaluated and exited"


async def test_symbol_is_stamped_into_per_member_config():
    seen: list[str] = []

    async def _evaluate(req):
        seen.append(req.strategy_config.symbol)
        return _no_action_resp(req.strategy_config.symbol)

    portfolio = LivePortfolioManager(total_capital=100_000.0, max_positions=5)
    await aggregate_member_evaluations(
        deployment_id="dep1", member_symbols=["AAA", "BBB"], base_config=_base_config(),
        evaluate_member=_evaluate, exec_state_for=_exec_state_factory,
        portfolio=portfolio, mode=EvaluationMode.paper,
    )
    assert seen == ["AAA", "BBB"]      # member symbol stamped, placeholder never leaks
