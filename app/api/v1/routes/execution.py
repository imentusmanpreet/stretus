"""
app/api/v1/routes/execution.py
───────────────────────────────
POST /strategy/evaluate/execute

Stateless decision endpoint — returns trade instructions to the OMS.
Never executes orders directly.

Supports two request modes:
  Mode 1: { "strategy_id": "...", "mode": "paper"|"live" }
          → fetches strategy config, risk config, and execution state from DB.

  Mode 2: { "strategy_config": {...}, "execution_state": {...} }
          → uses inline config (useful for backtesting runners / testing).
"""
from __future__ import annotations

import logging
from typing import Annotated, Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.execution import (
    EvaluateExecuteRequest,
    EvaluateExecuteResponse,
    EvaluationMode,
)
from app.schemas.universe_execution import UniverseEvaluateResponse
from app.services.execution.strategy_evaluator import StrategyEvaluator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/evaluate/execute",
    response_model=Union[EvaluateExecuteResponse, UniverseEvaluateResponse],
    status_code=status.HTTP_200_OK,
    summary="Evaluate a strategy and generate execution instructions",
    description="""
Stateless decision engine endpoint. Returns bracket orders (entry + SL + TP) and/or exit
instructions. Never submits orders — the OMS is responsible for execution.

**Static — single symbol:**
- *Mode 1:* supply `strategy_id` + optional `mode` (config fetched from the DB).
- *Mode 2:* supply `strategy_config` + `execution_state` inline.
→ returns an `EvaluateExecuteResponse` for the one symbol.

**Dynamic — universe (portfolio):** supply `strategy_config` (the per-member body) + a
`universe` block (deployment, members, shared capital, `max_positions`, open positions). Each
member is evaluated through the SAME per-symbol engine, then aggregated as one shared-capital
portfolio — exits always allowed (and free capital first), entries gated by `max_positions` +
available capital with explicit per-symbol skip reasons.
→ returns a `UniverseEvaluateResponse`. Feature-flagged behind `DYNAMIC_UNIVERSE_ENABLED`;
`max_positions` is clamped to `DYNAMIC_UNIVERSE_MAX_POSITIONS`.

### Static execution flow (strict order)
1. Load strategy config
2. Fetch market data (candles, LTP, circuit limits) — cached 60s
3. Fetch instrument metadata (tick/lot size)
4. Fetch execution state
5. **Exit phase** — for each open position: SL → TP → exit signal → time
6. **Entry phase** — entry signal → risk calc → account check → bracket order
""",
    tags=["⚡ Execution"],
)
async def evaluate_execute(
    body: EvaluateExecuteRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Union[EvaluateExecuteResponse, UniverseEvaluateResponse]:
    from app.core.tenant_context import get_current_tenant

    # ── Dynamic-universe branch: routed ONLY when a `universe` block is present, BEFORE any
    # static logic runs — the static single-symbol path below is byte-for-byte unchanged.
    if body.universe is not None:
        return await _evaluate_dynamic(body, db)

    tenant = get_current_tenant()
    logger.info(
        "evaluate/execute | mode=%s strategy_id=%s symbol=%s tenant_id=%s legacy=%s",
        body.mode,
        body.strategy_id,
        getattr(body.strategy_config, "symbol", None),
        tenant.tenant_id if tenant else "none",
        tenant.is_legacy if tenant else True,
    )

    evaluator = StrategyEvaluator(db)
    return await evaluator.evaluate(body)


async def _evaluate_dynamic(
    body: EvaluateExecuteRequest,
    db: AsyncSession,
) -> UniverseEvaluateResponse:
    """Portfolio-evaluation branch: fan out per-member and aggregate via the Portfolio Manager.

    Reached only when the request carries a ``universe`` block. The per-member strategy body is
    the request's own ``strategy_config`` (symbol stamped per member). Feature-flagged; clamps
    ``max_positions`` to the platform ceiling.
    """
    from datetime import datetime, timezone

    from fastapi import HTTPException

    from app.core.config import get_settings
    from app.services.execution.universe_evaluator import evaluate_universe_deployment

    settings = get_settings()
    if not settings.dynamic_universe_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dynamic-universe evaluation is disabled (DYNAMIC_UNIVERSE_ENABLED=false).",
        )

    uni = body.universe
    is_live = body.mode == EvaluationMode.live
    logger.info(
        "\n╔═🌌 DYNAMIC EVALUATE | deployment=%s | mode=%s ════════════════════════",
        uni.deployment_id, getattr(body.mode, "value", body.mode),
    )

    # ── FAIL CLOSED (live): the OMS owns state; never assume capital / flat positions ─────────
    # Paper may default (testing/parity); live MUST carry the OMS's truth.
    if is_live and (uni.total_capital is None or uni.open_positions is None):
        missing = [n for n, v in (("total_capital", uni.total_capital),
                                  ("open_positions", uni.open_positions)) if v is None]
        logger.warning(
            "🌌 ⛔ FAIL-CLOSED | deployment=%s | live evaluation missing OMS state: %s — refusing to trade",
            uni.deployment_id, missing,
        )
        return UniverseEvaluateResponse(
            status="error", deployment_id=uni.deployment_id, mode=body.mode,
            max_positions=min(uni.max_positions, settings.dynamic_universe_max_positions),
            error=(
                f"Live dynamic evaluation requires {missing} from the OMS. The platform never "
                "assumes capital or flat positions for real money — failing closed."
            ),
        )
    total_capital = uni.total_capital if uni.total_capital is not None else 100_000.0
    open_positions = uni.open_positions if uni.open_positions is not None else []

    # ── Member-resolution precedence ──────────────────────────────────────────
    #   1. explicit `members` → use as-is (override, for testing)
    #   2. live `rule`        → resolve as-of the CURRENT refresh window (stateless + cached);
    #                           stable within the window, re-picks at the boundary, OMS-agnostic
    #   3. neither            → load the deployment's active members from the DB (legacy/back-compat)
    if uni.members is not None:
        member_symbols = uni.members
        logger.info("🌌 members=EXPLICIT (%d): %s", len(member_symbols), member_symbols)
    elif uni.rule is not None:
        from app.services.execution.live_universe import resolve_universe_windowed

        resolved = await resolve_universe_windowed(
            uni.rule, now=datetime.now(timezone.utc), settings=settings, run_id=uni.deployment_id,
        )
        member_symbols = resolved.member_symbols
        logger.info(
            "🌌 members=RESOLVED LIVE (window-quantized) → %d: %s (snapshot=%s)",
            len(member_symbols), member_symbols, (resolved.snapshot_hash or "")[:12],
        )
    else:
        from app.services.execution.universe_persistence import load_active_members

        rows = await load_active_members(db, deployment_id=uni.deployment_id, include_retiring=False)
        member_symbols = [r.symbol for r in rows]
        logger.info("🌌 members=FROM DB (%d active)", len(member_symbols))

    # Clamp the concurrent-position cap to the platform ceiling (§7.1) — never silently exceed.
    max_positions = min(uni.max_positions, settings.dynamic_universe_max_positions)
    logger.info(
        "🌌 deployment=%s | members=%d | max_positions=%d (requested %d) | capital=%.2f | held=%d | mode=%s",
        uni.deployment_id, len(member_symbols), max_positions, uni.max_positions,
        total_capital, len(open_positions), getattr(body.mode, "value", body.mode),
    )

    return await evaluate_universe_deployment(
        db,
        deployment_id=uni.deployment_id,
        base_config=body.strategy_config,
        member_symbols=member_symbols,
        total_capital=total_capital,
        max_positions=max_positions,
        sizing_mode=uni.sizing_mode,
        risk_per_trade_pct=uni.risk_per_trade_pct,
        open_positions=open_positions,
        bars_since_last_trade=uni.bars_since_last_trade,
        mode=body.mode,
    )
