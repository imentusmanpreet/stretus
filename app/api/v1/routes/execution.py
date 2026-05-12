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
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.execution import EvaluateExecuteRequest, EvaluateExecuteResponse
from app.services.execution.strategy_evaluator import StrategyEvaluator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/evaluate/execute",
    response_model=EvaluateExecuteResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a strategy and generate execution instructions",
    description="""
Stateless decision engine endpoint.

Returns bracket orders (entry + SL + TP) and/or exit instructions.
Never submits orders — the OMS is responsible for execution.

**Mode 1 (production):** supply `strategy_id` + optional `mode`.
All config is fetched from the database (strategy_config, risk_config, execution_state).

**Mode 2 (direct / paper):** supply `strategy_config` + `execution_state` inline.
Useful for testing or when the strategy has not been confirmed yet.

### Execution flow (strict order)
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
) -> EvaluateExecuteResponse:
    logger.info(
        "evaluate/execute | mode=%s strategy_id=%s symbol=%s",
        body.mode,
        body.strategy_id,
        getattr(body.strategy_config, "symbol", None),
    )

    evaluator = StrategyEvaluator(db)
    return await evaluator.evaluate(body)
