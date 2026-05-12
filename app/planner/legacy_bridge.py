"""
app/planner/legacy_bridge.py — Drop-in replacement for the old retriever calls.

Goal: let chat_service.py swap from
    plan = plan_strategy_signals(builder, llm_preferences=llm_prefs)
    plan = enrich_plan_with_ohlcv(plan, ohlcv_records, ...)
to
    plan = await plan_signals_v2(builder, ohlcv_records=ohlcv_records)

without touching any downstream consumers (builder.apply_signal_plan,
build_strategy_object, build_strategy_config, response composers).

The returned dict matches the legacy signal_plan shape consumed by the
existing code paths.
"""
from __future__ import annotations

import logging
from typing import Any

from app.kb import kb
from app.planner.adapters import plan_to_signal_plan
from app.planner.pipeline import (
    NoValidCandidate,
    Pipeline,
    UnsupportedStock,
    UnsupportedTimeframe,
)
from app.planner.trace import remember_trace

logger = logging.getLogger(__name__)


def _ohlcv_records_to_df(records: list[dict] | None):
    """Convert the OHLCV record list used by chat_service into a pandas DataFrame
    that the planner's param_resolver / SL-TP scaler can consume.
    Returns None if no records or pandas unavailable."""
    if not records:
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    try:
        df = pd.DataFrame(records)
        rename = {}
        for raw in list(df.columns):
            low = raw.lower()
            if low in {"open", "high", "low", "close", "volume"}:
                rename[raw] = low.capitalize() if low != "volume" else "Volume"
        df = df.rename(columns=rename)
        return df
    except Exception as exc:
        logger.warning("legacy_bridge|ohlcv_df_conversion_failed|err=%s", exc)
        return None


async def plan_signals_v2(
    builder: Any,
    ohlcv_records: list[dict] | None = None,
    pipeline: Pipeline | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the new planner and return a dict in the legacy signal_plan shape.

    Raises the planner's typed errors (UnsupportedStock, UnsupportedTimeframe,
    NoValidCandidate) so chat_service can map them to user-facing messages.
    """
    pipe = pipeline or Pipeline()
    df = _ohlcv_records_to_df(ohlcv_records)
    bars = len(df) if df is not None else 0

    logger.info(
        "legacy_bridge|enter|session_id=%s|symbol=%s|tf=%s|sentiment=%s|ohlcv_bars=%d",
        session_id,
        getattr(builder, "symbol", None),
        getattr(builder, "timeframe", None),
        getattr(builder, "sentiment", None),
        bars,
    )

    plan = await pipe.plan(builder, ohlcv=df)
    legacy = plan_to_signal_plan(plan)
    legacy["signals_available"] = len(kb.signals)
    # Keep the trace addressable on the builder via the dict — chat_service can
    # log / surface it without needing direct access to the StrategyPlan object.
    legacy["_planner_trace"] = plan.trace
    legacy["_sl_pct"] = plan.sl_pct
    legacy["_tp_pct"] = plan.tp_pct
    if session_id:
        remember_trace(session_id, plan.trace)

    logger.info(
        "legacy_bridge|exit|session_id=%s|signals_used=%s|sl=%s%%|tp=%s%%",
        session_id, legacy.get("signals_used", []),
        plan.sl_pct, plan.tp_pct,
    )
    return legacy


__all__ = [
    "plan_signals_v2",
    "UnsupportedStock",
    "UnsupportedTimeframe",
    "NoValidCandidate",
]
