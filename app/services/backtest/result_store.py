"""
Persist backtest runs to ai_strategy.backtest: metrics and summary only (no per-trade rows).

Used by the HTTP backtest callback and the chat (sync) backtest flow.
"""
from __future__ import annotations

import copy
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.strategy import Backtest, BacktestStatus, Strategy, StrategyStatus

logger = logging.getLogger(__name__)

_TRADE_OUTCOME_ALIAS_KEYS = (
    "expectancy_pct",
    "average_outcome_per_trade_pct",
)
_TRADE_DURATION_ALIAS_KEYS = (
    "avg_trade_duration_days",
    "average_holding_duration_days",
)
_TRADE_OUTCOME_METRIC_KEYS = (
    "average_outcome_per_trade",
    *_TRADE_OUTCOME_ALIAS_KEYS,
)
_TRADE_DURATION_METRIC_KEYS = (
    "average_holding_duration",
    *_TRADE_DURATION_ALIAS_KEYS,
)
_DUPLICATE_TRADE_ACTIVITY_METRIC_KEYS = (
    *_TRADE_OUTCOME_ALIAS_KEYS,
    *_TRADE_DURATION_ALIAS_KEYS,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_metric_number(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    zero_value: float | None = None
    for key in keys:
        value = _coerce_float(metrics.get(key))
        if value is None:
            continue
        if value != 0.0:
            return value
        if zero_value is None:
            zero_value = value
    return zero_value


def _trade_date_duration_days(trade: dict[str, Any]) -> float | None:
    try:
        entry_raw = str(trade.get("entry_date") or "").replace("Z", "+00:00")
        exit_raw = str(trade.get("exit_date") or "").replace("Z", "+00:00")
        if not entry_raw or not exit_raw:
            return None
        entry = datetime.fromisoformat(entry_raw)
        exit_ = datetime.fromisoformat(exit_raw)
        return max(0.0, (exit_ - entry).total_seconds() / 86_400.0)
    except Exception:
        return None


def _average_trade_outcome_pct(trades: list[Any]) -> float | None:
    values: list[float] = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        value = _coerce_float(trade.get("outcome_pct"))
        if value is None:
            value = _coerce_float(trade.get("pnl_pct"))
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def _average_trade_duration_days(trades: list[Any]) -> float | None:
    values: list[float] = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        value = _trade_date_duration_days(trade)
        if value is None:
            value = _coerce_float(trade.get("holding_duration_days"))
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def _set_metric_aliases(
    metrics: dict[str, Any],
    aliases: tuple[str, ...],
    value: float | None,
    *,
    ndigits: int,
) -> None:
    if value is None:
        return
    rounded_value = round(float(value), ndigits)
    for alias in aliases:
        metrics[alias] = rounded_value


def normalize_backtest_metric_aliases(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Backfill user-facing trade activity metric names.

    The canonical output names are `average_outcome_per_trade` and
    `average_holding_duration`. Older payloads may still contain legacy aliases;
    use them as inputs, then remove those duplicate keys from the response.
    """
    if not isinstance(result, dict):
        return result

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return result

    trades = metrics.get("backtest_trades")
    trade_list = trades if isinstance(trades, list) else []

    average_outcome = _first_metric_number(
        metrics,
        _TRADE_OUTCOME_METRIC_KEYS,
    )
    if (average_outcome is None or average_outcome == 0.0) and trade_list:
        average_outcome = _average_trade_outcome_pct(trade_list)
    if average_outcome is None:
        average_outcome = 0.0

    average_duration = _first_metric_number(
        metrics,
        _TRADE_DURATION_METRIC_KEYS,
    )
    if (average_duration is None or average_duration == 0.0) and trade_list:
        average_duration = _average_trade_duration_days(trade_list)
    if average_duration is None:
        average_duration = 0.0

    _set_metric_aliases(
        metrics,
        ("average_outcome_per_trade",),
        average_outcome,
        ndigits=4,
    )
    _set_metric_aliases(
        metrics,
        ("average_holding_duration",),
        average_duration,
        ndigits=4,
    )

    for key in _DUPLICATE_TRADE_ACTIVITY_METRIC_KEYS:
        metrics.pop(key, None)

    return result


def summarize_backtest_for_db(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Return a copy of the quant-engine result safe to store in backtest.result_json
    (metrics summary only; drops metrics.backtest_trades). Chat API messages may
    still attach the full engine result separately for clients that need per-trade data.
    """
    if not isinstance(result, dict):
        return result

    out: dict[str, Any] = copy.deepcopy(result)
    normalize_backtest_metric_aliases(out)
    metrics = out.get("metrics")
    if isinstance(metrics, dict) and "backtest_trades" in metrics:
        metrics = dict(metrics)
        metrics.pop("backtest_trades", None)
        out["metrics"] = metrics
    return out


def _classify_backtest_status(sanitized: dict[str, Any]) -> tuple[BacktestStatus, bool]:
    """
    Receives already-sanitized (API) payload. Returns (row_status, is_execution_error).
    """
    failure_reason = str(sanitized.get("failure_reason") or "").strip()
    is_execution_error = failure_reason.startswith("Backtest execution failed:")
    if is_execution_error:
        return BacktestStatus.failed, True
    if failure_reason:
        # Strategy / threshold failure — still a completed backtest with diagnostics
        return BacktestStatus.completed, False
    return BacktestStatus.completed, False


def apply_sanitized_result_to_row(
    *,
    backtest: Backtest,
    strategy: Strategy | None,
    sanitized: dict[str, Any],
) -> None:
    """Set ORM fields from a sanitized result dict (used by API callback and in-memory tests)."""
    b_status, is_execution_error = _classify_backtest_status(sanitized)
    backtest.result_json = sanitized
    backtest.status = b_status
    backtest.error_message = str(sanitized.get("failure_reason") or "").strip() or None
    backtest.completed_at = _utcnow()

    if strategy:
        strategy.status = (
            StrategyStatus.confirmed if is_execution_error else StrategyStatus.backtest_complete
        )


async def insert_chat_backtest_row(
    db: AsyncSession,
    *,
    strategy_id: uuid.UUID,
    summary: dict[str, Any],
    started_at: datetime | None = None,
) -> uuid.UUID:
    """
    Create a backtest row for a chat-executed (sync) run. Commit is left to the caller.
    """
    s = summary if isinstance(summary, dict) else {}
    b_status, is_exec_err = _classify_backtest_status(s)
    fail_msg = str(s.get("failure_reason") or "").strip() or None
    backtest_id = uuid.uuid4()
    now = _utcnow()
    row = Backtest(
        id=backtest_id,
        strategy_id=strategy_id,
        status=b_status,
        result_json=summary,
        error_message=fail_msg,
        started_at=started_at or now,
        completed_at=now,
    )
    db.add(row)
    await db.flush()

    strategy = await db.get(Strategy, strategy_id)
    if strategy:
        strategy.status = StrategyStatus.confirmed if is_exec_err else StrategyStatus.backtest_complete

    logger.info(
        "backtest_persist|source=chat|outcome=row_inserted|backtest_id=%s|strategy_id=%s|"
        "status=%s|engine_ref=%s",
        backtest_id,
        strategy_id,
        b_status.value,
        s.get("backtest_ref_id"),
    )
    return backtest_id


async def insert_failed_chat_backtest(
    db: AsyncSession,
    *,
    strategy_id: uuid.UUID,
    error_message: str,
    started_at: datetime | None = None,
) -> uuid.UUID | None:
    """Record a failed chat backtest (orchestration / network / engine error)."""
    backtest_id = uuid.uuid4()
    now = _utcnow()
    row = Backtest(
        id=backtest_id,
        strategy_id=strategy_id,
        status=BacktestStatus.failed,
        result_json=None,
        error_message=error_message[:20000] if error_message else None,
        started_at=started_at or now,
        completed_at=now,
    )
    db.add(row)
    await db.flush()

    strategy = await db.get(Strategy, strategy_id)
    if strategy:
        strategy.status = StrategyStatus.confirmed

    logger.info(
        "backtest_persist|source=chat|outcome=failed|backtest_id=%s|strategy_id=%s|error_len=%s",
        backtest_id,
        strategy_id,
        len(error_message) if error_message else 0,
    )
    return backtest_id
