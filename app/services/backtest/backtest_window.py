"""
Resolve and validate backtest date windows for fetch and simulation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class BacktestWindowError(ValueError):
    """User-facing validation failure for a requested backtest window."""


def _parse_utc_timestamp(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_utc_iso(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def default_backtest_window() -> tuple[str, str]:
    """Return the product default inclusive UTC window (from config ? now)."""
    from quant_engine.engine.config import (
        BACKTEST_MARKET_DATA_FROM_UTC,
        BACKTEST_MARKET_DATA_TO_UTC,
    )

    return BACKTEST_MARKET_DATA_FROM_UTC, BACKTEST_MARKET_DATA_TO_UTC


def _earliest_allowed_start() -> datetime:
    return _parse_utc_timestamp(
        settings.backtest_earliest_from_utc,
        "backtest_earliest_from_utc",
    ) or datetime(2024, 1, 1, tzinfo=timezone.utc)


def earliest_backtest_from_display() -> str:
    """Human-readable earliest date for error messages."""
    return _earliest_allowed_start().strftime("%d %b %Y")


def resolve_backtest_window(
    from_utc: str | None,
    to_utc: str | None,
    *,
    interval: str = "1d",
    indicators: dict | None = None,
    objective: str = "positional",
) -> tuple[str, str]:
    """
    Resolve the simulation/report window for a backtest.

    If only one bound is provided, the other uses the product default
    (``BACKTEST_MARKET_DATA_FROM_UTC`` or current UTC for ``to``).
    If neither is provided, the full default window applies.
    """
    del interval, indicators, objective

    default_from, default_to = default_backtest_window()
    start_dt = _parse_utc_timestamp(from_utc, "from_utc")
    end_dt = _parse_utc_timestamp(to_utc, "to_utc")

    if start_dt is None and end_dt is None:
        logger.info("🕒 Backtest window | using product default | from=%s to=%s", default_from, default_to)
        return default_from, default_to

    if start_dt is None:
        start_dt = _parse_utc_timestamp(default_from, "from_utc")
    if end_dt is None:
        end_dt = _parse_utc_timestamp(default_to, "to_utc")

    now = datetime.now(timezone.utc).replace(microsecond=0)
    earliest = _earliest_allowed_start()

    if start_dt < earliest:
        raise BacktestWindowError(
            f"We currently support backtests from {earliest_backtest_from_display()} onward. "
            f"Please choose a start date on or after that date."
        )

    if start_dt >= end_dt:
        raise BacktestWindowError("The backtest start date must be earlier than the end date.")

    clamped_to_now = end_dt > now
    if clamped_to_now:
        end_dt = now

    if start_dt >= end_dt:
        raise BacktestWindowError(
            "The backtest end date must be after the start date (and not in the future)."
        )

    resolved_from, resolved_to = _to_utc_iso(start_dt), _to_utc_iso(end_dt)
    logger.info(
        "🕒 Backtest window resolved | from=%s to=%s%s",
        resolved_from,
        resolved_to,
        " (end clamped to now)" if clamped_to_now else "",
    )
    return resolved_from, resolved_to


def extend_fetch_start(
    from_utc: str,
    *,
    user_specified_window: bool = False,
    interval: str = "1d",
    indicators: dict | None = None,
    strategy: dict | None = None,
    objective: str = "positional",
) -> str:
    """Move fetch start earlier for indicator warm-up before the simulation window.

    Only applies when ``user_specified_window`` is True (caller gave both from/to).
    Default product windows are fetched as-is from the configured start date.
    """
    if not user_specified_window:
        return from_utc

    from app.services.backtest.market_data import _compute_user_range_padding_days

    start_dt = _parse_utc_timestamp(from_utc, "from_utc")
    if start_dt is None:
        return from_utc

    padding_days = _compute_user_range_padding_days(
        interval,
        indicators or {},
        strategy=strategy,
        objective=objective,
    )
    extended = start_dt - timedelta(days=padding_days)
    earliest = _earliest_allowed_start()
    if extended < earliest:
        extended = earliest
    return _to_utc_iso(extended)
