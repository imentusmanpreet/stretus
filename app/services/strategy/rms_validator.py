"""
app/services/strategy/rms_validator.py — sanity validation for risk-management
settings (RMS) extracted from user input.

The chat flow now treats the LLM as the *primary* extractor of RMS numbers
(stop loss, take profit, risk:reward, per-trade risk, daily loss cap, max
trades). LLMs are good at reading intent from free-form text but can transpose
or hallucinate a digit, so every captured value passes through this validator
before it is stored.

Contract: ``validate_rms(field, value)`` returns an :class:`RmsValidation`.

  * ``status == "ok"``                → store ``value`` (possibly normalised).
  * ``status == "needs_confirmation"``→ the value parsed but falls outside the
                                        sane range; DO NOT store it silently —
                                        ask the user to confirm/correct first.
  * ``status == "invalid"``           → the value is not usable at all
                                        (non-numeric / non-positive); drop it.

Ranges are intentionally wide: the goal is to catch typos and unit confusion
(e.g. "50% daily loss cap" or "200% take profit"), not to enforce a house view
on what a "good" risk number is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.kb.compat import get_daily_loss_cap_bounds

# Per-field sane bounds for percentage / ratio fields, as (low, high) inclusive.
# ``daily_loss_cap`` is sourced from the KB so it stays consistent with the
# clamp already applied by StrategyBuilder.apply_defaults.
_DAILY_LOSS_CAP_LOW, _DAILY_LOSS_CAP_HIGH = get_daily_loss_cap_bounds()  # 1.0 .. 5.0

_PCT_BOUNDS: dict[str, tuple[float, float]] = {
    "stop_loss_pct": (0.05, 25.0),
    "take_profit_pct": (0.05, 100.0),
    "per_trade_risk": (0.05, 10.0),
    "daily_loss_cap": (_DAILY_LOSS_CAP_LOW, _DAILY_LOSS_CAP_HIGH),
    "risk_reward": (0.1, 20.0),
}

# Fields that are non-negative integers.
_INT_FIELDS = {"max_trades"}
_MAX_TRADES_HIGH = 200

# Public set of fields this validator understands.
RMS_FIELDS = frozenset(_PCT_BOUNDS) | _INT_FIELDS


@dataclass(frozen=True)
class RmsValidation:
    """Outcome of validating one RMS field."""

    field: str
    status: str  # "ok" | "needs_confirmation" | "invalid"
    value: float | int | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def needs_confirmation(self) -> bool:
        return self.status == "needs_confirmation"


def _coerce_number(raw: Any) -> float | None:
    """Best-effort numeric coercion. Accepts the agent's ``{"value": n}`` shape
    and plain strings like ``"1%"``; returns ``None`` when not numeric."""
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is an int subclass — never an RMS value
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict) and "value" in raw:
        return _coerce_number(raw["value"])
    if isinstance(raw, str):
        cleaned = raw.strip().rstrip("%").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def validate_rms(field: str, value: Any) -> RmsValidation:
    """Validate a single RMS field. See module docstring for the contract."""
    if field not in RMS_FIELDS:
        return RmsValidation(field, "invalid", None, f"unknown RMS field {field!r}")

    num = _coerce_number(value)
    if num is None:
        return RmsValidation(field, "invalid", None, "value is not numeric")

    if field in _INT_FIELDS:
        ivalue = int(round(num))
        if ivalue < 0:
            return RmsValidation(field, "invalid", None, "max_trades cannot be negative")
        if ivalue > _MAX_TRADES_HIGH:
            return RmsValidation(
                field,
                "needs_confirmation",
                ivalue,
                f"{ivalue} trades/day is unusually high (>{_MAX_TRADES_HIGH})",
            )
        return RmsValidation(field, "ok", ivalue)

    # Percentage / ratio fields.
    if num <= 0:
        return RmsValidation(field, "invalid", None, f"{field} must be greater than 0")

    low, high = _PCT_BOUNDS[field]
    if num < low or num > high:
        return RmsValidation(
            field,
            "needs_confirmation",
            num,
            f"{field}={num:g} is outside the expected range {low:g}-{high:g}",
        )
    return RmsValidation(field, "ok", num)
