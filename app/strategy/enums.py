"""
app/strategy/enums.py — the legal enums for a StrategySpec, sourced from config/engine.

Timeframes are read at runtime from the existing ``app/kb/timeframes.yaml`` (the KB
stays in place; we only read it), so adding a timeframe there is automatically
honored here. Objective and direction literals mirror the engine's own contract
(``quant_engine/engine/loader.py``: ``StrategyConfig.objective`` /
``.direction``) — they are the structural data contract, not trading logic.
Market strings are normalised through the engine itself rather than enumerated.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from app.strategy import engine_bridge

# Range-based timeframe acceptance: the backtest fetches 1-minute data and the
# engine resamples up to any timeframe, so any well-formed 1m..1d interval is
# valid — not just the curated presets in timeframes.yaml.
_TF_RANGE_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_TF_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}
_TF_MIN_MINUTES = 1
_TF_MAX_MINUTES = 1440

# The engine's two objective outputs and three direction outputs. These match the
# literals in engine.loader (StrategyConfig.objective / .direction) and are kept
# here so the Pydantic model and the prompt agree with the backtester.
OBJECTIVES: tuple[str, ...] = ("intraday", "positional")
DIRECTIONS: tuple[str, ...] = ("long_only", "short_only", "both")

# Canonical market ids the engine's _normalise_market collapses to. Reference set
# for the prompt; market validation itself defers to the engine (see
# is_supported_market) so an unknown market is normalised, not rejected outright.
CANONICAL_MARKETS: tuple[str, ...] = (
    "indian_stocks",
    "indian_indices",
    "crypto",
    "us_stocks",
)

_TIMEFRAMES_YAML = (
    Path(__file__).resolve().parents[1] / "kb" / "timeframes.yaml"
)


@lru_cache(maxsize=1)
def supported_timeframes() -> tuple[str, ...]:
    """The ``supported`` timeframe list from app/kb/timeframes.yaml."""
    try:
        data = yaml.safe_load(_TIMEFRAMES_YAML.read_text(encoding="utf-8")) or {}
        values = data.get("supported") or []
        return tuple(str(v).strip() for v in values if str(v).strip())
    except Exception:  # noqa: BLE001 — config read must not crash the path
        return ()


def is_supported_timeframe(timeframe: str) -> bool:
    """True for any well-formed timeframe within the accepted range (1m..1d).

    Curated presets in timeframes.yaml match exactly; everything else is range-
    checked, so arbitrary intervals (2m, 7m, 45m, 4h) are accepted because the
    engine resamples them from 1-minute data.
    """
    raw = str(timeframe).strip()
    if not raw:
        return False
    if raw in supported_timeframes():
        return True
    match = _TF_RANGE_RE.match(raw)
    if not match:
        return False
    minutes = int(match.group(1)) * _TF_UNIT_MINUTES[match.group(2).lower()]
    return _TF_MIN_MINUTES <= minutes <= _TF_MAX_MINUTES


def normalise_market(market: str) -> str:
    """Engine-canonical market id (e.g. 'NSE' / 'nse' → 'indian_stocks')."""
    return engine_bridge.normalise_market(market)


def is_supported_market(market: str) -> bool:
    """True when the market normalises to a canonical engine market id."""
    return normalise_market(market) in CANONICAL_MARKETS
