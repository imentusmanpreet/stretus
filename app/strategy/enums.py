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

from functools import lru_cache
from pathlib import Path

import yaml

from app.strategy import engine_bridge

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
    tfs = supported_timeframes()
    return bool(tfs) and str(timeframe).strip() in tfs


def normalise_market(market: str) -> str:
    """Engine-canonical market id (e.g. 'NSE' / 'nse' → 'indian_stocks')."""
    return engine_bridge.normalise_market(market)


def is_supported_market(market: str) -> bool:
    """True when the market normalises to a canonical engine market id."""
    return normalise_market(market) in CANONICAL_MARKETS
