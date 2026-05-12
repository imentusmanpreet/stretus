"""
app/services/execution/indicator_engine.py
───────────────────────────────────────────
Utility layer between the RuleEngine and stretus_kb signals.

Responsibilities:
  1. compute_lookback()  — inspect signal params to determine how many
                           candles the rule engine needs.
  2. ensure_signal_imports() — ensure all stretus_kb signals are registered
                               before the first RuleRegistry.evaluate() call.

The actual evaluation is delegated to stretus_kb.RuleRegistry, which already
knows every registered signal.  This module does NOT duplicate indicator logic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Importing stretus_knowledge_base.stretus_kb triggers its __init__.py, which
# auto-registers all signals via @RuleRegistry.register decorators.
try:
    from stretus_knowledge_base.stretus_kb import RuleRegistry
except ImportError as exc:
    logger.warning("Could not import stretus_kb: %s", exc)
    RuleRegistry = None  # type: ignore[assignment,misc]


# ── Lookback calculation ──────────────────────────────────────────────────────

# Known param names that represent indicator window sizes
_WINDOW_PARAM_NAMES = {"window", "window_fast", "window_slow", "window_sign", "period"}

# Hard-coded minimum windows for signals that don't expose a 'window' param
_SIGNAL_MIN_WINDOWS: Dict[str, int] = {
    "macd_bullish_cross":     35,  # slow(26) + sign(9)
    "macd_bearish_cross":     35,
    "macd_histogram_positive": 35,
    "macd_histogram_negative": 35,
    "vwap_above":              1,
    "vwap_below":              1,
    "opening_range_breakout":  1,
    "circuit_proximity":       1,
    "nse_market_hours":        1,
    "expiry_day":              1,
}

_DEFAULT_WINDOW = 20


def _signal_window(signal_type: str, params: Dict[str, Any]) -> int:
    """Return the largest window size extracted from a signal's params."""
    windows = [
        int(v) for k, v in params.items()
        if k in _WINDOW_PARAM_NAMES and isinstance(v, (int, float)) and v > 0
    ]
    if windows:
        return max(windows)
    return _SIGNAL_MIN_WINDOWS.get(signal_type, _DEFAULT_WINDOW)


def compute_lookback(signal_rules: List[Dict[str, Any]]) -> int:
    """
    Given a flat list of signal rule dicts (triggers + filters), return the
    minimum number of candles that must be fetched so every signal has enough
    data.

    Formula: max(window per signal) * 2 + 20 buffer
    """
    if not signal_rules:
        return _DEFAULT_WINDOW * 2 + 20

    max_window = max(
        _signal_window(r.get("type", ""), r.get("params", {}))
        for r in signal_rules
    )
    return max_window * 2 + 20


def list_registered_signals() -> List[str]:
    """Return all signal names currently in the registry."""
    return RuleRegistry.list_all()
