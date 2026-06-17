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
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Extracts numeric args from indicator calls in formula strings, e.g.
# "EMA(9) > EMA(21) AND RSI(14) > 45" → [9, 21, 14]
_FORMULA_WINDOW_RE = re.compile(r"\b(?:EMA|SMA|RSI|ATR|VWAP|MACD|BB|BBANDS|WMA|HMA|DEMA|TEMA|KAMA|STOCH|ADX|CCI|MFI|OBV|CMF|DONCHIAN)\s*\(\s*(\d+)", re.IGNORECASE)

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
    """Return the largest window size extracted from a signal's params.

    For `condition` type rules the params carry a formula string instead of
    discrete window keys — parse the formula to find indicator call arguments.
    """
    # Named-param signals (KB structured signals)
    windows = [
        int(v) for k, v in params.items()
        if k in _WINDOW_PARAM_NAMES and isinstance(v, (int, float)) and v > 0
    ]
    if windows:
        return max(windows)

    # Formula-string conditions: parse "EMA(9) > EMA(21) AND RSI(14) > 45" → [9, 21, 14]
    formula = params.get("formula", "")
    if formula:
        formula_windows = [int(m) for m in _FORMULA_WINDOW_RE.findall(str(formula)) if int(m) > 0]
        if formula_windows:
            return max(formula_windows)

    return _SIGNAL_MIN_WINDOWS.get(signal_type, _DEFAULT_WINDOW)


# Minimum candles to fetch regardless of window size.  EMA and RSI are
# exponentially weighted — they need many bars to converge to values that
# match a charting platform (which initialises from full history).
# 300 bars covers ~200 "warmup" candles above max_window for common periods.
_MIN_LOOKBACK = 300
# Multiplier over max_window to allow indicator convergence.
_LOOKBACK_MULTIPLIER = 10


def compute_lookback(signal_rules: List[Dict[str, Any]]) -> int:
    """
    Given a flat list of signal rule dicts (triggers + filters), return the
    minimum number of candles that must be fetched so every signal has enough
    data AND its computed values converge to chart-platform values.

    Formula: max(max_window * MULTIPLIER, MIN_LOOKBACK)
    """
    if not signal_rules:
        return _MIN_LOOKBACK

    per_signal = {
        r.get("type", "?"): _signal_window(r.get("type", ""), r.get("params", {}))
        for r in signal_rules
    }
    max_window = max(per_signal.values())
    lookback = max(max_window * _LOOKBACK_MULTIPLIER, _MIN_LOOKBACK)
    logger.info("compute_lookback | per_signal=%s max_window=%d lookback=%d", per_signal, max_window, lookback)
    return lookback


def list_registered_signals() -> List[str]:
    """Return all signal names currently in the registry."""
    return RuleRegistry.list_all()
