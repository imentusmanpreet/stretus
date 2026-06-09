"""
app/planner/condition_satisfiability.py — "can this entry ever fire?" check.

The planner can assemble an entry_condition that is syntactically valid but
LOGICALLY can never be true — e.g. ``RSI(14) >= 0 AND RSI(14) <= 0`` (a
collapsed [0,0] band), or ``CLOSE > EMA(20) AND CLOSE < EMA(20)``. Such a
strategy backtests to ZERO trades and silently wastes the user's time.

We catch it behaviourally: build ONE rich synthetic OHLCV probe that sweeps the
full range of market conditions (strong up/down trends, reversals, RSI from
oversold to overbought, volume spikes, and explicit bullish candlestick
patterns), then evaluate the condition on every bar EXACTLY the way the real
backtest does (compile_condition + add_all_indicators + per-bar evaluate). If it
fires on zero bars across this rich probe, it is effectively unsatisfiable.

No-raise contract: if the quant engine can't be imported, or anything goes
wrong, we return None (no finding) rather than break strategy assembly.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _probe_df():
    """A deterministic, deliberately-varied OHLCV frame (cached). Engineered so
    that any *reasonable* entry condition fires at least once: it trends up, then
    crashes, then chops, then V-recovers — sweeping RSI ~5..95, crossing its EMAs
    and VWAP many times — with periodic volume spikes and a few hand-placed
    bullish engulfing / hammer candles so candlestick conditions can fire too."""
    import numpy as np
    import pandas as pd

    n = 800
    idx = pd.date_range("2026-01-01 03:45", periods=n, freq="5min", tz="UTC")
    t = np.arange(n)
    # Trend that rises, falls hard, then recovers — plus a slow sine and noise.
    trend = np.piecewise(
        t.astype(float),
        [t < 200, (t >= 200) & (t < 400), (t >= 400) & (t < 600), t >= 600],
        [lambda x: 0.10 * x,                       # steady rise
         lambda x: 20 - 0.18 * (x - 200),          # sharp drop
         lambda x: -16 + 6 * np.sin((x - 400) / 15.0),  # choppy/range
         lambda x: -16 + 0.14 * (x - 600)],        # V-recovery
    )
    rng = np.random.RandomState(20260530)
    close = 100.0 + trend + np.cumsum(rng.randn(n) * 0.15)
    close = np.maximum(close, 1.0)
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) + np.abs(rng.randn(n) * 0.3) + 0.05
    low = np.minimum(openp, close) - np.abs(rng.randn(n) * 0.3) - 0.05
    volume = 1e5 + np.abs(rng.randn(n)) * 2e4
    volume[::37] *= 6.0  # periodic volume spikes so volume-ratio conditions fire

    # Hand-place a few unmistakable BULLISH ENGULFING candles (a small down
    # candle followed by a big up candle that engulfs it) so CDL_ENGULFING>0 fires.
    for k in (120, 250, 470, 640):
        openp[k - 1], close[k - 1] = 100.0, 99.4          # small red
        openp[k], close[k] = 99.2, 101.2                  # big green engulfing
        high[k] = close[k] + 0.1; low[k] = openp[k] - 0.1
        high[k - 1] = openp[k - 1] + 0.1; low[k - 1] = close[k - 1] - 0.1

    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def entry_fire_count(entry_condition: str) -> int | None:
    """How many bars of the probe the condition fires on. None if it can't be
    evaluated (engine unavailable or condition won't compile). Raises nothing."""
    try:
        compiled = _compile(entry_condition)
    except Exception:
        return None
    if compiled is None:
        return None
    try:
        return _fire_count(compiled)
    except Exception as exc:  # noqa: BLE001 — no-raise contract
        logger.debug("condition_satisfiability|probe_failed|%s|%r", entry_condition, exc)
        return None


def _compile(entry_condition: str):
    """compile_condition() the entry rule. Returns the compiled condition, or None
    for an empty rule. Raises the engine's ParseError on an invalid/unknown
    function so callers can classify it. Returns None if the engine is missing."""
    if not entry_condition or not str(entry_condition).strip():
        return None
    try:
        from engine.conditions import compile_condition
    except Exception:  # engine not importable in this context
        return None
    return compile_condition(entry_condition)


def _fire_count(compiled) -> int:
    """Bars of the probe the (already-compiled) condition fires on."""
    from engine.conditions import ensure_scalar_columns
    from engine.indicators import add_all_indicators

    df = _probe_df().copy()
    cfg: dict[str, list[int]] = {}
    for ref in compiled.indicator_refs:
        cfg.setdefault(ref.name, [])
        if ref.period not in cfg[ref.name]:
            cfg[ref.name].append(ref.period)
    if cfg:
        df = add_all_indicators(df, cfg)
    ensure_scalar_columns(df, compiled.scalar_refs)
    n = len(df)
    return sum(1 for i in range(n) if compiled.evaluate(df, i))


def check_entry_satisfiable(entry_condition: str) -> dict[str, Any] | None:
    """Return a finding dict when the entry condition is INVALID (won't compile —
    e.g. references an unknown indicator like KC_UPPER) or can NEVER fire (zero
    trades), else None. Shape matches FidelityFinding fields.

    No-raise: if the quant engine can't be imported we return None (can't judge).
    """
    if not entry_condition or not str(entry_condition).strip():
        return None

    # 1. Does it even compile? An unknown/unsupported function (e.g. a half-wired
    # indicator the assembler emitted) means the strategy can't run at all.
    try:
        compiled = _compile(entry_condition)
    except Exception as exc:  # ParseError: unknown function / bad syntax
        return {
            "severity": "critical",
            "code": "entry_condition_invalid",
            "field": "entry_condition",
            "message": (
                "The assembled entry rule references something the engine can't "
                "run, so this strategy **cannot be backtested**. This usually "
                "means an indicator was requested that isn't wired up "
                "(check the indicator/function names).\n\n"
                f"Problem: {exc}\n\nEntry rule: `{entry_condition}`"
            ),
            "assembled_value": entry_condition,
        }
    if compiled is None:
        return None  # empty rule, or engine unavailable → can't judge

    # 2. It compiles — but does it ever fire across a rich probe?
    try:
        fires = _fire_count(compiled)
    except Exception as exc:  # noqa: BLE001
        logger.debug("condition_satisfiability|probe_failed|%s|%r", entry_condition, exc)
        return None
    if fires == 0:
        return {
            "severity": "critical",
            "code": "entry_never_fires",
            "field": "entry_condition",
            "message": (
                "The assembled entry rule can **never become true**, so this "
                "strategy would take **zero trades**. This usually means two "
                "conditions contradict each other (e.g. an indicator required to "
                "be both ≥ X and ≤ X). Please review the entry logic before "
                "running a backtest.\n\n"
                f"Entry rule: `{entry_condition}`"
            ),
            "assembled_value": entry_condition,
        }
    return None
