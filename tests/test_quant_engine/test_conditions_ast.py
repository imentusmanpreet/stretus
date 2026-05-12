"""Phase 2 AST primitives: STDEV, ZSCORE, PREV(EXPR, k)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from engine.conditions import (
    build_arrays_from_df,
    compile_condition,
    evaluate_condition,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _trending_close(n: int = 60) -> pd.DataFrame:
    """Monotonically rising series — perfect for slope and z-score tests."""
    closes = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "open":  closes,
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def _flat_close(n: int = 60, value: float = 100.0) -> pd.DataFrame:
    closes = [value] * n
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * n,
    })


# ── STDEV ─────────────────────────────────────────────────────────────────────


def test_stdev_zero_for_flat_series():
    df = _flat_close()
    # On a flat series the rolling stdev is zero. The condition `STDEV(...) == 0`
    # exercises the parser; the comparison itself returns True.
    assert evaluate_condition("STDEV(CLOSE, 20) == 0", df, 30) is True


def test_stdev_positive_for_trending_series():
    df = _trending_close()
    assert evaluate_condition("STDEV(CLOSE, 20) > 0", df, 30) is True


def test_stdev_returns_nan_before_warmup():
    """With period=20, bar 5 doesn't have enough history. NaN comparisons are
    False so the condition should never fire."""
    df = _trending_close()
    assert evaluate_condition("STDEV(CLOSE, 20) > 0", df, 5) is False


# ── ZSCORE ────────────────────────────────────────────────────────────────────


def test_zscore_oversold_below_lower_threshold():
    """A sharp drop after a long flat run should produce a strongly negative
    z-score — typical mean-reversion entry trigger."""
    closes = [100.0] * 30 + [95.0]      # 30 flat bars, then a drop
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * len(closes),
    })
    z = evaluate_condition("ZSCORE(CLOSE, 20) < -1", df, 30)
    # Stdev of the rolling window includes the drop, so z is large negative
    assert z is True


def test_zscore_returns_nan_when_stdev_is_zero():
    df = _flat_close()
    # On a flat series stdev=0, z-score is undefined (NaN). Comparisons → False.
    assert evaluate_condition("ZSCORE(CLOSE, 20) < -1", df, 30) is False
    assert evaluate_condition("ZSCORE(CLOSE, 20) > 1",  df, 30) is False


def test_zscore_overbought_for_strong_uptrend():
    """A spike at the end of an otherwise flat run should produce a strongly
    positive z-score."""
    closes = [100.0] * 30 + [110.0]
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * len(closes),
    })
    assert evaluate_condition("ZSCORE(CLOSE, 20) > 1", df, 30) is True


# ── PREV with expression argument ─────────────────────────────────────────────


def test_prev_field_still_works():
    """Backward-compat: PREV(CLOSE, 1) on a rising series gives yesterday's close."""
    df = _trending_close()
    # close[10] = 105.0; close[9] = 104.5 → CLOSE > PREV(CLOSE, 1) is True
    assert evaluate_condition("CLOSE > PREV(CLOSE, 1)", df, 10) is True


def test_prev_function_for_ema_slope_check():
    """Slope-of-EMA — the headline use case. Rising series ⇒ EMA(20) rising
    ⇒ EMA(20) > PREV(EMA(20), 3)."""
    df = _trending_close()
    assert evaluate_condition("EMA(20) > PREV(EMA(20), 3)", df, 40) is True


def test_prev_function_returns_false_on_falling_series():
    closes = [200.0 - i * 0.5 for i in range(60)]
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * len(closes),
    })
    assert evaluate_condition("EMA(20) > PREV(EMA(20), 3)", df, 40) is False


def test_prev_function_handles_zero_offset():
    df = _trending_close()
    # PREV(EMA(20), 0) == EMA(20)
    assert evaluate_condition("EMA(20) >= PREV(EMA(20), 0)", df, 30) is True


def test_prev_function_returns_nan_before_warmup():
    df = _trending_close()
    # bar 2 with offset 3 → prev_i = -1, NaN, condition False
    assert evaluate_condition("EMA(20) > PREV(EMA(20), 3)", df, 2) is False


# ── compile_condition flags ───────────────────────────────────────────────────


def test_compile_marks_stdev_as_fast_path_unsafe():
    cond = compile_condition("STDEV(CLOSE, 20) > 0.5")
    assert cond is not None
    assert cond.fast_path_safe is False


def test_compile_marks_zscore_as_fast_path_unsafe():
    cond = compile_condition("ZSCORE(CLOSE, 20) < -2")
    assert cond is not None
    assert cond.fast_path_safe is False


def test_compile_marks_prev_function_as_fast_path_unsafe():
    cond = compile_condition("EMA(20) > PREV(EMA(20), 3)")
    assert cond is not None
    assert cond.fast_path_safe is False
    # Inner EMA(20) should still be discovered for precomputation.
    refs = {(r.name, r.period) for r in cond.indicator_refs}
    assert ("EMA", 20) in refs


def test_compile_keeps_prev_field_fast_path_safe():
    cond = compile_condition("CLOSE > PREV(CLOSE, 1)")
    assert cond is not None
    assert cond.fast_path_safe is True


# ── Compiled fast/slow path agreement ─────────────────────────────────────────


def test_compiled_zscore_via_evaluate_matches_slow_path():
    """compile_condition().evaluate() should produce the same answer as
    evaluate_condition() since both use the same underlying AST."""
    closes = [100.0] * 25 + [108.0]
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * len(closes),
    })
    cond = compile_condition("ZSCORE(CLOSE, 20) > 1")
    assert cond is not None
    assert cond.evaluate(df, 25) is True
    assert evaluate_condition("ZSCORE(CLOSE, 20) > 1", df, 25) is True
