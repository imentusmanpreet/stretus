"""Phase 8a — near_52_week_high / near_52_week_low signal formulas.

These signals use the existing AST primitives (MAX/MIN) — we just verify the
proximity math is right and that the formulas evaluate cleanly. Real-world
use is via an HTF: 1d gate so the lookback aligns with calendar weeks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from engine.conditions import evaluate_condition


def _df_with_highs(highs: list[float]) -> pd.DataFrame:
    """Each bar's high == the given value; close==high, low/open arbitrary."""
    return pd.DataFrame({
        "open":   highs,
        "high":   highs,
        "low":    [h - 0.5 for h in highs],
        "close":  highs,
        "volume": [1000.0] * len(highs),
    })


def _df_with_lows(lows: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "open":   lows,
        "high":   [l + 0.5 for l in lows],
        "low":    lows,
        "close":  lows,
        "volume": [1000.0] * len(lows),
    })


# ── near_52_week_high (uses MAX(HIGH, window) * proximity threshold) ────────


def test_near_52_week_high_fires_when_close_is_within_2_percent():
    """Build a series whose 52-bar high is 100. A close at 99 (1% below) → True.
    A close at 95 (5% below) → False."""
    # 51 bars rising to 100, then a current bar at 99 — 1% below max
    highs = list(range(50, 100)) + [100.0, 99.0]
    df = _df_with_highs(highs)
    # MAX(HIGH, 52) at index 51 = 100; CLOSE = 99; 99 >= 100 * 0.98 (=98) → True
    assert evaluate_condition("CLOSE >= MAX(HIGH, 52) * 0.98", df, 51) is True


def test_near_52_week_high_does_not_fire_when_close_is_too_far():
    highs = list(range(50, 100)) + [100.0, 95.0]
    df = _df_with_highs(highs)
    # CLOSE = 95; 95 >= 98 → False
    assert evaluate_condition("CLOSE >= MAX(HIGH, 52) * 0.98", df, 51) is False


def test_near_52_week_high_at_exact_threshold():
    highs = list(range(50, 100)) + [100.0, 98.0]
    df = _df_with_highs(highs)
    # CLOSE = 98; 98 >= 98 → True (boundary inclusive)
    assert evaluate_condition("CLOSE >= MAX(HIGH, 52) * 0.98", df, 51) is True


def test_near_52_week_high_fires_on_a_new_high():
    """When today's bar IS the 52w high (or above), the proximity still holds.
    Need at least 52 bars of history before the test index — MAX(HIGH, 52)
    returns NaN otherwise."""
    highs = [50.0] * 52 + [101.0]      # 52 flat bars at 50, then new high at index 52
    df = _df_with_highs(highs)
    # MAX(HIGH, 52) at index 52 includes the spike → 101; CLOSE = 101 → True
    assert evaluate_condition("CLOSE >= MAX(HIGH, 52) * 0.98", df, 52) is True


# ── near_52_week_low (uses MIN(LOW, window) * proximity, proximity > 1) ─────


def test_near_52_week_low_fires_when_close_is_within_2_percent():
    """52-bar low is 50. Close at 51 (~2% above) is within proximity 1.02 → True."""
    lows = list(range(100, 50, -1)) + [50.0, 51.0]
    df = _df_with_lows(lows)
    # MIN(LOW, 52) at index 51 = 50; CLOSE = 51; 51 <= 50 * 1.02 (=51) → True
    assert evaluate_condition("CLOSE <= MIN(LOW, 52) * 1.02", df, 51) is True


def test_near_52_week_low_does_not_fire_when_close_is_too_far():
    lows = list(range(100, 50, -1)) + [50.0, 55.0]
    df = _df_with_lows(lows)
    # CLOSE = 55; 55 <= 51 → False
    assert evaluate_condition("CLOSE <= MIN(LOW, 52) * 1.02", df, 51) is False


# ── RSI 60 / 40 thresholds compile and resolve via the existing AST ─────────


def test_rsi_above_60_compiles_and_evaluates():
    """RSI > 60 needs no new AST work — same formula shape as rsi_above_50.
    Need at least one down bar in the window or RSI is NaN (avg_loss == 0 → div by 0)."""
    # Mostly up with one small down bar at index 5
    closes = [100.0 + i for i in range(30)]
    closes[5] = closes[4] - 0.1            # one tiny dip
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low":  [c - 0.5 for c in closes], "close": closes,
        "volume": [1000.0] * len(closes),
    })
    assert evaluate_condition("RSI(14) > 60", df, 29) is True


def test_rsi_below_40_compiles_and_evaluates():
    closes = [100.0 - i for i in range(30)]
    closes[5] = closes[4] + 0.1            # one tiny up bar so RSI is defined
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low":  [c - 0.5 for c in closes], "close": closes,
        "volume": [1000.0] * len(closes),
    })
    assert evaluate_condition("RSI(14) < 40", df, 29) is True
