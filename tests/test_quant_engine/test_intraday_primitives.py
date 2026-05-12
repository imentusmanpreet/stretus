"""Tests for the 9 intraday primitives added on top of the base engine:

  1. Pivot Points (PIVOT_P / R1 / R2 / S1 / S2)
  2. IN_TIME_WINDOW(start_hhmm, end_hhmm)
  3. VOL_SPIKE(window, multiplier)
  4. Gap detection + GAP_FILL_PCT
  5. Candlestick patterns (engulfing, hammer, shooting star)
  6. min_risk_reward gate
  7. RETEST_FROM_ABOVE / RETEST_FROM_BELOW
  8. Dynamic take-profit (level / expression / gap_fill)
  9. Multi-step entry sequence (state machine)
"""
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

from engine.conditions import evaluate_condition
from engine.entry_sequence import evaluate_entry_sequence, parse_entry_sequence
from engine.indicators import gap_series, pivot_points
from engine.loader import load_strategy_from_content
from engine.patterns import add_all_patterns


# ── Helpers ───────────────────────────────────────────────────────────────────


def _multi_day_df(rows_per_day: int = 5, days: int = 3) -> pd.DataFrame:
    """Build a multi-day intraday OHLCV frame with monotonically rising prices."""
    times: list[pd.Timestamp] = []
    base_dates = [
        f"2024-01-{d:02d}" for d in range(2, 2 + days)
    ]
    for d in base_dates:
        times += pd.date_range(
            f"{d} 09:15", periods=rows_per_day, freq="1h", tz="Asia/Kolkata"
        ).tolist()
    idx = pd.DatetimeIndex(times)
    closes = list(range(100, 100 + rows_per_day * days))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


# ── 1. Pivot Points ───────────────────────────────────────────────────────────


def test_pivot_points_uses_previous_session():
    df = _multi_day_df(rows_per_day=5, days=2)
    piv = pivot_points(df)
    # Day 1 H/L/C: H=max(101..105)=105, L=min(99..103)=99, C=104
    expected_p = (105 + 99 + 104) / 3.0
    assert pytest.approx(piv.iloc[5]["PIVOT_P"], abs=1e-9) == expected_p
    # R1, S1 cross-check
    assert pytest.approx(piv.iloc[5]["PIVOT_R1"], abs=1e-9) == 2 * expected_p - 99
    assert pytest.approx(piv.iloc[5]["PIVOT_S1"], abs=1e-9) == 2 * expected_p - 105


def test_pivot_points_first_session_is_nan():
    df = _multi_day_df(rows_per_day=5, days=2)
    piv = pivot_points(df)
    # No prior session for the first day → all pivots NaN.
    assert pd.isna(piv.iloc[0]["PIVOT_P"])


# ── 2. IN_TIME_WINDOW ─────────────────────────────────────────────────────────


def test_in_time_window_inclusive_bounds():
    idx = pd.date_range("2024-01-02 09:15", periods=8, freq="15min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        index=idx,
    )
    # 09:15 falls within [915, 930]; 09:45 does not.
    assert evaluate_condition("IN_TIME_WINDOW(915, 930)", df, 0) is True
    assert evaluate_condition("IN_TIME_WINDOW(915, 930)", df, 1) is True
    assert evaluate_condition("IN_TIME_WINDOW(915, 930)", df, 2) is False


def test_in_time_window_rejects_wrap_around():
    idx = pd.date_range("2024-01-02 09:15", periods=3, freq="15min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        index=idx,
    )
    # End < start (would wrap midnight) → always False.
    assert evaluate_condition("IN_TIME_WINDOW(1500, 0930)", df, 0) is False


# ── 3. VOL_SPIKE ──────────────────────────────────────────────────────────────


def test_vol_spike_fires_on_3x_baseline():
    idx = pd.date_range("2024-01-02 09:15", periods=10, freq="15min", tz="Asia/Kolkata")
    volumes = [1000] * 6 + [3500, 800, 5000, 900]
    df = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": volumes},
        index=idx,
    )
    # Bar 6: baseline avg(volume[1..5]) = 1000; current=3500 ≥ 1.5×1000 → True
    assert evaluate_condition("VOL_SPIKE(5, 1.5)", df, 6) is True
    # Bar 7: current 800 < threshold → False
    assert evaluate_condition("VOL_SPIKE(5, 1.5)", df, 7) is False


def test_vol_spike_false_before_warmup():
    idx = pd.date_range("2024-01-02 09:15", periods=5, freq="15min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": [10000] * 5},
        index=idx,
    )
    # Need 5 prior bars; bar 4 only has 4 prior → not enough.
    assert evaluate_condition("VOL_SPIKE(5, 1.5)", df, 4) is False


# ── 4. Gap detection / GAP_FILL_PCT ───────────────────────────────────────────


def _gap_test_df() -> pd.DataFrame:
    """Day 1 closes at 104. Day 2 gaps up to open 110 with low 108. Day 3 gaps
    down to open 100 with high 108 (full retrace of the down-gap)."""
    times: list[pd.Timestamp] = []
    for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        times += pd.date_range(f"{d} 09:15", periods=5, freq="1h", tz="Asia/Kolkata").tolist()
    idx = pd.DatetimeIndex(times)
    return pd.DataFrame(
        {
            "open":   [100, 101, 102, 103, 104,  110, 109, 107, 106, 108,  100, 101, 103, 106, 107],
            "high":   [101, 102, 103, 104, 105,  111, 110, 108, 107, 109,  101, 102, 104, 107, 108],
            "low":    [ 99, 100, 101, 102, 103,  108, 107, 106, 105, 107,   98, 100, 103, 106, 106],
            "close":  [101, 102, 103, 104, 104,  109, 108, 107, 107, 108,  100, 102, 104, 107, 107],
            "volume": [1000] * 15,
        },
        index=idx,
    )


def test_gap_size_pct_matches_formula():
    df = _gap_test_df()
    gap = gap_series(df)
    # Day 2 first bar — prev_close=104, open=110 → 5.769%.
    assert pytest.approx(gap.iloc[5]["GAP_SIZE_PCT"], abs=1e-4) == (110 - 104) / 104 * 100


def test_gap_fill_pct_runs_intraday():
    df = _gap_test_df()
    gap = gap_series(df)
    # Day 2 bar 4: min low across day 2 (105) → fill = (110-105)/6 = 83.33%.
    assert pytest.approx(gap.iloc[9]["GAP_FILL_PCT"], abs=1e-4) == 100 * 5 / 6


def test_is_gap_up_threshold():
    df = _gap_test_df()
    # Day 2 has 5.77% gap up; threshold 0.5% triggers.
    assert evaluate_condition("IS_GAP_UP(0.5)", df, 5) is True
    # Day 1 has no gap (first session) → False.
    assert evaluate_condition("IS_GAP_UP(0.5)", df, 0) is False


def test_is_gap_down_threshold():
    df = _gap_test_df()
    # Day 3 has a 7.4% gap down.
    assert evaluate_condition("IS_GAP_DOWN(0.5)", df, 10) is True
    assert evaluate_condition("IS_GAP_UP(0.5)", df, 10) is False


# ── 5. Candlestick patterns ───────────────────────────────────────────────────


def test_bullish_engulfing_fires():
    df = pd.DataFrame(
        {
            "open":  [100,  97],
            "high":  [100.5, 102.2],
            "low":   [ 97.5, 96.8],
            "close": [ 98, 102],
            "volume":[1000, 1000],
        }
    )
    out = add_all_patterns(df, {"bullish_engulfing": {}})
    assert out["IS_BULLISH_ENGULFING"].iloc[1] == 1.0
    assert out["IS_BULLISH_ENGULFING"].iloc[0] == 0.0


def test_hammer_detection():
    # Small bullish body 100→101, long lower wick to 95, tiny upper to 101.2.
    df = pd.DataFrame(
        {
            "open":  [100],
            "high":  [101.2],
            "low":   [95],
            "close": [101],
            "volume":[1000],
        }
    )
    out = add_all_patterns(df, {"hammer": {}})
    assert out["IS_HAMMER"].iloc[0] == 1.0


def test_shooting_star_detection():
    # Small bearish body 100→99, long upper wick to 105, tiny lower to 98.5.
    df = pd.DataFrame(
        {
            "open":  [100],
            "high":  [105],
            "low":   [98.5],
            "close": [99],
            "volume":[1000],
        }
    )
    out = add_all_patterns(df, {"shooting_star": {}})
    assert out["IS_SHOOTING_STAR"].iloc[0] == 1.0


def test_hammer_rejects_pure_doji():
    # Open == Close (zero body): hammer test must NOT fire.
    df = pd.DataFrame(
        {
            "open":  [100],
            "high":  [100.5],
            "low":   [95],
            "close": [100],
            "volume":[1000],
        }
    )
    out = add_all_patterns(df, {"hammer": {}})
    assert out["IS_HAMMER"].iloc[0] == 0.0


# ── 6. min_risk_reward gate ───────────────────────────────────────────────────


def test_min_risk_reward_loads_from_yaml():
    cfg = load_strategy_from_content(
        """
strategy:
  name: rr_test
  symbol: ABC.NS
  timeframe: 5m
  entry: { condition: "CLOSE > VWAP" }
  exit:  { condition: "CLOSE < VWAP" }
  indicators: { VWAP: [] }
  risk_management:
    stop_loss_percent: 2.0
    take_profit_percent: 5.0
    min_risk_reward: 2.0
"""
    )
    assert cfg.min_risk_reward == 2.0


def test_min_risk_reward_rejects_negative():
    with pytest.raises(ValueError, match="min_risk_reward"):
        load_strategy_from_content(
            """
strategy:
  name: rr_neg
  symbol: ABC.NS
  timeframe: 5m
  entry: { condition: "CLOSE > VWAP" }
  exit:  { condition: "CLOSE < VWAP" }
  indicators: { VWAP: [] }
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 1.0
    min_risk_reward: -1.0
"""
        )


# ── 7. RETEST_FROM_ABOVE / FROM_BELOW ─────────────────────────────────────────


def test_retest_from_above_bullish_breakout_retest():
    # Bar 4: breakout close=102; bar 6: low=99.5 retests 100 from above.
    df = pd.DataFrame(
        {
            "open":  [98, 99, 97, 99, 100, 102, 102, 101],
            "high":  [99, 100, 98, 100, 102, 104, 103, 104],
            "low":   [97, 98, 96, 98, 100, 102, 99.5, 101],
            "close": [98, 99, 97, 99, 102, 103, 102, 104],
            "volume":[1000] * 8,
        }
    )
    assert evaluate_condition("RETEST_FROM_ABOVE(100, 5)", df, 5) is False
    assert evaluate_condition("RETEST_FROM_ABOVE(100, 5)", df, 6) is True


def test_retest_from_below_bearish_breakdown_retest():
    # Bar 4: breakdown close=98 (< 100); bar 6: high=100.5 retests 100 from below.
    df = pd.DataFrame(
        {
            "open":  [102, 103, 101, 103, 100,  98,  98,  99],
            "high":  [103, 104, 102, 104, 102, 100, 100.5, 100],
            "low":   [101, 102, 100, 102,  98,  97,  97,  98],
            "close": [102, 103, 101, 103,  98,  97,  98,  97],
            "volume":[1000] * 8,
        }
    )
    assert evaluate_condition("RETEST_FROM_BELOW(100, 5)", df, 4) is False
    assert evaluate_condition("RETEST_FROM_BELOW(100, 5)", df, 6) is True


# ── 8. Dynamic take-profit ────────────────────────────────────────────────────


def test_take_profit_level_loads():
    cfg = load_strategy_from_content(
        """
strategy:
  name: tp_level
  symbol: ABC.NS
  timeframe: 5m
  entry: { condition: "CLOSE > VWAP" }
  exit:  { condition: "CLOSE < VWAP" }
  indicators: { VWAP: [] }
  take_profit:
    type: level
    level: PIVOT_R1
    padding_pct: 0.1
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 3.0
"""
    )
    assert cfg.take_profit_spec == {
        "type": "level",
        "level": "PIVOT_R1",
        "padding_pct": 0.1,
    }


def test_take_profit_gap_fill_loads():
    cfg = load_strategy_from_content(
        """
strategy:
  name: tp_gap
  symbol: ABC.NS
  timeframe: 5m
  entry: { condition: "IS_GAP_DOWN(0.5)" }
  exit:  { condition: "CLOSE > VWAP" }
  indicators: { VWAP: [] }
  take_profit:
    type: gap_fill
    percent: 80
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 3.0
"""
    )
    assert cfg.take_profit_spec == {"type": "gap_fill", "percent": 80.0}


def test_take_profit_rejects_invalid_level():
    with pytest.raises(ValueError, match="take_profit.level"):
        load_strategy_from_content(
            """
strategy:
  name: tp_bad
  symbol: ABC.NS
  timeframe: 5m
  entry: { condition: "CLOSE > VWAP" }
  exit:  { condition: "CLOSE < VWAP" }
  indicators: { VWAP: [] }
  take_profit:
    type: level
    level: NOT_A_REAL_LEVEL
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 3.0
"""
        )


# ── 9. Entry sequence (state machine) ─────────────────────────────────────────


def test_entry_sequence_completes_in_order():
    """trend → pullback → volume spike, all firing within windows."""
    df = pd.DataFrame(
        {
            "open":  [100, 101, 102, 103, 100,  99, 100, 102],
            "high":  [101, 102, 103, 104, 102, 100, 103, 103],
            "low":   [ 99, 100, 101, 102, 100,  98,  99, 101],
            "close": [101, 102, 103, 104, 100, 100, 103, 102],
            "volume":[1000, 1000, 1000, 1000, 1000, 1000, 3000, 1000],
        }
    )
    steps = parse_entry_sequence(
        [
            {"id": "trend",    "condition": "CLOSE > 102", "within_bars": 0},
            {"id": "pullback", "condition": "LOW <= 100",  "within_bars": 5},
            {"id": "volume",   "condition": "VOLUME > 2000", "within_bars": 3},
        ]
    )
    fired = evaluate_entry_sequence(steps, df)
    assert fired.iloc[6] is np.True_ or fired.iloc[6] is True
    # Only one bar should fire.
    assert int(fired.sum()) == 1


def test_entry_sequence_times_out_and_resets():
    df = pd.DataFrame(
        {
            "open":  [100, 101, 102, 103, 100, 99, 100, 101, 101, 100, 99],
            "high":  [101, 102, 103, 104, 102, 100, 103, 103, 101, 100, 99],
            "low":   [ 99, 100, 101, 102, 100, 98, 99,  98, 98,  98, 98],
            "close": [101, 102, 103, 104, 100, 100, 102, 100, 100, 99, 99],
            "volume":[1000, 1000, 1000, 1000, 1000, 1000, 1500, 1200, 1000, 3000, 5000],
        }
    )
    steps = parse_entry_sequence(
        [
            {"id": "trend",    "condition": "CLOSE > 102", "within_bars": 0},
            {"id": "pullback", "condition": "LOW <= 100",  "within_bars": 5},
            {"id": "volume",   "condition": "VOLUME > 2000", "within_bars": 3},
        ]
    )
    fired = evaluate_entry_sequence(steps, df)
    # Volume spike at bars 9-10 happens too late to satisfy the 3-bar window
    # after pullback (bars 4-5). Sequence must never complete.
    assert not fired.any()


def test_entry_sequence_loader_validation():
    with pytest.raises(ValueError, match="duplicate id"):
        load_strategy_from_content(
            """
strategy:
  name: seq_dup
  symbol: ABC.NS
  timeframe: 5m
  exit:  { condition: "CLOSE < VWAP" }
  indicators: { VWAP: [] }
  entry_sequence:
    - { id: a, condition: "CLOSE > VWAP", within_bars: 5 }
    - { id: a, condition: "LOW <= VWAP",  within_bars: 5 }
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 3.0
"""
        )


def test_entry_condition_optional_when_sequence_present():
    """A strategy with only entry_sequence (no entry.condition) loads cleanly."""
    cfg = load_strategy_from_content(
        """
strategy:
  name: seq_only
  symbol: ABC.NS
  timeframe: 5m
  exit:  { condition: "CLOSE < VWAP" }
  indicators: { VWAP: [] }
  entry_sequence:
    - { id: trend,    condition: "CLOSE > VWAP", within_bars: 0 }
    - { id: pullback, condition: "LOW <= VWAP",  within_bars: 5 }
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 3.0
"""
    )
    assert len(cfg.entry_sequence) == 2
    assert cfg.entry_sequence[0]["id"] == "trend"
