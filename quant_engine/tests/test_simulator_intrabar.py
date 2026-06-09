"""
Tests for the Phase 11 1-minute execution path in engine/simulator.py.

Two properties matter:

  1. PARITY — when each strategy bar maps to exactly one minute bar equal to
     itself, the 1-minute walk must reproduce the legacy strategy-bar
     resolution bit-for-bit. This is what guarantees a true 1m strategy (and
     any unchanged backtest) behaves exactly as before.

  2. ACCURACY — when a single strategy bar's range touches BOTH the stop and
     the target, the legacy engine resolves it pessimistically (stop first).
     Walking the underlying minutes must instead honour whichever was actually
     touched first: TAKE_PROFIT if price spiked to the target before dipping to
     the stop, STOP_LOSS if it dipped first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.simulator import build_minute_arrays, simulate_trades


# Common simulation kwargs: zero costs so fills land on exact stop/target
# prices and assertions stay simple. Entry signal is always true, so the
# engine enters on the first eligible bar and re-enters after each exit.
_COMMON = dict(
    symbol="TEST",
    entry_condition="CLOSE > 0",
    exit_condition="",
    stop_loss_pct=2.0,
    take_profit_pct=2.0,
    slippage_bps=0.0,
    commission_bps=0.0,
    warm_up_candles=0,
    objective="positional",
    stt_delivery_pct=0.0,
)


def _strategy_frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame(
        {
            "open":  [r[1] for r in rows],
            "high":  [r[2] for r in rows],
            "low":   [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0] * len(rows),
        },
        index=idx,
    )


def _identity_slices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Each strategy bar i owns exactly minute i."""
    return np.arange(n, dtype=np.int64), np.arange(1, n + 1, dtype=np.int64)


# ── Parity ──────────────────────────────────────────────────────────────────

def test_parity_identity_minutes_match_legacy():
    # A multi-bar path that produces several SL/TP round-trips.
    df = _strategy_frame([
        ("2024-01-01", 100, 101, 99, 100),
        ("2024-01-02", 100, 104, 99, 103),   # spikes up → TAKE_PROFIT (tp=102)
        ("2024-01-03", 103, 104, 102, 103),
        ("2024-01-04", 103, 104, 100, 101),   # dips → STOP_LOSS (entry 103, sl≈100.94)
        ("2024-01-05", 101, 103, 100, 102),
        ("2024-01-06", 102, 105, 101, 104),
        ("2024-01-07", 104, 105, 103, 104),
    ])

    legacy_trades, _ = simulate_trades(df=df, **_COMMON)

    minute_arrays = build_minute_arrays(df)
    starts, ends = _identity_slices(len(df))
    intrabar_trades, _ = simulate_trades(
        df=df, minute_arrays=minute_arrays, subbar_starts=starts, subbar_ends=ends, **_COMMON,
    )

    assert len(legacy_trades) == len(intrabar_trades) > 0
    # Frozen dataclass equality compares every field, including the structured
    # entry/exit rationale dicts — so this is a strict bit-for-bit check.
    assert legacy_trades == intrabar_trades


# ── Accuracy: same-bar stop+target ambiguity resolved by the minute path ──────

def _same_bar_both_touched_frame() -> pd.DataFrame:
    # Bar 0 fires the entry signal; fill at bar 1 open = 100. With sl/tp = 2%,
    # stop = 98, target = 102. Bar 1's daily range [97, 103] touches BOTH.
    return _strategy_frame([
        ("2024-01-01", 100, 100, 100, 100),
        ("2024-01-02", 100, 103, 97, 100),   # ambiguous on the daily bar
        ("2024-01-03", 100, 100, 100, 100),
    ])


def _minutes_for_bar1(bar1_minutes: list[tuple]):
    """Build minute_arrays + slices for the same-bar fixture.

    bar0→1 min, bar1→len(bar1_minutes) mins, bar2→1 min. Each bar1 minute is a
    (open, high, low, close) tuple, in chronological order.
    """
    rows = [("2024-01-01 00:00", 100, 100, 100, 100)]  # bar 0
    for k, mins in enumerate(bar1_minutes):
        rows.append((f"2024-01-02 00:{k:02d}", *mins))  # bar 1 minutes
    rows.append(("2024-01-03 00:00", 100, 100, 100, 100))  # bar 2
    mdf = _strategy_frame(rows)
    n1 = len(bar1_minutes)
    starts = np.array([0, 1, 1 + n1], dtype=np.int64)
    ends   = np.array([1, 1 + n1, 2 + n1], dtype=np.int64)
    return build_minute_arrays(mdf), starts, ends


def test_legacy_resolves_same_bar_pessimistically():
    df = _same_bar_both_touched_frame()
    trades, _ = simulate_trades(df=df, **_COMMON)
    assert len(trades) == 1
    assert trades[0].exit_reason == "STOP_LOSS_AND_TAKE_PROFIT_SAME_BAR"
    assert trades[0].exit_price == 98.0          # filled at the stop (pessimistic)


def test_intrabar_take_profit_first():
    df = _same_bar_both_touched_frame()
    # Minute 0 spikes to the target (high 103) without touching the stop
    # (low 99); minute 1 later dips to 97. Target was hit first.
    minute_arrays, starts, ends = _minutes_for_bar1([
        (100, 103, 99, 101),
        (101, 101, 97, 98),
    ])
    trades, _ = simulate_trades(
        df=df, minute_arrays=minute_arrays, subbar_starts=starts, subbar_ends=ends, **_COMMON,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "TAKE_PROFIT"
    assert trades[0].exit_price == 102.0
    assert trades[0].exit_date == "2024-01-02 00:00:00"   # the firing minute


def test_intrabar_stop_loss_first():
    df = _same_bar_both_touched_frame()
    # Minute 0 touches neither; minute 1 dips to the stop (low 97) with high 100
    # (no target); minute 2 only later spikes to 103. Stop was hit first, at
    # minute 1 — before the target was ever reached.
    minute_arrays, starts, ends = _minutes_for_bar1([
        (100, 100, 99, 100),
        (100, 100, 97, 98),
        (98, 103, 98, 102),
    ])
    trades, _ = simulate_trades(
        df=df, minute_arrays=minute_arrays, subbar_starts=starts, subbar_ends=ends, **_COMMON,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "STOP_LOSS"
    assert trades[0].exit_price == 98.0
    assert trades[0].exit_date == "2024-01-02 00:01:00"   # the firing minute


def test_intrabar_both_in_one_minute_is_pessimistic():
    df = _same_bar_both_touched_frame()
    # A single minute touches both → ambiguity legitimately remains, resolve
    # pessimistically (stop first), exactly like legacy but confined to 1 min.
    minute_arrays, starts, ends = _minutes_for_bar1([
        (100, 103, 97, 100),
        (100, 100, 100, 100),
    ])
    trades, _ = simulate_trades(
        df=df, minute_arrays=minute_arrays, subbar_starts=starts, subbar_ends=ends, **_COMMON,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "STOP_LOSS_AND_TAKE_PROFIT_SAME_BAR"
    assert trades[0].exit_price == 98.0
