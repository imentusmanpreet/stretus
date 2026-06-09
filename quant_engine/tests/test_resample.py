"""
Tests for engine/resample.py — the 1m → strategy-timeframe bridge that
underpins Phase 11 (1-minute execution).

Covers:
  • OHLCV aggregation correctness (open=first, high=max, low=min, close=last,
    volume=sum) at 15m / 1h.
  • Empty bins across session/weekend gaps are dropped.
  • "1m" resample is an identity (the property that gives bit-for-bit legacy
    parity for 1m strategies).
  • build_subbar_slices: each strategy bar maps to exactly the minutes inside
    its half-open window, with no look-ahead and clean handling of gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.htf import timeframe_to_timedelta
from engine.resample import build_subbar_slices, resample_ohlcv


def _minute_frame(start: str, periods: int, freq: str = "1min") -> pd.DataFrame:
    """A deterministic 1-minute OHLCV frame.

    Prices walk upward by minute so high/low/first/last are easy to predict:
    bar j has open=100+j, high=100+j+0.5, low=100+j-0.5, close=100+j+0.25.
    """
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    j = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "open": 100.0 + j,
            "high": 100.0 + j + 0.5,
            "low": 100.0 + j - 0.5,
            "close": 100.0 + j + 0.25,
            "volume": 10.0 + j,
        },
        index=idx,
    )


def test_resample_15m_aggregates_ohlcv_correctly():
    # 30 one-minute bars from 03:45 UTC → two clean 15m bars.
    df1 = _minute_frame("2024-01-02 03:45:00", periods=30)
    out = resample_ohlcv(df1, "15m")

    assert list(out.index) == [
        pd.Timestamp("2024-01-02 03:45:00"),
        pd.Timestamp("2024-01-02 04:00:00"),
    ]

    # First 15m bar = minutes 0..14.
    assert out.iloc[0]["open"] == df1.iloc[0]["open"]          # first
    assert out.iloc[0]["close"] == df1.iloc[14]["close"]       # last
    assert out.iloc[0]["high"] == df1.iloc[:15]["high"].max()  # max
    assert out.iloc[0]["low"] == df1.iloc[:15]["low"].min()    # min
    assert out.iloc[0]["volume"] == df1.iloc[:15]["volume"].sum()

    # Second 15m bar = minutes 15..29.
    assert out.iloc[1]["open"] == df1.iloc[15]["open"]
    assert out.iloc[1]["close"] == df1.iloc[29]["close"]
    assert out.iloc[1]["high"] == df1.iloc[15:]["high"].max()
    assert out.iloc[1]["low"] == df1.iloc[15:]["low"].min()


def test_resample_drops_empty_bins_across_gap():
    # Two separate sessions a day apart: 15 minutes each. A naive fixed-freq
    # resample would emit dozens of empty 15m bins in the overnight gap; we
    # must drop them and keep exactly the two real bars.
    day1 = _minute_frame("2024-01-02 03:45:00", periods=15)
    day2 = _minute_frame("2024-01-03 03:45:00", periods=15)
    df1 = pd.concat([day1, day2])

    out = resample_ohlcv(df1, "15m")
    assert list(out.index) == [
        pd.Timestamp("2024-01-02 03:45:00"),
        pd.Timestamp("2024-01-03 03:45:00"),
    ]


def test_resample_1m_is_identity():
    df1 = _minute_frame("2024-01-02 03:45:00", periods=20)
    out = resample_ohlcv(df1, "1m")
    pd.testing.assert_frame_equal(out, df1[["open", "high", "low", "close", "volume"]])


def test_resample_1h_aggregates():
    # 120 minutes → two 1h bars aligned to clock hours from 04:00 UTC.
    df1 = _minute_frame("2024-01-02 04:00:00", periods=120)
    out = resample_ohlcv(df1, "1h")
    assert list(out.index) == [
        pd.Timestamp("2024-01-02 04:00:00"),
        pd.Timestamp("2024-01-02 05:00:00"),
    ]
    assert out.iloc[0]["open"] == df1.iloc[0]["open"]
    assert out.iloc[0]["close"] == df1.iloc[59]["close"]
    assert out.iloc[0]["high"] == df1.iloc[:60]["high"].max()


def test_resample_rejects_non_datetime_index():
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    with pytest.raises(ValueError):
        resample_ohlcv(df, "15m")


def test_subbar_slices_partition_minutes_with_no_lookahead():
    df1 = _minute_frame("2024-01-02 03:45:00", periods=30)
    out = resample_ohlcv(df1, "15m")
    td = timeframe_to_timedelta("15m")

    starts, ends = build_subbar_slices(out.index, df1.index, td)

    # Two strategy bars, each owning exactly its 15 minutes, contiguous and
    # covering the whole minute series with no overlap.
    assert list(starts) == [0, 15]
    assert list(ends) == [15, 30]

    # Every minute in bar k's slice is inside [T_k, T_k + 15m); none leak from
    # the next bar (no look-ahead).
    for k, ts in enumerate(out.index):
        sl = df1.index[starts[k]:ends[k]]
        assert (sl >= ts).all()
        assert (sl < ts + td).all()


def test_subbar_slices_empty_when_gap_has_no_minutes():
    # Strategy index references a bar that has no underlying minutes.
    df1 = _minute_frame("2024-01-02 03:45:00", periods=15)
    td = timeframe_to_timedelta("15m")
    # A fabricated strategy index with a phantom 04:00 bar that has no minutes.
    strat_idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 03:45:00"), pd.Timestamp("2024-01-02 04:00:00")]
    )
    starts, ends = build_subbar_slices(strat_idx, df1.index, td)
    assert list(starts) == [0, 15]
    assert list(ends) == [15, 15]      # second bar: empty slice (start == end)


def test_subbar_slices_empty_inputs():
    td = timeframe_to_timedelta("15m")
    empty = pd.DatetimeIndex([])
    minutes = _minute_frame("2024-01-02 03:45:00", periods=5).index
    s, e = build_subbar_slices(empty, minutes, td)
    assert len(s) == 0 and len(e) == 0
    s, e = build_subbar_slices(minutes, empty, td)
    assert list(s) == [0] * len(minutes)
    assert list(e) == [0] * len(minutes)
