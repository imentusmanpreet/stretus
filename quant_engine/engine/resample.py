"""
engine/resample.py
──────────────────
Phase 11 — 1-minute execution.

A backtest's *signals* are evaluated at the strategy timeframe (15m / 1h / 1d),
but its *trade execution* (entry fills, stop-loss / take-profit / trailing-stop
resolution) is only accurate at 1-minute granularity. A single 1d candle hides
the intrabar price path: when both the stop and the target sit inside one daily
bar's range we cannot tell from the daily OHLC alone which was touched first,
nor the price at which a stop actually filled. The legacy simulator resolved
that ambiguity pessimistically (assume the stop filled first) — correct as a
worst case, but wrong often enough to distort results.

This module bridges signal-timeframe and execution-timeframe:

  • resample_ohlcv(df_1m, tf)
        aggregate a 1-minute OHLCV frame up to the strategy timeframe so
        indicators and entry/exit conditions evaluate exactly as before.

  • build_subbar_slices(strategy_index, minute_index, tf_duration)
        for each strategy bar, the half-open [start, end) slice of 1-minute
        bars that fall inside that strategy bar's window — so the simulator
        can walk the true price path minute by minute while in a trade.

Both follow the same left-labelled / left-closed convention the rest of the
engine assumes (see engine/htf.py): a bar stamped T represents the candle
[T, T + tf_duration) and is only "closed" at T + tf_duration.

Timestamp alignment note: resampling bins are anchored to the unix epoch
(origin="epoch"), i.e. calendar-UTC boundaries. For NSE the whole trading
session (03:45–10:00 UTC) falls inside one UTC date, so a "1d" bin groups a
full session correctly, and sub-hour bins (1m/5m/15m) align to the session
open because 03:45 is an exact multiple of those frequencies from midnight.
Hourly ("1h") bins align to clock hours, which may not match a provider that
anchors hourly bars to the 09:15 IST session open; if/when we need exact
parity with such a provider the bin origin becomes a parameter. Execution
accuracy (the point of this module) does not depend on that alignment.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.htf import timeframe_to_timedelta

logger = logging.getLogger(__name__)


# OHLCV aggregation rules. Any column not listed here (e.g. REF_-prefixed
# reference series, indicator columns) is dropped — resampling must run on the
# RAW 1-minute OHLCV, before indicators/reference merges, exactly like the
# legacy loader produced the strategy-timeframe frame.
_BASE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a 1-minute OHLCV frame up to ``timeframe``.

    The result is what the rest of the pipeline treats as the "strategy
    timeframe" frame: indicators are computed on it and entry/exit conditions
    evaluate against it. open=first, high=max, low=min, close=last, volume=sum.

    Empty bins (gaps between sessions, weekends, holidays) are dropped so the
    output contains only bars that actually had ticks — matching the dense,
    gap-free frames the market-data provider used to deliver pre-aggregated.

    Returns a new DataFrame; the input is not mutated. When ``timeframe`` is
    "1m" (or any timeframe equal to the input's own spacing) the data is
    returned essentially unchanged, which is what makes 1m-strategy backtests
    bit-for-bit identical to the legacy path.
    """
    if df_1m is None or df_1m.empty:
        raise ValueError("resample_ohlcv: input frame is empty.")
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise ValueError("resample_ohlcv: input must have a DatetimeIndex.")
    if not df_1m.index.is_monotonic_increasing:
        raise ValueError("resample_ohlcv: input index must be monotonically increasing.")

    td = timeframe_to_timedelta(timeframe)
    if td <= pd.Timedelta(0):
        raise ValueError(f"resample_ohlcv: non-positive timeframe duration for {timeframe!r}.")

    cols = [c for c in _BASE_AGG if c in df_1m.columns]
    if "open" not in cols or "close" not in cols:
        raise ValueError(
            "resample_ohlcv: input must contain at least 'open' and 'close' columns; "
            f"got {list(df_1m.columns)}."
        )
    agg = {c: _BASE_AGG[c] for c in cols}

    out = (
        df_1m.resample(td, label="left", closed="left", origin="epoch")
        .agg(agg)
        # A bin with no ticks yields NaN open/close — drop it so callers never
        # see synthetic flat bars across overnight/holiday gaps.
        .dropna(subset=["open", "close"], how="any")
    )
    out = out.astype(float)

    logger.info(
        "Resampled OHLCV | timeframe=%s in_rows=%s out_rows=%s span=%s→%s",
        timeframe, len(df_1m), len(out),
        out.index[0].isoformat() if len(out) else "∅",
        out.index[-1].isoformat() if len(out) else "∅",
    )
    return out


def build_subbar_slices(
    strategy_index: pd.DatetimeIndex,
    minute_index: pd.DatetimeIndex,
    tf_duration: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each strategy bar to the minute bars inside its window.

    For strategy bar ``k`` stamped ``T_k`` (covering ``[T_k, T_k + tf_duration)``)
    return ``starts[k]`` / ``ends[k]`` such that the minute bars belonging to
    that strategy bar are ``minute_index[starts[k]:ends[k]]`` — i.e. every
    minute bar ``m`` with ``T_k <= m_ts < T_k + tf_duration``.

    Both arrays have length ``len(strategy_index)``. An empty slice
    (``starts[k] == ends[k]``) means no minute data exists for that strategy
    bar; the caller decides how to handle it (the simulator falls back to the
    strategy bar's own OHLC so a gap never silently skips an exit check).

    No look-ahead by construction: the slice for bar ``k`` contains only
    minutes within ``k``'s own window — never a future bar's minutes.
    """
    n = len(strategy_index)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if len(minute_index) == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    if not minute_index.is_monotonic_increasing:
        raise ValueError("build_subbar_slices: minute_index must be monotonically increasing.")

    m_ns = minute_index.asi8
    starts = np.searchsorted(m_ns, strategy_index.asi8, side="left")
    ends = np.searchsorted(m_ns, (strategy_index + tf_duration).asi8, side="left")
    return starts.astype(np.int64), ends.astype(np.int64)
