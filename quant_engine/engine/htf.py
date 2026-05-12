"""
engine/htf.py
─────────────
Higher-timeframe context (Phase 5).

A HtfContext bundles everything the simulator needs to evaluate one HTF
entry-gate at every main bar:

  • the HTF's own OHLCV df (with indicators precomputed)
  • the compiled condition string
  • a precomputed mapping `main_to_htf_index[i]` — for each main-df bar i,
    the index in the HTF df of the most recently *closed* HTF bar at the
    time of that main bar. -1 when no HTF bar has closed yet (early in the
    backtest), which the simulator treats as "block entry".

The mapping is built once (O(n_main · log n_htf)) and looked up O(1) per
bar inside the hot loop.

Strict no-look-ahead: an HTF bar at timestamp T represents the candle
[T, T + tf_duration). It is "closed" only at T + tf_duration. So when
evaluating a main bar at timestamp t_main, the latest usable HTF index k
satisfies htf_ts[k] + tf_duration <= t_main.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

from engine.conditions import CompiledCondition, compile_condition
from engine.indicators import add_all_indicators

logger = logging.getLogger(__name__)


# ── Timeframe → pandas Timedelta ─────────────────────────────────────────────

# Same canonical set the rest of the system supports.
_TF_PATTERN = re.compile(r"^\s*(\d+)\s*(m|h|d|w)\s*$", re.IGNORECASE)


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    """Convert "5m" / "1h" / "1d" / "1w" to a pandas Timedelta.
    Raises ValueError for unrecognised input so HTF setup fails loudly."""
    if not tf:
        raise ValueError("timeframe cannot be empty")
    m = _TF_PATTERN.match(str(tf))
    if not m:
        raise ValueError(f"unsupported HTF timeframe: {tf!r}")
    n, unit = int(m.group(1)), m.group(2).lower()
    unit_map = {"m": "min", "h": "h", "d": "d", "w": "w"}
    return pd.Timedelta(f"{n}{unit_map[unit]}")


# ── Main → HTF index mapping ─────────────────────────────────────────────────


def build_main_to_htf_index(
    main_index: pd.DatetimeIndex,
    htf_index: pd.DatetimeIndex,
    htf_duration: pd.Timedelta,
) -> np.ndarray:
    """For each main-bar timestamp, return the index of the most recently
    *closed* HTF bar; -1 if no HTF bar has closed yet.

    An HTF bar at timestamp T is treated as closed at T + htf_duration, so
    the latest usable index k for main timestamp t satisfies
        htf_index[k] + htf_duration <= t   ⇔   htf_index[k] <= t - htf_duration

    No look-ahead: we explicitly require strict <= on the closed timestamp.
    """
    if len(htf_index) == 0:
        return np.full(len(main_index), -1, dtype=np.int64)
    if not htf_index.is_monotonic_increasing:
        raise ValueError("HTF index must be monotonically increasing.")

    cutoffs_ns = (main_index - htf_duration).asi8         # int64 ns array
    htf_ns = htf_index.asi8

    # searchsorted(side='right') returns the count of htf entries <= cutoff;
    # subtract 1 to get the index of the latest such entry.
    pos = np.searchsorted(htf_ns, cutoffs_ns, side="right") - 1
    return pos.astype(np.int64)


# ── HtfContext ───────────────────────────────────────────────────────────────


@dataclass
class HtfContext:
    timeframe: str
    df: pd.DataFrame
    compiled: CompiledCondition
    main_to_htf_index: np.ndarray   # length == len(main_df); values are htf indices or -1

    @property
    def raw_condition(self) -> str:
        return self.compiled.raw

    def evaluate(self, main_index_i: int) -> bool:
        """True iff the HTF condition holds at the most recently closed HTF
        bar before main bar i. False when no HTF bar has closed yet."""
        htf_i = int(self.main_to_htf_index[main_index_i])
        if htf_i < 0:
            return False
        return self.compiled.evaluate(self.df, htf_i)


def _ensure_indicators_for_condition(df: pd.DataFrame, compiled: CompiledCondition) -> pd.DataFrame:
    """Compute every periodic indicator the compiled condition references
    so the per-bar evaluator just reads precomputed columns. Mirrors what
    runner._merge_indicator_requirements does for the main df."""
    config: dict[str, set[int]] = {}
    for ref in compiled.indicator_refs:
        config.setdefault(ref.name, set()).add(ref.period)
    if not config:
        return df
    return add_all_indicators(df, {name: sorted(periods) for name, periods in config.items()})


def _ensure_scalar_indicators(df: pd.DataFrame, compiled: CompiledCondition) -> pd.DataFrame:
    """Compute MACD/VWAP scalar columns if the HTF condition uses them."""
    needed = set(compiled.scalar_refs)
    if not needed:
        return df
    out = df.copy()
    if "VWAP" in needed and "VWAP" not in out.columns:
        from engine.indicators import vwap
        out["VWAP"] = vwap(out)
    if {"MACD", "MACD_SIGNAL", "MACD_HIST"} & needed:
        from engine.indicators import macd_line, macd_signal
        if "MACD" not in out.columns:
            out["MACD"] = macd_line(out["close"])
        if "MACD_SIGNAL" not in out.columns:
            out["MACD_SIGNAL"] = macd_signal(out["close"])
        if "MACD_HIST" not in out.columns and {"MACD", "MACD_SIGNAL"}.issubset(out.columns):
            out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out


def build_htf_contexts(
    htf_rules: Iterable,
    htf_ohlcv_by_tf: dict[str, pd.DataFrame],
    main_index: pd.DatetimeIndex,
) -> list[HtfContext]:
    """Build one HtfContext per HTF rule. Validates that every declared
    timeframe has a matching OHLCV df supplied; raises a clear error
    otherwise so the caller can fail fast."""
    contexts: list[HtfContext] = []
    for rule in htf_rules:
        tf = rule.timeframe
        if tf not in htf_ohlcv_by_tf:
            raise ValueError(
                f"HTF rule declares timeframe={tf!r} but no OHLCV was supplied "
                f"for it (htf_ohlcv keys: {sorted(htf_ohlcv_by_tf.keys())})."
            )
        htf_df = htf_ohlcv_by_tf[tf]
        if htf_df is None or htf_df.empty:
            raise ValueError(f"HTF OHLCV for timeframe={tf!r} is empty.")
        if not isinstance(htf_df.index, pd.DatetimeIndex):
            raise ValueError(f"HTF OHLCV for timeframe={tf!r} must have a DatetimeIndex.")

        compiled = compile_condition(rule.condition)
        if compiled is None:
            raise ValueError(f"HTF condition is empty for timeframe={tf!r}.")

        # Precompute the indicators the HTF condition reads.
        htf_df = _ensure_indicators_for_condition(htf_df, compiled)
        htf_df = _ensure_scalar_indicators(htf_df, compiled)

        td = timeframe_to_timedelta(tf)
        mapping = build_main_to_htf_index(main_index, htf_df.index, td)

        contexts.append(HtfContext(
            timeframe=tf,
            df=htf_df,
            compiled=compiled,
            main_to_htf_index=mapping,
        ))
        logger.info(
            "🔭 HTF context built | tf=%s rows=%s condition=%r first_usable_main_bar=%s",
            tf, len(htf_df), rule.condition,
            int(np.argmax(mapping >= 0)) if (mapping >= 0).any() else -1,
        )
    return contexts


def all_htf_gates_pass(contexts: list[HtfContext], main_index_i: int) -> bool:
    """Bar-loop helper: True iff every HTF gate passes at main bar i."""
    for ctx in contexts:
        if not ctx.evaluate(main_index_i):
            return False
    return True
