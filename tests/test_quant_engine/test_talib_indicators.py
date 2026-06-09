"""
Phase 1 — TA-Lib indicator backend tests.

Three guarantees per the acceptance criteria:

  (a) Reference     — each wrapper equals TA-Lib's own output (the oracle),
                      aligned to df.index, with NaN through the warm-up window.
  (b) Look-ahead    — truncate the series at bar i, recompute, and assert
                      value[i] is unchanged. A causal indicator can NEVER use
                      bar i+1 to decide bar i; this proves it for every study.
  (c) Single source — a study computes IDENTICALLY whether requested via
                      add_all_indicators(), compute_indicator_column(), or
                      discovered in a formula and resolved by the slow path,
                      the fast (array) path, or precompute-if-missing. If these
                      ever diverge the backtest is fiction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant_engine"))

from engine import indicators as I
from engine.conditions import (
    build_arrays_from_df,
    compile_condition,
    ensure_scalar_columns,
    evaluate_condition,
)
from engine.runner import _merge_indicator_requirements


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Deterministic OHLCV frame — 15-minute bars, a couple of sessions."""
    rng = np.random.RandomState(20260529)
    n = 240
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="15min", tz="UTC")
    walk = np.cumsum(rng.randn(n) * 0.5)
    close = pd.Series(100.0 + walk, index=idx)
    prev = close.shift(1).bfill()
    high = np.maximum(close, prev) + np.abs(rng.randn(n) * 0.3)
    low = np.minimum(close, prev) - np.abs(rng.randn(n) * 0.3)
    return pd.DataFrame(
        {
            "open": prev.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": 1e5 + np.abs(rng.randn(n)) * 1e4,
        },
        index=idx,
    )


# ── (a) Reference: wrappers match TA-Lib, aligned + NaN-warmed ─────────────────

def _close(df):
    return np.ascontiguousarray(df["close"].to_numpy(dtype=float))


def _hlc(df):
    return (
        np.ascontiguousarray(df["high"].to_numpy(dtype=float)),
        np.ascontiguousarray(df["low"].to_numpy(dtype=float)),
        np.ascontiguousarray(df["close"].to_numpy(dtype=float)),
    )


def test_close_wrappers_match_talib(df):
    c = _close(df)
    cases = {
        "SMA": (I.sma(df["close"], 20), talib.SMA(c, 20)),
        "EMA": (I.ema(df["close"], 20), talib.EMA(c, 20)),
        "WMA": (I.wma(df["close"], 10), talib.WMA(c, 10)),
        "DEMA": (I.dema(df["close"], 10), talib.DEMA(c, 10)),
        "TEMA": (I.tema(df["close"], 10), talib.TEMA(c, 10)),
        "RSI": (I.rsi(df["close"], 14), talib.RSI(c, 14)),
        "ROC": (I.roc(df["close"], 10), talib.ROC(c, 10)),
        "MOM": (I.mom(df["close"], 10), talib.MOM(c, 10)),
        "CMO": (I.cmo(df["close"], 14), talib.CMO(c, 14)),
        "TRIX": (I.trix(df["close"], 15), talib.TRIX(c, 15)),
    }
    for name, (got, oracle) in cases.items():
        assert isinstance(got, pd.Series), name
        assert got.index.equals(df.index), f"{name}: index not aligned"
        np.testing.assert_allclose(
            got.to_numpy(dtype=float), oracle, rtol=1e-9, atol=1e-9,
            err_msg=f"{name} diverges from TA-Lib", equal_nan=True,
        )


def test_ohlc_wrappers_match_talib(df):
    h, l, c = _hlc(df)
    cases = {
        "ATR": (I.atr(df, 14), talib.ATR(h, l, c, 14)),
        "NATR": (I.natr(df, 14), talib.NATR(h, l, c, 14)),
        "ADX": (I.adx(df, 14), talib.ADX(h, l, c, 14)),
        "ADXR": (I.adxr(df, 14), talib.ADXR(h, l, c, 14)),
        "PLUS_DI": (I.plus_di(df, 14), talib.PLUS_DI(h, l, c, 14)),
        "MINUS_DI": (I.minus_di(df, 14), talib.MINUS_DI(h, l, c, 14)),
        "CCI": (I.cci(df, 20), talib.CCI(h, l, c, 20)),
        "WILLR": (I.willr(df, 14), talib.WILLR(h, l, c, 14)),
    }
    for name, (got, oracle) in cases.items():
        assert got.index.equals(df.index), f"{name}: index not aligned"
        np.testing.assert_allclose(
            got.to_numpy(dtype=float), oracle, rtol=1e-9, atol=1e-9,
            err_msg=f"{name} diverges from TA-Lib", equal_nan=True,
        )


def test_warmup_is_nan_then_valid(df):
    """NaN through the look-back window, then valid — the engine's convention."""
    rsi = I.rsi(df["close"], 14)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].notna().all()
    sma = I.sma(df["close"], 20)
    assert sma.iloc[:19].isna().all()
    assert sma.iloc[19:].notna().all()


def test_candlestick_pattern_is_signed_int(df):
    eng = I.candlestick_pattern(df, "CDL_ENGULFING")
    assert eng.index.equals(df.index)
    vals = eng.dropna().to_numpy()
    # TA-Lib pattern functions return signed integers (bullish > 0, bearish < 0,
    # 0 = no pattern); magnitudes are typically 80/100/200, never fractional.
    assert np.all(np.equal(np.mod(vals, 1), 0)), "CDL values must be integer-valued"
    assert np.all(np.abs(vals) <= 200), "CDL magnitude out of expected range"


# ── (b) Look-ahead / causality: value[i] independent of bars > i ───────────────

# (config-key, period, output column) — one representative period per family.
_LOOKAHEAD_CASES = [
    ("SMA", 20, "SMA_20"), ("EMA", 20, "EMA_20"), ("WMA", 10, "WMA_10"),
    ("DEMA", 10, "DEMA_10"), ("TEMA", 10, "TEMA_10"), ("TRIMA", 10, "TRIMA_10"),
    ("KAMA", 30, "KAMA_30"), ("T3", 5, "T3_5"),
    ("RSI", 14, "RSI_14"), ("ROC", 10, "ROC_10"), ("MOM", 10, "MOM_10"),
    ("CMO", 14, "CMO_14"), ("TRIX", 15, "TRIX_15"), ("STDDEV", 20, "STDDEV_20"),
    ("ATR", 14, "ATR_14"), ("NATR", 14, "NATR_14"),
    ("ADX", 14, "ADX_14"), ("ADXR", 14, "ADXR_14"), ("DX", 14, "DX_14"),
    ("PLUS_DI", 14, "PLUS_DI_14"), ("MINUS_DI", 14, "MINUS_DI_14"),
    ("CCI", 20, "CCI_20"), ("WILLR", 14, "WILLR_14"), ("MFI", 14, "MFI_14"),
    ("DONCHIAN_UPPER", 20, "DONCHIAN_UPPER_20"),
    ("DONCHIAN_LOWER", 20, "DONCHIAN_LOWER_20"),
    ("AROONOSC", 14, "AROONOSC_14"),
]


@pytest.mark.parametrize("name,period,col", _LOOKAHEAD_CASES)
def test_periodic_indicator_no_lookahead(df, name, period, col):
    full = I.add_all_indicators(df, {name: [period]})
    assert col in full.columns, f"{name} produced no {col}"
    for i in (60, 120, 200):
        trunc = I.add_all_indicators(df.iloc[: i + 1], {name: [period]})
        a, b = full[col].iloc[i], trunc[col].iloc[i]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"{col}@{i}: NaN-state differs"
        else:
            assert abs(a - b) < 1e-9, f"{col}@{i}: look-ahead leak {a} != {b}"


# (config-key, params, output column) for multi-output / scalar studies.
_LOOKAHEAD_SCALAR = [
    ("MACD", [], "MACD"), ("MACD", [], "MACD_SIGNAL"),
    ("BB", [20], "BB_UPPER_20"), ("BB", [20], "BB_LOWER_20"),
    ("STOCH", {}, "STOCH_K"), ("STOCH", {}, "STOCH_D"),
    ("STOCHRSI", {}, "STOCHRSI_K"), ("SAR", {}, "SAR"),
    ("OBV", [], "OBV"), ("AD", [], "AD"), ("ADOSC", {}, "ADOSC"),
    ("ULTOSC", {}, "ULTOSC"), ("PPO", {}, "PPO"), ("APO", {}, "APO"),
    ("TRANGE", [], "TRANGE"), ("SUPERTREND", {}, "SUPERTREND"),
    ("CDL_ENGULFING", [], "CDL_ENGULFING"),
]


@pytest.mark.parametrize("name,params,col", _LOOKAHEAD_SCALAR)
def test_scalar_indicator_no_lookahead(df, name, params, col):
    full = I.add_all_indicators(df, {name: params})
    assert col in full.columns, f"{name} produced no {col}"
    for i in (80, 150, 210):
        trunc = I.add_all_indicators(df.iloc[: i + 1], {name: params})
        a, b = full[col].iloc[i], trunc[col].iloc[i]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"{col}@{i}: NaN-state differs"
        else:
            assert abs(a - b) < 1e-9, f"{col}@{i}: look-ahead leak {a} != {b}"


# ── (c) Single source of truth: add_all == compute_column == formula paths ─────

def test_compute_column_matches_add_all_indicators(df):
    for name, period, col in _LOOKAHEAD_CASES:
        via_all = I.add_all_indicators(df, {name: [period]})[col]
        via_one = I.compute_indicator_column(df, name, period)
        np.testing.assert_allclose(
            via_all.to_numpy(dtype=float), via_one.to_numpy(dtype=float),
            rtol=1e-12, atol=1e-12, equal_nan=True,
            err_msg=f"{col}: add_all_indicators != compute_indicator_column",
        )


_FORMULAS = [
    "CCI(20) > 100", "WILLR(14) < -80", "MFI(14) > 70", "ADXR(14) > 20",
    "ROC(10) > 0", "CMO(14) > 0", "AROONOSC(14) > 0",
    "CLOSE > DONCHIAN_UPPER(20)", "CLOSE > KAMA(30)", "DEMA(10) > TEMA(10)",
    "NATR(14) > 0.1", "STDDEV(20) > 0.2",
    "STOCH_K > 80", "STOCH_K > STOCH_D", "CLOSE > SAR", "OBV > PREV(OBV, 3)",
    "MACD > MACD_SIGNAL", "ULTOSC > 50", "PPO > 0", "CLOSE > VWAP",
    "CDL_ENGULFING > 0", "AVG(ATR(14), 10) > 0",
]


@pytest.mark.parametrize("formula", _FORMULAS)
def test_slow_fast_and_precompute_agree(df, formula):
    """The guardrail: a formula evaluates identically via the slow (pandas) path,
    the fast (numpy array) path, AND precompute-if-missing on a raw frame. If any
    diverge, the backtest would not match live evaluation."""
    comp = compile_condition(formula)
    assert comp is not None

    # Runner-style precompute, then both compiled paths.
    cfg = _merge_indicator_requirements({}, comp)
    prepared = I.add_all_indicators(df, cfg)
    ensure_scalar_columns(prepared, comp.scalar_refs)
    arrays = build_arrays_from_df(prepared)
    n = len(prepared)

    slow = [comp.evaluate(prepared, i) for i in range(n)]
    fast = (
        [comp.evaluate_arrays(arrays, n, i) for i in range(n)]
        if comp.fast_path_safe else slow
    )
    # precompute-if-missing: direct evaluate on a RAW frame (no precomputed cols)
    raw = df.copy()
    lazy = [evaluate_condition(formula, raw, i) for i in range(n)]

    assert slow == fast, f"{formula}: slow != fast path"
    assert slow == lazy, f"{formula}: slow != precompute-if-missing"
    # A non-trivial formula should fire at least once on this data (sanity that
    # we're comparing real signals, not all-False).
    assert any(slow), f"{formula}: never True — check fixture/threshold"
