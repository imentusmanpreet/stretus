"""Phase 4 — cross-symbol reference data.

Covers the loader's reference_symbol field, the OHLCV merge helper, the new
AST identifiers (REF_*) and function (RS), and the end-to-end runner path.
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

from engine.conditions import compile_condition, evaluate_condition
from engine.data import merge_reference_data
from engine.loader import load_strategy_from_content


# ── Loader: reference_symbol field ──────────────────────────────────────────


def test_loader_parses_reference_symbol():
    yaml_content = """
strategy:
  name: rs_test
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  reference_symbol: "^NSEI"
  entry:
    condition: "RS(20) > 1.0"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.reference_symbol == "^NSEI"


def test_loader_uppercases_and_strips_reference_symbol():
    yaml_content = """
strategy:
  name: t
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  reference_symbol: "  ^nsei  "
  entry:
    condition: "CLOSE > 0"
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 2.0
"""
    assert load_strategy_from_content(yaml_content).reference_symbol == "^NSEI"


def test_loader_returns_none_when_reference_symbol_absent():
    yaml_content = """
strategy:
  name: t
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  entry:
    condition: "CLOSE > 0"
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 2.0
"""
    assert load_strategy_from_content(yaml_content).reference_symbol is None


# ── merge_reference_data ─────────────────────────────────────────────────────


def _df(dates, closes):
    return pd.DataFrame({
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low":  [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * len(closes),
    }, index=pd.DatetimeIndex(dates))


def test_merge_reference_adds_prefixed_columns():
    main = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [100.0] * 5)
    ref  = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [200.0] * 5)
    merged = merge_reference_data(main, ref)
    for col in ("REF_open", "REF_high", "REF_low", "REF_close", "REF_volume"):
        assert col in merged.columns
    # Original columns unchanged
    assert (merged["close"] == 100.0).all()
    assert (merged["REF_close"] == 200.0).all()


def test_merge_reference_forward_fills_missing_bars():
    """Reference is missing a bar (e.g. brief data outage) — should ffill."""
    main = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [100.0] * 5)
    ref_dates = [pd.Timestamp("2026-01-01"),
                 pd.Timestamp("2026-01-01 00:30"),
                 pd.Timestamp("2026-01-01 00:45"),
                 pd.Timestamp("2026-01-01 01:00")]
    ref = _df(ref_dates, [200.0, 201.0, 202.0, 203.0])
    merged = merge_reference_data(main, ref)
    # Bar at 00:15 is missing in ref → ffill should carry the 00:00 value
    assert merged["REF_close"].iloc[1] == 200.0


def test_merge_reference_leading_gap_stays_nan():
    """Reference starts AFTER main starts — leading bars stay NaN."""
    main = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [100.0] * 5)
    ref_dates = pd.date_range("2026-01-01 00:30", periods=3, freq="15min")
    ref = _df(ref_dates, [200.0, 201.0, 202.0])
    merged = merge_reference_data(main, ref)
    assert pd.isna(merged["REF_close"].iloc[0])
    assert pd.isna(merged["REF_close"].iloc[1])
    assert merged["REF_close"].iloc[2] == 200.0


def test_merge_reference_rejects_empty():
    main = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [100.0] * 5)
    with pytest.raises(ValueError, match="reference_df is empty"):
        merge_reference_data(main, pd.DataFrame())


# ── AST: REF_CLOSE identifier ────────────────────────────────────────────────


def test_ref_close_identifier_resolves_when_column_present():
    main = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [100.0, 101.0, 102.0, 103.0, 104.0])
    ref  = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [200.0, 201.0, 202.0, 203.0, 204.0])
    merged = merge_reference_data(main, ref)
    assert evaluate_condition("REF_CLOSE > 200", merged, 1) is True
    assert evaluate_condition("REF_CLOSE > 250", merged, 4) is False


def test_ref_close_returns_nan_when_no_reference_data():
    """When no reference is merged, REF_CLOSE should resolve to NaN and
    comparisons should evaluate to False — never crash."""
    main = _df(pd.date_range("2026-01-01", periods=5, freq="15min"), [100.0] * 5)
    assert evaluate_condition("REF_CLOSE > 100", main, 2) is False


# ── AST: RS(n) ──────────────────────────────────────────────────────────────


def test_rs_above_one_when_stock_outperforms_reference():
    """Stock doubles, reference flat → RS > 1."""
    dates = pd.date_range("2026-01-01", periods=12, freq="15min")
    stock_closes = [100.0] * 8 + [110.0, 120.0, 130.0, 140.0]   # rallies last 4 bars
    ref_closes   = [200.0] * 12                                  # flat
    main = _df(dates, stock_closes)
    ref  = _df(dates, ref_closes)
    merged = merge_reference_data(main, ref)
    assert evaluate_condition("RS(8) > 1.0", merged, 11) is True


def test_rs_below_one_when_stock_underperforms_reference():
    """Stock flat, reference rallies → RS < 1."""
    dates = pd.date_range("2026-01-01", periods=12, freq="15min")
    stock_closes = [100.0] * 12
    ref_closes   = [200.0] * 8 + [210.0, 220.0, 230.0, 240.0]
    main = _df(dates, stock_closes)
    ref  = _df(dates, ref_closes)
    merged = merge_reference_data(main, ref)
    assert evaluate_condition("RS(8) < 1.0", merged, 11) is True


def test_rs_equals_one_when_both_move_proportionally():
    """Both move by 10% → RS ~= 1."""
    dates = pd.date_range("2026-01-01", periods=12, freq="15min")
    stock_closes = [100.0] * 8 + [102.5, 105.0, 107.5, 110.0]
    ref_closes   = [200.0] * 8 + [205.0, 210.0, 215.0, 220.0]
    main = _df(dates, stock_closes)
    ref  = _df(dates, ref_closes)
    merged = merge_reference_data(main, ref)
    # RS(8) := (110/100) / (220/200) = 1.1 / 1.1 = 1.0
    assert evaluate_condition("RS(8) > 0.99", merged, 11) is True
    assert evaluate_condition("RS(8) < 1.01", merged, 11) is True


def test_rs_returns_nan_before_window_warmup():
    dates = pd.date_range("2026-01-01", periods=5, freq="15min")
    main = _df(dates, [100.0, 101.0, 102.0, 103.0, 104.0])
    ref  = _df(dates, [200.0, 201.0, 202.0, 203.0, 204.0])
    merged = merge_reference_data(main, ref)
    # RS(8) at bar 4 — only 5 bars of history, n=8 ⇒ NaN ⇒ comparison False
    assert evaluate_condition("RS(8) > 0", merged, 4) is False


def test_rs_returns_nan_when_reference_column_missing():
    """Strategy formula uses RS but caller forgot to merge reference data —
    silently NaN, not crash."""
    main = _df(pd.date_range("2026-01-01", periods=12, freq="15min"), [100.0] * 12)
    assert evaluate_condition("RS(5) > 1.0", main, 11) is False


# ── compile_condition: RS / REF_* are fast-path-unsafe ───────────────────────


def test_compile_marks_rs_as_fast_path_unsafe():
    cond = compile_condition("RS(20) > 1.05")
    assert cond is not None
    assert cond.fast_path_safe is False


# ── End-to-end through run_backtest ──────────────────────────────────────────


def test_run_backtest_requires_reference_when_strategy_declares_one():
    """Strategy with reference_symbol but no reference_ohlcv must fail clearly,
    not silently produce a broken backtest."""
    from engine.runner import run_backtest

    yaml_content = """
strategy:
  name: rs_test
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 1d
  reference_symbol: "^NSEI"
  entry:
    condition: "RS(5) > 1.05"
  risk_management:
    stop_loss_percent: 2.0
    take_profit_percent: 5.0
"""
    rows = [
        {"timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
         "open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i,
         "close": 101.0 + i, "volume": 1000}
        for i in range(30)
    ]
    with pytest.raises(ValueError, match="reference_ohlcv"):
        run_backtest(
            yaml_content,
            ohlcv_data=[{**r, "timestamp": r["timestamp"].isoformat()} for r in rows],
            run_config={},
            market_data_request={"from_utc": "2026-01-01T00:00:00Z", "to_utc": "2026-02-01T00:00:00Z"},
        )


def test_run_backtest_ignores_reference_when_strategy_omits_it(caplog):
    """Strategy without reference_symbol should NOT raise the
    'reference_ohlcv missing' error even if reference_ohlcv is supplied —
    the runner should silently log and proceed. We trigger a downstream
    failure (missing market_data_request keys) just to short-circuit before
    the heavy backtest path; the point is that the *Phase 4 guard* doesn't
    fire.
    """
    import logging
    from engine.runner import run_backtest

    yaml_content = """
strategy:
  name: simple
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 1d
  entry:
    condition: "CLOSE > 100"
  risk_management:
    stop_loss_percent: 2.0
    take_profit_percent: 5.0
"""
    rows = [
        {"timestamp": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).isoformat(),
         "open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i,
         "close": 101.0 + i, "volume": 1000}
        for i in range(30)
    ]
    caplog.set_level(logging.INFO, logger="engine.runner")
    # The runner may raise downstream (missing market_data_request fields),
    # but it must NOT raise the Phase 4 reference-missing ValueError.
    try:
        run_backtest(
            yaml_content,
            ohlcv_data=rows,
            run_config={},
            market_data_request={"from_utc": "2026-01-01T00:00:00Z", "to_utc": "2026-02-01T00:00:00Z"},
            reference_ohlcv=rows,
        )
    except ValueError as exc:
        assert "reference_ohlcv" not in str(exc), (
            "Phase 4 guard fired even though strategy declares no reference_symbol"
        )
    except KeyError:
        pass  # downstream path needs more fixture wiring; not the test's concern
    # Confirm the runner logged the "ignoring reference" message.
    assert any("ignoring" in r.message.lower() for r in caplog.records)
