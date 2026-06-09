"""
Phase 2 — volatility-band entry gate.

Proves the gate BLOCKS when it should and PASSES when it should, in BOTH the
backtest simulator (quant_engine) AND the live evaluator
(app/services/execution/entry_gates.py) — the parity contract that keeps a
backtest honest. Both sides read the SAME TA-Lib NATR/ATR study.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quant_engine"))

from engine import indicators as I
from engine.simulator import simulate_trades


def _df_with_volatility(n: int = 60, *, swing: float = 1.0) -> pd.DataFrame:
    """Intraday-ish frame with a controllable per-bar range so NATR is non-trivial.
    A steady up-drift with `swing` wide bars; entry condition fires throughout."""
    idx = pd.date_range("2026-01-05 03:45", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100.0 + np.arange(n) * 0.05, index=idx)
    df = pd.DataFrame(
        {
            "open": close.values,
            "high": (close + swing).values,
            "low": (close - swing).values,
            "close": close.values,
            "volume": [1000.0] * n,
        },
        index=idx,
    )
    return I.add_all_indicators(df, {"NATR": [14], "ATR": [14]})


_COMMON = dict(
    symbol="TEST.NS",
    entry_condition="CLOSE > 0",   # always true → only the gate can block
    exit_condition="",
    stop_loss_pct=5.0,
    take_profit_pct=10.0,
    slippage_bps=0.0,
    commission_bps=0.0,
    warm_up_candles=14,
    max_holding_candles=5,
    objective="intraday",
)


def _natr_last(df):
    return float(df["NATR_14"].dropna().iloc[-1])


# ── Backtest simulator parity ──────────────────────────────────────────────────

def test_backtest_gate_blocks_when_below_min():
    df = _df_with_volatility()
    floor = _natr_last(df) + 5.0  # min far above actual NATR → everything blocked
    trades, diags = simulate_trades(
        df=df, **_COMMON,
        vol_filter_metric="natr", vol_filter_window=14, vol_filter_min=floor,
    )
    assert trades == [], "no trade should fill when NATR is below the band floor"
    # At least one evaluated bar must be explicitly blocked by the volatility gate.
    assert any(d["entry_blocked_volatility"] for d in diags)


def test_backtest_gate_blocks_when_above_max():
    df = _df_with_volatility()
    ceil = max(_natr_last(df) - 0.001, 0.0)  # max below actual NATR → blocked
    trades, diags = simulate_trades(
        df=df, **_COMMON,
        vol_filter_metric="natr", vol_filter_window=14, vol_filter_max=ceil,
    )
    assert trades == []
    assert any(d["entry_blocked_volatility"] for d in diags)


def test_backtest_gate_passes_inside_band():
    df = _df_with_volatility()
    natr = _natr_last(df)
    trades, diags = simulate_trades(
        df=df, **_COMMON,
        vol_filter_metric="natr", vol_filter_window=14,
        vol_filter_min=0.0, vol_filter_max=natr + 5.0,   # band straddles NATR
    )
    assert len(trades) >= 1, "a trade should fill when NATR is inside the band"
    assert not any(d["entry_blocked_volatility"] for d in diags)


def test_backtest_gate_disabled_by_default():
    """No vol_filter_* args → identical to omitting the gate entirely."""
    df = _df_with_volatility()
    a, _ = simulate_trades(df=df, **_COMMON)
    b, _ = simulate_trades(df=df, **_COMMON, vol_filter_metric=None)
    assert [t.exit_reason for t in a] == [t.exit_reason for t in b]
    assert len(a) >= 1


# ── Live evaluator parity ───────────────────────────────────────────────────────

def _live(df, **gate_kwargs):
    from app.schemas.execution import AssetClass, GatesConfig
    from app.services.execution.entry_gates import evaluate_entry_gates

    class _NullRule:
        def evaluate_entry(self, *a, **k):
            return True

    # Use the crypto (24/7 UTC) asset class so the session entry_window gate
    # never fires — this test isolates the volatility gate specifically.
    return evaluate_entry_gates(
        df=df,
        gates=GatesConfig(**gate_kwargs),
        exec_state=None,
        side="BUY",
        rule_engine=_NullRule(),
        entry_block={"trigger": {}, "filters": []},
        asset_class=AssetClass.crypto_spot,
    )


def test_live_gate_blocks_when_below_min():
    df = _df_with_volatility()
    floor = _natr_last(df) + 5.0
    res = _live(df, vol_filter_metric="natr", vol_filter_window=14, vol_filter_min=floor)
    assert res.passed is False
    assert res.blocked_by == "volatility"


def test_live_gate_blocks_when_above_max():
    df = _df_with_volatility()
    ceil = max(_natr_last(df) - 0.001, 0.0)
    res = _live(df, vol_filter_metric="natr", vol_filter_window=14, vol_filter_max=ceil)
    assert res.passed is False
    assert res.blocked_by == "volatility"


def test_live_gate_passes_inside_band():
    df = _df_with_volatility()
    natr = _natr_last(df)
    res = _live(df, vol_filter_metric="natr", vol_filter_window=14,
                vol_filter_min=0.0, vol_filter_max=natr + 5.0)
    assert res.passed is True
    assert res.blocked_by is None


def test_live_gate_disabled_by_default():
    df = _df_with_volatility()
    res = _live(df)  # no vol_filter_* → gate inactive
    assert res.passed is True


def test_backtest_and_live_agree_on_same_band():
    """The decisive parity check: identical band + data → identical verdict."""
    df = _df_with_volatility()
    natr = _natr_last(df)
    band = dict(vol_filter_metric="natr", vol_filter_window=14, vol_filter_min=natr + 5.0)  # blocks

    trades, _ = simulate_trades(df=df, **_COMMON, **band)
    live = _live(df, **band)
    backtest_blocks = trades == []
    live_blocks = not live.passed
    assert backtest_blocks == live_blocks is True


# ── Loader wiring ───────────────────────────────────────────────────────────────

def test_loader_parses_volatility_filter_block():
    from engine.loader import load_strategy_from_content
    cfg = load_strategy_from_content(
        """
strategy:
  name: Vol Gate Test
  symbol: TEST.NS
  market: NSE
  timeframe: 15m
  entry: { condition: "CLOSE > 0" }
  exit: { condition: "" }
  risk_management: { stop_loss_percent: 1.5, take_profit_percent: 3.0 }
  volatility_filter: { metric: natr, window: 14, min: 0.5, max: 5.0 }
"""
    )
    assert cfg.vol_filter_metric == "natr"
    assert cfg.vol_filter_window == 14
    assert cfg.vol_filter_min == 0.5
    assert cfg.vol_filter_max == 5.0
