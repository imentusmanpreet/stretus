"""Phase 12 — short-side execution in the simulator + two-sided ("both") wiring.

The engine was long-only: the simulator hard-coded side="LONG" and ignored
`direction`. These tests pin the SHORT mechanics (stop above entry, target below,
mirrored P&L / MAE / MFE / costs) and the runner's two-pass "both" behaviour.
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

from engine.simulator import (
    simulate_trades,
    _compute_initial_stop_short,
    _compute_trailing_ceiling_short,
)


def _df(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "open":  close + 0.2,
            "high":  close + 0.5,
            "low":   close - 0.5,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


_COMMON = dict(
    symbol="TEST",
    entry_condition="CLOSE > 0",   # always true → enter at first opportunity
    exit_condition="",
    stop_loss_pct=5.0,
    take_profit_pct=5.0,
    slippage_bps=0.0,
    commission_bps=0.0,
    objective="positional",
    stt_intraday_sell_pct=0.0,
    stt_delivery_pct=0.0,
)


# ── Stop-helper unit tests ────────────────────────────────────────────────────

def test_short_percent_stop_sits_above_entry():
    stop = _compute_initial_stop_short(
        {"type": "percent", "pct": 2.0}, fallback_pct=5.0, entry_price=100.0,
        entry_index=0, arrays={}, high_arr=np.array([100.0]), day_ordinals=np.array([0]),
    )
    assert stop == pytest.approx(102.0)


def test_short_fallback_stop_above_entry_when_no_spec():
    stop = _compute_initial_stop_short(
        None, fallback_pct=5.0, entry_price=100.0, entry_index=0,
        arrays={}, high_arr=np.array([100.0]), day_ordinals=np.array([0]),
    )
    assert stop == pytest.approx(105.0)


def test_short_atr_stop_adds_atr_above_entry():
    arrays = {"ATR_14": np.array([2.0, 2.0])}
    stop = _compute_initial_stop_short(
        {"type": "atr", "multiplier": 1.5, "window": 14}, fallback_pct=5.0,
        entry_price=100.0, entry_index=1, arrays=arrays,
        high_arr=np.array([100.0, 100.0]), day_ordinals=np.array([0, 0]),
    )
    assert stop == pytest.approx(103.0)   # 100 + 1.5 * 2.0


def test_short_trailing_ceiling_ratchets_with_lowest_low():
    ceiling = _compute_trailing_ceiling_short(
        {"type": "percent", "distance_pct": 1.0}, entry_price=100.0,
        current_index=0, arrays={}, lowest_low_since_entry=90.0, current_close=90.0,
    )
    assert ceiling == pytest.approx(90.9)   # lowest low + 1%


def test_short_trailing_inactive_returns_inf_before_activation():
    ceiling = _compute_trailing_ceiling_short(
        {"type": "percent", "distance_pct": 1.0, "activate_after_pct": 3.0},
        entry_price=100.0, current_index=0, arrays={},
        lowest_low_since_entry=99.0, current_close=99.0,   # only +1% gain
    )
    assert ceiling == float("inf")


# ── Full-trade behaviour ──────────────────────────────────────────────────────

def test_short_profits_in_falling_market():
    trades, _ = simulate_trades(df=_df(np.linspace(100, 78, 12)), side="SHORT", **_COMMON)
    assert trades, "short should have taken a trade"
    t = trades[0]
    assert t.side == "SHORT"
    assert t.exit_reason == "TAKE_PROFIT"
    assert t.pnl_pct == pytest.approx(5.0, abs=1e-6)


def test_short_loses_in_rising_market():
    trades, _ = simulate_trades(df=_df(np.linspace(80, 100, 12)), side="SHORT", **_COMMON)
    t = trades[0]
    assert t.exit_reason == "STOP_LOSS"
    assert t.pnl_pct == pytest.approx(-5.0, abs=1e-6)


def test_short_is_the_mirror_of_long():
    falling = _df(np.linspace(100, 78, 12))
    long_t, _  = simulate_trades(df=falling, side="LONG",  **_COMMON)
    short_t, _ = simulate_trades(df=falling, side="SHORT", **_COMMON)
    # Same entry, opposite P&L, mirrored MAE/MFE.
    assert long_t[0].entry_price == pytest.approx(short_t[0].entry_price)
    assert long_t[0].pnl_pct == pytest.approx(-short_t[0].pnl_pct, abs=1e-6)
    assert long_t[0].mae_pct == pytest.approx(-short_t[0].mfe_pct, abs=1e-6)
    assert long_t[0].mfe_pct == pytest.approx(-short_t[0].mae_pct, abs=1e-6)


def test_short_entry_costs_use_sell_side():
    # With STT only on the sell leg, a short SELLS at entry → the effective
    # entry fill is BELOW the raw next-open (sell-side cost), unlike a long.
    df = _df(np.linspace(100, 78, 12))
    common = {**_COMMON, "stt_delivery_pct": 0.1, "slippage_bps": 0.0, "commission_bps": 0.0}
    short_t, _ = simulate_trades(df=df, side="SHORT", **common)
    raw_next_open = float(df["open"].iloc[1])
    assert short_t[0].entry_price < raw_next_open   # sell-side cost pulls it down


def test_default_side_is_long_unchanged():
    # Omitting side must preserve the legacy long behaviour exactly.
    df = _df(np.linspace(80, 100, 12))
    a, _ = simulate_trades(df=df, **_COMMON)
    b, _ = simulate_trades(df=df, side="LONG", **_COMMON)
    assert a[0].side == "LONG"
    assert a[0].pnl_pct == pytest.approx(b[0].pnl_pct)


# ── Loader: short leg parsing ──────────────────────────────────────────────────

def test_loader_parses_short_leg_conditions():
    from engine.loader import load_strategy_from_content
    cfg = load_strategy_from_content("\n".join([
        "strategy:",
        '  name: "Both"',
        "  symbol: TEST.NS",
        "  market: indian_stocks",
        "  timeframe: 1d",
        "  direction: both",
        "  entry: { condition: \"CLOSE > OPEN\" }",
        "  exit:  { condition: \"PROFIT >= TAKE_PROFIT_TARGET\" }",
        "  short_entry_condition: \"CLOSE < OPEN\"",
        "  short_exit_condition: \"PROFIT >= TAKE_PROFIT_TARGET\"",
        "  indicators: {}",
        "  risk_management: { stop_loss_percent: 1.0, take_profit_percent: 1.0 }",
    ]))
    assert cfg.direction == "both"
    assert cfg.short_entry_condition == "CLOSE < OPEN"
    assert cfg.short_exit_condition.strip()


# ── End-to-end: a "both" strategy produces BOTH long and short trades ──────────

def test_run_backtest_both_direction_produces_long_and_short_trades():
    from engine.runner import run_backtest

    yaml_content = "\n".join([
        "strategy:",
        '  name: "Both Direction"',
        "  symbol: BTC_USDT",
        "  market: crypto",        # 24/7 → no NSE session-window gate on daily bars
        "  timeframe: 1d",
        "  direction: both",
        "  entry: { condition: \"CLOSE > OPEN\" }",
        "  exit:  { condition: \"PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET\" }",
        "  short_entry_condition: \"CLOSE < OPEN\"",
        "  short_exit_condition: \"PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET\"",
        "  indicators: {}",
        "  risk_management: { stop_loss_percent: 1.5, take_profit_percent: 1.5 }",
    ])

    # Daily bars inside the enforced window [2024-01-01, now]: alternating up /
    # down candles so the long leg (CLOSE>OPEN) AND short leg (CLOSE<OPEN) both
    # fire across the series.
    ohlcv = []
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    price = 100.0
    for i in range(120):
        up = (i % 2 == 0)
        o = price
        c = price * (1.02 if up else 0.98)
        hi = max(o, c) * 1.01
        lo = min(o, c) * 0.99
        ohlcv.append({
            "timestamp": (base + pd.Timedelta(days=i)).isoformat().replace("+00:00", "Z"),
            "open": float(o), "high": float(hi), "low": float(lo),
            "close": float(c), "volume": 1000.0,
        })
        price = c

    result = run_backtest(
        yaml_content, ohlcv,
        {"starting_balance": 10000.0, "slippage_bps": 0.0, "commission_bps": 0.0},
        {"symbol": "TEST", "interval": "1d",
         "from_utc": "2024-01-01T00:00:00Z", "to_utc": "2026-01-01T00:00:00Z"},
        "both-dir-ref",
    )

    sides = {t["side"] for t in result["metrics"]["backtest_trades"]}
    assert "LONG" in sides, f"expected long trades, got sides={sides}"
    assert "SHORT" in sides, f"expected short trades, got sides={sides}"
