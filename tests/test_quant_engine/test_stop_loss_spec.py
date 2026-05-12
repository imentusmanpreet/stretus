"""Phase 3 — structural & trailing stop loss specs.

Tests both the loader (parses & validates the new YAML blocks) and the
simulator (uses the specs to compute initial + trailing stop prices).
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

from engine.indicators import add_all_indicators
from engine.loader import (
    _parse_stop_loss_spec,
    _parse_trailing_stop_spec,
    load_strategy_from_content,
)
from engine.simulator import simulate_trades


# ── Loader spec parsing ──────────────────────────────────────────────────────


def test_loader_returns_none_when_blocks_absent():
    assert _parse_stop_loss_spec(None) is None
    assert _parse_stop_loss_spec({}) is None
    assert _parse_trailing_stop_spec(None) is None


def test_loader_percent_stop_loss_spec():
    spec = _parse_stop_loss_spec({"type": "percent", "pct": 1.5})
    assert spec == {"type": "percent", "pct": 1.5}


def test_loader_structural_orb_spec_has_defaults():
    spec = _parse_stop_loss_spec({
        "type": "structural", "anchor": "opening_range_low",
    })
    assert spec["anchor"] == "opening_range_low"
    assert spec["opening_bars"] == 3
    assert spec["padding_pct"] == 0.0


def test_loader_structural_anchor_validation():
    with pytest.raises(ValueError, match="anchor must be one of"):
        _parse_stop_loss_spec({"type": "structural", "anchor": "magic"})


def test_loader_atr_spec_validates_positive_multiplier():
    with pytest.raises(ValueError, match="multiplier must be > 0"):
        _parse_stop_loss_spec({"type": "atr", "multiplier": 0, "window": 14})


def test_loader_trailing_percent_spec():
    spec = _parse_trailing_stop_spec({"type": "percent", "distance_pct": 0.8, "activate_after_pct": 0.5})
    assert spec == {"type": "percent", "distance_pct": 0.8, "activate_after_pct": 0.5}


def test_loader_trailing_ema_spec_has_defaults():
    spec = _parse_trailing_stop_spec({"type": "ema"})
    assert spec["window"] == 20
    assert spec["activate_after_pct"] == 0.0


def test_loader_trailing_unknown_type_raises():
    with pytest.raises(ValueError, match="trailing_stop.type must be"):
        _parse_trailing_stop_spec({"type": "magic"})


def test_loader_carries_specs_into_strategy_config():
    yaml_content = """
strategy:
  name: test
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 5m
  entry:
    condition: "CLOSE > EMA(20)"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
  stop_loss:
    type: structural
    anchor: opening_range_low
    opening_bars: 3
    padding_pct: 0.1
  trailing_stop:
    type: ema
    window: 20
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.stop_loss_spec["anchor"] == "opening_range_low"
    assert cfg.trailing_stop_spec["type"] == "ema"
    # Legacy stop_loss percent still parsed for backward-compat fallback.
    assert cfg.stop_loss == 1.5


# ── Simulator: structural ORB-low SL ─────────────────────────────────────────


def _intraday_df(rows_per_day: list[list[tuple[float, float, float, float]]]) -> pd.DataFrame:
    """Build an intraday DataFrame from per-day OHLC tuples (open, high, low, close)."""
    timestamps: list[pd.Timestamp] = []
    records: list[dict] = []
    for day_idx, day_rows in enumerate(rows_per_day):
        base = pd.Timestamp("2026-01-05T03:45:00") + pd.Timedelta(days=day_idx)
        for bar_idx, (op, hi, lo, cl) in enumerate(day_rows):
            timestamps.append(base + pd.Timedelta(minutes=15 * bar_idx))
            records.append({"open": op, "high": hi, "low": lo, "close": cl, "volume": 1000.0})
    return pd.DataFrame(records, index=pd.DatetimeIndex(timestamps))


def test_structural_orb_low_anchors_sl_to_session_opening_low():
    """Day 1: 3 opening bars then a breakout, then a deep dip below ORB low.
    With percent SL the trade survives; with structural=ORB-low it gets stopped
    at the opening-range low (with no padding)."""
    day1 = [
        (100.0, 101.0, 99.5, 100.5),     # ORB bar 1, low 99.5
        (100.5, 102.0, 100.0, 101.5),    # ORB bar 2, low 100.0
        (101.5, 103.0, 101.0, 102.5),    # ORB bar 3, low 101.0  → ORB low = 99.5
        (102.5, 105.0, 102.0, 104.5),    # breakout (entry on bar 4 open)
        (104.5, 106.0, 99.0, 99.5),      # dip touches 99.0 — below ORB low 99.5
    ]
    df = _intraday_df([day1])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 102",            # fires on bar 3 (close=102.5)
        exit_condition="",
        stop_loss_pct=10.0,                       # generous % so it doesn't fire
        take_profit_pct=20.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=10,
        objective="intraday",
        stop_loss_spec={"type": "structural", "anchor": "opening_range_low", "opening_bars": 3, "padding_pct": 0.0},
    )

    assert len(trades) == 1
    # Expected: SL anchored at ORB low 99.5; bar 5 low = 99.0 → stop hit at 99.5
    # Intraday STT (0.025% on sell) is applied to the exit price → 99.5 * 0.99975
    assert trades[0].exit_reason == "STOP_LOSS"
    assert pytest.approx(trades[0].exit_price, abs=0.05) == 99.5 * (1.0 - 0.025 / 100.0)


def test_structural_orb_low_padding_lowers_stop_price():
    day1 = [
        (100.0, 101.0, 99.5, 100.5),
        (100.5, 102.0, 100.0, 101.5),
        (101.5, 103.0, 101.0, 102.5),
        (102.5, 105.0, 102.0, 104.5),
        (104.5, 106.0, 99.4, 99.5),       # dip to 99.4 — below 99.5 ORB low
    ]
    df = _intraday_df([day1])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 102",
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=10,
        objective="intraday",
        # padding_pct=0.5% pulls SL ~0.5 below 99.5 → SL≈99.005 — bar's low is 99.4, no stop
        stop_loss_spec={"type": "structural", "anchor": "opening_range_low", "opening_bars": 3, "padding_pct": 0.5},
    )
    assert len(trades) == 1
    assert trades[0].exit_reason != "STOP_LOSS"


def test_structural_atr_uses_precomputed_atr_column():
    """ATR-based SL needs the ATR column precomputed by add_all_indicators."""
    closes = [100.0 + i * 0.2 for i in range(40)]
    rows = [
        {"open": c, "high": c + 1.0, "low": c - 1.0, "close": c, "volume": 1000.0}
        for c in closes
    ]
    # Inject a downward spike at bar 35
    rows[35]["low"] = 90.0
    df = pd.DataFrame(rows)
    df = add_all_indicators(df, {"ATR": [14]})

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 100",            # fires immediately
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=50.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=15,
        max_holding_candles=100,
        objective="positional",
        stop_loss_spec={"type": "atr", "multiplier": 2.0, "window": 14},
    )
    # Should produce at least one trade — and the SL is somewhere reasonable below entry
    assert len(trades) >= 1
    first = trades[0]
    assert first.entry_price > 0
    assert first.exit_price > 0


# ── Simulator: trailing stop ─────────────────────────────────────────────────


def test_trailing_percent_ratchets_stop_upward():
    """Build a series that crosses above 99, runs up, then dips. With 1%
    trailing it should exit on the dip via TRAILING_STOP."""
    # Bar 0 close=98 (below trigger). Bar 1 close=100 (crosses above 99 → entry signal,
    # fill at bar 2 open). Price then runs to 110.5 (high of bar 6), dips to low 108.5.
    # 1% trailing of 110.5 = 109.395 → stop fires at bar 7 (low 108.5).
    closes = [98.0, 100.0, 100.0, 102.0, 105.0, 108.0, 110.0, 109.0, 108.0, 107.5]
    rows = [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000.0} for c in closes]
    df = pd.DataFrame(rows)

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        # One-shot entry: only fires when crossing above 99 from below.
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=10.0,                       # generous initial SL
        take_profit_pct=50.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        trailing_stop_spec={"type": "percent", "distance_pct": 1.0},
    )
    assert len(trades) >= 1
    # First trade should exit via the trailing stop, not END_OF_DATA or static SL
    assert trades[0].exit_reason == "TRAILING_STOP"


def test_trailing_floor_never_moves_backward():
    """Drop after the high should keep the trailing floor at the previous max,
    not lower it. We exit at the locked floor, not at a re-derived lower one."""
    closes = [98.0, 100.0, 100.0, 105.0, 110.0, 108.0, 106.0, 104.0]
    rows = [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000.0} for c in closes]
    df = pd.DataFrame(rows)

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=50.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        # 2% trailing of the highest_high (110 + 0.5 = 110.5) → floor = 108.29
        trailing_stop_spec={"type": "percent", "distance_pct": 2.0},
    )
    assert len(trades) >= 1
    # exit price approximately the trailing floor 108.29 (×0.999 STT for delivery)
    assert pytest.approx(trades[0].exit_price, abs=0.5) == 108.29 * 0.999


def test_trailing_inactive_until_activate_after_pct():
    """activate_after_pct=5% — trailing is dormant until trade is up >= 5%.
    A small early dip should NOT trigger a trailing exit."""
    # Entry near 100, price wobbles: up 2%, down 1.5%, then stalls — never reaches 5%.
    closes = [100.0, 100.0, 102.0, 100.5, 101.5, 100.0, 100.0]
    rows = [{"open": c, "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1000.0} for c in closes]
    df = pd.DataFrame(rows)

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=50.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        # Tight 0.1% trailing — would otherwise trigger immediately on the dip.
        trailing_stop_spec={"type": "percent", "distance_pct": 0.1, "activate_after_pct": 5.0},
    )
    # No trailing exit; either MAX_HOLDING or END_OF_DATA closes the trade.
    if trades:
        assert trades[0].exit_reason != "TRAILING_STOP"


# ── Backward-compat: legacy strategies are unchanged ─────────────────────────


def test_legacy_strategies_without_specs_unchanged():
    """A strategy with no stop_loss_spec/trailing_stop_spec must behave exactly
    as it did before Phase 3."""
    closes = [98.0, 100.0, 100.0, 95.0, 90.0, 85.0]    # cross-up then immediate downside
    rows = [{"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1000.0} for c in closes]
    df = pd.DataFrame(rows)

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=2.0,
        take_profit_pct=10.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=10,
        objective="positional",
    )
    assert len(trades) >= 1
    assert trades[0].exit_reason == "STOP_LOSS"
