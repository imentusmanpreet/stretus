"""Phase 8b — wall-clock intraday cutoff (`time_exit` block).

Covers the loader's parser, end-to-end behavior in the simulator
(force-exit + block-new-entry), and backward compat for strategies
that don't declare a time_exit.
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

from engine.loader import _parse_time_exit_spec, load_strategy_from_content
from engine.simulator import simulate_trades


# ── Loader: _parse_time_exit_spec ────────────────────────────────────────────


def test_parse_time_exit_returns_none_when_block_absent():
    assert _parse_time_exit_spec(None) is None
    assert _parse_time_exit_spec({}) is None
    assert _parse_time_exit_spec("") is None


def test_parse_time_exit_converts_ist_to_utc_minutes():
    """15:15 IST = 09:45 UTC = 9*60 + 45 = 585 minutes since UTC midnight."""
    spec = _parse_time_exit_spec({"exit_time": "15:15", "timezone": "Asia/Kolkata"})
    assert spec["exit_time"] == "15:15"
    assert spec["timezone"] == "Asia/Kolkata"
    assert spec["utc_minutes_of_day"] == 585


def test_parse_time_exit_default_timezone_is_ist():
    spec = _parse_time_exit_spec({"exit_time": "15:15"})
    assert spec["timezone"] == "Asia/Kolkata"
    assert spec["utc_minutes_of_day"] == 585


def test_parse_time_exit_utc_passes_through():
    spec = _parse_time_exit_spec({"exit_time": "09:45", "timezone": "UTC"})
    assert spec["utc_minutes_of_day"] == 9 * 60 + 45


def test_parse_time_exit_handles_wrap_for_early_morning_local():
    """02:00 IST = 20:30 UTC the prior day. mod 1440 wraps correctly."""
    spec = _parse_time_exit_spec({"exit_time": "02:00", "timezone": "Asia/Kolkata"})
    # 2*60 - 330 = -210; mod 1440 = 1230 minutes = 20:30 UTC ✓
    assert spec["utc_minutes_of_day"] == 1230


def test_parse_time_exit_rejects_bad_format():
    with pytest.raises(ValueError, match="HH:MM"):
        _parse_time_exit_spec({"exit_time": "3pm"})


def test_parse_time_exit_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _parse_time_exit_spec({"exit_time": "25:00"})
    with pytest.raises(ValueError, match="out of range"):
        _parse_time_exit_spec({"exit_time": "10:99"})


def test_parse_time_exit_rejects_unsupported_timezone():
    with pytest.raises(ValueError, match="not supported"):
        _parse_time_exit_spec({"exit_time": "15:15", "timezone": "America/New_York"})


def test_parse_time_exit_requires_exit_time():
    with pytest.raises(ValueError, match="exit_time is required"):
        _parse_time_exit_spec({"timezone": "UTC"})


def test_parse_time_exit_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a mapping"):
        _parse_time_exit_spec("15:15")


def test_loader_carries_time_exit_into_strategy_config():
    yaml_content = """
strategy:
  name: t
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 5m
  entry: { condition: "CLOSE > 0" }
  risk_management: { stop_loss_percent: 1.5, take_profit_percent: 3.0 }
  time_exit:
    exit_time: "15:15"
    timezone:  "Asia/Kolkata"
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.time_exit_spec is not None
    assert cfg.time_exit_spec["exit_time"] == "15:15"
    assert cfg.time_exit_spec["utc_minutes_of_day"] == 585


# ── Simulator: end-to-end behavior ───────────────────────────────────────────


def _intraday_df(timestamps_utc: list[pd.Timestamp], closes: list[float]) -> pd.DataFrame:
    """Build an intraday OHLCV df with the given UTC timestamps."""
    assert len(timestamps_utc) == len(closes)
    return pd.DataFrame({
        "open":   closes,
        "high":   [c + 0.5 for c in closes],
        "low":    [c - 0.5 for c in closes],
        "close":  closes,
        "volume": [1000.0] * len(closes),
    }, index=pd.DatetimeIndex(timestamps_utc))


def test_simulator_force_exits_open_position_at_cutoff():
    """Construct a series where an entry fills BEFORE the cutoff and no other
    exit fires — TIME_EXIT must close it at the cutoff bar's close."""
    # 09:00-10:00 UTC, 5m bars (12 bars). 15:00 IST = 09:30 UTC starts the
    # window leading into the cutoff. We want one-shot entry on bar 1.
    ts = [pd.Timestamp("2026-01-05 09:00") + pd.Timedelta(minutes=5 * i) for i in range(13)]
    # Cross-up at index 1 (98 → 100): one-shot entry at index 2.
    closes = [98.0, 100.0] + [100.0 + 0.1 * i for i in range(11)]
    df = _intraday_df(ts, closes)

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=50,
        objective="intraday",
        time_exit_spec={"exit_time": "15:15", "timezone": "Asia/Kolkata",
                        "utc_minutes_of_day": 585},
    )
    assert len(trades) >= 1
    # First trade exits via TIME_EXIT at the first bar whose UTC minute >= 585.
    # That's 09:45 UTC = index 9 in our series (09:00 + 9 * 5m).
    assert trades[0].exit_reason == "TIME_EXIT"


def test_simulator_unchanged_without_time_exit_spec():
    """Backward compat: omitting time_exit_spec keeps prior behavior."""
    ts = [pd.Timestamp("2026-01-05 09:00") + pd.Timedelta(minutes=5 * i) for i in range(13)]
    closes = [98.0, 100.0] + [100.0 + 0.1 * i for i in range(11)]
    df = _intraday_df(ts, closes)
    common = dict(
        df=df, symbol="TEST.NS",
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=10.0, take_profit_pct=20.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0, max_holding_candles=50, objective="intraday",
    )
    a, _ = simulate_trades(**common)
    b, _ = simulate_trades(**common, time_exit_spec=None)
    assert [t.exit_reason for t in a] == [t.exit_reason for t in b]


def test_simulator_blocks_new_entries_past_cutoff():
    """An entry signal that fires at a post-cutoff bar must NOT fill;
    the diagnostic at that bar should set entry_blocked_time_exit."""
    # 09:30-10:00 UTC, 5m bars. Cross-up at index 5 (which is 09:55 UTC,
    # well past the 09:45 cutoff).
    ts = [pd.Timestamp("2026-01-05 09:30") + pd.Timedelta(minutes=5 * i) for i in range(8)]
    closes = [98.0, 98.0, 98.0, 98.0, 98.0, 100.0, 101.0, 102.0]
    df = _intraday_df(ts, closes)

    trades, diags = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE > 99 AND PREV(CLOSE, 1) <= 99",
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="intraday",
        time_exit_spec={"exit_time": "15:15", "timezone": "Asia/Kolkata",
                        "utc_minutes_of_day": 585},
    )
    assert trades == []
    blocked = [d for d in diags if d.get("entry_blocked_time_exit") and d.get("entry_signal")]
    assert len(blocked) >= 1
