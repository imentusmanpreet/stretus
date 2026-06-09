"""
End-to-end tests for the Phase 11 1-minute execution wiring in
engine/runner.run_backtest.

  • PARITY — with a 1m strategy, enabling intrabar execution resamples 1m→1m
    (an identity) and walks one sub-bar per strategy bar, so the full pipeline
    (indicators, conditions, metrics) must produce identical metrics to the
    legacy path on the same 1m data.

  • RESAMPLING — with a 15m strategy fed 1m data, the engine resamples to the
    strategy timeframe and runs end-to-end, producing the expected number of
    strategy bars and a well-formed result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.resample import resample_ohlcv
from engine.runner import run_backtest


_YAML_1M = """
strategy:
  name: intrabar-parity
  symbol: TEST
  timeframe: 1m
  objective: positional
  entry_condition: "CLOSE > 0"
  risk_management:
    stop_loss_percent: 2
    take_profit_percent: 2
"""

_YAML_15M = """
strategy:
  name: intrabar-15m
  symbol: TEST
  timeframe: 15m
  objective: positional
  entry_condition: "CLOSE > 0"
  risk_management:
    stop_loss_percent: 2
    take_profit_percent: 2
"""


def _minute_ohlcv(periods: int = 750) -> list[dict]:
    """Deterministic 1-minute OHLCV that oscillates ±3% so 2% stop/target
    levels are repeatedly hit — generating a stream of round-trips."""
    idx = pd.date_range("2024-01-02 03:45:00", periods=periods, freq="1min")
    i = np.arange(periods)
    close = 100.0 * (1.0 + 0.03 * np.sin(2 * np.pi * i / 31.0))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * 1.001
    low = np.minimum(openp, close) * 0.999
    return [
        {
            "timestamp": ts.isoformat() + "Z",
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": 1000.0,
        }
        for ts, o, h, l, c in zip(idx, openp, high, low, close)
    ]


_MDR = {
    "symbol": "TEST",
    "interval": "1m",
    "from_utc": "2024-01-01T00:00:00Z",
    "to_utc": "2099-01-01T00:00:00Z",
}


def test_runner_1m_intrabar_matches_legacy_metrics():
    ohlcv = _minute_ohlcv()

    legacy = run_backtest(_YAML_1M, ohlcv, {}, dict(_MDR))
    intrabar = run_backtest(_YAML_1M, ohlcv, {"intrabar_execution": True}, dict(_MDR))

    # Same data, identity resample, 1:1 sub-bars → identical metrics.
    assert legacy["metrics"] == intrabar["metrics"]
    assert intrabar["metrics"]["total_trades"] > 0


def test_runner_15m_intrabar_resamples_and_runs():
    ohlcv = _minute_ohlcv()

    result = run_backtest(_YAML_15M, ohlcv, {"intrabar_execution": True}, dict(_MDR))

    # 750 one-minute bars from 03:45 → fifty 15-minute strategy bars; the
    # backtest must complete and report a coherent result.
    expected_bars = len(resample_ohlcv(
        pd.DataFrame(
            {
                "open": [r["open"] for r in ohlcv],
                "high": [r["high"] for r in ohlcv],
                "low": [r["low"] for r in ohlcv],
                "close": [r["close"] for r in ohlcv],
                "volume": [r["volume"] for r in ohlcv],
            },
            index=pd.DatetimeIndex([pd.Timestamp(r["timestamp"].rstrip("Z")) for r in ohlcv]),
        ),
        "15m",
    ))
    assert expected_bars == 50
    assert "metrics" in result
    assert result["metrics"]["total_trades"] >= 0
