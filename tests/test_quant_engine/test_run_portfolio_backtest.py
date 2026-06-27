"""Phase B — end-to-end portfolio backtest (§8): wrap per-member sim under one pool.

Drives the real engine over two synthetic members and asserts the portfolio overlay:
one equity curve, members listed, membership-window filtering, and the static-path
no-behaviour-change guard (return_trades is additive and off by default).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from engine.runner import run_backtest, run_portfolio_backtest

_FROM = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _daily_ohlcv(n: int = 160, *, amp: float = 8.0, base: float = 100.0) -> list[dict]:
    """n daily bars with a gentle oscillation so a CLOSE-vs-EMA strategy trades."""
    import math
    recs = []
    for i in range(n):
        ts = _FROM + timedelta(days=i)
        close = base + amp * math.sin(i / 6.0) + i * 0.05
        recs.append({
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 100000 + (i % 7) * 1000,
        })
    return recs


def _template_yaml() -> str:
    return yaml.safe_dump({
        "strategy": {
            "name": "dyn", "symbol": "PLACEHOLDER", "market": "indian_stocks",
            "timeframe": "1d", "objective": "positional", "direction": "long_only",
            "entry": {"condition": "CLOSE > EMA(10)"},
            "exit": {"condition": "CLOSE < EMA(10)"},
            "risk_management": {"stop_loss_percent": 3, "take_profit_percent": 6},
            "entry_evaluation_mode": "formula", "exit_evaluation_mode": "formula",
        },
        "universe": {"source": {"kind": "watchlist", "symbols": ["AAA", "BBB"]},
                     "rank": {"by": "rvol"}, "take": 2},
    }, sort_keys=False)


def _market_request() -> dict:
    return {"symbol": "PLACEHOLDER", "interval": "1d",
            "from_utc": _FROM.isoformat().replace("+00:00", "Z"),
            "to_utc": (_FROM + timedelta(days=200)).isoformat().replace("+00:00", "Z")}


def _run_config() -> dict:
    return {"starting_balance": 100000.0, "objective": "positional"}


def test_portfolio_backtest_produces_single_result():
    member_ohlcv = {"AAA": _daily_ohlcv(amp=8.0), "BBB": _daily_ohlcv(amp=5.0, base=200.0)}
    out = run_portfolio_backtest(
        _template_yaml(), member_ohlcv, _run_config(), _market_request(),
        starting_capital=100000.0, max_positions=2,
    )
    assert out["members"] == ["AAA", "BBB"]
    assert out["starting_capital"] == 100000.0
    assert "equity_curve" in out and "metrics" in out
    assert out["survivorship_mode"] == "approximate"
    # Per-symbol attribution keys are a subset of the members.
    assert set(out["per_symbol_pnl"]).issubset({"AAA", "BBB"})


def test_membership_window_filters_out_of_window_entries():
    member_ohlcv = {"AAA": _daily_ohlcv()}
    # A window covering only the first 5 days → almost all entries filtered out.
    narrow = {"AAA": [(_FROM.isoformat().replace("+00:00", "Z"),
                       (_FROM + timedelta(days=5)).isoformat().replace("+00:00", "Z"))]}
    wide = {"AAA": [(_FROM.isoformat().replace("+00:00", "Z"),
                     (_FROM + timedelta(days=200)).isoformat().replace("+00:00", "Z"))]}
    out_narrow = run_portfolio_backtest(
        _template_yaml(), member_ohlcv, _run_config(), _market_request(),
        membership_windows=narrow, max_positions=2)
    out_wide = run_portfolio_backtest(
        _template_yaml(), member_ohlcv, _run_config(), _market_request(),
        membership_windows=wide, max_positions=2)
    assert out_narrow["metrics"]["total_trades"] <= out_wide["metrics"]["total_trades"]


def test_static_run_backtest_unchanged_without_return_trades():
    """Invariant 2/10: the additive return_trades hook is off by default — a normal
    single-symbol run never gains a 'trades' key."""
    result = run_backtest(
        yaml.safe_dump({"strategy": {
            "name": "s", "symbol": "AAA", "market": "indian_stocks", "timeframe": "1d",
            "objective": "positional", "direction": "long_only",
            "entry": {"condition": "CLOSE > EMA(10)"}, "exit": {"condition": "CLOSE < EMA(10)"},
            "risk_management": {"stop_loss_percent": 3, "take_profit_percent": 6},
            "entry_evaluation_mode": "formula", "exit_evaluation_mode": "formula",
        }}, sort_keys=False),
        _daily_ohlcv(), _run_config(), {"symbol": "AAA", "interval": "1d",
        "from_utc": _market_request()["from_utc"], "to_utc": _market_request()["to_utc"]},
    )
    assert "trades" not in result
    assert "metrics" in result
