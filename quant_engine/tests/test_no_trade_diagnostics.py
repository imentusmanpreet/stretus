"""
Regression tests for the no-trade diagnostics.

The engine used to blame "circuit-breakers (daily cap / max trades)" for ANY
case where entries fired but no trade filled — even when the real cause was
something else entirely. The most common real cause: an intraday-objective
strategy running on a timeframe that spans a whole session (e.g. 1d), where
every bar is its session's last bar so every entry is rejected. These tests
pin the truthful behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.metrics import _build_no_trade_hints
from engine.simulator import simulate_trades


def _daily_frame(days: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=days, freq="1D")
    j = np.arange(days, dtype=float)
    return pd.DataFrame(
        {
            "open":  100.0 + j,
            "high":  101.0 + j,
            "low":   99.0 + j,
            "close": 100.5 + j,
            "volume": 1000.0,
        },
        index=idx,
    )


def test_intraday_objective_on_daily_bars_records_session_last_block():
    df = _daily_frame(30)
    trades, diagnostics = simulate_trades(
        df=df,
        symbol="TEST",
        entry_condition="CLOSE > 0",   # always true → fires every bar
        exit_condition="",
        stop_loss_pct=2.0,
        take_profit_pct=2.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        warm_up_candles=0,
        objective="intraday",          # the contradiction: intraday on daily bars
    )

    # Every entry is rejected because each daily bar ends its own session.
    assert trades == []
    session_blocks = sum(1 for d in diagnostics if d.get("entry_blocked_session_last"))
    assert session_blocks > 0
    # And it was NOT the daily-cap / max-trades gates (the old misleading guess).
    assert all(not d.get("entry_blocked_daily_cap") for d in diagnostics)
    assert all(not d.get("entry_blocked_max_trades") for d in diagnostics)


def test_positional_objective_on_daily_bars_does_fill():
    # Same data, positional objective → the same-session constraint doesn't
    # apply, so entries fill and we get trades. Confirms the block is specific
    # to the intraday/timeframe mismatch, not the daily bars themselves.
    df = _daily_frame(30)
    trades, _ = simulate_trades(
        df=df,
        symbol="TEST",
        entry_condition="CLOSE > 0",
        exit_condition="",
        stop_loss_pct=2.0,
        take_profit_pct=2.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        warm_up_candles=0,
        objective="positional",
    )
    assert len(trades) > 0


def test_no_trade_hint_names_the_dominant_real_blocker():
    # Synthetic diagnostics: entries fired on every bar, all blocked by the
    # session-last constraint. The hint must name that cause, not circuit-breakers.
    diagnostics = [
        {"entry_evaluated": True, "entry_signal": True, "entry_blocked_session_last": True}
        for _ in range(50)
    ]
    hint = _build_no_trade_hints(diagnostics, pd.DataFrame())
    assert "objective is intraday" in hint
    assert "positional" in hint
    assert "daily cap" not in hint   # the old hardcoded guess is gone


def test_no_trade_hint_reports_confirmation_gate_when_that_is_the_cause():
    diagnostics = [
        {"entry_evaluated": True, "entry_signal": True, "entry_blocked_confirmation": True}
        for _ in range(10)
    ]
    hint = _build_no_trade_hints(diagnostics, pd.DataFrame())
    assert "confirmation" in hint
