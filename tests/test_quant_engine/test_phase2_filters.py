"""
Phase 2 — regime, event-date, and relative-strength entry gates.

Each gate is proven to BLOCK when it should and PASS when it should, in BOTH
the backtest simulator and the live evaluator (entry_gates.py). Both sides share
the same logic (regime → classify_regime; RS → same ratio formula), so a backtest
matches live execution — the Phase-2 parity contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quant_engine"))

from engine.regime import classify_regime_series
from engine.simulator import simulate_trades


def _df(n: int = 80, *, ref: str = "flat") -> pd.DataFrame:
    """Steadily up-drifting symbol. REF_close is flat (ref='flat') so the symbol
    outperforms, or up-faster (ref='strong') so it underperforms."""
    idx = pd.date_range("2026-01-05 03:45", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100.0 + np.arange(n) * 0.05, index=idx)
    if ref == "flat":
        ref_close = np.full(n, 100.0)
    else:  # reference rises twice as fast → symbol underperforms
        ref_close = 100.0 + np.arange(n) * 0.10
    return pd.DataFrame(
        {
            "open": close.values, "high": (close + 1).values, "low": (close - 1).values,
            "close": close.values, "volume": [1000.0] * n, "REF_close": ref_close,
        },
        index=idx,
    )


_COMMON = dict(
    symbol="T.NS", entry_condition="CLOSE > 0", exit_condition="",
    stop_loss_pct=5.0, take_profit_pct=10.0, slippage_bps=0.0, commission_bps=0.0,
    warm_up_candles=14, max_holding_candles=5, objective="intraday",
)

_ALL_REGIMES = ("trending_up", "trending_down", "ranging", "volatile")


def _live(df, **gate_kwargs):
    from app.schemas.execution import AssetClass, GatesConfig
    from app.services.execution.entry_gates import evaluate_entry_gates

    class _NullRule:
        def evaluate_entry(self, *a, **k):
            return True

    return evaluate_entry_gates(
        df=df, gates=GatesConfig(**gate_kwargs), exec_state=None, side="BUY",
        rule_engine=_NullRule(), entry_block={"trigger": {}, "filters": []},
        asset_class=AssetClass.crypto_spot,  # disable the session window for isolation
    )


# ── Regime gate ─────────────────────────────────────────────────────────────────

def test_regime_series_is_causal():
    """value[i] must not change when later bars are truncated away."""
    df = _df()
    full = classify_regime_series(df)
    for i in (40, 60, 79):
        trunc = classify_regime_series(df.iloc[: i + 1])
        assert full.iloc[i] == trunc.iloc[i], f"regime@{i} leaked future data"


def test_regime_gate_backtest_block_and_pass():
    df = _df()
    present = set(classify_regime_series(df).iloc[14:].unique())
    assert "volatile" not in present  # this steady df is never volatile
    blk, diags = simulate_trades(df=df, **_COMMON, regime_filter_allowed=("volatile",))
    assert blk == [] and any(d["entry_blocked_regime"] for d in diags)
    pss, diags = simulate_trades(df=df, **_COMMON, regime_filter_allowed=_ALL_REGIMES)
    assert len(pss) >= 1 and not any(d["entry_blocked_regime"] for d in diags)


def test_regime_gate_live_block_and_pass():
    df = _df()
    assert _live(df, regime_filter_allowed=["volatile"]).blocked_by == "regime"
    assert _live(df, regime_filter_allowed=list(_ALL_REGIMES)).passed is True


# ── Event-date gate ──────────────────────────────────────────────────────────────

def test_event_gate_backtest_block_and_pass():
    df = _df()
    all_dates = tuple(sorted({str(t)[:10] for t in df.index}))
    blk, diags = simulate_trades(df=df, **_COMMON, event_skip_dates=all_dates)
    assert blk == [] and any(d["entry_blocked_event"] for d in diags)
    pss, _ = simulate_trades(df=df, **_COMMON, event_skip_dates=("1999-01-01",))
    assert len(pss) >= 1


def test_event_gate_live_block_and_pass():
    df = _df()
    last_date = str(df.index[-1])[:10]
    assert _live(df, event_skip_dates=[last_date]).blocked_by == "event"
    assert _live(df, event_skip_dates=["1999-01-01"]).passed is True


# ── Relative-strength gate ───────────────────────────────────────────────────────

def test_rs_gate_backtest_block_and_pass():
    df = _df(ref="flat")  # symbol up vs flat ref → RS > 1
    pss, _ = simulate_trades(df=df, **_COMMON, rs_filter_window=10, rs_filter_min_ratio=1.0)
    assert len(pss) >= 1
    blk, diags = simulate_trades(df=df, **_COMMON, rs_filter_window=10, rs_filter_min_ratio=1.5)
    assert blk == [] and any(d["entry_blocked_relative_strength"] for d in diags)


def test_rs_gate_blocks_genuine_underperformer():
    df = _df(ref="strong")  # reference rises faster → symbol underperforms → RS < 1
    blk, diags = simulate_trades(df=df, **_COMMON, rs_filter_window=10, rs_filter_min_ratio=1.0)
    assert blk == [] and any(d["entry_blocked_relative_strength"] for d in diags)


def test_rs_gate_live_block_and_pass():
    df = _df(ref="flat")
    assert _live(df, rs_filter_window=10, rs_filter_min_ratio=1.0).passed is True
    assert _live(df, rs_filter_window=10, rs_filter_min_ratio=1.5).blocked_by == "relative_strength"


# ── Backtest ↔ live parity (decisive) ────────────────────────────────────────────

@pytest.mark.parametrize("gate_kwargs,should_block", [
    (dict(regime_filter_allowed=("volatile",)), True),
    (dict(regime_filter_allowed=_ALL_REGIMES), False),
    (dict(rs_filter_window=10, rs_filter_min_ratio=1.5), True),
    (dict(rs_filter_window=10, rs_filter_min_ratio=1.0), False),
])
def test_backtest_live_parity(gate_kwargs, should_block):
    df = _df(ref="flat")
    trades, _ = simulate_trades(df=df, **_COMMON, **gate_kwargs)
    # Live takes list (not tuple) for regime; normalise.
    live_kwargs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gate_kwargs.items()}
    live = _live(df, **live_kwargs)
    assert (trades == []) == (not live.passed) == should_block


# ── Lunch-lull gate ──────────────────────────────────────────────────────────────

def test_lunch_lull_backtest_block_and_pass():
    # Frame spans 03:45–~23:30 UTC; pick a lull window the bars actually hit.
    df = _df(n=80)
    mins = sorted({int(t.hour) * 60 + int(t.minute) for t in df.index})
    lull_lo, lull_hi = mins[0], mins[-1]  # whole range → all entries blocked
    blk, diags = simulate_trades(
        df=df, **_COMMON, lunch_lull_start_utc=lull_lo, lunch_lull_end_utc=lull_hi,
    )
    assert blk == [] and any(d["entry_blocked_lunch_lull"] for d in diags)
    # A 1-minute window no bar lands on → never blocks.
    pss, _ = simulate_trades(
        df=df, **_COMMON, lunch_lull_start_utc=1, lunch_lull_end_utc=1,
    )
    assert len(pss) >= 1


def test_lunch_lull_live_block_and_pass():
    df = _df(n=80)
    ts = df.index[-1]
    last_min = int(ts.hour) * 60 + int(ts.minute)
    assert _live(df, lunch_lull_start_utc=last_min, lunch_lull_end_utc=last_min).blocked_by == "lunch_lull"
    assert _live(df, lunch_lull_start_utc=1, lunch_lull_end_utc=1).passed is True


# ── Loader wiring ────────────────────────────────────────────────────────────────

def test_loader_parses_phase2_filter_blocks():
    from engine.loader import load_strategy_from_content
    cfg = load_strategy_from_content(
        """
strategy:
  name: Phase2 Filters
  symbol: T.NS
  market: NSE
  timeframe: 15m
  entry: { condition: "CLOSE > 0" }
  exit: { condition: "" }
  risk_management: { stop_loss_percent: 1.5, take_profit_percent: 3.0 }
  regime_filter: { allowed: [trending_up, trending_down] }
  relative_strength_filter: { window: 20, min_ratio: 1.1 }
  event_filter: { skip_dates: ["2026-01-31", "2026-02-27"] }
  lunch_lull: { start: "12:00", end: "13:00", timezone: "Asia/Kolkata" }
"""
    )
    assert cfg.regime_filter_allowed == ("trending_up", "trending_down")
    assert cfg.rs_filter_window == 20 and cfg.rs_filter_min_ratio == 1.1
    assert cfg.event_skip_dates == ("2026-01-31", "2026-02-27")
    # 12:00 IST = 06:30 UTC = 390 min; 13:00 IST = 07:30 UTC = 450 min.
    assert cfg.lunch_lull_start_utc == 390 and cfg.lunch_lull_end_utc == 450
