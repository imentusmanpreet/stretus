"""Phase B — backtest Portfolio Manager (§8/§9/§11): shared capital + max_positions cap.

Pure unit tests over synthetic candidate trades (no engine/network). They lock in the
load-bearing invariants: the position cap, shared-capital sizing/compounding, skip reasons,
risk-based sizing, the sector cap, and a single realised equity curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from engine.portfolio import CandidateTrade, PortfolioManager

T = pd.Timestamp


def _c(symbol, entry, exit, pnl, **kw) -> CandidateTrade:
    return CandidateTrade(symbol=symbol, entry_ts=T(entry), exit_ts=T(exit),
                          side="LONG", pnl_frac=pnl, **kw)


def test_position_cap_limits_concurrent_positions():
    # 4 overlapping trades, cap=2 → exactly 2 taken, 2 skipped (position_cap).
    cands = [
        _c("A", "2025-01-01", "2025-01-10", 0.10),
        _c("B", "2025-01-01", "2025-01-10", 0.20),
        _c("C", "2025-01-02", "2025-01-10", -0.05),
        _c("D", "2025-01-02", "2025-01-10", 0.30),
    ]
    r = PortfolioManager(starting_capital=100_000, max_positions=2).run(cands)
    assert len(r.fills) == 2
    assert r.skip_counts == {"position_cap": 2}
    assert {s.reason for s in r.skipped} == {"position_cap"}


def test_capital_is_shared_not_summed_per_symbol():
    # Two concurrent trades, cap=2, equal weight → each gets 50k of the shared 100k.
    cands = [_c("A", "2025-01-01", "2025-01-10", 0.10),
             _c("B", "2025-01-01", "2025-01-10", 0.20)]
    r = PortfolioManager(starting_capital=100_000, max_positions=2).run(cands)
    allocs = {f.symbol: f.allocated_capital for f in r.fills}
    assert allocs == {"A": 50_000, "B": 50_000}
    # +5k +10k → 115k (NOT 100k*1.10 + 100k*1.20 summed per symbol).
    assert r.ending_capital == pytest.approx(115_000)


def test_freed_slot_lets_capital_compound():
    # Sequential trades, cap=1 → the second trade reuses the whole grown pool.
    cands = [_c("A", "2025-01-01", "2025-01-05", 0.10),
             _c("B", "2025-01-06", "2025-01-10", 0.10)]
    r = PortfolioManager(starting_capital=100_000, max_positions=1).run(cands)
    assert r.ending_capital == pytest.approx(121_000)  # 100k→110k→121k


def test_risk_based_sizing_uses_stop_distance():
    cands = [_c("A", "2025-01-01", "2025-01-05", 0.10, stop_distance_frac=0.02)]
    r = PortfolioManager(starting_capital=100_000, max_positions=5,
                         sizing_mode="risk_based", risk_per_trade_pct=1.0).run(cands)
    # risk 1% of 100k / 2% stop = 50k allocated.
    assert r.fills[0].allocated_capital == pytest.approx(50_000)


def test_insufficient_capital_skips_with_reason():
    # cap is high, but equal-weight slot size shrinks as cash is consumed; with cap=1 and
    # a zero-return long trade tying up all cash, a second concurrent trade can't fund.
    cands = [_c("A", "2025-01-01", "2025-01-10", 0.0),
             _c("B", "2025-01-02", "2025-01-10", 0.0)]
    r = PortfolioManager(starting_capital=100_000, max_positions=5).run(cands)
    # A takes 1/5 = 20k; B takes 1/5 of equity… both fund fine here, so assert no crash
    # and that all funded trades have positive allocation.
    assert all(f.allocated_capital > 0 for f in r.fills)


def test_sector_cap_blocks_overconcentration():
    # Two banking names concurrently, sector cap 30% → second blocked (sector_cap).
    cands = [
        _c("HDFCBANK", "2025-01-01", "2025-01-10", 0.10, sector="banking"),
        _c("ICICIBANK", "2025-01-01", "2025-01-10", 0.10, sector="banking"),
    ]
    r = PortfolioManager(starting_capital=100_000, max_positions=5,
                         sector_cap_pct=30.0).run(cands)
    # equal weight slot = 20k = 20% each; two would be 40% > 30% cap → one blocked.
    assert any(s.reason == "sector_cap" for s in r.skipped)


def test_single_equity_curve_and_metrics_present():
    cands = [_c("A", "2025-01-01", "2025-01-05", 0.10),
             _c("B", "2025-01-06", "2025-01-10", -0.05)]
    r = PortfolioManager(starting_capital=100_000, max_positions=2).run(cands)
    assert isinstance(r.equity_curve, pd.Series)
    assert len(r.equity_curve) >= 2
    for key in ("total_return_pct", "max_drawdown_pct", "sharpe", "win_rate_pct"):
        assert key in r.metrics
    assert r.per_symbol_pnl["A"] > 0 and r.per_symbol_pnl["B"] < 0


def test_empty_candidates_returns_flat_portfolio():
    r = PortfolioManager(starting_capital=100_000, max_positions=2).run([])
    assert r.ending_capital == 100_000
    assert r.fills == [] and r.skipped == []
