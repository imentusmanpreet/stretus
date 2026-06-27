"""Phase D — the live Portfolio Manager (§9/§11): shared capital + position cap, live.

Stateful admission control. Pure (no DB/broker), so we test it directly: the cap, capital
sharing, sector cap, already-open guard, release/compounding, and reconciliation rebuild.
"""
from __future__ import annotations

import pytest

from app.services.execution.portfolio_manager import LivePortfolioManager, LivePosition


def _pm(**kw) -> LivePortfolioManager:
    base = dict(total_capital=100_000.0, max_positions=2)
    base.update(kw)
    return LivePortfolioManager(**base)


def test_admits_until_cap_then_position_cap():
    pm = _pm(max_positions=2)
    d1 = pm.evaluate_entry("A")
    assert d1.admit and d1.allocated_capital == 50_000  # equal weight: 100k/2
    pm.admit("A", d1.allocated_capital)
    d2 = pm.evaluate_entry("B")
    pm.admit("B", d2.allocated_capital)
    # third entry blocked by the cap
    d3 = pm.evaluate_entry("C")
    assert d3.admit is False and d3.reason == "position_cap"


def test_capital_is_shared():
    pm = _pm(max_positions=2)
    pm.admit("A", pm.evaluate_entry("A").allocated_capital)
    pm.admit("B", pm.evaluate_entry("B").allocated_capital)
    assert pm.free_cash == pytest.approx(0.0)      # 50k + 50k locked
    assert pm.equity == pytest.approx(100_000.0)
    assert pm.open_count == 2


def test_release_returns_capital_and_pnl_then_compounds():
    pm = _pm(max_positions=1)
    pm.admit("A", pm.evaluate_entry("A").allocated_capital)   # 100k in
    pm.release("A", realized_pnl=10_000.0)                    # +10% → pot 110k
    assert pm.equity == pytest.approx(110_000.0)
    d = pm.evaluate_entry("B")
    assert d.allocated_capital == pytest.approx(110_000.0)    # next trade uses grown pot


def test_already_open_is_not_readmitted():
    pm = _pm(max_positions=3)
    pm.admit("A", pm.evaluate_entry("A").allocated_capital)
    again = pm.evaluate_entry("A")
    assert again.admit is False and again.reason == "already_open"


def test_risk_based_sizing_uses_stop_distance():
    pm = _pm(max_positions=5, sizing_mode="risk_based", risk_per_trade_pct=1.0)
    d = pm.evaluate_entry("A", stop_distance_frac=0.02)       # 1% of 100k / 2% = 50k
    assert d.allocated_capital == pytest.approx(50_000.0)


def test_sector_cap_blocks_overconcentration():
    pm = _pm(max_positions=5, sector_cap_pct=30.0)
    d1 = pm.evaluate_entry("HDFCBANK", sector="banking")     # 100k/5 = 20k = 20%
    pm.admit("HDFCBANK", d1.allocated_capital, sector="banking")
    d2 = pm.evaluate_entry("ICICIBANK", sector="banking")    # would be 40% > 30%
    assert d2.admit is False and d2.reason == "sector_cap"


def test_insufficient_capital_skips():
    pm = _pm(max_positions=10)
    # consume nearly all cash via a big risk-based allocation, then a tiny pot remains
    pm.admit("A", 99_999.0)
    d = pm.evaluate_entry("B")
    assert d.admit is True  # tiny slice still fundable
    pm.admit("B", pm.free_cash)  # drain to zero
    d2 = pm.evaluate_entry("C")
    assert d2.admit is False and d2.reason == "insufficient_capital"


def test_rebuild_restores_open_positions_and_cash():
    pm = _pm(max_positions=3, total_capital=100_000.0)
    pm.rebuild([LivePosition("A", 30_000.0), LivePosition("B", 20_000.0)])
    assert pm.open_count == 2
    assert pm.free_cash == pytest.approx(50_000.0)   # 100k - 30k - 20k
    assert set(pm.open_symbols) == {"A", "B"}
