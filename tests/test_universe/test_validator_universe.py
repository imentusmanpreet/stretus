"""
Validator tests for the dynamic-universe block (§7.1 cap notes, screen compile, source checks).

These need the engine grammar (compile_condition); conftest puts quant_engine on the path.
If the engine is unavailable the validator fails closed on every spec, so we skip then.
"""
from __future__ import annotations

import pytest

from app.strategy import engine_bridge
from app.strategy.spec import StrategySpec
from app.strategy.validator import validate_spec

pytestmark = pytest.mark.skipif(
    not engine_bridge.is_available(), reason="engine grammar unavailable"
)


def _spec(universe: dict) -> StrategySpec:
    return StrategySpec(
        name="dyn", market="indian_stocks", timeframe="5m", objective="intraday",
        direction="long_only", entry_condition="CLOSE > VWAP",
        stop_loss={"type": "percent", "value": 1.0, "source": "user"},
        take_profit={"type": "percent", "value": 2.0, "source": "user"},
        universe=universe,
    )


def test_valid_universe_passes_with_cap_note():
    s = _spec({"source": {"kind": "index", "name": "NIFTY500"},
               "rank": {"by": "rvol"}, "take": 1000, "max_positions": 50,
               "screen": ["CLOSE > VWAP", "RSI(14) > 50"]})
    r = validate_spec(s)
    assert r.ok  # cap is informational, not blocking
    codes = {n.code for n in r.notes}
    assert "take_above_platform_cap" in codes
    assert "max_positions_above_platform_cap" in codes


def test_bad_screen_condition_is_blocking_error():
    s = _spec({"source": {"kind": "index", "name": "NIFTY500"},
               "rank": {"by": "rvol"}, "take": 5, "screen": ["BOGUSFN(3) > 1"]})
    r = validate_spec(s)
    assert not r.ok
    assert any(e.field.startswith("universe.screen") for e in r.errors)


def test_empty_watchlist_is_blocking_error():
    s = _spec({"source": {"kind": "watchlist", "symbols": []},
               "rank": {"by": "rvol"}, "take": 5, "screen": []})
    r = validate_spec(s)
    assert not r.ok
    assert any(e.code == "empty_watchlist" for e in r.errors)


def test_unwired_rank_metric_notes_feed():
    s = _spec({"source": {"kind": "f_and_o"}, "rank": {"by": "oi_change"},
               "take": 5, "screen": []})
    r = validate_spec(s)
    assert any(n.code == "rank_metric_needs_feed" for n in r.notes)
