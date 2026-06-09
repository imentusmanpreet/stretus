"""Tests for the SL/TP resolver and user RR parsing."""
from __future__ import annotations

import pandas as pd
import pytest

from app.kb.schemas import RiskTier
from app.planner.param_resolver import parse_risk_reward, resolve_sl_tp


_TIER = RiskTier(
    per_trade_risk_pct=2.0,
    daily_loss_cap_pct=2.0,
    max_open_positions=3,
    base_sl_pct=1.5,
    base_tp_pct=3.0,
)


# ── parse_risk_reward ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1:2", 2.0),
        ("1:3", 3.0),
        # "N:1" means N reward per 1 risk (trader convention), not 1/N.
        ("2:1", 2.0),
        ("3:1", 3.0),
        ("2", 2.0),
        ("2.5", 2.5),
        (3, 3.0),
        (None, None),
        ("", None),
        ("   ", None),
        ("garbage", None),
        ("0:5", None),
        ("1:0", None),
        ("-1:2", None),
    ],
)
def test_parse_risk_reward(raw, expected):
    assert parse_risk_reward(raw) == expected


# ── resolve_sl_tp ─────────────────────────────────────────────────────────────


def test_resolve_sl_tp_no_ohlcv_no_user_rr_uses_baseline():
    sl, tp, mult = resolve_sl_tp(_TIER)
    assert sl == 1.5
    assert tp == 3.0
    assert mult == 1.0


def test_resolve_sl_tp_no_ohlcv_with_user_rr_overrides_tp():
    sl, tp, mult = resolve_sl_tp(_TIER, user_risk_reward="1:2")
    assert sl == 1.5
    assert tp == 3.0      # 1.5 × 2
    assert mult == 1.0


def test_resolve_sl_tp_no_ohlcv_with_user_rr_3_to_1():
    sl, tp, _ = resolve_sl_tp(_TIER, user_risk_reward="1:3")
    assert tp == 4.5      # 1.5 × 3 — overrides baseline 3.0


@pytest.mark.parametrize("rr", ["0.36", "0.5", "1:0.4", 0.3])
def test_resolve_sl_tp_never_returns_tp_below_sl(rr):
    """A sub-1 risk:reward must NOT produce tp < sl. Previously this slipped
    through and crashed the planner's validate_plan (tp_pct < sl_pct). The
    resolver now floors TP at the SL (a 1:1 minimum)."""
    sl, tp, _ = resolve_sl_tp(_TIER, user_risk_reward=rr)
    assert tp >= sl, f"tp ({tp}) must be >= sl ({sl}) for rr={rr!r}"


def test_resolve_sl_tp_floor_does_not_touch_valid_rr():
    """A normal RR >= 1 is unaffected by the floor."""
    sl, tp, _ = resolve_sl_tp(_TIER, user_risk_reward="1:2")
    assert tp == round(sl * 2.0, 2)


def test_resolve_sl_tp_with_ohlcv_scales_sl_then_applies_user_rr():
    """High-vol OHLCV should widen SL; user RR sets TP = SL × ratio."""
    # Build a fake high-volatility OHLCV: ~3% ATR
    rows = []
    base = 100.0
    for i in range(60):
        bar = base + (i % 5) * 3.0      # bars swing 3 rupees on a ~100-rupee stock
        rows.append({
            "high": bar + 1.5,
            "low": bar - 1.5,
            "close": bar,
            "open": bar,
            "volume": 1000,
        })
    df = pd.DataFrame(rows)
    sl, tp, mult = resolve_sl_tp(_TIER, ohlcv=df, user_risk_reward="1:2")
    assert mult > 1.0                    # high vol → multiplier above baseline
    assert sl > 1.5                      # SL widened
    assert tp == round(sl * 2.0, 2)      # TP honors user RR exactly
