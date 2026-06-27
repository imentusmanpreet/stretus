"""
Multi-TP ladder end-to-end tests for simulate_trades.

IMPORTANT entry-price convention:
  Entry fires when entry_condition evaluates True on bar N.
  The actual FILL happens at bar N+1's OPEN.
  So rung prices are computed from bar(N+1).open, not bar(N).open.

We use bar(signal).close == bar(entry).open == 100.0 so the math is clean.
"""
from __future__ import annotations
import pandas as pd
from engine.simulator import simulate_trades
from engine.loader import _parse_multi_take_profit_spec


def _frame(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low":  [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [1e6] * len(rows)},
        index=idx,
    )


_BASE = dict(
    symbol="TEST",
    exit_condition="",
    slippage_bps=0.0,
    commission_bps=0.0,
    warm_up_candles=0,
    objective="positional",
    stt_delivery_pct=0.0,
    stt_intraday_sell_pct=0.0,
)


# ── TEST 1: all 3 rungs hit → MULTI_TP_COMPLETE ───────────────────────────────
def test_multi_tp_complete():
    """
    bar 0: signal fires (CLOSE > 0)
    bar 1: ENTRY at open=100.0  SL=99.0  TP1=101  TP2=102  TP3=104
           high=101.1 → TP1 fires (@101), SL→100 (breakeven)
    bar 2: high=102.2 → TP2 fires (@102), SL→101 (prev_tp)
    bar 3: high=104.5 → TP3 fires (@104) → MULTI_TP_COMPLETE
    blended pnl = 0.30*(1/100) + 0.30*(2/100) + 0.40*(4/100) = 2.5%
    """
    # low of each bar must stay above the new SL after TP fires on that bar,
    # otherwise the same-bar collision rule triggers STOP_LOSS instead.
    # After TP1: SL → 100.0 (breakeven). Bar1 low=100.1 > 100.0 ✓
    # After TP2: SL → 101.0 (prev_tp=TP1 fill). Bar2 low=101.1 > 101.0 ✓
    df = _frame([
        ("2024-01-02 09:15", 100.0, 100.5, 99.8, 100.0),  # bar 0: signal
        ("2024-01-02 09:30", 100.0, 101.1, 100.1, 100.9),  # bar 1: entry + TP1 hit
        ("2024-01-02 09:45", 100.9, 102.2, 101.1, 101.8),  # bar 2: TP2 hit
        ("2024-01-02 10:00", 101.8, 104.5, 101.5, 103.5),  # bar 3: TP3 hit
    ])
    rungs = _parse_multi_take_profit_spec([
        {"type": "risk_reward", "value": 1.0, "exit_fraction": 0.30, "risk_action": "move_sl_to_breakeven"},
        {"type": "risk_reward", "value": 2.0, "exit_fraction": 0.30, "risk_action": "move_sl_to_previous_tp"},
        {"type": "risk_reward", "value": 4.0, "exit_fraction": 0.40, "risk_action": "none"},
    ])
    trades, _ = simulate_trades(
        df=df, entry_condition="CLOSE > 0",
        stop_loss_pct=1.0, take_profit_pct=0.0,
        take_profit_targets=rungs, **_BASE,
    )
    assert len(trades) >= 1, f"Expected ≥1 trade, got {len(trades)}"
    t = trades[0]
    print(f"  exit={t.exit_reason}  pnl={t.pnl_pct:.4f}%  partial_exits={len(t.partial_exits)}")
    for pe in t.partial_exits:
        print(f"    rung={pe.get('rung_index')} frac={pe.get('exit_fraction')} fill={pe.get('fill_price')} action={pe.get('risk_action')}")
    assert t.exit_reason == "MULTI_TP_COMPLETE", f"Expected MULTI_TP_COMPLETE, got {t.exit_reason}"
    assert len(t.partial_exits) == 3, f"Expected 3 partial exits, got {len(t.partial_exits)}"
    expected_pnl = 2.5
    assert abs(t.pnl_pct - expected_pnl) < 0.1, f"pnl_pct={t.pnl_pct:.4f} expected≈{expected_pnl}"
    assert t.partial_exits[0].get("risk_action") == "move_sl_to_breakeven"
    assert t.partial_exits[1].get("risk_action") == "move_sl_to_previous_tp"


# ── TEST 2: SL fires after TP1 moved SL to breakeven ─────────────────────────
def test_sl_fires_after_tp1_moves_sl():
    """
    bar 0: signal
    bar 1: ENTRY at 100.0  SL=99.0  TP1=101, high=101.1 → TP1 fires, SL→100 (breakeven)
    bar 2: low=99.5 which is < new SL=100 → STOP_LOSS at 100
    blended pnl = 0.30*(1/100) + 0.70*(0/100) = 0.3%
    """
    df = _frame([
        ("2024-01-02 09:15", 100.0, 100.5, 99.8, 100.0),  # bar 0: signal
        ("2024-01-02 09:30", 100.0, 101.1, 100.0, 100.9),  # bar 1: entry + TP1 fires
        ("2024-01-02 09:45", 100.9, 101.0, 99.5, 100.4),   # bar 2: low < breakeven SL
    ])
    rungs = _parse_multi_take_profit_spec([
        {"type": "risk_reward", "value": 1.0, "exit_fraction": 0.30, "risk_action": "move_sl_to_breakeven"},
        {"type": "risk_reward", "value": 2.0, "exit_fraction": 0.30, "risk_action": "none"},
        {"type": "risk_reward", "value": 4.0, "exit_fraction": 0.40, "risk_action": "none"},
    ])
    trades, _ = simulate_trades(
        df=df, entry_condition="CLOSE > 0",
        stop_loss_pct=1.0, take_profit_pct=0.0,
        take_profit_targets=rungs, **_BASE,
    )
    assert len(trades) >= 1
    t = trades[0]
    print(f"  exit={t.exit_reason}  pnl={t.pnl_pct:.4f}%  partial_exits={len(t.partial_exits)}")
    assert t.exit_reason == "STOP_LOSS", f"Expected STOP_LOSS, got {t.exit_reason}"
    assert len(t.partial_exits) == 1, f"Expected 1 partial exit (TP1 only), got {len(t.partial_exits)}"
    expected_pnl = 0.3   # 30% at +1R, 70% at 0 (breakeven)
    assert abs(t.pnl_pct - expected_pnl) < 0.15, f"pnl_pct={t.pnl_pct:.4f} expected≈{expected_pnl}"


# ── TEST 3: scalar TP backward compatibility ──────────────────────────────────
def test_scalar_tp_backward_compat():
    """Single TP still fires correctly; no partial_exits on Trade."""
    df = _frame([
        ("2024-01-02 09:15", 100.0, 100.5, 99.8, 100.0),   # bar 0: signal
        ("2024-01-02 09:30", 100.0, 102.2, 99.9, 101.5),    # bar 1: entry@100, TP=102, high=102.2 → HIT
    ])
    trades, _ = simulate_trades(
        df=df, entry_condition="CLOSE > 0",
        stop_loss_pct=1.0, take_profit_pct=2.0,
        take_profit_targets=None, **_BASE,
    )
    assert len(trades) >= 1
    t = trades[0]
    print(f"  exit={t.exit_reason}  pnl={t.pnl_pct:.4f}%")
    assert t.exit_reason == "TAKE_PROFIT", f"Expected TAKE_PROFIT, got {t.exit_reason}"
    assert t.partial_exits == (), f"Expected no partial_exits, got {t.partial_exits}"


# ── TEST 4: gap-up fills TP1 at open (not rung price) ────────────────────────
def test_gap_up_fills_tp_at_open():
    """
    bar 0: signal
    bar 1: entry at 100.0, no TP hit
    bar 2: opens at 101.5 (gap above TP1=101) → TP1 fills at 101.5 (open), not 101.0
           high=104.5 → TP2=104 also hits → MULTI_TP_COMPLETE
    """
    df = _frame([
        ("2024-01-02 09:15", 100.0, 100.5, 99.8, 100.0),   # bar 0: signal
        ("2024-01-02 09:30", 100.0, 100.8, 99.9, 100.5),   # bar 1: entry@100, no hit
        ("2024-01-02 09:45", 101.5, 104.5, 101.2, 103.8),  # bar 2: gap-up, both rungs hit
    ])
    rungs = _parse_multi_take_profit_spec([
        {"type": "risk_reward", "value": 1.0, "exit_fraction": 0.40, "risk_action": "move_sl_to_breakeven"},
        {"type": "risk_reward", "value": 4.0, "exit_fraction": 0.60, "risk_action": "none"},
    ])
    trades, _ = simulate_trades(
        df=df, entry_condition="CLOSE > 0",
        stop_loss_pct=1.0, take_profit_pct=0.0,
        take_profit_targets=rungs, **_BASE,
    )
    assert len(trades) >= 1
    t = trades[0]
    print(f"  exit={t.exit_reason}  pnl={t.pnl_pct:.4f}%  partial_exits={len(t.partial_exits)}")
    assert t.exit_reason == "MULTI_TP_COMPLETE", f"Expected MULTI_TP_COMPLETE, got {t.exit_reason}"
    assert len(t.partial_exits) == 2
    pe0 = t.partial_exits[0]
    # TP1 rung = 101.0; bar opens at 101.5 → gap fill at 101.5
    assert pe0.get("fill_price") >= 101.0, f"fill_price={pe0.get('fill_price')} should be ≥ 101 (rung)"
    assert pe0.get("fill_price") >= 101.5 - 0.01, f"fill_price={pe0.get('fill_price')} should be ≥ bar_open=101.5 (gap fill)"
    print(f"  TP1 gap-fill at {pe0.get('fill_price')} (rung=101.0, bar_open=101.5)")


# ── TEST 5: loader mutual-exclusion: multi_tp + trailing_tp ──────────────────
def test_loader_mutual_exclusion():
    from engine.loader import _build_strategy_config
    import yaml
    raw = """
strategy:
  name: test
  risk_management:
    stop_loss_percent: 1.0
    take_profit_percent: 0.0
    trailing_take_profit:
      activate_after_pct: 1.0
      distance_pct: 0.5
    multi_take_profit:
      - type: risk_reward
        value: 2.0
        exit_fraction: 1.0
  entry_condition: "CLOSE > 0"
  exit_condition: ""
  variables: {}
  timeframe: "15m"
  direction: long_only
"""
    try:
        _build_strategy_config(yaml.safe_load(raw))
        raise AssertionError("Should have raised for mutual exclusion")
    except (ValueError, Exception) as e:
        if "AssertionError" in type(e).__name__:
            raise
        print(f"  raised: {e}")


# ── TEST 6: percent-type rung ─────────────────────────────────────────────────
def test_percent_type_rung():
    """Percent-based rung: exit_fraction=1.0 → single rung, fires at +2%."""
    df = _frame([
        ("2024-01-02 09:15", 100.0, 100.5, 99.8, 100.0),
        ("2024-01-02 09:30", 100.0, 102.2, 99.9, 101.5),
    ])
    rungs = _parse_multi_take_profit_spec([
        {"type": "percent", "value": 2.0, "exit_fraction": 1.0, "risk_action": "none"},
    ])
    trades, _ = simulate_trades(
        df=df, entry_condition="CLOSE > 0",
        stop_loss_pct=1.0, take_profit_pct=0.0,
        take_profit_targets=rungs, **_BASE,
    )
    assert len(trades) >= 1
    t = trades[0]
    print(f"  exit={t.exit_reason}  pnl={t.pnl_pct:.4f}%  partial_exits={len(t.partial_exits)}")
    # Single rung with 100% fraction → fires MULTI_TP_COMPLETE when it hits
    assert t.exit_reason == "MULTI_TP_COMPLETE", f"Expected MULTI_TP_COMPLETE, got {t.exit_reason}"
    assert len(t.partial_exits) == 1
    assert abs(t.pnl_pct - 2.0) < 0.1, f"pnl_pct={t.pnl_pct:.4f} expected≈2.0"
