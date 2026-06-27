"""Phase 1 — trailing take-profit.

Covers the loader (parses/validates the new `trailing_take_profit` block and the
mutual-exclusion with `trailing_stop`) and the simulator (a ratcheting give-back
line that fires on a pull-back from the peak/trough and books TRAILING_TAKE_PROFIT
— for both LONG and SHORT — while leaving static take-profit behaviour intact).
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

from engine.loader import (  # noqa: E402
    _parse_trailing_take_profit_spec,
    load_strategy_from_content,
)
from engine.simulator import simulate_trades  # noqa: E402


def _df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame from (open, high, low, close) tuples."""
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 1000.0} for o, h, l, c in bars]
    )


# ── Loader: parsing & validation ─────────────────────────────────────────────


def test_loader_returns_none_when_block_absent():
    assert _parse_trailing_take_profit_spec(None) is None
    assert _parse_trailing_take_profit_spec({}) is None
    assert _parse_trailing_take_profit_spec("") is None


def test_loader_percent_spec_with_activation():
    spec = _parse_trailing_take_profit_spec(
        {"type": "percent", "distance_pct": 1.5, "activate_after_pct": 3.0}
    )
    assert spec == {"type": "percent", "distance_pct": 1.5, "activate_after_pct": 3.0}


def test_loader_percent_spec_defaults_activation_to_zero():
    spec = _parse_trailing_take_profit_spec({"type": "percent", "distance_pct": 2.0})
    assert spec["activate_after_pct"] == 0.0


def test_loader_requires_positive_distance():
    with pytest.raises(ValueError, match="distance_pct must be > 0"):
        _parse_trailing_take_profit_spec({"type": "percent", "distance_pct": 0})


def test_loader_rejects_non_percent_type_in_phase1():
    # ATR/EMA are deliberately out of scope until the OMS can honour them.
    with pytest.raises(ValueError, match="trailing_take_profit.type must be one of"):
        _parse_trailing_take_profit_spec({"type": "atr", "multiplier": 2.0, "window": 14})


def test_loader_carries_spec_into_strategy_config():
    yaml_content = """
strategy:
  name: test
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 5m
  entry:
    condition: "CLOSE > EMA(20)"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
  trailing_take_profit:
    type: percent
    distance_pct: 1.5
    activate_after_pct: 3.0
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.trailing_take_profit_spec == {
        "type": "percent",
        "distance_pct": 1.5,
        "activate_after_pct": 3.0,
    }


def test_loader_allows_trailing_stop_and_trailing_tp_together():
    """A position MAY carry BOTH trailing exits — the engine runs both lines and
    exits on whichever a reversal reaches first (a staged trail)."""
    yaml_content = """
strategy:
  name: test
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 5m
  entry:
    condition: "CLOSE > EMA(20)"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
  trailing_stop:
    type: percent
    distance_pct: 2.0
  trailing_take_profit:
    type: percent
    distance_pct: 1.0
    activate_after_pct: 8.0
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.trailing_stop_spec["distance_pct"] == 2.0
    assert cfg.trailing_take_profit_spec["distance_pct"] == 1.0


# ── Simulator: LONG ──────────────────────────────────────────────────────────

# Shared one-shot entry: fires only when CLOSE crosses above 99 from below, so
# every scenario opens exactly one trade and the exit is unambiguous.
_ENTRY = "CLOSE > 99 AND PREV(CLOSE, 1) <= 99"


def test_long_trailing_tp_fires_on_pullback_from_peak():
    """Run up to a peak of 116.5, then pull back. A 2% give-back line sits at
    114.17; the pull-back bar straddles it → TRAILING_TAKE_PROFIT at that line."""
    df = _df([
        (98.0,  98.5,  97.5,  98.0),    # below trigger
        (100.0, 100.5, 99.5,  100.0),   # cross above 99 → entry signal
        (100.0, 100.5, 99.5,  100.0),   # entry fill at this bar's open (100)
        (104.0, 104.5, 103.5, 104.0),
        (108.0, 108.5, 107.5, 108.0),
        (112.0, 112.5, 111.5, 112.0),
        (116.0, 116.5, 115.5, 116.0),   # peak high = 116.5
        (114.0, 115.0, 113.0, 114.0),   # pull-back straddles give-back line 114.17
    ])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition=_ENTRY,
        exit_condition="",
        stop_loss_pct=10.0,           # generous — must not fire
        take_profit_pct=50.0,         # far away — must not fire
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        trailing_take_profit_spec={"type": "percent", "distance_pct": 2.0},
    )

    assert len(trades) == 1
    assert trades[0].exit_reason == "TRAILING_TAKE_PROFIT"
    # Fill at the give-back line 116.5 * 0.98 = 114.17 (× 0.999 delivery STT on sell).
    assert pytest.approx(trades[0].exit_price, abs=0.5) == 114.17 * 0.999
    assert trades[0].pnl_pct > 0
    # The structured rationale must carry the new machine code + trigger type.
    assert trades[0].exit_reason_detail["code"] == "TRAILING_TAKE_PROFIT"
    assert trades[0].exit_reason_detail["trigger"]["type"] == "trailing_take_profit"


def test_long_trailing_tp_line_survives_deactivation_ratchet():
    """With activate_after_pct=3 and a wide 3% give-back, the line locks while in
    profit, then HOLDS even after price dips back below the 3% activation level
    (the ratchet), so a later dip still exits via TRAILING_TAKE_PROFIT."""
    df = _df([
        (98.0,  98.5,  97.5,  98.0),
        (100.0, 100.5, 99.5,  100.0),   # entry signal
        (100.0, 100.5, 99.5,  100.0),   # fill at 100
        (104.0, 104.5, 103.5, 104.0),   # +4% → trailing armed; line = 104.5*0.97 = 101.365
        (102.0, 102.5, 101.8, 102.0),   # +2% (< activation) — line must NOT reset
        (101.0, 101.5, 100.8, 101.0),   # low 100.8 <= 101.365 → fire at the locked line
    ])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition=_ENTRY,
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=50.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        trailing_take_profit_spec={"type": "percent", "distance_pct": 3.0, "activate_after_pct": 3.0},
    )

    assert len(trades) == 1
    assert trades[0].exit_reason == "TRAILING_TAKE_PROFIT"
    # Locked line = 104.5 * 0.97 = 101.365 (× 0.999 STT).
    assert pytest.approx(trades[0].exit_price, abs=0.3) == 101.365 * 0.999


def test_long_trailing_tp_dormant_until_activation():
    """activate_after_pct=5 with a tight 0.1% give-back. The trade never gets 5%
    in profit, so the give-back line never arms and no trailing TP fires."""
    df = _df([
        (100.0, 100.3, 99.7,  100.0),
        (100.0, 100.3, 99.7,  100.0),   # entry signal won't fire (no cross) — see below
        (100.0, 102.3, 99.7,  102.0),
        (100.5, 101.0, 100.2, 100.5),
        (101.5, 101.8, 101.2, 101.5),
        (100.0, 100.3, 99.7,  100.0),
    ])
    # Use a trigger that fires on the first bar's open instead of a cross.
    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition="CLOSE >= 100",   # fires on bar 0 → fill bar 1
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=50.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        trailing_take_profit_spec={"type": "percent", "distance_pct": 0.1, "activate_after_pct": 5.0},
    )
    if trades:
        assert trades[0].exit_reason != "TRAILING_TAKE_PROFIT"


def test_long_static_take_profit_still_fires_without_pullback():
    """A clean run straight into the static target (no pull-back) must still book
    TAKE_PROFIT — the trailing line is configured but never breached."""
    df = _df([
        (98.0,  98.5,  97.5,  98.0),
        (100.0, 100.5, 99.5,  100.0),   # entry signal → fill at 100
        (100.0, 100.5, 99.5,  100.0),
        (103.0, 103.5, 102.5, 103.0),
        (106.0, 106.5, 105.5, 106.0),   # high 106.5 >= target 105 → TAKE_PROFIT
    ])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition=_ENTRY,
        exit_condition="",
        stop_loss_pct=10.0,
        take_profit_pct=5.0,            # target = 105
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        # Wide give-back so it never arms ahead of the clean target hit.
        trailing_take_profit_spec={"type": "percent", "distance_pct": 5.0},
    )

    assert len(trades) == 1
    assert trades[0].exit_reason == "TAKE_PROFIT"


# ── Simulator: SHORT (mirror) ────────────────────────────────────────────────


def test_short_trailing_tp_fires_on_bounce_from_trough():
    """Short from ~98, price falls to a trough of 89.5, then bounces. A 2%
    give-back ceiling sits at 91.29; the bounce bar straddles it → exit."""
    df = _df([
        (101.0, 101.5, 100.5, 101.0),   # above trigger
        (98.0,  98.5,  97.5,  98.0),    # cross below 99 → short entry signal
        (98.0,  98.5,  97.5,  98.0),    # entry fill at 98
        (94.0,  94.5,  93.5,  94.0),
        (90.0,  90.5,  89.5,  90.0),    # trough low = 89.5
        (92.0,  93.0,  91.0,  92.0),    # bounce straddles give-back ceiling 91.29
    ])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        side="SHORT",
        entry_condition="CLOSE < 99 AND PREV(CLOSE, 1) >= 99",
        exit_condition="",
        stop_loss_pct=10.0,           # 107.8 — must not fire
        take_profit_pct=50.0,         # 49 — must not fire
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=20,
        objective="positional",
        trailing_take_profit_spec={"type": "percent", "distance_pct": 2.0},
    )

    assert len(trades) == 1
    assert trades[0].exit_reason == "TRAILING_TAKE_PROFIT"
    assert trades[0].pnl_pct > 0          # covering below entry = profit on a short
    # Ceiling = 89.5 * 1.02 = 91.29.
    assert pytest.approx(trades[0].exit_price, abs=0.6) == 91.29 * (1.0 + 0.0)  # cost-free raw ≈ line


# ── Optional static take-profit (trailing supplies the profit exit) ──────────


def test_trailing_only_captures_runner_when_static_tp_omitted():
    """The exact failure from the BTC example: a small static target equal to the
    trailing activation exits FIRST and starves the trail. With the static target
    OMITTED (take_profit_pct=0), the trade trails the peak and books a much larger
    TRAILING_TAKE_PROFIT."""
    df = _df([
        (98.0,  98.2,  97.8,  98.0),
        (100.0, 100.2, 99.8,  100.0),   # entry signal → fill at 100
        (100.0, 100.2, 99.8,  100.0),
        (101.5, 101.7, 101.3, 101.5),   # +1.5% (a static 1% TP would have exited here)
        (102.5, 102.7, 102.3, 102.5),
        (103.5, 103.7, 103.3, 103.5),   # peak 103.7
        (102.0, 102.2, 101.8, 102.0),   # pull-back > 0.5% off peak → trailing fires
        (101.0, 101.2, 100.8, 101.0),
    ])

    trades, _ = simulate_trades(
        df=df, symbol="BTC", entry_condition=_ENTRY, exit_condition="",
        stop_loss_pct=2.0, take_profit_pct=0.0,        # <-- no static target
        slippage_bps=0.0, commission_bps=0.0, warm_up_candles=0,
        max_holding_candles=20, objective="intraday",
        trailing_take_profit_spec={"type": "percent", "distance_pct": 0.5, "activate_after_pct": 1.0},
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "TRAILING_TAKE_PROFIT"
    # Captured well beyond the 1% a static target would have given.
    assert trades[0].pnl_pct > 2.5


def test_no_static_tp_and_no_trailing_raises():
    df = _df([(100.0, 100.2, 99.8, 100.0)] * 5)
    with pytest.raises(ValueError, match="take_profit_pct must be greater than zero"):
        simulate_trades(
            df=df, symbol="X", entry_condition="CLOSE > 0", exit_condition="",
            stop_loss_pct=2.0, take_profit_pct=0.0,    # no static TP and no trailing → invalid
            slippage_bps=0.0, commission_bps=0.0, objective="positional",
        )


def test_trailing_stop_alone_supplies_the_exit_and_books_trailing_stop():
    """"Ride until stopped out": a trailing STOP with no static target must NOT raise
    (it is a complete exit), and a run-then-pullback books TRAILING_STOP in profit.
    This is the engine half of the NAUKRI prompt "once up 1%, trail my stop by 0.5%"."""
    df = _df([
        (98.0,  98.2,  97.8,  98.0),
        (100.0, 100.2, 99.8,  100.0),   # entry signal → fill at 100
        (100.0, 100.2, 99.8,  100.0),
        (101.5, 101.7, 101.3, 101.5),   # +1.5% → trailing armed (activate 1%)
        (102.5, 102.7, 102.3, 102.5),
        (103.5, 103.7, 103.3, 103.5),   # peak 103.7 → floor = 103.7*0.995 = 103.18
        (102.0, 102.2, 101.8, 102.0),   # low 101.8 <= 103.18 → trailing stop fires
        (101.0, 101.2, 100.8, 101.0),
    ])

    trades, _ = simulate_trades(
        df=df, symbol="TEST.NS", entry_condition=_ENTRY, exit_condition="",
        stop_loss_pct=2.0, take_profit_pct=0.0,        # <-- no static target
        slippage_bps=0.0, commission_bps=0.0, warm_up_candles=0,
        max_holding_candles=20, objective="positional",
        trailing_stop_spec={"type": "percent", "distance_pct": 0.5, "activate_after_pct": 1.0},
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "TRAILING_STOP"
    assert trades[0].pnl_pct > 2.5          # booked the ratcheted gain, not a loss


def test_both_trailing_lines_stage_and_tighter_late_line_governs():
    """A trailing STOP (2%, immediate) AND a trailing TAKE-PROFIT (1%, after +8%) on
    the same trade: below +8% the wide stop governs; once past +8% the tight TP line
    is closer to the peak and fires first on a small pull-back → TRAILING_TAKE_PROFIT.
    Proves both lines run together and the tightest active one wins (staged trail)."""
    df = _df([
        (98.0,  98.2,  97.8,  98.0),
        (100.0, 100.2, 99.8,  100.0),   # entry signal → fill at 100
        (100.0, 100.2, 99.8,  100.0),
        (104.0, 104.2, 103.8, 104.0),   # +4% → stop trails (2%); TP not yet armed (<8%)
        (110.0, 110.5, 109.8, 110.0),   # peak 110.5, +10% → TP armed: line 109.395 > stop line 108.29
        (109.5, 109.6, 109.0, 109.5),   # low 109.0 <= TP line 109.395 but > stop line 108.29
    ])

    trades, _ = simulate_trades(
        df=df, symbol="TEST.NS", entry_condition=_ENTRY, exit_condition="",
        stop_loss_pct=10.0, take_profit_pct=0.0,       # no static target; trailing supplies it
        slippage_bps=0.0, commission_bps=0.0, warm_up_candles=0,
        max_holding_candles=20, objective="positional",
        trailing_stop_spec={"type": "percent", "distance_pct": 2.0},
        trailing_take_profit_spec={"type": "percent", "distance_pct": 1.0, "activate_after_pct": 8.0},
    )
    assert len(trades) == 1
    # The tight TP line (1% off the +10% peak) governs — not the wide 2% stop.
    assert trades[0].exit_reason == "TRAILING_TAKE_PROFIT"
    assert trades[0].pnl_pct > 8.0


def test_loader_makes_take_profit_optional_with_trailing_stop():
    """A YAML with a trailing_stop and NO take_profit_percent loads with take_profit=0
    (no static cap injected) — the trailing stop is the exit."""
    from engine.loader import load_strategy_from_content

    content = """
strategy:
  name: t
  symbol: NAUKRI.NS
  market: indian_stocks
  timeframe: 1m
  entry:
    condition: "CLOSE > EMA(20)"
  risk_management:
    stop_loss_percent: 1.0
  trailing_stop:
    type: percent
    distance_pct: 0.5
    activate_after_pct: 2.0
"""
    cfg = load_strategy_from_content(content)
    assert cfg.take_profit == 0.0
    assert cfg.trailing_stop_spec["distance_pct"] == 0.5
    assert cfg.trailing_take_profit_spec is None


def test_loader_makes_take_profit_optional_with_trailing():
    """A YAML with a trailing_take_profit and NO take_profit_percent loads with
    take_profit=0 (no static cap) instead of the legacy 5% default."""
    from engine.loader import load_strategy_from_content

    content = """
strategy:
  name: t
  symbol: BTC_USDT
  market: crypto
  timeframe: 3m
  entry:
    condition: "CLOSE > EMA(20)"
  risk_management:
    stop_loss_percent: 2.0
  trailing_take_profit:
    type: percent
    distance_pct: 0.5
    activate_after_pct: 1.0
"""
    cfg = load_strategy_from_content(content)
    assert cfg.take_profit == 0.0
    assert cfg.trailing_take_profit_spec["distance_pct"] == 0.5


def test_loader_still_requires_take_profit_without_trailing():
    from engine.loader import load_strategy_from_content
    content = """
strategy:
  name: t
  symbol: BTC_USDT
  market: crypto
  timeframe: 3m
  entry:
    condition: "CLOSE > EMA(20)"
  risk_management:
    stop_loss_percent: 2.0
    take_profit_percent: 0
"""
    with pytest.raises(ValueError, match="take_profit_percent must be greater than zero"):
        load_strategy_from_content(content)


# ── Backward-compat ──────────────────────────────────────────────────────────


def test_no_trailing_tp_spec_is_unchanged():
    """Without the spec the simulator behaves exactly as before: a clean downside
    move books STOP_LOSS."""
    df = _df([
        (98.0, 98.2, 97.8, 98.0),
        (100.0, 100.2, 99.8, 100.0),    # entry signal → fill 100
        (100.0, 100.2, 99.8, 100.0),
        (95.0, 95.2, 94.8, 95.0),
        (90.0, 90.2, 89.8, 90.0),       # well below 2% stop → STOP_LOSS
    ])

    trades, _ = simulate_trades(
        df=df,
        symbol="TEST.NS",
        entry_condition=_ENTRY,
        exit_condition="",
        stop_loss_pct=2.0,
        take_profit_pct=10.0,
        slippage_bps=0.0, commission_bps=0.0,
        warm_up_candles=0,
        max_holding_candles=10,
        objective="positional",
    )
    assert len(trades) >= 1
    assert trades[0].exit_reason == "STOP_LOSS"
