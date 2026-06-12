"""Entry-window parsing/gating for crypto (24/7) strategies.

Regression for the bug where a crypto strategy's entry_window (e.g. 00:00–22:00)
was parsed as IST — shifting it by −5:30, wrapping it past midnight
(start_utc > end_utc) — and the simulator's wrap-blind gate then blocked the
ENTIRE day, so no trade ever filled. Two fixes are covered:

  1. loader: crypto entry-window (and lunch-lull) times are parsed as UTC,
     regardless of the (stale equity) timezone stamped in the YAML.
  2. simulator: a wrap-around window (start_utc > end_utc) is honoured as a
     midnight-crossing window instead of blocking everything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quant_engine"))

from engine.loader import load_strategy_from_content
from engine.simulator import simulate_trades


def _crypto_yaml(entry_window_block: str) -> str:
    return f"""
strategy:
  name: Crypto EW
  symbol: ETH_USDT
  market: crypto
  asset_class: crypto_spot
  timeframe: 5m
  entry: {{ condition: "CLOSE > 0" }}
  exit: {{ condition: "" }}
  risk_management: {{ stop_loss_percent: 2.0, take_profit_percent: 4.0 }}
  {entry_window_block}
"""


# ── Loader: crypto times are UTC ──────────────────────────────────────────────

def test_loader_crypto_entry_window_parsed_as_utc_not_ist():
    # The assembler stamps timezone="Asia/Kolkata" even for crypto; the loader
    # must override to UTC. 00:00–22:00 UTC → 0 and 1320 (NOT 1110/990, which is
    # the −5:30 IST shift that wrapped past midnight and blocked the whole day).
    cfg = load_strategy_from_content(
        _crypto_yaml('entry_window: { start: "00:00", end: "22:00", timezone: "Asia/Kolkata" }')
    )
    assert cfg.entry_window_start_utc == 0
    assert cfg.entry_window_end_utc == 1320
    # And critically: not a wrap-around after parsing.
    assert cfg.entry_window_start_utc <= cfg.entry_window_end_utc


def test_loader_crypto_entry_window_utc_via_asset_class_only():
    yaml = """
strategy:
  name: Crypto EW2
  symbol: BTC_USDC
  asset_class: crypto_spot
  timeframe: 5m
  entry: { condition: "CLOSE > 0" }
  exit: { condition: "" }
  risk_management: { stop_loss_percent: 2.0, take_profit_percent: 4.0 }
  entry_window: { start: "09:00", end: "17:00", timezone: "Asia/Kolkata" }
"""
    cfg = load_strategy_from_content(yaml)
    assert cfg.entry_window_start_utc == 9 * 60      # 09:00 UTC
    assert cfg.entry_window_end_utc == 17 * 60       # 17:00 UTC


def test_loader_equity_entry_window_still_ist():
    yaml = """
strategy:
  name: Equity EW
  symbol: SBIN.NS
  market: indian_stocks
  timeframe: 15m
  entry: { condition: "CLOSE > 0" }
  exit: { condition: "" }
  risk_management: { stop_loss_percent: 1.5, take_profit_percent: 3.0 }
  entry_window: { start: "09:15", end: "15:30", timezone: "Asia/Kolkata" }
"""
    cfg = load_strategy_from_content(yaml)
    # 09:15 IST = 03:45 UTC = 225 min; 15:30 IST = 10:00 UTC = 600 min.
    assert cfg.entry_window_start_utc == 225
    assert cfg.entry_window_end_utc == 600


# ── Simulator: window gating ──────────────────────────────────────────────────

def _df(n: int = 80) -> pd.DataFrame:
    # 15-min bars from 03:45 UTC → spans ~03:45 to ~23:45 same day.
    idx = pd.date_range("2026-01-05 03:45", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100.0 + np.arange(n) * 0.05, index=idx)
    return pd.DataFrame(
        {
            "open": close.values, "high": (close + 1).values, "low": (close - 1).values,
            "close": close.values, "volume": [1000.0] * n,
        },
        index=idx,
    )


_COMMON = dict(
    symbol="ETH_USDT", entry_condition="CLOSE > 0", exit_condition="",
    stop_loss_pct=5.0, take_profit_pct=10.0, slippage_bps=0.0, commission_bps=0.0,
    warm_up_candles=14, max_holding_candles=5, objective="intraday",
)


def test_sim_crypto_full_day_window_allows_trades():
    # 00:00–22:00 UTC (the fixed crypto case): daytime bars are in-window → trades.
    trades, diags = simulate_trades(
        df=_df(), **_COMMON, entry_window_start_utc=0, entry_window_end_utc=22 * 60,
    )
    assert len(trades) >= 1


def test_sim_wrap_around_window_does_not_block_whole_day():
    # 22:00–06:00 UTC wraps past midnight (start 1320 > end 360). The early bars
    # (03:45–06:00 ≤ 360) are inside the window, so entries must still fire.
    # Pre-fix, the wrap-blind gate blocked every bar → zero trades.
    trades, _ = simulate_trades(
        df=_df(), **_COMMON, entry_window_start_utc=22 * 60, entry_window_end_utc=6 * 60,
    )
    assert len(trades) >= 1


def test_sim_window_outside_all_bars_blocks():
    # A window covering only 00:02–00:03 UTC matches no bar (bars start 03:45) →
    # the gate still blocks everything (proves we didn't disable the gate).
    trades, diags = simulate_trades(
        df=_df(), **_COMMON, entry_window_start_utc=2, entry_window_end_utc=3,
    )
    assert trades == []
    assert any(d.get("entry_blocked_entry_window") for d in diags)
