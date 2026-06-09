"""
A beginner-friendly demo: watch a "filter" (a bouncer at the door) block trades.

Run it:   python3 scripts/demo_filter_blocking.py

We build 40 days of fake prices in two halves:
  • Days 1-20  = CALM market  (small up/down moves)
  • Days 21-40 = WILD market  (big, jumpy moves)

Our robot's rule is dead simple:  "BUY whenever today closed higher than yesterday."

Then we run it twice:
  1. WITHOUT any filter        -> it trades in both the calm AND the wild half.
  2. WITH a VOLATILITY filter  -> it REFUSES to trade once the market gets wild.
And once more with an EVENT filter that blocks a specific "earnings day".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quant_engine"))

from engine import indicators as I
from engine.simulator import simulate_trades


def make_prices() -> pd.DataFrame:
    """40 daily bars: a calm first half, then a wild second half."""
    days = pd.date_range("2026-01-01", periods=40, freq="D")
    rng = np.random.RandomState(7)
    closes, highs, lows, opens = [], [], [], []
    price = 100.0
    for i in range(40):
        wild = i >= 20
        step = rng.choice([-1.0, 1.2]) * (3.0 if wild else 0.6)   # bigger moves when wild
        band = 4.0 if wild else 0.4                                # bigger high-low range when wild
        opens.append(price)
        price = max(1.0, price + step)
        closes.append(price)
        highs.append(max(opens[-1], price) + band)
        lows.append(min(opens[-1], price) - band)
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000.0] * 40},
        index=days,
    )
    # Precompute NATR (a volatility gauge) so the volatility filter has its column.
    return I.add_all_indicators(df, {"NATR": [14]})


COMMON = dict(
    symbol="DEMO",
    entry_condition="CLOSE > PREV(CLOSE, 1)",   # buy when price rose vs yesterday
    exit_condition="",
    stop_loss_pct=3.0, take_profit_pct=5.0,
    slippage_bps=0.0, commission_bps=0.0,
    warm_up_candles=14,            # NATR needs ~14 bars to warm up
    max_holding_candles=2,
    objective="swing",             # daily bars (not intraday)
)


def show(title: str, trades, diags, block_flag: str | None):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"  Trades taken: {len(trades)}")
    for t in trades:
        d = str(t.entry_date)[:10] if t.entry_date else "?"
        print(f"    • BOUGHT around {d}  ->  P/L {t.pnl_pct:+.2f}%")
    if block_flag:
        blocked = [d for d in diags if d.get("entry_signal") and d.get(block_flag)]
        if blocked:
            print(f"\n  The bouncer said NO on these days (signal fired, but blocked):")
            for d in blocked:
                print(f"    ⛔ {str(d['timestamp'])[:10]}  — blocked by [{block_flag.replace('entry_blocked_','')}]")


def main():
    df = make_prices()
    calm_end = str(df.index[19])[:10]
    print(f"\nMarket is CALM through {calm_end}, then turns WILD after that.")

    # 1) No filter — robot trades in both halves.
    t0, d0 = simulate_trades(df=df, **COMMON)
    show("RUN 1 — NO FILTER (robot trades in calm AND wild markets)", t0, d0, None)

    # 2) Volatility filter — block entries when NATR is too high (market too wild).
    #    Pick a ceiling between the calm-half NATR and the wild-half NATR.
    natr = df["NATR_14"].dropna()
    ceiling = float(natr.quantile(0.5))
    t1, d1 = simulate_trades(
        df=df, **COMMON,
        vol_filter_metric="natr", vol_filter_window=14, vol_filter_max=ceiling,
    )
    show(f"RUN 2 — VOLATILITY FILTER (skip when NATR > {ceiling:.2f}; i.e. too wild)",
         t1, d1, "entry_blocked_volatility")

    # 3) Event filter — pretend 2026-01-18 is an earnings day; block it.
    #    (Jan 18 had a winning trade in RUN 1 — watch it disappear.)
    t2, d2 = simulate_trades(df=df, **COMMON, event_skip_dates=("2026-01-18",))
    show("RUN 3 — EVENT FILTER (skip the 'earnings day' 2026-01-18)",
         t2, d2, "entry_blocked_event")

    print("\n" + "-" * 64)
    print("Takeaway:")
    print(f"  • No filter      -> {len(t0)} trades (some in the risky wild half)")
    print(f"  • Volatility on  -> {len(t1)} trades (it stepped aside when the market got wild)")
    print(f"  • Event filter   -> {len(t2)} trades (it skipped the earnings day)")
    print("  Same prices, same buy rule — the FILTER changed which trades were allowed.")


if __name__ == "__main__":
    main()
