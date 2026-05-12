# Signal Planning & Parameter System — Complete Reference

This document explains every component of how Stretus selects signals and
computes their parameters — from what the user types in chat to the final
backtest condition string like `SMA(30) > SMA(86) AND CLOSE > VWAP`.

---

## Table of Contents

1. [What is Signal Planning?](#1-what-is-signal-planning)
2. [The Three-Layer Parameter System](#2-the-three-layer-parameter-system)
3. [Layer 1 — YAML Default Values](#3-layer-1--yaml-default-values)
4. [Layer 2 — OHLCV Statistical Estimation](#4-layer-2--ohlcv-statistical-estimation)
5. [Layer 3 — Performance Cache](#5-layer-3--performance-cache)
6. [Signal Selection Flow](#6-signal-selection-flow)
7. [Signal Scoring System](#7-signal-scoring-system)
8. [All Supported Signals and Their Defaults](#8-all-supported-signals-and-their-defaults)
9. [Complete End-to-End Example](#9-complete-end-to-end-example)
10. [How to Add a New Signal](#10-how-to-add-a-new-signal)
11. [File Map](#11-file-map)

---

## 1. What is Signal Planning?

When a user finishes entering their trading preferences (stock, timeframe,
objective, sentiment, experience, goal), the system must automatically decide:

- **Entry trigger** — the primary signal that says "NOW is the time to enter"
- **Entry filter** — a confirmation signal that says "yes, the entry is valid"
- **Exit trigger** — the signal that says "time to get out"

And for each signal, it must decide the **parameters** — for example, RSI needs
a period (how many candles to look at) and a threshold (what level counts as
oversold).

This entire process is called **signal planning** and happens inside
`plan_strategy_signals()` in
[app/services/knowledge/retriever.py](../app/services/knowledge/retriever.py).

---

## 2. The Three-Layer Parameter System

Parameters are resolved in this exact order. The first layer that has data wins.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — Performance Cache                                │
│  "We ran 25 backtests for TCS 15m with RSI(9) and got 72%  │
│   win rate. Use RSI(9) next time."                          │
│  File: app/core/signal_performance_cache.json               │
│  Requires: ≥ 10 recorded backtests for this signal+stock+tf │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — OHLCV Statistical Estimation                     │
│  "HDFCBANK 1m has ATR=0.04% per bar. Use SMA(30,86)        │
│   instead of the generic SMA(20,50)."                       │
│  File: app/core/signal_param_estimator.py                   │
│  Requires: ≥ 60 OHLCV bars for this stock+timeframe        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — YAML Defaults                                    │
│  "RSI period is 14, threshold is 30. Always."              │
│  File: app/core/signals_config.yaml                         │
│  Always available. Used as fallback.                        │
└─────────────────────────────────────────────────────────────┘
```

**Which layer is used when:**

| Situation | Layer used |
|---|---|
| Brand new stock, first ever backtest, no OHLCV data yet | Layer 1 (YAML) |
| OHLCV data fetched, first backtest | Layer 2 (statistical estimate from actual price data) |
| 10 or more backtests completed for this stock+signal+timeframe | Layer 3 (cache with empirically validated params) |

---

## 3. Layer 1 — YAML Default Values

### Where it lives

All defaults are in
[app/core/signals_config.yaml](../app/core/signals_config.yaml)
under the `signal_parameters` section.

### How it works

The YAML has three context keys for each signal:

```yaml
signal_parameters:
  sma_cross_up:
    intraday:          # used when objective = "intraday"
      window_fast: 20
      window_slow: 50
    daily_or_higher:   # used when timeframe is 1d, 3d, 1w, or 1m
      window_fast: 20
      window_slow: 50
    default:           # used for all other cases (15m positional etc.)
      window_fast: 20
      window_slow: 100
```

The code in `get_signal_params()` in
[app/core/signals_config_loader.py](../app/core/signals_config_loader.py)
picks the right section:

```
objective == "intraday"          → use "intraday" section
timeframe in {1d, 3d, 1w, 1m}   → use "daily_or_higher" section
everything else                  → use "default" section
```

### Complete table of all YAML defaults

#### EMA Crossover Signals

| Signal | Context | window_fast | window_slow |
|---|---|---|---|
| `ema_cross_up` | intraday | 9 | 21 |
| `ema_cross_up` | daily_or_higher | 10 | 20 |
| `ema_cross_up` | default (15m–1h positional) | 12 | 26 |
| `ema_cross_down` | intraday | 9 | 21 |
| `ema_cross_down` | daily_or_higher | 10 | 20 |
| `ema_cross_down` | default | 12 | 26 |

#### SMA Crossover Signals

| Signal | Context | window_fast | window_slow |
|---|---|---|---|
| `sma_cross_up` | intraday | 20 | 50 |
| `sma_cross_up` | daily_or_higher | 20 | 50 |
| `sma_cross_up` | default | 20 | 100 |
| `sma_cross_down` | intraday | 20 | 50 |
| `sma_cross_down` | daily_or_higher | 20 | 50 |
| `sma_cross_down` | default | 20 | 100 |

#### Price vs MA Filter Signals

| Signal | Context | window |
|---|---|---|
| `is_above_sma` | intraday | 50 |
| `is_above_sma` | daily_or_higher | 50 |
| `is_above_sma` | default | 100 |
| `price_above_ema` | intraday | 20 |
| `price_above_ema` | daily_or_higher | 20 |
| `price_above_ema` | default | 34 |
| `price_below_ema` | intraday | 20 |
| `price_below_ema` | daily_or_higher | 20 |
| `price_below_ema` | default | 34 |

#### RSI Signals

| Signal | window | threshold | What it means |
|---|---|---|---|
| `rsi_oversold` | 14 | 30.0 | RSI(14) < 30 → buy signal |
| `rsi_overbought` | 14 | 70.0 | RSI(14) > 70 → sell signal |
| `rsi_cross_up` | 14 | 50.0 | RSI(14) crosses above 50 → upward momentum |
| `rsi_cross_down` | 14 | 50.0 | RSI(14) crosses below 50 → downward momentum |

#### MACD Signals

| Signal | window_fast | window_slow | window_sign |
|---|---|---|---|
| `macd_bullish_cross` | 12 | 26 | 9 |
| `macd_bearish_cross` | 12 | 26 | 9 |
| `macd_positive` | 12 | 26 | 9 |

#### Bollinger Band Signals

| Signal | window | window_dev | What it means |
|---|---|---|---|
| `price_above_bb_upper` | 20 | 2 | CLOSE > upper band |
| `price_below_bb_lower` | 20 | 2 | CLOSE < lower band |
| `bb_pct_b_high` | 20 | 2 | threshold=0.8 — %B > 0.8 |
| `bb_pct_b_low` | 20 | 2 | threshold=0.2 — %B < 0.2 |

#### Volume Signals

| Signal | window | multiplier | What it means |
|---|---|---|---|
| `volume_spike` | 20 | 1.5 | Volume > 20-bar average × 1.5 |
| `volume_dry_up` | 20 | 0.5 | Volume < 20-bar average × 0.5 |
| `high_delivery_volume` | 10 | 1.3 | Bullish close + volume > 10-bar avg × 1.3 |

#### VWAP Signals

| Signal | Parameters | What it means |
|---|---|---|
| `vwap_bullish` | none | CLOSE > VWAP |
| `vwap_bearish` | none | CLOSE < VWAP |

#### India-Specific Signals

| Signal | Parameter | Default | What it means |
|---|---|---|---|
| `opening_range_breakout` | opening_bars | 4 | CLOSE > highest price of first 4 candles |
| `gap_up_open` | gap_pct | 0.5% | OPEN > previous CLOSE × 1.005 |
| `gap_down_open` | gap_pct | 0.5% | OPEN < previous CLOSE × 0.995 |
| `inside_bar` | none | — | HIGH ≤ prev HIGH AND LOW ≥ prev LOW |

#### Rate of Change Signals

| Signal | window | What it means |
|---|---|---|
| `roc_positive` | 12 | Price today > Price 12 bars ago |
| `roc_negative` | 12 | Price today < Price 12 bars ago |

### Signals NOT in YAML defaults (no free parameters)

These signals have fixed logic that doesn't need tunable parameters:

| Signal | Why no params |
|---|---|
| `vwap_bullish` / `vwap_bearish` | VWAP is computed for the whole session, no period to tune |
| `inside_bar` | It's a candle pattern: current High ≤ prev High AND current Low ≥ prev Low |
| `macd_bullish_cross` / `macd_bearish_cross` | 12/26/9 is universally accepted; not estimated |

---

## 4. Layer 2 — OHLCV Statistical Estimation

### What OHLCV is

OHLCV is a table of price candles. Every row is one time period (1 minute, 5
minutes, 15 minutes, or 1 day).

```
Timestamp            Open     High     Low      Close    Volume
2026-01-02 09:15    1705.0   1712.0   1701.0   1709.5   850,000
2026-01-02 09:16    1709.5   1711.0   1707.0   1708.0   310,000
2026-01-02 09:17    1708.0   1714.5   1707.5   1713.0   420,000
```

- **Open** = price the candle started at
- **High** = highest price touched during the candle
- **Low** = lowest price touched during the candle
- **Close** = price the candle ended at
- **Volume** = shares traded during this candle

For HDFCBANK on 1m, 3 months of data ≈ 16,875 rows (375 bars/session × 45
trading days).

### Where it lives

[app/core/signal_param_estimator.py](../app/core/signal_param_estimator.py)

The main function is `enrich_plan_with_ohlcv()` in
[app/services/knowledge/retriever.py](../app/services/knowledge/retriever.py).

### When it runs

After OHLCV is fetched from the market data service but BEFORE the backtest
runs. The call happens in `chat_service.py`:

```python
ohlcv_records = await fetch_ohlcv_records(market_data_request)
builder.signal_plan = enrich_plan_with_ohlcv(
    plan          = builder.signal_plan,
    ohlcv_records = ohlcv_records,
    objective     = builder.objective,
    timeframe     = builder.timeframe,
)
```

### Minimum requirement

All estimators need **at least 60 OHLCV bars**. Below that the YAML defaults
are returned unchanged because a statistical estimate from fewer than 60 data
points is unreliable.

---

### Estimator 1 — RSI Period and Threshold

**File:** `estimate_rsi_params()` in signal_param_estimator.py  
**Applies to:** `rsi_oversold`, `rsi_overbought`, `rsi_cross_up`, `rsi_cross_down`

#### What RSI is

RSI (Relative Strength Index) is a number from 0 to 100 that measures how fast
price has been rising vs falling. Below 30 = oversold (beaten down, likely to
bounce). Above 70 = overbought (run up too much, likely to fall).

The **period** (default 14) is how many candles RSI looks at. Period 9 = fast
and reactive. Period 21 = slow and smooth.

#### The problem with RSI(14) for everyone

ADANIENT 1m can move 1% in a single candle. HDFCBANK 1m moves 0.05% per candle.

If ADANIENT drops 3% in 10 candles and starts recovering, RSI(14) might still
be reading "oversold" 5 candles after the bounce already started. The signal is
too late. RSI(9) or RSI(10) would have caught it.

For HDFCBANK, RSI(9) gives 4–5 false "oversold" readings per hour from tiny
noise moves. RSI(15) or RSI(16) filters that noise.

#### How it's computed from OHLCV

**Step 1 — Compute returns per candle**

```
Close prices:   1709.5, 1708.0, 1713.0, 1710.5, 1715.0
Returns:        —,      -0.09%, +0.29%, -0.15%, +0.26%
```

**Step 2 — Measure recent volatility (last 20 candles)**

Standard deviation of the last 20 returns. This measures how spread out the
returns are right now.

```
HDFCBANK 1m, last 20 returns: [-0.08%, +0.05%, -0.12%, ...]
standard deviation = 0.09%   ← calm

ADANIENT 1m, last 20 returns: [-0.95%, +1.20%, -0.80%, ...]
standard deviation = 0.95%   ← volatile
```

**Step 3 — Measure long-term volatility (last 60 candles)**

Same calculation but over 60 candles. This is the "normal" baseline.

**Step 4 — Compute volatility ratio**

```
vol_ratio = recent_volatility / long_term_volatility
```

If `vol_ratio > 1`: the stock is MORE volatile right now than usual → need
faster RSI (shorter period).

If `vol_ratio < 1`: the stock is CALMER right now than usual → need slower RSI
(longer period).

**Step 5 — Scale the RSI period**

```
new_period = round(14 / sqrt(vol_ratio))
Clamped to range [7, 21]
```

Why `sqrt`? Because the relationship is not linear. If a stock is 4× more
volatile, you don't need a period 4× shorter — roughly 2× shorter works
(square root of 4 = 2).

**Example calculation:**

```
HDFCBANK:  recent_vol = 0.09%,  long_vol = 0.10%
           vol_ratio = 0.9
           new_period = round(14 / sqrt(0.9)) = round(14 / 0.95) = 15

ADANIENT:  recent_vol = 0.95%,  long_vol = 0.50%
           vol_ratio = 1.9
           new_period = round(14 / sqrt(1.9)) = round(14 / 1.38) = 10
```

**Step 6 — Compute the threshold from RSI history**

Instead of always using 30 (oversold) or 70 (overbought), compute actual RSI
values for the whole OHLCV history using the new period, then find:

- Oversold threshold = 15th percentile of all RSI values
- Overbought threshold = 85th percentile of all RSI values

Meaning: "The stock is oversold only in the bottom 15% of historical RSI
readings — that's the REAL extreme level for this stock."

```
HDFCBANK 1m — RSI(15) history sorted:
[18.2, 21.0, 24.1, 25.5, 27.2, 28.0, 29.1, 30.2, ...]
                        ^
              15th percentile = 27.5

Result: rsi_oversold threshold changes from 30.0 → 27.5
Formula becomes: RSI(15) < 27.5  (was RSI(14) < 30.0)
```

---

### Estimator 2 — SMA/EMA Periods

**File:** `estimate_ma_params()` in signal_param_estimator.py  
**Applies to:** `sma_cross_up`, `sma_cross_down`, `ema_cross_up`, `ema_cross_down`,
`price_above_ema`, `price_below_ema`, `is_above_sma`

#### What ATR is

ATR (Average True Range) measures how much price moves per candle. It takes the
biggest of three values for each candle:

```
True Range = max(
  High - Low,               ← candle's internal range
  |High - previous Close|,  ← gap up + candle range
  |Low  - previous Close|   ← gap down + candle range
)
```

ATR% = ATR expressed as a percentage of the closing price.

#### Why ATR determines MA periods

Moving averages smooth out price noise. How much "noise" is in each candle
determines how many candles you need to confirm a real trend vs a random wiggle.

If each candle moves 0.04% (HDFCBANK 1m), a "trend" needs many candles to be
visible above the noise. Short MAs like SMA(20) cross up and down constantly
from pure noise, giving dozens of false signals per day.

If each candle moves 2% (ADANIENT 15m), a trend is visible very quickly. Long
MAs like SMA(50) lag too far behind — by the time they signal, the trend is
half over.

#### How it's computed from OHLCV

**Step 1 — Compute ATR% for each candle**

```
For each of the last 20 candles:
  TR = max(High-Low, |High-prevClose|, |Low-prevClose|)
  ATR% = TR / Close × 100

ATR% (rolling 20-bar average):
  HDFCBANK 1m: ATR% ≈ 0.041%  ← very calm per candle
  ADANIENT 15m: ATR% ≈ 2.15%  ← very noisy per candle
```

**Step 2 — Compare to reference ATR**

The code uses reference values calibrated for typical NSE stocks:
- Intraday reference ATR = 0.20% per bar (a typical 15m intraday stock)
- Positional reference ATR = 1.50% per bar (a typical daily stock)

**Step 3 — Compute scaling factor**

```
vol_factor = (reference_ATR / stock_ATR) ^ 0.35
Clamped to [0.50, 1.80]
```

If `stock_ATR > reference_ATR`: stock is noisier than average → `vol_factor < 1`
→ shorter periods (react faster, reduce lag-induced false signals).

If `stock_ATR < reference_ATR`: stock is calmer than average → `vol_factor > 1`
→ longer periods (smooth out the already-small noise).

The exponent 0.35 makes the scaling gentle — doubling ATR doesn't halve the
period, it reduces it by about 21%.

**Example calculation:**

```
HDFCBANK 1m:
  ATR% = 0.041%
  reference = 0.20% (intraday)
  vol_factor = (0.20 / 0.041)^0.35 = (4.88)^0.35 = 1.72
  (clamped to max 1.80 → stays 1.72)

  new_fast = round(20 × 1.72) = 34  (clamped to max 30 → 30)
  new_slow = round(50 × 1.72) = 86

  Formula: SMA(20) > SMA(50) becomes SMA(30) > SMA(86)

ADANIENT 15m:
  ATR% = 2.15%
  vol_factor = (0.20 / 2.15)^0.35 = (0.093)^0.35 = 0.43
  (clamped to min 0.50 → becomes 0.50)

  new_fast = round(20 × 0.50) = 10
  new_slow = max(10+10, round(50 × 0.50)) = max(20, 25) = 25

  Formula: SMA(20) > SMA(50) becomes SMA(10) > SMA(25)
```

---

### Estimator 3 — Volume Spike Multiplier

**File:** `estimate_volume_params()` in signal_param_estimator.py  
**Applies to:** `volume_spike`, `volume_dry_up`, `high_delivery_volume`

#### What volume_spike does

The signal fires when: `current volume > 20-bar average volume × 1.5`

The 1.5 multiplier says "volume is 50% higher than the recent average = spike."

#### The problem

"Unusual" is different for every stock. HDFCBANK on a busy day trades 1.8×
its average. ADANIENT on a busy day trades 3× its average. Using 1.5× for both:

- HDFCBANK 1.5× fires almost every day — not really unusual
- ADANIENT 1.5× fires all the time — nearly every day qualifies

#### How it's computed from OHLCV

**Step 1 — Compute volume ratio for every bar**

```
rolling_avg = average volume over last 20 bars (updated each bar)
vol_ratio   = current bar's volume / rolling_avg

Example (HDFCBANK 1m):
Bar 1: volume=850,000 / avg=600,000 = ratio 1.42
Bar 2: volume=310,000 / avg=620,000 = ratio 0.50
Bar 3: volume=1,200,000 / avg=640,000 = ratio 1.88
...
(16,000 ratios for 3 months of data)
```

**Step 2 — Find the 80th percentile**

Sort all ratios from lowest to highest. The 80th percentile is the value where
80% of bars are below it.

```
HDFCBANK 1m sorted ratios (sample):
[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.1, 2.4]
                                                                  ^
                                              80th percentile ≈ 1.8
```

This means: "80% of all candles have volume below 1.8× the 20-bar average.
Only the top 20% of candles exceed this — those are genuinely unusual."

**Result:**

```
HDFCBANK: multiplier 1.5 → 1.8  (signals only on top-20% volume days)
ADANIENT: multiplier 1.5 → 2.9  (signals only on genuine spikes for Adani)
TCS:      multiplier 1.5 → 1.5  (happens to match — no change)
```

---

### Estimator 4 — Bollinger Band Width

**File:** `estimate_bb_params()` in signal_param_estimator.py  
**Applies to:** `price_above_bb_upper`, `price_below_bb_lower`, `bb_pct_b_high`,
`bb_pct_b_low`

#### What Bollinger Bands are

Bollinger Bands draw two lines around a moving average:

```
Upper Band = SMA(20) + 2 × standard_deviation_of_last_20_closes
Lower Band = SMA(20) - 2 × standard_deviation_of_last_20_closes
```

The 2 is the `window_dev` parameter. At 2 standard deviations, statistically
95% of all values should fall inside the bands (assuming normal distribution).

#### The problem

At high volatility (ADANIENT, 80%+ annual vol), price breaks the 2-std bands
constantly — it's not a statistical extreme, it's just normal daily movement.
This gives constant false breakout signals.

At low volatility (HDFCBANK stable phase, <25% annual vol), price rarely
touches the 2-std bands — the signal almost never fires, missing real
opportunities.

#### How it's computed from OHLCV

**Step 1 — Compute per-bar return**

```
return_t = (Close_t - Close_{t-1}) / Close_{t-1}
```

**Step 2 — Compute annualized realized volatility**

```
realized_vol = std(last 60 returns) × sqrt(bars_per_year)
```

Bars per year:
- Intraday 1m: 375 bars/session × 250 sessions = 93,750
- Intraday 15m: 25 bars/session × 250 sessions = 6,250
- Daily: 250 bars per year

```
HDFCBANK 1m:
  std(60 returns) = 0.00090  (0.09% per bar)
  bars_per_year   = 93,750
  realized_vol    = 0.00090 × sqrt(93750) = 0.00090 × 306 = 0.275 = 27.5% annual

ADANIENT 1m:
  std(60 returns) = 0.00750  (0.75% per bar)
  realized_vol    = 0.00750 × 306 = 2.295 = 229% annual
```

**Step 3 — Map volatility to band width**

| Annual Volatility | window_dev | Effect |
|---|---|---|
| < 25% | 1.5 | Narrow bands — more signals, stock is calm enough |
| 25% – 50% | 2.0 | Standard (the original Bollinger setting) |
| 50% – 80% | 2.5 | Wider — reduces false breakouts from normal volatility |
| > 80% | 3.0 | Very wide — only extreme moves trigger |

```
HDFCBANK: 27.5% annual → 2.0 (standard, no change from default)
ADANIENT: 229% annual  → 3.0 (very wide)

Formula changes:
  CLOSE > BB_UPPER(20, 2) becomes CLOSE > BB_UPPER(20, 3)
  Only genuine breakouts fire — not just normal daily swings.
```

---

### Estimator 5 — Gap Threshold

**File:** `estimate_gap_params()` in signal_param_estimator.py  
**Applies to:** `gap_up_open`, `gap_down_open`

#### What a gap signal does

```
gap_up_open:  OPEN > previous_CLOSE × 1.005   (opened 0.5% higher)
gap_down_open: OPEN < previous_CLOSE × 0.995  (opened 0.5% lower)
```

#### The problem

Every stock has a "normal open variation" — a small random difference between
yesterday's close and today's open caused by overnight orders and global news.

HDFCBANK: normal overnight variation = 0.1–0.2%
ADANIENT: normal overnight variation = 0.3–0.6%

Using 0.5% for HDFCBANK fires on almost every trading day. Using 0.5% for
ADANIENT misses many genuine gaps.

#### How it's computed from OHLCV

**Step 1 — Compute overnight gap for every trading day**

```
gap_pct = |Open_today - Close_yesterday| / Close_yesterday × 100

Day 1: Open=1705.0, prev_Close=1703.5 → gap = 0.09%
Day 2: Open=1712.0, prev_Close=1709.5 → gap = 0.15%
Day 3: Open=1698.0, prev_Close=1710.0 → gap = 0.70%  ← real gap
...
```

**Step 2 — Find the 75th percentile**

The 75th percentile means "75% of trading days have a gap SMALLER than this.
The top 25% of days have gaps larger — these are genuinely unusual."

```
HDFCBANK sorted gap sizes:
[0.02, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.19, 0.22, 0.28, 0.35, 0.70, ...]
                                                    ^
                              75th percentile ≈ 0.28%

ADANIENT sorted gap sizes:
[0.10, 0.18, 0.25, 0.32, 0.41, 0.50, 0.62, 0.75, 0.90, 1.10, ...]
                                           ^
                    75th percentile ≈ 0.62%
```

**Result:**

```
HDFCBANK: gap_pct 0.5% → 0.28%
  Formula: OPEN > PREV(CLOSE,1) × 1.00500  becomes  OPEN > PREV(CLOSE,1) × 1.00280

ADANIENT: gap_pct 0.5% → 0.62%
  Formula: OPEN > PREV(CLOSE,1) × 1.00500  becomes  OPEN > PREV(CLOSE,1) × 1.00620
```

---

### Signals NOT estimated from OHLCV

These signals are intentionally left at YAML defaults even when OHLCV is
available:

| Signal | Reason |
|---|---|
| `vwap_bullish` / `vwap_bearish` | No parameters — formula is just `CLOSE > VWAP` |
| `inside_bar` | No parameters — it's a candle pattern comparison |
| `macd_bullish_cross` / `macd_bearish_cross` / `macd_positive` | 12/26/9 is universally accepted and empirically stable |
| `opening_range_breakout` | `opening_bars` is session-timing (NSE session structure), not price-derived |
| `roc_positive` / `roc_negative` | Rate of change needs no per-stock calibration |

---

## 5. Layer 3 — Performance Cache

### What it does

After every backtest the system records which signal parameters were used and
what win rate was achieved. Next time the same (signal + stock + timeframe)
combination is requested, the historically best parameters are used instead of
the YAML defaults.

### Where it lives

- Logic: [app/core/signal_performance_cache.py](../app/core/signal_performance_cache.py)
- Storage: `app/core/signal_performance_cache.json` (created automatically)

### The cache file format

```json
{
  "rsi_oversold:TCS.NS:15m": {
    "signal_name":  "rsi_oversold",
    "symbol":       "TCS.NS",
    "timeframe":    "15m",
    "best_params":  {"window": 9, "threshold": 28.0},
    "win_rate":     0.72,
    "total_trades": 75,
    "last_updated": "2026-04-29"
  },
  "sma_cross_up:HDFCBANK.NS:1m": {
    "best_params":  {"window_fast": 30, "window_slow": 86},
    "win_rate":     0.61,
    "total_trades": 12,
    ...
  }
}
```

### Cache key format

```
{signal_name}:{symbol}:{timeframe}
```

Examples:
- `rsi_oversold:TCS.NS:15m`
- `volume_spike:RELIANCE.NS:5m`
- `gap_up_open:ADANIENT.NS:1m`

### When the cache is updated

`record_performance()` is called after every completed backtest. It compares
the new win rate to the stored best:

```
New win rate > stored best win rate  → update the cache entry
Stored total_trades < 10             → update even if win rate is not better
                                       (not enough data yet to trust the old entry)
Everything else                      → keep existing entry
```

### When the cache is used

`get_best_params()` is called from `_default_params()` in retriever.py. It
checks:

```
1. Does an entry exist for this (signal, symbol, timeframe)?
   NO  → return YAML defaults

2. Is total_trades >= 10?
   NO  → return YAML defaults (not enough backtests to trust this entry)

3. Does the entry have valid best_params?
   YES → return cached params (skip YAML defaults entirely)
```

The minimum of 10 trades is a safety threshold. Below 10 trades, the win rate
estimate is too noisy to trust over the validated YAML defaults.

### How to record performance from your code

```python
from app.core.signal_performance_cache import record_performance

# After a backtest completes:
for signal in plan.get("entry", []) + plan.get("exit", []):
    record_performance(
        signal_name  = signal["name"],          # e.g. "rsi_oversold"
        symbol       = builder.format_symbol(), # e.g. "TCS.NS"
        timeframe    = builder.timeframe or "", # e.g. "15m"
        params       = signal.get("params", {}),
        win_rate     = backtest_metrics["win_rate"] / 100.0,  # 0.0 to 1.0
        total_trades = backtest_metrics["total_trades"],
    )
```

---

## 6. Signal Selection Flow

This section explains HOW the system picks which signals to use from the
available pool.

### The candidate pool

The system does NOT evaluate all 28 formula-supported signals for every user.
It first narrows down to a shortlist (6–9 signals per category), then picks the
best one from each category.

**Three categories:**
- **Triggers** — signals that initiate entry ("NOW is the right time")
- **Filters** — signals that confirm the entry ("yes, the setup is valid")
- **Exits** — signals that close the trade ("time to get out")

### Step 1 — Build the base shortlist from YAML

Based on `objective + sentiment + timeframe`, the code looks up a key in the
`candidate_sets` section of `signals_config.yaml`:

```
intraday + bullish            → "intraday_bullish"
intraday + bearish            → "intraday_bearish"
positional + bullish + daily  → "positional_bullish_daily"
positional + bullish + other  → "positional_bullish_default"
positional + bearish + daily  → "positional_bearish_daily"
positional + bearish + other  → "positional_bearish_default"
```

Example for `intraday_bullish`:
```yaml
intraday_bullish:
  triggers: [opening_range_breakout, gap_up_open, rsi_cross_up, inside_bar,
             price_above_bb_upper, ema_cross_up, sma_cross_up,
             macd_bullish_cross, vwap_bullish]
  filters:  [vwap_bullish, high_delivery_volume, volume_spike,
             price_above_ema, is_above_sma, volume_dry_up]
  exits:    [rsi_cross_down, gap_down_open, ema_cross_down,
             sma_cross_down, macd_bearish_cross, rsi_overbought]
```

### Step 2 — Market overrides (push India signals to front)

For `indian_stocks` on `intraday`, India-specific signals are pushed to the
front of the shortlist:

```yaml
market_candidate_overrides:
  indian_stocks:
    intraday:
      bullish:
        prioritize_triggers: [opening_range_breakout, gap_up_open, inside_bar]
        prioritize_filters:  [vwap_bullish, high_delivery_volume, volume_spike]
```

"Push to front" means these signals will be at the top of the shortlist and
will win the scoring competition unless something else scores significantly
higher.

### Step 3 — Experience overrides

Beginner users get simpler, more reliable signals. Expert users get more
advanced signals:

```yaml
experience_candidate_overrides:
  beginner:
    bullish:
      intraday:
        prioritize_triggers: [opening_range_breakout, rsi_cross_up,
                              sma_cross_up, ema_cross_up]
  expert:
    bullish:
      intraday:
        prioritize_triggers: [opening_range_breakout, inside_bar,
                              price_above_bb_upper]
```

### Step 4 — Goal keyword overrides

If the user typed a goal, keywords in it move specific signals to the front:

```yaml
goal_preferences:
  - keywords: [breakout, opening range, continuation]
    bullish:
      prioritize_triggers: [opening_range_breakout, gap_up_open,
                            inside_bar, price_above_bb_upper]
      prioritize_filters:  [vwap_bullish, volume_spike, high_delivery_volume]

  - keywords: [trend, moving average, ema, sma]
    bullish:
      prioritize_triggers: [ema_cross_up, sma_cross_up, macd_bullish_cross]
      prioritize_exits:    [ema_cross_down, sma_cross_down, rsi_cross_down]
```

### Step 5 — LLM interprets the goal

For goal phrases that don't match any keyword (e.g. "I want to ride explosive
moves"), the LLM is called with the full list of available signals and the
user's goal. It returns up to 5 preferred signals and up to 3 signals to avoid.

These preferences are applied to the shortlist BEFORE scoring.

### Step 6 — Scoring and picking

Each candidate signal in the shortlist is scored. Higher score = more likely to
be picked.

The score has several components:

| Component | Points | Description |
|---|---|---|
| Intraday signal bonus | +2.5 | ORB, gap, VWAP, volume signals for intraday |
| India market bonus | +1.4 | ORB, gap, inside_bar, high_delivery_volume |
| Positional signal bonus | +2.0 | MA crosses, RSI cross for positional |
| Daily timeframe bonus | +2.2 | RSI, EMA, MACD on daily+ timeframes |
| Beginner signal bonus | +0.9 | Simple signals for beginners |
| Expert signal bonus | +0.9 | Advanced signals for experts |
| Sentiment match | +1.1 | Signal name/description matches bullish/bearish |
| Sentiment mismatch | -0.4 | Signal direction opposes sentiment |
| KB phrase match | +2.8–5.0 | Signal name found in knowledge base documents |
| Goal breakout bonus | +1.8 | Breakout signals when goal mentions breakout |
| Goal trend bonus | +2.3 | MA signals when goal mentions trend |
| Goal reversal bonus | +1.9 | RSI oversold when goal mentions reversal |
| Avoid RSI penalty | -2.8 | RSI signals when goal says "avoid RSI" |
| Position in shortlist | +0.35/position | Higher position in shortlist = small tiebreak bonus |

All numeric values are configurable in `signals_config.yaml` under
`scoring_weights`. Changing them requires no code deployment.

### Step 7 — Diversity enforcement

The filter signal is constrained to NOT use the same signal family as the
trigger. If the trigger is `ema_cross_up` (EMA family), the filter cannot
be `ema_cross_down` or `price_above_ema`. This forces signal diversity so
entry conditions are not redundant.

Families: `ema`, `sma`, `macd`, `rsi`, `bollinger`, `volume`, `vwap`,
`opening_range`, `gap`, `inside_bar`

### What comes out

```python
{
    "entry": [
        {
            "name": "opening_range_breakout",
            "params": {"opening_bars": 4},   # ← from Layer 1/2/3
            "timeframe": "1m",
            "signal_type": "TRIGGER"
        },
        {
            "name": "vwap_bullish",
            "params": {},
            "timeframe": "1m",
            "signal_type": "FILTER"
        }
    ],
    "exit": [
        {
            "name": "rsi_cross_down",
            "params": {"window": 15, "threshold": 50.0},
            "timeframe": "1m",
            "signal_type": "TRIGGER"
        }
    ],
    "entry_condition": "CLOSE > OPENING_RANGE_HIGH(4) AND CLOSE > VWAP",
    "exit_condition": "RSI(15) < 50.0",
    "signals_used": ["opening_range_breakout", "vwap_bullish", "rsi_cross_down"],
    "signals_available": 28
}
```

---

## 7. Signal Scoring System

### `_profile_bonus()` — context fit score

Called for every candidate signal. Returns a float (positive = good fit,
negative = poor fit).

Inputs: signal name, category, description, and the user's builder state
(objective, sentiment, experience, market, timeframe, goal).

All numeric values (`2.5`, `1.4`, `0.9`, etc.) are read from
`scoring_weights` in `signals_config.yaml`. You can tune them without touching
Python code.

### `_score_signal()` — total score

Calls `_profile_bonus()` and adds KB-document-based scoring.

The KB scoring works by searching the knowledge base documents
(Indicators & Signals.docx, Signals Quick Reference.docx) for mentions of the
signal name or related terms. Signals that appear in the KB documents get
higher scores.

Reference documents (from `_SIGNAL_REFERENCE_DOCS`) get a 1.8× multiplier on
their scoring contribution. All other KB documents get 1.0×.

### `_pick_signal()` — the final selector

Takes the ranked shortlist, applies diversity constraints, and returns the name
of the winning signal.

The final ranking formula:
```
total_score = _score_signal(candidate) + (shortlist_position × 0.35)
```

The position bonus is a small tiebreaker — if two signals have very similar
scores, the one listed earlier in the shortlist (from the YAML candidate set or
overrides) wins.

---

## 8. All Supported Signals and Their Defaults

### What "supported" means

Only signals in `formula_supported_signals` in `signals_config.yaml` can be
selected during strategy planning. A signal must have a formula that the quant
engine's condition parser can evaluate.

### Full list with formula templates and OHLCV estimation support

| Signal | Formula Template | YAML Default Params | OHLCV Estimated? |
|---|---|---|---|
| `ema_cross_up` | `EMA({window_fast}) > EMA({window_slow})` | intraday: 9/21 | Yes (ATR-based) |
| `ema_cross_down` | `EMA({window_fast}) < EMA({window_slow})` | intraday: 9/21 | Yes (ATR-based) |
| `sma_cross_up` | `SMA({window_fast}) > SMA({window_slow})` | intraday: 20/50 | Yes (ATR-based) |
| `sma_cross_down` | `SMA({window_fast}) < SMA({window_slow})` | intraday: 20/50 | Yes (ATR-based) |
| `price_above_ema` | `CLOSE > EMA({window})` | intraday: 20 | Yes (ATR-based) |
| `price_below_ema` | `CLOSE < EMA({window})` | intraday: 20 | Yes (ATR-based) |
| `is_above_sma` | `CLOSE > SMA({window})` | intraday: 50 | Yes (ATR-based) |
| `macd_bullish_cross` | `MACD > 0` | 12/26/9 | No (universal) |
| `macd_bearish_cross` | `MACD < 0` | 12/26/9 | No (universal) |
| `macd_positive` | `MACD > 0` | 12/26/9 | No (universal) |
| `rsi_oversold` | `RSI({window}) < {threshold}` | 14 / 30.0 | Yes (vol-based) |
| `rsi_overbought` | `RSI({window}) > {threshold}` | 14 / 70.0 | Yes (vol-based) |
| `rsi_cross_up` | `RSI({window}) > {threshold}` | 14 / 50.0 | Yes (period only) |
| `rsi_cross_down` | `RSI({window}) < {threshold}` | 14 / 50.0 | Yes (period only) |
| `price_above_bb_upper` | `CLOSE > BB_UPPER({window}, {window_dev})` | 20 / 2 | Yes (vol-based dev) |
| `price_below_bb_lower` | `CLOSE < BB_LOWER({window}, {window_dev})` | 20 / 2 | Yes (vol-based dev) |
| `bb_pct_b_high` | `%B > {threshold}` | 20 / 2 / 0.8 | Yes (vol-based dev) |
| `bb_pct_b_low` | `%B < {threshold}` | 20 / 2 / 0.2 | Yes (vol-based dev) |
| `volume_spike` | `VOL > AVG(VOL,{window}) * {multiplier}` | 20 / 1.5 | Yes (percentile) |
| `volume_dry_up` | `VOL < AVG(VOL,{window}) * {multiplier}` | 20 / 0.5 | No (inverse logic) |
| `high_delivery_volume` | `CLOSE > OPEN AND VOL > AVG(VOL,{window}) * {multiplier}` | 10 / 1.3 | Yes (percentile) |
| `vwap_bullish` | `CLOSE > VWAP` | none | No (no params) |
| `vwap_bearish` | `CLOSE < VWAP` | none | No (no params) |
| `roc_positive` | `(CLOSE - PREV(CLOSE,1)) / PREV(CLOSE,1) > 0` | window: 12 | No |
| `roc_negative` | `(CLOSE - PREV(CLOSE,1)) / PREV(CLOSE,1) < 0` | window: 12 | No |
| `opening_range_breakout` | `CLOSE > OPENING_RANGE_HIGH({opening_bars})` | opening_bars: 4 | No (session-timing) |
| `gap_up_open` | `OPEN > PREV(CLOSE,1) * {gap_factor}` | gap_pct: 0.5% | Yes (percentile) |
| `gap_down_open` | `OPEN < PREV(CLOSE,1) * {gap_factor}` | gap_pct: 0.5% | Yes (percentile) |
| `inside_bar` | `HIGH <= PREV(HIGH,1) AND LOW >= PREV(LOW,1)` | none | No (no params) |

---

## 9. Complete End-to-End Example

**User input:** HDFCBANK, 1m, intraday, bullish, beginner, "quick profit"

### Stage 1 — Signal selection

```
1. Build shortlist from YAML:
   Key = "intraday_bullish"
   Triggers: [ORB, gap_up, rsi_cross_up, inside_bar, bb_upper, ema_cross_up, ...]

2. Market overrides (indian_stocks, intraday, bullish):
   Push to front: [ORB, gap_up, inside_bar]
   Triggers: [ORB, gap_up, inside_bar, rsi_cross_up, bb_upper, ema_cross_up, ...]

3. Experience overrides (beginner, bullish, intraday):
   Push to front: [ORB, rsi_cross_up, sma_cross_up, ema_cross_up]
   Triggers: [ORB, rsi_cross_up, sma_cross_up, ema_cross_up, gap_up, inside_bar, ...]

4. Goal "quick profit":
   Matches keyword "quick" in the scalp preference group:
   Push to front: [ORB, rsi_cross_up, gap_up]
   Triggers: [ORB, rsi_cross_up, gap_up, sma_cross_up, ema_cross_up, ...]

5. Scoring (top candidates):
   opening_range_breakout: +2.5 (intraday) +1.4 (india) +0.9 (beginner)
                           +1.8 (goal breakout) +1.1 (bullish) = 7.7
   rsi_cross_up:           +0.9 (beginner) +1.1 (bullish) +1.2 (query blob) = 3.2
   gap_up_open:            +2.5 (intraday) +1.4 (india) +1.8 (goal) = 5.7

   Winner: opening_range_breakout (score 7.7)

6. Filter selection (avoid same family as ORB = "opening_range"):
   Filters: [vwap_bullish, volume_spike, high_delivery_volume, ...]
   vwap_bullish: +2.5 (intraday) +1.1 (bullish) = 3.6  Winner

7. Exit selection:
   Exits: [rsi_cross_down, gap_down, ema_cross_down, ...]
   rsi_cross_down: +0.9 (beginner) +1.1 (bearish match) = 2.0  Winner

Result:
  entry = ORB (TRIGGER) + vwap_bullish (FILTER)
  exit  = rsi_cross_down
```

### Stage 2 — YAML default parameters applied

```
opening_range_breakout: {opening_bars: 4}    (no intraday override in YAML)
vwap_bullish:           {}                    (no parameters)
rsi_cross_down:         {window: 14, threshold: 50.0}   (default section)

Condition strings built:
  entry_condition = "CLOSE > OPENING_RANGE_HIGH(4) AND CLOSE > VWAP"
  exit_condition  = "RSI(14) < 50.0"
```

### Stage 3 — Performance cache check

```
Cache key: "rsi_cross_down:HDFCBANK.NS:1m"
Cache result: not found (first time)
→ use YAML defaults unchanged
```

### Stage 4 — OHLCV statistical estimation (at backtest time)

```
fetch_ohlcv_records() returns 16,875 rows of HDFCBANK 1m data.

enrich_plan_with_ohlcv() is called:

For rsi_cross_down:
  recent_vol (20 bars) = 0.09%
  long_vol   (60 bars) = 0.10%
  vol_ratio  = 0.9
  new_period = round(14 / sqrt(0.9)) = 15
  (threshold=50 unchanged — crossover signal, not oversold/overbought)
  Updated: {window: 15, threshold: 50.0}

For opening_range_breakout:
  No estimation — session-timing signal
  Unchanged: {opening_bars: 4}

Conditions re-rendered:
  entry_condition = "CLOSE > OPENING_RANGE_HIGH(4) AND CLOSE > VWAP"  (unchanged)
  exit_condition  = "RSI(15) < 50.0"  (updated)
```

### Stage 5 — Backtest runs

```
Quant engine receives:
  entry_condition = "CLOSE > OPENING_RANGE_HIGH(4) AND CLOSE > VWAP"
  exit_condition  = "RSI(15) < 50.0"
  stop_loss_pct   = 2.0  (from market_config.indian_stocks.default_stop_loss)
  take_profit_pct = 5.0  (from market_config.indian_stocks.default_take_profit)
  daily_loss_cap  = 1.0% (from experience_risk_tiers.beginner.daily_loss_cap_pct)
  per_trade_risk  = 1.0% (from experience_risk_tiers.beginner.per_trade_risk_pct)

Result: 12 trades, 58% win rate, 4.2% return
```

### Stage 6 — Cache updated

```
record_performance(
    signal_name  = "rsi_cross_down",
    symbol       = "HDFCBANK.NS",
    timeframe    = "1m",
    params       = {"window": 15, "threshold": 50.0},
    win_rate     = 0.58,
    total_trades = 12,
)

Cache file now contains:
{
  "rsi_cross_down:HDFCBANK.NS:1m": {
    "best_params": {"window": 15, "threshold": 50.0},
    "win_rate": 0.58,
    "total_trades": 12,
    "last_updated": "2026-04-29"
  }
}

Note: 12 < 10 minimum → cache DOES NOT override YAML defaults yet.
After 10 trades this entry is trusted.
```

---

## 10. How to Add a New Signal

To add a completely new signal (e.g. `keltner_breakout_up`) to the system:

### Step 1 — Register the signal function

Create the Python function in
`stretus_knowledge_base/stretus_kb/signals/volatility.py`:

```python
@RuleRegistry.register(
    "keltner_breakout_up",
    formula="CLOSE > EMA({window}) + {multiplier} * ATR({window})",
)
def keltner_breakout_up(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    # compute Keltner Channel upper band and compare to CLOSE
    ...
```

### Step 2 — Add parameter defaults to YAML

In `signals_config.yaml` under `signal_parameters`:

```yaml
keltner_breakout_up:
  intraday:
    window: 14
    multiplier: 1.5
  default:
    window: 20
    multiplier: 2.0
```

### Step 3 — Add to formula_supported_signals

In `signals_config.yaml` under `formula_supported_signals`:

```yaml
formula_supported_signals:
  ...
  - keltner_breakout_up   # ← add here
```

### Step 4 — Add to candidate sets

In `signals_config.yaml` under `candidate_sets`:

```yaml
candidate_sets:
  intraday_bullish:
    triggers: [..., keltner_breakout_up]  # ← add to relevant lists
```

### Step 5 (optional) — Add OHLCV estimation

If this signal has tunable parameters that depend on volatility, add an
estimator branch in `estimate_params_for_signal()` in
`signal_param_estimator.py`:

```python
if name == "keltner_breakout_up":
    return estimate_bb_params(df, yaml_params, objective)  # reuse similar estimator
```

**Zero Python changes needed for steps 2–4.** Only step 1 (new Python function)
and optional step 5 (new estimator) require code.

---

## 11. File Map

| File | Purpose |
|---|---|
| [app/core/signals_config.yaml](../app/core/signals_config.yaml) | All signal defaults, candidate sets, scoring weights, risk tiers |
| [app/core/signals_config_loader.py](../app/core/signals_config_loader.py) | Reads YAML and exposes typed getter functions |
| [app/core/signal_performance_cache.py](../app/core/signal_performance_cache.py) | Layer 3: reads/writes the JSON performance cache |
| [app/core/signal_performance_cache.json](../app/core/signal_performance_cache.json) | The runtime cache file (auto-created, gitignored) |
| [app/core/signal_param_estimator.py](../app/core/signal_param_estimator.py) | Layer 2: all 5 OHLCV-based estimators |
| [app/services/knowledge/retriever.py](../app/services/knowledge/retriever.py) | Signal planning orchestration (`plan_strategy_signals`, `_candidate_sets`, `_score_signal`, `_pick_signal`, `enrich_plan_with_ohlcv`) |
| [stretus_knowledge_base/stretus_kb/registry.py](../stretus_knowledge_base/stretus_kb/registry.py) | Signal registry (maps signal name → Python function + formula template) |
| [stretus_knowledge_base/stretus_kb/signals/](../stretus_knowledge_base/stretus_kb/signals/) | Signal function implementations (momentum, trend, volatility, volume, india_specific) |
| [quant_engine/engine/loader.py](../quant_engine/engine/loader.py) | Parses strategy YAML into `StrategyConfig` for the backtest engine |
| [quant_engine/engine/simulator.py](../quant_engine/engine/simulator.py) | Runs the bar-by-bar backtest simulation |
| [quant_engine/engine/kb_signals.py](../quant_engine/engine/kb_signals.py) | Registry-mode signal evaluation in the quant engine |
