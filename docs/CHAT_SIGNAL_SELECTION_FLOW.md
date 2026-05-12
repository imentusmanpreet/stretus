# Chat Flow: From User Question to Backtest Strategy

## Overview

When a user says **"I want to trade HDFCBANK on 5-minute timeframe with a bullish sentiment"**, Stretus goes through a complete pipeline:

1. **Signal Planning** — Select the best entry/exit signals for this specific setup
2. **Parameter Estimation** — Tune signal parameters to match recent stock behavior
3. **Strategy Assembly** — Build a YAML strategy file with entry/exit conditions
4. **Backtest** — Test the strategy over 2 years of historical data

This document explains each step in detail.

---

## Part 1: The Big Picture (Architecture)

```
User Input
    ↓
    └─→ Intent Extraction (LLM) → structured intent dict
        (goal="I want quick profits" → {style: "scalping", frequency: "high", ...})
    ↓
    └─→ Signal Planner (deterministic)
        ├─ Fetch last 30 days of OHLCV for parameter estimation
        ├─ Select entry signal (hard filter by direction + soft rank by intent fit)
        ├─ Select exit signal (same process)
        ├─ Estimate signal params from recent volatility
        └─ Create signal_plan dict
    ↓
    └─→ Chat shows strategy to user
        Entry: RSI(12) > 50 with Close > SMA(20)
        Exit:  RSI(12) < 30
        SL: 2.5% | TP: 5.0%
    ↓
    └─→ User clicks "Run Backtest"
        ├─ Fetch full 2 years of OHLCV (in 90-day chunks)
        ├─ Re-estimate params on full dataset
        ├─ Generate final YAML strategy
        └─ Run backtest on all candles 2024-01-01 → 2026-03-31
    ↓
    └─→ User sees results
        Total trades: 247 | Win rate: 62% | Return: +18.5%
```

---

## Part 2: Signal Planning in Detail

### Step 1: Intent Extraction

**What happens:** The LLM reads the user's free-text goal and outputs a structured intent.

**User says:**
```
"I want quick profits with low risk. I trade for a few hours and like scalping."
```

**LLM outputs:**
```json
{
  "hold_horizon": "minutes",
  "frequency": "high",
  "profit_size": "small",
  "style": "scalping",
  "risk_appetite": "conservative"
}
```

**Constraints:** The LLM MUST pick from a closed list (see `app/kb/taxonomy.yaml`). If it invents a new value, it's rejected and defaults are used.

**Code:** `app/planner/intent_extractor.py`

---

### Step 2: Signal Selection (The Smart Part)

The planner has **63 signals** registered. It must pick 1 entry, 1 exit, and optionally 1 entry filter.

#### How signals are filtered (Hard Rules)

**Rule 1: Direction Enforcement**
```
User sentiment: BULLISH
Available entry signals: 50+ signals
Filter 1: Keep only signals with role="entry_trigger"
Filter 2: Keep only signals with direction="bullish"
          (remove signals marked direction="bearish")
Survivors: ~8 bullish entry signals
```

Example signals that survive:
- `ema_cross_up` — EMA crossover upward (bullish)
- `rsi_above_50` — RSI above 50 (bullish regime)
- `volume_spike_up` — Volume increasing on green candles (bullish)

Signals that DON'T survive:
- `ema_cross_down` — (bearish, removed)
- `rsi_below_50` — (bearish, removed)

**Rule 2: Timeframe Compatibility**
```
User timeframe: 5m
Signal: ema_cross_up
Does this signal work on 5m? Check signal card's works_best_on list
  works_best_on: ["1m", "5m", "15m", "30m", "1h"]
  → YES, keep it
```

**Rule 3: Avoid Contradictions**
```
Signal: volume_spike_up
Contraindicated_when: {"volatility": "very_high"}
Current ATR%: 3.5% (very high)
  → YES, this signal is contraindicated, REMOVE it
```

#### How signals are ranked (Soft Scoring)

After hard filtering, ~8 candidates remain. Now rank them by how well they fit the user's intent.

**Ranking formula:**
```
score = intent_fit × experience_fit + timeframe_affinity

where:
  intent_fit        = how well the signal matches the intent
                      (scalping strategy likes fast signals like rsi_above_50)
  experience_fit    = how suitable for the user's experience level
                      (beginners get simpler signals like sma_cross_up)
  timeframe_affinity = how much does this signal like 5m timeframe?
                       (0.8 for good match, 0.5 for mediocre)
```

**Example calculation (entry signal):**

| Signal | Intent Fit | Exp Fit | TF Affinity | Score |
|--------|-----------|---------|------------|-------|
| rsi_above_50 | 0.95 (scalping) | 0.9 | 0.9 | **1.75** ← WINNER |
| ema_cross_up | 0.75 | 0.85 | 0.8 | 1.43 |
| volume_spike | 0.7 | 0.6 | 0.8 | 1.16 |

**Winner:** `rsi_above_50` — best fit for a scalping setup on 5m

**Code:** `app/planner/pipeline.py` (the `pick_*` functions)

---

### Step 3: Fetch OHLCV for Parameter Estimation

**Time window for planning:**
```
Config: signal_eval_lookback_days = 30
Fetch: last 30 days from today
Example: 2026-04-01 → 2026-05-01 (today)
Candles: ~11,250 one-minute bars for HDFCBANK
```

**Why 30 days?**
- Enough data to estimate volatility (20+ bars minimum)
- Small enough to fetch in 1 request (no rate limit errors)
- Recent enough to match current stock behavior

**Code:** `app/services/chat/chat_service.py` (_signal_eval_window function)

---

### Step 4: Parameter Estimation (3-Tier System)

Every signal has **default parameters**. For `rsi_above_50`, the default is `window=14, threshold=50`.

These defaults are universal (set decades ago for US daily charts). We tune them using 3 tiers:

#### Tier 1: Performance Cache
```
Question: Has (rsi_above_50, HDFCBANK, 5m) been backtested before?

If YES and ≥ 10 trades recorded:
  → Use the learned params
  → Example: RSI(window=11) won 8/10 times
  → USE RSI(window=11) ← learned from history

If NO or < 10 trades:
  → Go to Tier 2
```

**Code:** `app/core/signal_performance_cache.py`

#### Tier 2: Statistical Estimation from OHLCV
```
Question: How volatile is HDFCBANK on 5m right now?

Measure the recent 30 days:
  ├─ ATR (Average True Range) = typical price swing per bar
  │   Example: 0.85 points (stock price ≈ 1800, so 0.047%)
  ├─ Close behavior = how much bars usually move
  ├─ Volume patterns = spike threshold for volume signals
  └─ Return volatility = how fast prices change

For RSI:
  ├─ If stock is very volatile (fast swings)
  │   → Use faster RSI like window=11 (reacts quicker)
  ├─ If stock is stable (slow swings)
  │   → Use slower RSI like window=18 (filters noise)
  └─ Output: window=12 ← statistically estimated

For SMA:
  ├─ Fast SMA: SMA(5) or SMA(8)
  ├─ Slow SMA: SMA(18) or SMA(21)
  └─ Chosen based on volatility

For Volume Spike:
  └─ spike_multiplier = 1.8x recent average volume
```

**How it works:**
```python
estimate_params_for_signal(
  signal_name="rsi_above_50",
  df=last_30_days_ohlcv,
  objective="intraday",
  timeframe="5m",
  yaml_params={"window": 14, "threshold": 50}  ← defaults
) → {"window": 12, "threshold": 50}  ← estimated
```

**Why?** Different stocks have different volatility. ADANIENT is 2x more volatile than HDFCBANK, so its optimal RSI window is different.

**Code:** `app/core/signal_param_estimator.py`

#### Tier 3: YAML Card Defaults
```
If no cache and no OHLCV data (or < 20 bars):
  → Use hardcoded defaults
  → window=14, threshold=50
```

#### SL/TP Volatility Scaling

Stop-loss and take-profit are also scaled:

```
Base SL = 2.0% (from user's risk tier)
Base TP = 4.0%

Measure ATR% from recent data:
  ATR% = 0.6% (current volatility)
  Baseline ATR% = 1.5% (historical median)
  vol_mult = 0.6 / 1.5 = 0.4x (stock is stable)

Scaled SL = 2.0% × 0.4 = 0.8%  ← tighter on stable stocks
Scaled TP = 4.0% × 0.4 = 1.6%

(Clamped to 0.5x–3.0x to prevent extreme values)
```

**Result for the user:**
```
Entry: RSI(12) > 50
Exit:  RSI(12) < 30
SL: 0.8%
TP: 1.6%
```

**Code:** `app/planner/param_resolver.py`

---

## Part 3: Formula Rendering

Once signal selection is done, we need to show the user a readable condition string.

### Formula Templates

Every signal has a formula template:

```python
# From app/kb/signals/rsi_oversold.yaml
formula: "RSI({window}) < {threshold}"
params: {"window": 14, "threshold": 30}
```

### Rendering

The planner **renders** the template with actual params:

```python
render_formula("rsi_oversold", {"window": 12, "threshold": 25})
  ↓
"RSI({window}) < {threshold}".format(window=12, threshold=25)
  ↓
"RSI(12) < 25"
```

### Complex Conditions

Entry might use multiple signals:

```
Entry Signal: rsi_above_50
Entry Filter: close_above_sma

Rendered:
  Entry trigger: "RSI(12) > 50"
  Entry filter:  "Close > SMA(20)"
  Combined:      "RSI(12) > 50 AND Close > SMA(20)"

Exit Signal: rsi_below_30
Rendered:
  Exit trigger: "RSI(12) < 30"
```

**Code:** `app/planner/formulas.py`

---

## Part 4: YAML Strategy Generation & Enrichment

### What is the YAML?

A strategy is a YAML file that tells the quant engine exactly what to do:

```yaml
strategy:
  symbol: HDFCBANK
  timeframe: 5m
  objective: intraday
  
  entry:
    trigger:
      name: rsi_above_50
      params: {window: 12, threshold: 50}
    filters:
      - name: close_above_sma
        params: {window: 20}
  
  exit:
    trigger:
      name: rsi_below_30
      params: {window: 12, threshold: 30}
  
  sl_tp:
    stop_loss_pct: 0.8
    take_profit_pct: 1.6
  
  risk:
    max_risk_per_trade_pct: 1.0
    max_open_positions: 3
```

### Two Phases of Enrichment

#### Phase 1: Planning Phase (30-day data)
```
signal_plan → enrich_plan_with_ohlcv(last_30_days)
  ├─ Re-estimate params on 30 days
  └─ Generate YAML
```

#### Phase 2: Backtest Phase (2-year data)
```
Before backtest actually runs:
  └─ enrich_plan_with_ohlcv(all_2_years)
     ├─ Re-estimate params on full 2 years (more accurate)
     └─ Regenerate YAML with final params
```

**Why two phases?**
- Phase 1 (30 days): Show the user quickly what the strategy looks like
- Phase 2 (2 years): Before running backtest, ensure params are calibrated on full historical window

**Code:** `app/planner/strategy_assembler.py` (enrich_plan_with_ohlcv function)

---

## Part 5: Backtest Flow

### Step 1: Fetch OHLCV (Chunked)

**Config:**
```
BACKTEST_MARKET_DATA_FROM_UTC = "2024-01-01T00:00:00Z"
BACKTEST_MARKET_DATA_TO_UTC   = "2026-03-31T23:59:59Z"
backtest_fetch_chunk_days = 90
```

**What happens:**
```
Full range: 2024-01-01 → 2026-03-31 (821 days)
Split into 90-day chunks:

Chunk 1:  2024-01-01 → 2024-03-31  ┐
Chunk 2:  2024-03-31 → 2024-06-29  │
Chunk 3:  2024-06-29 → 2024-09-27  │
...                                 ├─ Sequential fetch
Chunk 9:  2025-12-21 → 2026-03-21  │ with 0.3s pause
Chunk 10: 2026-03-21 → 2026-03-31  ┘ between chunks

Each chunk: ~11,250 one-minute bars (for 1m timeframe)
All chunks merged + deduplicated by timestamp
Final: ~262,500 one-minute bars total
```

**Why chunks?**
- Fetching all 2 years in one request = HTTP 429 (rate limit)
- Fetching in 90-day chunks = 10 small requests that succeed
- Pauses between chunks prevent overwhelming the server

**Code:** `app/services/backtest/market_data.py` (_fetch_single_chunk, _build_date_chunks)

### Step 2: Re-enrich Params on Full Dataset

```
Before running backtest:
  enrich_plan_with_ohlcv(all_2_year_data)
  
Re-estimate RSI window, SMA periods, etc.
on full 2 years (much more data than 30 days)

Updated YAML is regenerated with final params
```

### Step 3: Pass to Quant Engine

```python
await run_quant_backtest_sync(
  yaml_path="./strategies/hdfcbank_5m_strategy.yaml",
  ohlcv_data=all_2_year_candles,
  run_config={...}
)
```

### Step 4: Quant Engine Enforcement

Inside the quant engine (`quant_engine/engine/runner.py`):

```python
_enforce_market_data_window(dataframe, market_data_request):
  ├─ Trims dataframe to exactly 2024-01-01 → 2026-03-31
  ├─ Checks that all candles fall within this window
  └─ If not → ERROR
```

### Step 5: Backtest Simulation

For every candle from 2024-01-01 to 2026-03-31:

```
if entry_condition_met and no_open_position:
  → open_position()
  
if exit_condition_met and position_open:
  → close_position()
  → record_trade(entry_price, exit_price)
  
if stop_loss_hit:
  → close_position()
  
if take_profit_hit:
  → close_position()

Compute metrics:
  total_trades, win_rate, avg_return, sharpe, etc.
```

### Step 6: Results

```
Results object:
{
  "pass": true,
  "metrics": {
    "total_trades": 247,
    "winning_trades": 153,
    "losing_trades": 94,
    "win_rate": 61.9,
    "total_return": 18.5,
    "avg_trade_return": 0.075,
    "sharpe_ratio": 1.2,
    ...
  },
  "backtest_ref_id": "abc-123-def"
}
```

---

## Part 6: Key Configs (Where to Tweak)

### Chat Planning Phase

**File:** `app/core/config.py`

```python
signal_eval_lookback_days = 30
  # Days of recent data for param estimation in planning phase
  # Lower = faster but less accurate params
  # Higher = more accurate but slower and more 429 errors
```

### Backtest Phase

**File 1:** `quant_engine/engine/config.py`
```python
BACKTEST_MARKET_DATA_FROM_UTC = "2024-01-01T00:00:00Z"
BACKTEST_MARKET_DATA_TO_UTC   = "2026-03-31T23:59:59Z"
  # The exact date range tested
  # Only edit if you want different historical window
```

**File 2:** `app/core/config.py`
```python
backtest_fetch_chunk_days = 90
  # Size of each chunk when fetching OHLCV
  # Lower = avoid 429 but more round-trips
  # Higher = fewer requests but risk 429
```

### Param Estimation

**File:** `app/core/signal_param_estimator.py`

```python
_MIN_BARS_FOR_VOL = 20
  # Minimum bars needed to estimate params
  # If actual data < 20 bars, fall back to YAML defaults
```

---

## Part 7: Example Walkthrough

### User Input
```
User: "I want to trade HDFCBANK on 5-minute chart. 
        I have 1 year experience and want to scalp 
        for quick small profits with conservative risk."
```

### Step 1: Intent Extraction
```
LLM reads goal and outputs:
{
  "hold_horizon": "minutes",
  "frequency": "high",
  "profit_size": "small",
  "style": "scalping",
  "risk_appetite": "conservative"
}
```

### Step 2: Signal Selection
```
Sentiment inferred: BULLISH
Timeframe: 5m
Experience: intermediate
Objective: intraday

Hard filters:
  ├─ Keep entry_trigger signals only
  ├─ Keep bullish direction only
  └─ Remove if contraindicated

Soft ranking on ~8 survivors:
  Winner entry: rsi_above_50 (score 1.75)
  Winner filter: close_above_sma (score 1.62)
  Winner exit: rsi_below_30 (score 1.68)
```

### Step 3: Fetch 30 Days & Estimate Params
```
Fetch: 2026-04-01 → 2026-05-01 (11,250 candles)

Measure volatility:
  ATR% = 0.7% (fairly volatile)
  → RSI window = 12 (faster response)
  → SMA period = 18 (faster moving average)

Measure SL/TP:
  vol_mult = 0.7 / 1.5 = 0.47
  SL = 2.0% × 0.47 = 0.94%
  TP = 4.0% × 0.47 = 1.88%
```

### Step 4: Show Strategy to User
```
ENTRY:
  Trigger: RSI(12) > 50
  Filter:  Close > SMA(18)
  → Combined: RSI(12) > 50 AND Close > SMA(18)

EXIT:
  Trigger: RSI(12) < 30

RISK:
  SL: 0.94%
  TP: 1.88%
  Max 3 concurrent positions
```

### Step 5: User Clicks "Run Backtest"
```
Fetch 2024-01-01 → 2026-03-31 in 10 chunks
Re-estimate params on full 2-year data:
  RSI window: 12 (confirmed)
  SMA period: 18 (confirmed)

Generate final YAML and pass to quant engine
Quant engine simulates all 262,500 one-minute candles
```

### Step 6: Results
```
✅ Backtest Complete

Total Trades: 247
Winning Trades: 153 (61.9% win rate)
Losing Trades: 94
Total Return: +18.5%
Avg Trade Return: +0.075%
Sharpe Ratio: 1.2

(These are actual results from the 2-year backtest period)
```

---

## Part 8: Formulas Reference

### Common Signal Formulas

**RSI-based:**
```
rsi_above_50:      RSI({window}) > 50
rsi_below_50:      RSI({window}) < 50
rsi_oversold:      RSI({window}) < {threshold}
rsi_overbought:    RSI({window}) > {threshold}
```

**SMA/EMA-based:**
```
close_above_sma:   Close > SMA({window})
close_below_sma:   Close < SMA({window})
ema_above_sma:     EMA({window_fast}) > SMA({window_slow})
ema_cross_up:      EMA({window_fast}) crossed above SMA({window_slow})
ema_cross_down:    EMA({window_fast}) crossed below SMA({window_slow})
```

**Volume-based:**
```
volume_spike:      Volume > {spike_multiplier} × Avg Volume
volume_increasing: Current Volume > Previous 5 candles' avg
```

**Volatility-based:**
```
atr_above_threshold: ATR({window}) > {threshold_pct}%
```

All formulas support **parameter substitution**:
```
Template: "RSI({window}) > {threshold}"
Params:   {"window": 12, "threshold": 50}
Result:   "RSI(12) > 50"
```

**Code:** `app/planner/formulas.py`, `app/kb/signals/` (each signal YAML card)

---

## Part 9: Troubleshooting

### Problem: "HTTP 429 Too Many Requests"

**During planning (signal selection):**
- Increase `signal_eval_lookback_days` won't help (already 30 days = small)
- Problem is likely network issue

**During backtest:**
- Lower `backtest_fetch_chunk_days` from 90 to 60 or 30
- This makes chunks smaller (fewer candles per request)

### Problem: "Insufficient bars for parameter estimation"

**Cause:** Less than 20 bars of OHLCV data

**Solution:**
- Check if `signal_eval_lookback_days` is too small
- Increase it to 40 or 60 (takes longer but gets more data)

### Problem: "Signal params look generic, not tuned to my stock"

**Check:**
- Did param estimation succeed? (Look for `param_resolver|source=ohlcv_estimator` in logs)
- If source is `card_default`, that means:
  - Not enough OHLCV bars, or
  - Performance cache had no prior data, or
  - Estimator failed silently (check logs for `param_resolver|estimator_failed`)

---

## Part 10: Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      CHAT FLOW ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input                                                     │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Intent Extractor (LLM)                                   │   │
│  │ Input: Free-text goal                                    │   │
│  │ Output: {style, frequency, hold_horizon, ...}            │   │
│  │ Files: app/planner/intent_extractor.py                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Signal Planner (Deterministic)                           │   │
│  │ Inputs: sentiment, timeframe, intent, experience         │   │
│  │ Step 1: Hard filter (direction, timeframe, contra...)    │   │
│  │ Step 2: Soft rank survivors (intent + exp + tf affinity) │   │
│  │ Step 3: Pick top entry, top exit, optional filter        │   │
│  │ Files: app/planner/pipeline.py                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OHLCV Fetch #1 (Planning Phase)                          │   │
│  │ Config: signal_eval_lookback_days = 30                   │   │
│  │ Data: Last 30 days (e.g., 11,250 one-minute bars)       │   │
│  │ Files: app/services/chat/chat_service.py                 │   │
│  │         app/services/backtest/market_data.py             │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Param Estimator (3-Tier)                                 │   │
│  │ Tier 1: Check perf cache (learned params)                │   │
│  │ Tier 2: Estimate from OHLCV (volatility analysis)        │   │
│  │ Tier 3: Fall back to YAML defaults                       │   │
│  │ Files: app/planner/param_resolver.py                     │   │
│  │         app/core/signal_param_estimator.py               │   │
│  │         app/core/signal_performance_cache.py             │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Formula Rendering                                         │   │
│  │ Input: signal name + params                              │   │
│  │ Output: readable condition string                         │   │
│  │ Example: "RSI(12) > 50 AND Close > SMA(18)"              │   │
│  │ Files: app/planner/formulas.py                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ YAML Generation & User Display                           │   │
│  │ Strategy shown to user for approval                       │   │
│  │ Files: app/core/main.py (generate_yaml)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  User clicks "Run Backtest"                                     │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OHLCV Fetch #2 (Backtest Phase) — CHUNKED               │   │
│  │ Config: backtest_fetch_chunk_days = 90                   │   │
│  │ Date range: 2024-01-01 → 2026-03-31 (821 days)          │   │
│  │ Strategy: Split into 10 chunks, fetch sequentially       │   │
│  │ Data: Full 2 years (e.g., 262,500 one-minute bars)       │   │
│  │ Files: app/services/backtest/market_data.py              │   │
│  │         app/services/chat/chat_service.py                │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Re-enrich Params on Full Data                            │   │
│  │ Re-estimate all signal params on 2-year data             │   │
│  │ (More accurate than 30-day estimates)                     │   │
│  │ Files: app/planner/strategy_assembler.py                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Quant Engine Backtest                                    │   │
│  │ Input: YAML strategy + 2-year OHLCV data                 │   │
│  │ Process: Simulate every candle in date range             │   │
│  │ Output: metrics (win rate, total return, sharpe, etc.)   │   │
│  │ Files: quant_engine/engine/runner.py                     │   │
│  │         quant_engine/engine/evaluator.py                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Results Persistence                                       │   │
│  │ Store results in database                                 │   │
│  │ Store performance data for future param caching           │   │
│  │ Files: app/services/backtest/result_store.py             │   │
│  │         app/core/signal_performance_cache.py             │   │
│  └──────────────────────────────────────────────────────────┘   │
│    ↓                                                             │
│  Display Results to User                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

**Key takeaways:**

1. **Two OHLCV fetches:**
   - Planning (30 days) — quick param estimation
   - Backtest (2 years, chunked) — full historical test

2. **Three param sources (in priority order):**
   - Performance cache (learned from past backtests)
   - Statistical estimator (from recent volatility)
   - YAML defaults (universal fallback)

3. **Signal selection is smart:**
   - Hard filters eliminate incompatible signals
   - Soft ranking picks the best match for the user's intent

4. **Formulas are templated:**
   - Abstract: `RSI({window}) > {threshold}`
   - Concrete: `RSI(12) > 50` (after param substitution)

5. **Configs you can tweak:**
   - `signal_eval_lookback_days` — planning phase data freshness
   - `backtest_fetch_chunk_days` — backtest chunking (avoid 429)
   - `BACKTEST_MARKET_DATA_FROM_UTC/TO_UTC` — historical range

---

**Questions?** Check the corresponding Python files for implementation details.
