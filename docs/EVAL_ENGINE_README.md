# Strategy Execute & Eval Engine — Complete Technical Reference

> **Scope:** This document covers every component of the Stretus AI strategy evaluation and execution engine: live signal evaluation (`POST /strategy/evaluate/execute`), the quant backtest engine (`quant_engine/`), market data fetching, risk management, account management, indicator computation, trade simulation, and all metrics/scoring formulas.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Market Data Service](#3-market-data-service)
4. [Strategy Evaluator — Live Execution Pipeline](#4-strategy-evaluator--live-execution-pipeline)
5. [Rule Engine — Signal Evaluation](#5-rule-engine--signal-evaluation)
6. [Risk Manager](#6-risk-manager)
7. [Account Manager](#7-account-manager)
8. [Trade Manager — Bracket Orders](#8-trade-manager--bracket-orders)
9. [Reference Data Service](#9-reference-data-service)
10. [Risk Execution Config Service](#10-risk-execution-config-service)
11. [Execution Cache](#11-execution-cache)
12. [Quant Backtest Engine](#12-quant-backtest-engine)
13. [Technical Indicators — All Formulas](#13-technical-indicators--all-formulas)
14. [Trade Simulator — All Formulas](#14-trade-simulator--all-formulas)
15. [Backtest Metrics — All Formulas](#15-backtest-metrics--all-formulas)
16. [Assessment & Grading System](#16-assessment--grading-system)
17. [Market Phase & Alignment Classifier](#17-market-phase--alignment-classifier)
18. [Configuration Reference — All Variables](#18-configuration-reference--all-variables)
19. [Environment Variables](#19-environment-variables)
20. [Data Flow Summary](#20-data-flow-summary)

---

## 1. System Overview

The platform has **two evaluation surfaces**:

| Surface | Path | Purpose |
|---|---|---|
| **Live Execution Evaluator** | `app/services/execution/` | Real-time per-bar signal evaluation on live market data via Upstox API. Returns bracket order instructions. Stateless — never submits orders. |
| **Quant Backtest Engine** | `quant_engine/` | Historical OHLCV-based strategy simulation. Returns P&L metrics, grades, and market phase analysis. Runs in a separate FastAPI service. |

Both surfaces share the same signal logic. The live evaluator uses `stretus_kb.RuleRegistry` for signal evaluation; the backtest engine uses string-based condition expressions evaluated by `engine/conditions.py`.

---

## 2. Architecture Diagram

```
POST /strategy/evaluate/execute
            │
            ▼
    StrategyEvaluator  (app/services/execution/strategy_evaluator.py)
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 1: Load strategy config (DB by strategy_id OR inline body) │
    │  Step 2: Resolve Upstox instrument key (ref_data.equities)       │
    │  Step 3: Fetch market data (candles + LTP + circuit limits)      │
    │  Step 4a: Build InstrumentDefaults (ref_data.system_configs)     │
    │  Step 4b: Load execution state + risk config from DB             │
    │  Step 5: EXIT PHASE  — SL / TP / exit signal check              │
    │  Step 6: ENTRY PHASE — signal → risk → account → bracket order  │
    └──────────────────────────────────────────────────────────────────┘
            │                                │
            ▼                                ▼
    MarketDataService               RuleEngine
    (Upstox v2 API)                 (stretus_kb.RuleRegistry)
            │                                │
            ▼                                ▼
    RiskManager                     AccountManager
    (SL/TP/size calc)               (margin/position checks)
            │
            ▼
    TradeManager
    (builds BracketOrder)

POST /run  (quant_engine)
            │
            ▼
    run_backtest()  (quant_engine/engine/runner.py)
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. Parse YAML strategy config                                    │
    │  2. Load + trim OHLCV to backtest window                         │
    │  3. Data sufficiency check                                        │
    │  4. Compute all indicators (add_all_indicators)                   │
    │  5. simulate_trades() — bar-by-bar loop                          │
    │  6. calculate_metrics() → build_assessment() → grading           │
    └──────────────────────────────────────────────────────────────────┘
```

---

## 3. Market Data Service

**File:** `app/services/execution/market_data_service.py`

### 3.1 Data Source

All live execution market data comes from a **single source**: Upstox v2 API.

```
Base URL: settings.market_data_url  (default: https://api.upstox.com/v2)
Auth:     Bearer {settings.upstox_access_token}
```

Backtest historical data comes from a **separate** source: `settings.historical_data_url` (ngrok tunnel to a local data server). The market data service **never** touches the historical data URL.

---

### 3.2 Candle Fetching

**Endpoint:**
```
GET {MARKET_DATA_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
```

**Upstox only supports these native intervals:**
- `1minute`
- `30minute`
- `day`
- `week`
- `month`

**Strategy timeframes are handled by fetching the nearest finer-grain interval, then resampling:**

| Strategy Timeframe | Upstox Fetch Interval | Pandas Resample Rule |
|---|---|---|
| `1m` | `1minute` | none (passthrough) |
| `3m` | `1minute` | `3min` |
| `5m` | `1minute` | `5min` |
| `10m` | `1minute` | `10min` |
| `15m` | `1minute` | `15min` |
| `30m` | `30minute` | none (passthrough) |
| `45m` | `30minute` | `45min` |
| `1h` | `30minute` | `60min` |
| `2h` | `30minute` | `120min` |
| `4h` | `30minute` | `240min` |
| `1d` | `day` | none (passthrough) |

**Resample aggregation rules (OHLCV):**
```python
Open   = first
High   = max
Low    = min
Close  = last
Volume = sum
```

**Date window calculation:**

The service computes how many calendar days to fetch based on lookback bars requested:

```python
fetch_minutes = fi_mins  # 1, 30, or 1440 depending on fetch interval
total_mins    = lookback_candles × fi_mins × 2      # 2× safety buffer
trading_days  = max(2, (total_mins ÷ 375) + 1)     # 375 min/day = NSE session length
calendar_days = trading_days × 2 + 14              # weekends + holidays buffer

to_date   = today
from_date = today - calendar_days
```

**Variables:**
| Variable | Description |
|---|---|
| `lookback_candles` | Number of target-timeframe bars needed by the indicator engine |
| `fi_mins` | Minutes per fetch interval bar (1, 30, or 1440) |
| `ratio` | `tf_mins ÷ fi_mins` — how many fetch bars produce 1 strategy bar |
| `fetch_lookback` | `lookback_candles × ratio` — raw bars to request |
| `market_mins_per_day` | 375 (hardcoded NSE session: 9:15–15:30) |

**Response format:**
Upstox returns candles newest-first as arrays: `[timestamp, open, high, low, close, volume, oi]`. The service reverses them to chronological order and builds a UTC-indexed DataFrame.

**Final output:** `df.tail(lookback)` — only as many bars as requested.

---

### 3.3 LTP (Last Traded Price) Fetching

**Endpoint:**
```
GET {MARKET_DATA_URL}/market-quote/ltp?instrument_key={key}
```

**Response shape:**
```json
{
  "status": "success",
  "data": {
    "NSE_EQ:RELIANCE": { "last_price": 2345.60 }
  }
}
```

**Fallback on error:** scans the candle cache across all timeframes (`1m`, `5m`, `15m`, `30m`, `1h`, `1d`) and returns the last `Close` from the first non-empty DataFrame found.

---

### 3.4 Circuit Limits Fetching

**Endpoint:**
```
GET {MARKET_DATA_URL}/market-quote/quotes?instrument_key={key}
```

Parses `upper_circuit_limit` / `lower_circuit_limit` (or aliases `upper_circuit` / `lower_circuit`) from the response. Returns `None` on error — the evaluator falls back to DB-derived percentage defaults.

---

### 3.5 Instrument Key Resolution

**Priority:**
1. DB lookup via `ref_data.equities → equity_data_source_mappings` (returns ISIN-based key, e.g. `NSE_EQ|INE002A01018`)
2. Best-effort fallback: constructed as `NSE_EQ|{TICKER}` (works for most NSE equities via the market-quote endpoint)

**Symbol normalization:**
```
RELIANCE.NS  → RELIANCE
reliance.ns  → RELIANCE
NSE:RELIANCE → RELIANCE
Reliance     → RELIANCE
```

---

### 3.6 Lookback Calculation

**File:** `app/services/execution/indicator_engine.py`

The lookback (number of candles to fetch) is computed from all signal rules in the strategy:

```python
compute_lookback(rules) → max_period_across_all_rules × 2
```

Each rule's `params` is scanned for any key whose value is a positive integer (period parameter). The largest such value across all entry trigger, entry filters, exit trigger, and exit filters is taken as `max_window`. The lookback is `max(50, max_window × 2)`.

---

## 4. Strategy Evaluator — Live Execution Pipeline

**File:** `app/services/execution/strategy_evaluator.py`  
**Entry point:** `POST /api/v1/strategy/evaluate/execute`  
**Class:** `StrategyEvaluator` (one instance per request)

### 4.1 Two Loading Modes

**Mode 1 — DB strategy** (when `strategy_id` is provided):
- Fetches strategy row from `ai_strategy.strategies`
- Fetches global risk config from `ref_data.risk_execution_configs`
- Fetches execution state from `ai_strategy.execution_states`
- Constructs `StrategyConfigPayload` from JSONB `strategy_config` field

**Mode 2 — Inline** (when `strategy_id` is absent):
- Uses request body `strategy_config` and `execution_state` directly
- Loads global risk config as baseline; inline values overlay it

### 4.2 Execution Order

Every request follows these steps in order (no step can be skipped):

```
1. Load strategy config
2. Resolve Upstox instrument key (ref_data)
3. Fetch market data: candles + LTP + circuit limits
4a. Build InstrumentDefaults (tick/lot from ref_data.system_configs; circuits from live API or pct default)
4b. Load execution state (DB row or synthetic from inline payload)
5. EXIT PHASE: for each open position → SL hit? → TP hit? → exit signal?
6. ENTRY PHASE: entry signal? → risk calc → account check → build bracket order
```

### 4.3 Effective Config Resolution

In Mode 1 (DB strategy):
- `effective_risk_config` = `runtime_risk_config` (the ref_data global config)
- `effective_exec_state` = the DB `ExecutionState` row (or synthetic fallback if missing)

In Mode 2 (inline):
- `effective_risk_config` = `_synthetic_risk_config`: a `SimpleNamespace` built from `strategy_cfg.sl_tp` + `strategy_cfg.risk` (inline values override ref_data baseline)
- `effective_exec_state` = `_synthetic_exec_state`: built from `exec_payload.capital` + `strategy_cfg.risk` limits

**Key variables resolved:**
| Variable | Source | Description |
|---|---|---|
| `symbol` | strategy_cfg | Trading symbol |
| `timeframe` | strategy_cfg | Bar timeframe |
| `strategy_type` | strategy_cfg | `"intraday"` or `"positional"` |
| `available_margin` | exec_state_payload | Current available margin in ₹ |
| `open_positions` | exec_state_payload | List of currently open positions |
| `bars_since_last_trade` | exec_state_payload | Candles since last trade (for cooldown) |
| `capital` | exec_state_payload | Total capital in ₹ |
| `ltp` | MarketDataService | Last traded price |
| `lookback` | indicator_engine | Number of candles to fetch |
| `instrument` | ref_data_service | Tick size, lot size, circuit limits |

### 4.4 Exit Phase Logic

For **each open position** in `open_positions`, checks in this order:

1. **Stop Loss:** `ltp <= position.stop_loss_price` → exit reason `"stop_loss"`
2. **Take Profit:** `ltp >= position.take_profit_price` → exit reason `"take_profit"`
3. **Exit Signal:** evaluates `strategy_cfg.exit` (trigger + filters via RuleEngine) → exit reason `"exit_signal"`

If any exit condition fires, a `ExitInstruction` is built via `TradeManager.build_exit_instruction()` and the response returns `action = "exit_triggered"`. The entry phase is **skipped** entirely when there are exits.

**Unrealized P&L displayed in logs:**
```python
unrealised_PnL = (ltp - entry_price) × quantity
```

### 4.5 Entry Phase Logic

1. Evaluate entry signal via `RuleEngine.evaluate_entry()` (trigger AND all filters must pass)
2. If no signal → return `action = "no_action"`
3. Run `RiskManager.calculate()` — computes SL/TP prices and position size
4. If risk fails → return `action = "no_action"`
5. Run `AccountManager.validate()` — checks margin / positions / cooldown
6. If account fails → return `action = "no_action"`
7. Build bracket order via `TradeManager.build_bracket_order()` → return `action = "entry_created"`

### 4.6 Response Actions

| `action` | Meaning |
|---|---|
| `entry_created` | Signal fired, risk passed, account passed — bracket order generated |
| `exit_triggered` | One or more open positions have an exit condition met |
| `no_action` | No signal, or signal fired but blocked by risk/account checks |

---

## 5. Rule Engine — Signal Evaluation

**File:** `app/services/execution/rule_engine.py`  
**Class:** `RuleEngine`

### 5.1 Signal Block Structure

Both entry and exit are evaluated from a "block":
```json
{
  "trigger": { "type": "RSI_OVERSOLD", "params": { "period": 14, "threshold": 30 } },
  "filters": [
    { "type": "ABOVE_EMA", "params": { "period": 50 } },
    { "type": "VOLUME_SPIKE", "params": { "multiplier": 1.5 } }
  ]
}
```

### 5.2 Evaluation Logic

**Entry:** `trigger AND filter[1] AND filter[2] AND ... filter[n]`
- If trigger fails → filters are skipped (short-circuit)
- Any filter failure → entry blocked

**Exit:** same AND logic — trigger AND all filters must fire

### 5.3 Underlying Signal Dispatch

Each rule is dispatched via `stretus_kb.RuleRegistry.evaluate({"name": rule_type, "params": params}, df)`. This calls the registered signal function with the full OHLCV DataFrame and returns a boolean evaluated against the **last row** of the DataFrame.

### 5.4 SL/TP Price Checks

```python
check_stop_loss(ltp, sl_price)  → ltp <= sl_price
check_take_profit(ltp, tp_price) → ltp >= tp_price
```

---

## 6. Risk Manager

**File:** `app/services/execution/risk_manager.py`  
**Class:** `RiskManager` (stateless, one instance per request)  
**Input:** `RiskInput` dataclass  
**Output:** `RiskOutput` dataclass

### 6.1 Input Variables

| Variable | Type | Source | Description |
|---|---|---|---|
| `entry_price` | `float` | LTP from market data | Current price at which the trade would be entered |
| `risk_config` | `object` | `effective_risk_config` | Holds `stop_loss_pct`, `take_profit_pct` |
| `exec_state` | `object` | `effective_exec_state` | Holds `capital`, `max_risk_per_trade_pct`, `min_trade_value`, `max_position_capital_pct` |
| `instrument` | `InstrumentDefaults` | ref_data | Holds `tick_size`, `lot_size`, `upper_circuit`, `lower_circuit` |
| `circuit_threshold_pct` | `float` | settings | Fraction of upper circuit at which trading is blocked (default `0.98`) |

### 6.2 Safe Default Values

Used only when ORM rows are `None` (should not happen in production):

| Default Constant | Value | Applies To |
|---|---|---|
| `_DEFAULT_STOP_LOSS_PCT` | `2.0` | stop_loss_pct |
| `_DEFAULT_TAKE_PROFIT_PCT` | `5.0` | take_profit_pct |
| `_DEFAULT_TICK_SIZE` | `0.05` | tick_size |
| `_DEFAULT_LOT_SIZE` | `1` | lot_size |
| `_DEFAULT_CAPITAL` | `100,000.0` | capital |
| `_DEFAULT_MAX_RISK_PCT` | `2.0` | max_risk_per_trade_pct |
| `_DEFAULT_MIN_TRADE_VALUE` | `500.0` | min_trade_value |
| `_DEFAULT_MAX_POSITION_PCT` | `20.0` | max_position_capital_pct |

### 6.3 Calculation Steps (in order)

#### Step 1 — Extract Parameters
Reads SL%, TP%, capital, max_risk_pct, min_trade_value, tick_size, lot_size, upper_circuit, lower_circuit from the respective inputs (with fallback to defaults).

#### Step 2 — Circuit Limit Guard

**Upper circuit guard:**
```
threshold = upper_circuit × circuit_threshold_pct

if entry_price >= threshold:
    → BLOCKED (near upper circuit)
```

**Lower circuit guard:**
```
if entry_price <= lower_circuit:
    → BLOCKED (at or below lower circuit)
```

When blocked, `ok = False` and `position_size = 0`.

#### Step 3 — SL/TP Price Calculation

```
raw_sl = entry_price × (1 - stop_loss_pct / 100)
raw_tp = entry_price × (1 + take_profit_pct / 100)

sl_price = round_tick(raw_sl, tick_size)
tp_price = round_tick(raw_tp, tick_size)
```

**Tick rounding:**
```python
round_tick(price, tick_size) = round(round(price / tick_size) × tick_size, 4)
```

**Example:** entry=₹1353, SL=2%, TP=5%, tick=₹0.05
```
raw_sl = 1353 × 0.98 = 1325.94   → sl_price = 1325.95
raw_tp = 1353 × 1.05 = 1420.65   → tp_price = 1420.65
```

#### Step 4 — Position Size Calculation

**Step A: Risk-based sizing**
```
risk_amount    = capital × max_risk_per_trade_pct / 100
risk_per_share = entry_price - sl_price
raw_qty        = risk_amount / risk_per_share
quantity       = round_lot(raw_qty, lot_size)
```

If `risk_per_share <= 0` → `ok = False` (impossible to size).

**Lot rounding:**
```python
round_lot(qty, lot_size):
    if lot_size <= 1:
        return max(1, floor(qty))
    lots = floor(qty / lot_size)
    return max(lot_size, lots × lot_size)
```

**Example:** capital=₹1,00,000, max_risk=2%, entry=₹1353, sl=₹1325.95
```
risk_amount    = 1,00,000 × 2 / 100 = ₹2,000
risk_per_share = 1353 - 1325.95 = ₹27.05
raw_qty        = 2000 / 27.05 = 73.94
quantity       = 73  (floor, lot_size=1)
```

**Step B: Capital cap (prevents over-sized positions)**
```
max_pos_pct    = exec_state.max_position_capital_pct  (default: 20%)
max_principal  = capital × max_pos_pct / 100
max_qty_by_cap = round_lot(max_principal / entry_price, lot_size)

if quantity > max_qty_by_cap:
    quantity = max_qty_by_cap
```

**Example (cap in action):** capital=₹1,00,000, SL=0.5%, entry=₹1353
```
risk_amount    = ₹2,000
risk_per_share = 1353 × 0.005 = ₹6.765
raw_qty        = 2000 / 6.765 = 295.6  (very large!)
max_principal  = 1,00,000 × 0.20 = ₹20,000
max_qty_by_cap = floor(20000 / 1353) = 14
quantity       = 14  (capped)
```

#### Step 5 — Minimum Trade Value Check

```
principal = entry_price × quantity

if principal < min_trade_value:
    → BLOCKED (trade too small)
```

### 6.4 Output Variables

| Variable | Formula | Description |
|---|---|---|
| `stop_loss_price` | `entry × (1 - sl_pct/100)` rounded to tick | Absolute SL price |
| `take_profit_price` | `entry × (1 + tp_pct/100)` rounded to tick | Absolute TP price |
| `position_size` | `risk_amount / risk_per_share` floored to lot | Number of shares |
| `principal_amount` | `entry_price × position_size` | Total deployment value in ₹ |
| `ok` | `True` if all checks pass | Whether trade should proceed |

---

## 7. Account Manager

**File:** `app/services/execution/account_manager.py`  
**Class:** `AccountManager` (stateless)  
**Input:** `AccountCheckInput` dataclass  
**Output:** `AccountCheckResult` dataclass

### 7.1 Input Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `trade_value` | `float` | — | `entry_price × quantity` (from RiskManager) |
| `available_margin` | `float` | — | Current available margin in ₹ |
| `current_open_positions` | `int` | — | Number of currently open positions |
| `bars_since_last_trade` | `int` | — | Candles elapsed since the last trade |
| `max_open_positions` | `int` | `3` | Maximum concurrent open positions allowed |
| `cash_reserve_pct` | `float` | `0.10` | Fraction of capital to keep as cash buffer |
| `cooldown_bars` | `int` | `5` | Minimum bars between consecutive trades |
| `capital` | `float` | `100,000.0` | Total capital in ₹ |

### 7.2 Checks (in order — any failure returns immediately)

#### Check 1 — Margin Affordability

```
if trade_value > available_margin:
    → BLOCKED
```

**Must hold:** `trade_value ≤ available_margin`

#### Check 2 — Cash Reserve

```
reserve_required = capital × cash_reserve_pct
margin_after     = available_margin - trade_value

if margin_after < reserve_required:
    → BLOCKED
```

**Must hold:** `(available_margin - trade_value) ≥ capital × cash_reserve_pct`

**Example:** capital=₹1,00,000, reserve=10%, available_margin=₹50,000, trade_value=₹46,000
```
reserve_required = 1,00,000 × 0.10 = ₹10,000
margin_after     = 50,000 - 46,000 = ₹4,000
4,000 < 10,000 → BLOCKED
```

#### Check 3 — Maximum Open Positions

```
if current_open_positions >= max_open_positions:
    → BLOCKED
```

#### Check 4 — Cooldown Period

```
if cooldown_bars > 0 AND bars_since_last_trade < cooldown_bars:
    → BLOCKED
```

If all 4 checks pass → `ok = True`.

### 7.3 Note

The AccountManager is **pure read-only**. It does not update balances, positions, or any state. It only validates whether a new trade is permissible given the current state snapshot passed in the request.

---

## 8. Trade Manager — Bracket Orders

**File:** `app/services/execution/trade_manager.py`  
**Class:** `TradeManager` (stateless)

### 8.1 Bracket Order Structure

A bracket order consists of 3 legs:

| Leg | Side | Type | Price |
|---|---|---|---|
| Entry | BUY | LIMIT | `entry_price` (LTP) |
| Stop Loss Exit | SELL | SL-M | `stop_loss_price` (trigger = price) |
| Take Profit Exit | SELL | LIMIT | `take_profit_price` |

### 8.2 Product Type

```python
strategy_type == "intraday" → ProductType.MIS  (Margin Intraday Square-off)
strategy_type == "positional" → ProductType.CNC (Cash and Carry)
```

### 8.3 Idempotency Key

```python
raw = f"{strategy_id}::{symbol}::{bar_datetime}"
idempotency_key = SHA256(raw.encode()).hexdigest()[:32]
```

This stable 32-character fingerprint ensures that evaluating the same bar twice does not produce duplicate orders when forwarded to an OMS.

### 8.4 Order Validity

All legs: `DAY` (valid for the current trading session).

---

## 9. Reference Data Service

**File:** `app/services/execution/ref_data_service.py`

### 9.1 Upstox Instrument Key Lookup

Queries the `ref_data` schema (SQL via SQLAlchemy `text()`):

```sql
SELECT m.source_symbol
FROM   ref_data.equities e
JOIN   ref_data.equity_data_source_mappings m
       ON m.equity_id = e.id AND m.is_active = true
WHERE  e.ticker      = :ticker
  AND  m.provider_id = :provider_id   -- Upstox provider UUID
LIMIT 1
```

The Upstox provider UUID is cached at process level after the first lookup.

### 9.2 InstrumentDefaults

`InstrumentDefaults` is a dataclass holding per-instrument trading parameters:

| Field | Source | Description |
|---|---|---|
| `tick_size` | `ref_data.system_configs['nse.tick_size_default']` | Price rounding granularity (e.g. `0.05`) |
| `lot_size` | `ref_data.system_configs['nse.lot_size_default']` | Minimum quantity multiple (e.g. `1` for NSE equities) |
| `upper_circuit` | Live Upstox API or `ltp × (1 + upper_circuit_pct)` | Absolute upper circuit price |
| `lower_circuit` | Live Upstox API or `ltp × (1 - lower_circuit_pct)` | Absolute lower circuit price |

**Circuit price derivation (when live data is unavailable):**
```
upper_circuit = ltp × (1 + nse.upper_circuit_default)   e.g. ltp × 1.20 for 20% limit
lower_circuit = ltp × (1 - nse.lower_circuit_default)   e.g. ltp × 0.80 for 20% limit
```

**Fallback values (when DB is unreachable):**

| Fallback | Value |
|---|---|
| tick_size | `0.05` |
| lot_size | `1` |
| upper_circuit_pct | `0.20` (20%) |
| lower_circuit_pct | `0.20` (20%) |

### 9.3 system_configs Keys Queried

```
nse.tick_size_default
nse.lot_size_default
nse.upper_circuit_default
nse.lower_circuit_default
```

These are cached at the module level for the process lifetime (populated on first request).

---

## 10. Risk Execution Config Service

**File:** `app/services/execution/risk_execution_config_service.py`

### 10.1 Storage Format

Values are stored in `ref_data.risk_execution_configs` as key-value pairs:

- **Global scope:** key = field name (e.g. `stop_loss_pct`, `per_trade_risk`)
- **Strategy scope:** key = `strategy:{uuid}.field_name`
- **Session scope:** key = `session:{uuid}.field_name`

### 10.2 Default Seed Values (Global)

| Key | Default | Type |
|---|---|---|
| `max_trades` | `2` | int |
| `risk_reward` | `2.5` | float |
| `daily_loss_cap` | `3.0` | float (%) |
| `execution_mode` | `"Backtest"` | string |
| `per_trade_risk` | `2.0` | float (%) |
| `trading_window` | `"9:15 - 15:30"` | string |
| `position_sizing` | `"Risk based"` | string |
| `risk_validation` | `"system risk guardials"` | string |
| `stop_loss_pct` | `2.0` | float (%) |
| `take_profit_pct` | `5.0` | float (%) |
| `minimum_trade_value` | `500.0` | float (₹) |

### 10.3 RiskExecutionConfigSnapshot Fields

| Field | Property Alias | Description |
|---|---|---|
| `config_scope` | — | `"global"`, `"session"`, or `"strategy"` |
| `scope_id` | — | UUID or `"global-default"` |
| `max_trades` | `max_trades_per_day` | Maximum trades per day |
| `risk_reward` | — | `take_profit_pct / stop_loss_pct` |
| `daily_loss_cap` | `daily_loss_cap_pct` | % of capital — daily loss limit |
| `per_trade_risk` | `per_trade_risk_pct` | % of capital risked per trade |
| `stop_loss_pct` | — | Default SL % |
| `take_profit_pct` | — | Default TP % |
| `minimum_trade_value` | `min_trade_value` | Minimum notional value per trade |

### 10.4 Risk-Reward Formula

```python
risk_reward = take_profit_pct / stop_loss_pct  (when stop_loss_pct > 0)
```

### 10.5 Resolution Logic

`resolve_active_risk_execution_config()` currently always returns the **global** config (strategy/session-specific resolution is scaffolded in `_fetch_snapshot_by_scope` but not wired into the resolution path yet).

---

## 11. Execution Cache

**File:** `app/services/execution/execution_cache.py`  
**Class:** `MarketDataCache`

A **thread-safe in-process TTL cache** with three independent stores: candles, LTP, and circuit limits.

| Store | Cache Key | TTL |
|---|---|---|
| Candles | `"{symbol}::{timeframe}"` | `settings.market_data_cache_ttl_seconds` |
| LTP | `"{symbol}"` | same TTL |
| Circuit limits | `"{symbol}"` | same TTL |

**TTL check:** `time.monotonic() < expires_at` — monotonic clock avoids wall-clock skew.

**Cache behavior:**
- On a fresh request: fetch from Upstox → store in cache → return
- Cache hit within TTL: return cached value directly
- Network error: log warning → check stale cache (returns even if expired) → raise `RuntimeError` if no data at all

---

## 12. Quant Backtest Engine

**Files:** `quant_engine/engine/`  
**Entry:** `POST /run` (async, callback-based) or `POST /run-sync`

### 12.1 Request Body (RunConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `starting_balance` | `float` | `10,000.0` | Initial portfolio balance in ₹ |
| `slippage_bps` | `float` | `5.0` | Slippage in basis points |
| `commission_bps` | `float` | `2.0` | Broker commission in basis points |
| `max_holding_candles` | `int\|null` | from YAML | Max bars to hold a position |
| `objective` | `str` | from YAML | `"intraday"` or `"positional"` |
| `daily_loss_cap_pct` | `float` | `0.0` | % portfolio loss halting new entries today |
| `max_trades_per_day` | `int` | `0` | Max round-trips per trading day (0=unlimited) |
| `stt_intraday_sell_pct` | `float` | `0.025` | STT on sell for intraday (0.025%) |
| `stt_delivery_pct` | `float` | `0.1` | STT for delivery on both sides (0.1%) |

### 12.2 Backtest Pipeline (run_backtest)

```
Step 1: Parse YAML strategy config
Step 2: load_ohlcv_data() — normalize records to DataFrame
Step 3: _enforce_market_data_window() — trim to configured UTC window
Step 4: _check_data_sufficiency() — min 20 candles, warn on warm-up shortage
Step 5: add_all_indicators() — compute all YAML-specified indicators
Step 6: _ensure_common_indicators() — compute any indicators used in condition strings
Step 7: simulate_trades() — bar-by-bar loop
Step 8: build_backtest_result() → calculate_metrics() → build_assessment()
```

### 12.3 Backtest Market Data Window

All backtests are **enforced** to this UTC window (configurable in `config.py`):

```python
BACKTEST_MARKET_DATA_FROM_UTC = "2024-01-01T00:00:00Z"
BACKTEST_MARKET_DATA_TO_UTC   = "2026-03-31T23:59:59Z"
```

Data outside this range is trimmed. An error is raised if no data falls within the window.

### 12.4 Data Sufficiency Check

```python
if len(df) < 20:
    raise ValueError("Insufficient data")

eligible_after_warmup = len(df) - required_warmup
if required_warmup > 0 and eligible_after_warmup < required_warmup:
    logger.warning("Insufficient candles after warm-up")
```

Recommended minimum: `total_candles >= required_warmup × 2`

### 12.5 OHLCV Data Normalization

Accepts either:
- List of dicts: `[{"timestamp": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}]`
- List of arrays: `[[timestamp, open, high, low, close, volume], ...]`

Produces a UTC-indexed pandas DataFrame with columns: `open`, `high`, `low`, `close`, `volume`.

---

## 13. Technical Indicators — All Formulas

**File:** `quant_engine/engine/indicators.py`

All indicators operate on the `close` price series (or the full DataFrame for ATR/VWAP).

### 13.1 SMA — Simple Moving Average

```python
SMA(n) = rolling_mean(close, window=n, min_periods=n)
```

Column name: `SMA_{n}` (e.g. `SMA_50`)

### 13.2 EMA — Exponential Moving Average

```python
EMA(n) = ewm(close, span=n, adjust=False, min_periods=n).mean()
```

`adjust=False` uses the recursive formula: `EMA_t = α × price_t + (1-α) × EMA_{t-1}` where `α = 2/(n+1)`.

Column name: `EMA_{n}` (e.g. `EMA_21`)

### 13.3 RSI — Relative Strength Index (Wilder's Method)

```
delta     = close.diff()
gain      = max(delta, 0)       — clip to zero for losses
loss      = max(-delta, 0)      — clip to zero for gains

avg_gain  = EWM(gain,  alpha=1/n, adjust=False, min_periods=n)
avg_loss  = EWM(loss,  alpha=1/n, adjust=False, min_periods=n)

RS        = avg_gain / avg_loss
RSI       = 100 - (100 / (1 + RS))
```

Wilder's smoothing uses `alpha = 1/n` (not `2/(n+1)` like standard EMA). `avg_loss = 0` is replaced with `NaN` before division.

Column name: `RSI_{n}` (e.g. `RSI_14`)

### 13.4 MACD — Moving Average Convergence Divergence

```
MACD        = EMA(12) - EMA(26)       — fast minus slow EMA of close
MACD_SIGNAL = EMA(9) of MACD          — signal line
MACD_HIST   = MACD - MACD_SIGNAL      — histogram (momentum)
```

Column names: `MACD`, `MACD_SIGNAL`, `MACD_HIST`

### 13.5 Bollinger Bands

```
BB_MID(n)   = SMA(close, n)
std(n)      = rolling_std(close, window=n, min_periods=n)
BB_UPPER(n) = BB_MID(n) + 2 × std(n)
BB_LOWER(n) = BB_MID(n) - 2 × std(n)
```

Default period: `n=20`, multiplier: `2`. Column names: `BB_UPPER_{n}`, `BB_LOWER_{n}`, `BB_MID_{n}`.

### 13.6 ATR — Average True Range (Wilder's Smoothing)

```
TR_t = max(
    High_t - Low_t,
    |High_t - Close_{t-1}|,
    |Low_t  - Close_{t-1}|
)

ATR(n) = EWM(TR, alpha=1/n, adjust=False, min_periods=n)
```

Column name: `ATR_{n}` (e.g. `ATR_14`)

### 13.7 VWAP — Volume Weighted Average Price

**Intraday bars** (median gap between bars < 6 hours):
```
TP_t = (High_t + Low_t + Close_t) / 3      — typical price

VWAP_t = Σ(TP × Volume, session_start to t)
         ────────────────────────────────────
           Σ(Volume, session_start to t)
```
Session = calendar day. The cumulative sum resets at midnight.

**Daily bars** (median gap >= 6 hours):
```
VWAP = (High + Low + Close) / 3    — typical price for the completed session
```
(Cumulating across multiple daily bars would produce a volume-weighted moving average, not a session VWAP.)

Column name: `VWAP`

### 13.8 Indicator Warm-up Requirements

| Indicator | Warm-up Candles |
|---|---|
| SMA(n) | n |
| EMA(n) | n |
| RSI(n) | n |
| BB(n) | n |
| ATR(n) | n |
| MACD | 26 + 9 = 35 (EMA26 + signal EMA9) |
| VWAP | 0 |

`max_indicator_warmup(indicator_config)` returns the maximum across all configured indicators. The simulator skips the first `warm_up_candles` bars of evaluation.

---

## 14. Trade Simulator — All Formulas

**File:** `quant_engine/engine/simulator.py`  
**Function:** `simulate_trades()`

### 14.1 Cost Model

**Indian equity cost model (all costs as fractions of price):**

```
bps_cost = (slippage_bps + commission_bps) / 10,000

Effective entry price = raw_price × (1 + bps_cost + stt_entry_pct / 100)
Effective exit price  = raw_price × (1 - bps_cost - stt_exit_pct / 100)
```

**STT (Securities Transaction Tax) rates:**

| Trade Type | STT on Buy | STT on Sell |
|---|---|---|
| Intraday equity | `0%` | `0.025%` |
| Delivery equity | `0.1%` | `0.1%` |

**Default bps values:**
- `slippage_bps = 5.0` (5 basis points = 0.05%)
- `commission_bps = 2.0` (2 basis points = 0.02%)

**Example for intraday entry (raw price ₹1000):**
```
bps_cost       = (5 + 2) / 10000 = 0.0007
stt_entry_pct  = 0% (intraday)
effective_entry = 1000 × (1 + 0.0007 + 0) = 1000.70
```

**Example for intraday exit (raw price ₹1020):**
```
stt_exit_pct   = 0.025%
effective_exit  = 1020 × (1 - 0.0007 - 0.00025) = 1020 × 0.99905 = 1019.03
```

### 14.2 Entry Logic

- Evaluated on bar `i` (current bar)
- Filled at bar `i+1` open price (next-bar execution — realistic simulation)
- Entry is **blocked** if:
  - It is the session's last bar (can't fill the next day for intraday)
  - `i >= len(df) - 1` (no next bar exists)
  - Daily loss cap is breached (`_daily_cap_breached()`)
  - Max trades per day reached (`_max_trades_breached()`)

### 14.3 In-Trade Exit Logic (checked in this order per bar)

**1. Stop Loss:**
```
stop_price = entry_price × (1 - stop_loss_pct / 100)
stop_hit   = low_price <= stop_price
```
Exit price = `apply_exit_costs(stop_price)`

**2. Take Profit:**
```
take_profit_price = entry_price × (1 + take_profit_pct / 100)
take_profit_hit   = high_price >= take_profit_price
```
Exit price = `apply_exit_costs(take_profit_price)`

**Worst-case rule** (both hit same bar):
```
if stop_hit AND take_profit_hit:
    exit at stop_price  (conservative — assumes stop was hit first)
    exit_reason = "STOP_LOSS_AND_TAKE_PROFIT_SAME_BAR"
```

**3. Exit Signal:**
Evaluated via `evaluate_condition(exit_condition, df, i)`. If fired:
- If session last bar or no next bar: exit at `apply_exit_costs(close_price)`
- Otherwise: exit at `apply_exit_costs(next_bar_open)`
- `exit_reason = "EXIT_SIGNAL"`

**4. Max Holding Candles (Time-based stop):**
```
holding_candles = i - entry_index
if max_holding_candles is not None AND holding_candles >= max_holding_candles:
    → exit (same next-bar logic as exit signal)
    exit_reason = "MAX_HOLDING"
```

**5. Session End (Intraday only):**
```
if is_session_last_bar:
    → exit at apply_exit_costs(close_price)
    exit_reason = "SESSION_END"
```

**6. End of Data:**
Any open trade at the end of the DataFrame is force-closed at the last bar's close:
```
exit_reason = "END_OF_DATA"
```

### 14.4 Trade P&L Formulas

```
pnl_inr = exit_price - entry_price           — per-unit absolute P&L in ₹
pnl_abs = pnl_inr / entry_price              — fractional return (used for compounding)
pnl_pct = pnl_abs × 100.0                   — percentage return
```

### 14.5 MAE and MFE

Tracked bar-by-bar during the holding period:

```
_trade_min_low  = min of all bar lows during holding
_trade_max_high = max of all bar highs during holding

MAE (Maximum Adverse Excursion):
  mae_pct = (_trade_min_low - entry_price) / entry_price × 100   (≤ 0 for long)

MFE (Maximum Favorable Excursion):
  mfe_pct = (_trade_max_high - entry_price) / entry_price × 100  (≥ 0 for long)
```

### 14.6 Daily Circuit Breakers (Intraday Only)

**Daily loss cap:**
```
_session_cumulative_pnl_pct += pnl_pct  (after each trade exits)

daily_cap_breached = _session_cumulative_pnl_pct <= -daily_loss_cap_pct
```
New entries are blocked when breached. Resets at session start (new calendar day).

**Max trades per day:**
```
_session_trades_today += 1  (after each trade exits)

max_trades_breached = _session_trades_today >= max_trades_per_day
```
Resets at session start.

### 14.7 Portfolio Compounding

After each trade:
```
_portfolio_balance_factor × = (1 + pnl_abs)
```

This is used internally to track portfolio growth across the simulation.

---

## 15. Backtest Metrics — All Formulas

**File:** `quant_engine/engine/metrics.py`  
**Function:** `calculate_metrics(trades, df, starting_balance, start_utc, end_utc)`

### 15.1 Trade Snapshots (Compounding)

The portfolio balance compounds trade-by-trade:
```
For each trade (sorted by entry_date):
    pnl_value      = starting_balance_for_this_trade × pnl_abs
    ending_balance = starting_balance + pnl_value
    next trade's starting_balance = this trade's ending_balance
```

### 15.2 Basic Trade Metrics

```
total_trades    = count(trades)
wins            = count(trades where pnl_pct > 0)
win_rate        = wins / total_trades × 100           (%)

gross_return_pct = Σ(pnl_pct for all trades)          (sum of individual % returns)
average_outcome_per_trade = gross_return_pct / total_trades
                            (average return per trade %)
```

### 15.3 Net Return

```
net_return_pct = (ending_balance - starting_balance) / starting_balance × 100
total_return_pct = net_return_pct   (alias)
```

### 15.4 Profit Factor

```
gross_profit = Σ(pnl_abs for winning trades)
gross_loss   = |Σ(pnl_abs for losing trades)|

profit_factor = gross_profit / gross_loss   (when gross_loss > 0)
              = Infinity → capped at 9,999  (when no losing trades)
              = 0                           (no winning trades)
```

Cap: `PROFIT_FACTOR_MAX_CAP = 9,999.0` (prevents JSON serialization issues with `Infinity`).

### 15.5 Annual Return (CAGR)

```
annual_return = ((ending_balance / starting_balance)^(365 / num_days) - 1) × 100
```

Where `num_days = calendar_days(start_utc, end_utc)`.

### 15.6 Daily Portfolio Values

An intermediate time series is built to mark the portfolio to market on each bar:

- **During a trade:** portfolio value on bar `t` = `starting_balance_of_trade × (close_t / entry_price)`
- **Outside trades:** portfolio value = last trade's `ending_balance` (held flat)

This is then resampled to daily (last bar of each day) and reindexed to the full calendar range.

### 15.7 Daily Returns

```
daily_returns     = daily_values.pct_change().fillna(0.0)       — decimal form (e.g. 0.01 = 1%)
daily_returns_pct = daily_returns × 100.0                       — percentage form
avg_daily_return  = daily_returns.mean() × 100.0                (%)
```

### 15.8 Sharpe Ratio (Annualized)

```
Sharpe = (mean(daily_returns) / std(daily_returns, ddof=1)) × √252

TRADING_DAYS_PER_YEAR = 252
Risk-free rate = 0 (not subtracted)
```

Returns `0.0` when standard deviation is `0`.

### 15.9 Sortino Ratio (Annualized)

```
downside_returns = daily_returns[daily_returns < 0]
downside_std     = std(downside_returns, ddof=1)

Sortino = (mean(daily_returns) / downside_std) × √252
```

Returns `0.0` when downside deviation is `0`.

### 15.10 Volatility (Annualized)

```
volatility_pct = std(daily_returns_pct, ddof=1) × √252
```

### 15.11 Downside Deviation (Annualized)

```
downside_returns_pct = daily_returns_pct[daily_returns_pct < 0]
downside_deviation_pct = std(downside_returns_pct, ddof=1) × √252
```

### 15.12 Drawdown Analysis

**Running peak:**
```
running_peak = cumulative_max(portfolio_values)
```

**Drawdown series (in %):**
```
drawdown = (portfolio_values / running_peak - 1) × 100
```

**Max drawdown:**
```
max_drawdown = |min(drawdown)|
```

**Drawdown duration (calendar days from peak to trough):**
```
trough_idx  = argmin(drawdown)
peak_idx    = LAST date on/before trough where drawdown == 0   # most recent high
dd_duration = (trough_idx - peak_idx).days
```
`peak_idx` is the most recent new high before the trough (the peak the curve
declined from), not `argmax`'s earliest occurrence — otherwise a curve that
revisits the same high before its worst drop overstates the duration.

**Recovery date:** First date after trough where `portfolio_value >= peak_value`; `null` if never recovered.

**Recovery time:** `(recovery_idx - trough_idx).days`; `null` (not `0`) if never recovered.

**Max drawdown duration (days underwater):**
Longest consecutive run of days where `drawdown < 0` (any prior peak not recovered).

### 15.13 Calmar Ratio

```
calmar_ratio = annual_return / max_drawdown   (when max_drawdown > 0)
             = 0                              (otherwise)
```

### 15.14 IRR (Internal Rate of Return)

```
irr_daily      = (ending_balance / starting_balance)^(1 / num_days) - 1
irr_annualized = (1 + irr_daily)^365 - 1
```

### 15.15 Historical VaR and Expected Shortfall (CVaR)

```
VAR_CONFIDENCE_LEVEL = 0.05  (5% tail = 95th percentile VaR)

sorted_returns = sort(daily_returns_pct, ascending)
cutoff_idx     = max(1, floor(len(sorted_returns) × 0.05))
var_95_pct     = sorted_returns[cutoff_idx - 1]        — 5th percentile loss
es_95_pct      = mean(sorted_returns[:cutoff_idx])     — mean of tail beyond VaR
```

Both returned as negative numbers representing potential loss (%).

### 15.16 Average Trade Duration

```
average_holding_duration = mean((exit_ts - entry_ts).total_seconds() / 86400)
                           across all trades
```

### 15.17 Trades per Month

```
trades_per_month = total_trades / (num_days / CALENDAR_DAYS_PER_MONTH)
CALENDAR_DAYS_PER_MONTH = 30.4375   (= 365.25 / 12)
```

### 15.18 Longest Losing Streak

Longest consecutive sequence of trades where `pnl_pct <= 0` (break-even counts as a loss).

### 15.19 Pass / Fail Criteria

| Objective | Condition | Pass? |
|---|---|---|
| `intraday` | `total_trades > 0` AND `win_rate >= 40%` | ✅ |
| `positional` | `total_trades > 0` AND `win_rate >= 40%` AND `profit_factor >= 1.2` | ✅ |

---

## 16. Assessment & Grading System

**Files:** `quant_engine/engine/assessment.py`, `quant_engine/engine/descriptions.py`

### 16.1 Overall Grade — Score-Based

The grade is derived from a composite score (0–100):

```
score = return_score + sharpe_score + drawdown_score + consistency_score + sample_score
```

| Component | Max Points | Metric |
|---|---|---|
| Return | 30 | `total_return_pct` |
| Sharpe | 25 | `sharpe_ratio` |
| Drawdown | 25 | `max_drawdown` |
| Consistency | 15 | `profit_factor` + `win_rate` combined |
| Sample Size | 5 | `total_trades` |

**Return score:**
| `total_return_pct` | Score |
|---|---|
| ≥ 20% | 30 |
| ≥ 12% | 24 |
| ≥ 5% | 16 |
| ≥ 0% | 8 |
| < 0% | 0 |

**Sharpe score:**
| `sharpe_ratio` | Score |
|---|---|
| ≥ 1.5 | 25 |
| ≥ 1.0 | 18 |
| ≥ 0.5 | 10 |
| > 0.0 | 5 |
| ≤ 0.0 | 0 |

**Drawdown score:**
| `max_drawdown` | Score |
|---|---|
| ≤ 8% | 25 |
| ≤ 12% | 18 |
| ≤ 15% | 12 |
| ≤ 20% | 8 |
| ≤ 30% | 4 |
| > 30% | 0 |

**Consistency score (profit_factor AND win_rate):**
| `profit_factor` ≥ | `win_rate` ≥ | Score |
|---|---|---|
| 1.5 | 60% | 15 |
| 1.2 | 50% | 12 |
| 1.0 | 45% | 8 |
| 0.8 | any | 4 |
| < 0.8 | any | 0 |

**Sample size score:**
| `total_trades` | Score |
|---|---|
| ≥ 100 | 5 |
| ≥ 40 | 4 |
| ≥ 20 | 2 |
| < 20 | 0 |

**Grade thresholds:**
| Score | Grade |
|---|---|
| ≥ 85 | A |
| ≥ 72 | B+ |
| ≥ 62 | B |
| ≥ 52 | C+ |
| ≥ 42 | C |
| < 42 | D |

### 16.2 Return Potential Classification

| Condition | Label |
|---|---|
| `total_return_pct >= 15%` AND `sharpe >= 1.0` AND `total_trades >= 20` | **Strong** |
| `total_return_pct >= 5%` AND `total_trades >= 10` | **Moderate** |
| Otherwise | **Weak** |

### 16.3 Risk Profile Classification

| Condition | Label |
|---|---|
| `max_drawdown <= 8%` AND `volatility <= 12%` AND `var_95 >= -2%` | **Conservative** |
| `max_drawdown <= 15%` AND `volatility <= 20%` AND `var_95 >= -4%` | **Moderate** |
| Otherwise | **Aggressive** |

### 16.4 Drawdown Tolerance Required

| Condition | Label |
|---|---|
| `max_drawdown <= 8%` AND `recovery_time <= 30 days` | **Low** |
| `max_drawdown <= 15%` AND `recovery_time <= 60 days` | **Medium** |
| Otherwise | **High** |

### 16.5 Recommended For

| Condition | Label |
|---|---|
| `total_return <= 0` OR `sharpe <= 0` | **Not Recommended** |
| `risk_profile == "Aggressive"` OR `drawdown_tolerance == "High"` OR `trades/month >= 20` | **Experienced Traders** |
| `risk_profile == "Moderate"` OR `drawdown_tolerance == "Medium"` OR `trades/month >= 8` | **Intermediate Traders** |
| Otherwise | **Beginner Traders** |

---

## 17. Market Phase & Alignment Classifier

**File:** `quant_engine/engine/market_classifier.py`

### 17.1 Market Type (Bull / Bear / Sideways)

```
price_return_pct = (last_close - first_close) / first_close × 100

if price_return_pct >= 8%  → "Bull"
if price_return_pct <= -8% → "Bear"
otherwise                  → "Sideways"
```

Constants: `BULL_MARKET_MIN_RETURN_PCT = 8.0`, `BEAR_MARKET_MAX_RETURN_PCT = -8.0`

### 17.2 Market Phase (Uptrend / Downtrend / Range-bound)

Linear regression on normalized close prices:

```
price_range = max(close) - min(close)
normalized  = (close - min(close)) / price_range   — maps to [0, 1]
x           = [0, 1, 2, ..., n-1]
slope       = polyfit(x, normalized, degree=1)[0]  — slope of best-fit line

if slope >= 0.0003  → "Uptrend"
if slope <= -0.0003 → "Downtrend"
otherwise           → "Range-bound"
```

Using normalized prices makes the slope dimensionless (independent of absolute price level).

### 17.3 Strategy Alignment

Base alignment from matrix lookup:

| Strategy Side | Market Type | Base Alignment |
|---|---|---|
| LONG | Bull | Strong |
| LONG | Sideways | Moderate |
| LONG | Bear | Weak |
| SHORT | Bear | Strong |
| SHORT | Sideways | Moderate |
| SHORT | Bull | Weak |

Adjustment for "Moderate" base:
```
if phase_win_rate >= 60% → upgrade to "Strong"
if phase_win_rate <= 30% → downgrade to "Weak"
```

### 17.4 Entry Condition Classification

Evaluated at the entry bar using fast and slow rolling averages:

```
fast_avg = mean(close[bar_idx-10 : bar_idx])
slow_avg = mean(close[bar_idx-30 : bar_idx])
current  = close[bar_idx]

if current > fast_avg > slow_avg → "Bull"
if current < fast_avg < slow_avg → "Bear"
otherwise                        → "Sideways"
```

Window constants: `ENTRY_CONDITION_FAST_WINDOW = 10`, `ENTRY_CONDITION_SLOW_WINDOW = 30`.
Returns `"Unknown"` if `bar_idx < 30`.

### 17.5 Market Phase Analysis

Data is split into calendar quarters (`pd.Grouper(freq="QE")`). For each quarter:

- Price return over the quarter → `market_type`
- Linear regression slope → `market_phase`
- Trades entered in the quarter → win rate, cumulative return
- Alignment = `classify_alignment(strategy_side, market_type, phase_win_rate)`

### 17.6 Monthly Performance

```
monthly_values  = daily_portfolio.resample("ME").last()
monthly_returns = monthly_values.pct_change().fillna(0) × 100   (%)
```

**Monthly statistics:**
```
highest_monthly_gain_pct     = max(monthly_returns)
lowest_monthly_gain_pct      = min(monthly_returns)
monthly_performance_range_pct = highest - lowest

return_vs_drawdown_efficiency = total_strategy_return / monthly_drawdown
```
Where `monthly_drawdown` = peak-to-trough across the cumulative monthly return series.

---

## 18. Configuration Reference — All Variables

### 18.1 quant_engine/engine/config.py — All Constants

| Constant | Value | Description |
|---|---|---|
| **Pass/Fail** | | |
| `PASS_FAIL_THRESHOLDS["intraday"]["min_win_rate_pct"]` | `40.0` | Minimum win rate for intraday pass |
| `PASS_FAIL_THRESHOLDS["intraday"]["min_profit_factor"]` | `0.0` | PF not enforced for intraday |
| `PASS_FAIL_THRESHOLDS["positional"]["min_win_rate_pct"]` | `40.0` | Minimum win rate for positional pass |
| `PASS_FAIL_THRESHOLDS["positional"]["min_profit_factor"]` | `1.2` | Minimum profit factor for positional pass |
| **Grade Boundaries** | | |
| `GRADE_A_MIN_SCORE` | `85` | Score ≥ 85 → Grade A |
| `GRADE_B_PLUS_MIN_SCORE` | `72` | Score ≥ 72 → Grade B+ |
| `GRADE_B_MIN_SCORE` | `62` | Score ≥ 62 → Grade B |
| `GRADE_C_PLUS_MIN_SCORE` | `52` | Score ≥ 52 → Grade C+ |
| `GRADE_C_MIN_SCORE` | `42` | Score ≥ 42 → Grade C |
| **Return Score** | | |
| `GRADE_RETURN_SCORE_EXCELLENT_MIN_PCT` | `20.0` | → 30 pts |
| `GRADE_RETURN_SCORE_GOOD_MIN_PCT` | `12.0` | → 24 pts |
| `GRADE_RETURN_SCORE_MODERATE_MIN_PCT` | `5.0` | → 16 pts |
| `GRADE_RETURN_SCORE_POSITIVE_MIN_PCT` | `0.0` | → 8 pts |
| **Sharpe Score** | | |
| `GRADE_SHARPE_SCORE_EXCELLENT_MIN` | `1.5` | → 25 pts |
| `GRADE_SHARPE_SCORE_GOOD_MIN` | `1.0` | → 18 pts |
| `GRADE_SHARPE_SCORE_MODERATE_MIN` | `0.5` | → 10 pts |
| `GRADE_SHARPE_SCORE_POSITIVE_MIN` | `0.0` | → 5 pts |
| **Drawdown Score** | | |
| `GRADE_DRAWDOWN_SCORE_EXCELLENT_MAX_PCT` | `8.0` | → 25 pts |
| `GRADE_DRAWDOWN_SCORE_GOOD_MAX_PCT` | `12.0` | → 18 pts |
| `GRADE_DRAWDOWN_SCORE_MODERATE_MAX_PCT` | `15.0` | → 12 pts |
| `GRADE_DRAWDOWN_SCORE_FAIR_MAX_PCT` | `20.0` | → 8 pts |
| `GRADE_DRAWDOWN_SCORE_POOR_MAX_PCT` | `30.0` | → 4 pts |
| **Consistency Score** | | |
| `GRADE_CONSISTENCY_SCORE_EXCELLENT_MIN_PF` | `1.5` | Combined with WR ≥ 60% → 15 pts |
| `GRADE_CONSISTENCY_SCORE_EXCELLENT_MIN_WR` | `60.0` | |
| `GRADE_CONSISTENCY_SCORE_GOOD_MIN_PF` | `1.2` | Combined with WR ≥ 50% → 12 pts |
| `GRADE_CONSISTENCY_SCORE_GOOD_MIN_WR` | `50.0` | |
| `GRADE_CONSISTENCY_SCORE_MODERATE_MIN_PF` | `1.0` | Combined with WR ≥ 45% → 8 pts |
| `GRADE_CONSISTENCY_SCORE_MODERATE_MIN_WR` | `45.0` | |
| `GRADE_CONSISTENCY_SCORE_FAIR_MIN_PF` | `0.8` | → 4 pts (any WR) |
| **Sample Score** | | |
| `GRADE_SAMPLE_SCORE_LARGE_MIN_TRADES` | `100` | → 5 pts |
| `GRADE_SAMPLE_SCORE_MEDIUM_MIN_TRADES` | `40` | → 4 pts |
| `GRADE_SAMPLE_SCORE_SMALL_MIN_TRADES` | `20` | → 2 pts |
| **Return Potential** | | |
| `RETURN_POTENTIAL_STRONG_MIN_PCT` | `15.0` | total_return_pct threshold for Strong |
| `RETURN_POTENTIAL_MODERATE_MIN_PCT` | `5.0` | total_return_pct threshold for Moderate |
| `RETURN_POTENTIAL_STRONG_MIN_SHARPE` | `1.0` | Also required for Strong |
| `RETURN_POTENTIAL_STRONG_MIN_TRADES` | `20` | Also required for Strong |
| `RETURN_POTENTIAL_MODERATE_MIN_TRADES` | `10` | Required for Moderate |
| **Risk Profile** | | |
| `RISK_CONSERVATIVE_MAX_DRAWDOWN_PCT` | `8.0` | |
| `RISK_CONSERVATIVE_MAX_VOLATILITY_PCT` | `12.0` | |
| `RISK_CONSERVATIVE_MIN_VAR_95_PCT` | `-2.0` | VaR must not be worse than this |
| `RISK_MODERATE_MAX_DRAWDOWN_PCT` | `15.0` | |
| `RISK_MODERATE_MAX_VOLATILITY_PCT` | `20.0` | |
| `RISK_MODERATE_MIN_VAR_95_PCT` | `-4.0` | |
| **Drawdown Tolerance** | | |
| `DRAWDOWN_TOLERANCE_LOW_MAX_PCT` | `8.0` | |
| `DRAWDOWN_TOLERANCE_LOW_MAX_RECOVERY_DAYS` | `30` | |
| `DRAWDOWN_TOLERANCE_MEDIUM_MAX_PCT` | `15.0` | |
| `DRAWDOWN_TOLERANCE_MEDIUM_MAX_RECOVERY_DAYS` | `60` | |
| **Recommended Trader Type** | | |
| `RECOMMENDED_EXPERIENCED_MIN_TRADES_PER_MONTH` | `20` | |
| `RECOMMENDED_INTERMEDIATE_MIN_TRADES_PER_MONTH` | `8` | |
| **Market Phase** | | |
| `BULL_MARKET_MIN_RETURN_PCT` | `8.0` | |
| `BEAR_MARKET_MAX_RETURN_PCT` | `-8.0` | |
| `TREND_SLOPE_UPTREND_MIN` | `0.0003` | Normalized price slope |
| `TREND_SLOPE_DOWNTREND_MAX` | `-0.0003` | |
| **Alignment** | | |
| `ALIGNMENT_STRONG_UPGRADE_WIN_RATE_PCT` | `60.0` | Upgrade Moderate → Strong |
| `ALIGNMENT_WEAK_DOWNGRADE_WIN_RATE_PCT` | `30.0` | Downgrade Moderate → Weak |
| **Entry Condition** | | |
| `ENTRY_CONDITION_FAST_WINDOW` | `10` | Fast MA bars for entry condition label |
| `ENTRY_CONDITION_SLOW_WINDOW` | `30` | Slow MA bars for entry condition label |
| **Backtest Window** | | |
| `BACKTEST_MARKET_DATA_FROM_UTC` | `"2024-01-01T00:00:00Z"` | Fixed backtest start |
| `BACKTEST_MARKET_DATA_TO_UTC` | `"2026-03-31T23:59:59Z"` | Fixed backtest end |
| **Metrics Computation** | | |
| `DEFAULT_STARTING_BALANCE` | `10,000.0` | Default portfolio balance |
| `TRADING_DAYS_PER_YEAR` | `252` | Annualization factor |
| `MONTHS_PER_YEAR` | `12.0` | |
| `CALENDAR_DAYS_PER_MONTH` | `30.4375` | 365.25 / 12 |
| `PROFIT_FACTOR_MAX_CAP` | `9,999.0` | Prevents Infinity in JSON |
| `VAR_CONFIDENCE_LEVEL` | `0.05` | 5% tail = 95th percentile VaR |

---

## 19. Environment Variables

**Files:** `app/core/config.py`, `.env.example`

### 19.1 Live Execution (Market Data)

| Variable | Default | Description |
|---|---|---|
| `MARKET_DATA_URL` | `https://api.upstox.com/v2` | Upstox v2 API base URL |
| `UPSTOX_ACCESS_TOKEN` | — | Bearer token for Upstox API auth |
| `MARKET_DATA_TIMEOUT_SECONDS` | — | HTTP request timeout (seconds) |
| `MARKET_DATA_CACHE_TTL_SECONDS` | — | In-process cache TTL for candles/LTP/circuits |
| `MARKET_DATA_CIRCUIT_THRESHOLD_PCT` | `0.98` | Fraction of upper circuit at which entries are blocked |

### 19.2 Backtest / Historical Data

| Variable | Default | Description |
|---|---|---|
| `HISTORICAL_DATA_URL` | — | URL of historical OHLCV API (ngrok tunnel or internal service) |
| `HISTORICAL_DATA_TIMEOUT_SECONDS` | — | Request timeout |
| `BACKTEST_DEFAULT_LOOKBACK_DAYS` | — | Default historical data window |

### 19.3 Quant Engine

| Variable | Default | Description |
|---|---|---|
| `QUANT_ENGINE_URL` | — | URL of the quant engine FastAPI service |
| `QUANT_ENGINE_TIMEOUT` | — | Timeout for quant engine HTTP calls |

### 19.4 Database

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |

The `ref_data` schema (equities, equity_data_source_mappings, system_configs, risk_execution_configs, data_providers) is owned by a separate service but queried read-only.

The `ai_strategy` schema (strategies, execution_states) is owned by this service.

### 19.5 Risk Manager Hard-coded Defaults (source code)

These override behaviour when DB rows are missing (should not occur in production):

| Constant | File | Value |
|---|---|---|
| `_DEFAULT_STOP_LOSS_PCT` | `risk_manager.py` | `2.0%` |
| `_DEFAULT_TAKE_PROFIT_PCT` | `risk_manager.py` | `5.0%` |
| `_DEFAULT_TICK_SIZE` | `risk_manager.py` | `0.05` |
| `_DEFAULT_LOT_SIZE` | `risk_manager.py` | `1` |
| `_DEFAULT_CAPITAL` | `risk_manager.py` | `₹1,00,000` |
| `_DEFAULT_MAX_RISK_PCT` | `risk_manager.py` | `2.0%` |
| `_DEFAULT_MIN_TRADE_VALUE` | `risk_manager.py` | `₹500` |
| `_DEFAULT_MAX_POSITION_PCT` | `risk_manager.py` | `20%` |
| `_FALLBACK_TICK_SIZE` | `ref_data_service.py` | `0.05` |
| `_FALLBACK_LOT_SIZE` | `ref_data_service.py` | `1` |
| `_FALLBACK_UPPER_CIRCUIT_PCT` | `ref_data_service.py` | `20%` |
| `_FALLBACK_LOWER_CIRCUIT_PCT` | `ref_data_service.py` | `20%` |

---

## 20. Data Flow Summary

### 20.1 Live Evaluation Request

```
Client → POST /strategy/evaluate/execute
              ↓
         StrategyEvaluator.evaluate()
              │
              ├─ DB: ai_strategy.strategies (strategy_id mode)
              ├─ DB: ref_data.risk_execution_configs (global risk config)
              ├─ DB: ai_strategy.execution_states (position/capital state)
              │
              ├─ DB: ref_data.equities + equity_data_source_mappings
              │      → Upstox instrument key (NSE_EQ|INE002A01018)
              │
              ├─ Upstox: /historical-candle  → OHLCV DataFrame (resampled)
              ├─ Upstox: /market-quote/ltp   → LTP float
              ├─ Upstox: /market-quote/quotes → upper/lower circuit prices
              │
              ├─ DB: ref_data.system_configs → tick_size, lot_size
              │      + circuit prices from Upstox or pct defaults
              │
              ├─ EXIT PHASE (per open position):
              │    LTP ≤ SL price? → ExitInstruction(stop_loss)
              │    LTP ≥ TP price? → ExitInstruction(take_profit)
              │    RuleRegistry.evaluate(exit block) → ExitInstruction(exit_signal)
              │
              └─ ENTRY PHASE:
                   RuleRegistry.evaluate(entry block)  → signal bool
                   RiskManager.calculate()             → SL/TP/size
                   AccountManager.validate()           → margin/positions
                   TradeManager.build_bracket_order()  → BracketOrder
              ↓
         EvaluateExecuteResponse (entry_created | exit_triggered | no_action)
```

### 20.2 Backtest Request

```
Client → POST /api/v1/strategy/backtest
              ↓
         BacktestService
              ├─ fetch OHLCV from HISTORICAL_DATA_URL
              └─ POST {QUANT_ENGINE_URL}/run
                       ↓
                  run_backtest()
                       ├─ Parse YAML strategy
                       ├─ Normalize + trim OHLCV (to configured UTC window)
                       ├─ add_all_indicators()  → RSI, SMA, EMA, MACD, BB, ATR, VWAP columns
                       ├─ simulate_trades()     → list[Trade] + per-candle diagnostics
                       └─ calculate_metrics()  → all metrics dict
                            └─ build_assessment() → grade, labels, notes
                       ↓
                  PUT {APP_URL}/api/v1/strategy/backtest/{id}/result
```

---

*All source code references are relative to the repository root at `/home/im3/Desktop/stretus/stretus-ai/`.*
