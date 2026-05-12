# Backtesting: end-to-end flow

This document describes how backtests are started, which data they use, how trades are simulated, and how every major metric is computed in this repository. The implementation lives in the main FastAPI app under `app/` and the **quant engine** under `quant_engine/`.

---

## 1. Big-picture architecture

1. A client calls **`POST /api/v1/strategy/backtest`** with a `BacktestTriggerRequest` (strategy id, optional overrides for symbol, interval, costs, and objective, etc.).
2. The API validates the strategy, creates a **`Backtest`** row (status `running`), sets the strategy to `backtesting`, and queues a **background task**.
3. The background task:
   - Reads the strategy **YAML** from disk.
   - Builds a market-data request (symbol, interval, time window) — see [Section 2](#2-what-data-is-used-and-how-its-fetched).
   - **`GET`**s OHLCV candles from the configured **historical market data** HTTP service.
   - **POST**s a payload to the quant engine **`POST {quant_engine_url}/run`**, which returns immediately; the engine runs the backtest in a thread and **PUT**s the JSON result to **`/api/v1/strategy/backtest/{backtest_id}/result`** on the main app.
4. The API stores the result, marks the backtest `completed` or `failed`, and updates the strategy status.

Synchronous backtests are also possible for tooling by calling the quant engine’s **`POST /run-sync`**, which returns the full result in one response (no callback).

**Important files**

| Area | File |
|------|------|
| API trigger, polling, result webhook | `app/api/v1/routes/backtest.py` |
| Request/response models | `app/schemas/backtest.py` |
| Market data extraction + fetch | `app/services/backtest/market_data.py` |
| Queue to quant engine | `app/services/backtest/quant_engine_client.py` |
| Engine HTTP + async run | `quant_engine/main.py` |
| End-to-end pipeline | `quant_engine/engine/runner.py` |
| Trade simulation | `quant_engine/engine/simulator.py` |
| Metrics and pass/fail | `quant_engine/engine/metrics.py` |
| Global thresholds, date window, constants | `quant_engine/engine/config.py` |

---

## 2. What data is used and how it is fetched

### 2.1 Input: OHLCV

Each bar must include, at minimum: **timestamp**, **open**, **high**, **low**, **close**, **volume**. The fetch layer accepts dicts (flexible key names) or 6-tuples: `[timestamp, open, high, low, close, volume]`. Rows are sorted by time and basic sanity checks are applied (non-negative prices, `high`/`low` consistent with other prices, etc.).

In the engine, this becomes a **pandas** `DataFrame` indexed by UTC-normalized datetimes, columns `open, high, low, close, volume` — see `quant_engine/engine/data.py` (`load_ohlcv_data`).

### 2.2 Symbol, interval, and time window

- **Symbol and interval** come from the backtest request overrides if present, otherwise from the strategy YAML (`strategy.symbol`, `strategy.timeframe`). Symbols are normalized for the data API (e.g. `HDFCBANK.NS` → `HDFCBANK`).

- **Time window (fixed in config)**  
  All backtests are aligned to a single **inclusive UTC window** defined in `quant_engine/engine/config.py`:

  - `BACKTEST_MARKET_DATA_FROM_UTC` (e.g. `2024-01-01T00:00:00Z`)  
  - `BACKTEST_MARKET_DATA_TO_UTC` (e.g. `2026-03-31T23:59:59Z`)  

  The function `extract_strategy_market_data_request` in `app/services/backtest/market_data.py` resolves the fetch to those boundaries (user-supplied `from_utc` / `to_utc` in the trigger body do not expand beyond this product rule — the service uses the configured backtest window for the request).

- After load, the engine’s `run_backtest` **trims** the DataFrame to the same window in `_enforce_market_data_window` so the run always matches the configured range even if the caller sent extra rows.

### 2.3 How much history to request (lookback)

For UI/orchestration, `market_data.py` can compute a suggested **lookback in calendar days** from indicator periods (e.g. long SMA) so upstream fetches are large enough. The idea is: need enough bars so that after **indicator warm-up** you still have tradeable history. The formula used there is along the lines of:

- `max_period` = largest indicator lookback in the YAML (e.g. `SMA(50)` → 50; `MACD` is treated as 26+9 for period purposes).
- `total_candles_needed = max_period × 2` (safety factor).
- Convert to days using **bars per Indian session** (e.g. 375 for 1m) plus an **interval-specific buffer** (e.g. more buffer for `1d` data).

The engine separately enforces a minimum **20 candles** and warns if `len(df) - warm_up` is small compared to `warm_up`.

### 2.4 Indicator warm-up

`max_indicator_warmup` in `quant_engine/engine/indicators.py` returns the number of **initial bars to skip** before evaluating entry conditions. It is the maximum, across the YAML `indicators` block, of:

- `SMA, EMA, RSI, BB*, ATR`: the configured period `n`  
- `MACD`: `26 + 9`  

The first `warm_up_candles` rows are not used for **entry** evaluation (NaN would make comparisons unreliable). The simulator also skips them explicitly and marks them in per-candle diagnostics.

---

## 3. Strategy YAML → engine config

`load_strategy` in `quant_engine/engine/loader.py` reads a YAML file with a top-level `strategy` object. It builds a `StrategyConfig` with at least:

| Field | Meaning |
|-------|--------|
| `name`, `symbol`, `timeframe`, `objective` | `intraday` or `positional` (normalized from YAML) |
| `entry_condition`, `exit_condition` | Strings passed to the condition engine |
| `indicators` | e.g. `SMA: [20]`, `RSI: [14]` — drives columns like `SMA_20`, `RSI_14` |
| `stop_loss`, `take_profit` | Percent values used for **bar-level** stop/TP and for **exit** expression variables `STOP_LOSS_TARGET`, `TAKE_PROFIT_TARGET` if you reference them (defaults come from `risk_management` / `variables` in YAML) |
| `daily_loss_cap_pct`, `max_trades_per_day` | **Intraday** circuit breakers; defaults depend on objective |
| `max_holding_candles` | Optional cap on bars in a trade; if omitted, inferred from objective + timeframe (e.g. intraday 1m → full NSE session bars, positional 1d → 20) |

`run_config` from the API (slippage, commission, `objective`, `max_holding_candles` override, etc.) can **override** some YAML values when the pipeline runs in `runner.py`.

---

## 4. Pipeline inside `run_backtest` (quant engine)

Order of operations in `quant_engine/engine/runner.py`:

1. **`load_strategy(yaml_path)`** → `StrategyConfig`.
2. **`load_ohlcv_data(ohlcv_data)`** → sorted `DataFrame` indexed by time.
3. **`_enforce_market_data_window`** — trim to `BACKTEST_MARKET_DATA_FROM_UTC` / `TO_UTC`; error if no rows remain.
4. **`_check_data_sufficiency`** — at least 20 rows; log warning if few bars remain after warm-up.
5. **`add_all_indicators(df, cfg.indicators)`** — add indicator columns.
6. **`_ensure_common_indicators`** — if entry/exit strings reference `RSI(n)` etc. but those columns were not in the YAML `indicators` block, compute them.
7. **`simulate_trades(...)`** — produces a list of `Trade` objects and per-bar **diagnostics**.
8. **`build_backtest_result(...)`** — computes **metrics**, **pass/fail**, **assessment**, **monthly** series, **market phase** analysis, and diagnostic summary.

---

## 5. How trades are simulated (`simulate_trades`)

**Assumption:** the engine currently implements **long-only** trades (`side` is `LONG`).

### 5.1 Objective: intraday vs positional

- **Intraday:** STT on **buy = 0%**, on **sell** uses `stt_intraday_sell_pct` (default 0.025% per NSE-style intraday sell). **Session** ends when the **calendar date** of the bar changes: last bar of a session can force a **SESSION_END** exit. **Daily loss cap** and **max trades per day** apply (see below).
- **Positional (delivery-style):** STT applies on **both** buy and sell at `stt_delivery_pct` (default 0.1%). No per-day trade limit unless you set it (still only enforced when objective is intraday in the current code).

`stt_entry_pct` / `stt_exit_pct` in the simulator are chosen from the objective and these defaults.

### 5.2 Cost model (prices, not a separate cash ledger in cents)

- **Slippage + commission** are combined in **basis points (bps)**: `bps_cost = (slippage_bps + commission_bps) / 10_000`.
- **Entry** (effective price paid, long):
  - `entry = raw_open_next_bar * (1 + bps_cost + stt_entry_pct/100)`.
- **Exit** (effective price received, long):
  - `exit = raw_price * (1 - bps_cost - stt_exit_pct/100)`.

`raw` prices for fills: typically **next bar’s open** after a signal, or **stop/TP** levels, or **close** on session end / end of data, depending on exit type.

### 5.3 When entries happen

- On bar `i`, if **not** in a trade, the code evaluates the **entry condition** on bar `i` (after warm-up).
- If the condition is `True` and the trade is not blocked and it is not the last bar and (for intraday) not the last bar of the session, the **entry fill** is at the **next bar’s open** (`i+1`), with entry costs applied.
- **Blocks (intraday):** if cumulative **realized daily P&L%** (sum of `pnl_pct` for completed trades that session) is below `daily_loss_cap_pct` (negative of cap), or if `max_trades_per_day` is hit, a new entry can be **blocked** (see diagnostics: `entry_blocked_daily_cap`, `entry_blocked_max_trades`).

### 5.4 Exits: stop, take-profit, condition, time, session

On each bar while in a trade, using **unadjusted** highs/lows vs **effective** entry after costs:

- **Stop price** = `entry_price * (1 - stop_loss_pct/100)`.
- **Take-profit price** = `entry_price * (1 + take_profit_pct/100)`.
- If **low ≤ stop** and **high ≥ take_profit** on the **same** bar, the model assumes the **stop hits first** (conservative), with exit at stop after exit costs.
- Otherwise: stop or TP as usual; or evaluate **exit condition** (with `PROFIT`, `LOSS`, `TAKE_PROFIT_TARGET`, `STOP_LOSS_TARGET` from `_trade_variables`); or **max_holding_candles**; or **last bar of session** (intraday) / **end of data**.

### 5.5 Trade P&L fields

On exit:

- `pnl_inr = exit_price - entry_price` (per one unit, in price units).
- `pnl_abs = pnl_inr / entry_price` (fractional return for compounding in metrics).
- `pnl_pct = pnl_abs * 100`.
- **MAE / MFE** (long): worst/favorable move vs entry using bar **low** / **high** over the holding window:
  - `mae_pct = (min_low - entry_price) / entry_price * 100`
  - `mfe_pct = (max_high - entry_price) / entry_price * 100`

If a position is still open at the last bar, it is **force-closed** at the last close (with exit costs) with reason `END_OF_DATA`.

### 5.6 Diagnostics

Each candle can produce a record with flags such as: `warm_up_skip`, `entry_evaluated`, `entry_signal`, `entry_blocked_*`, `stop_hit`, `tp_hit`, `exit_evaluated`, `exit_signal`, etc. These feed **no-trade** hints and `diagnostic_summary` in the result.

---

## 6. How portfolio metrics are computed (`calculate_metrics`)

The metrics layer (`quant_engine/engine/metrics.py`) builds a **notional account** that **compounds** trade returns in **sequence** (trades ordered by **entry** time).

### 6.1 Compounding and ending balance

For each trade, in order:

- `pnl_value = current_balance * pnl_abs`
- `ending_balance = current_balance + pnl_value`
- The balance rolls forward to the next trade.

If there are no trades, **ending balance = starting balance**.

**Note:** `gross_return_pct` in the metrics dict is the **sum of per-trade `pnl_pct`** (percent points), not the same as compound total return, which is reflected in `ending_balance` / `net_return_pct` / `total_return_pct`.

### 6.2 Returns over the backtest window

- **`num_days`** = calendar days from `start_utc` to `end_utc` inclusive:  
  `floor(end.normalize() - start.normalize()).days + 1`.

- **`net_return_pct`** = **`total_return_pct`** =  
  `(ending_balance - starting_balance) / starting_balance * 100`.

- **`annual_return`** =  
  `((ending_balance/starting_balance)^(365/num_days) - 1) * 100`  
  (if `num_days > 0` and balance positive).

- **`average_outcome_per_trade`** = `gross_return_pct / total_trades` (average **percentage points** per trade, not the same as expectancy of fractional return unless you convert mentally).

- **`irr_daily`** =  
  `(ending/starting)^(1/num_days) - 1`.
- **`irr_annualized`** = `(1 + irr_daily)^365 - 1` (with a guard for `irr_daily <= -1`).

- **`win_rate`** = (number of trades with `pnl_pct > 0`) / `total_trades * 100`.

- **`profit_factor`**: let `gross_profit` = sum of `pnl_abs` for winning trades, `gross_loss` = absolute value of sum of `pnl_abs` for losing trades. Then  
  `gross_profit / gross_loss` if `gross_loss > 0`, else `∞` is **capped** to `PROFIT_FACTOR_MAX_CAP` in config (e.g. 9999) for JSON safety.

### 6.3 Daily portfolio curve (for risk metrics)

A **synthetic daily** equity series is built across **every calendar day** in `[start_utc, end_utc]`. Intrabar logic:

- Before the first bar of a trade, the portfolio is the **pre-trade balance** (carried from prior trades / cash).
- While in a trade, the series marks **mark-to-close** (long):  
  `balance_at_bar = starting_balance_for_trade * (close / entry_price)` on that slice, using the per-trade `starting_balance` from the compounding step (see `_daily_portfolio_values`).

The series is then **resampled to one value per calendar day** (last value of the day, forward-filled), and **daily returns** are `pct_change` of that series.

From **daily returns in decimal** (`R_t`):

- **Sharpe (annualized, 0% risk-free):**  
  `(mean(R) / std(R, ddof=1)) * sqrt(252)` when `std > 0`.
- **Sortino:** same, but the denominator uses **std of negative daily returns** only, still scaled by `sqrt(252)`.

From **daily returns in percent** (`R_pct`):

- **volatility_pct** = `std(R_pct, ddof=1) * sqrt(252)`.
- **downside_deviation_pct** = `std(negative R_pct) * sqrt(252)` (needs at least two negative days for non-zero in implementation).

- **VaR 95%** and **expected shortfall 95%** use a **historical** approach: sort daily `%` returns ascending; with `VAR_CONFIDENCE_LEVEL = 0.05`, pick the 5% tail (implementation uses index `max(1, int(n * 0.05))` and the **mean of the left tail** for ES).

- **`avg_daily_return`** = `mean(daily decimal returns) * 100`.

- **`max_drawdown`** and related: from the **daily** equity curve, running **peak**; drawdown at t is `((V_t/peak) - 1) * 100%`; the worst drawdown is the **maximum negative** excursion; detailed dates use **trough** then **peak before trough**, and **recovery** is first day after the trough when value ≥ the **pre-drawdown peak**.

- **`max_drawdown_duration`**: longest **consecutive** stretch of “underwater” (drawdown series < 0), in days in the current implementation.
- **Calmar** = `annual_return / max_drawdown` (if `max_drawdown > 0`).

- **`trades_per_month`** = `total_trades / (num_days / CALENDAR_DAYS_PER_MONTH)` with `CALENDAR_DAYS_PER_MONTH = 30.4375`.

- **`longest_losing_streak`**: max consecutive trades with `pnl_pct <= 0`.

- **`average_holding_duration`**: mean **calendar** seconds between entry and exit timestamps, divided by 86,400.

### 6.4 Monthly performance

`compute_monthly_performance` in `market_classifier.py` takes the **daily** portfolio series, resamples to **month-end** (`ME`), and computes **month-over-month** percentage returns. It also counts trades by **entry month**. Statistics include min/max month and a **return-vs-drawdown efficiency** on the **cumulative** monthly return series (not the equity drawdown from section 6.3).

### 6.5 Pass / fail (not the same as letter grade)

`PASS_FAIL_THRESHOLDS` in `config.py`:

- **intraday:** e.g. minimum **20** trades, **40%** win rate; **profit factor** not required (`min = 0`).
- **positional:** e.g. minimum **5** trades, **40%** win rate, **profit factor ≥ 1.2**.

`pass` is `True` only if all applicable thresholds for that objective are met. If not, a **`failure_reason`** string explains which rule failed, or that **no trades** were executed. Technical failures of the run produce a result with `failure_reason` like `Backtest execution failed: ...` (execution error path in `build_execution_error_result`).

The **A/B/C/…** style **assessment** (`build_assessment` in `assessment.py`) is a **separate** scoring/labeling layer for UX; it is **not** the same boolean as `pass` above (you can have metrics with a grade and still fail the objective thresholds).

---

## 7. What you get in the API payload

`BacktestResultPayload` in `app/schemas/backtest.py` mirrors the engine output:

- **metrics:** balances, return stats, risk ratios, drawdown, trades list with MAE/MFE, etc.  
- **assessment:** grade and narrative labels.  
- **monthly_performance / monthly_statistics**  
- **market_phase_analysis:** per-quarter (or period) view vs benchmark-style classification in `market_classifier`  
- **config:** including `warm_up_candles`, cost parameters, and enforced window  
- **pass** / **failure_reason**  
- **diagnostic_summary** (and internal condition diagnostics if exposed)

---

## 8. Worked example (simplified, positional)

**Strategy (conceptually similar to** `strategies/hdfcbank_ns_1m_strategy.yaml` **):**

- **Entry:** `RSI(14) > 50` and `close > EMA(20)`.
- **Exit:** e.g. when `RSI(14) < 50` or when stop/TP or max-holding or **session** rules apply (exact behavior depends on YAML + objective).

**Steps with round numbers**

1. **History** is loaded for the configured backtest window (Q1 2026 in the current `config` constants).
2. **Warm-up:** the first `max(14, 20) = 20` bars (from RSI/EMA periods) are skipped for **new entries**; indicators fill in.
3. On bar 50, entry condition is **true**; **entry** fills on bar 51 at **open** of bar 51 with slippage, commission, and (positional) **STT on buy**.
4. A few bars later, price hits the **stop**; exit fills with exit costs and sell-side STT. The trade’s `pnl_abs` is negative.
5. Another trade later is a **winner**. The second trade’s P&L is applied to the **balance left after the first** — that is the **compounding** rule in `_trade_snapshots`.
6. **Metrics:** win rate, profit factor, Sharpe, etc. are computed from the **trade list** and the **daily** reconstructed equity curve.
7. If objective is **positional** and the strategy has enough trades, high enough win rate, and **profit factor ≥ 1.2**, **`pass: true`**.

This matches the design: **strategic rules** in YAML, **operational frictions** (bps + STT) in the simulator, and **account-level reporting** in `metrics.py`.

---

## 9. Changing the backtest calendar window

The live window is **one place** in `quant_engine/engine/config.py` (`BACKTEST_MARKET_DATA_FROM_UTC` / `BACKTEST_MARKET_DATA_TO_UTC`). The fetch layer in `app/services/backtest/market_data.py` imports the same symbols so the **HTTP** request and the **engine** trim use the same bounds.

---

## 10. Glossary (quick)

| Term | Here it means |
|------|----------------|
| bps | Basis points; `1 bps = 0.01%` |
| Warm-up | Earliest bars skipped so indicators are defined |
| `pnl_abs` | Per-trade fractional P&L on notional, used to compound the account |
| `pnl_pct` | `pnl_abs * 100` (percentage **points** for one trade) |
| Objective | `intraday` vs `positional` — changes STT, session, and pass thresholds |

---

## 11. Backtest response field reference (how each value is calculated)

**Source of truth in code:** `quant_engine/engine/metrics.py` (`calculate_metrics`, `_drawdown_stats`, trade serialization), `quant_engine/engine/assessment.py`, `quant_engine/engine/market_classifier.py`, `quant_engine/engine/config.py`. The API schema is `app/schemas/backtest.py`.

**Foundation:** Per-trade P&L comes from `simulator.py` (`pnl_abs`, `pnl_pct` after costs). The account **compounds** in `_trade_snapshots` (`pnl_value = balance × pnl_abs`). A **synthetic daily equity** curve is built in `_daily_portfolio_values` from bar closes while in a trade, then resampled to **one value per calendar day** from `start_utc` to `end_utc`. Most risk/Sharpe/vol uses **that daily series**.

### 11.1 `metrics` — “Key performance” and headline numbers

| Field / UI label | Calculation |
|------------------|------------|
| **Total return** (`total_return_pct`, `net_return_pct`, same value) | `(ending_balance - starting_balance) / starting_balance × 100` after compounding all trades. |
| **Gross return** (`gross_return_pct`) | Sum of each trade’s `pnl_pct` (percentage **points**); *not* the same as compound total return. |
| **Win rate** | `100 × (wins / total_trades)` where a win is `pnl_pct > 0`. |
| **Profit factor** (“Profile factor” in UI is usually this) | `sum(pnl_abs for winning trades) / abs(sum(pnl_abs for losing trades))` with cap `PROFIT_FACTOR_MAX_CAP` if loss side is 0. |
| **Average outcome per trade** (`average_outcome_per_trade`) | `gross_return_pct / total_trades` (avg points per trade; not R-multiple style expectancy). |
| **Sharpe ratio** | `mean(daily R) / std(daily R, ddof=1) × sqrt(252)` on **decimal** daily returns, risk-free 0. |
| **Sortino ratio** | Like Sharpe, but denominator is std of **negative** daily returns only, same annualization. |
| **Max drawdown** | From daily portfolio: `running_peak = cummax(values)`; drawdown at t = `(V/peak - 1)×100%`; **max drawdown** = absolute value of the most negative. |
| **Max drawdown duration** (`max_drawdown_duration`) | **Longest consecutive days** with drawdown &lt; 0 (streak in `_drawdown_stats`), not the depth of the single worst move. |
| **Volatility** (`volatility_pct`) | `std(daily return %, ddof=1) × sqrt(252)` on the **%** daily returns. |
| **Downside deviation** (`downside_deviation_pct`) | `std(negative daily % only) × sqrt(252)` (needs ≥2 negative days). |
| **VaR 95%** / **Expected shortfall 95%** | Sort daily **%** returns ascending; 5% tail: VaR = value at cutoff index; ES = **mean** of the left tail. |
| **Calmar ratio** | `annual_return / max_drawdown` (if `max_drawdown &gt; 0`). |
| **Annual return** | `((ending/starting)^(365/num_days) - 1) × 100`. |
| **avg_daily_return** | `mean(daily decimal returns) × 100`. |
| **num_days** | Calendar **inclusive** day count between `start_utc` and `end_utc` dates. |
| **trades_per_month** | `total_trades / (num_days / 30.4375)`. |
| **longest_losing_streak** | Longest run of consecutive trades with `pnl_pct ≤ 0` (chronological by entry). |
| **average_holding_duration** | Mean of `(exit - entry)` in **calendar** days. |

### 11.2 Worst drawdown period (same block as `metrics`)

| Field | Calculation |
|-------|-------------|
| `worst_drawdown_start_date` | Date of the **running peak** of portfolio value that precedes the **trough** where drawdown is worst. |
| `worst_drawdown_end_date` | Date of the **trough** (min of the percent drawdown series from running peak). |
| `drawdown_duration_days` | **Calendar** days from peak date to **trough** date (the depth of the *single* worst leg). |
| `recovery_date` | First day **on or after** the trough when portfolio ≥ that **pre-drawdown** peak. |
| `recovery_time_days` | Calendar days from trough to `recovery_date`. |

### 11.3 Monthly performance and `monthly_statistics` (`PerformanceStatistics`)

- **monthly_performance[]:** Month-end values of the same daily portfolio series; **monthly % return** = `pct_change` of month-ends × 100. **trades_count** = entries whose `entry_date` falls in that **YYYY-MM**.

- **monthly_statistics** includes: **highest_monthly_gain_pct** / **lowest_monthly_gain_pct** / **range**; **return_vs_drawdown_efficiency** = (sum of monthly % returns) / a **cumulative** “monthly” drawdown built by walking monthly returns (see `compute_monthly_performance` in `market_classifier.py`).

### 11.4 Strategy assessment (`assessment` object)

Generated by `build_assessment` in `assessment.py`. It **does not** recompute the Sharpe, etc.; it **reads** the `metrics` dict and applies **thresholds from** `config.py`.

| Field | How it is set |
|-------|----------------|
| `overall_grade` (A, B+, B, C+, C, D) | Sum of sub-scores (0–100): return (0–30), Sharpe (0–25), drawdown (0–25), **consistency** = profit_factor + win_rate rules (0–15), sample size (0–5). Thresholds: `GRADE_*_MIN_SCORE`. If `total_trades == 0` → **D**. |
| `return_potential` | “Strong” / “Moderate” / “Weak” from `total_return_pct`, `sharpe_ratio`, and trade count (see `RETURN_POTENTIAL_*` in config). |
| `risk_profile` | “Conservative” / “Moderate” / “Aggressive” from `max_drawdown`, `volatility_pct`, and `var_95_pct` vs `RISK_*` thresholds. |
| `drawdown_tolerance_required` | “Low” / “Medium” / “High” from `max_drawdown` and `recovery_time_days` vs `DRAWDOWN_TOLERANCE_*`. |
| `recommended_for` | e.g. “Not Recommended” if return ≤0 or Sharpe ≤0; else rules combining risk/tolerance and `trades_per_month` vs `RECOMMENDED_*` constants. |
| `notes` | Templated text from `descriptions.build_assessment_notes`. |

### 11.5 Pass/fail (not the letter grade)

`build_backtest_result` in `metrics.py` sets `pass` and `failure_reason` using `PASS_FAIL_THRESHOLDS[objective]` (min trades, min win %, min profit factor for positional). This is **independent** of `overall_grade`.

### 11.6 Performance by market condition (`market_phase_analysis`)

`build_market_phase_analysis` in `market_classifier.py` splits the OHLCV `DataFrame` by **calendar quarter** (`QE`).

Per quarter: **price_change_pct** = first-to-last **close** % in that slice. **market_type** = `classify_market_type` (Bull if return ≥ +8%, Bear if ≤ -8% else Sideways, from `BULL_MARKET_MIN_RETURN_PCT` / `BEAR_MARKET_MAX_RETURN_PCT`). **market_phase** = `classify_market_phase` (slope of **normalized** close: Uptrend / Downtrend / Range-bound vs `TREND_SLOPE_*`).

**strategy_trades**, **strategy_win_rate_%**, **strategy_return_%** are computed on trades with **entry** in that quarter. **observed_alignment** = `classify_alignment(strategy_side, market_type, phase_win_rate)` (lookup table + win-rate nudge). **description** = `build_phase_description` text template.

### 11.7 `backtest_trades[]` (per trade)

Prices and `pnl_*` from simulation. **entry_market_condition** = `classify_entry_condition` at the entry bar: compares **close** to fast (10) and slow (30) **means of prior** closes. **mae** / **mfe** = worst / best % vs entry from lows/highs in the position window (see `simulator.py`).

This README reflects the code as of the repository; if you add short selling, a different compounding model, or risk-free Sharpe, update the corresponding modules and this document together.
