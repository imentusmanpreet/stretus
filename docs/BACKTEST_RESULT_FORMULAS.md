# Backtest Result Formulas

This file lists the formulas used to calculate the fields in the backtest result payload.

Source files:

- `quant_engine/engine/simulator.py`
- `quant_engine/engine/metrics.py`
- `quant_engine/engine/assessment.py`
- `quant_engine/engine/market_classifier.py`
- `quant_engine/engine/config.py`

## 1. Trade-Level Calculations

### Cost Model

```text
bps_cost = (slippage_bps + commission_bps) / 10000
```

Entry price after buy-side costs:

```text
entry_price = raw_entry_price * (1 + bps_cost + stt_entry_pct / 100)
```

Exit price after sell-side costs:

```text
exit_price = raw_exit_price * (1 - bps_cost - stt_exit_pct / 100)
```

STT rules:

```text
intraday:
  stt_entry_pct = 0.0
  stt_exit_pct  = 0.025

positional / delivery:
  stt_entry_pct = 0.1
  stt_exit_pct  = 0.1
```

### Stop Loss and Take Profit

```text
stop_price        = entry_price * (1 - stop_loss_pct / 100)
take_profit_price = entry_price * (1 + take_profit_pct / 100)
```

If stop loss and take profit are both touched in the same candle, the engine assumes the stop loss was hit first.

### Trade P&L

```text
pnl_inr = exit_price - entry_price
pnl_abs = pnl_inr / entry_price
pnl_pct = pnl_abs * 100
```

`pnl_abs` is the fractional return used for compounding.

Example:

```text
pnl_abs = 0.05 means 5% gain
pnl_pct = 5.0
```

### MAE and MFE

Maximum adverse excursion:

```text
max_adverse_excursion_pct =
  (lowest_low_during_trade - entry_price) / entry_price * 100
```

Maximum favorable excursion:

```text
max_favorable_excursion_pct =
  (highest_high_during_trade - entry_price) / entry_price * 100
```

### Holding Duration

```text
holding_duration_days =
  (exit_timestamp - entry_timestamp).total_seconds() / 86400
```

### Holding Candles

```text
holding_candles = exit_candle_index - entry_candle_index
```

## 2. Balance Compounding

Trades are applied sequentially by entry date.

```text
pnl_value      = balance_before_trade * trade.pnl_abs
ending_balance = balance_before_trade + pnl_value
next_balance   = ending_balance
```

Equivalently:

```text
ending_balance = balance_before_trade * (1 + trade.pnl_abs)
```

## 3. Basic Metrics

### Number of Days

Calendar day count is inclusive.

```text
num_days = (end_date - start_date).days + 1
```

### Total Trades

```text
total_trades = count(trades)
```

### Wins

```text
wins = count(trades where pnl_pct > 0)
```

Break-even trades are not wins.

### Win Rate

```text
win_rate = wins / total_trades * 100
```

If there are no trades:

```text
win_rate = 0
```

### Ending Balance

```text
ending_balance = final compounded balance after all trades
```

If there are no trades:

```text
ending_balance = starting_balance
```

## 4. Return Metrics

### Net Return / Total Return

`total_return_pct` and `net_return_pct` are the same value.

```text
net_return_pct =
  (ending_balance - starting_balance) / starting_balance * 100

total_return_pct = net_return_pct
```

### Gross Return

```text
gross_return_pct = sum(trade.pnl_pct for all trades)
```

Note: this is the sum of individual trade percentage returns. It is not compounded.

### Average Outcome Per Trade

```text
average_outcome_per_trade = gross_return_pct / total_trades
```

If there are no trades:

```text
average_outcome_per_trade = 0
```

### Annual Return

```text
annual_return =
  ((ending_balance / starting_balance) ^ (365 / num_days) - 1) * 100
```

If `starting_balance <= 0` or `num_days <= 0`:

```text
annual_return = 0
```

### Daily IRR

```text
irr_daily =
  (ending_balance / starting_balance) ^ (1 / num_days) - 1
```

If `starting_balance <= 0`, `ending_balance <= 0`, or `num_days <= 0`:

```text
irr_daily = 0
```

### Annualized IRR

```text
irr_annualized = (1 + irr_daily) ^ 365 - 1
```

If `irr_daily <= -1`:

```text
irr_annualized = -1
```

## 5. Profit Factor

```text
gross_profit = sum(trade.pnl_abs where trade.pnl_abs > 0)
gross_loss   = abs(sum(trade.pnl_abs where trade.pnl_abs < 0))
```

```text
profit_factor = gross_profit / gross_loss
```

Special cases:

```text
if gross_loss > 0:
  profit_factor = gross_profit / gross_loss

if gross_loss == 0 and gross_profit > 0:
  profit_factor = 9999

if gross_loss == 0 and gross_profit == 0:
  profit_factor = 0
```

`9999` is the cap used to avoid infinity in JSON.

## 6. Daily Portfolio Series

The engine creates a synthetic daily equity curve.

During a trade:

```text
portfolio_value_at_bar =
  trade_starting_balance * (bar_close / entry_price)
```

Outside a trade:

```text
portfolio_value_at_bar = last_realized_balance
```

Then:

```text
daily_values = last portfolio value of each calendar day
```

Missing calendar days are forward-filled. If no value exists yet, `starting_balance` is used.

### Daily Returns

Decimal daily return:

```text
daily_returns = daily_values.pct_change().fillna(0)
```

Percentage daily return:

```text
daily_returns_pct = daily_returns * 100
```

Average daily return:

```text
avg_daily_return = mean(daily_returns) * 100
```

## 7. Risk-Adjusted Metrics

### Sharpe Ratio

Risk-free rate is treated as `0`.

```text
sharpe_ratio =
  mean(daily_returns) / std(daily_returns, ddof=1) * sqrt(252)
```

If standard deviation is `0`:

```text
sharpe_ratio = 0
```

### Sortino Ratio

```text
downside_returns = daily_returns where daily_returns < 0
```

```text
sortino_ratio =
  mean(daily_returns) / std(downside_returns, ddof=1) * sqrt(252)
```

If downside standard deviation is `0`:

```text
sortino_ratio = 0
```

### Calmar Ratio

```text
calmar_ratio = annual_return / max_drawdown
```

If `max_drawdown <= 0`:

```text
calmar_ratio = 0
```

## 8. Volatility and Downside Deviation

### Annualized Volatility

```text
volatility_pct =
  std(daily_returns_pct, ddof=1) * sqrt(252)
```

If fewer than 2 daily return values exist:

```text
volatility_pct = 0
```

### Annualized Downside Deviation

```text
negative_daily_returns_pct =
  daily_returns_pct where daily_returns_pct < 0
```

```text
downside_deviation_pct =
  std(negative_daily_returns_pct, ddof=1) * sqrt(252)
```

If fewer than 2 negative daily return values exist:

```text
downside_deviation_pct = 0
```

## 9. VaR and Expected Shortfall

The engine uses the 5% left tail of daily percentage returns.

```text
sorted_returns = sort(daily_returns_pct ascending)
cutoff_idx     = max(1, int(len(sorted_returns) * 0.05))
```

### VaR 95%

```text
var_95_pct = sorted_returns[cutoff_idx - 1]
```

### Expected Shortfall 95%

```text
expected_shortfall_95_pct =
  mean(sorted_returns[0 : cutoff_idx])
```

Both values are usually negative when they represent loss.

## 10. Drawdown Metrics

### Running Peak

```text
running_peak = cumulative_max(daily_values)
```

### Drawdown Series

```text
drawdown_pct = (daily_values / running_peak - 1) * 100
```

### Max Drawdown

```text
max_drawdown = abs(min(drawdown_pct))
```

If there is no drawdown:

```text
max_drawdown = 0
```

### Worst Drawdown Dates

```text
trough_date = date where drawdown_pct is minimum
peak_date   = MOST RECENT date on/before trough_date where drawdown_pct == 0
              (i.e. the last equity high the curve declined from)
```

Note: `peak_date` is the *last* new high before the trough, not the earliest
occurrence of the peak value. When an equity curve revisits the same high
several times before its worst drop, picking the earliest occurrence (e.g. via
`idxmax`) would overstate the drawdown duration.

```text
worst_drawdown_start_date = peak_date
worst_drawdown_end_date   = trough_date
```

### Drawdown Duration Days

```text
drawdown_duration_days = (trough_date - peak_date).days
```

This is the duration of the worst peak-to-trough drawdown leg.

### Recovery Date

```text
recovery_date =
  first date on or after trough_date where daily_value >= peak_value
```

If recovery does not happen:

```text
recovery_date = null
```

### Recovery Time Days

```text
recovery_time_days = (recovery_date - trough_date).days
```

If there is no recovery (curve still underwater at the end of the window):

```text
recovery_time_days = null
```

`null` (not `0`) signals "never recovered" so risk classification treats it as
worst-case rather than as an instant recovery.

### Max Drawdown Duration

```text
max_drawdown_duration =
  longest consecutive day streak where drawdown_pct < 0
```

This is the longest underwater period.

## 11. Trade Activity Metrics

### Average Holding Duration

```text
average_holding_duration =
  mean((exit_timestamp - entry_timestamp).total_seconds() / 86400)
```

If there are no trades:

```text
average_holding_duration = 0
```

### Trades Per Month

```text
trades_per_month = total_trades / (num_days / 30.4375)
```

`30.4375` is the average calendar days per month.

### Longest Losing Streak

Trades are sorted by entry date.

```text
longest_losing_streak =
  longest consecutive sequence where trade.pnl_pct <= 0
```

Break-even trades count as losing trades for this streak.

## 12. Monthly Performance

### Monthly Returns

```text
monthly_values =
  daily_values.resample(month_end).last()
```

```text
monthly_return_pct =
  monthly_values.pct_change().fillna(0) * 100
```

### Monthly Trades Count

```text
trades_count =
  count(trades where entry_date falls in that YYYY-MM month)
```

### Monthly Statistics

```text
highest_monthly_gain_pct = max(monthly_return_pct)
lowest_monthly_gain_pct  = min(monthly_return_pct)
```

```text
monthly_performance_range_pct =
  highest_monthly_gain_pct - lowest_monthly_gain_pct
```

Monthly return-vs-drawdown efficiency:

```text
running_monthly_return = cumulative sum of monthly_return_pct
monthly_peak           = cumulative max of running_monthly_return
monthly_drawdown       = max(monthly_peak - running_monthly_return)
```

```text
return_vs_drawdown_efficiency =
  sum(monthly_return_pct) / monthly_drawdown
```

If monthly drawdown is `0`:

```text
return_vs_drawdown_efficiency = 0
```

## 13. Market Phase Analysis

The OHLCV data is grouped by calendar quarter.

### Price Change

```text
price_change_pct =
  (last_close - first_close) / first_close * 100
```

If `first_close == 0`:

```text
price_change_pct = 0
```

### Market Type

```text
if price_change_pct >= 8:
  market_type = "Bull"

elif price_change_pct <= -8:
  market_type = "Bear"

else:
  market_type = "Sideways"
```

### Market Phase

Close prices are normalized first:

```text
price_range = max(close) - min(close)
normalized_close = (close - min(close)) / price_range
```

A linear regression slope is calculated:

```text
slope = polyfit(x, normalized_close, degree=1)[0]
```

Classification:

```text
if slope >= 0.0003:
  market_phase = "Uptrend"

elif slope <= -0.0003:
  market_phase = "Downtrend"

else:
  market_phase = "Range-bound"
```

If there are fewer than 2 closes, or the price range is almost zero:

```text
market_phase = "Range-bound"
```

### Quarter Strategy Trades

```text
strategy_trades =
  count(trades where entry_date is inside that quarter)
```

### Quarter Strategy Win Rate

```text
strategy_win_rate_pct =
  count(quarter trades where pnl_pct > 0) / strategy_trades * 100
```

If there are no quarter trades:

```text
strategy_win_rate_pct = 0
```

### Quarter Strategy Return

```text
strategy_return_pct =
  sum(pnl_pct for trades entered in that quarter)
```

### Observed Alignment

Base matrix:

| Strategy Side | Market Type | Base Alignment |
| --- | --- | --- |
| LONG | Bull | Strong |
| LONG | Sideways | Moderate |
| LONG | Bear | Weak |
| SHORT | Bear | Strong |
| SHORT | Sideways | Moderate |
| SHORT | Bull | Weak |

If base alignment is `Moderate`, it is adjusted using quarter win rate:

```text
if strategy_win_rate_pct >= 60:
  observed_alignment = "Strong"

elif strategy_win_rate_pct <= 30:
  observed_alignment = "Weak"

else:
  observed_alignment = "Moderate"
```

## 14. Entry Market Condition

For each trade, entry condition is classified using the entry bar.

If there are fewer than 30 previous bars:

```text
entry_market_condition = "Unknown"
```

Otherwise:

```text
fast_avg = mean(close[entry_bar_index - 10 : entry_bar_index])
slow_avg = mean(close[entry_bar_index - 30 : entry_bar_index])
current  = close[entry_bar_index]
```

Classification:

```text
if current > fast_avg > slow_avg:
  entry_market_condition = "Bull"

elif current < fast_avg < slow_avg:
  entry_market_condition = "Bear"

else:
  entry_market_condition = "Sideways"
```

## 15. Assessment Fields

Assessment fields are labels derived from the calculated metrics.

### Overall Grade

```text
score =
  return_score
  + sharpe_score
  + drawdown_score
  + consistency_score
  + sample_size_score
```

If there are no trades:

```text
overall_grade = "D"
```

### Return Score

| `total_return_pct` | Score |
| --- | ---: |
| `>= 20` | 30 |
| `>= 12` | 24 |
| `>= 5` | 16 |
| `>= 0` | 8 |
| `< 0` | 0 |

### Sharpe Score

| `sharpe_ratio` | Score |
| --- | ---: |
| `>= 1.5` | 25 |
| `>= 1.0` | 18 |
| `>= 0.5` | 10 |
| `> 0` | 5 |
| `<= 0` | 0 |

### Drawdown Score

| `max_drawdown` | Score |
| --- | ---: |
| `<= 8` | 25 |
| `<= 12` | 18 |
| `<= 15` | 12 |
| `<= 20` | 8 |
| `<= 30` | 4 |
| `> 30` | 0 |

### Consistency Score

| Condition | Score |
| --- | ---: |
| `profit_factor >= 1.5` and `win_rate >= 60` | 15 |
| `profit_factor >= 1.2` and `win_rate >= 50` | 12 |
| `profit_factor >= 1.0` and `win_rate >= 45` | 8 |
| `profit_factor >= 0.8` | 4 |
| otherwise | 0 |

### Sample Size Score

| `total_trades` | Score |
| --- | ---: |
| `>= 100` | 5 |
| `>= 40` | 4 |
| `>= 20` | 2 |
| `< 20` | 0 |

### Grade Mapping

| Score | Grade |
| --- | --- |
| `>= 85` | A |
| `>= 72` | B+ |
| `>= 62` | B |
| `>= 52` | C+ |
| `>= 42` | C |
| `< 42` | D |

### Return Potential

```text
if total_return_pct >= 15 and sharpe_ratio >= 1.0 and total_trades >= 20:
  return_potential = "Strong"

elif total_return_pct >= 5 and total_trades >= 10:
  return_potential = "Moderate"

else:
  return_potential = "Weak"
```

### Risk Profile

```text
if max_drawdown <= 8 and volatility_pct <= 12 and var_95_pct >= -2:
  risk_profile = "Conservative"

elif max_drawdown <= 15 and volatility_pct <= 20 and var_95_pct >= -4:
  risk_profile = "Moderate"

else:
  risk_profile = "Aggressive"
```

### Drawdown Tolerance Required

```text
if max_drawdown <= 8 and recovery_time_days <= 30:
  drawdown_tolerance_required = "Low"

elif max_drawdown <= 15 and recovery_time_days <= 60:
  drawdown_tolerance_required = "Medium"

else:
  drawdown_tolerance_required = "High"
```

### Recommended For

```text
if total_return_pct <= 0 or sharpe_ratio <= 0:
  recommended_for = "Not Recommended"

elif risk_profile == "Aggressive"
  or drawdown_tolerance_required == "High"
  or trades_per_month >= 20:
  recommended_for = "Experienced Traders"

elif risk_profile == "Moderate"
  or drawdown_tolerance_required == "Medium"
  or trades_per_month >= 8:
  recommended_for = "Intermediate Traders"

else:
  recommended_for = "Beginner Traders"
```

## 16. Pass / Fail

Pass/fail is separate from `overall_grade`.

### Intraday

```text
pass =
  total_trades > 0
  and win_rate >= 40
```

Profit factor is not enforced for intraday.

### Positional

```text
pass =
  total_trades > 0
  and win_rate >= 40
  and profit_factor >= 1.2
```

### Failure Reason

If no trades:

```text
failure_reason =
  strategy failed because no trades were executed during the backtest window
```

If win rate is below threshold:

```text
failure_reason =
  strategy failed because win_rate is below required threshold
```

If positional profit factor is below threshold:

```text
failure_reason =
  strategy failed because profit_factor is below 1.2
```

Otherwise:

```text
failure_reason = ""
```

## 17. Empty / Error Result Defaults

When no data is available or the result is built as an execution error, calculated numeric fields default to zero except balances.

```text
starting_balance = configured starting_balance
ending_balance   = configured starting_balance
total_trades     = 0
win_rate         = 0
all return/risk metrics = 0
backtest_trades  = []
monthly_performance = []
monthly_statistics  = {}
market_phase_analysis = []
```

## 18. Rounding

Most metric values are rounded before being returned.

```text
default numeric rounding = 4 decimal places
balance rounding         = 2 decimal places
trades_per_month         = 2 decimal places
pnl_abs in trades        = 6 decimal places
```

