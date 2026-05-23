# Quant Engine Timeout Fix 🔧

## Problem

```
ERROR | ❌ backtest.quant_engine_client | Failed to reach quant engine for synchronous backtest
httpx.ReadTimeout
```

The backtest is taking longer than the configured timeout (was 180 seconds / 3 minutes).

## Root Causes

1. **Complex backtests** - Large date ranges or complex strategies take time
2. **Data fetching delays** - Historical data API might be slow
3. **Too short timeout** - Default 180s isn't enough for heavy backtests

## Solutions Applied

### 1. Increased Timeout Settings

**Updated `app/core/config.py`:**
```python
quant_engine_timeout_seconds: float = 600.0  # 10 minutes (was 180s)
historical_data_timeout_seconds: float = 120.0  # 2 minutes (was 60s)
```

**Updated `.env` and `.env.example`:**
```env
QUANT_ENGINE_TIMEOUT_SECONDS=600
HISTORICAL_DATA_TIMEOUT_SECONDS=120
```

### 2. How to Apply the Fix

#### Option A: Restart Containers (Quick)
```bash
docker compose restart stretus_api
```

#### Option B: Rebuild (If restart doesn't work)
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### 3. Adjust Timeout Based on Your Needs

Edit `.env` file:

```env
# For very complex backtests (15+ minutes)
QUANT_ENGINE_TIMEOUT_SECONDS=900

# For simple backtests (5 minutes)
QUANT_ENGINE_TIMEOUT_SECONDS=300

# For extremely long backtests (30 minutes)
QUANT_ENGINE_TIMEOUT_SECONDS=1800
```

Then restart:
```bash
docker compose restart stretus_api
```

## Additional Optimizations

### 1. Reduce Backtest Date Range

In your `.env`, adjust the lookback period:

```env
# Reduce from 868 days to something smaller
BACKTEST_DEFAULT_LOOKBACK_DAYS=180  # 6 months instead of 2+ years

# Reduce signal evaluation lookback
SIGNAL_EVAL_LOOKBACK_DAYS=30  # Keep this small
```

### 2. Use Smaller Fetch Chunks

```env
# Fetch data in smaller chunks to avoid timeouts
BACKTEST_FETCH_CHUNK_DAYS=30  # Was 90, now 30
```

### 3. Check Historical Data API

Make sure your historical data API is responding quickly:

```bash
# Test the API
curl -w "\nTime: %{time_total}s\n" https://dev-api.stretus.com/health
```

If it's slow (>5 seconds), that's your bottleneck.

### 4. Monitor Quant Engine Logs

Check what the quant engine is doing:

```bash
docker compose logs -f stretus_quant_engine
```

Look for:
- Data fetching delays
- Processing bottlenecks
- Memory issues

## Recommended Settings

### For Development (Fast Iteration)
```env
QUANT_ENGINE_TIMEOUT_SECONDS=300
BACKTEST_DEFAULT_LOOKBACK_DAYS=90
BACKTEST_FETCH_CHUNK_DAYS=30
SIGNAL_EVAL_LOOKBACK_DAYS=30
```

### For Production (Comprehensive Backtests)
```env
QUANT_ENGINE_TIMEOUT_SECONDS=900
BACKTEST_DEFAULT_LOOKBACK_DAYS=365
BACKTEST_FETCH_CHUNK_DAYS=60
SIGNAL_EVAL_LOOKBACK_DAYS=60
```

### For Quick Testing (Fastest)
```env
QUANT_ENGINE_TIMEOUT_SECONDS=180
BACKTEST_DEFAULT_LOOKBACK_DAYS=30
BACKTEST_FETCH_CHUNK_DAYS=15
SIGNAL_EVAL_LOOKBACK_DAYS=15
```

## Verification

After applying the fix, you should see:

```
✅ Backtest completed successfully
✅ No ReadTimeout errors
```

## If Still Timing Out

### 1. Check Quant Engine Health
```bash
curl http://localhost:8001/health
```

### 2. Check Container Resources
```bash
docker stats
```

Look for:
- High CPU usage (>90%)
- High memory usage (>80%)
- Container restarts

### 3. Increase Docker Resources

Edit Docker Desktop settings:
- **Memory**: Increase to 8GB+ (was 4GB)
- **CPU**: Increase to 4+ cores (was 2)

### 4. Check Network Connectivity

```bash
# From inside the API container
docker compose exec stretus_api curl http://quant_engine:8001/health
```

Should return quickly (<1 second).

## Understanding the Timeout Flow

```
User Request → API Service → Quant Engine → Historical Data API
                   ↓              ↓                ↓
              600s timeout   Processing      Fetching OHLCV
                                              (120s timeout)
```

If any step takes too long, you'll get a timeout.

## Quick Fix Summary

**Immediate fix:**
```bash
# Just restart
docker compose restart stretus_api
```

**If that doesn't work:**
```bash
# Rebuild
docker compose down
docker compose build --no-cache
docker compose up
```

**Adjust timeout in `.env` if needed:**
```env
QUANT_ENGINE_TIMEOUT_SECONDS=600  # Increase if still timing out
```

---

**The timeout has been increased from 3 minutes to 10 minutes.** This should handle most backtests. If you still see timeouts, increase it further in `.env`. 🚀
