"""
engine/indicators.py
═════════════════════
Calculates all technical indicators supported by Stretus.

Supported indicators:
  SMA(n)         — Simple Moving Average
  EMA(n)         — Exponential Moving Average
  RSI(n)         — Relative Strength Index
  MACD           — MACD line (12, 26, 9)
  BB_UPPER(n)    — Bollinger Band Upper (n period, 2 std dev)
  BB_LOWER(n)    — Bollinger Band Lower (n period, 2 std dev)
  ATR(n)         — Average True Range (Wilder's smoothing)
  VWAP           — Session VWAP (intraday: cumulative; daily bars: typical price per bar)
  SUPERTREND     — Supertrend (period, multiplier) — direction column (+1/-1) and line value
  PREV_DAY_HIGH  — Previous calendar day's high, forward-filled to intraday bars
  PREV_DAY_LOW   — Previous calendar day's low, forward-filled to intraday bars

VWAP note:
  On intraday timeframes (sub-daily index), VWAP is computed as the
  cumulative (price × volume) / cumulative volume within each trading session
  (session = calendar day).

  On daily timeframes (where each bar already represents one full session),
  VWAP reduces to the typical price (H+L+C)/3 — this IS the VWAP for
  the completed session, since there is no intrabar granularity to cumulate.
  Using the full-history cumulative approach on daily bars is semantically wrong
  and produces a lagging moving-average-like value, NOT session VWAP.

All functions accept pandas Series/DataFrame and return a Series.
"""

import pandas as pd
import numpy as np


# ─── Core indicators ──────────────────────────────────────────────────────────

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average over n periods."""
    return series.rolling(window=n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average over n periods."""
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing method).
    Returns values between 0 and 100.
    """
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd_line(series: pd.Series) -> pd.Series:
    """MACD line = EMA(12) - EMA(26)."""
    return ema(series, 12) - ema(series, 26)


def macd_signal(series: pd.Series) -> pd.Series:
    """MACD signal line = EMA(9) of MACD line."""
    return ema(macd_line(series), 9)


def bb_upper(series: pd.Series, n: int = 20) -> pd.Series:
    """Bollinger Band Upper = SMA(n) + 2 * std(n)."""
    mid = sma(series, n)
    std = series.rolling(window=n, min_periods=n).std()
    return mid + (2 * std)


def bb_lower(series: pd.Series, n: int = 20) -> pd.Series:
    """Bollinger Band Lower = SMA(n) - 2 * std(n)."""
    mid = sma(series, n)
    std = series.rolling(window=n, min_periods=n).std()
    return mid - (2 * std)


def bb_middle(series: pd.Series, n: int = 20) -> pd.Series:
    """Bollinger Band Middle = SMA(n)."""
    return sma(series, n)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = EWM of TR with Wilder's alpha = 1/n
    """
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ─── VWAP ─────────────────────────────────────────────────────────────────────

def _is_daily_timeframe(df: pd.DataFrame) -> bool:
    """
    Detect whether the DataFrame represents daily (or lower frequency) bars.

    We check by sampling the median timedelta between consecutive bars:
    - If the median gap >= 6 hours, it's daily or coarser.
    - If the index is not a DatetimeIndex, fall back to False (treat as intraday).
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return False
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return False
    median_gap = deltas.median()
    return median_gap >= pd.Timedelta(hours=6)


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP calculation, automatically adapted to timeframe:

    Intraday bars (gap < 6h between bars):
        Session-aware cumulative VWAP groupped by calendar day.
        VWAP_t = Σ(TP × Vol, session) / Σ(Vol, session)
        where TP = (High + Low + Close) / 3

    Daily bars (gap >= 6h between bars):
        Each bar IS its own session. VWAP for a completed daily session
        is simply the session's typical price = (H + L + C) / 3.
        Cumulating across multiple days does NOT produce session VWAP;
        it produces a volume-weighted moving average (different concept).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].fillna(0.0)

    if _is_daily_timeframe(df):
        # Daily bars: VWAP = typical price for that session (H+L+C)/3
        # If volume is meaningful, weight by volume within the bar (still just TP
        # for a single bar since there is only one data point per session).
        return typical_price

    # Intraday: proper session-cumulated VWAP
    if isinstance(df.index, pd.DatetimeIndex):
        session_key = df.index.normalize()
        pv = (typical_price * volume).groupby(session_key).cumsum()
        cumulative_volume = volume.groupby(session_key).cumsum()
    else:
        pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

    safe_volume = cumulative_volume.replace(0, np.nan)
    return pv / safe_volume


# ─── Supertrend ───────────────────────────────────────────────────────────────

def supertrend(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.

    Returns (direction, line) where:
        direction: +1.0 = bullish (price above Supertrend line)
                   -1.0 = bearish (price below Supertrend line)
                   NaN  = warm-up period
        line: the Supertrend value itself (upper band when bearish, lower when bullish)

    Usage in formula conditions:
        SUPERTREND > 0           → bullish regime
        CLOSE > SUPERTREND_LINE  → price above supertrend
    """
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    atr_val = atr(df, period)
    hl2     = (high + low) / 2.0

    basic_upper = hl2 + multiplier * atr_val
    basic_lower = hl2 - multiplier * atr_val

    n = len(close)
    final_upper = basic_upper.values.copy()
    final_lower = basic_lower.values.copy()
    st_line      = np.full(n, np.nan)
    st_direction = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(basic_upper.iloc[i]) or np.isnan(basic_lower.iloc[i]):
            continue

        # Ratchet upper band down, lower band up
        if basic_upper.iloc[i] < final_upper[i - 1] or close.iloc[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower.iloc[i] > final_lower[i - 1] or close.iloc[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Supertrend line flips between upper and lower band
        if np.isnan(st_line[i - 1]):
            st_line[i] = final_upper[i] if close.iloc[i] <= final_upper[i] else final_lower[i]
        elif st_line[i - 1] == final_upper[i - 1]:
            # Was bearish: stays upper unless price closes above upper
            st_line[i] = final_upper[i] if close.iloc[i] <= final_upper[i] else final_lower[i]
        else:
            # Was bullish: stays lower unless price closes below lower
            st_line[i] = final_lower[i] if close.iloc[i] >= final_lower[i] else final_upper[i]

        st_direction[i] = 1.0 if close.iloc[i] > st_line[i] else -1.0

    return (
        pd.Series(st_direction, index=df.index, name="SUPERTREND"),
        pd.Series(st_line,      index=df.index, name="SUPERTREND_LINE"),
    )


# ─── Previous-day high / low ──────────────────────────────────────────────────

def prev_day_high(df: pd.DataFrame) -> pd.Series:
    """
    Previous calendar day's high, forward-filled to all intraday bars of the next day.
    On daily timeframes returns the prior bar's high (shift(1)).
    Returns NaN for the first session (no prior day available).
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return pd.Series(np.nan, index=df.index, name="PREV_DAY_HIGH")

    if _is_daily_timeframe(df):
        return df["high"].astype(float).shift(1).rename("PREV_DAY_HIGH")

    daily_high = df["high"].astype(float).resample("1D").max()
    shifted    = daily_high.shift(1)
    return shifted.reindex(df.index, method="ffill").rename("PREV_DAY_HIGH")


def prev_day_low(df: pd.DataFrame) -> pd.Series:
    """
    Previous calendar day's low, forward-filled to all intraday bars of the next day.
    On daily timeframes returns the prior bar's low (shift(1)).
    Returns NaN for the first session (no prior day available).
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return pd.Series(np.nan, index=df.index, name="PREV_DAY_LOW")

    if _is_daily_timeframe(df):
        return df["low"].astype(float).shift(1).rename("PREV_DAY_LOW")

    daily_low = df["low"].astype(float).resample("1D").min()
    shifted   = daily_low.shift(1)
    return shifted.reindex(df.index, method="ffill").rename("PREV_DAY_LOW")


# ─── Indicator orchestrator ────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    """
    Add all required indicator columns to the DataFrame.

    indicator_config is the 'indicators' dict from the YAML, e.g.:
      {"RSI": [14], "SMA": [10, 200], "EMA": [9, 21], "ATR": [14]}

    Adds columns like: RSI_14, SMA_200, EMA_9, MACD, BB_UPPER_20, BB_LOWER_20, ATR_14, VWAP
    """
    close = df["close"]
    df    = df.copy()

    for indicator, periods in indicator_config.items():
        ind = indicator.upper()

        if ind == "RSI":
            for n in (periods or [14]):
                df[f"RSI_{n}"] = rsi(close, int(n))

        elif ind == "SMA":
            for n in (periods or [20]):
                df[f"SMA_{n}"] = sma(close, int(n))

        elif ind == "EMA":
            for n in (periods or [20]):
                df[f"EMA_{n}"] = ema(close, int(n))

        elif ind == "MACD":
            df["MACD"]        = macd_line(close)
            df["MACD_SIGNAL"] = macd_signal(close)
            df["MACD_HIST"]   = df["MACD"] - df["MACD_SIGNAL"]

        elif ind in ("BB_UPPER", "BB_LOWER", "BB"):
            for n in (periods or [20]):
                df[f"BB_UPPER_{n}"] = bb_upper(close, int(n))
                df[f"BB_LOWER_{n}"] = bb_lower(close, int(n))
                df[f"BB_MID_{n}"]   = bb_middle(close, int(n))

        elif ind == "ATR":
            for n in (periods or [14]):
                df[f"ATR_{n}"] = atr(df, int(n))

        elif ind == "VWAP":
            df["VWAP"] = vwap(df)

        elif ind == "SUPERTREND":
            params   = periods if isinstance(periods, dict) else {}
            st_period = int(params.get("period", 7))
            st_mult   = float(params.get("multiplier", 3.0))
            direction, line = supertrend(df, period=st_period, multiplier=st_mult)
            df["SUPERTREND"]      = direction
            df["SUPERTREND_LINE"] = line

        elif ind in ("PREV_DAY_HIGH", "PREV_DAY"):
            df["PREV_DAY_HIGH"] = prev_day_high(df)
            df["PREV_DAY_LOW"]  = prev_day_low(df)

        elif ind == "PREV_DAY_LOW":
            df["PREV_DAY_HIGH"] = prev_day_high(df)
            df["PREV_DAY_LOW"]  = prev_day_low(df)

    # Always add PREV_DAY_HIGH / PREV_DAY_LOW when the DataFrame spans multiple
    # days — they cost almost nothing and make the columns available for
    # conditions without requiring explicit config.
    if "PREV_DAY_HIGH" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df["PREV_DAY_HIGH"] = prev_day_high(df)
        df["PREV_DAY_LOW"]  = prev_day_low(df)

    return df


def max_indicator_warmup(indicator_config: dict) -> int:
    """
    Return the maximum warm-up candles needed across all indicators.

    This is used by the data fetcher to ensure enough history is requested
    so that indicators are available from the very first tradeable candle.

    Example: SMA(50) → 50, RSI(14) → 14, MACD → 26.  Max = 50.
    """
    warmup = 0

    for indicator, periods in indicator_config.items():
        ind = indicator.upper()

        if ind in ("SMA", "EMA", "RSI", "BB_UPPER", "BB_LOWER", "BB", "ATR"):
            for n in (periods or [20]):
                warmup = max(warmup, int(n))

        elif ind == "MACD":
            warmup = max(warmup, 26 + 9)  # EMA(26) + signal EMA(9)

        elif ind == "SUPERTREND":
            params = periods if isinstance(periods, dict) else {}
            warmup = max(warmup, int(params.get("period", 7)))

    return warmup
