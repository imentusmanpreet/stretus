"""
engine/indicators.py
═════════════════════
Calculates all technical indicators supported by Stretus.

Supported indicators:
  SMA(n)       — Simple Moving Average
  EMA(n)       — Exponential Moving Average
  RSI(n)       — Relative Strength Index
  MACD         — MACD line (12, 26, 9)
  BB_UPPER(n)  — Bollinger Band Upper (n period, 2 std dev)
  BB_LOWER(n)  — Bollinger Band Lower (n period, 2 std dev)
  ATR(n)       — Average True Range (Wilder's smoothing)
  VWAP         — Session VWAP (intraday: cumulative; daily bars: typical price per bar)

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

    # Hand off to the extended-indicator pack for everything outside the
    # original eight-indicator core (ADX, Stochastic, OBV, MFI, Donchian,
    # Keltner, Supertrend, …). Indicators the extended pack doesn't know
    # are ignored silently — the chat layer's catalog is the source of
    # truth for what's supported, and unknown names are dead weight here.
    from .indicators_ext import add_extended_indicators
    df = add_extended_indicators(df, indicator_config)

    return df


_EXTENDED_INDICATOR_WARMUP: dict[str, int] = {
    "ADX": 14, "DMI": 14,
    "STOCH_K": 14, "STOCH_D": 17, "WILLR": 14,
    "ROC": 10, "CCI": 20, "MOMENTUM": 10, "CMO": 9, "TRIX": 45,
    "STDEV": 20, "HV": 20, "CHOPPINESS": 14, "DISPARITY": 14,
    "BB_WIDTH": 20, "BB_PCT_B": 20,
    "VOLUME_SMA": 20, "VROC": 12, "MFI": 14, "CMF": 20,
    "AROON_UP": 15, "AROON_DOWN": 15, "AROON_OSC": 15,
    "HHV": 20, "LLV": 20, "DONCHIAN": 20,
    "SUPERTREND": 10, "KELTNER": 20, "ATR_UPPER": 14, "ATR_LOWER": 14,
    "PSAR": 5,
}


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

        elif ind in _EXTENDED_INDICATOR_WARMUP:
            base = _EXTENDED_INDICATOR_WARMUP[ind]
            for entry in (periods or [base]):
                if isinstance(entry, (list, tuple)) and entry:
                    candidate = entry[0]
                else:
                    candidate = entry
                try:
                    warmup = max(warmup, int(candidate))
                except (TypeError, ValueError):
                    warmup = max(warmup, base)

    return warmup
