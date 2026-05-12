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
  PIVOT        — Classic Pivot Points P / R1 / R2 / S1 / S2 from prior session

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


# ─── Pivot Points (classic) ───────────────────────────────────────────────────

def gap_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bar session gap diagnostics.

    Returns a DataFrame aligned to df.index with columns:
        SESSION_OPEN      — open price of the bar that started this session
        PREV_SESSION_CLOSE— close of the previous session's last bar
        GAP_SIZE_PCT      — (SESSION_OPEN - PREV_SESSION_CLOSE)/PREV_SESSION_CLOSE × 100
        GAP_FILL_PCT      — how much of the gap has been retraced *so far today*,
                            on the scale 0% (not filled) → 100% (fully filled).
                            Negative values mean price has moved AWAY from
                            previous close (the gap widened).

    Conventions
    ───────────
      Gap Up:    OPEN > PrevClose → GAP_SIZE_PCT > 0
                 fill happens when LOW retraces DOWN toward PrevClose.
                 GAP_FILL_PCT = (SESSION_OPEN − min_low_so_far) / GapSize × 100
      Gap Down:  OPEN < PrevClose → GAP_SIZE_PCT < 0
                 fill happens when HIGH retraces UP toward PrevClose.
                 GAP_FILL_PCT = (max_high_so_far − SESSION_OPEN) / |GapSize| × 100

    Intraday timeframes only. Daily-bar inputs return all-NaN values for these
    columns (a "gap" between consecutive daily bars uses prev_close vs today's
    open and can be expressed directly via PREV(CLOSE, 1)).

    No look-ahead: at bar i, only data from session-start through i is used.
    """
    columns = ["SESSION_OPEN", "PREV_SESSION_CLOSE", "GAP_SIZE_PCT", "GAP_FILL_PCT"]
    out = pd.DataFrame(index=df.index, columns=columns, dtype=float)

    if len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex) or _is_daily_timeframe(df):
        return out

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    sessions = df.index.normalize()

    # Per-session: opening price (first bar's open) and previous session's last close.
    session_open = open_.groupby(sessions).transform("first")
    session_last_close = close.groupby(sessions).last()
    prev_session_close = session_last_close.shift(1)
    prev_close_per_bar = pd.Series(
        sessions.map(prev_session_close), index=df.index, dtype=float
    )

    gap_size = session_open - prev_close_per_bar
    safe_prev = prev_close_per_bar.replace(0, np.nan)
    gap_pct = (gap_size / safe_prev) * 100.0

    # Running session lows/highs (cumulative within each session, includes
    # the current bar — this is the "max retracement so far" measure).
    cum_low = low.groupby(sessions).cummin()
    cum_high = high.groupby(sessions).cummax()

    # Fill % depends on gap direction. Avoid divide-by-zero when gap≈0.
    abs_gap = gap_size.abs().replace(0, np.nan)
    fill_pct = pd.Series(np.nan, index=df.index, dtype=float)
    up_mask = gap_size > 0
    down_mask = gap_size < 0
    # Gap up: how far down from session_open toward prev_close
    fill_pct = fill_pct.mask(
        up_mask, ((session_open - cum_low) / abs_gap) * 100.0
    )
    # Gap down: how far up from session_open toward prev_close
    fill_pct = fill_pct.mask(
        down_mask, ((cum_high - session_open) / abs_gap) * 100.0
    )

    out["SESSION_OPEN"] = session_open
    out["PREV_SESSION_CLOSE"] = prev_close_per_bar
    out["GAP_SIZE_PCT"] = gap_pct
    out["GAP_FILL_PCT"] = fill_pct
    return out


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic Pivot Points from the *previous* trading session.

    Returns a DataFrame aligned to df.index with columns:
        PIVOT_P, PIVOT_R1, PIVOT_R2, PIVOT_S1, PIVOT_S2

    Definitions
    ───────────
      P  = (PrevHigh + PrevLow + PrevClose) / 3
      R1 = (2 × P) − PrevLow
      S1 = (2 × P) − PrevHigh
      R2 = P + (PrevHigh − PrevLow)
      S2 = P − (PrevHigh − PrevLow)

    Intraday bars
        Each bar inside a trading session shares the same pivot block
        (computed from the *prior* day's H/L/C). The first session in the
        DataFrame has NaN pivots because there is no prior session yet.

    Daily / coarser bars
        The pivots of bar i are computed from the previous bar (i-1). Bar 0
        has NaN.

    No look-ahead: pivots use only data from sessions that ended strictly
    before the current bar.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    columns = ["PIVOT_P", "PIVOT_R1", "PIVOT_R2", "PIVOT_S1", "PIVOT_S2"]
    out = pd.DataFrame(index=df.index, columns=columns, dtype=float)

    if len(df) == 0:
        return out

    if not _is_daily_timeframe(df) and isinstance(df.index, pd.DatetimeIndex):
        # Intraday: collapse each session to one (H, L, C) tuple, shift one
        # session forward, then broadcast back to every bar inside that session.
        sessions = df.index.normalize()
        session_h = high.groupby(sessions).max()
        session_l = low.groupby(sessions).min()
        session_c = close.groupby(sessions).last()
        prev_h = session_h.shift(1)
        prev_l = session_l.shift(1)
        prev_c = session_c.shift(1)
        p = (prev_h + prev_l + prev_c) / 3.0
        rng = prev_h - prev_l
        r1 = (2.0 * p) - prev_l
        s1 = (2.0 * p) - prev_h
        r2 = p + rng
        s2 = p - rng
        # Map session-level values back to every bar. .map() on a DatetimeIndex
        # returns an Index; column assignment coerces it to a Series.
        out["PIVOT_P"] = sessions.map(p)
        out["PIVOT_R1"] = sessions.map(r1)
        out["PIVOT_R2"] = sessions.map(r2)
        out["PIVOT_S1"] = sessions.map(s1)
        out["PIVOT_S2"] = sessions.map(s2)
        return out

    # Daily (or non-datetime index): each bar uses the previous bar.
    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)
    p = (prev_h + prev_l + prev_c) / 3.0
    rng = prev_h - prev_l
    out["PIVOT_P"] = p
    out["PIVOT_R1"] = (2.0 * p) - prev_l
    out["PIVOT_S1"] = (2.0 * p) - prev_h
    out["PIVOT_R2"] = p + rng
    out["PIVOT_S2"] = p - rng
    return out


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

        elif ind in ("PIVOT", "PIVOTS", "PIVOT_POINTS"):
            piv = pivot_points(df)
            for col in piv.columns:
                df[col] = piv[col]

        elif ind in ("GAP", "GAPS"):
            gap = gap_series(df)
            for col in gap.columns:
                df[col] = gap[col]

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

        elif ind in ("PIVOT", "PIVOTS", "PIVOT_POINTS", "GAP", "GAPS"):
            # Pivots / gap stats need at least one prior session of bars. The
            # session size varies by timeframe; use a conservative 375 (one
            # full NSE session in 1m bars). Daily strategies only need 1 prior
            # bar but the safe over-allocation costs little.
            warmup = max(warmup, 375)

    return warmup
