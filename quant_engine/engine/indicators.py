"""
engine/indicators.py
═════════════════════
Calculates all technical indicators supported by Stretus.

Backend
───────
TA-Lib (the official C-library wrapper, ``TA-Lib==0.6.8``) is the indicator
backend for every standard study. Each wrapper takes a pandas Series/DataFrame
and returns a ``pd.Series`` aligned to ``df.index``, NaN until the indicator's
warm-up (look-back) is satisfied — the same convention the engine has always
used. TA-Lib computes purely left-to-right (value[i] depends only on bars
≤ i), so every wrapper here is causal and safe for point-in-time evaluation.

A small number of studies have NO TA-Lib equivalent and stay hand-rolled, but
remain causal: session VWAP, SuperTrend, previous-day H/L, opening range,
Donchian channel, classic pivots and Keltner channels (ATR ± EMA).

Seeding note (deliberate migration shift)
──────────────────────────────────────────
TA-Lib seeds Wilder-smoothed studies (EMA/RSI/ATR/ADX) with an initial SMA and
its MACD signal needs the slow EMA fully seeded. Its early values therefore
differ slightly from the engine's former hand-rolled pandas ``ewm`` math, and
its first-valid index can be one bar later. This is expected and intentional —
TA-Lib is now the single source of truth.

Causality / look-ahead
───────────────────────
Only causal studies are exposed. Repainting / centered / forward-looking
TA-Lib functions (e.g. the Hilbert-Transform ``HT_*`` family, ``MAVP``) are
deliberately NOT registered, so they can never appear in an entry rule.

All functions accept pandas Series/DataFrame and return a Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib


# ─── Array helpers ──────────────────────────────────────────────────────────────

def _arr(series: pd.Series) -> np.ndarray:
    """Contiguous float64 view of a Series — the input shape TA-Lib requires."""
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64, copy=False))


def _s(values: np.ndarray, index, name: str | None = None) -> pd.Series:
    """Wrap a TA-Lib output array back into an index-aligned Series."""
    return pd.Series(values, index=index, name=name)


def rolling_agg(series: pd.Series, n: int, agg: str) -> pd.Series | None:
    """TA-Lib rolling aggregate over an arbitrary series — the backend for the
    evaluator's AVG/MAX/MIN/STDEV/ZSCORE/SMA(field,n)/EMA(field,n) primitives, so
    those compute on TA-Lib too (no pandas .rolling anywhere). NaN until the full
    n-bar window is valid (TA-Lib propagates NaN), matching the engine convention.

    agg ∈ {mean, max, min, stdev, ema}. STDEV uses TA-Lib's sample-equivalent
    nbdev=1 standard deviation. Returns None for an unknown agg."""
    arr = _arr(series)
    n = int(n)
    if agg == "mean":
        out = talib.SMA(arr, timeperiod=n)
    elif agg == "max":
        out = talib.MAX(arr, timeperiod=n)
    elif agg == "min":
        out = talib.MIN(arr, timeperiod=n)
    elif agg == "stdev":
        out = talib.STDDEV(arr, timeperiod=n, nbdev=1.0)
    elif agg == "ema":
        out = talib.EMA(arr, timeperiod=n)
    else:
        return None
    return _s(out, series.index)


# ─── Overlap / trend ────────────────────────────────────────────────────────────

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average over n periods."""
    return _s(talib.SMA(_arr(series), timeperiod=int(n)), series.index)


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average over n periods (TA-Lib seeding: SMA then EMA)."""
    return _s(talib.EMA(_arr(series), timeperiod=int(n)), series.index)


def wma(series: pd.Series, n: int) -> pd.Series:
    """Weighted Moving Average over n periods."""
    return _s(talib.WMA(_arr(series), timeperiod=int(n)), series.index)


def dema(series: pd.Series, n: int) -> pd.Series:
    """Double Exponential Moving Average."""
    return _s(talib.DEMA(_arr(series), timeperiod=int(n)), series.index)


def tema(series: pd.Series, n: int) -> pd.Series:
    """Triple Exponential Moving Average."""
    return _s(talib.TEMA(_arr(series), timeperiod=int(n)), series.index)


def trima(series: pd.Series, n: int) -> pd.Series:
    """Triangular Moving Average."""
    return _s(talib.TRIMA(_arr(series), timeperiod=int(n)), series.index)


def kama(series: pd.Series, n: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    return _s(talib.KAMA(_arr(series), timeperiod=int(n)), series.index)


def t3(series: pd.Series, n: int = 5, vfactor: float = 0.7) -> pd.Series:
    """T3 (Tillson) smoothed moving average."""
    return _s(talib.T3(_arr(series), timeperiod=int(n), vfactor=float(vfactor)), series.index)


def macd_line(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line (fast EMA − slow EMA). TA-Lib MACD(12, 26, 9)."""
    macd, _signal, _hist = talib.MACD(
        _arr(series), fastperiod=int(fast), slowperiod=int(slow), signalperiod=int(signal)
    )
    return _s(macd, series.index, "MACD")


def macd_signal(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD signal line = EMA(signal) of the MACD line."""
    _macd, sig, _hist = talib.MACD(
        _arr(series), fastperiod=int(fast), slowperiod=int(slow), signalperiod=int(signal)
    )
    return _s(sig, series.index, "MACD_SIGNAL")


def macd_all(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """(macd, signal, hist) in one TA-Lib pass — avoids three separate calls."""
    macd, sig, hist = talib.MACD(
        _arr(series), fastperiod=int(fast), slowperiod=int(slow), signalperiod=int(signal)
    )
    idx = series.index
    return _s(macd, idx, "MACD"), _s(sig, idx, "MACD_SIGNAL"), _s(hist, idx, "MACD_HIST")


def ppo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Percentage Price Oscillator = 100·(EMA_fast − EMA_slow)/EMA_slow."""
    return _s(talib.PPO(_arr(series), fastperiod=int(fast), slowperiod=int(slow)), series.index)


def apo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Absolute Price Oscillator = EMA_fast − EMA_slow."""
    return _s(talib.APO(_arr(series), fastperiod=int(fast), slowperiod=int(slow)), series.index)


def sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Parabolic SAR — stop-and-reverse trend level."""
    return _s(
        talib.SAR(_arr(df["high"]), _arr(df["low"]), acceleration=float(acceleration), maximum=float(maximum)),
        df.index,
        "SAR",
    )


def aroon(df: pd.DataFrame, n: int = 14):
    """Aroon (down, up) over n periods, each 0–100."""
    down, up = talib.AROON(_arr(df["high"]), _arr(df["low"]), timeperiod=int(n))
    return _s(down, df.index, "AROON_DOWN"), _s(up, df.index, "AROON_UP")


def aroonosc(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Aroon Oscillator = Aroon Up − Aroon Down (−100..+100)."""
    return _s(talib.AROONOSC(_arr(df["high"]), _arr(df["low"]), timeperiod=int(n)), df.index, "AROONOSC")


# ─── Bollinger Bands ────────────────────────────────────────────────────────────

def _bbands(series: pd.Series, n: int, nbdev: float = 2.0):
    upper, middle, lower = talib.BBANDS(
        _arr(series), timeperiod=int(n), nbdevup=float(nbdev), nbdevdn=float(nbdev), matype=0
    )
    idx = series.index
    return _s(upper, idx), _s(middle, idx), _s(lower, idx)


def bb_upper(series: pd.Series, n: int = 20) -> pd.Series:
    """Bollinger Band Upper = SMA(n) + 2·stddev(n) (TA-Lib population stddev)."""
    return _bbands(series, n)[0]


def bb_middle(series: pd.Series, n: int = 20) -> pd.Series:
    """Bollinger Band Middle = SMA(n)."""
    return _bbands(series, n)[1]


def bb_lower(series: pd.Series, n: int = 20) -> pd.Series:
    """Bollinger Band Lower = SMA(n) − 2·stddev(n)."""
    return _bbands(series, n)[2]


# ─── Momentum ───────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), 0–100."""
    return _s(talib.RSI(_arr(series), timeperiod=int(n)), series.index)


def roc(series: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change = 100·(price/price[n] − 1)."""
    return _s(talib.ROC(_arr(series), timeperiod=int(n)), series.index)


def rocp(series: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change ratio = price/price[n] − 1."""
    return _s(talib.ROCP(_arr(series), timeperiod=int(n)), series.index)


def mom(series: pd.Series, n: int = 10) -> pd.Series:
    """Momentum = price − price[n]."""
    return _s(talib.MOM(_arr(series), timeperiod=int(n)), series.index)


def cmo(series: pd.Series, n: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (−100..+100)."""
    return _s(talib.CMO(_arr(series), timeperiod=int(n)), series.index)


def trix(series: pd.Series, n: int = 30) -> pd.Series:
    """TRIX — 1-day ROC of a triple-smoothed EMA."""
    return _s(talib.TRIX(_arr(series), timeperiod=int(n)), series.index)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    return _s(talib.CCI(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def willr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R (−100..0)."""
    return _s(talib.WILLR(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI), 0–100."""
    return _s(
        talib.MFI(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), _arr(df["volume"]), timeperiod=int(n)),
        df.index,
    )


def ultosc(df: pd.DataFrame, n1: int = 7, n2: int = 14, n3: int = 28) -> pd.Series:
    """Ultimate Oscillator across three look-backs, 0–100."""
    return _s(
        talib.ULTOSC(
            _arr(df["high"]), _arr(df["low"]), _arr(df["close"]),
            timeperiod1=int(n1), timeperiod2=int(n2), timeperiod3=int(n3),
        ),
        df.index,
    )


def stoch(df: pd.DataFrame, fastk: int = 14, slowk: int = 3, slowd: int = 3):
    """Stochastic oscillator (%K, %D), 0–100."""
    k, d = talib.STOCH(
        _arr(df["high"]), _arr(df["low"]), _arr(df["close"]),
        fastk_period=int(fastk), slowk_period=int(slowk), slowk_matype=0,
        slowd_period=int(slowd), slowd_matype=0,
    )
    return _s(k, df.index, "STOCH_K"), _s(d, df.index, "STOCH_D")


def stochf(df: pd.DataFrame, fastk: int = 14, fastd: int = 3):
    """Fast Stochastic (%K, %D), 0–100."""
    k, d = talib.STOCHF(
        _arr(df["high"]), _arr(df["low"]), _arr(df["close"]),
        fastk_period=int(fastk), fastd_period=int(fastd), fastd_matype=0,
    )
    return _s(k, df.index, "STOCHF_K"), _s(d, df.index, "STOCHF_D")


def stochrsi(series: pd.Series, n: int = 14, fastk: int = 5, fastd: int = 3):
    """Stochastic RSI (%K, %D), 0–100."""
    k, d = talib.STOCHRSI(
        _arr(series), timeperiod=int(n), fastk_period=int(fastk), fastd_period=int(fastd), fastd_matype=0
    )
    return _s(k, series.index, "STOCHRSI_K"), _s(d, series.index, "STOCHRSI_D")


# ─── Directional movement (ADX / +DI / -DI) ─────────────────────────────────────

def dmi(df: pd.DataFrame, n: int = 14):
    """
    Wilder's Directional Movement system via TA-Lib.

    Returns (adx, plus_di, minus_di), each a Series aligned to df.index.
    Causal: every value depends only on bars ≤ i.
    """
    high, low, close = _arr(df["high"]), _arr(df["low"]), _arr(df["close"])
    adx_arr = talib.ADX(high, low, close, timeperiod=int(n))
    plus = talib.PLUS_DI(high, low, close, timeperiod=int(n))
    minus = talib.MINUS_DI(high, low, close, timeperiod=int(n))
    idx = df.index
    return _s(adx_arr, idx), _s(plus, idx), _s(minus, idx)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0–100), direction-agnostic."""
    return _s(talib.ADX(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def adxr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Movement Rating (smoothed ADX)."""
    return _s(talib.ADXR(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def dx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Directional Movement Index (un-smoothed ADX numerator)."""
    return _s(talib.DX(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def plus_di(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """+DI — strength of upward directional movement."""
    return _s(talib.PLUS_DI(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def minus_di(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """-DI — strength of downward directional movement."""
    return _s(talib.MINUS_DI(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


# ─── Volatility ─────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return _s(talib.ATR(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def natr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Normalised ATR = 100·ATR/close — volatility as a % of price."""
    return _s(talib.NATR(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), timeperiod=int(n)), df.index)


def trange(df: pd.DataFrame) -> pd.Series:
    """True Range (single-bar, no smoothing)."""
    return _s(talib.TRANGE(_arr(df["high"]), _arr(df["low"]), _arr(df["close"])), df.index, "TRANGE")


def stddev(series: pd.Series, n: int = 20, nbdev: float = 1.0) -> pd.Series:
    """Rolling standard deviation (population, TA-Lib convention)."""
    return _s(talib.STDDEV(_arr(series), timeperiod=int(n), nbdev=float(nbdev)), series.index)


def keltner_upper(df: pd.DataFrame, n: int = 20, atr_period: int = 10, mult: float = 2.0) -> pd.Series:
    """Keltner Channel Upper = EMA(close, n) + mult·ATR(atr_period). Hand-rolled (no TA-Lib study)."""
    return ema(df["close"], n) + mult * atr(df, atr_period)


def keltner_lower(df: pd.DataFrame, n: int = 20, atr_period: int = 10, mult: float = 2.0) -> pd.Series:
    """Keltner Channel Lower = EMA(close, n) − mult·ATR(atr_period)."""
    return ema(df["close"], n) - mult * atr(df, atr_period)


def keltner_middle(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Keltner Channel Middle = EMA(close, n)."""
    return ema(df["close"], n)


# ─── Volume ─────────────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    return _s(talib.OBV(_arr(df["close"]), _arr(df["volume"])), df.index, "OBV")


def ad(df: pd.DataFrame) -> pd.Series:
    """Chaikin Accumulation/Distribution line."""
    return _s(
        talib.AD(_arr(df["high"]), _arr(df["low"]), _arr(df["close"]), _arr(df["volume"])),
        df.index,
        "AD",
    )


def adosc(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin A/D Oscillator."""
    return _s(
        talib.ADOSC(
            _arr(df["high"]), _arr(df["low"]), _arr(df["close"]), _arr(df["volume"]),
            fastperiod=int(fast), slowperiod=int(slow),
        ),
        df.index,
        "ADOSC",
    )


# ─── VWAP (hand-rolled, session-aware — no TA-Lib equivalent) ────────────────────

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
        return typical_price

    if isinstance(df.index, pd.DatetimeIndex):
        session_key = df.index.normalize()
        pv = (typical_price * volume).groupby(session_key).cumsum()
        cumulative_volume = volume.groupby(session_key).cumsum()
    else:
        pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

    safe_volume = cumulative_volume.replace(0, np.nan)
    return pv / safe_volume


# ─── Supertrend (hand-rolled — no TA-Lib equivalent) ─────────────────────────────

def supertrend(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 3.0,
):
    """
    Supertrend indicator.

    Returns (direction, line) where:
        direction: +1.0 = bullish (price above Supertrend line)
                   -1.0 = bearish (price below Supertrend line)
                   NaN  = warm-up period
        line: the Supertrend value itself (upper band when bearish, lower when bullish)
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

        if basic_upper.iloc[i] < final_upper[i - 1] or close.iloc[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower.iloc[i] > final_lower[i - 1] or close.iloc[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if np.isnan(st_line[i - 1]):
            st_line[i] = final_upper[i] if close.iloc[i] <= final_upper[i] else final_lower[i]
        elif st_line[i - 1] == final_upper[i - 1]:
            st_line[i] = final_upper[i] if close.iloc[i] <= final_upper[i] else final_lower[i]
        else:
            st_line[i] = final_lower[i] if close.iloc[i] >= final_lower[i] else final_upper[i]

        st_direction[i] = 1.0 if close.iloc[i] > st_line[i] else -1.0

    return (
        pd.Series(st_direction, index=df.index, name="SUPERTREND"),
        pd.Series(st_line,      index=df.index, name="SUPERTREND_LINE"),
    )


# ─── Previous-day high / low (hand-rolled, session-aware) ────────────────────────

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


# ─── Donchian channel (hand-rolled — rolling max/min of prior bars) ──────────────

def donchian_upper(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Donchian upper = highest HIGH of the PRIOR n bars (excludes the current bar,
    so a close above it is a genuine breakout, not a tautology). NaN until n priors.

    TA-Lib MAX(high, n) is the rolling max INCLUDING the current bar; shifting the
    result by one bar drops the current bar, yielding the prior-n-bar high."""
    raw = talib.MAX(_arr(df["high"]), timeperiod=int(n))
    return _s(raw, df.index).shift(1).rename(f"DONCHIAN_UPPER_{int(n)}")


def donchian_lower(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Donchian lower = lowest LOW of the PRIOR n bars (excludes the current bar)."""
    raw = talib.MIN(_arr(df["low"]), timeperiod=int(n))
    return _s(raw, df.index).shift(1).rename(f"DONCHIAN_LOWER_{int(n)}")


def donchian_middle(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Donchian middle = (upper + lower) / 2."""
    return ((donchian_upper(df, n) + donchian_lower(df, n)) / 2.0).rename(f"DONCHIAN_MID_{int(n)}")


# ─── Classic floor-trader pivots (hand-rolled, prior-session based) ──────────────

def _prev_session_ohlc(df: pd.DataFrame):
    """Return (prev_high, prev_low, prev_close) Series aligned to df.index, using
    the PRIOR calendar session's H/L/C (intraday) or the prior bar (daily)."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        nan = pd.Series(np.nan, index=df.index)
        return nan, nan, nan
    if _is_daily_timeframe(df):
        h = df["high"].astype(float).shift(1)
        l = df["low"].astype(float).shift(1)
        c = df["close"].astype(float).shift(1)
        return h, l, c
    daily_high  = df["high"].astype(float).resample("1D").max().shift(1)
    daily_low   = df["low"].astype(float).resample("1D").min().shift(1)
    daily_close = df["close"].astype(float).resample("1D").last().shift(1)
    h = daily_high.reindex(df.index, method="ffill")
    l = daily_low.reindex(df.index, method="ffill")
    c = daily_close.reindex(df.index, method="ffill")
    return h, l, c


def pivot_levels(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Classic pivot point + R1/S1/R2/S2 from the prior session. All causal."""
    h, l, c = _prev_session_ohlc(df)
    pivot = (h + l + c) / 3.0
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    return {
        "PIVOT": pivot.rename("PIVOT"),
        "PIVOT_R1": r1.rename("PIVOT_R1"),
        "PIVOT_S1": s1.rename("PIVOT_S1"),
        "PIVOT_R2": r2.rename("PIVOT_R2"),
        "PIVOT_S2": s2.rename("PIVOT_S2"),
    }


# ─── Opening range (per-session first-N max / min) ────────────────────────────

def opening_range_series(
    df: pd.DataFrame, field: str, n: int, agg: str
) -> pd.Series:
    """Vectorized session-level "opening range" computation.

    For every row, returns the max (agg="max") or min (agg="min") of `field`
    over the first `n` bars of the SAME session (calendar day, UTC). Bars 0..n-1
    of each session return NaN — they're inside the warm-up window of their own
    session — matching the semantics of the legacy per-bar implementation.
    """
    col_lower = str(field).lower()
    if (
        n <= 0
        or not isinstance(df.index, pd.DatetimeIndex)
        or col_lower not in df.columns
    ):
        return pd.Series(np.nan, index=df.index)

    session = df.index.normalize()                       # UTC calendar day
    pos_in_session = df.groupby(session, sort=False).cumcount()  # 0,1,2,... per session
    in_opening_window = pos_in_session < n

    opening_vals = df[col_lower].astype(float).where(in_opening_window)
    grouped = opening_vals.groupby(session, sort=False)
    per_session = grouped.max() if str(agg).lower() == "max" else grouped.min()

    result = per_session.reindex(session).to_numpy()
    result_series = pd.Series(result, index=df.index)
    result_series = result_series.mask(in_opening_window.to_numpy())
    return result_series


# ─── Candlestick patterns (TA-Lib CDL*) ──────────────────────────────────────────
#
# Every TA-Lib pattern function takes (open, high, low, close) and returns an int
# array: +100 (bullish), -100 (bearish), 0 (no pattern); a few output ±100/±200.
# Inherently causal (each value uses only the current and prior 1–3 bars). We
# expose them as confirmation columns named CDL_<PATTERN> holding that signed
# integer, so a card formula can say `CDL_ENGULFING > 0` (bullish confirmation)
# or `CDL_ENGULFING < 0` (bearish).

# TA-Lib names are like "CDLENGULFING"; our column/alias drops to "CDL_ENGULFING".
def _talib_cdl_functions() -> dict[str, str]:
    """Map our alias (CDL_ENGULFING) → TA-Lib function name (CDLENGULFING)."""
    out: dict[str, str] = {}
    for fn in talib.get_function_groups().get("Pattern Recognition", []):
        alias = "CDL_" + fn[len("CDL"):]          # CDLENGULFING -> CDL_ENGULFING
        out[alias] = fn
    return out


_CDL_FUNCTIONS = _talib_cdl_functions()

# A curated subset that gets first-class catalog SignalCards. The full set is
# still computable on demand via add_all_indicators / candlestick_pattern().
COMMON_CANDLESTICK_PATTERNS = [
    "CDL_ENGULFING", "CDL_DOJI", "CDL_HAMMER", "CDL_INVERTEDHAMMER",
    "CDL_SHOOTINGSTAR", "CDL_HARAMI", "CDL_MORNINGSTAR", "CDL_EVENINGSTAR",
    "CDL_3WHITESOLDIERS", "CDL_3BLACKCROWS", "CDL_PIERCING", "CDL_DARKCLOUDCOVER",
    "CDL_HANGINGMAN", "CDL_MARUBOZU",
]


def candlestick_pattern(df: pd.DataFrame, alias: str) -> pd.Series:
    """Signed candlestick-pattern Series for a CDL_* alias (+bullish / −bearish / 0)."""
    fn = _CDL_FUNCTIONS.get(alias.upper())
    if fn is None:
        return pd.Series(np.nan, index=df.index, name=alias.upper())
    func = getattr(talib, fn)
    vals = func(_arr(df["open"]), _arr(df["high"]), _arr(df["low"]), _arr(df["close"]))
    return pd.Series(vals.astype(float), index=df.index, name=alias.upper())


# ─── Indicator registry ──────────────────────────────────────────────────────────
#
# Single source of truth for "given an indicator NAME and period(s), which
# columns appear and how are they computed". Each spec yields one or more
# (column_name, Series) pairs. `add_all_indicators` and conditions.py's
# precompute-if-missing path both consult this — so a study computes IDENTICALLY
# whether it's requested via YAML config or discovered in a formula.

# Single-output, period-parameterised studies over the CLOSE series.
_CLOSE_PERIODIC = {
    "SMA": sma, "EMA": ema, "WMA": wma, "DEMA": dema, "TEMA": tema,
    "TRIMA": trima, "KAMA": kama, "T3": t3,
    "RSI": rsi, "ROC": roc, "ROCP": rocp, "MOM": mom, "CMO": cmo,
    "TRIX": trix, "STDDEV": stddev,
}

# Single-output, period-parameterised studies over the full OHLC(V) frame.
_OHLC_PERIODIC = {
    "ATR": atr, "NATR": natr,
    "ADX": adx, "ADXR": adxr, "DX": dx, "PLUS_DI": plus_di, "MINUS_DI": minus_di,
    "CCI": cci, "WILLR": willr, "MFI": mfi,
    "DONCHIAN_UPPER": donchian_upper, "DONCHIAN_LOWER": donchian_lower,
    "DONCHIAN_MID": donchian_middle,
}

# Periodic indicators where one config entry should also fan out sibling columns
# (compute once, populate the whole family).
_DMI_FAMILY = ("ADX", "PLUS_DI", "MINUS_DI")

# Default periods used when a config entry passes no explicit periods.
_DEFAULT_PERIODS: dict[str, list[int]] = {
    "RSI": [14], "ATR": [14], "NATR": [14], "ADX": [14], "ADXR": [14], "DX": [14],
    "PLUS_DI": [14], "MINUS_DI": [14], "CCI": [20], "WILLR": [14], "MFI": [14],
    "CMO": [14], "ROC": [10], "ROCP": [10], "MOM": [10], "TRIX": [30],
    "STDDEV": [20], "KAMA": [30], "T3": [5], "DONCHIAN_UPPER": [20],
    "DONCHIAN_LOWER": [20], "DONCHIAN_MID": [20], "AROON": [14], "AROONOSC": [14],
}

# Wilder-smoothed studies need roughly 2× the period to stabilise.
_WILDER_WARMUP = {"RSI", "ATR", "NATR", "ADX", "ADXR", "DX", "PLUS_DI", "MINUS_DI", "CMO", "MFI"}
# Cascaded-EMA studies need a multiple of the period.
_CASCADED_WARMUP = {"DEMA": 2, "TEMA": 3, "T3": 6, "TRIX": 3}


def _default_periods(name: str) -> list[int]:
    return _DEFAULT_PERIODS.get(name, [20])


# ─── Indicator orchestrator ────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    """
    Add all required indicator columns to the DataFrame.

    indicator_config is the 'indicators' dict from the YAML, e.g.:
      {"RSI": [14], "SMA": [10, 200], "EMA": [9, 21], "ATR": [14], "STOCH": {...}}

    Periodic studies produce columns like RSI_14, SMA_200, ATR_14, CCI_20.
    Multi-output / parameterised studies (MACD, BB, STOCH, SAR, ...) produce
    their named columns. Candlestick patterns produce CDL_* columns.
    """
    df = df.copy()
    has_volume = "volume" in df.columns
    has_ohlc = all(c in df.columns for c in ("high", "low", "close"))

    for indicator, periods in indicator_config.items():
        ind = indicator.upper()

        # ── Close-series periodic studies (SMA/EMA/RSI/...) ──────────────────
        if ind in _CLOSE_PERIODIC:
            func = _CLOSE_PERIODIC[ind]
            for n in (periods or _default_periods(ind)):
                col = f"{ind}_{int(n)}"
                if col not in df.columns:
                    df[col] = func(df["close"], int(n))

        # ── DMI family — compute ADX/+DI/-DI together per period ─────────────
        elif ind in _DMI_FAMILY or ind == "DMI":
            for n in (periods or [14]):
                if f"ADX_{int(n)}" in df.columns:
                    continue
                adx_s, plus_s, minus_s = dmi(df, int(n))
                df[f"ADX_{int(n)}"]      = adx_s
                df[f"PLUS_DI_{int(n)}"]  = plus_s
                df[f"MINUS_DI_{int(n)}"] = minus_s

        # ── Other OHLC periodic studies (ATR/NATR/CCI/WILLR/MFI/Donchian/...) ─
        elif ind in _OHLC_PERIODIC:
            func = _OHLC_PERIODIC[ind]
            for n in (periods or _default_periods(ind)):
                col = f"{ind}_{int(n)}"
                if col not in df.columns:
                    df[col] = func(df, int(n))

        # ── Aroon (two outputs + oscillator) ─────────────────────────────────
        elif ind == "AROON":
            for n in (periods or [14]):
                down_s, up_s = aroon(df, int(n))
                df[f"AROON_DOWN_{int(n)}"] = down_s
                df[f"AROON_UP_{int(n)}"]   = up_s
        elif ind in ("AROON_UP", "AROON_DOWN"):
            # Allow either family member to be requested directly (e.g. from a
            # formula reference AROON_UP(14)); compute both, populate both columns.
            for n in (periods or [14]):
                if f"AROON_UP_{int(n)}" in df.columns:
                    continue
                down_s, up_s = aroon(df, int(n))
                df[f"AROON_DOWN_{int(n)}"] = down_s
                df[f"AROON_UP_{int(n)}"]   = up_s
        elif ind == "AROONOSC":
            for n in (periods or [14]):
                df[f"AROONOSC_{int(n)}"] = aroonosc(df, int(n))

        # ── MACD (scalar, no period suffix) ──────────────────────────────────
        elif ind == "MACD":
            macd_s, sig_s, hist_s = macd_all(df["close"])
            df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd_s, sig_s, hist_s

        # ── Bollinger Bands ──────────────────────────────────────────────────
        elif ind in ("BB_UPPER", "BB_LOWER", "BB"):
            for n in (periods or [20]):
                u, m, l = _bbands(df["close"], int(n))
                df[f"BB_UPPER_{int(n)}"] = u
                df[f"BB_LOWER_{int(n)}"] = l
                df[f"BB_MID_{int(n)}"]   = m

        # ── Stochastics ──────────────────────────────────────────────────────
        elif ind == "STOCH":
            params = periods if isinstance(periods, dict) else {}
            k, d = stoch(df, int(params.get("fastk", 14)), int(params.get("slowk", 3)), int(params.get("slowd", 3)))
            df["STOCH_K"], df["STOCH_D"] = k, d
        elif ind == "STOCHF":
            params = periods if isinstance(periods, dict) else {}
            k, d = stochf(df, int(params.get("fastk", 14)), int(params.get("fastd", 3)))
            df["STOCHF_K"], df["STOCHF_D"] = k, d
        elif ind == "STOCHRSI":
            params = periods if isinstance(periods, dict) else {}
            k, d = stochrsi(df["close"], int(params.get("window", 14)), int(params.get("fastk", 5)), int(params.get("fastd", 3)))
            df["STOCHRSI_K"], df["STOCHRSI_D"] = k, d

        # ── Price oscillators (scalar; default 12/26) ────────────────────────
        elif ind == "PPO":
            params = periods if isinstance(periods, dict) else {}
            df["PPO"] = ppo(df["close"], int(params.get("fast", 12)), int(params.get("slow", 26)))
        elif ind == "APO":
            params = periods if isinstance(periods, dict) else {}
            df["APO"] = apo(df["close"], int(params.get("fast", 12)), int(params.get("slow", 26)))

        # ── Ultimate oscillator ──────────────────────────────────────────────
        elif ind == "ULTOSC":
            params = periods if isinstance(periods, dict) else {}
            df["ULTOSC"] = ultosc(df, int(params.get("n1", 7)), int(params.get("n2", 14)), int(params.get("n3", 28)))

        # ── Parabolic SAR ────────────────────────────────────────────────────
        elif ind == "SAR":
            params = periods if isinstance(periods, dict) else {}
            df["SAR"] = sar(df, float(params.get("acceleration", 0.02)), float(params.get("maximum", 0.2)))

        # ── Keltner channel ──────────────────────────────────────────────────
        elif ind == "KELTNER":
            params = periods if isinstance(periods, dict) else {}
            n = int(params.get("window", 20)); ap = int(params.get("atr_period", 10)); mult = float(params.get("multiplier", 2.0))
            df[f"KELTNER_UPPER_{n}"]  = keltner_upper(df, n, ap, mult)
            df[f"KELTNER_LOWER_{n}"]  = keltner_lower(df, n, ap, mult)
            df[f"KELTNER_MID_{n}"]    = keltner_middle(df, n)

        # ── Volume studies (period-less) ─────────────────────────────────────
        elif ind == "OBV" and has_volume:
            df["OBV"] = obv(df)
        elif ind == "AD" and has_volume:
            df["AD"] = ad(df)
        elif ind == "ADOSC" and has_volume:
            params = periods if isinstance(periods, dict) else {}
            df["ADOSC"] = adosc(df, int(params.get("fast", 3)), int(params.get("slow", 10)))

        # ── True range (period-less) ─────────────────────────────────────────
        elif ind == "TRANGE":
            df["TRANGE"] = trange(df)

        # ── VWAP ─────────────────────────────────────────────────────────────
        elif ind == "VWAP":
            df["VWAP"] = vwap(df)

        # ── Opening range ────────────────────────────────────────────────────
        elif ind == "OPENING_RANGE_HIGH":
            for n in (periods or [15]):
                df[f"OPENING_RANGE_HIGH_{int(n)}"] = opening_range_series(df, "high", int(n), "max")
        elif ind == "OPENING_RANGE_LOW":
            for n in (periods or [15]):
                df[f"OPENING_RANGE_LOW_{int(n)}"] = opening_range_series(df, "low", int(n), "min")

        # ── Supertrend ───────────────────────────────────────────────────────
        elif ind == "SUPERTREND":
            params   = periods if isinstance(periods, dict) else {}
            st_period = int(params.get("period", 7))
            st_mult   = float(params.get("multiplier", 3.0))
            direction, line = supertrend(df, period=st_period, multiplier=st_mult)
            df["SUPERTREND"]      = direction
            df["SUPERTREND_LINE"] = line

        # ── Pivots ───────────────────────────────────────────────────────────
        elif ind in ("PIVOT", "PIVOTS"):
            for col, series in pivot_levels(df).items():
                df[col] = series

        # ── Prev-day H/L ─────────────────────────────────────────────────────
        elif ind in ("PREV_DAY_HIGH", "PREV_DAY_LOW", "PREV_DAY"):
            df["PREV_DAY_HIGH"] = prev_day_high(df)
            df["PREV_DAY_LOW"]  = prev_day_low(df)

        # ── Candlestick patterns ─────────────────────────────────────────────
        elif ind == "CDL_ALL":
            for alias in _CDL_FUNCTIONS:
                df[alias] = candlestick_pattern(df, alias)
        elif ind in _CDL_FUNCTIONS:
            df[ind] = candlestick_pattern(df, ind)

    # PREV_DAY_HIGH / PREV_DAY_LOW are cheap and broadly useful — always add when
    # the frame spans multiple days, matching the engine's historical behaviour.
    if "PREV_DAY_HIGH" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df["PREV_DAY_HIGH"] = prev_day_high(df)
        df["PREV_DAY_LOW"]  = prev_day_low(df)

    return df


# Names addressable as a single periodic column NAME(period) → column NAME_period.
# This is the contract conditions.py relies on for precompute-if-missing: any
# study here computes IDENTICALLY whether requested via YAML config or discovered
# in a formula — one implementation, no per-bar recompute, no divergence.
PERIODIC_INDICATOR_NAMES = frozenset(
    set(_CLOSE_PERIODIC) | set(_OHLC_PERIODIC) | {
        "BB_UPPER", "BB_LOWER", "BB_MID", "AROON_UP", "AROON_DOWN", "AROONOSC",
    }
)


def compute_indicator_column(df: pd.DataFrame, name: str, period: int) -> pd.Series | None:
    """Return the full vectorised Series for a single periodic indicator column
    `NAME(period)` (e.g. ('RSI', 14) → RSI_14), or None if `name` is not a known
    periodic indicator. Used by conditions.py to precompute a missing column ONCE
    rather than recomputing it per bar — the single source of truth shared with
    add_all_indicators()."""
    name = name.upper()
    period = int(period)
    if name in _CLOSE_PERIODIC:
        return _CLOSE_PERIODIC[name](df["close"], period)
    if name in _OHLC_PERIODIC:
        return _OHLC_PERIODIC[name](df, period)
    if name in ("BB_UPPER", "BB_LOWER", "BB_MID"):
        u, m, l = _bbands(df["close"], period)
        return {"BB_UPPER": u, "BB_MID": m, "BB_LOWER": l}[name]
    if name in ("AROON_UP", "AROON_DOWN"):
        down_s, up_s = aroon(df, period)
        return up_s if name == "AROON_UP" else down_s
    if name == "AROONOSC":
        return aroonosc(df, period)
    if name == "OPENING_RANGE_HIGH":
        return opening_range_series(df, "high", period, "max")
    if name == "OPENING_RANGE_LOW":
        return opening_range_series(df, "low", period, "min")
    return None


def max_indicator_warmup(indicator_config: dict) -> int:
    """
    Return the maximum warm-up candles needed across all indicators.

    Used by the data fetcher to request enough history so that indicators are
    available from the very first tradeable candle. Uses conservative, TA-Lib-
    aware factors (Wilder studies ~2×, cascaded EMAs 2–6×, MACD = slow+signal).
    """
    warmup = 0

    for indicator, periods in indicator_config.items():
        ind = indicator.upper()

        if ind == "MACD":
            warmup = max(warmup, 26 + 9)
            continue
        if ind in ("BB_UPPER", "BB_LOWER", "BB"):
            for n in (periods or [20]):
                warmup = max(warmup, int(n))
            continue
        if ind == "SUPERTREND":
            params = periods if isinstance(periods, dict) else {}
            warmup = max(warmup, int(params.get("period", 7)))
            continue
        if ind == "STOCHRSI":
            params = periods if isinstance(periods, dict) else {}
            warmup = max(warmup, 2 * int(params.get("window", 14)))
            continue
        if ind == "KELTNER":
            params = periods if isinstance(periods, dict) else {}
            warmup = max(warmup, max(int(params.get("window", 20)), 2 * int(params.get("atr_period", 10))))
            continue
        if ind == "ULTOSC":
            params = periods if isinstance(periods, dict) else {}
            warmup = max(warmup, int(params.get("n3", 28)) + 1)
            continue

        plist = periods if isinstance(periods, (list, tuple)) and periods else _default_periods(ind)
        for n in plist:
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if ind in _WILDER_WARMUP or ind in _DMI_FAMILY or ind == "DMI":
                warmup = max(warmup, 2 * n)
            elif ind in _CASCADED_WARMUP:
                warmup = max(warmup, _CASCADED_WARMUP[ind] * n)
            else:
                warmup = max(warmup, n)

    return warmup
