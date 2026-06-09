"""India-specific signal implementations."""
"""
India-specific signals — ye Mangrove me nahi the.
NSE/BSE ke liye: circuit breakers, F&O ban, India VIX, market hours.
Uses TA-Lib (C-backed) for the SMA-based broad-market filter.
"""
import numpy as np
import pandas as pd
import talib
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry

NSE_OPEN_HOUR  = 9
NSE_OPEN_MIN   = 15
NSE_CLOSE_HOUR = 15
NSE_CLOSE_MIN  = 30


@RuleRegistry.register("within_market_hours")
def within_market_hours(df: pd.DataFrame) -> bool:
    """
    True if last candle is within NSE market hours (9:15 - 15:30 IST).
    Use as a FILTER — don't trade pre/post market.
    """
    if df.index.tzinfo is None:
        return True
    last = df.index[-1]
    after_open   = (last.hour, last.minute) >= (NSE_OPEN_HOUR, NSE_OPEN_MIN)
    before_close = (last.hour, last.minute) <= (NSE_CLOSE_HOUR, NSE_CLOSE_MIN)
    return after_open and before_close


@RuleRegistry.register("not_near_circuit")
def not_near_circuit(df: pd.DataFrame, circuit_pct: float = 5.0, buffer_pct: float = 1.0) -> bool:
    """
    True if price is NOT near upper/lower circuit limit.
    Prevents getting trapped at circuit limits (NSE 5/10/20% circuits).
    """
    if len(df) < 2:
        return True
    prev_close = float(df["Close"].iloc[-2])
    curr_close = float(df["Close"].iloc[-1])
    upper_circuit = prev_close * (1 + circuit_pct / 100)
    lower_circuit = prev_close * (1 - circuit_pct / 100)
    near_upper = curr_close >= upper_circuit * (1 - buffer_pct / 100)
    near_lower = curr_close <= lower_circuit * (1 + buffer_pct / 100)
    return not (near_upper or near_lower)


@RuleRegistry.register("avoid_expiry_day")
def avoid_expiry_day(df: pd.DataFrame) -> bool:
    """
    True if today is NOT Thursday (NSE weekly F&O expiry).
    Expiry day has abnormal volatility — avoid for positional trades.
    """
    last_date = df.index[-1]
    if hasattr(last_date, 'weekday'):
        return last_date.weekday() != 3  # 3 = Thursday
    return True


@RuleRegistry.register("india_vix_low")
def india_vix_low(df: pd.DataFrame, vix_threshold: float = 20.0) -> bool:
    """
    True if India VIX is below threshold (low fear = good to trade).
    Pass India VIX OHLCV as df (fetch ^INDIAVIX from yfinance).
    Below 15 = calm, 15-20 = normal, above 20 = fearful.
    """
    if "Close" not in df.columns or len(df) < 1:
        return True
    return float(df["Close"].iloc[-1]) < vix_threshold


@RuleRegistry.register("gap_up_open")
def gap_up_open(df: pd.DataFrame, gap_pct: float = 0.5) -> bool:
    """
    Today's open > yesterday's close by gap_pct%.
    Common gap-up strategy trigger in Indian markets.
    """
    if len(df) < 2:
        return False
    prev_close = float(df["Close"].iloc[-2])
    today_open = float(df["Open"].iloc[-1])
    if prev_close == 0:
        return False
    return (today_open - prev_close) / prev_close * 100 >= gap_pct


@RuleRegistry.register("gap_down_open")
def gap_down_open(df: pd.DataFrame, gap_pct: float = 0.5) -> bool:
    """
    Today's open < yesterday's close by gap_pct%.
    Gap-down signals weakness — bearish setup.
    """
    if len(df) < 2:
        return False
    prev_close = float(df["Close"].iloc[-2])
    today_open = float(df["Open"].iloc[-1])
    if prev_close == 0:
        return False
    return (prev_close - today_open) / prev_close * 100 >= gap_pct


@RuleRegistry.register("nifty50_trend_filter")
def nifty50_trend_filter(df: pd.DataFrame, window: int = 50) -> bool:
    """
    Is NIFTY 50 above its 50-day SMA?
    Pass NIFTY 50 OHLCV as df — broad market filter.
    If Nifty is below 50-SMA, avoid long positions.
    """
    if len(df) < window:
        return True
    close = df["Close"].to_numpy(dtype=float)
    sma = talib.SMA(close, timeperiod=window)
    if np.isnan(sma[-1]):
        return True
    return float(close[-1]) > float(sma[-1])


@RuleRegistry.register(
    "high_delivery_volume",
    formula="CLOSE > OPEN AND VOL > AVG(VOL, {window}) * {multiplier}",
)
def high_delivery_volume(df: pd.DataFrame, window: int = 10, multiplier: float = 1.3) -> bool:
    """
    Volume significantly above recent average on a green candle.
    High delivery = institutional buying in Indian cash market.
    """
    if len(df) < window + 1:
        return False
    is_green = float(df["Close"].iloc[-1]) > float(df["Open"].iloc[-1])
    avg_vol = df["Volume"].rolling(window).mean()
    if pd.isna(avg_vol.iloc[-1]) or float(avg_vol.iloc[-1]) == 0:
        return False
    high_vol = float(df["Volume"].iloc[-1]) > float(avg_vol.iloc[-1]) * multiplier
    return is_green and high_vol


@RuleRegistry.register("inside_bar", formula="HIGH <= PREV(HIGH, 1) AND LOW >= PREV(LOW, 1)")
def inside_bar(df: pd.DataFrame) -> bool:
    """
    Current candle's high/low inside previous candle's range.
    Inside bar = consolidation, breakout likely. Common in Nifty/BankNifty.
    """
    if len(df) < 2:
        return False
    curr_high = float(df["High"].iloc[-1])
    curr_low  = float(df["Low"].iloc[-1])
    prev_high = float(df["High"].iloc[-2])
    prev_low  = float(df["Low"].iloc[-2])
    return curr_high <= prev_high and curr_low >= prev_low


@RuleRegistry.register("opening_range_breakout", formula="CLOSE > OPENING_RANGE_HIGH({opening_bars})")
def opening_range_breakout(df: pd.DataFrame, opening_bars: int = 4) -> bool:
    """
    Price breaks above the high of first N bars (Opening Range Breakout).
    ORB is hugely popular intraday strategy on NSE — especially 9:15-9:30 range.
    opening_bars=4 means first 4 x 5min candles = 20 min opening range.
    """
    if len(df) < opening_bars + 1:
        return False
    opening_high = df["High"].iloc[:opening_bars].max()
    return float(df["Close"].iloc[-1]) > float(opening_high)
