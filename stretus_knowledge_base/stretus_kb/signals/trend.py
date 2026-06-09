"""Trend signal implementations."""
import numpy as np
import pandas as pd
import talib
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry


def _close(df: pd.DataFrame) -> np.ndarray:
    return df["Close"].to_numpy(dtype=float)


def _hlc(df: pd.DataFrame):
    return (
        df["High"].to_numpy(dtype=float),
        df["Low"].to_numpy(dtype=float),
        df["Close"].to_numpy(dtype=float),
    )


@RuleRegistry.register("sma_cross_up", formula="SMA({window_fast}) > SMA({window_slow})")
def sma_cross_up(df: pd.DataFrame, window_fast: int = 20, window_slow: int = 50) -> bool:
    """Fast SMA crosses above slow SMA — golden cross, bullish."""
    if len(df) < window_slow + 1:
        return False
    close = _close(df)
    fast = talib.SMA(close, timeperiod=window_fast)
    slow = talib.SMA(close, timeperiod=window_slow)
    if len(fast) < 2 or np.isnan(fast[-1]) or np.isnan(slow[-1]) or np.isnan(fast[-2]) or np.isnan(slow[-2]):
        return False
    return float(fast[-2]) <= float(slow[-2]) and float(fast[-1]) > float(slow[-1])


@RuleRegistry.register("sma_cross_down", formula="SMA({window_fast}) < SMA({window_slow})")
def sma_cross_down(df: pd.DataFrame, window_fast: int = 20, window_slow: int = 50) -> bool:
    """Fast SMA crosses below slow SMA — death cross, bearish."""
    if len(df) < window_slow + 1:
        return False
    close = _close(df)
    fast = talib.SMA(close, timeperiod=window_fast)
    slow = talib.SMA(close, timeperiod=window_slow)
    if len(fast) < 2 or np.isnan(fast[-1]) or np.isnan(slow[-1]) or np.isnan(fast[-2]) or np.isnan(slow[-2]):
        return False
    return float(fast[-2]) >= float(slow[-2]) and float(fast[-1]) < float(slow[-1])


@RuleRegistry.register("is_above_sma", formula="CLOSE > SMA({window})")
def is_above_sma(df: pd.DataFrame, window: int = 50) -> bool:
    """Price above SMA — bullish filter."""
    if len(df) < window:
        return False
    close = _close(df)
    sma = talib.SMA(close, timeperiod=window)
    if np.isnan(sma[-1]):
        return False
    return float(close[-1]) > float(sma[-1])


@RuleRegistry.register("ema_cross_up", formula="EMA({window_fast}) > EMA({window_slow})")
def ema_cross_up(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA crosses above slow EMA — bullish."""
    if len(df) < window_slow + 1:
        return False
    close = _close(df)
    fast = talib.EMA(close, timeperiod=window_fast)
    slow = talib.EMA(close, timeperiod=window_slow)
    if len(fast) < 2 or np.isnan(fast[-1]) or np.isnan(slow[-1]) or np.isnan(fast[-2]) or np.isnan(slow[-2]):
        return False
    return float(fast[-2]) <= float(slow[-2]) and float(fast[-1]) > float(slow[-1])


@RuleRegistry.register("ema_cross_down", formula="EMA({window_fast}) < EMA({window_slow})")
def ema_cross_down(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA crosses below slow EMA — bearish."""
    if len(df) < window_slow + 1:
        return False
    close = _close(df)
    fast = talib.EMA(close, timeperiod=window_fast)
    slow = talib.EMA(close, timeperiod=window_slow)
    if len(fast) < 2 or np.isnan(fast[-1]) or np.isnan(slow[-1]) or np.isnan(fast[-2]) or np.isnan(slow[-2]):
        return False
    return float(fast[-2]) >= float(slow[-2]) and float(fast[-1]) < float(slow[-1])


@RuleRegistry.register("ema_above", formula="EMA({window_fast}) > EMA({window_slow})")
def ema_above(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA is above slow EMA (bullish regime, fires every bar in trend)."""
    if len(df) < window_slow + 1:
        return False
    close = _close(df)
    fast = talib.EMA(close, timeperiod=window_fast)
    slow = talib.EMA(close, timeperiod=window_slow)
    if np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return False
    return float(fast[-1]) > float(slow[-1])


@RuleRegistry.register("ema_below", formula="EMA({window_fast}) < EMA({window_slow})")
def ema_below(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA is below slow EMA (bearish regime, fires every bar in downtrend)."""
    if len(df) < window_slow + 1:
        return False
    close = _close(df)
    fast = talib.EMA(close, timeperiod=window_fast)
    slow = talib.EMA(close, timeperiod=window_slow)
    if np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return False
    return float(fast[-1]) < float(slow[-1])


@RuleRegistry.register("price_above_ema", formula="CLOSE > EMA({window})")
def price_above_ema(df: pd.DataFrame, window: int = 20) -> bool:
    """Price above EMA — bullish filter."""
    if len(df) < window:
        return False
    close = _close(df)
    ema = talib.EMA(close, timeperiod=window)
    if np.isnan(ema[-1]):
        return False
    return float(close[-1]) > float(ema[-1])


@RuleRegistry.register("price_below_ema", formula="CLOSE < EMA({window})")
def price_below_ema(df: pd.DataFrame, window: int = 20) -> bool:
    """Price below EMA — bearish filter."""
    if len(df) < window:
        return False
    close = _close(df)
    ema = talib.EMA(close, timeperiod=window)
    if np.isnan(ema[-1]):
        return False
    return float(close[-1]) < float(ema[-1])


@RuleRegistry.register("adx_strong_trend")
def adx_strong_trend(df: pd.DataFrame, window: int = 14, threshold: float = 25.0) -> bool:
    """ADX above threshold — strong trend present (either direction)."""
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    adx = talib.ADX(high, low, close, timeperiod=window)
    if np.isnan(adx[-1]):
        return False
    return float(adx[-1]) > threshold


@RuleRegistry.register("adx_bullish_di")
def adx_bullish_di(df: pd.DataFrame, window: int = 14) -> bool:
    """DI+ > DI- — bulls in control."""
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    dip = talib.PLUS_DI(high, low, close, timeperiod=window)
    din = talib.MINUS_DI(high, low, close, timeperiod=window)
    if np.isnan(dip[-1]) or np.isnan(din[-1]):
        return False
    return float(dip[-1]) > float(din[-1])


@RuleRegistry.register("adx_bearish_di")
def adx_bearish_di(df: pd.DataFrame, window: int = 14) -> bool:
    """DI- > DI+ — bears in control."""
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    dip = talib.PLUS_DI(high, low, close, timeperiod=window)
    din = talib.MINUS_DI(high, low, close, timeperiod=window)
    if np.isnan(dip[-1]) or np.isnan(din[-1]):
        return False
    return float(din[-1]) > float(dip[-1])


@RuleRegistry.register("supertrend_bullish")
def supertrend_bullish(df: pd.DataFrame, window: int = 7, multiplier: float = 3.0) -> bool:
    """
    Supertrend bullish — price above supertrend line.
    Very popular intraday signal in Indian markets (Nifty, BankNifty).
    Manually computed since TA-Lib doesn't have supertrend.
    """
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    atr = talib.ATR(high, low, close, timeperiod=window)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if close[i] > upper_band[i - 1]:
            direction[i] = 1
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return int(direction[-1]) == 1


@RuleRegistry.register("supertrend_bearish")
def supertrend_bearish(df: pd.DataFrame, window: int = 7, multiplier: float = 3.0) -> bool:
    """Supertrend bearish — price below supertrend line."""
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    atr = talib.ATR(high, low, close, timeperiod=window)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if close[i] > upper_band[i - 1]:
            direction[i] = 1
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return int(direction[-1]) == -1


@RuleRegistry.register("cci_oversold")
def cci_oversold(df: pd.DataFrame, window: int = 20, threshold: float = -100.0) -> bool:
    """CCI below -100 — oversold, bullish reversal setup."""
    if len(df) < window:
        return False
    high, low, close = _hlc(df)
    cci = talib.CCI(high, low, close, timeperiod=window)
    if np.isnan(cci[-1]):
        return False
    return float(cci[-1]) < threshold


@RuleRegistry.register("cci_overbought")
def cci_overbought(df: pd.DataFrame, window: int = 20, threshold: float = 100.0) -> bool:
    """CCI above 100 — overbought, bearish reversal setup."""
    if len(df) < window:
        return False
    high, low, close = _hlc(df)
    cci = talib.CCI(high, low, close, timeperiod=window)
    if np.isnan(cci[-1]):
        return False
    return float(cci[-1]) > threshold


@RuleRegistry.register("psar_bullish")
def psar_bullish(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> bool:
    """Parabolic SAR below price — bullish trend."""
    if len(df) < 3:
        return False
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    psar_val = talib.SAR(high, low, acceleration=step, maximum=max_step)
    if np.isnan(psar_val[-1]):
        return False
    return float(psar_val[-1]) < float(close[-1])


@RuleRegistry.register("psar_bearish")
def psar_bearish(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> bool:
    """Parabolic SAR above price — bearish trend."""
    if len(df) < 3:
        return False
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    psar_val = talib.SAR(high, low, acceleration=step, maximum=max_step)
    if np.isnan(psar_val[-1]):
        return False
    return float(psar_val[-1]) > float(close[-1])
