"""Momentum signal implementations."""
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


@RuleRegistry.register("rsi_oversold", formula="RSI({window}) < {threshold}")
def rsi_oversold(df: pd.DataFrame, window: int = 14, threshold: float = 30.0) -> bool:
    """RSI below oversold threshold — bullish setup. Indian market: 30/70 standard."""
    if len(df) < window + 1:
        return False
    rsi = talib.RSI(_close(df), timeperiod=window)
    if np.isnan(rsi[-1]):
        return False
    return float(rsi[-1]) < threshold


@RuleRegistry.register("rsi_overbought", formula="RSI({window}) > {threshold}")
def rsi_overbought(df: pd.DataFrame, window: int = 14, threshold: float = 70.0) -> bool:
    """RSI above overbought threshold — bearish setup."""
    if len(df) < window + 1:
        return False
    rsi = talib.RSI(_close(df), timeperiod=window)
    if np.isnan(rsi[-1]):
        return False
    return float(rsi[-1]) > threshold


@RuleRegistry.register("rsi_cross_up", formula="RSI({window}) > {threshold}")
def rsi_cross_up(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI crosses above threshold — momentum shift bullish."""
    if len(df) < window + 2:
        return False
    rsi = talib.RSI(_close(df), timeperiod=window)
    if len(rsi) < 2 or np.isnan(rsi[-1]) or np.isnan(rsi[-2]):
        return False
    return float(rsi[-2]) <= threshold and float(rsi[-1]) > threshold


@RuleRegistry.register("rsi_cross_down", formula="RSI({window}) < {threshold}")
def rsi_cross_down(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI crosses below threshold — momentum shift bearish."""
    if len(df) < window + 2:
        return False
    rsi = talib.RSI(_close(df), timeperiod=window)
    if len(rsi) < 2 or np.isnan(rsi[-1]) or np.isnan(rsi[-2]):
        return False
    return float(rsi[-2]) >= threshold and float(rsi[-1]) < threshold


@RuleRegistry.register("rsi_above_50", formula="RSI({window}) > {threshold}")
def rsi_above_50(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI is above the bullish-momentum threshold (regime, not crossover bar)."""
    if len(df) < window + 1:
        return False
    rsi = talib.RSI(_close(df), timeperiod=window)
    if np.isnan(rsi[-1]):
        return False
    return float(rsi[-1]) > threshold


@RuleRegistry.register("rsi_below_50", formula="RSI({window}) < {threshold}")
def rsi_below_50(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI is below the bullish-momentum threshold (regime, not crossover bar)."""
    if len(df) < window + 1:
        return False
    rsi = talib.RSI(_close(df), timeperiod=window)
    if np.isnan(rsi[-1]):
        return False
    return float(rsi[-1]) < threshold


@RuleRegistry.register("macd_negative", formula="MACD < 0")
def macd_negative(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD line is below zero (bearish regime)."""
    if len(df) < window_slow + window_sign:
        return False
    macd_line, _signal, _hist = talib.MACD(
        _close(df), fastperiod=window_fast, slowperiod=window_slow, signalperiod=window_sign
    )
    if np.isnan(macd_line[-1]):
        return False
    return float(macd_line[-1]) < 0


@RuleRegistry.register("macd_bullish_cross", formula="MACD > 0")
def macd_bullish_cross(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD line crosses above signal line — bullish."""
    if len(df) < window_slow + window_sign:
        return False
    macd_line, signal_line, _hist = talib.MACD(
        _close(df), fastperiod=window_fast, slowperiod=window_slow, signalperiod=window_sign
    )
    if len(macd_line) < 2:
        return False
    if np.isnan(macd_line[-1]) or np.isnan(signal_line[-1]) or np.isnan(macd_line[-2]) or np.isnan(signal_line[-2]):
        return False
    prev_below = float(macd_line[-2]) <= float(signal_line[-2])
    curr_above = float(macd_line[-1]) > float(signal_line[-1])
    return prev_below and curr_above


@RuleRegistry.register("macd_bearish_cross", formula="MACD < 0")
def macd_bearish_cross(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD line crosses below signal line — bearish."""
    if len(df) < window_slow + window_sign:
        return False
    macd_line, signal_line, _hist = talib.MACD(
        _close(df), fastperiod=window_fast, slowperiod=window_slow, signalperiod=window_sign
    )
    if len(macd_line) < 2:
        return False
    if np.isnan(macd_line[-1]) or np.isnan(signal_line[-1]) or np.isnan(macd_line[-2]) or np.isnan(signal_line[-2]):
        return False
    prev_above = float(macd_line[-2]) >= float(signal_line[-2])
    curr_below = float(macd_line[-1]) < float(signal_line[-1])
    return prev_above and curr_below


@RuleRegistry.register("macd_positive", formula="MACD > 0")
def macd_positive(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD histogram positive — bullish momentum."""
    if len(df) < window_slow + window_sign:
        return False
    _macd, _signal, hist = talib.MACD(
        _close(df), fastperiod=window_fast, slowperiod=window_slow, signalperiod=window_sign
    )
    if np.isnan(hist[-1]):
        return False
    return float(hist[-1]) > 0


@RuleRegistry.register("stoch_oversold")
def stoch_oversold(df: pd.DataFrame, window: int = 14, smooth_window: int = 3, threshold: float = 20.0) -> bool:
    """Stochastic %K below oversold — bullish setup."""
    if len(df) < window + smooth_window:
        return False
    high, low, close = _hlc(df)
    slowk, _slowd = talib.STOCH(
        high, low, close,
        fastk_period=window, slowk_period=smooth_window, slowk_matype=0,
        slowd_period=smooth_window, slowd_matype=0,
    )
    if np.isnan(slowk[-1]):
        return False
    return float(slowk[-1]) < threshold


@RuleRegistry.register("stoch_overbought")
def stoch_overbought(df: pd.DataFrame, window: int = 14, smooth_window: int = 3, threshold: float = 80.0) -> bool:
    """Stochastic %K above overbought — bearish setup."""
    if len(df) < window + smooth_window:
        return False
    high, low, close = _hlc(df)
    slowk, _slowd = talib.STOCH(
        high, low, close,
        fastk_period=window, slowk_period=smooth_window, slowk_matype=0,
        slowd_period=smooth_window, slowd_matype=0,
    )
    if np.isnan(slowk[-1]):
        return False
    return float(slowk[-1]) > threshold


@RuleRegistry.register("williams_r_oversold")
def williams_r_oversold(df: pd.DataFrame, lbp: int = 14, threshold: float = -80.0) -> bool:
    """Williams %R below -80 — oversold, bullish setup."""
    if len(df) < lbp:
        return False
    high, low, close = _hlc(df)
    wr = talib.WILLR(high, low, close, timeperiod=lbp)
    if np.isnan(wr[-1]):
        return False
    return float(wr[-1]) < threshold


@RuleRegistry.register("williams_r_overbought")
def williams_r_overbought(df: pd.DataFrame, lbp: int = 14, threshold: float = -20.0) -> bool:
    """Williams %R above -20 — overbought, bearish setup."""
    if len(df) < lbp:
        return False
    high, low, close = _hlc(df)
    wr = talib.WILLR(high, low, close, timeperiod=lbp)
    if np.isnan(wr[-1]):
        return False
    return float(wr[-1]) > threshold


@RuleRegistry.register("roc_positive", formula="(CLOSE - PREV(CLOSE, {window})) / PREV(CLOSE, {window}) > 0")
def roc_positive(df: pd.DataFrame, window: int = 12) -> bool:
    """Rate of Change positive — bullish momentum."""
    if len(df) < window + 1:
        return False
    roc = talib.ROC(_close(df), timeperiod=window)
    if np.isnan(roc[-1]):
        return False
    return float(roc[-1]) > 0


@RuleRegistry.register("roc_negative", formula="(CLOSE - PREV(CLOSE, {window})) / PREV(CLOSE, {window}) < 0")
def roc_negative(df: pd.DataFrame, window: int = 12) -> bool:
    """Rate of Change negative — bearish momentum."""
    if len(df) < window + 1:
        return False
    roc = talib.ROC(_close(df), timeperiod=window)
    if np.isnan(roc[-1]):
        return False
    return float(roc[-1]) < 0
