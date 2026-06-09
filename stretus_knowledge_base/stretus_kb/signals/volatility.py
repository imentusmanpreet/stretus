"""Volatility signal implementations."""
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


@RuleRegistry.register("bb_squeeze")
def bb_squeeze(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> bool:
    """
    Bollinger Band squeeze — bands narrowing, volatility compression.
    Breakout likely soon. Common setup before big moves in Nifty/BankNifty.
    """
    if len(df) < window * 2:
        return False
    upper, mid, lower = talib.BBANDS(_close(df), timeperiod=window, nbdevup=window_dev, nbdevdn=window_dev, matype=0)
    if np.isnan(upper[-1]):
        return False
    width = (upper - lower) / mid
    avg_width = pd.Series(width).rolling(window).mean().to_numpy()
    if np.isnan(avg_width[-1]):
        return False
    return float(width[-1]) < float(avg_width[-1]) * 0.8


@RuleRegistry.register("price_above_bb_upper", formula="CLOSE > BB_UPPER({window})")
def price_above_bb_upper(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> bool:
    """Price above upper Bollinger Band — strong breakout signal."""
    if len(df) < window:
        return False
    close = _close(df)
    upper, _mid, _lower = talib.BBANDS(close, timeperiod=window, nbdevup=window_dev, nbdevdn=window_dev, matype=0)
    if np.isnan(upper[-1]):
        return False
    return float(close[-1]) > float(upper[-1])


@RuleRegistry.register("price_below_bb_lower", formula="CLOSE < BB_LOWER({window})")
def price_below_bb_lower(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> bool:
    """Price below lower Bollinger Band — oversold, potential reversal."""
    if len(df) < window:
        return False
    close = _close(df)
    _upper, _mid, lower = talib.BBANDS(close, timeperiod=window, nbdevup=window_dev, nbdevdn=window_dev, matype=0)
    if np.isnan(lower[-1]):
        return False
    return float(close[-1]) < float(lower[-1])


@RuleRegistry.register(
    "bb_pct_b_high",
    formula="(CLOSE - BB_LOWER({window})) / (BB_UPPER({window}) - BB_LOWER({window})) > {threshold}",
)
def bb_pct_b_high(df: pd.DataFrame, window: int = 20, window_dev: int = 2, threshold: float = 0.8) -> bool:
    """
    %B above 0.8 — price near upper band, momentum strong.
    %B = (Price - Lower) / (Upper - Lower)
    """
    if len(df) < window:
        return False
    close = _close(df)
    upper, _mid, lower = talib.BBANDS(close, timeperiod=window, nbdevup=window_dev, nbdevdn=window_dev, matype=0)
    if np.isnan(upper[-1]) or np.isnan(lower[-1]) or upper[-1] == lower[-1]:
        return False
    pct_b = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
    return float(pct_b) > threshold


@RuleRegistry.register(
    "bb_pct_b_low",
    formula="(CLOSE - BB_LOWER({window})) / (BB_UPPER({window}) - BB_LOWER({window})) < {threshold}",
)
def bb_pct_b_low(df: pd.DataFrame, window: int = 20, window_dev: int = 2, threshold: float = 0.2) -> bool:
    """%B below 0.2 — price near lower band, oversold."""
    if len(df) < window:
        return False
    close = _close(df)
    upper, _mid, lower = talib.BBANDS(close, timeperiod=window, nbdevup=window_dev, nbdevdn=window_dev, matype=0)
    if np.isnan(upper[-1]) or np.isnan(lower[-1]) or upper[-1] == lower[-1]:
        return False
    pct_b = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
    return float(pct_b) < threshold


@RuleRegistry.register("atr_high_volatility")
def atr_high_volatility(df: pd.DataFrame, window: int = 14, multiplier: float = 1.5) -> bool:
    """ATR above its own moving average — high volatility, be careful with position size."""
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    atr = talib.ATR(high, low, close, timeperiod=window)
    if np.isnan(atr[-1]):
        return False
    atr_ma = pd.Series(atr).rolling(window).mean().to_numpy()
    if np.isnan(atr_ma[-1]):
        return False
    return float(atr[-1]) > float(atr_ma[-1]) * multiplier


@RuleRegistry.register("atr_low_volatility")
def atr_low_volatility(df: pd.DataFrame, window: int = 14, multiplier: float = 0.7) -> bool:
    """ATR below its own MA — low volatility, tight stops possible."""
    if len(df) < window * 2:
        return False
    high, low, close = _hlc(df)
    atr = talib.ATR(high, low, close, timeperiod=window)
    if np.isnan(atr[-1]):
        return False
    atr_ma = pd.Series(atr).rolling(window).mean().to_numpy()
    if np.isnan(atr_ma[-1]):
        return False
    return float(atr[-1]) < float(atr_ma[-1]) * multiplier


@RuleRegistry.register("keltner_breakout_up")
def keltner_breakout_up(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    """
    Price above Keltner Channel upper band — strong bullish breakout.
    Keltner = EMA(close, N) ± multiplier * ATR(N). TA-Lib has no native Keltner,
    so we compose it from EMA + ATR primitives.
    """
    if len(df) < window:
        return False
    high, low, close = _hlc(df)
    ema = talib.EMA(close, timeperiod=window)
    atr = talib.ATR(high, low, close, timeperiod=window)
    if np.isnan(ema[-1]) or np.isnan(atr[-1]):
        return False
    upper = ema[-1] + multiplier * atr[-1]
    return float(close[-1]) > float(upper)


@RuleRegistry.register("keltner_breakout_down")
def keltner_breakout_down(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    """Price below Keltner Channel lower band — strong bearish breakout."""
    if len(df) < window:
        return False
    high, low, close = _hlc(df)
    ema = talib.EMA(close, timeperiod=window)
    atr = talib.ATR(high, low, close, timeperiod=window)
    if np.isnan(ema[-1]) or np.isnan(atr[-1]):
        return False
    lower = ema[-1] - multiplier * atr[-1]
    return float(close[-1]) < float(lower)
