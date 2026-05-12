"""Momentum signal implementations."""
import pandas as pd
import ta
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry


@RuleRegistry.register("rsi_oversold", formula="RSI({window}) < {threshold}")
def rsi_oversold(df: pd.DataFrame, window: int = 14, threshold: float = 30.0) -> bool:
    """RSI below oversold threshold — bullish setup. Indian market: 30/70 standard."""
    if len(df) < window + 1:
        return False
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    if rsi is None or pd.isna(rsi.iloc[-1]):
        return False
    return float(rsi.iloc[-1]) < threshold


@RuleRegistry.register("rsi_overbought", formula="RSI({window}) > {threshold}")
def rsi_overbought(df: pd.DataFrame, window: int = 14, threshold: float = 70.0) -> bool:
    """RSI above overbought threshold — bearish setup."""
    if len(df) < window + 1:
        return False
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    if rsi is None or pd.isna(rsi.iloc[-1]):
        return False
    return float(rsi.iloc[-1]) > threshold


@RuleRegistry.register("rsi_cross_up", formula="RSI({window}) > {threshold}")
def rsi_cross_up(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI crosses above threshold — momentum shift bullish."""
    if len(df) < window + 2:
        return False
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    if rsi is None or len(rsi) < 2:
        return False
    return float(rsi.iloc[-2]) <= threshold and float(rsi.iloc[-1]) > threshold


@RuleRegistry.register("rsi_cross_down", formula="RSI({window}) < {threshold}")
def rsi_cross_down(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI crosses below threshold — momentum shift bearish."""
    if len(df) < window + 2:
        return False
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    if rsi is None or len(rsi) < 2:
        return False
    return float(rsi.iloc[-2]) >= threshold and float(rsi.iloc[-1]) < threshold


@RuleRegistry.register("rsi_above_50", formula="RSI({window}) > {threshold}")
def rsi_above_50(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI is above the bullish-momentum threshold (regime, not crossover bar)."""
    if len(df) < window + 1:
        return False
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    if rsi is None or pd.isna(rsi.iloc[-1]):
        return False
    return float(rsi.iloc[-1]) > threshold


@RuleRegistry.register("rsi_below_50", formula="RSI({window}) < {threshold}")
def rsi_below_50(df: pd.DataFrame, window: int = 14, threshold: float = 50.0) -> bool:
    """RSI is below the bullish-momentum threshold (regime, not crossover bar)."""
    if len(df) < window + 1:
        return False
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    if rsi is None or pd.isna(rsi.iloc[-1]):
        return False
    return float(rsi.iloc[-1]) < threshold


@RuleRegistry.register("macd_negative", formula="MACD < 0")
def macd_negative(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD line is below zero (bearish regime)."""
    if len(df) < window_slow + window_sign:
        return False
    macd_obj = ta.trend.MACD(df["Close"], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
    macd_line = macd_obj.macd()
    if macd_line is None or pd.isna(macd_line.iloc[-1]):
        return False
    return float(macd_line.iloc[-1]) < 0


@RuleRegistry.register("macd_bullish_cross", formula="MACD > 0")
def macd_bullish_cross(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD line crosses above signal line — bullish."""
    if len(df) < window_slow + window_sign:
        return False
    macd_obj = ta.trend.MACD(df["Close"], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
    macd_line = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    if macd_line is None or len(macd_line) < 2:
        return False
    if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return False
    prev_below = float(macd_line.iloc[-2]) <= float(signal_line.iloc[-2])
    curr_above = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
    return prev_below and curr_above


@RuleRegistry.register("macd_bearish_cross", formula="MACD < 0")
def macd_bearish_cross(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD line crosses below signal line — bearish."""
    if len(df) < window_slow + window_sign:
        return False
    macd_obj = ta.trend.MACD(df["Close"], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
    macd_line = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    if macd_line is None or len(macd_line) < 2:
        return False
    if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return False
    prev_above = float(macd_line.iloc[-2]) >= float(signal_line.iloc[-2])
    curr_below = float(macd_line.iloc[-1]) < float(signal_line.iloc[-1])
    return prev_above and curr_below


@RuleRegistry.register("macd_positive", formula="MACD > 0")
def macd_positive(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9) -> bool:
    """MACD histogram positive — bullish momentum."""
    if len(df) < window_slow + window_sign:
        return False
    macd_obj = ta.trend.MACD(df["Close"], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
    hist = macd_obj.macd_diff()
    if hist is None or pd.isna(hist.iloc[-1]):
        return False
    return float(hist.iloc[-1]) > 0


@RuleRegistry.register("stoch_oversold")
def stoch_oversold(df: pd.DataFrame, window: int = 14, smooth_window: int = 3, threshold: float = 20.0) -> bool:
    """Stochastic %K below oversold — bullish setup."""
    if len(df) < window + smooth_window:
        return False
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"],
        window=window, smooth_window=smooth_window
    )
    k = stoch.stoch()
    if k is None or pd.isna(k.iloc[-1]):
        return False
    return float(k.iloc[-1]) < threshold


@RuleRegistry.register("stoch_overbought")
def stoch_overbought(df: pd.DataFrame, window: int = 14, smooth_window: int = 3, threshold: float = 80.0) -> bool:
    """Stochastic %K above overbought — bearish setup."""
    if len(df) < window + smooth_window:
        return False
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"],
        window=window, smooth_window=smooth_window
    )
    k = stoch.stoch()
    if k is None or pd.isna(k.iloc[-1]):
        return False
    return float(k.iloc[-1]) > threshold


@RuleRegistry.register("williams_r_oversold")
def williams_r_oversold(df: pd.DataFrame, lbp: int = 14, threshold: float = -80.0) -> bool:
    """Williams %R below -80 — oversold, bullish setup."""
    if len(df) < lbp:
        return False
    wr = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=lbp).williams_r()
    if wr is None or pd.isna(wr.iloc[-1]):
        return False
    return float(wr.iloc[-1]) < threshold


@RuleRegistry.register("williams_r_overbought")
def williams_r_overbought(df: pd.DataFrame, lbp: int = 14, threshold: float = -20.0) -> bool:
    """Williams %R above -20 — overbought, bearish setup."""
    if len(df) < lbp:
        return False
    wr = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=lbp).williams_r()
    if wr is None or pd.isna(wr.iloc[-1]):
        return False
    return float(wr.iloc[-1]) > threshold


@RuleRegistry.register("roc_positive", formula="(CLOSE - PREV(CLOSE, {window})) / PREV(CLOSE, {window}) > 0")
def roc_positive(df: pd.DataFrame, window: int = 12) -> bool:
    """Rate of Change positive — bullish momentum."""
    if len(df) < window + 1:
        return False
    roc = ta.momentum.ROCIndicator(df["Close"], window=window).roc()
    if roc is None or pd.isna(roc.iloc[-1]):
        return False
    return float(roc.iloc[-1]) > 0


@RuleRegistry.register("roc_negative", formula="(CLOSE - PREV(CLOSE, {window})) / PREV(CLOSE, {window}) < 0")
def roc_negative(df: pd.DataFrame, window: int = 12) -> bool:
    """Rate of Change negative — bearish momentum."""
    if len(df) < window + 1:
        return False
    roc = ta.momentum.ROCIndicator(df["Close"], window=window).roc()
    if roc is None or pd.isna(roc.iloc[-1]):
        return False
    return float(roc.iloc[-1]) < 0
