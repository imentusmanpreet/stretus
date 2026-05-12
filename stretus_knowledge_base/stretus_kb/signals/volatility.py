"""Volatility signal implementations."""
import pandas as pd
import ta
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry


@RuleRegistry.register("bb_squeeze")
def bb_squeeze(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> bool:
    """
    Bollinger Band squeeze — bands narrowing, volatility compression.
    Breakout likely soon. Common setup before big moves in Nifty/BankNifty.
    """
    if len(df) < window * 2:
        return False
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=window_dev)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    mid   = bb.bollinger_mavg()
    if upper is None or pd.isna(upper.iloc[-1]):
        return False
    width = (upper - lower) / mid
    avg_width = width.rolling(window).mean()
    if pd.isna(avg_width.iloc[-1]):
        return False
    return float(width.iloc[-1]) < float(avg_width.iloc[-1]) * 0.8


@RuleRegistry.register("price_above_bb_upper", formula="CLOSE > BB_UPPER({window})")
def price_above_bb_upper(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> bool:
    """Price above upper Bollinger Band — strong breakout signal."""
    if len(df) < window:
        return False
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=window_dev)
    upper = bb.bollinger_hband()
    if upper is None or pd.isna(upper.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) > float(upper.iloc[-1])


@RuleRegistry.register("price_below_bb_lower", formula="CLOSE < BB_LOWER({window})")
def price_below_bb_lower(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> bool:
    """Price below lower Bollinger Band — oversold, potential reversal."""
    if len(df) < window:
        return False
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=window_dev)
    lower = bb.bollinger_lband()
    if lower is None or pd.isna(lower.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) < float(lower.iloc[-1])


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
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=window_dev)
    pct_b = bb.bollinger_pband()
    if pct_b is None or pd.isna(pct_b.iloc[-1]):
        return False
    return float(pct_b.iloc[-1]) > threshold


@RuleRegistry.register(
    "bb_pct_b_low",
    formula="(CLOSE - BB_LOWER({window})) / (BB_UPPER({window}) - BB_LOWER({window})) < {threshold}",
)
def bb_pct_b_low(df: pd.DataFrame, window: int = 20, window_dev: int = 2, threshold: float = 0.2) -> bool:
    """%B below 0.2 — price near lower band, oversold."""
    if len(df) < window:
        return False
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=window_dev)
    pct_b = bb.bollinger_pband()
    if pct_b is None or pd.isna(pct_b.iloc[-1]):
        return False
    return float(pct_b.iloc[-1]) < threshold


@RuleRegistry.register("atr_high_volatility")
def atr_high_volatility(df: pd.DataFrame, window: int = 14, multiplier: float = 1.5) -> bool:
    """ATR above its own moving average — high volatility, be careful with position size."""
    if len(df) < window * 2:
        return False
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=window).average_true_range()
    if atr is None or pd.isna(atr.iloc[-1]):
        return False
    atr_ma = atr.rolling(window).mean()
    if pd.isna(atr_ma.iloc[-1]):
        return False
    return float(atr.iloc[-1]) > float(atr_ma.iloc[-1]) * multiplier


@RuleRegistry.register("atr_low_volatility")
def atr_low_volatility(df: pd.DataFrame, window: int = 14, multiplier: float = 0.7) -> bool:
    """ATR below its own MA — low volatility, tight stops possible."""
    if len(df) < window * 2:
        return False
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=window).average_true_range()
    if atr is None or pd.isna(atr.iloc[-1]):
        return False
    atr_ma = atr.rolling(window).mean()
    if pd.isna(atr_ma.iloc[-1]):
        return False
    return float(atr.iloc[-1]) < float(atr_ma.iloc[-1]) * multiplier


@RuleRegistry.register("keltner_breakout_up")
def keltner_breakout_up(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    """Price above Keltner Channel upper band — strong bullish breakout."""
    if len(df) < window:
        return False
    kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=window, multiplier=multiplier)
    upper = kc.keltner_channel_hband()
    if upper is None or pd.isna(upper.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) > float(upper.iloc[-1])


@RuleRegistry.register("keltner_breakout_down")
def keltner_breakout_down(df: pd.DataFrame, window: int = 20, multiplier: float = 2.0) -> bool:
    """Price below Keltner Channel lower band — strong bearish breakout."""
    if len(df) < window:
        return False
    kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=window, multiplier=multiplier)
    lower = kc.keltner_channel_lband()
    if lower is None or pd.isna(lower.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) < float(lower.iloc[-1])
