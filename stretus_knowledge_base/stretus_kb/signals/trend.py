"""Trend signal implementations."""
import pandas as pd
import ta
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry


@RuleRegistry.register("sma_cross_up", formula="SMA({window_fast}) > SMA({window_slow})")
def sma_cross_up(df: pd.DataFrame, window_fast: int = 20, window_slow: int = 50) -> bool:
    """Fast SMA crosses above slow SMA — golden cross, bullish."""
    if len(df) < window_slow + 1:
        return False
    fast = ta.trend.SMAIndicator(df["Close"], window=window_fast).sma_indicator()
    slow = ta.trend.SMAIndicator(df["Close"], window=window_slow).sma_indicator()
    if fast is None or slow is None or len(fast) < 2:
        return False
    if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return False
    return float(fast.iloc[-2]) <= float(slow.iloc[-2]) and float(fast.iloc[-1]) > float(slow.iloc[-1])


@RuleRegistry.register("sma_cross_down", formula="SMA({window_fast}) < SMA({window_slow})")
def sma_cross_down(df: pd.DataFrame, window_fast: int = 20, window_slow: int = 50) -> bool:
    """Fast SMA crosses below slow SMA — death cross, bearish."""
    if len(df) < window_slow + 1:
        return False
    fast = ta.trend.SMAIndicator(df["Close"], window=window_fast).sma_indicator()
    slow = ta.trend.SMAIndicator(df["Close"], window=window_slow).sma_indicator()
    if fast is None or slow is None or len(fast) < 2:
        return False
    if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return False
    return float(fast.iloc[-2]) >= float(slow.iloc[-2]) and float(fast.iloc[-1]) < float(slow.iloc[-1])


@RuleRegistry.register("is_above_sma", formula="CLOSE > SMA({window})")
def is_above_sma(df: pd.DataFrame, window: int = 50) -> bool:
    """Price above SMA — bullish filter."""
    if len(df) < window:
        return False
    sma = ta.trend.SMAIndicator(df["Close"], window=window).sma_indicator()
    if sma is None or pd.isna(sma.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) > float(sma.iloc[-1])


@RuleRegistry.register("ema_cross_up", formula="EMA({window_fast}) > EMA({window_slow})")
def ema_cross_up(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA crosses above slow EMA — bullish."""
    if len(df) < window_slow + 1:
        return False
    fast = ta.trend.EMAIndicator(df["Close"], window=window_fast).ema_indicator()
    slow = ta.trend.EMAIndicator(df["Close"], window=window_slow).ema_indicator()
    if fast is None or slow is None or len(fast) < 2:
        return False
    if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return False
    return float(fast.iloc[-2]) <= float(slow.iloc[-2]) and float(fast.iloc[-1]) > float(slow.iloc[-1])


@RuleRegistry.register("ema_cross_down", formula="EMA({window_fast}) < EMA({window_slow})")
def ema_cross_down(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA crosses below slow EMA — bearish."""
    if len(df) < window_slow + 1:
        return False
    fast = ta.trend.EMAIndicator(df["Close"], window=window_fast).ema_indicator()
    slow = ta.trend.EMAIndicator(df["Close"], window=window_slow).ema_indicator()
    if fast is None or slow is None or len(fast) < 2:
        return False
    if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return False
    return float(fast.iloc[-2]) >= float(slow.iloc[-2]) and float(fast.iloc[-1]) < float(slow.iloc[-1])


@RuleRegistry.register("ema_above", formula="EMA({window_fast}) > EMA({window_slow})")
def ema_above(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA is above slow EMA (bullish regime, fires every bar in trend)."""
    if len(df) < window_slow + 1:
        return False
    fast = ta.trend.EMAIndicator(df["Close"], window=window_fast).ema_indicator()
    slow = ta.trend.EMAIndicator(df["Close"], window=window_slow).ema_indicator()
    if fast is None or slow is None or pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return False
    return float(fast.iloc[-1]) > float(slow.iloc[-1])


@RuleRegistry.register("ema_below", formula="EMA({window_fast}) < EMA({window_slow})")
def ema_below(df: pd.DataFrame, window_fast: int = 9, window_slow: int = 21) -> bool:
    """Fast EMA is below slow EMA (bearish regime, fires every bar in downtrend)."""
    if len(df) < window_slow + 1:
        return False
    fast = ta.trend.EMAIndicator(df["Close"], window=window_fast).ema_indicator()
    slow = ta.trend.EMAIndicator(df["Close"], window=window_slow).ema_indicator()
    if fast is None or slow is None or pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return False
    return float(fast.iloc[-1]) < float(slow.iloc[-1])


@RuleRegistry.register("price_above_ema", formula="CLOSE > EMA({window})")
def price_above_ema(df: pd.DataFrame, window: int = 20) -> bool:
    """Price above EMA — bullish filter."""
    if len(df) < window:
        return False
    ema = ta.trend.EMAIndicator(df["Close"], window=window).ema_indicator()
    if ema is None or pd.isna(ema.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) > float(ema.iloc[-1])


@RuleRegistry.register("price_below_ema", formula="CLOSE < EMA({window})")
def price_below_ema(df: pd.DataFrame, window: int = 20) -> bool:
    """Price below EMA — bearish filter."""
    if len(df) < window:
        return False
    ema = ta.trend.EMAIndicator(df["Close"], window=window).ema_indicator()
    if ema is None or pd.isna(ema.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) < float(ema.iloc[-1])


@RuleRegistry.register("adx_strong_trend")
def adx_strong_trend(df: pd.DataFrame, window: int = 14, threshold: float = 25.0) -> bool:
    """ADX above threshold — strong trend present (either direction)."""
    if len(df) < window * 2:
        return False
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window).adx()
    if adx is None or pd.isna(adx.iloc[-1]):
        return False
    return float(adx.iloc[-1]) > threshold


@RuleRegistry.register("adx_bullish_di")
def adx_bullish_di(df: pd.DataFrame, window: int = 14) -> bool:
    """DI+ > DI- — bulls in control."""
    if len(df) < window * 2:
        return False
    adx_obj = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window)
    dip = adx_obj.adx_pos()
    din = adx_obj.adx_neg()
    if dip is None or din is None or pd.isna(dip.iloc[-1]) or pd.isna(din.iloc[-1]):
        return False
    return float(dip.iloc[-1]) > float(din.iloc[-1])


@RuleRegistry.register("adx_bearish_di")
def adx_bearish_di(df: pd.DataFrame, window: int = 14) -> bool:
    """DI- > DI+ — bears in control."""
    if len(df) < window * 2:
        return False
    adx_obj = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window)
    dip = adx_obj.adx_pos()
    din = adx_obj.adx_neg()
    if dip is None or din is None or pd.isna(dip.iloc[-1]) or pd.isna(din.iloc[-1]):
        return False
    return float(din.iloc[-1]) > float(dip.iloc[-1])


@RuleRegistry.register("supertrend_bullish")
def supertrend_bullish(df: pd.DataFrame, window: int = 7, multiplier: float = 3.0) -> bool:
    """
    Supertrend bullish — price above supertrend line.
    Very popular intraday signal in Indian markets (Nifty, BankNifty).
    Manually computed since `ta` library doesn't have supertrend.
    """
    if len(df) < window * 2:
        return False
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=window).average_true_range()
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1   # bullish
        elif df["Close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1  # bearish
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    return int(direction.iloc[-1]) == 1


@RuleRegistry.register("supertrend_bearish")
def supertrend_bearish(df: pd.DataFrame, window: int = 7, multiplier: float = 3.0) -> bool:
    """Supertrend bearish — price below supertrend line."""
    if len(df) < window * 2:
        return False
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=window).average_true_range()
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    return int(direction.iloc[-1]) == -1


@RuleRegistry.register("cci_oversold")
def cci_oversold(df: pd.DataFrame, window: int = 20, threshold: float = -100.0) -> bool:
    """CCI below -100 — oversold, bullish reversal setup."""
    if len(df) < window:
        return False
    cci = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=window).cci()
    if cci is None or pd.isna(cci.iloc[-1]):
        return False
    return float(cci.iloc[-1]) < threshold


@RuleRegistry.register("cci_overbought")
def cci_overbought(df: pd.DataFrame, window: int = 20, threshold: float = 100.0) -> bool:
    """CCI above 100 — overbought, bearish reversal setup."""
    if len(df) < window:
        return False
    cci = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=window).cci()
    if cci is None or pd.isna(cci.iloc[-1]):
        return False
    return float(cci.iloc[-1]) > threshold


@RuleRegistry.register("psar_bullish")
def psar_bullish(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> bool:
    """Parabolic SAR below price — bullish trend."""
    if len(df) < 3:
        return False
    psar = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"], step=step, max_step=max_step)
    psar_val = psar.psar()
    if psar_val is None or pd.isna(psar_val.iloc[-1]):
        return False
    return float(psar_val.iloc[-1]) < float(df["Close"].iloc[-1])


@RuleRegistry.register("psar_bearish")
def psar_bearish(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> bool:
    """Parabolic SAR above price — bearish trend."""
    if len(df) < 3:
        return False
    psar = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"], step=step, max_step=max_step)
    psar_val = psar.psar()
    if psar_val is None or pd.isna(psar_val.iloc[-1]):
        return False
    return float(psar_val.iloc[-1]) > float(df["Close"].iloc[-1])
