"""Volume signal implementations."""
import pandas as pd
import ta
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry


@RuleRegistry.register("volume_spike", formula="VOL > AVG(VOL, {window}) * {multiplier}")
def volume_spike(df: pd.DataFrame, window: int = 20, multiplier: float = 1.5) -> bool:
    """
    Volume > 1.5x 20-bar average.
    In Indian markets: high delivery volume = institutional interest.
    Strong confirmation signal for any breakout.
    """
    if len(df) < window:
        return False
    avg_vol = df["Volume"].rolling(window).mean()
    if pd.isna(avg_vol.iloc[-1]) or float(avg_vol.iloc[-1]) == 0:
        return False
    return float(df["Volume"].iloc[-1]) > float(avg_vol.iloc[-1]) * multiplier


@RuleRegistry.register("volume_dry_up", formula="VOL < AVG(VOL, {window}) * {multiplier}")
def volume_dry_up(df: pd.DataFrame, window: int = 20, multiplier: float = 0.5) -> bool:
    """
    Volume < 50% of average — volume dry up.
    Often seen before big moves. Low conviction in current direction.
    """
    if len(df) < window:
        return False
    avg_vol = df["Volume"].rolling(window).mean()
    if pd.isna(avg_vol.iloc[-1]) or float(avg_vol.iloc[-1]) == 0:
        return False
    return float(df["Volume"].iloc[-1]) < float(avg_vol.iloc[-1]) * multiplier


@RuleRegistry.register("obv_rising")
def obv_rising(df: pd.DataFrame, window: int = 10) -> bool:
    """OBV rising over last N bars — accumulation by smart money."""
    if len(df) < window + 1:
        return False
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    if obv is None or len(obv) < window:
        return False
    return float(obv.iloc[-1]) > float(obv.iloc[-window])


@RuleRegistry.register("obv_falling")
def obv_falling(df: pd.DataFrame, window: int = 10) -> bool:
    """OBV falling over last N bars — distribution by smart money."""
    if len(df) < window + 1:
        return False
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    if obv is None or len(obv) < window:
        return False
    return float(obv.iloc[-1]) < float(obv.iloc[-window])


@RuleRegistry.register("vwap_bullish", formula="CLOSE > VWAP")
def vwap_bullish(df: pd.DataFrame) -> bool:
    """
    Price above VWAP — strong intraday bullish signal.
    VWAP is the most important intraday level in NSE/BSE.
    Institutions buy above VWAP, sell below.
    """
    if len(df) < 2:
        return False
    vwap = ta.volume.VolumeWeightedAveragePrice(
        df["High"], df["Low"], df["Close"], df["Volume"]
    ).volume_weighted_average_price()
    if vwap is None or pd.isna(vwap.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) > float(vwap.iloc[-1])


@RuleRegistry.register("vwap_bearish", formula="CLOSE < VWAP")
def vwap_bearish(df: pd.DataFrame) -> bool:
    """Price below VWAP — intraday bearish signal."""
    if len(df) < 2:
        return False
    vwap = ta.volume.VolumeWeightedAveragePrice(
        df["High"], df["Low"], df["Close"], df["Volume"]
    ).volume_weighted_average_price()
    if vwap is None or pd.isna(vwap.iloc[-1]):
        return False
    return float(df["Close"].iloc[-1]) < float(vwap.iloc[-1])


@RuleRegistry.register("mfi_oversold")
def mfi_oversold(df: pd.DataFrame, window: int = 14, threshold: float = 20.0) -> bool:
    """Money Flow Index below 20 — oversold with volume confirmation."""
    if len(df) < window + 1:
        return False
    mfi = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=window).money_flow_index()
    if mfi is None or pd.isna(mfi.iloc[-1]):
        return False
    return float(mfi.iloc[-1]) < threshold


@RuleRegistry.register("mfi_overbought")
def mfi_overbought(df: pd.DataFrame, window: int = 14, threshold: float = 80.0) -> bool:
    """Money Flow Index above 80 — overbought with volume confirmation."""
    if len(df) < window + 1:
        return False
    mfi = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=window).money_flow_index()
    if mfi is None or pd.isna(mfi.iloc[-1]):
        return False
    return float(mfi.iloc[-1]) > threshold


@RuleRegistry.register("chaikin_money_flow_positive")
def chaikin_money_flow_positive(df: pd.DataFrame, window: int = 20) -> bool:
    """CMF positive — money flowing into the stock (bullish)."""
    if len(df) < window:
        return False
    cmf = ta.volume.ChaikinMoneyFlowIndicator(
        df["High"], df["Low"], df["Close"], df["Volume"], window=window
    ).chaikin_money_flow()
    if cmf is None or pd.isna(cmf.iloc[-1]):
        return False
    return float(cmf.iloc[-1]) > 0


@RuleRegistry.register("chaikin_money_flow_negative")
def chaikin_money_flow_negative(df: pd.DataFrame, window: int = 20) -> bool:
    """CMF negative — money flowing out of the stock (bearish)."""
    if len(df) < window:
        return False
    cmf = ta.volume.ChaikinMoneyFlowIndicator(
        df["High"], df["Low"], df["Close"], df["Volume"], window=window
    ).chaikin_money_flow()
    if cmf is None or pd.isna(cmf.iloc[-1]):
        return False
    return float(cmf.iloc[-1]) < 0
