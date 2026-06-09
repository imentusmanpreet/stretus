"""Volume signal implementations."""
import numpy as np
import pandas as pd
import talib
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
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    obv = talib.OBV(close, volume)
    if len(obv) < window:
        return False
    return float(obv[-1]) > float(obv[-window])


@RuleRegistry.register("obv_falling")
def obv_falling(df: pd.DataFrame, window: int = 10) -> bool:
    """OBV falling over last N bars — distribution by smart money."""
    if len(df) < window + 1:
        return False
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    obv = talib.OBV(close, volume)
    if len(obv) < window:
        return False
    return float(obv[-1]) < float(obv[-window])


def _cumulative_vwap(df: pd.DataFrame) -> float:
    # TA-Lib has no VWAP — compute the standard cumulative session VWAP.
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"]
    cum_vol = vol.cumsum()
    if cum_vol.iloc[-1] == 0:
        return float("nan")
    return float((typical * vol).cumsum().iloc[-1] / cum_vol.iloc[-1])


@RuleRegistry.register("vwap_bullish", formula="CLOSE > VWAP")
def vwap_bullish(df: pd.DataFrame) -> bool:
    """
    Price above VWAP — strong intraday bullish signal.
    VWAP is the most important intraday level in NSE/BSE.
    Institutions buy above VWAP, sell below.
    """
    if len(df) < 2:
        return False
    vwap = _cumulative_vwap(df)
    if np.isnan(vwap):
        return False
    return float(df["Close"].iloc[-1]) > vwap


@RuleRegistry.register("vwap_bearish", formula="CLOSE < VWAP")
def vwap_bearish(df: pd.DataFrame) -> bool:
    """Price below VWAP — intraday bearish signal."""
    if len(df) < 2:
        return False
    vwap = _cumulative_vwap(df)
    if np.isnan(vwap):
        return False
    return float(df["Close"].iloc[-1]) < vwap


@RuleRegistry.register("mfi_oversold")
def mfi_oversold(df: pd.DataFrame, window: int = 14, threshold: float = 20.0) -> bool:
    """Money Flow Index below 20 — oversold with volume confirmation."""
    if len(df) < window + 1:
        return False
    mfi = talib.MFI(
        df["High"].to_numpy(dtype=float),
        df["Low"].to_numpy(dtype=float),
        df["Close"].to_numpy(dtype=float),
        df["Volume"].to_numpy(dtype=float),
        timeperiod=window,
    )
    if np.isnan(mfi[-1]):
        return False
    return float(mfi[-1]) < threshold


@RuleRegistry.register("mfi_overbought")
def mfi_overbought(df: pd.DataFrame, window: int = 14, threshold: float = 80.0) -> bool:
    """Money Flow Index above 80 — overbought with volume confirmation."""
    if len(df) < window + 1:
        return False
    mfi = talib.MFI(
        df["High"].to_numpy(dtype=float),
        df["Low"].to_numpy(dtype=float),
        df["Close"].to_numpy(dtype=float),
        df["Volume"].to_numpy(dtype=float),
        timeperiod=window,
    )
    if np.isnan(mfi[-1]):
        return False
    return float(mfi[-1]) > threshold


def _chaikin_money_flow(df: pd.DataFrame, window: int) -> float:
    # TA-Lib exposes ADOSC (Chaikin A/D Oscillator), not CMF. CMF is a separate
    # indicator: sum(money-flow-volume, N) / sum(volume, N). Computed manually.
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    hl_range = (high - low).replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = (mf_multiplier * volume).fillna(0.0)
    vol_sum = volume.rolling(window).sum()
    mfv_sum = mf_volume.rolling(window).sum()
    if vol_sum.iloc[-1] == 0 or pd.isna(vol_sum.iloc[-1]) or pd.isna(mfv_sum.iloc[-1]):
        return float("nan")
    return float(mfv_sum.iloc[-1] / vol_sum.iloc[-1])


@RuleRegistry.register("chaikin_money_flow_positive")
def chaikin_money_flow_positive(df: pd.DataFrame, window: int = 20) -> bool:
    """CMF positive — money flowing into the stock (bullish)."""
    if len(df) < window:
        return False
    cmf = _chaikin_money_flow(df, window)
    if np.isnan(cmf):
        return False
    return cmf > 0


@RuleRegistry.register("chaikin_money_flow_negative")
def chaikin_money_flow_negative(df: pd.DataFrame, window: int = 20) -> bool:
    """CMF negative — money flowing out of the stock (bearish)."""
    if len(df) < window:
        return False
    cmf = _chaikin_money_flow(df, window)
    if np.isnan(cmf):
        return False
    return cmf < 0
