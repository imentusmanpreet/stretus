"""
app/services/universe/metrics.py — pure, vectorized screening/ranking metric math.

Every function here is a PURE function over a candidate's tail OHLCV (no I/O), so the
resolver is trivially testable with synthetic frames and no network (§12.2/§12.4). The
liquidity / volatility / momentum helpers mirror ``scanner.py``'s ``_safe_*`` functions
(Wilder RSI, ATR%, relative volume, 52-week distance) so a screen/rank metric means the
same thing here as in the engine. All return ``NaN`` on incomplete warm-up rather than
raising — the resolver treats ``NaN`` as fail-closed (drop / rank-last).
"""
from __future__ import annotations

import math

import pandas as pd


def safe_relative_volume(df: pd.DataFrame, window: int = 20) -> float:
    """VOL[-1] / AVG(VOL, window)[-1]. NaN when warm-up incomplete."""
    if "volume" not in df.columns or len(df) < window + 1:
        return float("nan")
    avg = df["volume"].iloc[-(window + 1):-1].mean()
    if avg <= 0 or pd.isna(avg):
        return float("nan")
    return float(df["volume"].iloc[-1]) / float(avg)


def safe_rsi(df: pd.DataFrame, window: int = 14) -> float:
    """RSI(window) at the last bar — Wilder smoothing, identical to engine.indicators."""
    if "close" not in df.columns or len(df) < window + 1:
        return float("nan")
    delta = df["close"].astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    last_loss = float(avg_loss.iloc[-1])
    if last_loss == 0 or pd.isna(last_loss):
        return float("nan")
    rs = float(avg_gain.iloc[-1]) / last_loss
    return 100.0 - (100.0 / (1.0 + rs))


def safe_atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    """ATR(window) as a percentage of the last close. NaN when warm-up incomplete."""
    if len(df) < window + 1:
        return float("nan")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean().iloc[-1]
    last_close = float(close.iloc[-1])
    if pd.isna(atr) or last_close <= 0:
        return float("nan")
    return float(atr) / last_close * 100.0


def safe_distance_52w(df: pd.DataFrame, side: str = "high", window: int = 252) -> float:
    """Percent distance from last close to the window extreme (always ≥ 0). NaN if short.

    ``side='high'`` → (max_high - close)/close*100 ("how far below the high");
    ``side='low'``  → (close - min_low)/close*100.
    """
    if len(df) < window:
        return float("nan")
    close = float(df["close"].iloc[-1])
    if close <= 0:
        return float("nan")
    tail = df.tail(window)
    if side == "high":
        return (float(tail["high"].max()) - close) / close * 100.0
    if side == "low":
        return (close - float(tail["low"].min())) / close * 100.0
    return float("nan")


def safe_pct_change(df: pd.DataFrame, window: int = 1) -> float:
    """Percent return over ``window`` bars: (close[-1]/close[-1-window] - 1) * 100."""
    if "close" not in df.columns or len(df) < window + 1:
        return float("nan")
    base = float(df["close"].iloc[-1 - window])
    if base <= 0:
        return float("nan")
    return (float(df["close"].iloc[-1]) / base - 1.0) * 100.0


def safe_momentum(df: pd.DataFrame, window: int = 20) -> float:
    """Price momentum = percent return over ``window`` bars (alias of pct_change)."""
    return safe_pct_change(df, window=window)


def safe_relative_strength(
    df: pd.DataFrame, reference: pd.DataFrame | None, window: int = 20
) -> float:
    """Symbol return minus reference (index) return over ``window`` bars, in percent.

    When no reference frame is supplied the metric degrades to the symbol's own momentum
    (the resolver notes the degradation). Positive ⇒ outperforming the reference.
    """
    sym = safe_pct_change(df, window=window)
    if reference is None or math.isnan(sym):
        return sym
    ref = safe_pct_change(reference, window=window)
    if math.isnan(ref):
        return sym
    return sym - ref


def average_daily_value(df: pd.DataFrame, window: int = 20) -> float:
    """Average daily traded VALUE (close × volume) over ``window`` bars — the ADV used
    for the eligibility liquidity floor. Computed from OHLCV (no vendor feed). NaN if short.
    """
    if not {"close", "volume"}.issubset(df.columns) or len(df) < window:
        return float("nan")
    tail = df.tail(window)
    value = (tail["close"].astype(float) * tail["volume"].astype(float))
    adv = float(value.mean())
    return adv if adv > 0 else float("nan")


# rank.by → (metric function, whether it consumes a reference frame).
def rank_metric(
    df: pd.DataFrame, *, by: str, window: int | None, reference: pd.DataFrame | None = None
) -> float:
    """Compute the ranking metric named by ``by`` for one candidate frame.

    Returns ``NaN`` for metrics whose feed is not wired yet (delivery_volume / oi_change)
    or when warm-up is incomplete; the resolver treats ``NaN`` as rank-last / drop.
    """
    if by == "rvol":
        return safe_relative_volume(df, window=window or 20)
    if by == "volume":
        return float(df["volume"].iloc[-1]) if "volume" in df.columns and len(df) else float("nan")
    if by == "pct_change":
        return safe_pct_change(df, window=window or 1)
    if by == "momentum":
        return safe_momentum(df, window=window or 20)
    if by == "rel_strength":
        return safe_relative_strength(df, reference, window=window or 20)
    if by == "atr_pct":
        return safe_atr_pct(df, window=window or 14)
    if by == "distance_52w":
        return safe_distance_52w(df, side="high", window=window or 252)
    if by == "rsi":
        return safe_rsi(df, window=window or 14)
    # delivery_volume / oi_change — no feed yet (validator notes this).
    return float("nan")
