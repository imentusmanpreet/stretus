"""
engine/patterns.py
──────────────────
Phase 6 — structural pattern detectors.

Each detector takes a DataFrame (with lowercase OHLCV columns) and returns a
pandas Series aligned to the same index, where True means "this pattern is
confirmed at this bar". The runner adds these as boolean/0-1 columns
(`IS_SWING_HIGH`, `IS_BULLISH_FVG`, etc.) to the main df before condition
evaluation, so AST formulas can reference them as plain identifiers.

NO LOOK-AHEAD INVARIANT
─────────────────────────────────────────────────────────────────────
Every detector's value at bar i depends only on bars 0..i. For "centered"
patterns like swing highs (which need bars on either side of the pivot),
the True label lands at the bar where the swing is *confirmed* — i.e. shifted
forward by `window` bars from the actual pivot. This matches real-world ICT
trading: you don't know a swing high formed until `window` bars have passed
without exceeding it. The signal fires when you would actually act on it.
─────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Pattern column names (single source of truth) ───────────────────────────

# Maps the AST identifier → DataFrame column name. Same in v1 (no rename).
PATTERN_COLUMNS = {
    "IS_SWING_HIGH":     "IS_SWING_HIGH",
    "IS_SWING_LOW":      "IS_SWING_LOW",
    "IS_HIGHER_HIGH":    "IS_HIGHER_HIGH",
    "IS_LOWER_LOW":      "IS_LOWER_LOW",
    "IS_BULLISH_FVG":    "IS_BULLISH_FVG",
    "IS_BEARISH_FVG":    "IS_BEARISH_FVG",
    "IS_BOS_BULLISH":    "IS_BOS_BULLISH",
    "IS_BOS_BEARISH":    "IS_BOS_BEARISH",
    # Phase 4 — candlestick pattern columns (rule #22).
    "IS_HAMMER":             "IS_HAMMER",
    "IS_HANGING_MAN":        "IS_HANGING_MAN",
    "IS_ENGULFING":          "IS_ENGULFING",
    "IS_BULLISH_ENGULFING":  "IS_BULLISH_ENGULFING",
    "IS_BEARISH_ENGULFING":  "IS_BEARISH_ENGULFING",
    "IS_PIN_BAR":            "IS_PIN_BAR",
    "IS_DOJI":               "IS_DOJI",
    "IS_MORNING_STAR":       "IS_MORNING_STAR",
    "IS_EVENING_STAR":       "IS_EVENING_STAR",
}

# Default parameters. A strategy can override via the `patterns:` YAML block.
PATTERN_DEFAULTS = {
    "swing":  {"window": 5},
    "fvg":    {},
    "bos":    {"swing_window": 5},
    "trend":  {"swing_window": 5},
}


# ─── Detectors ────────────────────────────────────────────────────────────────


def swing_high(high: pd.Series, window: int = 5) -> pd.Series:
    """At bar i, True iff bar (i - window) was a local high that was not
    exceeded for `window` bars on either side. Confirmation lags by `window`
    bars — strict no-look-ahead.

    Example with window=2: a high at bar 10 is confirmed at bar 12 (i.e.
    IS_SWING_HIGH[12] = True), provided no bar in [8, 12] had a higher high.
    """
    if window <= 0:
        return pd.Series(False, index=high.index)
    centered_max = high.rolling(window=2 * window + 1, center=True).max()
    # Cast to bool first to avoid pandas FutureWarning about downcasting
    # object dtype on fillna.
    is_local_high = (high == centered_max).astype(bool)
    return is_local_high.shift(window, fill_value=False).astype(bool)


def swing_low(low: pd.Series, window: int = 5) -> pd.Series:
    """Mirror of swing_high — confirmed local minimum, lagged by `window`."""
    if window <= 0:
        return pd.Series(False, index=low.index)
    centered_min = low.rolling(window=2 * window + 1, center=True).min()
    is_local_low = (low == centered_min).astype(bool)
    return is_local_low.shift(window, fill_value=False).astype(bool)


def bullish_fvg(df: pd.DataFrame) -> pd.Series:
    """3-bar bullish fair value gap: bar i's LOW > bar (i-2)'s HIGH.

    The middle bar is an impulsive bullish candle whose range left a gap
    between the prior bar's high and the next bar's low — institutional
    aggression that often gets revisited.
    """
    return (df["low"] > df["high"].shift(2)).fillna(False).astype(bool)


def bearish_fvg(df: pd.DataFrame) -> pd.Series:
    """Mirror of bullish_fvg: bar i's HIGH < bar (i-2)'s LOW."""
    return (df["high"] < df["low"].shift(2)).fillna(False).astype(bool)


def break_of_structure_bullish(
    df: pd.DataFrame,
    swing_high_col: pd.Series,
) -> pd.Series:
    """At bar i, True iff close[i] strictly exceeds the most recent confirmed
    swing high level prior to bar i. The "structure break" is the moment
    bullish intent is proven — same definition ICT traders use.
    """
    swing_levels = df["high"].where(swing_high_col).ffill()
    # shift(1) ensures we compare against the level KNOWN at bar i (set at
    # bar i-1 or earlier); using the unshifted series would let bar i compare
    # against itself, which is meaningless.
    return (df["close"] > swing_levels.shift(1)).fillna(False).astype(bool)


def break_of_structure_bearish(
    df: pd.DataFrame,
    swing_low_col: pd.Series,
) -> pd.Series:
    """Mirror of break_of_structure_bullish — close beneath last swing low."""
    swing_levels = df["low"].where(swing_low_col).ffill()
    return (df["close"] < swing_levels.shift(1)).fillna(False).astype(bool)


def higher_high(
    swing_high_col: pd.Series,
    high: pd.Series,
    window: int = 5,
) -> pd.Series:
    """At each confirmed swing-high bar, True iff that swing's actual peak
    high (the high `window` bars BEFORE confirmation, i.e. the pivot bar) is
    higher than the previous swing's peak. The True lands at the
    confirmation bar so the signal fires when traders would actually act.
    """
    if not swing_high_col.any():
        return pd.Series(False, index=high.index)
    # Look up the high at the pivot bar (window bars before confirmation),
    # not at the confirmation bar itself.
    pivot_highs = high.shift(window).where(swing_high_col).dropna()
    is_hh = (pivot_highs > pivot_highs.shift(1)).fillna(False)
    out = pd.Series(False, index=high.index)
    out.loc[is_hh.index] = is_hh.values
    return out


def lower_low(
    swing_low_col: pd.Series,
    low: pd.Series,
    window: int = 5,
) -> pd.Series:
    """Mirror of higher_high — current swing's actual pivot low is lower
    than the previous swing's pivot low."""
    if not swing_low_col.any():
        return pd.Series(False, index=low.index)
    pivot_lows = low.shift(window).where(swing_low_col).dropna()
    is_ll = (pivot_lows < pivot_lows.shift(1)).fillna(False)
    out = pd.Series(False, index=low.index)
    out.loc[is_ll.index] = is_ll.values
    return out


# ─── Candlestick patterns (Phase 4 — rule #22) ───────────────────────────────


def _body_and_shadow(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (body_size, range_size, upper_shadow, lower_shadow) series."""
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    body = (c - o).abs()
    rng  = (h - l).clip(lower=1e-9)
    upper_shadow = h - c.where(c >= o, o)
    lower_shadow = c.where(c <= o, o) - l
    return body, rng, upper_shadow, lower_shadow


def hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish hammer: small body near the top, long lower wick (≥ 2× body),
    short upper wick (≤ 0.5× body). Detected on any bar, regardless of
    preceding trend (callers can layer trend context separately)."""
    body, rng, upper, lower = _body_and_shadow(df)
    body_ratio = body / rng
    return (
        (lower >= 2 * body)
        & (upper <= 0.5 * body.clip(lower=1e-9))
        & (body_ratio <= 0.35)
    ).astype(bool)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """Hanging man: same candle shape as hammer but the prior trend should
    have been up. We return the candle-shape match; downstream conditions
    can require an uptrend confirmation."""
    return hammer(df)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bar (i): bullish (close > open) AND prev bar (i-1) bearish AND
    bar(i).body fully engulfs bar(i-1).body (open[i] <= close[i-1] AND
    close[i] >= open[i-1])."""
    o, c = df["open"].astype(float), df["close"].astype(float)
    prev_o, prev_c = o.shift(1), c.shift(1)
    return (
        (c > o)
        & (prev_c < prev_o)
        & (o <= prev_c)
        & (c >= prev_o)
    ).astype(bool)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"].astype(float), df["close"].astype(float)
    prev_o, prev_c = o.shift(1), c.shift(1)
    return (
        (c < o)
        & (prev_c > prev_o)
        & (o >= prev_c)
        & (c <= prev_o)
    ).astype(bool)


def pin_bar(df: pd.DataFrame) -> pd.Series:
    """Either side. Body ≤ 30% of range AND one shadow ≥ 2× the body AND ≥
    60% of range."""
    body, rng, upper, lower = _body_and_shadow(df)
    body_ratio = body / rng
    long_lower = (lower >= 2 * body.clip(lower=1e-9)) & (lower / rng >= 0.6)
    long_upper = (upper >= 2 * body.clip(lower=1e-9)) & (upper / rng >= 0.6)
    return ((body_ratio <= 0.3) & (long_lower | long_upper)).astype(bool)


def doji(df: pd.DataFrame, body_pct_max: float = 0.10) -> pd.Series:
    """Body ≤ `body_pct_max` of the bar range."""
    body, rng, _, _ = _body_and_shadow(df)
    return ((body / rng) <= body_pct_max).astype(bool)


def morning_star(df: pd.DataFrame) -> pd.Series:
    """3-bar pattern: bearish, small body, bullish closing above midpoint of
    first body."""
    o, c = df["open"].astype(float), df["close"].astype(float)
    body = (c - o).abs()
    rng  = (df["high"] - df["low"]).clip(lower=1e-9)
    body_pct = body / rng

    b1_bearish = (c.shift(2) < o.shift(2))
    b2_small   = (body_pct.shift(1) <= 0.30)
    b3_bullish = (c > o)
    b3_recovers = c > (o.shift(2) + c.shift(2)) / 2.0
    return (b1_bearish & b2_small & b3_bullish & b3_recovers).astype(bool)


def evening_star(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"].astype(float), df["close"].astype(float)
    body = (c - o).abs()
    rng  = (df["high"] - df["low"]).clip(lower=1e-9)
    body_pct = body / rng

    b1_bullish = (c.shift(2) > o.shift(2))
    b2_small   = (body_pct.shift(1) <= 0.30)
    b3_bearish = (c < o)
    b3_falls   = c < (o.shift(2) + c.shift(2)) / 2.0
    return (b1_bullish & b2_small & b3_bearish & b3_falls).astype(bool)


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def add_all_patterns(
    df: pd.DataFrame,
    pattern_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add the requested boolean pattern columns to df.

    `pattern_config` is a {pattern_name: params_dict} mapping. Pattern names
    correspond to the IS_* identifiers above (lowercased and without IS_):

        {"swing_high": {"window": 5},
         "bullish_fvg": {},
         "bos_bullish": {"swing_window": 5}}

    A pattern that was already added by a previous call is skipped. Detectors
    that depend on others (BOS needs swing levels; HH/LL need swing markers)
    are scheduled in dependency order so callers don't need to specify the
    transitive set explicitly.
    """
    if not pattern_config:
        return df
    out = df.copy()

    # Resolve dependencies: BOS, HH, LL all need swing markers. We compute
    # those once and reuse.
    needs_swing_high = any(
        p in pattern_config for p in ("swing_high", "bos_bullish", "higher_high")
    )
    needs_swing_low = any(
        p in pattern_config for p in ("swing_low", "bos_bearish", "lower_low")
    )

    # Window for swing detection: prefer the explicit swing_* config if given,
    # otherwise fall back to the BOS/trend swing_window param, otherwise default.
    swing_window_high = int(
        pattern_config.get("swing_high", {}).get("window")
        or pattern_config.get("bos_bullish", {}).get("swing_window")
        or pattern_config.get("higher_high", {}).get("swing_window")
        or PATTERN_DEFAULTS["swing"]["window"]
    )
    swing_window_low = int(
        pattern_config.get("swing_low", {}).get("window")
        or pattern_config.get("bos_bearish", {}).get("swing_window")
        or pattern_config.get("lower_low", {}).get("swing_window")
        or PATTERN_DEFAULTS["swing"]["window"]
    )

    swing_high_col: pd.Series | None = None
    swing_low_col: pd.Series | None = None
    if needs_swing_high:
        swing_high_col = swing_high(out["high"], window=swing_window_high)
        out["IS_SWING_HIGH"] = swing_high_col.astype(float)
    if needs_swing_low:
        swing_low_col = swing_low(out["low"], window=swing_window_low)
        out["IS_SWING_LOW"] = swing_low_col.astype(float)

    if "bullish_fvg" in pattern_config:
        out["IS_BULLISH_FVG"] = bullish_fvg(out).astype(float)
    if "bearish_fvg" in pattern_config:
        out["IS_BEARISH_FVG"] = bearish_fvg(out).astype(float)

    if "bos_bullish" in pattern_config:
        assert swing_high_col is not None     # dependency was scheduled above
        out["IS_BOS_BULLISH"] = break_of_structure_bullish(out, swing_high_col).astype(float)
    if "bos_bearish" in pattern_config:
        assert swing_low_col is not None
        out["IS_BOS_BEARISH"] = break_of_structure_bearish(out, swing_low_col).astype(float)

    if "higher_high" in pattern_config:
        assert swing_high_col is not None
        out["IS_HIGHER_HIGH"] = higher_high(
            swing_high_col, out["high"], window=swing_window_high,
        ).astype(float)
    if "lower_low" in pattern_config:
        assert swing_low_col is not None
        out["IS_LOWER_LOW"] = lower_low(
            swing_low_col, out["low"], window=swing_window_low,
        ).astype(float)

    # Phase 4 — candlestick patterns. Each is a pure single-bar (or 2-3 bar)
    # detector with no dependency on swing markers.
    if "hammer" in pattern_config:
        out["IS_HAMMER"] = hammer(out).astype(float)
    if "hanging_man" in pattern_config:
        out["IS_HANGING_MAN"] = hanging_man(out).astype(float)
    if "engulfing" in pattern_config:
        bull = bullish_engulfing(out).astype(float)
        bear = bearish_engulfing(out).astype(float)
        out["IS_BULLISH_ENGULFING"] = bull
        out["IS_BEARISH_ENGULFING"] = bear
        out["IS_ENGULFING"]         = ((bull > 0) | (bear > 0)).astype(float)
    if "bullish_engulfing" in pattern_config:
        out["IS_BULLISH_ENGULFING"] = bullish_engulfing(out).astype(float)
    if "bearish_engulfing" in pattern_config:
        out["IS_BEARISH_ENGULFING"] = bearish_engulfing(out).astype(float)
    if "pin_bar" in pattern_config:
        out["IS_PIN_BAR"] = pin_bar(out).astype(float)
    if "doji" in pattern_config:
        params = pattern_config.get("doji") or {}
        body_max = float(params.get("body_pct_max", 0.10))
        out["IS_DOJI"] = doji(out, body_pct_max=body_max).astype(float)
    if "morning_star" in pattern_config:
        out["IS_MORNING_STAR"] = morning_star(out).astype(float)
    if "evening_star" in pattern_config:
        out["IS_EVENING_STAR"] = evening_star(out).astype(float)

    return out


# ─── AST integration helpers ──────────────────────────────────────────────────


# Mapping AST identifier → top-level pattern name (used by the runner to know
# which detector to compute when a condition references IS_BOS_BULLISH etc.).
IDENT_TO_PATTERN_NAME = {
    "IS_SWING_HIGH":     "swing_high",
    "IS_SWING_LOW":      "swing_low",
    "IS_HIGHER_HIGH":    "higher_high",
    "IS_LOWER_LOW":      "lower_low",
    "IS_BULLISH_FVG":    "bullish_fvg",
    "IS_BEARISH_FVG":    "bearish_fvg",
    "IS_BOS_BULLISH":    "bos_bullish",
    "IS_BOS_BEARISH":    "bos_bearish",
    # Phase 4 — candlestick patterns (rule #22).
    "IS_HAMMER":              "hammer",
    "IS_HANGING_MAN":         "hanging_man",
    "IS_ENGULFING":           "engulfing",
    "IS_BULLISH_ENGULFING":   "bullish_engulfing",
    "IS_BEARISH_ENGULFING":   "bearish_engulfing",
    "IS_PIN_BAR":             "pin_bar",
    "IS_DOJI":                "doji",
    "IS_MORNING_STAR":        "morning_star",
    "IS_EVENING_STAR":        "evening_star",
}


def patterns_required_by_identifiers(idents: set[str]) -> dict[str, dict]:
    """Convert a set of AST identifiers (e.g. {'IS_BOS_BULLISH'}) into a
    pattern_config dict suitable for add_all_patterns(). Defaults applied."""
    config: dict[str, dict] = {}
    for ident in idents:
        name = IDENT_TO_PATTERN_NAME.get(ident)
        if name is None:
            continue
        config[name] = {}    # defaults; runner can merge YAML overrides
    return config


def merge_pattern_configs(*configs: dict[str, Any] | None) -> dict[str, Any]:
    """Merge multiple {pattern: params} dicts. Later wins on conflicting params."""
    merged: dict[str, dict] = {}
    for cfg in configs:
        if not cfg:
            continue
        for name, params in cfg.items():
            merged.setdefault(name, {})
            if isinstance(params, dict):
                merged[name].update(params)
    return merged
