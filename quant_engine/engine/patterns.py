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
    # Single-bar candlestick patterns
    "IS_BULLISH_ENGULFING": "IS_BULLISH_ENGULFING",
    "IS_BEARISH_ENGULFING": "IS_BEARISH_ENGULFING",
    "IS_HAMMER":            "IS_HAMMER",
    "IS_SHOOTING_STAR":     "IS_SHOOTING_STAR",
}

# Default parameters. A strategy can override via the `patterns:` YAML block.
PATTERN_DEFAULTS = {
    "swing":  {"window": 5},
    "fvg":    {},
    "bos":    {"swing_window": 5},
    "trend":  {"swing_window": 5},
    # Hammer / shooting star: lower (resp. upper) wick must be at least
    # `wick_body_ratio` times the body length, and the opposite wick must be
    # small (≤ `opposite_wick_ratio` × body). Defaults reflect the textbook
    # 2:1 wick-to-body ratio with a small head/tail.
    "hammer":        {"wick_body_ratio": 2.0, "opposite_wick_ratio": 0.5},
    "shooting_star": {"wick_body_ratio": 2.0, "opposite_wick_ratio": 0.5},
    # Engulfing patterns: no tunable parameters.
    "bullish_engulfing": {},
    "bearish_engulfing": {},
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


# ─── Candlestick detectors ────────────────────────────────────────────────────


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """2-bar bullish engulfing: previous candle was bearish (close < open),
    current candle is bullish (close > open) AND the current real body
    completely contains the previous real body — i.e. today's close ≥ prev
    open, and today's open ≤ prev close. Body comparison (NOT wick) per the
    textbook ICT/Nison definition.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    open_ = df["open"]
    close = df["close"]

    prev_bearish = prev_close < prev_open
    today_bullish = close > open_
    engulfs = (open_ <= prev_close) & (close >= prev_open)
    return (prev_bearish & today_bullish & engulfs).fillna(False).astype(bool)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Mirror: prev bullish candle fully engulfed by today's bearish body."""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    open_ = df["open"]
    close = df["close"]

    prev_bullish = prev_close > prev_open
    today_bearish = close < open_
    engulfs = (open_ >= prev_close) & (close <= prev_open)
    return (prev_bullish & today_bearish & engulfs).fillna(False).astype(bool)


def hammer(
    df: pd.DataFrame,
    wick_body_ratio: float = 2.0,
    opposite_wick_ratio: float = 0.5,
) -> pd.Series:
    """Single-bar hammer (bullish reversal candle):
       • Small real body near the top of the bar's range.
       • Lower wick ≥ wick_body_ratio × body length.
       • Upper wick ≤ opposite_wick_ratio × body length.
       • Body length itself must be non-zero (skip pure doji bars).
    The candle direction (bullish vs bearish body) is not enforced — both
    "green hammer" and "hanging man" shapes qualify; trend context decides
    bullish vs bearish meaning.
    """
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bot = pd.concat([open_, close], axis=1).min(axis=1)
    body = (body_top - body_bot).abs()
    upper_wick = high - body_top
    lower_wick = body_bot - low
    # Guard against zero-body doji where the ratio is undefined.
    nonzero_body = body > 0
    has_long_lower = lower_wick >= wick_body_ratio * body
    has_small_upper = upper_wick <= opposite_wick_ratio * body
    return (nonzero_body & has_long_lower & has_small_upper).fillna(False).astype(bool)


def shooting_star(
    df: pd.DataFrame,
    wick_body_ratio: float = 2.0,
    opposite_wick_ratio: float = 0.5,
) -> pd.Series:
    """Single-bar shooting star (bearish reversal): mirror of hammer.
    Long upper wick, small lower wick, small body near the bottom.
    """
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bot = pd.concat([open_, close], axis=1).min(axis=1)
    body = (body_top - body_bot).abs()
    upper_wick = high - body_top
    lower_wick = body_bot - low
    nonzero_body = body > 0
    has_long_upper = upper_wick >= wick_body_ratio * body
    has_small_lower = lower_wick <= opposite_wick_ratio * body
    return (nonzero_body & has_long_upper & has_small_lower).fillna(False).astype(bool)


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

    if "bullish_engulfing" in pattern_config:
        out["IS_BULLISH_ENGULFING"] = bullish_engulfing(out).astype(float)
    if "bearish_engulfing" in pattern_config:
        out["IS_BEARISH_ENGULFING"] = bearish_engulfing(out).astype(float)

    if "hammer" in pattern_config:
        params = {**PATTERN_DEFAULTS["hammer"], **pattern_config["hammer"]}
        out["IS_HAMMER"] = hammer(
            out,
            wick_body_ratio=float(params.get("wick_body_ratio", 2.0)),
            opposite_wick_ratio=float(params.get("opposite_wick_ratio", 0.5)),
        ).astype(float)
    if "shooting_star" in pattern_config:
        params = {**PATTERN_DEFAULTS["shooting_star"], **pattern_config["shooting_star"]}
        out["IS_SHOOTING_STAR"] = shooting_star(
            out,
            wick_body_ratio=float(params.get("wick_body_ratio", 2.0)),
            opposite_wick_ratio=float(params.get("opposite_wick_ratio", 0.5)),
        ).astype(float)

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
    "IS_BULLISH_ENGULFING": "bullish_engulfing",
    "IS_BEARISH_ENGULFING": "bearish_engulfing",
    "IS_HAMMER":            "hammer",
    "IS_SHOOTING_STAR":     "shooting_star",
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
