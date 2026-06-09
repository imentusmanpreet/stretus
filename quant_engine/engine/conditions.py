"""
Safe condition parser and evaluator for strategy formulas.

Supported syntax:
  - Comparison:   CLOSE > SMA(20),  RSI(14) < 30
  - Boolean:      ... AND ...,  ... OR ...,  NOT ...
  - Arithmetic:   CLOSE * 1.02,  (HIGH - LOW) / CLOSE,  SMA(20) + 10
  - Functions:    SMA(n), EMA(n), RSI(n), BB_UPPER(n), BB_LOWER(n), BB_MID(n)
                  ATR(n), ADX(n), PLUS_DI(n), MINUS_DI(n)   ← regime / volatility filters
                  AVG(FIELD or INDICATOR(n), n), MAX(...), MIN(...)
                  STDEV(FIELD, n), ZSCORE(FIELD, n)
                  PREV(FIELD or EXPR, offset)   ← e.g. PREV(EMA(20), 3) for slope checks
                  OPENING_RANGE_HIGH(n), OPENING_RANGE_LOW(n)

                  AVG/MAX/MIN accept a nested periodic indicator as their first
                  argument, e.g. AVG(ATR(14), 50) — the rolling stat of ATR(14)
                  over the last 50 bars (used by the ATR-volatility regime cards).
  - Identifiers:  CLOSE, OPEN, HIGH, LOW, VOLUME, VWAP, MACD, MACD_SIGNAL
                  REF_CLOSE, REF_OPEN, REF_HIGH, REF_LOW, REF_VOLUME  ← Phase 4: reference symbol
                  IS_SWING_HIGH, IS_SWING_LOW, IS_BOS_BULLISH, IS_BOS_BEARISH,
                  IS_BULLISH_FVG, IS_BEARISH_FVG, IS_HIGHER_HIGH, IS_LOWER_LOW
                                                                       ← Phase 6: structural patterns
  - Functions:    ... RS(n)                                            ← Phase 4: relative strength vs reference
  - Variables:    PROFIT, LOSS, TAKE_PROFIT_TARGET, STOP_LOSS_TARGET

Parse hierarchy (highest to lowest precedence):
  primary → unary → multiplicative → additive → comparison → not → and → or
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

import engine.indicators as _ind
from engine.indicators import (
    PERIODIC_INDICATOR_NAMES, candlestick_pattern, compute_indicator_column,
)

logger = logging.getLogger(__name__)

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    (?P<NUMBER>\d+(?:\.\d+)?)
    |(?P<OP><=|>=|==|!=|<|>)
    |(?P<LPAREN>\()
    |(?P<RPAREN>\))
    |(?P<COMMA>,)
    |(?P<PLUS>\+)
    |(?P<MINUS>-)
    |(?P<MUL>\*)
    |(?P<DIV>/)
    |(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<WS>\s+)
    |(?P<MISMATCH>.)
    """,
    re.VERBOSE,
)

# Track conditions that already triggered a warning so we don't spam the logs.
# One warning per unique (condition text, error type) pair — then silenced.
_warned_conditions: set[str] = set()


# ── AST node types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Token:
    kind: str
    value: str


@dataclass(frozen=True)
class NumberNode:
    value: float


@dataclass(frozen=True)
class IdentifierNode:
    name: str


@dataclass(frozen=True)
class UnaryNode:
    op: str
    operand: Any


@dataclass(frozen=True)
class BinaryNode:
    """Arithmetic binary operation: +, -, *, /"""
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class FunctionNode:
    name: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class ComparisonNode:
    left: Any
    op: str
    right: Any


@dataclass(frozen=True)
class BooleanNode:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class NotNode:
    operand: Any


class ParseError(ValueError):
    pass


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(expression: str) -> list[Token]:
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(expression):
        kind = match.lastgroup or "MISMATCH"
        value = match.group(0)
        if kind == "WS":
            continue
        if kind == "MISMATCH":
            raise ParseError(f"Unexpected character {value!r} in expression.")
        tokens.append(Token(kind=kind, value=value))
    return tokens


# ── Parser ────────────────────────────────────────────────────────────────────

class _Parser:
    """
    Recursive-descent parser.

    Precedence (low → high):
      or → and → not → comparison → additive(+/-) → multiplicative(*/÷) → unary(-) → primary
    """

    def __init__(self, tokens: Iterable[Token]):
        self.tokens = list(tokens)
        self.pos = 0

    def parse(self) -> Any:
        if not self.tokens:
            raise ParseError("Condition is empty.")
        node = self._parse_or()
        if self._peek() is not None:
            raise ParseError(
                f"Unexpected token {self._peek().value!r} at the end of the expression."
            )
        return node

    def _peek(self) -> Token | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self._peek()
        if token is None:
            raise ParseError("Unexpected end of expression.")
        self.pos += 1
        return token

    def _match(self, *values: str, kind: str | None = None) -> Token | None:
        token = self._peek()
        if token is None:
            return None
        if kind is not None and token.kind != kind:
            return None
        if values and token.value.upper() not in values:
            return None
        self.pos += 1
        return token

    def _expect(self, value: str | None = None, kind: str | None = None) -> Token:
        token = self._advance()
        if value is not None and token.value.upper() != value:
            raise ParseError(f"Expected {value!r}, got {token.value!r}.")
        if kind is not None and token.kind != kind:
            raise ParseError(f"Expected token kind {kind!r}, got {token.kind!r}.")
        return token

    # ── Boolean layer ─────────────────────────────────────────────────────────

    def _parse_or(self) -> Any:
        node = self._parse_and()
        while self._match("OR", kind="IDENT"):
            node = BooleanNode(op="OR", left=node, right=self._parse_and())
        return node

    def _parse_and(self) -> Any:
        node = self._parse_not()
        while self._match("AND", kind="IDENT"):
            node = BooleanNode(op="AND", left=node, right=self._parse_not())
        return node

    def _parse_not(self) -> Any:
        if self._match("NOT", kind="IDENT"):
            return NotNode(operand=self._parse_not())
        return self._parse_comparison()

    # ── Comparison layer ──────────────────────────────────────────────────────

    def _parse_comparison(self) -> Any:
        node = self._parse_additive()
        token = self._peek()
        if token is not None and token.kind == "OP":
            op = self._advance().value
            return ComparisonNode(left=node, op=op, right=self._parse_additive())
        return node

    # ── Arithmetic layer ──────────────────────────────────────────────────────

    def _parse_additive(self) -> Any:
        """Handles binary + and - (lower precedence than * and /)."""
        node = self._parse_multiplicative()
        while True:
            if self._match(kind="PLUS"):
                node = BinaryNode(op="+", left=node, right=self._parse_multiplicative())
            elif self._match(kind="MINUS"):
                node = BinaryNode(op="-", left=node, right=self._parse_multiplicative())
            else:
                break
        return node

    def _parse_multiplicative(self) -> Any:
        """Handles binary * and / (higher precedence than + and -)."""
        node = self._parse_unary()
        while True:
            if self._match(kind="MUL"):
                node = BinaryNode(op="*", left=node, right=self._parse_unary())
            elif self._match(kind="DIV"):
                node = BinaryNode(op="/", left=node, right=self._parse_unary())
            else:
                break
        return node

    def _parse_unary(self) -> Any:
        """Handles unary + and - (e.g. -STOP_LOSS_TARGET)."""
        if self._match(kind="PLUS"):
            return self._parse_unary()
        if self._match(kind="MINUS"):
            return UnaryNode(op="-", operand=self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Any:
        """Handles numbers, identifiers, function calls, and parenthesised groups."""
        token = self._peek()
        if token is None:
            raise ParseError("Unexpected end of expression.")

        if token.kind == "NUMBER":
            self._advance()
            return NumberNode(value=float(token.value))

        if token.kind == "IDENT":
            ident = self._advance().value.upper()
            if self._match(kind="LPAREN"):
                args: list[Any] = []
                if not self._match(kind="RPAREN"):
                    while True:
                        args.append(self._parse_or())
                        if self._match(kind="COMMA"):
                            continue
                        self._expect(kind="RPAREN")
                        break
                return FunctionNode(name=ident, args=tuple(args))
            return IdentifierNode(name=ident)

        if self._match(kind="LPAREN"):
            node = self._parse_or()
            self._expect(kind="RPAREN")
            return node

        raise ParseError(f"Unexpected token {token.value!r}.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_name(field: str) -> str:
    mapping = {
        "CLOSE":       "close",
        "OPEN":        "open",
        "HIGH":        "high",
        "LOW":         "low",
        "VOL":         "volume",
        "VOLUME":      "volume",
        "VWAP":        "VWAP",
        "MACD":        "MACD",
        "MACD_SIGNAL": "MACD_SIGNAL",
        "MACD_HIST":   "MACD_HIST",
        # Phase 4 — reference symbol columns. The REF_ prefix is preserved so
        # runner.merge_reference_data() can place columns under predictable
        # names (REF_close, REF_open, …).
        "REF_CLOSE":   "REF_close",
        "REF_OPEN":    "REF_open",
        "REF_HIGH":    "REF_high",
        "REF_LOW":     "REF_low",
        "REF_VOL":     "REF_volume",
        "REF_VOLUME":  "REF_volume",
    }
    return mapping.get(field.upper(), field.lower())


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _truthy(value: Any) -> bool:
    if _is_nan(value):
        return False
    return bool(value)


# ── Single-source precompute cache ─────────────────────────────────────────────
#
# conditions.py NEVER recomputes an indicator value bar-by-bar. The first time a
# value is needed, the FULL vectorised Series is computed once — via the same
# engine.indicators code add_all_indicators() uses — and cached as a column on
# df. Every later bar is then a pure column read. This keeps the slow (test /
# registry) path and the fast (simulator) path on IDENTICAL math, so a backtest
# can never diverge from live evaluation.
#
# In the simulator hot loop the runner has already precomputed every referenced
# column up front (runner._merge_indicator_requirements), so these builders only
# ever fire on the direct evaluate_condition() path (tests, ad-hoc calls).

_CACHE_PREFIX = "__qe_"


def _cache_column(df: pd.DataFrame, col: str, builder) -> pd.Series | None:
    """Ensure df[col] exists, computing it ONCE via builder() (returns a Series
    or None). Returns the cached Series (or None). Caches by mutating df so
    repeated per-bar calls in build_entry_signals reuse the first computation."""
    if col in df.columns:
        return df[col]
    series = builder()
    if series is None:
        return None
    try:
        df[col] = np.asarray(series, dtype=float)
        return df[col]
    except Exception:
        return series


def _value_at(series: pd.Series | None, i: int) -> float:
    if series is None or i < 0 or i >= len(series):
        return float("nan")
    value = series.iloc[i]
    return float(value) if not pd.isna(value) else float("nan")


def _ensure_field_rolling(df: pd.DataFrame, field: str, n: int, agg: str) -> pd.Series | None:
    """Vectorised rolling aggregate over a raw field (CLOSE, VOLUME, …), cached.
    Replaces the per-bar _rolling_mean/_rolling_ema/_rolling_max/_rolling_min/
    _rolling_stdev, which walked bars 0..i on every call."""
    col = _col_name(field)
    if n <= 0 or col not in df.columns:
        return None
    cache_key = f"{_CACHE_PREFIX}{agg}_{col}_{int(n)}"
    return _cache_column(
        df, cache_key, lambda: _ind.rolling_agg(df[col].astype(float), int(n), agg)
    )


def _ensure_zscore(df: pd.DataFrame, field: str, n: int) -> pd.Series | None:
    """Vectorised rolling z-score of `field`, cached. NaN where rolling stdev is
    zero (flat series — no meaningful z-score)."""
    col = _col_name(field)
    if n <= 0 or col not in df.columns:
        return None
    cache_key = f"{_CACHE_PREFIX}zscore_{col}_{int(n)}"

    def build():
        s = df[col].astype(float)
        mean = _ind.rolling_agg(s, int(n), "mean")
        std = _ind.rolling_agg(s, int(n), "stdev").replace(0.0, np.nan)
        return (s - mean) / std

    return _cache_column(df, cache_key, build)


def _session_opening_range(df: pd.DataFrame, field: str, n: int, i: int, agg: str) -> float:
    if n <= 0 or i < 0 or i >= len(df) or not isinstance(df.index, pd.DatetimeIndex):
        return float("nan")

    col = _col_name(field)
    if col not in df.columns:
        return float("nan")

    normalized_days = df.index.normalize()
    current_day = normalized_days[i]
    session_positions = np.flatnonzero(normalized_days == current_day)
    current_matches = np.flatnonzero(session_positions == i)
    if len(current_matches) == 0:
        return float("nan")

    current_session_index = int(current_matches[0])
    if current_session_index < n:
        return float("nan")

    opening_values = df.iloc[session_positions[:n]][col].dropna().astype(float)
    if len(opening_values) < n:
        return float("nan")

    value = opening_values.max() if agg == "max" else opening_values.min()
    return float(value)


def _ensure_indicator(df: pd.DataFrame, name: str, period: int) -> pd.Series | None:
    """Full Series for a periodic indicator column NAME_period — read from df if
    already precomputed, otherwise computed ONCE via the indicators registry and
    cached. The single source of truth shared with add_all_indicators()."""
    col = f"{name}_{int(period)}"
    return _cache_column(df, col, lambda: compute_indicator_column(df, name, int(period)))


def _resolve_indicator(df: pd.DataFrame, name: str, period: int, i: int) -> float:
    """Value of a periodic indicator at bar i — pure column read after a
    one-time precompute. No per-bar recompute."""
    return _value_at(_ensure_indicator(df, name, period), i)


def _periodic_column_for(node: Any) -> str | None:
    """If `node` is a single-period indicator call like ATR(14), return its
    precomputed column name 'ATR_14'. Otherwise None."""
    if (
        isinstance(node, FunctionNode)
        and node.name in _PERIODIC_INDICATORS
        and len(node.args) == 1
        and isinstance(node.args[0], NumberNode)
    ):
        return f"{node.name}_{int(node.args[0].value)}"
    return None


def _arg_series(df: pd.DataFrame, arg: Any) -> tuple[pd.Series | None, str | None]:
    """Resolve the first argument of AVG/MAX/MIN/STDEV to (Series, cache_label).

    Accepts either a raw field identifier (CLOSE, VOLUME, …) or a nested periodic
    indicator (ATR(14)) — precomputing the indicator column once if absent.
    Returns (None, None) when it's neither — the caller turns that into a
    ParseError."""
    if isinstance(arg, IdentifierNode):
        col = _col_name(arg.name)
        if col in df.columns:
            return df[col].astype(float), col
        return None, None
    col = _periodic_column_for(arg)
    if col is not None:
        name, _, period = col.rpartition("_")
        series = _ensure_indicator(df, name, int(period))
        if series is not None:
            return series.astype(float), col
    return None, None


def _roll_over_arg(df: pd.DataFrame, arg: Any, n: int, i: int, agg: str) -> float:
    """Vectorised rolling aggregate over a field or nested-indicator arg, cached.
    NaN until n non-NaN values are available."""
    series, label = _arg_series(df, arg)
    if series is None or n <= 0:
        return float("nan")
    cache_key = f"{_CACHE_PREFIX}{agg}_{label}_{int(n)}"
    return _value_at(
        _cache_column(df, cache_key, lambda: _ind.rolling_agg(series, int(n), agg)), i
    )


def _resolve_identifier(name: str, df: pd.DataFrame, i: int, variables: dict[str, float] | None) -> Any:
    row = df.iloc[i]

    if variables and name in variables:
        return variables[name]

    if name in {"TRUE", "FALSE"}:
        return name == "TRUE"

    if name in {"CLOSE", "OPEN", "HIGH", "LOW", "VOL", "VOLUME"}:
        col = _col_name(name)
        value = row.get(col, float("nan"))
        return float(value) if not pd.isna(value) else float("nan")

    # Scalar / fixed-param studies (VWAP, MACD*, OBV, SAR, STOCH_*, CDL_*, …).
    # Read the precomputed column if present (runner fast path), else compute the
    # full Series once and cache it — never per bar.
    if name in _SCALAR_IDENTIFIERS_ALL or name.startswith("CDL_"):
        value = row.get(name, float("nan"))
        if not pd.isna(value):
            return float(value)
        return _resolve_scalar(df, name, i)

    # Phase 4 — REF_* identifiers resolve to the reference symbol's OHLCV
    # columns. NaN when the runner wasn't given reference data; comparisons
    # against NaN return False so reference-dependent conditions stay safely
    # silent instead of crashing.
    if name in {"REF_CLOSE", "REF_OPEN", "REF_HIGH", "REF_LOW", "REF_VOL", "REF_VOLUME"}:
        col = _col_name(name)
        value = row.get(col, float("nan"))
        return float(value) if not pd.isna(value) else float("nan")

    # Phase 6 — structural-pattern identifiers (IS_SWING_HIGH, IS_BOS_BULLISH,
    # IS_BULLISH_FVG, ...). The runner pre-computes these as 0/1 columns on
    # the df. If the column is missing, return NaN ⇒ comparison False ⇒
    # condition silently stays inactive instead of crashing.
    if name in _PATTERN_IDENTIFIERS:
        value = row.get(name, float("nan"))
        return float(value) if not pd.isna(value) else float("nan")

    return float("nan")


# Period-less / fixed-param scalar studies referenceable as identifiers in a
# formula (e.g. CLOSE > VWAP, MACD > 0, CLOSE > SAR, STOCH_K < 20, OBV > 0).
# Each builder returns the full Series; computed once and cached. CDL_* pattern
# identifiers are handled by prefix (any TA-Lib candlestick alias).
_SCALAR_BUILDERS = {
    "VWAP":            lambda df: _ind.vwap(df),
    "MACD":            lambda df: _ind.macd_all(df["close"])[0],
    "MACD_SIGNAL":     lambda df: _ind.macd_all(df["close"])[1],
    "MACD_HIST":       lambda df: _ind.macd_all(df["close"])[2],
    "OBV":             lambda df: _ind.obv(df),
    "AD":              lambda df: _ind.ad(df),
    "ADOSC":           lambda df: _ind.adosc(df),
    "SAR":             lambda df: _ind.sar(df),
    "PPO":             lambda df: _ind.ppo(df["close"]),
    "APO":             lambda df: _ind.apo(df["close"]),
    "ULTOSC":          lambda df: _ind.ultosc(df),
    "TRANGE":          lambda df: _ind.trange(df),
    "STOCH_K":         lambda df: _ind.stoch(df)[0],
    "STOCH_D":         lambda df: _ind.stoch(df)[1],
    "STOCHF_K":        lambda df: _ind.stochf(df)[0],
    "STOCHF_D":        lambda df: _ind.stochf(df)[1],
    "STOCHRSI_K":      lambda df: _ind.stochrsi(df["close"])[0],
    "STOCHRSI_D":      lambda df: _ind.stochrsi(df["close"])[1],
    "SUPERTREND":      lambda df: _ind.supertrend(df)[0],
    "SUPERTREND_LINE": lambda df: _ind.supertrend(df)[1],
}

_SCALAR_IDENTIFIERS_ALL = frozenset(_SCALAR_BUILDERS)


def _ensure_scalar(df: pd.DataFrame, name: str) -> pd.Series | None:
    """Precompute-if-missing for a scalar identifier column (read-only after)."""
    builder = _SCALAR_BUILDERS.get(name)
    if builder is None:
        if name.startswith("CDL_"):
            return _cache_column(df, name, lambda: candlestick_pattern(df, name))
        return None
    return _cache_column(df, name, lambda: builder(df))


def _resolve_scalar(df: pd.DataFrame, name: str, i: int) -> float:
    return _value_at(_ensure_scalar(df, name), i)


def ensure_scalar_columns(df: pd.DataFrame, names) -> None:
    """Precompute the given scalar-study columns (VWAP/MACD*/OBV/SAR/STOCH_*/
    CDL_*/…) onto df in one vectorised pass, so the simulator's fast array path
    can read them. Used by the runner before building the per-bar arrays."""
    for name in names:
        _ensure_scalar(df, name)


def _evaluate_function(
    node: FunctionNode,
    df: pd.DataFrame,
    i: int,
    variables: dict[str, float] | None,
) -> Any:
    name = node.name
    args = node.args

    if name in PERIODIC_INDICATOR_NAMES:
        if len(args) == 1:
            period = int(_as_float(_eval_node(args[0], df, i, variables)))
            return _resolve_indicator(df, name, period, i)
        # Two-arg field form is only meaningful for the plain moving averages,
        # e.g. SMA(VOLUME, 20) / EMA(VOLUME, 20).
        if len(args) == 2 and isinstance(args[0], IdentifierNode) and name in {"SMA", "EMA"}:
            field = args[0].name
            period = int(_as_float(_eval_node(args[1], df, i, variables)))
            agg = "mean" if name == "SMA" else "ema"
            return _value_at(_ensure_field_rolling(df, field, period, agg), i)
        raise ParseError(f"{name} expects a single period argument.")

    # AVG / MAX / MIN / STDEV accept either a raw FIELD (CLOSE, VOLUME, …) or a
    # nested periodic indicator (e.g. AVG(ATR(14), 50) for the ATR-volatility
    # regime cards). _arg_series resolves both; a non-resolvable first arg is a
    # ParseError so the problem surfaces loudly at strategy-load, not silently.
    if name in {"AVG", "MAX", "MIN", "STDEV"}:
        if len(args) != 2:
            raise ParseError(f"{name} expects FIELD (or INDICATOR(n)), period.")
        series, _label = _arg_series(df, args[0])
        if series is None:
            raise ParseError(f"{name} first argument must be a field or a periodic indicator.")
        period = int(_as_float(_eval_node(args[1], df, i, variables)))
        agg = {"AVG": "mean", "MAX": "max", "MIN": "min", "STDEV": "stdev"}[name]
        return _roll_over_arg(df, args[0], period, i, agg)

    if name == "ZSCORE":
        if len(args) != 2 or not isinstance(args[0], IdentifierNode):
            raise ParseError("ZSCORE expects FIELD, period.")
        period = int(_as_float(_eval_node(args[1], df, i, variables)))
        return _value_at(_ensure_zscore(df, args[0].name, period), i)

    if name == "PREV":
        if len(args) != 2:
            raise ParseError("PREV expects (FIELD or EXPR), offset.")
        offset = int(_as_float(_eval_node(args[1], df, i, variables)))
        prev_i = i - offset
        if prev_i < 0 or prev_i >= len(df):
            return float("nan")

        # Form 1: PREV(FIELD, k) — fast path for raw OHLCV columns. Pure column
        # lookup at the shifted bar.
        if isinstance(args[0], IdentifierNode):
            ident = args[0].name
            field = _col_name(ident)
            if field in df.columns:
                value = df.iloc[prev_i][field]
                return float(value) if not pd.isna(value) else float("nan")
            # Identifier wasn't a raw column (e.g. VWAP, MACD). Fall through to
            # the generic expression path so the indicator's value at prev_i is
            # computed correctly.
            return _eval_node(args[0], df, prev_i, variables)

        # Form 2: PREV(EXPR, k) — re-evaluate any expression at the shifted bar.
        # Used for slope checks like EMA(20) > PREV(EMA(20), 3).
        return _eval_node(args[0], df, prev_i, variables)

    if name == "RS":
        # Phase 4 — Relative Strength = stock return / reference return over n bars.
        #   RS(n) := (CLOSE / PREV(CLOSE, n)) / (REF_CLOSE / PREV(REF_CLOSE, n))
        # > 1 → stock outperformed reference over the window.
        # NaN-safe: missing reference columns or insufficient history → NaN.
        if len(args) != 1:
            raise ParseError("RS expects a single period argument.")
        n = int(_as_float(_eval_node(args[0], df, i, variables)))
        if n <= 0 or i - n < 0:
            return float("nan")
        ref_col = _col_name("REF_CLOSE")
        if "close" not in df.columns or ref_col not in df.columns:
            return float("nan")
        try:
            close_now = float(df.iloc[i]["close"])
            close_then = float(df.iloc[i - n]["close"])
            ref_now = float(df.iloc[i][ref_col])
            ref_then = float(df.iloc[i - n][ref_col])
        except (KeyError, IndexError):
            return float("nan")
        if any(_is_nan(v) for v in (close_now, close_then, ref_now, ref_then)):
            return float("nan")
        if close_then == 0.0 or ref_then == 0.0 or ref_now == 0.0:
            return float("nan")
        stock_ret = close_now / close_then
        ref_ret = ref_now / ref_then
        if ref_ret == 0.0:
            return float("nan")
        return stock_ret / ref_ret

    if name == "OPENING_RANGE_HIGH":
        if len(args) != 1:
            raise ParseError("OPENING_RANGE_HIGH expects a single period argument.")
        return _session_opening_range(df, "HIGH", int(_as_float(_eval_node(args[0], df, i, variables))), i, "max")

    if name == "OPENING_RANGE_LOW":
        if len(args) != 1:
            raise ParseError("OPENING_RANGE_LOW expects a single period argument.")
        return _session_opening_range(df, "LOW", int(_as_float(_eval_node(args[0], df, i, variables))), i, "min")

    raise ParseError(f"Unsupported function '{name}'.")


# ── AST evaluator ─────────────────────────────────────────────────────────────

def _eval_node(node: Any, df: pd.DataFrame, i: int, variables: dict[str, float] | None) -> Any:
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, IdentifierNode):
        return _resolve_identifier(node.name, df, i, variables)

    if isinstance(node, UnaryNode):
        value = _as_float(_eval_node(node.operand, df, i, variables))
        return -value if node.op == "-" else value

    if isinstance(node, BinaryNode):
        left  = _as_float(_eval_node(node.left,  df, i, variables))
        right = _as_float(_eval_node(node.right, df, i, variables))
        if _is_nan(left) or _is_nan(right):
            return float("nan")
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right if right != 0.0 else float("nan")
        raise ParseError(f"Unsupported binary operator '{node.op}'.")

    if isinstance(node, FunctionNode):
        return _evaluate_function(node, df, i, variables)

    if isinstance(node, ComparisonNode):
        left  = _eval_node(node.left,  df, i, variables)
        right = _eval_node(node.right, df, i, variables)
        if _is_nan(left) or _is_nan(right):
            return False
        if node.op == ">":  return left >  right
        if node.op == "<":  return left <  right
        if node.op == ">=": return left >= right
        if node.op == "<=": return left <= right
        if node.op == "==": return left == right
        if node.op == "!=": return left != right
        raise ParseError(f"Unsupported comparison operator '{node.op}'.")

    if isinstance(node, BooleanNode):
        if node.op == "AND":
            return _truthy(_eval_node(node.left, df, i, variables)) and _truthy(
                _eval_node(node.right, df, i, variables)
            )
        if node.op == "OR":
            return _truthy(_eval_node(node.left, df, i, variables)) or _truthy(
                _eval_node(node.right, df, i, variables)
            )
        raise ParseError(f"Unsupported boolean operator '{node.op}'.")

    if isinstance(node, NotNode):
        return not _truthy(_eval_node(node.operand, df, i, variables))

    raise ParseError(f"Unsupported AST node: {type(node).__name__}")


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _parse_expression(expression: str) -> Any:
    return _Parser(_tokenize(expression.strip())).parse()


def evaluate_condition(
    condition: str,
    df: pd.DataFrame,
    i: int,
    variables: dict[str, float] | None = None,
) -> bool:
    if not condition or i < 0 or i >= len(df):
        return False

    try:
        tree = _parse_expression(condition)
        return bool(_truthy(_eval_node(tree, df, i, variables)))
    except Exception as exc:
        # Warn once per unique (condition, error-type) pair to avoid log flooding.
        warn_key = f"{condition}::{type(exc).__name__}"
        if warn_key not in _warned_conditions:
            _warned_conditions.add(warn_key)
            logger.warning(
                "Condition evaluation error for %r: %s. "
                "This warning will not repeat for the same condition. "
                "Check your entry/exit condition syntax.",
                condition,
                exc,
            )
        return False


def build_entry_signals(
    condition: str,
    df: pd.DataFrame,
    variables: dict[str, float] | None = None,
) -> pd.Series:
    return pd.Series(
        [evaluate_condition(condition, df, i, variables=variables) for i in range(len(df))],
        index=df.index,
    )


# ── Compiled-condition fast path ──────────────────────────────────────────────
#
# `evaluate_condition` re-parses the condition string and walks the AST on EVERY
# bar — for a 210k-bar backtest with 2 conditions that's ~420k parses + 420k AST
# walks. Worse, AST nodes for SMA/EMA/RSI/etc. used to call helpers that
# re-computed the rolling stat from bar 0 to i on every call.
#
# `compile_condition()` parses ONCE up front, scans the AST to find every
# referenced indicator (so the runner can vectorise them as DataFrame columns),
# and returns a `CompiledCondition` whose `.evaluate(df, i, variables)` does
# pure column lookups inside the bar loop. Same semantics as
# `evaluate_condition`, just orders of magnitude faster.

@dataclass(frozen=True)
class IndicatorRef:
    """A single indicator reference found in a condition, e.g. RSI(14) → ('RSI', 14)."""
    name: str
    period: int

    @property
    def column(self) -> str:
        return f"{self.name}_{self.period}"


@dataclass
class CompiledCondition:
    """A pre-parsed condition; evaluate() runs the AST without re-tokenising.

    Two evaluation modes:

      • evaluate(df, i)              — slow, pandas-based. Kept for callers that
                                       still pass a DataFrame (tests, registry path).
      • evaluate_arrays(arrs, n, i)  — fast, numpy-based. The simulator uses this:
                                       `arrs` is a dict of column name → np.ndarray
                                       built ONCE before the bar loop, so per-bar
                                       cost is just numpy[i] reads, no pandas at all.

    `fast_path_safe` is False when the AST uses functions whose values can't be
    served from a precomputed numpy column (AVG/MAX/MIN/OPENING_RANGE_*). In
    that case the simulator must call evaluate(df, i) instead.

    `pattern_refs` (Phase 6) lists the IS_* identifiers the condition uses;
    the runner reads this to know which structural patterns to precompute on
    the df before the bar loop.
    """
    raw: str
    tree: Any
    indicator_refs: tuple[IndicatorRef, ...]
    scalar_refs: tuple[str, ...]  # MACD, MACD_SIGNAL, VWAP — no period
    fast_path_safe: bool = True
    pattern_refs: tuple[str, ...] = ()

    def evaluate(self, df: pd.DataFrame, i: int, variables: dict[str, float] | None = None) -> bool:
        if i < 0 or i >= len(df):
            return False
        try:
            return bool(_truthy(_eval_node(self.tree, df, i, variables)))
        except Exception as exc:
            warn_key = f"{self.raw}::{type(exc).__name__}"
            if warn_key not in _warned_conditions:
                _warned_conditions.add(warn_key)
                logger.warning(
                    "Compiled condition evaluation error for %r: %s. "
                    "This warning will not repeat for the same condition.",
                    self.raw,
                    exc,
                )
            return False

    def evaluate_arrays(
        self,
        arrays: dict[str, np.ndarray],
        n_rows: int,
        i: int,
        variables: dict[str, float] | None = None,
    ) -> bool:
        """Hot-path evaluator — reads numpy arrays directly, no pandas indexing."""
        if i < 0 or i >= n_rows:
            return False
        try:
            return bool(_truthy_fast(_eval_node_arr(self.tree, arrays, n_rows, i, variables)))
        except Exception as exc:
            warn_key = f"{self.raw}::{type(exc).__name__}"
            if warn_key not in _warned_conditions:
                _warned_conditions.add(warn_key)
                logger.warning(
                    "Compiled condition evaluation error for %r: %s. "
                    "This warning will not repeat for the same condition.",
                    self.raw,
                    exc,
                )
            return False


# Single-period studies referenceable as NAME(period) and precomputed as
# NAME_{period} columns by add_all_indicators(). Sourced from the indicators
# registry (PERIODIC_INDICATOR_NAMES) so the two never drift apart, plus the
# session-relative opening range (computed by add_all_indicators the same way).
# _evaluate_function_arr's periodic block reads these directly from the arrays
# dict on the hot path.
_PERIODIC_INDICATORS = set(PERIODIC_INDICATOR_NAMES) | {
    "OPENING_RANGE_HIGH", "OPENING_RANGE_LOW",
}
# Period-less / fixed-param studies referenceable as bare identifiers (VWAP,
# MACD*, OBV, SAR, STOCH_*, PPO, …). CDL_* candlestick identifiers are matched
# by prefix. Sourced from the scalar-builder registry above.
_SCALAR_INDICATORS = set(_SCALAR_IDENTIFIERS_ALL)

# Phase 6 — structural-pattern identifiers. Kept in sync with the keys of
# patterns.PATTERN_COLUMNS via a startup assertion in tests/test_patterns.py.
# We keep the literal set here rather than importing patterns to avoid a
# module-load cycle (patterns imports conditions in the runner side).
_PATTERN_IDENTIFIERS = {
    "IS_SWING_HIGH", "IS_SWING_LOW",
    "IS_HIGHER_HIGH", "IS_LOWER_LOW",
    "IS_BULLISH_FVG", "IS_BEARISH_FVG",
    "IS_BOS_BULLISH", "IS_BOS_BEARISH",
}

# Identifier name → array key in the prebuilt arrays dict.
_IDENT_TO_ARRAY = {
    "CLOSE": "close", "OPEN": "open", "HIGH": "high", "LOW": "low",
    "VOL": "volume", "VOLUME": "volume",
    "VWAP": "VWAP", "MACD": "MACD",
    "MACD_SIGNAL": "MACD_SIGNAL", "MACD_HIST": "MACD_HIST",
    # Phase 4 — reference symbol columns. build_arrays_from_df() only
    # populates these when the runner has merged in reference data.
    "REF_CLOSE": "REF_close", "REF_OPEN": "REF_open",
    "REF_HIGH":  "REF_high",  "REF_LOW":  "REF_low",
    "REF_VOL":   "REF_volume", "REF_VOLUME": "REF_volume",
    # Phase 6 — pattern columns are 0/1 floats added to df by add_all_patterns
    # before simulation. Identifier name == column name.
    "IS_SWING_HIGH":  "IS_SWING_HIGH",  "IS_SWING_LOW":  "IS_SWING_LOW",
    "IS_HIGHER_HIGH": "IS_HIGHER_HIGH", "IS_LOWER_LOW":  "IS_LOWER_LOW",
    "IS_BULLISH_FVG": "IS_BULLISH_FVG", "IS_BEARISH_FVG": "IS_BEARISH_FVG",
    "IS_BOS_BULLISH": "IS_BOS_BULLISH", "IS_BOS_BEARISH": "IS_BOS_BEARISH",
}


def build_arrays_from_df(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert every numeric column to a contiguous float64 array, once.

    The simulator calls this before the bar loop. Lookups inside the loop become
    `arrays[col][i]` — a single numpy memory read — instead of `df.iloc[i][col]`,
    which builds a Series on every call.
    """
    out: dict[str, np.ndarray] = {}
    for col in df.columns:
        try:
            out[col] = np.ascontiguousarray(df[col].to_numpy(dtype=np.float64, copy=False))
        except (TypeError, ValueError):
            # Skip non-numeric columns; they can't appear in conditions anyway.
            continue
    return out


def _truthy_fast(value: Any) -> bool:
    """Like `_truthy` but skips the pd.isna call — caller guarantees float/bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # NaN check via self-comparison (NaN != NaN); 30× faster than math.isnan.
        return value == value and value != 0.0
    return bool(value)


def _eval_node_arr(
    node: Any,
    arrays: dict[str, np.ndarray],
    n_rows: int,
    i: int,
    variables: dict[str, float] | None,
) -> Any:
    """numpy-only mirror of `_eval_node`. No pandas calls anywhere on this path."""
    t = type(node)

    if t is NumberNode:
        return node.value

    if t is IdentifierNode:
        name = node.name
        if variables is not None and name in variables:
            return variables[name]
        if name == "TRUE":
            return True
        if name == "FALSE":
            return False
        # Scalar / pattern / CDL identifiers map to a precomputed column of the
        # same name; fall back to the identifier itself so any column the runner
        # precomputed (OBV, SAR, STOCH_K, CDL_ENGULFING, …) is readable here.
        col = _IDENT_TO_ARRAY.get(name, name)
        arr = arrays.get(col)
        if arr is None:
            return float("nan")
        return float(arr[i])

    if t is BinaryNode:
        left = _eval_node_arr(node.left, arrays, n_rows, i, variables)
        right = _eval_node_arr(node.right, arrays, n_rows, i, variables)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return float("nan")
        # NaN propagation via self-comparison (faster than pd.isna)
        if left != left or right != right:
            return float("nan")
        op = node.op
        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/": return left / right if right != 0.0 else float("nan")
        raise ParseError(f"Unsupported binary operator '{op}'.")

    if t is ComparisonNode:
        left = _eval_node_arr(node.left, arrays, n_rows, i, variables)
        right = _eval_node_arr(node.right, arrays, n_rows, i, variables)
        # NaN comparison → False
        try:
            if (isinstance(left, float) and left != left) or (isinstance(right, float) and right != right):
                return False
        except TypeError:
            return False
        op = node.op
        if op == ">":  return left >  right
        if op == "<":  return left <  right
        if op == ">=": return left >= right
        if op == "<=": return left <= right
        if op == "==": return left == right
        if op == "!=": return left != right
        raise ParseError(f"Unsupported comparison operator '{op}'.")

    if t is BooleanNode:
        op = node.op
        if op == "AND":
            return _truthy_fast(_eval_node_arr(node.left, arrays, n_rows, i, variables)) and \
                   _truthy_fast(_eval_node_arr(node.right, arrays, n_rows, i, variables))
        if op == "OR":
            return _truthy_fast(_eval_node_arr(node.left, arrays, n_rows, i, variables)) or \
                   _truthy_fast(_eval_node_arr(node.right, arrays, n_rows, i, variables))
        raise ParseError(f"Unsupported boolean operator '{op}'.")

    if t is NotNode:
        return not _truthy_fast(_eval_node_arr(node.operand, arrays, n_rows, i, variables))

    if t is UnaryNode:
        v = _eval_node_arr(node.operand, arrays, n_rows, i, variables)
        if isinstance(v, (int, float)):
            return -v if node.op == "-" else v
        return float("nan")

    if t is FunctionNode:
        return _evaluate_function_arr(node, arrays, n_rows, i, variables)

    raise ParseError(f"Unsupported AST node: {type(node).__name__}")


def _evaluate_function_arr(
    node: "FunctionNode",
    arrays: dict[str, np.ndarray],
    n_rows: int,
    i: int,
    variables: dict[str, float] | None,
) -> Any:
    """numpy-fast function evaluator. Periodic indicators MUST be precomputed."""
    name = node.name
    args = node.args

    # SMA(20) / EMA(20) / RSI(14) / BB_UPPER(20) / etc. — single arg form
    if name in _PERIODIC_INDICATORS and len(args) == 1 and type(args[0]) is NumberNode:
        period = int(args[0].value)
        col = f"{name}_{period}"
        arr = arrays.get(col)
        if arr is None:
            # Slow fallback: if the runner forgot to precompute, fall back to
            # the original pandas path so we still produce a correct value.
            return float("nan")
        return float(arr[i])

    if name == "PREV":
        if len(args) != 2:
            raise ParseError("PREV expects (FIELD or EXPR), offset.")
        offset_val = _eval_node_arr(args[1], arrays, n_rows, i, variables)
        if not isinstance(offset_val, (int, float)) or offset_val != offset_val:
            return float("nan")
        prev_i = i - int(offset_val)
        if prev_i < 0:
            return float("nan")

        # Form 1: PREV(IDENT, k) — column lookup at the shifted bar. Fall back to
        # the identifier name itself so scalar columns (OBV, SAR, STOCH_K, …) —
        # which are stored under their uppercase name — resolve correctly.
        if type(args[0]) is IdentifierNode:
            col = _IDENT_TO_ARRAY.get(args[0].name, args[0].name)
            arr = arrays.get(col)
            if arr is None:
                return float("nan")
            return float(arr[prev_i])

        # Form 2: PREV(EXPR, k) — re-evaluate the expression at the shifted bar.
        # Conditions that use this form are marked fast-path-unsafe in
        # _collect_refs so they normally take the slow pandas path. We still
        # support it here as a defensive fallback for callers that bypass
        # CompiledCondition and use evaluate_arrays() directly.
        return _eval_node_arr(args[0], arrays, n_rows, prev_i, variables)

    # AVG / MAX / MIN / STDEV / ZSCORE / RS — not yet pre-vectorised; conditions
    # using these are marked fast_path_unsafe in _collect_refs and take the slow
    # pandas path. Same fix shape as OPENING_RANGE_*: precompute as columns.
    return float("nan")


_FAST_PATH_UNSAFE_FUNCS = {
    "AVG", "MAX", "MIN", "STDEV", "ZSCORE", "RS",
}


def _collect_refs(
    node: Any,
    indicators: set[IndicatorRef],
    scalars: set[str],
    patterns: set[str],
    flags: dict[str, bool],
) -> None:
    """Walk the AST: collect indicator refs, scalar refs, pattern refs, and
    detect any fast-path-unsafe func."""
    if isinstance(node, FunctionNode):
        if node.name in _FAST_PATH_UNSAFE_FUNCS:
            flags["fast_path_safe"] = False
        # PREV(FIELD, k) is fast-path-safe; PREV(EXPR, k) — e.g. PREV(EMA(20), 3)
        # for slope checks — needs to re-evaluate the inner expression at a
        # shifted bar, which the prebuilt-array path doesn't precompute.
        if node.name == "PREV" and len(node.args) >= 1 and not isinstance(node.args[0], IdentifierNode):
            flags["fast_path_safe"] = False
        if node.name in _PERIODIC_INDICATORS and len(node.args) == 1 and isinstance(node.args[0], NumberNode):
            indicators.add(IndicatorRef(name=node.name, period=int(node.args[0].value)))
        elif node.name in _PERIODIC_INDICATORS:
            # Two-arg form e.g. SMA(CLOSE, 20) — can't be pre-served from a column.
            flags["fast_path_safe"] = False
        for arg in node.args:
            _collect_refs(arg, indicators, scalars, patterns, flags)
    elif isinstance(node, IdentifierNode):
        if node.name in _SCALAR_INDICATORS or node.name.startswith("CDL_"):
            scalars.add(node.name)
        elif node.name in _PATTERN_IDENTIFIERS:
            patterns.add(node.name)
    elif isinstance(node, (UnaryNode, NotNode)):
        _collect_refs(getattr(node, "operand", None), indicators, scalars, patterns, flags)
    elif isinstance(node, (BinaryNode, ComparisonNode, BooleanNode)):
        _collect_refs(node.left, indicators, scalars, patterns, flags)
        _collect_refs(node.right, indicators, scalars, patterns, flags)


# Every function the evaluator can actually compute. Used to fail loudly at
# compile time: an unknown function (e.g. a regime filter the formula path can't
# evaluate) previously raised mid-backtest, got swallowed to `False`, and — since
# entry filters are AND-ed — silently turned the whole strategy into zero trades.
# Validating here surfaces it as a clear strategy-load error instead.
_KNOWN_FUNCTIONS = set(_PERIODIC_INDICATORS) | {
    "AVG", "MAX", "MIN", "STDEV", "ZSCORE",
    "PREV", "RS",
}


def _validate_known_functions(node: Any) -> None:
    """Walk the AST and raise ParseError on any function the evaluator can't run."""
    if isinstance(node, FunctionNode):
        if node.name not in _KNOWN_FUNCTIONS:
            raise ParseError(
                f"Unsupported function {node.name!r} in condition. "
                f"Supported functions: {', '.join(sorted(_KNOWN_FUNCTIONS))}."
            )
        for arg in node.args:
            _validate_known_functions(arg)
    elif isinstance(node, (UnaryNode, NotNode)):
        _validate_known_functions(node.operand)
    elif isinstance(node, (BinaryNode, ComparisonNode, BooleanNode)):
        _validate_known_functions(node.left)
        _validate_known_functions(node.right)


def compile_condition(condition: str) -> CompiledCondition | None:
    """
    Parse `condition` once and return a CompiledCondition.

    Returns None for empty conditions. Raises ParseError on malformed input or
    an unsupported function so callers can surface the problem at strategy-load
    time, not silently mid-backtest.
    """
    if not condition or not str(condition).strip():
        return None
    tree = _parse_expression(condition)
    _validate_known_functions(tree)
    indicators: set[IndicatorRef] = set()
    scalars: set[str] = set()
    patterns: set[str] = set()
    flags: dict[str, bool] = {"fast_path_safe": True}
    _collect_refs(tree, indicators, scalars, patterns, flags)
    return CompiledCondition(
        raw=condition,
        tree=tree,
        indicator_refs=tuple(sorted(indicators, key=lambda r: (r.name, r.period))),
        scalar_refs=tuple(sorted(scalars)),
        fast_path_safe=flags["fast_path_safe"],
        pattern_refs=tuple(sorted(patterns)),
    )
