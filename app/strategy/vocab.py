"""
app/strategy/vocab.py — the legal condition vocabulary, derived LIVE from the engine.

Both the LLM prompt (what the model is told it may use) and the validator (what it
is allowed to accept) read from this one module, which in turn reads from
:mod:`app.strategy.engine_bridge`. Nothing about indicators or trading logic is
hand-typed here, so the prompt, the validator, and the backtester can never drift.

The only literals defined locally are the evaluator's structural grammar tokens —
the boolean constants and the exit-only variables (``PROFIT``, ``LOSS``,
``TAKE_PROFIT_TARGET``, ``STOP_LOSS_TARGET``) that the simulator injects at runtime
rather than exposing as an importable constant. These are grammar, not signals.
"""
from __future__ import annotations

from functools import lru_cache

from app.strategy import engine_bridge

# Evaluator grammar tokens (from engine.conditions._eval_node_arr + the simulator's
# per-bar `variables` dict). Not trading logic — the boolean literals and the
# exit-condition variables the grammar always understands.
GRAMMAR_BOOLEANS: frozenset[str] = frozenset({"TRUE", "FALSE"})
EXIT_VARIABLES: frozenset[str] = frozenset(
    {"PROFIT", "LOSS", "TAKE_PROFIT_TARGET", "STOP_LOSS_TARGET"}
)

# Rolling-window helper functions that take a (FIELD-or-indicator, period) form.
# These are a documented subset of the engine's known functions; we surface them
# separately only to describe their call shape in the prompt. Membership is still
# validated against the engine's live `known_functions()`.
_ROLLING_FUNCTION_HINTS: frozenset[str] = frozenset(
    {"AVG", "MAX", "MIN", "STDEV", "ZSCORE", "PREV", "RS"}
)


@lru_cache(maxsize=1)
def periodic_functions() -> frozenset[str]:
    """Functions that take a single period arg, e.g. ``EMA(20)``, ``RSI(14)``."""
    return engine_bridge.periodic_indicator_names() | frozenset(
        {"OPENING_RANGE_HIGH", "OPENING_RANGE_LOW"}
    )


@lru_cache(maxsize=1)
def rolling_functions() -> frozenset[str]:
    """Known functions that are NOT single-period periodic indicators (AVG, PREV…)."""
    return engine_bridge.known_functions() - periodic_functions()


@lru_cache(maxsize=1)
def scalar_identifiers() -> frozenset[str]:
    """Bare-identifier studies: VWAP, MACD, MACD_SIGNAL, OBV, SAR, STOCH_*, …."""
    return engine_bridge.scalar_identifiers()


@lru_cache(maxsize=1)
def pattern_identifiers() -> frozenset[str]:
    """Structural-pattern flags: IS_SWING_HIGH, IS_BOS_BULLISH, IS_BULLISH_FVG, …."""
    return engine_bridge.pattern_identifiers()


@lru_cache(maxsize=1)
def candlestick_names() -> frozenset[str]:
    """Every real TA-Lib candlestick pattern (CDL_ENGULFING, CDL_HAMMER, …)."""
    return engine_bridge.candlestick_names()


@lru_cache(maxsize=1)
def common_candlestick_names() -> tuple[str, ...]:
    """A curated subset of frequently-used candlestick patterns (for the prompt)."""
    return engine_bridge.common_candlestick_names()


@lru_cache(maxsize=1)
def all_legal_identifiers() -> frozenset[str]:
    """Every bare identifier (no parentheses) the evaluator can resolve to a value.

    Used by the validator to reject fabricated identifiers — the engine's
    ``compile_condition`` rejects unknown *functions* but silently resolves an
    unknown bare identifier to NaN, so we backstop it here. The CDL_* candlestick
    names are the EXACT TA-Lib patterns the engine can compute — a made-up CDL_*
    name resolves to all-NaN (⇒ silent zero trades), so we list the real ones here
    and reject the rest in :func:`is_legal_identifier`.
    """
    return (
        engine_bridge.identifier_to_array_keys()
        | scalar_identifiers()
        | pattern_identifiers()
        | candlestick_names()
        | GRAMMAR_BOOLEANS
        | EXIT_VARIABLES
    )


def is_legal_identifier(name: str) -> bool:
    """True if ``name`` is a resolvable bare identifier (incl. real CDL_* patterns).

    A CDL_* name that TA-Lib does not define is NOT legal: it would compute to an
    all-NaN column and silently produce zero trades, so we reject it here.
    """
    return name in all_legal_identifiers()


# ── Risk vocabulary (read live from the engine loader) ────────────────────────

def stop_loss_types() -> frozenset[str]:
    return engine_bridge.stop_loss_types()


def stop_loss_anchors() -> frozenset[str]:
    return engine_bridge.stop_loss_anchors()


def trailing_stop_types() -> frozenset[str]:
    return engine_bridge.trailing_stop_types()


def trailing_take_profit_types() -> frozenset[str]:
    return engine_bridge.trailing_take_profit_types()


# ── Prompt-facing grammar description (generated) ─────────────────────────────

def grammar_summary_for_prompt() -> str:
    """A generated, human-readable description of the engine condition grammar.

    Built entirely from the live engine vocabulary so the model is never told
    about a function the backtester can't run. No example strategies.
    """
    def _fmt(names: frozenset[str]) -> str:
        return ", ".join(sorted(names))

    periodic = sorted(periodic_functions())
    rolling = sorted(rolling_functions() & _ROLLING_FUNCTION_HINTS) or sorted(rolling_functions())

    lines = [
        "CONDITION GRAMMAR (the backtest engine compiles these strings directly):",
        "",
        "Operators:",
        "  comparison: >  <  >=  <=  ==  !=",
        "  boolean:    AND  OR  NOT",
        "  arithmetic: +  -  *  /  and parentheses ( )",
        "",
        "Price/volume fields (bare identifiers):",
        "  CLOSE, OPEN, HIGH, LOW, VOLUME (alias VOL), VWAP",
        "",
        f"Periodic functions — call as NAME(period), e.g. EMA(20), RSI(14):",
        f"  {_fmt(frozenset(periodic))}",
        "",
        "Window/utility functions:",
        "  AVG(FIELD or INDICATOR(n), n)  MAX(...)  MIN(...)  STDEV(FIELD, n)",
        "  ZSCORE(FIELD, n)  PREV(FIELD or EXPR, offset)  RS(n)",
        f"  (engine-known: {_fmt(frozenset(rolling))})",
        "",
        "Period-less studies (bare identifiers):",
        f"  {_fmt(scalar_identifiers())}",
        "",
        "Structural-pattern flags (bare identifiers, 0/1):",
        f"  {_fmt(pattern_identifiers())}",
        "",
        "Reference-symbol fields (only when a reference_symbol is set):",
        "  REF_CLOSE, REF_OPEN, REF_HIGH, REF_LOW, REF_VOLUME",
        "",
        "Exit-only variables (usable in exit conditions):",
        f"  {_fmt(EXIT_VARIABLES)}",
        "",
        f"Candlestick patterns — {len(candlestick_names())} CDL_* bare identifiers, "
        "signed pattern strength: +100 bullish, -100 bearish, 0 none (a few combine "
        "to ±200). Use an EXACT TA-Lib pattern name (made-up names are rejected). "
        f"Common ones: {', '.join(common_candlestick_names())}.",
        "",
        "Rules: every function/identifier you use MUST come from the lists above. "
        "Do not invent indicators. Conditions must be able to evaluate to true on "
        "real data (no self-contradictions like RSI(14) >= 70 AND RSI(14) <= 30).",
    ]
    return "\n".join(lines)
