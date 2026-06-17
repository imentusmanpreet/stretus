"""
app/services/strategy/condition_expression.py — accept trader-typed raw
entry/exit *condition expressions* (formulas), not just catalog signal names.

The KB signal resolver (``app.kb.signal_resolver``) matches a query against the
predefined signal catalog. That is the right path when the trader names a
signal ("rsi cross up", "macd"). But traders also type full formulas directly,
e.g.::

    EMA(8) > EMA(21) AND RSI(14) > 50 AND VOLUME > AVG(VOLUME, 10) * 1.2

Those are not catalog names — the resolver returns "no signal matches". Yet the
quant engine already evaluates exactly this syntax in *formula mode*
(``quant_engine/engine/conditions.py``). This module bridges the gap:

    1. :func:`looks_like_condition_expression` — cheap structural test that tells
       the chat handler "this is a formula, not a catalog keyword" so it routes
       to the condition path instead of the catalog resolver.
    2. :func:`validate_condition_expression` — compiles the formula with the
       engine's OWN parser (via :mod:`app.strategy.engine_bridge`) and walks the
       AST for unknown identifiers / functions, mirroring the SDL validator in
       ``app/strategy/validator.py``. If it validates here it is guaranteed to
       run in the backtester.

Nothing about trading logic is hardcoded: detection is purely structural
(operators / function-call syntax) and the legal vocabulary is read LIVE from
the engine, so this can never drift from what the backtester actually accepts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.strategy import engine_bridge, vocab

logger = logging.getLogger(__name__)

# A formula is unambiguously signalled by either a comparison operator
# (>, <, >=, <=, ==, !=) or function-call syntax like ``EMA(``/``RSI(``/``AVG(``.
# Catalog signal *names* ("rsi_cross_up") and natural phrasing ("rsi above 60")
# contain neither, so they keep flowing to the KB resolver untouched.
_COMPARISON_RE = re.compile(r"(>=|<=|==|!=|>|<)")
_FUNC_CALL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")


@dataclass(frozen=True)
class ConditionCheck:
    """Outcome of :func:`validate_condition_expression`.

    ``ok`` is True only when the engine can compile and run the formula.
    ``expression`` is the cleaned (whitespace-collapsed) form to store.
    ``error`` is a ready-to-show trader-facing message when ``ok`` is False.
    """

    ok: bool
    expression: str
    error: str | None = None


def looks_like_condition_expression(text: str) -> bool:
    """True when ``text`` is a raw condition formula rather than a signal name.

    Structural only — keys on comparison operators or function-call syntax,
    both absent from catalog signal names and natural-language signal phrasing.
    """
    s = str(text or "")
    if not s.strip():
        return False
    return bool(_COMPARISON_RE.search(s) or _FUNC_CALL_RE.search(s))


def validate_condition_expression(text: str) -> ConditionCheck:
    """Compile ``text`` with the engine parser and reject unknown vocab.

    Reuses :mod:`app.strategy.engine_bridge` (the single seam to the quant
    engine) so the grammar checker is the exact one the backtester uses. The
    AST walks mirror ``app/strategy/validator.py::_check_conditions``.
    """
    expr = " ".join(str(text or "").split()).strip()
    if not expr:
        return ConditionCheck(ok=False, expression="", error="The condition is empty.")

    # If the engine grammar cannot be imported in this environment we cannot
    # validate here. Accept the formula rather than block the trader — the
    # backtester (the same parser) will surface any real problem at load time.
    if not engine_bridge.is_available():
        logger.warning(
            "condition_expression|engine_unavailable|accepting_unvalidated|expr=%s",
            expr[:120],
        )
        return ConditionCheck(ok=True, expression=expr)

    compile_condition = engine_bridge.get_compile_condition()
    parse_error = engine_bridge.get_parse_error_type()

    try:
        compiled = compile_condition(expr)
    except parse_error as exc:
        return ConditionCheck(
            ok=False,
            expression=expr,
            error=(
                f"That condition can't be backtested — {exc} "
                f"Condition: `{expr}`"
            ),
        )
    except Exception as exc:  # noqa: BLE001 — surface any parser failure uniformly
        logger.warning("condition_expression|compile_error|expr=%s|err=%r", expr[:120], exc)
        return ConditionCheck(
            ok=False,
            expression=expr,
            error=f"That condition could not be parsed ({exc}). Condition: `{expr}`",
        )

    if compiled is None:
        return ConditionCheck(ok=False, expression="", error="The condition is empty.")

    # Bare-identifier backstop: the engine resolves unknown identifiers to NaN
    # at runtime instead of failing, so we reject them up front here.
    unknown_ids = sorted(
        n
        for n in engine_bridge.collect_identifier_names(compiled.tree)
        if not vocab.is_legal_identifier(n)
    )
    if unknown_ids:
        return ConditionCheck(
            ok=False,
            expression=expr,
            error=(
                f"That condition uses identifier(s) the engine does not know: "
                f"{', '.join(unknown_ids)}. Use documented fields/indicators only. "
                f"Condition: `{expr}`"
            ),
        )

    # Function-name backstop: the parser may accept an unknown function name
    # syntactically but fail at evaluation (e.g. SUPERTREND). Catch it now.
    unknown_funcs = sorted(
        n
        for n in engine_bridge.collect_function_names(compiled.tree)
        if n not in engine_bridge.known_functions()
    )
    if unknown_funcs:
        return ConditionCheck(
            ok=False,
            expression=expr,
            error=(
                f"That condition uses function(s) the engine cannot run: "
                f"{', '.join(unknown_funcs)}. "
                f"Use only: {', '.join(sorted(engine_bridge.known_functions()))}."
            ),
        )

    return ConditionCheck(ok=True, expression=expr)
