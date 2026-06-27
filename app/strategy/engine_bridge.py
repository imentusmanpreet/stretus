"""
app/strategy/engine_bridge.py — the ONLY seam between the app and the quant engine.

The quant engine (``quant_engine/engine``) imports itself as the top-level
package ``engine`` (e.g. ``conditions.py`` does ``import engine.indicators``), so
the app can only import it after ``quant_engine/`` is on ``sys.path``. This module
does that once, idempotently (the same trick ``tests/conftest.py`` and
``app/planner/condition_satisfiability.py`` already rely on), and re-exports the
engine's grammar primitives and vocabulary.

Nothing here is hardcoded about trading logic: every vocabulary the new strategy
path uses — legal functions, indicator names, stop-loss/trailing types — is read
LIVE from the engine, so the LLM prompt and the validator can never drift from
what the backtester actually accepts.

Fail-closed: if the engine cannot be imported (e.g. TA-Lib missing in a bare
environment), accessors raise :class:`EngineBridgeUnavailable`. The validator
treats that as a hard error rather than silently passing an unvalidated strategy.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Repo root = three parents up from this file: app/strategy/engine_bridge.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QUANT_ENGINE_ROOT = _REPO_ROOT / "quant_engine"


class EngineBridgeUnavailable(RuntimeError):
    """The quant engine grammar could not be imported in this environment."""


def _ensure_engine_on_path() -> None:
    """Prepend ``quant_engine/`` to ``sys.path`` once (idempotent).

    Mirrors ``tests/conftest.py`` so the engine resolves as ``engine.*``. Safe to
    call repeatedly and from any process (does NOT depend on the test conftest).
    """
    for candidate in (_REPO_ROOT, _QUANT_ENGINE_ROOT):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


@lru_cache(maxsize=1)
def _import_conditions() -> Any:
    _ensure_engine_on_path()
    try:
        import engine.conditions as conditions  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 — surface any import failure uniformly
        raise EngineBridgeUnavailable(
            f"quant engine grammar (engine.conditions) is not importable: {exc!r}. "
            "Ensure quant_engine/ is present and its deps (TA-Lib) are installed."
        ) from exc
    return conditions


@lru_cache(maxsize=1)
def _import_loader() -> Any:
    _ensure_engine_on_path()
    try:
        import engine.loader as loader  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise EngineBridgeUnavailable(
            f"quant engine loader (engine.loader) is not importable: {exc!r}."
        ) from exc
    return loader


@lru_cache(maxsize=1)
def _import_indicators() -> Any:
    _ensure_engine_on_path()
    try:
        import engine.indicators as indicators  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise EngineBridgeUnavailable(
            f"quant engine indicators (engine.indicators) is not importable: {exc!r}."
        ) from exc
    return indicators


def is_available() -> bool:
    """True when the engine grammar can be imported. Never raises."""
    try:
        _import_conditions()
        return True
    except EngineBridgeUnavailable:
        return False


# ── Grammar primitives ────────────────────────────────────────────────────────

def get_compile_condition() -> Callable[[str], Any]:
    """The engine's own ``compile_condition`` — the single grammar checker.

    Returns ``None`` for an empty string; raises the engine's ``ParseError`` on a
    malformed condition or an unsupported function.
    """
    return _import_conditions().compile_condition


def get_parse_error_type() -> type[Exception]:
    return _import_conditions().ParseError


# ── Vocabulary (read live from the engine) ────────────────────────────────────

def known_functions() -> frozenset[str]:
    """Every function the evaluator can run (periodic indicators + AVG/MAX/MIN/…)."""
    return frozenset(_import_conditions()._KNOWN_FUNCTIONS)


def periodic_indicator_names() -> frozenset[str]:
    return frozenset(_import_indicators().PERIODIC_INDICATOR_NAMES)


def scalar_identifiers() -> frozenset[str]:
    """Period-less studies usable as bare identifiers (VWAP, MACD, OBV, SAR, …)."""
    return frozenset(_import_conditions()._SCALAR_INDICATORS)


def pattern_identifiers() -> frozenset[str]:
    return frozenset(_import_conditions()._PATTERN_IDENTIFIERS)


def candlestick_names() -> frozenset[str]:
    """Every TA-Lib candlestick pattern the engine can compute, as ``CDL_*`` aliases.

    Read LIVE from the engine's TA-Lib pattern map (``talib`` Pattern Recognition
    group), so the prompt and validator know exactly which CDL_* names are real —
    a name not in this set produces an all-NaN column (⇒ silent zero trades), so
    the validator rejects it instead of letting it through.
    """
    return frozenset(_import_indicators()._CDL_FUNCTIONS.keys())


def common_candlestick_names() -> tuple[str, ...]:
    """A curated subset of frequently-used candlestick patterns, in catalog order.

    Surfaced in the prompt as concrete examples; the full set (``candlestick_names``)
    remains legal — these are just the ones worth naming explicitly.
    """
    return tuple(_import_indicators().COMMON_CANDLESTICK_PATTERNS)


def identifier_to_array_keys() -> frozenset[str]:
    """All bare identifiers the evaluator maps to a column (CLOSE, REF_CLOSE, IS_*…)."""
    return frozenset(_import_conditions()._IDENT_TO_ARRAY.keys())


def stop_loss_types() -> frozenset[str]:
    return frozenset(_import_loader()._STOP_LOSS_TYPES)


def stop_loss_anchors() -> frozenset[str]:
    return frozenset(_import_loader()._STOP_LOSS_ANCHORS)


def trailing_stop_types() -> frozenset[str]:
    return frozenset(_import_loader()._TRAILING_TYPES)


def trailing_take_profit_types() -> frozenset[str]:
    return frozenset(_import_loader()._TRAILING_TAKE_PROFIT_TYPES)


def normalise_market(market: str) -> str:
    """The engine's canonical market id for a free-text market string."""
    return _import_loader()._normalise_market(market)


def normalise_objective(objective: str) -> str:
    return _import_loader()._normalise_objective(objective)


# ── AST inspection ────────────────────────────────────────────────────────────

def collect_identifier_names(tree: Any) -> set[str]:
    """Walk a compiled condition's AST and return every bare-identifier name.

    The engine's ``compile_condition`` rejects unknown *functions* but resolves an
    unknown bare *identifier* (e.g. ``FOO``) to NaN at runtime instead of failing.
    The validator uses this to catch fabricated identifiers strictly. We isinstance
    against the engine's real node classes so the walk can't silently miss a node
    shape the parser added.
    """
    cond = _import_conditions()
    names: set[str] = set()

    def _walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, cond.IdentifierNode):
            names.add(node.name)
            return
        if isinstance(node, cond.FunctionNode):
            for arg in node.args:
                _walk(arg)
            return
        # Binary / comparison / boolean nodes expose .left and .right; unary / not
        # nodes expose .operand. Use getattr so a new node shape degrades safely.
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        operand = getattr(node, "operand", None)
        if left is not None or right is not None:
            _walk(left)
            _walk(right)
        elif operand is not None:
            _walk(operand)

    _walk(tree)
    return names


def collect_function_names(tree: Any) -> set[str]:
    """Walk a compiled condition's AST and return every function name used.

    The engine's parser does NOT always reject unknown function names at parse
    time — it may only fail at evaluation. This lets the validator catch
    fabricated functions (e.g. SUPERTREND) before they reach the backtest.
    """
    cond = _import_conditions()
    names: set[str] = set()

    def _walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, cond.FunctionNode):
            names.add(node.name)
            for arg in node.args:
                _walk(arg)
            return
        if isinstance(node, cond.IdentifierNode):
            return
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        operand = getattr(node, "operand", None)
        if left is not None or right is not None:
            _walk(left)
            _walk(right)
        elif operand is not None:
            _walk(operand)

    _walk(tree)
    return names
