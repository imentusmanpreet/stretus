"""
app/planner/engine_contract.py — the SINGLE source of truth for what the
backtest engine (`quant_engine`) actually accepts for risk execution (§2b).

Why this exists
───────────────
The SDL stop-loss vocabulary is rich (swing_low, orb_high, vwap_deviation, …),
but the engine's loader only accepts THREE stop types — percent / atr /
structural — and structural only with FOUR anchors. The old compiler papered
over the gap with a local `anchor_map` whose `.get(type, "opening_range_low")`
default SILENTLY turned any unmapped anchor (e.g. `vwap_deviation`) into an ORB
stop — a wrong strategy the user never asked for, and `vwap_deviation` SL still
reached the engine and errored.

This module is imported by BOTH sides:
  * the planner gate normalises every SDL stop to an engine-legal form (or marks
    it NEEDS_CLARIFY) BEFORE render, so the planner can never emit an anchor the
    engine rejects;
  * `quant_engine/engine/loader.py` imports `STOP_LOSS_ANCHORS`/`STOP_LOSS_TYPES`
    from here when co-located (it keeps a local fallback for its standalone
    deployment, guarded by a sync test).

Nothing here is silent: a stop that can't be mapped deterministically becomes a
clarification at the gate, never a substituted concept.
"""
from __future__ import annotations

from typing import Final

# ── Engine-accepted vocabulary (mirror of quant_engine/engine/loader.py) ──────
# These MUST stay in lock-step with the engine loader. tests/test_planner/
# test_engine_contract.py asserts equality against loader._STOP_LOSS_ANCHORS /
# _STOP_LOSS_TYPES so the two can never silently diverge.
STOP_LOSS_TYPES: Final[tuple[str, ...]] = ("percent", "structural", "atr")

STOP_LOSS_ANCHORS: Final[tuple[str, ...]] = (
    "opening_range_high",
    "opening_range_low",
    "prev_n_bar_high",
    "prev_n_bar_low",
)

TRAILING_TYPES: Final[tuple[str, ...]] = ("percent", "atr", "ema", "chandelier")

# Trailing TAKE-PROFIT types. Phase 1 ships `percent` only: a ratcheting give-back
# line measured as a % below the running peak (long) / above the trough (short).
# ATR/EMA/chandelier are intentionally excluded for now because the live OMS that
# enforces trailing exits can only honour price-based (percent) trailing — keeping
# this set to percent guarantees backtest↔live parity. Widen only when the OMS can
# compute indicator-based trailing too.
TRAILING_TAKE_PROFIT_TYPES: Final[tuple[str, ...]] = ("percent",)

# Engine take-profit is consumed as a single percent; every SDL TP type reduces
# to a pct EXCEPT `points` (absolute price distance — needs the price to convert).
TAKE_PROFIT_TYPES: Final[tuple[str, ...]] = ("percent", "rr", "atr")

# Sentinels returned by the mapping helpers.
NEEDS_CLARIFY: Final[str] = "NEEDS_CLARIFY"
NEEDS_DIRECTION: Final[str] = "NEEDS_DIRECTION"


# ── Planner-anchor → engine-anchor map ────────────────────────────────────────
# Keys are SDL `risk.stop_loss.type` literals that denote a STRUCTURAL stop.
# Values are the engine structural anchor id, NEEDS_DIRECTION (resolve via the
# leg direction), or NEEDS_CLARIFY (the engine genuinely can't express it).
_PLANNER_ANCHOR_TO_ENGINE: Final[dict[str, str]] = {
    "swing_low": "prev_n_bar_low",
    "swing_high": "prev_n_bar_high",
    "candle_low": "prev_n_bar_low",
    "candle_high": "prev_n_bar_high",
    "orb_low": "opening_range_low",
    "orb_high": "opening_range_high",
    "opposite_side": NEEDS_DIRECTION,   # long → prev_n_bar_low, short → prev_n_bar_high
    "vwap_deviation": NEEDS_CLARIFY,    # engine has no VWAP-distance stop
}

# `opposite_side` resolved by leg direction.
_OPPOSITE_SIDE_BY_DIRECTION: Final[dict[str, str]] = {
    "long": "prev_n_bar_low",
    "short": "prev_n_bar_high",
}

# SDL stop types the engine takes directly (non-structural).
_DIRECT_SL_TYPES: Final[frozenset[str]] = frozenset({"percent", "atr"})


def is_structural_anchor(planner_type: str) -> bool:
    """True if the SDL stop type denotes a structural anchor (mappable or not)."""
    return str(planner_type or "").lower().strip() in _PLANNER_ANCHOR_TO_ENGINE


def map_stop_loss_type(planner_type: str, direction: str | None = None) -> dict | str:
    """Normalise an SDL `risk.stop_loss.type` to the engine contract.

    Returns one of:
      * {"type": "percent"}                              — direct
      * {"type": "atr"}                                  — direct
      * {"type": "structural", "anchor": <engine anchor>}— mapped structural
      * NEEDS_CLARIFY                                    — engine can't express it
        (vwap_deviation, points, or an unknown type)

    `direction` ("long"/"short") resolves the `opposite_side` anchor; when it's
    needed but absent the result is NEEDS_CLARIFY (don't guess a side).
    """
    t = str(planner_type or "").lower().strip()
    if t in _DIRECT_SL_TYPES:
        return {"type": t}
    if t in _PLANNER_ANCHOR_TO_ENGINE:
        anchor = _PLANNER_ANCHOR_TO_ENGINE[t]
        if anchor == NEEDS_CLARIFY:
            return NEEDS_CLARIFY
        if anchor == NEEDS_DIRECTION:
            resolved = _OPPOSITE_SIDE_BY_DIRECTION.get(str(direction or "").lower().strip())
            if not resolved:
                return NEEDS_CLARIFY
            anchor = resolved
        if anchor not in STOP_LOSS_ANCHORS:
            return NEEDS_CLARIFY
        return {"type": "structural", "anchor": anchor}
    # Everything else (points, or any value not in the SDL vocabulary) → clarify.
    return NEEDS_CLARIFY


def map_take_profit_type(planner_type: str) -> str | None:
    """Return the engine-legal TP family, or NEEDS_CLARIFY for `points`/unknown.

    The engine ultimately takes a take-profit PERCENT; percent/rr/atr all reduce
    to one. `points` is an absolute price distance with no pct without the live
    price → clarify rather than silently approximate.
    """
    t = str(planner_type or "").lower().strip()
    if t in TAKE_PROFIT_TYPES:
        return t
    return NEEDS_CLARIFY
