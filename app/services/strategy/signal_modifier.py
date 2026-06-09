"""
app/services/strategy/signal_modifier.py — orchestrate trader-driven signal swaps.

This module ties together the three primitives that already exist:

    1. ``app.kb.signal_resolver.resolve_signal_name``   — fuzzy → candidate set
    2. ``app.kb.signal_validation.validate_signal_placement`` — role gate
    3. ``StrategyBuilder.modify_signal`` / ``remove_signal``  — surgical edit

The chat handler (or any future agent tool) calls
:func:`apply_signal_request` with the trader's typed query. The helper:

    - resolves the query against the KB (slot-restricted)
    - if exactly one signal matches → applies it to ``builder.signal_plan``
      and returns ``status="applied"``
    - if 2–4 match → returns ``status="needs_choice"`` with the candidate
      list and a ready-to-show message for the trader
    - if zero match → returns ``status="not_found"``
    - if the resolved signal is illegal in the slot → returns
      ``status="wrong_slot"`` with placement error + suggestions

When the trader responds to a ``needs_choice`` prompt with a specific
candidate, the caller invokes this helper again with that full name; the
exact match path applies it.

The downstream pipeline does *not* need any change — once ``signal_plan``
is updated, the existing ``StrategyBuilder.to_yaml_dict`` writes the new
``entry_signals`` / ``exit_signals`` block into the strategy YAML and the
quant engine evaluates the modified signal set against the OHLCV candles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.kb import kb
from app.kb.signal_resolver import (
    SignalResolution,
    normalize_signal_query,
    resolve_signal_name,
)
from app.kb.signal_validation import (
    SignalPlacementError,
    Slot,
    validate_signal_placement,
)
from app.services.strategy.builder import StrategyBuilder


ApplyStatus = Literal["applied", "needs_choice", "not_found", "wrong_slot"]


@dataclass(frozen=True)
class SignalApplyResult:
    """Outcome of :func:`apply_signal_request`.

    Inspect ``status`` first; ``message`` is a ready-to-show user-facing
    string for every status.
    """

    status: ApplyStatus
    slot: Slot
    query: str
    message: str
    resolved_name: str | None = None
    candidates: tuple[str, ...] = ()
    placement_error: SignalPlacementError | None = None
    resolution: SignalResolution | None = None


def _apply_verb_and_clause(
    action: str, replace_name: str | None
) -> tuple[str, str]:
    """Produce the user-facing verb + parenthetical for the success message."""
    if replace_name:
        return "Replaced", f" (replaced '{replace_name}')"
    if action == "add":
        return "Added", ""
    return "Set", ""  # "replace" without a specific target == wipe-and-set


def apply_signal_request(
    builder: StrategyBuilder,
    slot: Slot,
    query: str,
    *,
    params: dict | None = None,
    replace_name: str | None = None,
    action: str = "replace",
    max_candidates: int = 4,
) -> SignalApplyResult:
    """Resolve, validate, and (when unambiguous) apply a signal to the plan.

    Args:
        builder: the active :class:`StrategyBuilder` for the session.
        slot: ``"entry"`` or ``"exit"``.
        query: trader-typed signal name, possibly partial.
        params: optional params dict to pass through on apply.
        replace_name: if given, swap this existing signal in-place; otherwise
            ``action`` governs whether the slot is wiped or appended to.
        action: ``"replace"`` (default) wipes the slot and sets the new
            signal — the right behavior for "change entry signal to X".
            ``"add"`` appends without removing existing entries — for
            "add X to entry" phrasing.
        max_candidates: cap on ambiguous candidate list (default 4).

    Returns:
        :class:`SignalApplyResult`. The chat / API layer reads ``status``
        and surfaces ``message`` (plus ``candidates`` when applicable) to
        the trader.
    """
    # If the trader typed a complete, real signal name we honor it directly —
    # placement validation then decides whether it belongs in `slot`. This
    # avoids falling into fuzzy/family suggestions when the user's intent is
    # already unambiguous (e.g. they typed `ema_cross_up` in the exit slot;
    # we should say "this is entry-only", not list ema_* alternatives).
    # The query is normalized first so natural phrasing ("rsi above 60",
    # "MACD Negative") also lands on the canonical KB name when the user
    # spelled out the full signal in words.
    raw_name = query.strip()
    normalized_name = normalize_signal_query(raw_name)
    direct_name = normalized_name if normalized_name in kb.signals else raw_name
    direct_hit = kb.signals.get(direct_name) if direct_name else None
    if direct_hit is not None:
        placement_err = validate_signal_placement(direct_name, slot)
        if placement_err is None:
            builder.modify_signal(
                slot=slot,
                signal_name=direct_name,
                params=params,
                replace_name=replace_name,
                action=action,
            )
            verb, swap_clause = _apply_verb_and_clause(action, replace_name)
            return SignalApplyResult(
                status="applied",
                slot=slot,
                query=query,
                resolved_name=direct_name,
                message=f"{verb} {slot} signal '{direct_name}'{swap_clause}.",
                candidates=(direct_name,),
            )
        return SignalApplyResult(
            status="wrong_slot",
            slot=slot,
            query=query,
            resolved_name=direct_name,
            message=placement_err.message,
            candidates=placement_err.suggestions[:max_candidates],
            placement_error=placement_err,
        )

    resolution = resolve_signal_name(query, slot=slot, max_candidates=max_candidates)

    if resolution.state == "none":
        return SignalApplyResult(
            status="not_found",
            slot=slot,
            query=query,
            message=resolution.message,
            resolution=resolution,
        )

    if resolution.state == "ambiguous":
        return SignalApplyResult(
            status="needs_choice",
            slot=slot,
            query=query,
            message=resolution.message,
            candidates=resolution.candidates,
            resolution=resolution,
        )

    # Exact match from fuzzy path — re-check placement defensively.
    name = resolution.exact_match or ""
    placement_err = validate_signal_placement(name, slot)
    if placement_err is not None:
        return SignalApplyResult(
            status="wrong_slot",
            slot=slot,
            query=query,
            resolved_name=name,
            message=placement_err.message,
            candidates=placement_err.suggestions[:max_candidates],
            placement_error=placement_err,
            resolution=resolution,
        )

    builder.modify_signal(
        slot=slot,
        signal_name=name,
        params=params,
        replace_name=replace_name,
        action=action,
    )
    verb, swap_clause = _apply_verb_and_clause(action, replace_name)
    return SignalApplyResult(
        status="applied",
        slot=slot,
        query=query,
        resolved_name=name,
        message=f"{verb} {slot} signal '{name}'{swap_clause}.",
        candidates=(name,),
        resolution=resolution,
    )
