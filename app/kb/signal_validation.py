"""
app/kb/signal_validation.py — role-based placement checks for signal names.

Every :class:`~app.kb.schemas.SignalCard` declares which roles it may play
via the ``roles:`` field in its YAML card:

    - ``entry_trigger`` : primary entry signal
    - ``entry_filter``  : confirmation / regime filter on the entry side
    - ``exit_trigger``  : primary exit signal

Signals split into four trader-facing categories (by TRIGGER roles):

    * ``entry_only``  — has ``entry_trigger`` and no ``exit_trigger``
    * ``exit_only``   — has ``exit_trigger`` and no ``entry_trigger``
    * ``both``        — has both triggers
    * ``filter_only`` — no triggers at all (just ``entry_filter``)

## Two validation modes

``strict`` (default — used for trader-driven edits):
    Only triggers are accepted at the corresponding slot. Filters are not
    a valid standalone trigger anywhere.

        entry slot OK → category ``entry_only`` or ``both``
        exit slot  OK → category ``exit_only``  or ``both``

    Scenario A — entry-only in exit slot →
        "The signal you have mentioned is used only in entry signals.
         It cannot be used as an exit signal."

    Scenario B — exit-only in entry slot →
        "The signal you have mentioned is used only in exit signals.
         It cannot be used as an entry signal."

    Scenario D — filter_only in either slot →
        "This signal is strictly a filter/confirmation tool. It cannot be
         used as a standalone <slot> signal."

``lenient`` (used by ``StrategyBuilder.to_yaml_dict`` as a safety net):
    Entry slot also accepts ``entry_filter`` signals — this matches the
    catalog signal picker, which intentionally mixes triggers and filters
    into the same ``entry_signals`` list so the engine evaluates them as
    ``entry_trigger AND entry_filter1 AND …``. Without lenient mode, every
    planner-generated plan would fail the YAML gate.

Suggestions are *always* trigger-only — when the trader needs an entry
or exit replacement, listing filters would be misleading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.kb import kb


Slot = Literal["entry", "exit"]
Mode = Literal["strict", "lenient"]
Category = Literal["entry_only", "exit_only", "both", "filter_only"]

# Used by app.kb.signal_resolver — published names so the resolver can match
# the eligibility rules without duplicating them.
_ENTRY_OK_ROLES = frozenset({"entry_trigger", "entry_filter"})  # lenient
_EXIT_OK_ROLES = frozenset({"exit_trigger"})
_ENTRY_TRIGGER_ONLY = frozenset({"entry_trigger"})              # strict


# ── Public exception / error model ───────────────────────────────────────


@dataclass(frozen=True)
class SignalPlacementError:
    """One signal name that does not belong in the slot where the trader put it."""

    signal_name: str
    intended_slot: Slot
    actual_roles: tuple[str, ...]
    category: Category | None
    message: str
    suggestions: tuple[str, ...]

    @property
    def is_unknown(self) -> bool:
        return not self.actual_roles


class SignalPlacementValidationError(ValueError):
    """Raised when one or more signals are placed in the wrong slot."""

    def __init__(self, errors: list[SignalPlacementError]) -> None:
        self.errors: list[SignalPlacementError] = list(errors)
        super().__init__("; ".join(err.message for err in errors))


# ── Internal helpers ─────────────────────────────────────────────────────


def _category_of(card_roles: list[str] | tuple[str, ...]) -> Category:
    """Categorize a signal by its TRIGGER roles."""
    has_entry = "entry_trigger" in card_roles
    has_exit = "exit_trigger" in card_roles
    if has_entry and has_exit:
        return "both"
    if has_entry:
        return "entry_only"
    if has_exit:
        return "exit_only"
    return "filter_only"


def _trigger_names_for_slot(slot: Slot) -> list[str]:
    """All signal names that can act as a TRIGGER in ``slot``."""
    role = "entry_trigger" if slot == "entry" else "exit_trigger"
    return sorted(
        name for name, card in kb.signals.items()
        if role in card.roles
    )


def _suggest_for_slot(name: str, slot: Slot) -> tuple[str, ...]:
    """Trigger-only suggestions for ``slot``, with same-family names first."""
    candidates = _trigger_names_for_slot(slot)
    if not name:
        return tuple(candidates)
    family = name.split("_", 1)[0].lower()
    family_first = [n for n in candidates if n.split("_", 1)[0].lower() == family]
    others = [n for n in candidates if n not in family_first]
    return tuple(family_first + others)


# ── Public API ───────────────────────────────────────────────────────────


def validate_signal_placement(
    name: str,
    slot: Slot,
    *,
    mode: Mode = "strict",
) -> SignalPlacementError | None:
    """Return ``None`` if ``name`` is allowed in ``slot``, else a structured error.

    Args:
        name: KB signal name as typed / chosen by the trader.
        slot: ``"entry"`` or ``"exit"``.
        mode: ``"strict"`` (trader-driven UX rules — Scenarios A/B/C/D) or
            ``"lenient"`` (allow ``entry_filter`` in the entry slot to match
            planner output; still rejects wrong-trigger placements).

    The returned error carries:
        - ``message``      → ready-to-show text
        - ``suggestions``  → trigger-only alternatives, family-first
        - ``category``     → ``entry_only`` / ``exit_only`` / ``both`` / ``filter_only``
    """
    if slot not in ("entry", "exit"):
        raise ValueError(f"Unknown slot {slot!r}; expected 'entry' or 'exit'.")
    if mode not in ("strict", "lenient"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'strict' or 'lenient'.")

    clean = str(name or "").strip()
    if not clean:
        return None

    card = kb.signals.get(clean)
    if card is None:
        return SignalPlacementError(
            signal_name=clean,
            intended_slot=slot,
            actual_roles=(),
            category=None,
            message=(
                f"The signal '{clean}' is not a known signal. "
                f"Below are the valid {slot} signals — please select one of them."
            ),
            suggestions=_suggest_for_slot(clean, slot),
        )

    category = _category_of(card.roles)
    roles = tuple(card.roles)

    # ── Strict mode: trader-UX scenarios A / B / C / D ───────────────────
    if mode == "strict":
        if category == "filter_only":
            return SignalPlacementError(
                signal_name=clean,
                intended_slot=slot,
                actual_roles=roles,
                category=category,
                message=(
                    f"The signal '{clean}' is strictly a filter/confirmation "
                    f"tool. It cannot be used as a standalone {slot} signal. "
                    f"Below are the valid {slot} signals — please select one of them."
                ),
                suggestions=_suggest_for_slot(clean, slot),
            )
        if slot == "entry" and category == "exit_only":
            return SignalPlacementError(
                signal_name=clean,
                intended_slot=slot,
                actual_roles=roles,
                category=category,
                message=(
                    "The signal you have mentioned is used only in exit signals. "
                    "It cannot be used as an entry signal. "
                    "Below are the valid entry signals — please select one of them."
                ),
                suggestions=_suggest_for_slot(clean, slot),
            )
        if slot == "exit" and category == "entry_only":
            return SignalPlacementError(
                signal_name=clean,
                intended_slot=slot,
                actual_roles=roles,
                category=category,
                message=(
                    "The signal you have mentioned is used only in entry signals. "
                    "It cannot be used as an exit signal. "
                    "Below are the valid exit signals — please select one of them."
                ),
                suggestions=_suggest_for_slot(clean, slot),
            )
        # entry+entry, exit+exit, or 'both' → OK
        return None

    # ── Lenient mode: planner-friendly (entry_filter allowed on entry) ───
    if slot == "entry":
        if "entry_trigger" in card.roles or "entry_filter" in card.roles:
            return None
        return SignalPlacementError(
            signal_name=clean,
            intended_slot=slot,
            actual_roles=roles,
            category=category,
            message=(
                "The signal you have mentioned is used only in exit signals. "
                "It cannot be used as an entry signal. "
                "Below are the valid entry signals — please select one of them."
            ),
            suggestions=_suggest_for_slot(clean, slot),
        )

    # slot == "exit"
    # Symmetric with the entry branch above: in lenient (planner) mode an exit
    # FILTER is a legitimate occupant of the exit list (the ranker picks exit
    # filters alongside the exit trigger), so allow exit_filter too. Strict mode
    # (handled earlier) still rejects filter-only cards — that path is for a
    # user explicitly naming a single signal as "the exit".
    if "exit_trigger" in card.roles or "exit_filter" in card.roles:
        return None
    if category == "filter_only":
        return SignalPlacementError(
            signal_name=clean,
            intended_slot=slot,
            actual_roles=roles,
            category=category,
            message=(
                f"The signal '{clean}' is strictly a filter/confirmation "
                f"tool. It cannot be used as a standalone exit signal. "
                f"Below are the valid exit signals — please select one of them."
            ),
            suggestions=_suggest_for_slot(clean, slot),
        )
    return SignalPlacementError(
        signal_name=clean,
        intended_slot=slot,
        actual_roles=roles,
        category=category,
        message=(
            "The signal you have mentioned is used only in entry signals. "
            "It cannot be used as an exit signal. "
            "Below are the valid exit signals — please select one of them."
        ),
        suggestions=_suggest_for_slot(clean, slot),
    )


def validate_signal_lists(
    entry_names: list[str] | tuple[str, ...] | None,
    exit_names: list[str] | tuple[str, ...] | None,
    *,
    mode: Mode = "strict",
) -> list[SignalPlacementError]:
    """Batch validation. Returns one error per misplaced signal."""
    errors: list[SignalPlacementError] = []
    for n in entry_names or ():
        err = validate_signal_placement(n, "entry", mode=mode)
        if err is not None:
            errors.append(err)
    for n in exit_names or ():
        err = validate_signal_placement(n, "exit", mode=mode)
        if err is not None:
            errors.append(err)
    return errors


def assert_valid_signal_lists(
    entry_names: list[str] | tuple[str, ...] | None,
    exit_names: list[str] | tuple[str, ...] | None,
    *,
    mode: Mode = "strict",
) -> None:
    """Raise :class:`SignalPlacementValidationError` if any name is misplaced."""
    errors = validate_signal_lists(entry_names, exit_names, mode=mode)
    if errors:
        raise SignalPlacementValidationError(errors)


def signal_category(name: str) -> Category | None:
    """Public lookup: trader-facing category of a known signal, or ``None``."""
    card = kb.signals.get(str(name or "").strip())
    if card is None:
        return None
    return _category_of(card.roles)
