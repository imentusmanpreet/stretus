"""
signal_suggestions.py
---------------------
Returns randomised entry / exit signal suggestions seeded by session_id so
that every session sees a fresh, different set while the same session always
gets the same examples (deterministic replay).
"""

from __future__ import annotations

import hashlib
import random
from typing import NamedTuple

from app.kb import kb
from app.kb.schemas import SignalCard


# ── Helpers ──────────────────────────────────────────────────────────────────


def _trigger_signals_for_slot(slot: str) -> list[SignalCard]:
    """Return all SignalCards whose roles include the trigger for *slot*."""
    role = "entry_trigger" if slot == "entry" else "exit_trigger"
    return [card for card in kb.signals.values() if role in card.roles]


def _seed_from_session(session_id: str) -> int:
    """Derive a stable integer seed from the session UUID."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ── Public API ────────────────────────────────────────────────────────────────


class SignalSuggestions(NamedTuple):
    entry: list[str]   # canonical signal names
    exit: list[str]    # canonical signal names


def get_signal_suggestions(session_id: str, n: int = 5) -> SignalSuggestions:
    """
    Pick *n* entry and *n* exit signal examples, seeded by *session_id*.

    Different sessions → different examples.
    Same session → same examples (repeatable on refresh).
    """
    rng = random.Random(_seed_from_session(session_id))

    entry_pool = _trigger_signals_for_slot("entry")
    exit_pool  = _trigger_signals_for_slot("exit")

    entry_sample = rng.sample(entry_pool, min(n, len(entry_pool)))
    exit_sample  = rng.sample(exit_pool,  min(n, len(exit_pool)))

    return SignalSuggestions(
        entry=[c.name for c in entry_sample],
        exit=[c.name  for c in exit_sample],
    )


def format_signal_suggestions(session_id: str, n: int = 5) -> str:
    """
    Build a trader-facing plain-text message listing entry and exit
    suggestions — no markdown formatting (no backticks, bold, or bullets).
    """
    suggestions = get_signal_suggestions(session_id, n=n)

    def _line(name: str) -> str:
        card = kb.signals.get(name)
        desc = card.description if card else ""
        if len(desc) > 60:
            desc = desc[:57].rstrip() + "..."
        return f"{name}: {desc}" if desc else name

    entry_lines = "\n".join(_line(n) for n in suggestions.entry)
    exit_lines  = "\n".join(_line(n) for n in suggestions.exit)

    return (
        "I need to know which signal you'd like to use. "
        "Here are some options to choose from.\n\n"
        f"Entry signals (pick one):\n{entry_lines}\n\n"
        f"Exit signals (pick one):\n{exit_lines}\n\n"
        "Say something like: change entry to (signal_name), "
        "or use (signal_name) for exit."
    )
