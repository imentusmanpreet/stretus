"""
app/services/universe/errors.py — typed, fail-closed exceptions for the universe path.

Errors are explicit and never silently admit on uncertainty (§12.4). Each carries a
clear, actionable message intended to surface to the user/operator, not just a stack trace.
"""
from __future__ import annotations


class UniverseError(Exception):
    """Base class for all dynamic-universe errors."""


class UniverseSourceError(UniverseError):
    """The candidate pool could not be resolved from the source (e.g. an unknown index,
    an unsupported source kind in this phase, or a registry/feed failure).

    Distinct from "no candidates matched the screen": a source error means we could not
    even build the pool, so the caller must surface the root cause (Invariant: surface
    the real reason, never a misleading 'no matches')."""


class ScaleGuardExceeded(UniverseError):
    """A run's projected working set would breach the configured memory/time budget (§4.5).

    Raised BEFORE doing the expensive work, with an actionable message (which limit was
    hit, by how much, and how to bring the run within budget) — never an OOM."""
