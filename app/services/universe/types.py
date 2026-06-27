"""
app/services/universe/types.py — value objects for universe resolution.

Internal, in-flight value objects are **frozen dataclasses** (cheap, immutable, trivially
testable — mirroring ``scanner.py``'s ``Candidate``). The persisted/replayable snapshot
shape is a **Pydantic v2 model** with ``extra='forbid'`` (matching ``spec.py``), because it
crosses a boundary: it is hashed, stored, and returned in API payloads (§12.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

# Why a symbol was dropped from a resolution (stable, machine-usable codes).
DropReason = Literal[
    "fetch_error",        # OHLCV fetch raised
    "data_gap",           # no usable frame returned
    "insufficient_data",  # too few bars to evaluate screen/metrics
    "screen_failed",      # failed one of the screen conditions
    "ineligible_adv",     # below the average-daily-value floor (fail-closed)
    "ineligible_other",   # failed an injected tradability/circuit check (fail-closed)
    "rank_unavailable",   # ranking metric could not be computed (e.g. unwired feed)
    "breadth_blocked",    # admitted-then-blocked by the breadth gate
]

SurvivorshipMode = Literal["approximate", "point_in_time"]
CapReason = Literal["platform_limit", "none"]


@dataclass(frozen=True)
class Candidate:
    """One pool symbol with its pre-computed screening/ranking metrics.

    ``metrics`` carries everything the screen, eligibility and rank stages need so they
    are pure O(N) passes with no further I/O (mirrors ``scanner.py``).
    """

    symbol: str
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Dropped:
    """A symbol excluded during resolution, with the reason (transparency, §12.1)."""

    symbol: str
    reason: DropReason


class ResolvedMember(BaseModel):
    """One member of a resolved universe — its rank and the metrics behind it."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    rank: int = Field(ge=1)               # 1 = top by the ranking metric
    rank_value: float                     # the ranking-metric value used
    metrics: dict[str, float] = Field(default_factory=dict)


class DroppedMember(BaseModel):
    """A dropped symbol in the persisted snapshot (serializable form of ``Dropped``)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    reason: str


class ResolvedUniverse(BaseModel):
    """A point-in-time membership snapshot — content-hashed, persisted, fully replayable.

    This is the resolver's public return type and the shape stored in ``universe_snapshots``
    (Phase B/C). Backward-compatible by construction: it is a NEW payload, additive to the
    existing single-symbol responses (Invariant 10).
    """

    model_config = ConfigDict(extra="forbid")

    asof: str                              # ISO-8601 UTC instant of the resolution
    members: list[ResolvedMember] = Field(default_factory=list)
    dropped: list[DroppedMember] = Field(default_factory=list)

    # Asset-count cap transparency (§7.1) — never silently dropped.
    requested_take: int
    effective_take: int
    cap_reason: CapReason = "none"

    # Pipeline counts (logged + surfaced for observability, §14).
    pool_size: int = 0
    screened_count: int = 0               # passed the screen
    eligible_count: int = 0               # passed eligibility (fail-closed)

    # Breadth gate verdict (fail-closed).
    breadth_ok: bool = True
    breadth_reason: str = ""

    survivorship_mode: SurvivorshipMode = "approximate"

    # Reproducibility (§8).
    rule_hash: str = ""
    snapshot_hash: str = ""

    @property
    def member_symbols(self) -> list[str]:
        """Just the symbols, in rank order."""
        return [m.symbol for m in self.members]

    def cap_message(self) -> str | None:
        """User-facing one-liner when the platform cap clamped the take (§7.1), else None."""
        if self.cap_reason != "platform_limit":
            return None
        return (
            f"Requested {self.requested_take}; platform limit applied → "
            f"using {self.effective_take}."
        )
