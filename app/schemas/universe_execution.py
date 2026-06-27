"""
app/schemas/universe_execution.py — portfolio-level response for a dynamic-universe tick.

The static path returns one :class:`EvaluateExecuteResponse` per symbol. A dynamic deployment
evaluates MANY members on one tick and must answer as a single portfolio: which members opened
(after the shared-capital + ``max_positions`` gate), which were skipped and WHY, and which
positions exited. This schema is that aggregate — additive, never altering the per-symbol
response shape (Invariant 10). Each instruction keeps its owning ``symbol`` so the OMS can
attribute every order.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.execution import (
    ActionType,
    EvaluateExecuteResponse,
    EvaluationMode,
    ExitInstruction,
)


class MemberOutcome(BaseModel):
    """One member's verdict on this tick, with its full per-symbol response attached."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    action: ActionType
    admitted: Optional[bool] = Field(
        None,
        description="Entry only: True if the portfolio layer admitted it, False if skipped. "
        "None for exits / no_action (admission not applicable).",
    )
    skip_reason: Optional[str] = Field(
        None, description="Why an entry was skipped by the portfolio layer (e.g. position_cap)."
    )
    allocated_capital: float = 0.0
    response: EvaluateExecuteResponse


class UniverseEvaluateResponse(BaseModel):
    """The portfolio-level outcome of one dynamic-universe evaluation tick."""

    model_config = ConfigDict(extra="forbid")

    status: str = "success"
    deployment_id: str
    mode: EvaluationMode = EvaluationMode.paper
    max_positions: int
    members_evaluated: int = 0
    open_positions_after: int = 0

    # Admitted entries, skipped entries, and exits — each member result carries its symbol.
    entries: List[MemberOutcome] = Field(default_factory=list)
    skipped: List[MemberOutcome] = Field(default_factory=list)
    # Exit instructions across all members, flattened (each already carries its symbol).
    exits: List[ExitInstruction] = Field(default_factory=list)
    # Every member's outcome (entries + skipped + exits + no_action), for full auditability.
    results: List[MemberOutcome] = Field(default_factory=list)

    messages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
