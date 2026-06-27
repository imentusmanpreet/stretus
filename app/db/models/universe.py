"""
app/db/models/universe.py
─────────────────────────
ORM model for point-in-time universe constituents (dynamic-universe Phase C).

``ai_strategy.universe_membership`` records WHICH symbols belonged to a named universe
(an index / sector / F&O set) over WHICH date interval — the data that makes a dynamic
backtest **survivorship-safe** (Invariant 7). A historical resolution reads the membership
as it stood at ``asof``, so a since-delisted name correctly appears in a 2021 backtest and is
correctly absent today.

Membership rule — "the members of ``universe_key`` on date D":
    ``valid_from <= D AND (valid_to IS NULL OR valid_to > D)``
``valid_to IS NULL`` ⇒ still a current member. Half-open intervals ``[valid_from, valid_to)``
so a name that leaves on date X is a member up to but not including X.

KB-free: this table REPLACES the ``kb.stocks`` coupling (Invariant 11). It is populated by an
ingestion job (constituent history), never derived from the knowledge base.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Numeric, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UniverseMembership(Base):
    """One point-in-time membership interval of ``symbol`` in ``universe_key``.

    ``universe_key`` is the source identifier the resolver maps a ``UniverseSource`` to
    (e.g. ``"NIFTY500"`` for an index, ``"banking"`` for a sector, ``"f_and_o"`` for the F&O
    set). ``tenant_id`` is nullable for forward-compatible per-tenant universes (§15); a NULL
    tenant means a platform-shared universe.
    """

    __tablename__ = "universe_membership"
    __table_args__ = (
        # One interval row per (tenant, universe, symbol, valid_from) — re-ingesting the
        # same constituent snapshot is a no-op (idempotent backfill, §4.4).
        UniqueConstraint(
            "tenant_id", "universe_key", "symbol", "valid_from",
            name="uq_universe_membership_interval",
        ),
        # Point-in-time lookups filter by universe_key then the validity window.
        Index("ix_universe_membership_lookup", "universe_key", "valid_from", "valid_to"),
        {"schema": "ai_strategy"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    universe_key: Mapped[str] = mapped_column(Text, nullable=False)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Provenance of this interval (e.g. the ingestion source/vendor); informational.
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )


class UniverseMember(Base):
    """Live-runtime membership of a symbol in a dynamic deployment (Phase D, §9).

    Where ``UniverseMembership`` is *historical* constituents (for survivorship-safe
    backtests), this table is the *running* state of a LIVE dynamic deployment: which
    symbols are currently active, retiring, or retired, and since when. The Universe
    Scheduler writes it on each refresh (spawn/retire), and reconciliation reads it on
    restart to rebuild active membership before resuming (§9 step 7).

    ``status`` lifecycle:  ``active`` → ``retiring`` (dropped from the universe, squaring
    off) → ``retired`` (flat). ``snapshot_hash`` ties a member to the resolution that
    admitted it (auditability, §14).
    """

    __tablename__ = "universe_members"
    __table_args__ = (
        # One live row per (deployment, symbol) — the scheduler upserts state transitions.
        UniqueConstraint("deployment_id", "symbol", name="uq_universe_members_deployment_symbol"),
        Index("ix_universe_members_active", "deployment_id", "status"),
        {"schema": "ai_strategy"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    deployment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    # active | retiring | retired
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    # Free-form per-member runtime sub-state (warm-up done, last bar seen, …).
    member_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Capital currently allocated to this member's open position (0 when flat).
    allocated_capital: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0.0)
    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    snapshot_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow, onupdate=_utcnow
    )


class UniverseSnapshot(Base):
    """One per-resolution AUDIT row for a live dynamic deployment (Phase D, §9/§14).

    Every scheduler refresh tick that actually resolves the universe writes a row here: the
    ``asof`` instant, the resolved member symbols, the content ``snapshot_hash`` that ties the
    resolution to the admitted members, and the resolver funnel counts (pool → screened →
    eligible → members). This is the immutable, replayable record of WHY the running membership
    looks the way it does (auditability + reproducibility, §8/§14) — distinct from
    ``universe_members`` which holds only the CURRENT runtime state.

    Append-only: a new row per refresh. ``cap_reason`` / ``breadth_ok`` preserve the
    transparency the resolver already logs (the platform cap clamped the take; a breadth gate
    admitted nobody) so an audit never has to re-run the resolver to explain a tick.
    """

    __tablename__ = "universe_snapshots"
    __table_args__ = (
        # Replay a deployment's history in time order; one row per (deployment, asof, hash).
        UniqueConstraint(
            "deployment_id", "asof", "snapshot_hash",
            name="uq_universe_snapshots_deployment_asof_hash",
        ),
        Index("ix_universe_snapshots_deployment_asof", "deployment_id", "asof"),
        {"schema": "ai_strategy"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    deployment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    # The point-in-time instant the universe was resolved at (ISO-8601 UTC on the resolver).
    asof: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Resolved member symbols, in rank order (the snapshot's membership).
    members: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    snapshot_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Resolver funnel counts + transparency flags (pool→screened→eligible→members, cap, breadth).
    funnel: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    cap_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    survivorship_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=_utcnow
    )
