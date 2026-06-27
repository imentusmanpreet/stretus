"""Dynamic-universe Phase D: per-resolution audit table universe_snapshots

Revision ID: e0f1a2b3c4d5
Revises: d9e0f1a2b3c4
Create Date: 2026-06-25 00:00:00.000000

Adds ``ai_strategy.universe_snapshots`` — the append-only AUDIT trail of a live dynamic
deployment's refresh ticks (§9/§14). Each row records the ``asof`` instant, the resolved
member symbols, the content ``snapshot_hash`` that admitted them, and the resolver funnel
counts (pool → screened → eligible → members) plus the cap / survivorship transparency. This
is the immutable "why does the membership look like this" record, distinct from the mutable
``universe_members`` runtime-state table (0008).

Additive, idempotent (table_exists guard), reversible, tenant-scoped, ai_strategy schema.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

from migrations.helpers import table_exists

revision: str = "e0f1a2b3c4d5"
down_revision: Union[str, None] = "d9e0f1a2b3c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SCHEMA = "ai_strategy"
_TABLE = "universe_snapshots"


def upgrade() -> None:
    if table_exists(_TABLE, schema=_SCHEMA):
        return
    op.create_table(
        _TABLE,
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("deployment_id", UUID(as_uuid=True), nullable=False,
                  comment="The live dynamic deployment this resolution belongs to."),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("asof", sa.DateTime(timezone=True), nullable=False,
                  comment="Point-in-time instant the universe was resolved at (UTC)."),
        sa.Column("members", JSONB(), nullable=False, server_default="[]",
                  comment="Resolved member symbols in rank order."),
        sa.Column("snapshot_hash", sa.Text(), nullable=True),
        sa.Column("rule_hash", sa.Text(), nullable=True),
        sa.Column("funnel", JSONB(), nullable=True,
                  comment="Resolver funnel counts + breadth flags (pool→screened→eligible→members)."),
        sa.Column("cap_reason", sa.Text(), nullable=True),
        sa.Column("survivorship_mode", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("deployment_id", "asof", "snapshot_hash",
                            name="uq_universe_snapshots_deployment_asof_hash"),
        schema=_SCHEMA,
    )
    op.create_index("ix_universe_snapshots_deployment_asof", _TABLE,
                    ["deployment_id", "asof"], schema=_SCHEMA)


def downgrade() -> None:
    if not table_exists(_TABLE, schema=_SCHEMA):
        return
    op.drop_index("ix_universe_snapshots_deployment_asof", table_name=_TABLE, schema=_SCHEMA)
    op.drop_table(_TABLE, schema=_SCHEMA)
