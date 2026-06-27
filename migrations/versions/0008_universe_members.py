"""Dynamic-universe Phase D: live-runtime universe_members table

Revision ID: d9e0f1a2b3c4
Revises: c8d9e0f1a2b3
Create Date: 2026-06-18 00:30:00.000000

Adds ``ai_strategy.universe_members`` — the RUNNING membership state of a live dynamic
deployment (§9): which symbols are active / retiring / retired, since when, with how much
capital allocated, and which snapshot admitted them. Read on restart to reconcile before
resuming (§9 step 7).

Additive, idempotent (table_exists guard), reversible, tenant-scoped, ai_strategy schema.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

from migrations.helpers import table_exists

revision: str = "d9e0f1a2b3c4"
down_revision: Union[str, None] = "c8d9e0f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SCHEMA = "ai_strategy"
_TABLE = "universe_members"


def upgrade() -> None:
    if table_exists(_TABLE, schema=_SCHEMA):
        return
    op.create_table(
        _TABLE,
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("deployment_id", UUID(as_uuid=True), nullable=False,
                  comment="The live dynamic deployment this member belongs to."),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="active",
                  comment="active | retiring | retired"),
        sa.Column("member_state", JSONB(), nullable=True,
                  comment="Per-member runtime sub-state (warm-up done, last bar, …)."),
        sa.Column("allocated_capital", sa.Numeric(18, 4), nullable=False, server_default="0"),
        sa.Column("activated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("snapshot_hash", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("deployment_id", "symbol",
                            name="uq_universe_members_deployment_symbol"),
        schema=_SCHEMA,
    )
    op.create_index("ix_universe_members_active", _TABLE,
                    ["deployment_id", "status"], schema=_SCHEMA)


def downgrade() -> None:
    if not table_exists(_TABLE, schema=_SCHEMA):
        return
    op.drop_index("ix_universe_members_active", table_name=_TABLE, schema=_SCHEMA)
    op.drop_table(_TABLE, schema=_SCHEMA)
