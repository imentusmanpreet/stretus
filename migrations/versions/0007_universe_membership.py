"""Dynamic-universe Phase C: point-in-time universe_membership table

Revision ID: c8d9e0f1a2b3
Revises: a7b8c9d0e1f2
Create Date: 2026-06-18 00:00:00.000000

Adds ``ai_strategy.universe_membership`` — the survivorship-safe record of which symbols
belonged to a named universe (index / sector / F&O set) over which date interval (§6).
Members on date D = ``valid_from <= D AND (valid_to IS NULL OR valid_to > D)``.

Additive, idempotent (guarded by table_exists), reversible, tenant-scoped, ai_strategy schema.
KB-free — replaces the kb.stocks coupling for the dynamic path (Invariant 11).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

from migrations.helpers import table_exists

revision: str = "c8d9e0f1a2b3"
down_revision: Union[str, None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SCHEMA = "ai_strategy"
_TABLE = "universe_membership"


def upgrade() -> None:
    if table_exists(_TABLE, schema=_SCHEMA):
        return

    op.create_table(
        _TABLE,
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=True,
                  comment="Owning tenant; NULL = platform-shared universe (§15)."),
        sa.Column("universe_key", sa.Text(), nullable=False,
                  comment="Source identifier: index name (NIFTY500), sector (banking), "
                          "or set key (f_and_o)."),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False,
                  comment="Interval start (inclusive)."),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True,
                  comment="Interval end (exclusive); NULL = still a current member."),
        sa.Column("source", sa.Text(), nullable=True,
                  comment="Provenance of this interval (ingestion vendor/source)."),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("tenant_id", "universe_key", "symbol", "valid_from",
                            name="uq_universe_membership_interval"),
        schema=_SCHEMA,
    )
    op.create_index(
        "ix_universe_membership_lookup", _TABLE,
        ["universe_key", "valid_from", "valid_to"], schema=_SCHEMA,
    )


def downgrade() -> None:
    if not table_exists(_TABLE, schema=_SCHEMA):
        return
    op.drop_index("ix_universe_membership_lookup", table_name=_TABLE, schema=_SCHEMA)
    op.drop_table(_TABLE, schema=_SCHEMA)
