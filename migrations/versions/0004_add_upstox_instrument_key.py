"""Add upstox_instrument_key column to instrument_metadata

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-20 10:00:00.000000

Adds upstox_instrument_key (nullable Text) to ai_strategy.instrument_metadata.
This stores the Upstox v2 instrument key (e.g. NSE_EQ|INE002A01018) used
by the execution service for live LTP and circuit limit calls.

If NULL, the UpstoxClient constructs a best-effort key as NSE_EQ|SYMBOL.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'ai_strategy' "
            "  AND table_name   = 'instrument_metadata' "
            "  AND column_name  = 'upstox_instrument_key'"
        )
    ).scalar()
    if exists:
        return

    op.add_column(
        "instrument_metadata",
        sa.Column(
            "upstox_instrument_key",
            sa.Text(),
            nullable=True,
            comment="Upstox v2 key e.g. NSE_EQ|INE002A01018. Used for live LTP/circuit calls.",
        ),
        schema="ai_strategy",
    )


def downgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'ai_strategy' "
            "  AND table_name   = 'instrument_metadata' "
            "  AND column_name  = 'upstox_instrument_key'"
        )
    ).scalar()
    if not exists:
        return

    op.drop_column("instrument_metadata", "upstox_instrument_key", schema="ai_strategy")
