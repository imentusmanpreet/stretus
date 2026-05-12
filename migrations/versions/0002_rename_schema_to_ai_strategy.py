"""Rename schema: strategy → ai_strategy

Revision ID: f2b3c4d5e6f7
Revises: e1a2b3c4d5e6
Create Date: 2026-04-17 01:00:00.000000

Renames the PostgreSQL schema from 'strategy' to 'ai_strategy'.
PostgreSQL's ALTER SCHEMA … RENAME TO moves all contained objects
(tables, enums, functions, triggers, sequences) automatically.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "f2b3c4d5e6f7"
down_revision: Union[str, None] = "e1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    import sqlalchemy as sa
    conn = op.get_bind()

    has_old = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.schemata "
            "WHERE schema_name = 'strategy'"
        )
    ).scalar()

    has_new = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.schemata "
            "WHERE schema_name = 'ai_strategy'"
        )
    ).scalar()

    if not has_old:
        # Nothing to rename — schema was never called 'strategy'.
        return

    if has_new:
        # ai_strategy already exists (e.g. created by env.py or a prior run).
        # The rename is a no-op; nothing to do.
        return

    op.execute("ALTER SCHEMA strategy RENAME TO ai_strategy")


def downgrade() -> None:
    import sqlalchemy as sa
    conn = op.get_bind()

    has_ai = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.schemata "
            "WHERE schema_name = 'ai_strategy'"
        )
    ).scalar()

    has_old = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.schemata "
            "WHERE schema_name = 'strategy'"
        )
    ).scalar()

    if not has_ai or has_old:
        return

    op.execute("ALTER SCHEMA ai_strategy RENAME TO strategy")
