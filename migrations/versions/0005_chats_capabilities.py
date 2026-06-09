"""Add capabilities column to ai_strategy.chats

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-26 10:45:00.000000

Adds a nullable JSONB `capabilities` column on `ai_strategy.chats` so that
each chat session can record which asset classes the requesting tenant
has enabled (e.g. {"asset_classes": [{"asset_class_id": "crypto_spot",
"enabled": true}, {"asset_class_id": "equity_cash", "enabled": true}]}).

The column is nullable so existing rows continue to work without a backfill.
Defaults are applied at the application layer.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chats",
        sa.Column("capabilities", JSONB(), nullable=True),
        schema="ai_strategy",
    )


def downgrade() -> None:
    op.drop_column("chats", "capabilities", schema="ai_strategy")
