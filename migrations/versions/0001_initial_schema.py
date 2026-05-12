"""Initial schema: strategy schema, enums, and all tables

Revision ID: e1a2b3c4d5e6
Revises:
Create Date: 2026-04-17 00:00:00.000000

This is the single authoritative migration that creates the entire
strategy schema from scratch. It replaces the old init_db.sql + stamp
approach with proper Alembic-managed DDL.

For databases that were already provisioned via init_db.sql:
    alembic stamp e1a2b3c4d5e6
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e1a2b3c4d5e6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_enum(name: str, *values: str) -> None:
    op.execute(
        f"""
        DO $$ BEGIN
            CREATE TYPE ai_strategy.{name} AS ENUM ({', '.join(f"'{v}'" for v in values)});
        EXCEPTION WHEN duplicate_object THEN NULL; END $$
        """
    )


def _drop_enum(name: str) -> None:
    op.execute(f"DROP TYPE IF EXISTS ai_strategy.{name}")


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # Idempotency guard: if the schema objects already exist (e.g. created by a
    # previous deployment with different migration IDs), skip all DDL here.
    # Alembic will still record this revision in alembic_version so future
    # migrations apply correctly.
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'ai_strategy' AND table_name = 'strategies'"
        )
    )
    if result.scalar() > 0:
        return

    # The ai_strategy schema is created by env.py before any migration runs so
    # that Alembic can place its version table inside it.

    # ── ENUM types ──────────────────────────────────────────────────────────
    _create_enum("chat_role", "user", "assistant", "system")
    _create_enum(
        "strategy_status",
        "draft", "confirmed", "backtesting",
        "backtest_complete", "live", "paused", "archived",
    )
    _create_enum("backtest_status", "pending", "running", "completed", "failed")
    _create_enum("message_status", "pending", "processing", "completed", "failed")

    # ── Trigger function ─────────────────────────────────────────────────────
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ai_strategy.set_updated_at()
        RETURNS TRIGGER LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$
        """
    )

    # ── ai_strategy.chats ───────────────────────────────────────────────────────
    op.create_table(
        "chats",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        schema="ai_strategy",
    )
    op.create_index(
        "idx_chats_user_id",
        "chats",
        ["user_id"],
        schema="ai_strategy",
        postgresql_where=sa.text("user_id IS NOT NULL"),
    )
    op.execute(
        """
        CREATE TRIGGER trg_chats_updated_at
            BEFORE UPDATE ON ai_strategy.chats
            FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at()
        """
    )

    # ── ai_strategy.strategies ──────────────────────────────────────────────────
    op.create_table(
        "strategies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("market", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), server_default="1d", nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "draft", "confirmed", "backtesting", "backtest_complete",
                "live", "paused", "archived",
                name="strategy_status",
                schema="ai_strategy",
                create_type=False,
            ),
            server_default="draft",
            nullable=False,
        ),
        sa.Column("yaml_path", sa.Text(), nullable=True),
        sa.Column("strategy_config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["ai_strategy.chats.id"],
            name="strategies_session_id_fkey",
            ondelete="CASCADE",
        ),
        schema="ai_strategy",
    )
    op.create_index("idx_strategies_session_id", "strategies", ["session_id"], schema="ai_strategy")
    op.create_index("idx_strategies_status", "strategies", ["status"], schema="ai_strategy")
    op.create_index("idx_strategies_symbol", "strategies", ["symbol"], schema="ai_strategy")
    op.execute(
        """
        CREATE TRIGGER trg_strategies_updated_at
            BEFORE UPDATE ON ai_strategy.strategies
            FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at()
        """
    )

    # ── ai_strategy.backtest ────────────────────────────────────────────────────
    op.create_table(
        "backtest",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending", "running", "completed", "failed",
                name="backtest_status",
                schema="ai_strategy",
                create_type=False,
            ),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("result_json", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["ai_strategy.strategies.id"],
            name="backtest_strategy_id_fkey",
            ondelete="CASCADE",
        ),
        schema="ai_strategy",
    )
    op.create_index("idx_backtest_strategy_id", "backtest", ["strategy_id"], schema="ai_strategy")
    op.create_index("idx_backtest_status", "backtest", ["status"], schema="ai_strategy")

    # ── ai_strategy.chat_messages ───────────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM(
                "user", "assistant", "system",
                name="chat_role",
                schema="ai_strategy",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending", "processing", "completed", "failed",
                name="message_status",
                schema="ai_strategy",
                create_type=False,
            ),
            server_default="completed",
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("parent_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("is_final", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("strategy_draft", postgresql.JSONB(), nullable=True),
        sa.Column("backtest_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ref_backtest_id", sa.Text(), nullable=True),
        sa.Column(
            "strategy_json",
            postgresql.JSONB(),
            nullable=True,
            comment="Parsed strategy_json from assemble_strategy AI response (entry/exit signals, asset, parameters)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["ai_strategy.chats.id"],
            name="chat_messages_session_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["backtest_id"],
            ["ai_strategy.backtest.id"],
            name="chat_messages_backtest_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["ai_strategy.strategies.id"],
            name="chat_messages_strategy_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["parent_message_id"],
            ["ai_strategy.chat_messages.id"],
            name="chat_messages_parent_message_id_fkey",
            ondelete="SET NULL",
        ),
        schema="ai_strategy",
    )
    op.create_index(
        "idx_chat_messages_chat", "chat_messages", ["session_id", "created_at"], schema="ai_strategy"
    )
    op.create_index("idx_chat_messages_role", "chat_messages", ["role"], schema="ai_strategy")
    op.create_index("idx_chat_messages_status", "chat_messages", ["status"], schema="ai_strategy")
    op.create_index(
        "idx_chat_messages_backtest_id",
        "chat_messages",
        ["backtest_id"],
        schema="ai_strategy",
        postgresql_where=sa.text("backtest_id IS NOT NULL"),
    )
    op.create_index(
        "idx_chat_messages_strategy_id",
        "chat_messages",
        ["strategy_id"],
        schema="ai_strategy",
        postgresql_where=sa.text("strategy_id IS NOT NULL"),
    )
    op.create_index(
        "idx_chat_messages_ref_backtest_id",
        "chat_messages",
        ["ref_backtest_id"],
        schema="ai_strategy",
        postgresql_where=sa.text("ref_backtest_id IS NOT NULL"),
    )
    op.execute(
        """
        CREATE TRIGGER trg_chat_messages_updated_at
            BEFORE UPDATE ON ai_strategy.chat_messages
            FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at()
        """
    )


# ---------------------------------------------------------------------------
# Downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("chat_messages", schema="ai_strategy")
    op.drop_table("backtest", schema="ai_strategy")
    op.drop_table("strategies", schema="ai_strategy")
    op.drop_table("chats", schema="ai_strategy")

    op.execute("DROP FUNCTION IF EXISTS ai_strategy.set_updated_at() CASCADE")

    _drop_enum("message_status")
    _drop_enum("backtest_status")
    _drop_enum("strategy_status")
    _drop_enum("chat_role")

    op.execute("DROP SCHEMA IF EXISTS ai_strategy CASCADE")
