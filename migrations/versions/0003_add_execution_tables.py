"""Add execution tables: strategy_risk_config, instrument_metadata, execution_states

Revision ID: a1b2c3d4e5f6
Revises: f2b3c4d5e6f7
Create Date: 2026-04-17 12:00:00.000000

Adds three tables to the ai_strategy schema:

  strategy_risk_config  — 1-to-1 with strategies; stores all risk_and_execution
                          fields (stop_loss_pct, take_profit_pct, daily_loss_cap_pct,
                          per_trade_risk_pct, max_trades_per_day, risk_reward,
                          execution_mode, position_sizing, trading_window,
                          risk_validation).  Written on strategy confirmation,
                          read by the execution RiskManager.

  instrument_metadata   — per-symbol tick_size and lot_size for price/qty
                          normalisation.  Populated by the daily instrument
                          refresh job.

  execution_states      — per-strategy runtime state (capital, limits, cooldown,
                          OMS positions snapshot).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _updated_at_trigger(table: str) -> None:
    """Attach the existing set_updated_at() trigger to a new table."""
    op.execute(
        f"""
        CREATE TRIGGER trg_{table}_updated_at
            BEFORE UPDATE ON ai_strategy.{table}
            FOR EACH ROW EXECUTE FUNCTION ai_strategy.set_updated_at()
        """
    )


def _drop_updated_at_trigger(table: str) -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON ai_strategy.{table}"
    )


# ── Upgrade ────────────────────────────────────────────────────────────────────

def upgrade() -> None:
    # ── ai_strategy.strategy_risk_config ──────────────────────────────────────
    op.create_table(
        "strategy_risk_config",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=False),
        # SL / TP numeric pcts
        sa.Column("stop_loss_pct",     sa.Numeric(6, 3),  nullable=False, server_default="2.0"),
        sa.Column("take_profit_pct",   sa.Numeric(6, 3),  nullable=False, server_default="5.0"),
        # Risk sizing
        sa.Column("daily_loss_cap_pct", sa.Numeric(6, 3), nullable=False, server_default="3.0"),
        sa.Column("per_trade_risk_pct", sa.Numeric(6, 3), nullable=False, server_default="2.0"),
        sa.Column("max_trades_per_day", sa.Integer(),     nullable=False, server_default="2"),
        # Derived / display
        sa.Column("risk_reward",        sa.Text(),        nullable=True),
        # Execution policy
        sa.Column("execution_mode",  sa.Text(), nullable=False, server_default="'Backtest'"),
        sa.Column(
            "position_sizing",
            sa.Text(),
            nullable=False,
            server_default="'Risk based'",
        ),
        sa.Column(
            "trading_window",
            sa.Text(),
            nullable=False,
            server_default="'9:15 - 15:30'",
        ),
        sa.Column(
            "risk_validation",
            sa.Text(),
            nullable=False,
            server_default=(
                "'system risk guardials'"
            ),
        ),
        # Timestamps
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
            ["strategy_id"],
            ["ai_strategy.strategies.id"],
            name="risk_config_strategy_id_fkey",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("strategy_id", name="uq_risk_config_strategy_id"),
        schema="ai_strategy",
    )
    op.create_index(
        "idx_risk_config_strategy_id",
        "strategy_risk_config",
        ["strategy_id"],
        schema="ai_strategy",
    )
    _updated_at_trigger("strategy_risk_config")

    # ── ai_strategy.instrument_metadata ───────────────────────────────────────
    op.create_table(
        "instrument_metadata",
        sa.Column("symbol",         sa.Text(),              primary_key=True, nullable=False),
        sa.Column("tick_size",      sa.Numeric(10, 5),      nullable=False, server_default="0.05"),
        sa.Column("lot_size",       sa.Integer(),           nullable=False, server_default="1"),
        sa.Column("upper_circuit",  sa.Numeric(12, 4),      nullable=True),
        sa.Column("lower_circuit",  sa.Numeric(12, 4),      nullable=True),
        sa.Column(
            "upstox_instrument_key", sa.Text(), nullable=True,
            comment="Upstox v2 key e.g. NSE_EQ|INE002A01018. Used for live LTP/circuit calls.",
        ),
        sa.Column("last_refreshed", sa.DateTime(timezone=True), nullable=True),
        schema="ai_strategy",
    )

    # ── ai_strategy.execution_states ──────────────────────────────────────────
    op.create_table(
        "execution_states",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Capital / sizing defaults
        sa.Column("capital",               sa.Numeric(18, 4), nullable=False, server_default="100000"),
        sa.Column("max_risk_per_trade_pct", sa.Numeric(6, 3), nullable=False, server_default="2.0"),
        sa.Column("max_open_positions",    sa.Integer(),      nullable=False, server_default="3"),
        sa.Column("cash_reserve_pct",      sa.Numeric(6, 3), nullable=False, server_default="0.10"),
        sa.Column("cooldown_bars",         sa.Integer(),      nullable=False, server_default="5"),
        sa.Column("min_trade_value",       sa.Numeric(12, 4), nullable=False, server_default="500"),
        # Runtime counters
        sa.Column("bars_since_last_trade", sa.Integer(),      nullable=False, server_default="0"),
        # OMS snapshot
        sa.Column(
            "state_json",
            postgresql.JSONB(),
            nullable=True,
            comment="Latest open_positions snapshot from OMS",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["ai_strategy.strategies.id"],
            name="exec_state_strategy_id_fkey",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("strategy_id", name="uq_exec_state_strategy_id"),
        schema="ai_strategy",
    )
    op.create_index(
        "idx_exec_states_strategy_id",
        "execution_states",
        ["strategy_id"],
        schema="ai_strategy",
    )
    _updated_at_trigger("execution_states")


# ── Downgrade ──────────────────────────────────────────────────────────────────

def downgrade() -> None:
    _drop_updated_at_trigger("execution_states")
    op.drop_table("execution_states", schema="ai_strategy")

    op.drop_table("instrument_metadata", schema="ai_strategy")

    _drop_updated_at_trigger("strategy_risk_config")
    op.drop_table("strategy_risk_config", schema="ai_strategy")
