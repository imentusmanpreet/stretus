"""Add ref_data.risk_execution_configs table

Revision ID: d4e5f6a7b8c9
Revises: b2c3d4e5f6a7
Create Date: 2026-04-24 20:30:00.000000

Creates or normalizes the ref_data risk/execution config table into the
final vertical key/value shape:
  - keys
  - values
  - created_at
  - updated_at
"""
from __future__ import annotations

import re
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONFIG_FIELDS = (
    "max_trades",
    "risk_reward",
    "daily_loss_cap",
    "execution_mode",
    "per_trade_risk",
    "trading_window",
    "position_sizing",
    "risk_validation",
    "stop_loss_pct",
    "take_profit_pct",
    "minimum_trade_value",
)
_DEFAULT_SEED_VALUES = {
    "max_trades": "2.0",
    "risk_reward": "2.5",
    "daily_loss_cap": "3.0",
    "execution_mode": "Backtest",
    "per_trade_risk": "2.0",
    "trading_window": "9:15 - 15:30",
    "position_sizing": "Risk based",
    "risk_validation": "system risk guardials",
    "stop_loss_pct": "2.0",
    "take_profit_pct": "5.0",
    "minimum_trade_value": "500.0",
}
_STRING_FIELDS = {
    "execution_mode",
    "trading_window",
    "position_sizing",
    "risk_validation",
}
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _table_exists(conn: sa.engine.Connection, table_name: str) -> bool:
    return bool(
        conn.execute(
            sa.text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'ref_data'
                  AND table_name = :table_name
                """
            ),
            {"table_name": table_name},
        ).scalar()
    )


def _column_names(conn: sa.engine.Connection, table_name: str) -> list[str]:
    rows = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'ref_data'
              AND table_name = :table_name
            ORDER BY ordinal_position
            """
        ),
        {"table_name": table_name},
    ).fetchall()
    return [str(row[0]) for row in rows]


def _scope_prefix(config_scope: str, scope_id: str) -> str:
    if config_scope == "global":
        return ""
    return f"{config_scope}:{scope_id}"


def _storage_key(config_scope: str, scope_id: str, field_name: str) -> str:
    prefix = _scope_prefix(config_scope, scope_id)
    return field_name if not prefix else f"{prefix}.{field_name}"


def _serialize_vertical_value(field_name: str, value: object) -> str | None:
    if value is None:
        return None
    if field_name in _STRING_FIELDS:
        return str(value)

    match = _FLOAT_RE.search(str(value).strip())
    if not match:
        return None
    return str(float(match.group(0)))


def _create_vertical_table(table_name: str) -> None:
    op.create_table(
        table_name,
        sa.Column("keys", sa.Text(), nullable=False),
        sa.Column("values", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("keys", name=f"pk_{table_name}"),
        schema="ref_data",
    )


def _migrate_wide_rows_to_vertical(conn: sa.engine.Connection) -> None:
    temp_table_name = "risk_execution_configs_v2"
    op.execute(f"DROP TABLE IF EXISTS ref_data.{temp_table_name}")
    _create_vertical_table(temp_table_name)

    rows = conn.execute(
        sa.text(
            """
            SELECT
                config_scope,
                scope_id,
                max_trades,
                risk_reward,
                daily_loss_cap,
                execution_mode,
                per_trade_risk,
                trading_window,
                position_sizing,
                risk_validation,
                stop_loss_pct,
                take_profit_pct,
                minimum_trade_value,
                created_at,
                updated_at
            FROM ref_data.risk_execution_configs
            """
        )
    ).mappings().all()

    payloads: list[dict[str, object]] = []
    for row in rows:
        prefix = _scope_prefix(str(row["config_scope"]), str(row["scope_id"]))
        created_at = row["created_at"]
        updated_at = row["updated_at"]
        for field_name in _CONFIG_FIELDS:
            serialized_value = _serialize_vertical_value(field_name, row[field_name])
            if serialized_value is None:
                continue
            payloads.append(
                {
                    "key": _storage_key(str(row["config_scope"]), str(row["scope_id"]), field_name),
                    "value": serialized_value,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )

    if payloads:
        conn.execute(
            sa.text(
                """
                INSERT INTO ref_data.risk_execution_configs_v2 (
                    "keys",
                    "values",
                    created_at,
                    updated_at
                )
                VALUES (
                    :key,
                    :value,
                    COALESCE(:created_at, now()),
                    COALESCE(:updated_at, now())
                )
                ON CONFLICT ("keys") DO UPDATE
                SET
                    "values" = EXCLUDED."values",
                    updated_at = EXCLUDED.updated_at
                """
            ),
            payloads,
        )

    op.drop_table("risk_execution_configs", schema="ref_data")
    op.execute(
        "ALTER TABLE ref_data.risk_execution_configs_v2 "
        "RENAME TO risk_execution_configs"
    )


def _seed_defaults(conn: sa.engine.Connection) -> None:
    payloads = [
        {
            "key": field_name,
            "value": value,
        }
        for field_name, value in _DEFAULT_SEED_VALUES.items()
    ]
    conn.execute(
        sa.text(
            """
            INSERT INTO ref_data.risk_execution_configs ("keys", "values")
            VALUES (:key, :value)
            ON CONFLICT ("keys") DO NOTHING
            """
        ),
        payloads,
    )


def upgrade() -> None:
    conn = op.get_bind()
    op.execute("CREATE SCHEMA IF NOT EXISTS ref_data")

    if not _table_exists(conn, "risk_execution_configs"):
        _create_vertical_table("risk_execution_configs")
    else:
        existing_columns = _column_names(conn, "risk_execution_configs")
        if existing_columns != ["keys", "values", "created_at", "updated_at"]:
            _migrate_wide_rows_to_vertical(conn)

    _seed_defaults(conn)


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "risk_execution_configs"):
        return
    op.drop_table("risk_execution_configs", schema="ref_data")
