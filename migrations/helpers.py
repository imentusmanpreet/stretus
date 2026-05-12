"""
migrations/helpers.py
─────────────────────
Idempotency helpers for Alembic migration files.

Usage
─────
    from migrations.helpers import column_exists, table_exists

    def upgrade() -> None:
        if column_exists("strategies", "new_col", schema="ai_strategy"):
            return
        op.add_column("strategies", sa.Column("new_col", sa.Text()), schema="ai_strategy")
"""
import sqlalchemy as sa
from alembic import op

SCHEMA = "ai_strategy"


def schema_exists(schema: str) -> bool:
    return _scalar(
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = :v",
        schema,
    )


def table_exists(table: str, *, schema: str = SCHEMA) -> bool:
    return _scalar(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_schema = :s AND table_name = :v",
        table, schema=schema,
    )


def column_exists(table: str, column: str, *, schema: str = SCHEMA) -> bool:
    return _scalar(
        "SELECT COUNT(*) FROM information_schema.columns"
        " WHERE table_schema = :s AND table_name = :t AND column_name = :v",
        column, schema=schema, table=table,
    )


def index_exists(index: str, *, schema: str = SCHEMA) -> bool:
    return _scalar(
        "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = :s AND indexname = :v",
        index, schema=schema,
    )


def enum_exists(name: str, *, schema: str = SCHEMA) -> bool:
    return _scalar(
        "SELECT COUNT(*) FROM pg_type t"
        " JOIN pg_namespace n ON n.oid = t.typnamespace"
        " WHERE t.typtype = 'e' AND n.nspname = :s AND t.typname = :v",
        name, schema=schema,
    )


def constraint_exists(constraint: str, table: str, *, schema: str = SCHEMA) -> bool:
    return _scalar(
        "SELECT COUNT(*) FROM information_schema.table_constraints"
        " WHERE constraint_schema = :s AND table_name = :t AND constraint_name = :v",
        constraint, schema=schema, table=table,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _scalar(sql: str, value: str, *, schema: str = SCHEMA, table: str | None = None) -> bool:
    params: dict = {"s": schema, "v": value}
    if table is not None:
        params["t"] = table
    result = op.get_bind().execute(sa.text(sql), params)
    return result.scalar() > 0
