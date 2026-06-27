"""
migrations/env.py
──────────────────
Alembic migration environment.

Uses the SYNC database URL (psycopg2) because Alembic does not support
async connections natively.

Auto-generates migrations from the SQLAlchemy ORM models defined under
app/db/models/.
"""
import os
import sys
from logging.config import fileConfig

import sqlalchemy as sa
from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the app package importable from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings
from app.db.session import Base

# Import ALL models so Alembic can detect them for autogenerate
import app.db.models.strategy   # noqa: F401
import app.db.models.execution  # noqa: F401
import app.db.models.universe   # noqa: F401

# ── Alembic config ─────────────────────────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()

# Override sqlalchemy.url with the SYNC url from .env / environment
config.set_main_option("sqlalchemy.url", settings.database_url_sync)

target_metadata = Base.metadata


# ── Schema bootstrap ───────────────────────────────────────────────────────────
def _ensure_strategy_schema(connection: sa.engine.Connection) -> None:
    """
    Create the 'ai_strategy' schema before Alembic tries to place its
    alembic_version table inside it.  This must run outside a transaction
    block so the schema is visible to subsequent DDL.
    """
    connection.execute(sa.text("CREATE SCHEMA IF NOT EXISTS ai_strategy"))
    connection.commit()


# ── Offline migrations ─────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    """
    Emit SQL to stdout without a live DB connection.
    Useful for reviewing what will be executed (e.g. alembic upgrade head --sql).
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="ai_strategy",
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations ──────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Ensure the ai_strategy schema exists BEFORE Alembic tries to create
        # its version table inside it (ai_strategy.alembic_version).
        _ensure_strategy_schema(connection)

        # target_metadata is only needed for `alembic revision --autogenerate`.
        # During plain `upgrade`/`downgrade` runs, passing target_metadata causes
        # Alembic's op.create_table() to resolve Enum columns from the ORM
        # MetaData (create_type=True by default), which conflicts with the
        # migration's own CREATE TYPE statements and produces DuplicateObject
        # errors.  We therefore omit it here; autogenerate callers should set
        # it in a separate env-var-controlled branch if needed.
        context.configure(
            connection=connection,
            include_schemas=True,
            version_table_schema="ai_strategy",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
