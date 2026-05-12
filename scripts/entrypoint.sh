#!/usr/bin/env bash
# Docker entrypoint: wait for Postgres → migrate → start Uvicorn.
set -euo pipefail

: "${POSTGRES_HOST:=localhost}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_USER:=stretus}"
: "${POSTGRES_DB:=stretus}"

# ---------------------------------------------------------------------------
# 1. Wait for PostgreSQL
# ---------------------------------------------------------------------------
echo "[entrypoint] Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."

for attempt in $(seq 1 30); do
    python -c "
import psycopg2, sys
try:
    psycopg2.connect(host='${POSTGRES_HOST}', port=${POSTGRES_PORT},
                     user='${POSTGRES_USER}', password='${POSTGRES_PASSWORD}',
                     dbname='${POSTGRES_DB}', connect_timeout=3)
except Exception:
    sys.exit(1)
" 2>/dev/null && break
    [ "$attempt" -eq 30 ] && { echo "[entrypoint] Timed out waiting for PostgreSQL."; exit 1; }
    sleep 2
done

echo "[entrypoint] PostgreSQL is ready."

# ---------------------------------------------------------------------------
# 2. Run Alembic migrations
#
# Recovery path handles two known inconsistency states:
#   - KeyError / "can't locate revision": stale revision IDs left in
#     alembic_version from a previous deployment with different migration files.
#   - DuplicateObject / "already exists": schema objects exist but are not
#     tracked (alembic_version is empty or missing). Migration files have
#     idempotency guards so a clean retry skips existing objects safely.
#
# Both states share the same fix: reset alembic_version and retry.
# ---------------------------------------------------------------------------
echo "[entrypoint] Running migrations..."

# NOTE: We intentionally disable set -e for the migration command so that
# the script does not exit before we can inspect the output and run recovery.
MIGRATION_EXIT=0
MIGRATION_LOG=$(alembic upgrade head 2>&1) || MIGRATION_EXIT=$?
echo "$MIGRATION_LOG"

if [ "$MIGRATION_EXIT" -ne 0 ]; then
    if echo "$MIGRATION_LOG" | grep -qE "KeyError|can't locate revision|DuplicateObject|already exists"; then
        echo "[entrypoint] Stale migration state detected — resetting alembic_version and retrying..."
        python -c "
import os, sqlalchemy as sa
with sa.create_engine(os.environ['DATABASE_URL_SYNC']).connect() as conn:
    conn.execute(sa.text('CREATE SCHEMA IF NOT EXISTS ai_strategy'))
    conn.execute(sa.text(
        'CREATE TABLE IF NOT EXISTS ai_strategy.alembic_version'
        ' (version_num VARCHAR(32) NOT NULL PRIMARY KEY)'
    ))
    conn.execute(sa.text('DELETE FROM ai_strategy.alembic_version'))
    conn.commit()
"
        RETRY_EXIT=0
        RETRY_LOG=$(alembic upgrade head 2>&1) || RETRY_EXIT=$?
        echo "$RETRY_LOG"
        if [ "$RETRY_EXIT" -ne 0 ]; then
            echo "[entrypoint] Migration retry failed — see logs above."
            exit "$RETRY_EXIT"
        fi
    else
        echo "[entrypoint] Migration failed — see logs above."
        exit "$MIGRATION_EXIT"
    fi
fi

echo "[entrypoint] Migrations complete."

# ---------------------------------------------------------------------------
# 3. Seed risk / execution config defaults
# ---------------------------------------------------------------------------
echo "[entrypoint] Seeding risk / execution config..."
python scripts/seed_risk_execution_config.py

# ---------------------------------------------------------------------------
# 4. Start application
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting Uvicorn..."
exec uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    ${ROOT_PATH:+--root-path "$ROOT_PATH"} \
    ${APP_DEBUG:+--reload}
