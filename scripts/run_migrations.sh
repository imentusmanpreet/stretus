#!/usr/bin/env bash
# =============================================================================
#  scripts/run_migrations.sh
#  Local-development helper: apply all pending Alembic migrations.
#
#  Usage:
#    ./scripts/run_migrations.sh                 # upgrade to head (default)
#    ./scripts/run_migrations.sh downgrade -1    # rollback one revision
#    ./scripts/run_migrations.sh history         # show migration history
#    ./scripts/run_migrations.sh current         # show current DB revision
#    ./scripts/run_migrations.sh stamp <rev>     # mark revision without running
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Auto-activate a virtual environment if one exists
# ---------------------------------------------------------------------------
for VENV_DIR in .venv venv env .env_venv; do
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        echo "[migrations] Activating virtualenv: ${VENV_DIR}"
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
        break
    fi
done

# ---------------------------------------------------------------------------
# Resolve the alembic command
# Use the venv/system binary if available, otherwise fall back to python -m alembic
# ---------------------------------------------------------------------------
if command -v alembic &>/dev/null; then
    ALEMBIC="alembic"
elif python3 -m alembic --version &>/dev/null 2>&1; then
    ALEMBIC="python3 -m alembic"
elif python -m alembic --version &>/dev/null 2>&1; then
    ALEMBIC="python -m alembic"
else
    echo "[migrations] ERROR: alembic is not installed or not on PATH."
    echo "  Install it with:  pip install alembic"
    exit 1
fi

# ---------------------------------------------------------------------------
# Load .env for local runs (without Docker)
# ---------------------------------------------------------------------------
if [ -f ".env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

COMMAND="${1:-upgrade}"
ARGS="${*:2}"

case "$COMMAND" in
    upgrade)
        TARGET="${ARGS:-head}"
        echo "==> Running: alembic upgrade ${TARGET}"
        $ALEMBIC upgrade "$TARGET"
        echo "==> Done."
        ;;
    downgrade)
        TARGET="${ARGS:?'Usage: run_migrations.sh downgrade <revision|-1>'}"
        echo "==> Running: alembic downgrade ${TARGET}"
        $ALEMBIC downgrade "$TARGET"
        echo "==> Done."
        ;;
    history|current|heads|branches|show)
        $ALEMBIC "$COMMAND" ${ARGS}
        ;;
    stamp)
        TARGET="${ARGS:?'Usage: run_migrations.sh stamp <revision>'}"
        echo "==> Running: alembic stamp ${TARGET}"
        $ALEMBIC stamp "$TARGET"
        echo "==> Done."
        ;;
    *)
        echo "Usage: $0 {upgrade [rev]|downgrade <rev>|history|current|stamp <rev>}"
        exit 1
        ;;
esac
