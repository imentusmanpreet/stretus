#!/usr/bin/env bash
# Creates k8s Secrets required by the python-ai Helm chart.
#
# Secrets created:
#   stretus-ai-credentials  – APP_SECRET_KEY, GROQ_API_KEY, DATABASE_URL,
#                             DATABASE_URL_SYNC, UPSTOX_API_KEY, UPSTOX_ACCESS_TOKEN,
#                             OLLAMA_BASE_URL, HISTORICAL_DATA_URL (HTTP fallback only)
#
# Backtest OHLCV gRPC (MARKET_DATA_GRPC_*) is set by the Helm chart deployment
# template, not this secret — see deploy/helm/values.yaml → marketData.
#
# The postgres-credentials secret is shared with other services; create/update
# it using deploy/helm/charts/user/scripts/create-required-secrets.sh and pass
# the POSTGRES_* vars you want. This script only touches stretus-ai-credentials.
#
# Usage:
#   export GROQ_API_KEY='gsk_...'
#   export OPENROUTER_API_KEY='sk-or-v1-...'
#   export APP_SECRET_KEY='<random 64-char string>'
#   export DATABASE_URL='postgresql+asyncpg://stretus:<password>@postgres-postgresql:5432/stretus'
#   export DATABASE_URL_SYNC='postgresql+psycopg2://stretus:<password>@postgres-postgresql:5432/stretus'
#   ./create-required-secrets.sh stretus-ai-dev2
#
# All vars can be overridden by exporting them before calling the script.
# The script uses `kubectl apply` (idempotent) so it is safe to re-run.

set -euo pipefail

NS="${1:-stretus-ai-dev2}"

POSTGRES_HOST="${POSTGRES_HOST:-postgres-postgresql}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stretus}"
POSTGRES_USER="${POSTGRES_USER:-stretus}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-stretus123}"

# Derive DATABASE_URL from postgres vars if not explicitly set.
DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}}"
DATABASE_URL_SYNC="${DATABASE_URL_SYNC:-postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}}"

APP_SECRET_KEY="${APP_SECRET_KEY:?ERROR: APP_SECRET_KEY must be set (generate with: openssl rand -hex 32)}"
GROQ_API_KEY="${GROQ_API_KEY:-}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
UPSTOX_API_KEY="${UPSTOX_API_KEY:-}"
UPSTOX_ACCESS_TOKEN="${UPSTOX_ACCESS_TOKEN:-}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-}"
HISTORICAL_DATA_URL="${HISTORICAL_DATA_URL:-}"

echo "Applying stretus-ai-credentials in namespace ${NS}..."

kubectl create secret generic stretus-ai-credentials \
  --namespace "${NS}" \
  --from-literal=APP_SECRET_KEY="${APP_SECRET_KEY}" \
  --from-literal=GROQ_API_KEY="${GROQ_API_KEY}" \
  --from-literal=OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  --from-literal=DATABASE_URL="${DATABASE_URL}" \
  --from-literal=DATABASE_URL_SYNC="${DATABASE_URL_SYNC}" \
  --from-literal=UPSTOX_API_KEY="${UPSTOX_API_KEY}" \
  --from-literal=UPSTOX_ACCESS_TOKEN="${UPSTOX_ACCESS_TOKEN}" \
  --from-literal=OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
  --from-literal=HISTORICAL_DATA_URL="${HISTORICAL_DATA_URL}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "stretus-ai-credentials applied in namespace ${NS}."

DEPLOY_API="${PYTHON_AI_API_DEPLOY:-python-ai-api}"
DEPLOY_QUANT="${PYTHON_AI_QUANT_DEPLOY:-python-ai-quant}"

if [[ "${RESTART_AFTER_APPLY:-true}" == "true" ]]; then
  for deploy in "${DEPLOY_API}" "${DEPLOY_QUANT}"; do
    if kubectl rollout restart "deployment/${deploy}" -n "${NS}" >/dev/null 2>&1; then
      echo "Restarted deployment/${deploy} so containers load the updated secret values."
    else
      echo "Note: deployment/${deploy} not found — skipping restart (it may not exist yet)."
    fi
  done
fi
