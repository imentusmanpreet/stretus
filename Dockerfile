FROM python:3.12-slim

WORKDIR /app

# Non-root user for k8s runAsNonRoot / runAsUser: 1000
RUN useradd -m -u 1000 -s /bin/bash stretus

# System deps for psycopg2 (libpq) and any pip packages that need a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Make scripts executable and hand ownership to the non-root user.
# Mounted PVC directories (/app/strategies) are chowned to fsGroup: 1000
# by the kubelet at pod start.
RUN chmod +x scripts/entrypoint.sh scripts/run_migrations.sh \
    && chown -R 1000:1000 /app

USER 1000

EXPOSE 8000

# Default: run migrations then start the application server.
# The entrypoint waits for Postgres, upgrades the schema, then starts Uvicorn.
CMD ["scripts/entrypoint.sh"]