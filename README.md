# Stretus AI — Trading Strategy Builder

An AI-powered trading strategy platform for the Indian market. Describe your strategy in plain language through a chat interface, and Stretus will generate a backtestable YAML strategy, run a full historical simulation, and return detailed performance metrics.

---

## Architecture

Stretus is composed of three cooperating services:

```
┌─────────────────────────────────────────────────────────┐
│                     Client / Browser                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│          Main API  (FastAPI · port 8000)                 │
│  • Chat sessions + LLM (Groq / Ollama)                  │
│  • Strategy generation → YAML                           │
│  • ChromaDB knowledge retrieval                         │
│  • Postgres persistence (SQLAlchemy + Alembic)          │
└──────────┬──────────────────────────┬───────────────────┘
           │ triggers backtest        │ queries signals/docs
┌──────────▼──────────┐  ┌───────────▼───────────────────┐
│  Quant Engine        │  │  KB Server (port 8002)         │
│  (FastAPI · 8001)    │  │  • SQLite full-text search     │
│  • OHLCV ingestion   │  │  • Signal registry             │
│  • Indicators        │  │  • Document metadata           │
│  • Simulator         │  └────────────────────────────────┘
│  • Metrics / report  │
│  • Callback → API    │
└──────────────────────┘
```

| Service | Port | Description |
|---|---|---|
| Main API | 8000 | Strategy chat, YAML generation, backtest orchestration |
| Quant Engine | 8001 | Backtest execution (indicators, simulation, metrics) |
| KB Server | 8002 | Knowledge-base search and signal metadata |
| PostgreSQL | 5432 | Persistent storage for sessions and strategies |
| pgAdmin | 5050 | Database GUI (Docker only) |

---

## Project Structure

```
stretus-ai/
├── app/                                  # Main FastAPI application
│   ├── main.py                           # App factory, lifespan, health routes
│   ├── core/
│   │   ├── config.py                     # Settings (pydantic-settings, reads .env)
│   │   └── errors.py                     # Global exception handlers
│   ├── api/
│   │   └── v1/
│   │       └── routes/
│   │           ├── chat.py               # Chat session endpoints
│   │           ├── strategy.py           # Strategy confirm + YAML endpoints
│   │           └── backtest.py           # Backtest trigger + callback receiver
│   ├── services/
│   │   ├── ai/                           # LLM service (Groq / Ollama / auto)
│   │   ├── chat/                         # Chat orchestration + response templates
│   │   ├── knowledge/                    # ChromaDB embedder + retriever
│   │   ├── strategy/                     # Strategy builder + YAML generator
│   │   └── backtest/                     # Market-data fetch + quant engine client
│   └── db/                               # Async SQLAlchemy session + models
│
├── quant_engine/                         # Standalone backtest service
│   ├── main.py                           # FastAPI app (port 8001)
│   ├── engine/
│   │   ├── runner.py                     # End-to-end backtest orchestration
│   │   ├── loader.py                     # YAML strategy parser
│   │   ├── data.py                       # OHLCV normalisation
│   │   ├── indicators.py                 # Technical indicators
│   │   ├── conditions.py                 # Entry / exit condition evaluator
│   │   ├── simulator.py                  # Position simulator
│   │   ├── metrics.py                    # PnL, Sharpe, drawdown, win-rate
│   │   ├── assessment.py                 # Strategy quality assessment
│   │   └── market_classifier.py          # Market regime detection
│   ├── Dockerfile
│   └── requirements.txt
│
├── kb_server/                    # Knowledge-base search service
│   ├── main.py                   # FastAPI app (port 8002)
│   └── services/
│       ├── document_loader.py    # Loads docs from stretus_knowledge_base/
│       ├── search_engine.py      # SQLite FTS index (data/kb.db)
│       ├── signal_service.py     # Signal registry bridge
│       └── synonyms.py           # Query expansion
│
├── stretus_kb/                   # Signal / rule registry
│   └── registry.py               # RuleRegistry + signal implementations
│
├── stretus_knowledge_base/       # Source documents indexed by KB server
├── strategies/                   # Generated strategy YAML files
├── data/                         # kb.db (SQLite index for KB server)
├── migrations/                   # Alembic migrations
│   ├── env.py                    # Migration environment (schema bootstrap)
│   ├── versions/
│   │   ├── 0001_initial_schema.py          # Creates ai_strategy schema + all tables
│   │   └── 0002_rename_schema_to_ai_strategy.py  # Renames legacy 'strategy' schema
│   └── script.py.mako
├── scripts/
│   ├── entrypoint.sh             # Docker entrypoint: wait-for-db → migrate → serve
│   ├── run_migrations.sh         # Local dev migration helper
│   └── init_db.sql               # Reference SQL (not used at runtime; Alembic owns DDL)
├── tests/                        # pytest test suite
│
├── Dockerfile                            # Main API image
├── docker-compose.yml                    # Full stack
├── docker-compose.local.yml              # DB + pgAdmin only
├── alembic.ini
├── requirements.txt
└── .env.example
```

---

## User Flow

```
1. POST /api/v1/strategy/chats
        → creates a chat session

2. POST /api/v1/strategy/chats/{chat_id}/messages  (repeat)
        → chat with the AI to describe your strategy requirements
          (LLM uses ChromaDB knowledge retrieval for context)

3. POST /api/v1/strategy/strategies
        → confirm; LLM generates a YAML strategy and persists it

4. GET  /api/v1/strategy/strategies/{strategy_id}
        → inspect the generated strategy object

5. POST /api/v1/strategy/backtest
        → main API fetches OHLCV from market-data source,
          calls quant engine → engine posts results back via callback
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below.

> **Security:** Never commit your real `.env` to version control. Rotate any API keys that may have been exposed.

```env
# ── App ───────────────────────────────────────────────
APP_ENV=development
APP_SECRET_KEY=change-this-secret
APP_DEBUG=true

# ── PostgreSQL ────────────────────────────────────────
POSTGRES_HOST=localhost          # use "postgres" when running inside Docker
POSTGRES_PORT=5432
POSTGRES_DB=stretus
POSTGRES_USER=stretus
POSTGRES_PASSWORD=stretus_db

DATABASE_URL=postgresql+asyncpg://stretus:stretus_db@localhost:5432/stretus
DATABASE_URL_SYNC=postgresql+psycopg2://stretus:stretus_db@localhost:5432/stretus

# ── LLM Provider ──────────────────────────────────────
LLM_PROVIDER=auto               # groq | ollama | auto

# Groq Cloud (fast inference, free tier available)
GROQ_API_KEY=1:gsk_primary,2:gsk_backup
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama Local (private, no API key required)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# ── Paths ─────────────────────────────────────────────
CHROMA_DB_PATH=./stretus_knowledge_base/data/vector_db
STRATEGY_FOLDER=./strategies

# ── External Services ─────────────────────────────────
QUANT_ENGINE_URL=http://localhost:8001      # http://quant_engine:8001 in Docker
MARKET_DATA_API_URL=https://your-market-data-api.example.com
```

### LLM Provider notes

| `LLM_PROVIDER` | What happens |
|---|---|
| `groq` | Uses Groq Cloud. Requires `GROQ_API_KEY` (single key or indexed key pool). |
| `ollama` | Uses local Ollama. Requires Ollama running and model pulled. |
| `auto` | Tries Groq first; falls back to Ollama automatically. |

Check which models are available via `GET /health/llm/models`.

---

## Running with Docker (recommended)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2+

### 1. Clone and configure

```bash
git clone <repo-url>
cd stretus-ai
cp .env.example .env
# edit .env with your API keys and secrets
```

### 2. Start the full stack

```bash
docker compose up --build
```

This brings up five containers:

| Container | URL |
|---|---|
| `stretus_api` | http://localhost:8000 |
| `stretus_quant_engine` | http://localhost:8001 |
| `stretus_kb_server` | http://localhost:8002 |
| `stretus_postgres` | localhost:5432 |
| `stretus_pgadmin` | http://localhost:5050 |

On every `docker compose up --build`, the API container runs `scripts/entrypoint.sh` which:
1. Waits for Postgres to accept connections
2. Runs `alembic upgrade head` to apply any pending migrations
3. Starts the Uvicorn server

No manual database setup is required for fresh environments.

### 3. Verify services

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### Useful Docker commands

```bash
# Run in background
docker compose up -d

# View logs for a specific service
docker compose logs -f api
docker compose logs -f quant_engine

# Stop all services
docker compose down

# Stop and remove volumes (wipes the database)
docker compose down -v

# Rebuild a single service after code changes
docker compose up --build api
```

### DB only (for running the API natively)

If you want Postgres in Docker but the Python services on your host:

```bash
docker compose up postgres
```

---

## Running without Docker

### Prerequisites

- Python 3.12+
- PostgreSQL 15 or 16 running locally
- (Optional) [Ollama](https://ollama.com/) if using a local LLM

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd stretus-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `POSTGRES_HOST=localhost` (and the rest of your credentials). If using Ollama, set `OLLAMA_BASE_URL=http://localhost:11434`.

### 3. Set up the database

Ensure the `stretus` user and database exist in your local Postgres instance (one-time):

```sql
CREATE USER stretus WITH PASSWORD 'stretus_db';
CREATE DATABASE stretus OWNER stretus;
```

Then apply all migrations (creates the `ai_strategy` schema and all tables automatically):

```bash
./scripts/run_migrations.sh
```

Or directly:

```bash
alembic upgrade head
```

### 4. Start the Main API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup the API will:
- Create required directories (`strategies/`, `stretus_knowledge_base/data/vector_db/`, `logs/`, `data/`)
- Print the active LLM provider and model
- Index knowledge-base documents into ChromaDB (first run takes a few seconds)

### 5. Start the Quant Engine

Open a second terminal:

```bash
cd quant_engine
pip install -r requirements.txt     # only needed once

FASTAPI_URL=http://localhost:8000 uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 6. Start the KB Server (optional)

Open a third terminal (from the repo root):

```bash
uvicorn stretus_knowledge_base.kb_server.main:app --host 0.0.0.0 --port 8002 --reload
```

### 7. (Optional) Pull an Ollama model

If `LLM_PROVIDER=ollama` or `auto`, make sure Ollama is running and the model is pulled:

```bash
ollama pull qwen2.5:7b
```

---

## Health & Diagnostic Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Root info + links |
| `GET /health` | API liveness |
| `GET /health/llm` | Active LLM provider + connectivity check |
| `GET /health/llm/models` | Available Groq + Ollama models |
| `GET /health/kb` | ChromaDB index status (stale / ready) |
| `POST /health/kb/reindex` | Force re-embed all knowledge-base documents |

---

## Running Tests

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_backtest.py -v
```

---

## Database & Migrations

### Schema

All application tables live in the **`ai_strategy`** PostgreSQL schema.

| Table | Description |
|---|---|
| `ai_strategy.chats` | Chat sessions |
| `ai_strategy.strategies` | Generated trading strategies |
| `ai_strategy.backtest` | Backtest runs and results |
| `ai_strategy.chat_messages` | Individual chat messages (with async-processing support) |

The Alembic version table is stored at `ai_strategy.alembic_version`.

### Connection credentials

| Setting | Value |
|---|---|
| Host (local) | `localhost` |
| Host (Docker) | `postgres` (service name) |
| Port | `5432` |
| Database | `stretus` |
| Username | `stretus` |
| Password | `stretus_db` |

### Migration workflow

Migrations are managed with **Alembic**. The `migrations/env.py` automatically creates the `ai_strategy` schema on first run, so no manual SQL setup is needed.

**Apply all pending migrations (upgrade to latest):**
```bash
./scripts/run_migrations.sh
```

**Check current revision:**
```bash
./scripts/run_migrations.sh current
```

**View migration history:**
```bash
./scripts/run_migrations.sh history
```

**Roll back one revision:**
```bash
./scripts/run_migrations.sh downgrade -1
```

**Generate a new migration after model changes:**
```bash
alembic revision --autogenerate -m "describe your change"
```

### Migration history

| Revision | Description |
|---|---|
| `e1a2b3c4d5e6` | Initial schema — creates `ai_strategy` schema, all ENUM types, tables, indexes, and triggers |
| `f2b3c4d5e6f7` | Renames legacy `strategy` schema to `ai_strategy` |

### Docker startup flow

```
postgres (healthy)
    └─► api container starts
            └─► scripts/entrypoint.sh
                    ├─ waits for Postgres to accept connections
                    ├─ alembic upgrade head   ← applies any pending migrations
                    └─ uvicorn app.main:app   ← starts the API server
```

---

## Deployment

CI/CD is configured in `.github/workflows/deployment.yml` and deploys to an AWS EC2 instance via Instance Connect. On the server the stack runs with:

```bash
docker compose up -d api quant_engine
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| FastAPI + Uvicorn | Web framework for all three services |
| SQLAlchemy (async) + asyncpg | Async Postgres ORM |
| Alembic | Database migrations |
| ChromaDB + sentence-transformers | Vector embeddings for knowledge retrieval |
| Groq SDK | Fast cloud LLM inference |
| Ollama | Local LLM support |
| pandas + numpy + ta | OHLCV processing and technical indicators |
| yfinance + nsepython | Market data sources |
| PyYAML | Strategy YAML parsing in quant engine |
| pytest + httpx | Test suite |
