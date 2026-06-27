# Dynamic Universe — end-to-end backtest glue

Closes the gap between the Phase-B/C machinery and a runnable backtest: the cross-service
transport (`/run-portfolio-sync`), the app orchestrator, the route branch, and the membership
ingestion job. With this, a dynamic-universe strategy backtests end-to-end. Static path unchanged.

## The end-to-end flow
```
POST /backtest (dynamic spec)
  → _call_quant_engine detects a top-level `universe:` block (plain YAML read)
  → _run_dynamic_backtest_and_persist
       → run_dynamic_backtest (app/services/universe/backtest_orchestrator.py)
            1. resolve_universe(asof)            # KB-free; survivorship auto-detected
            2. load_member_execution_frames()    # Tier-B lazy load, members only
            3. POST member OHLCV → engine /run-portfolio-sync
       → summarize_backtest_for_db + apply_sanitized_result_to_row  # persist to Backtest row
```

## What shipped

| Component | Change |
|---|---|
| `quant_engine/main.py` | `POST /run-portfolio-sync` — `RunPortfolioSyncRequest` (template_yaml, member_ohlcv, membership_windows, portfolio config, snapshots, survivorship_mode) → runs `run_portfolio_backtest` in an executor → returns the single portfolio result. Mirrors `/run-sync`; empty members → 400. |
| `app/services/backtest/quant_engine_client.py` | `run_quant_portfolio_sync(payload)` — posts to `/run-portfolio-sync`, surfaces transport/engine errors (never a silent empty), 404 → clear "rebuild the engine" message. |
| `app/services/universe/backtest_orchestrator.py` | `run_dynamic_backtest(...)` — the bridge: resolve → Tier-B load → engine call. Injectable `fetch_ohlcv`/`pool_provider`/`engine_call` (network-free tests). Applies the platform `max_positions` cap (§7.1); passes `survivorship_mode` + snapshot through. v1 resolves a single snapshot at `asof`; multi-cadence refresh is the documented extension (the `membership_windows_from_snapshots` helper is ready). |
| `app/api/v1/routes/backtest.py` | Dynamic branch in `_call_quant_engine`: detect `universe:` (plain YAML, no engine import) → `_run_dynamic_backtest_and_persist` → reuse `summarize_backtest_for_db` + `apply_sanitized_result_to_row` to land the result on the Backtest row (additive). Static specs fall straight through (Invariant 2). index/sector/f_and_o use a `MembershipPoolProvider(SqlMembershipStore)`; watchlist/crypto need no provider. |
| `app/services/universe/ingestion.py` | Membership backfill: pure `diff_snapshot` (open-new / close-departed incremental rule), `MembershipIngestor` (idempotent `upsert` + `apply_snapshot`), and a CLI (`python -m app.services.universe.ingestion --universe NIFTY500 --symbols …`). KB-free. |
| `app/services/universe/membership.py` | `MembershipRow` gains `source` (provenance carried end-to-end). |

## Tests
- **Engine endpoint** (`tests/test_quant_engine/test_run_portfolio_sync_endpoint.py`) — real engine app via TestClient: portfolio result shape, member pass-through, empty-members 400.
- **Orchestrator** (`tests/test_universe/test_backtest_orchestrator.py`) — resolve→load→engine with an injected engine stub; watchlist → approximate; membership provider → point_in_time + delisted name resolves.
- **Ingestion** (`tests/test_universe/test_ingestion.py`) — `diff_snapshot` opens new / closes departed / no-op on unchanged / deterministic.
- **Route helpers** (`tests/test_api/test_backtest_dynamic_branch.py`) — `universe:` detection + ISO parsing.

## How to run it for real
1. **Install declared deps** (this dev venv was missing some): `pip install -r requirements.txt` (grpcio, asyncpg, protobuf, orjson, alembic, uvicorn, and **pyarrow** for the Parquet store).
2. **Migrate**: `alembic upgrade head` (creates `ai_strategy.universe_membership`).
3. **Seed membership** (for index/sector/f_and_o sources): `python -m app.services.universe.ingestion --universe NIFTY500 --symbols TCS,INFY,HDFCBANK,… --asof 2024-01-01`. (Watchlist/crypto sources skip this.)
4. **Start** the quant engine (now exposing `/run-portfolio-sync`) and the API.
5. **Backtest** a dynamic strategy (one whose StrategySpec has a `universe` block) via `POST /backtest` — the route auto-branches.

## Known limitations / next
- **Single-snapshot resolution** in v1 (membership at `asof`); per-cadence refresh over a long
  range is the next enhancement (helpers already present).
- **Result persistence** stores the portfolio result in `result_json` and marks the row
  completed; mapping portfolio metrics into any typed summary columns can be refined once
  exercised against a live DB (the response/equity-curve payload is fully present in `result_json`).
- A real **constituent-history feed** still needs wiring into the ingestion CLI/job; until then,
  seed via the CLI. `pyarrow` enables the Parquet store (else `CachingFetchProvider` is used).
