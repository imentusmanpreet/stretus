# Dynamic Universe — Phase B design note

Phase B of `docs/dynamic-universe-implementation.md` (§8, §17): the **portfolio backtest** —
wrap the existing per-symbol simulator under one shared capital pool, plus the columnar store
and the Tier-B lazy-load bridge. Builds on Phase A (`docs/dynamic-universe-phase-a.md`). The
static single-symbol path stays byte-for-byte unchanged (Invariant 2).

## What shipped

| Component | Change |
|---|---|
| `quant_engine/engine/portfolio.py` *(new)* | **Backtest Portfolio Manager** — the single authority for shared capital, the `max_positions` cap, sizing (`equal_weight` / `risk_based`), and the one realised equity curve (Invariant 9). Records every skip with a reason (`position_cap` / `insufficient_capital` / `sector_cap`); produces metrics (total return, max drawdown, Sharpe/Sortino/Calmar, win rate) + per-symbol attribution. Pure module (numpy/pandas only) — no engine/app imports. `candidate_from_trade` adapts a simulator `Trade`. |
| `quant_engine/engine/runner.py` | `run_portfolio_backtest(...)` — wraps per-member `run_backtest` (reuse, not rewrite — Invariant 1), restricts trades to membership windows, arbitrates via the Portfolio Manager → ONE portfolio result (equity curve, attribution, skip counts, snapshots, `survivorship_mode`). Plus a **guarded, additive** `return_trades` hook (default off → static result byte-for-byte unchanged). |
| `quant_engine/engine/loader.py` | Documents + makes explicit that a top-level `universe:` block is tolerated (loader reads only `strategy:`); adds `extract_universe_block()` so dynamic mode is detectable. |
| `app/services/data/parquet_store.py` *(new)* | **`ParquetOhlcvStore`** implementing the `DataProvider` Protocol (§4.2): range-sliced reads by `(symbol, timeframe, range)` bounded at `to` (no look-ahead), idempotent/resumable ingest, optional lazy fetch+ingest on miss. Swaps in behind callers with no change (Parquet→ClickHouse upgrade path). |
| `app/services/universe/backtest_orchestrator.py` *(new)* | **Tier-B loader** (§8 step 4): `load_member_execution_frames` lazily fetches execution-tier OHLCV for resolved members only, over their window + warm-up, bounded concurrency, per-symbol tolerance — peak memory O(active members) (Invariant 6). `membership_windows_from_snapshots` derives per-symbol [from,to] active windows from a refresh sequence. |

## Portfolio model (how the overlay wraps per-symbol sim)
Each member is simulated independently by the **existing** `simulate_trades` (via `run_backtest`),
yielding candidate trades with entry/exit timestamps and a fractional return. The Portfolio
Manager replays them on **one unified clock**: at each candidate entry it releases positions
that have since exited (returning capital to the shared pool), admits the entry only if a
slot is free (`max_positions`) **and** capital is available (else skip-with-reason), and on
exit realises `allocated × pnl_frac` back into the pool. The equity curve is the realised
shared-pool equity (open positions marked at cost — trade granularity). This is a faithful,
transparent overlay that never forks the simulator (Invariants 1 & 9).

Sizing: `equal_weight` = 1/`max_positions` of current equity per slot; `risk_based` =
`risk_per_trade_pct` of equity ÷ stop distance. Optional `sector_cap_pct` caps concurrent
capital per sector.

## Tests (`tests/test_quant_engine/`, `tests/test_universe/`)
- **Portfolio cap** — N overlapping entries, `max_positions=k` → exactly k taken, N−k skipped (`position_cap`).
- **Capital sharing** — concurrent trades split the shared pool (not summed per symbol); freed slots compound.
- **Risk-based sizing**, **sector cap**, **single equity curve + metrics**, **empty portfolio**.
- **End-to-end** `run_portfolio_backtest` over two synthetic members → one result; **membership-window filtering**.
- **Static no-change** — `run_backtest` without `return_trades` never gains a `trades` key.
- **Loader** — `universe:` tolerated; `extract_universe_block` returns it / `None` / tolerates malformed.
- **Parquet store** — idempotent ingest, range-slice bounded at `to`, lazy fallback (skips when no Parquet engine).
- **Orchestrator** — Tier-B loads only members with data; membership windows open/close correctly.

## Remaining Phase-B glue (next slice)
The engine runs as a **separate service** (the API posts OHLCV to it). A dynamic backtest
therefore needs the **transport layer**:
1. A quant-engine `POST /run_portfolio` endpoint accepting `{template_yaml, member_ohlcv,
   membership_windows, portfolio_config, snapshots}` → calls `run_portfolio_backtest`.
2. A backtest-route branch (`app/api/v1/routes/backtest.py`): when the spec is dynamic,
   resolve the universe → `load_member_execution_frames` → post to `/run_portfolio` → persist
   the portfolio result (additive schema, Invariant 10).

All the reusable machinery (resolver, provider, Parquet store, Portfolio Manager, runner
entry point, Tier-B loader) is in place and tested; the above is wiring, best landed as its
own integration-tested PR. Survivorship-safe point-in-time constituents are Phase C; live is
Phase D. No Kafka/ClickHouse required (§18) — the Parquet store covers the scale need.

## Dependency note
`ParquetOhlcvStore` needs `pyarrow` (or `fastparquet`); not installed in this environment, so
its tests skip. Until installed, `CachingFetchProvider` (Phase A) satisfies the same
`DataProvider` interface.
