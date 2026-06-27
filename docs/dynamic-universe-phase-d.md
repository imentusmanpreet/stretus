# Dynamic Universe — Phase D design note (live execution)

Phase D of `docs/dynamic-universe-implementation.md` (§9, §17): take a dynamic universe
**live** — re-resolve on a cadence, spawn/retire members, enforce the shared-capital +
`max_positions` cap in real time, and reconcile after a restart. Builds on A/B/C. Static
single-symbol live execution is untouched (Invariant 2).

## What shipped (the testable core + persistence)

| Component | Change |
|---|---|
| `app/db/models/universe.py` (+ `migration 0008`) | `UniverseMember` — the **running** membership of a live deployment: `(deployment_id, symbol, status[active/retiring/retired], member_state, allocated_capital, snapshot_hash)`. Read on restart to reconcile (§9 step 7). Migration `d9e0f1a2b3c4` (single head), additive/idempotent/reversible. |
| `app/services/execution/portfolio_manager.py` *(new)* | **Live `LivePortfolioManager`** — stateful authority for a deployment's shared capital + `max_positions` cap (Invariant 9). `evaluate_entry` → admit/skip-with-reason (`position_cap`/`insufficient_capital`/`sector_cap`/`already_open`); `admit`/`release` track the pool; `rebuild` restores it on restart. Same admission logic as the backtest manager (parity, §11). Pure — no DB/broker. |
| `app/services/execution/universe_scheduler.py` *(new)* | The scheduler **core**: `diff_membership` (spawn/retire/unchanged — pure), the `AdvisoryLock` contract with `PostgresAdvisoryLock` (one replica fires each tick via `pg_try_advisory_lock`) + `InMemoryAdvisoryLock` (tests), `reconcile_state` (rebuild from DB rows ∧ broker positions), and `run_refresh_tick` (lock → resolve → diff → spawn/retire over injectable deps). |

## How a live tick works (§9)
```
APScheduler fires (e.g. daily 10:15)                      ← thin driver (edge, see below)
  → run_refresh_tick:
       lock.try_acquire()         # Postgres advisory lock → exactly ONE replica proceeds
       resolved = await resolve() # live resolver → snapshot (persisted + hashed)
       diff = diff_membership(active, resolved.members)
       on_spawn(diff.to_spawn)    # warm up indicators, create sub-state, mark active
       on_retire(diff.to_retire)  # square off (or hold-to-exit), mark retiring → retired
       lock.release()
```
Per bar/tick, each active member drives the existing `strategy_evaluator.evaluate()`; **every
entry intent first passes `LivePortfolioManager.evaluate_entry`** — only admitted intents
become bracket orders fanned out to the existing Upstox/Binance adapters with idempotent client
order ids.

## Fail-safe behaviours (§13)
- **Multi-replica**: the advisory lock guarantees one fire per tick; others skip (tested).
- **Missed/overlapping tick**: `diff_membership` is idempotent — a tick that changes nothing
  is a clean no-op; a missed cadence simply resolves on the next tick.
- **Restart/failover**: `reconcile_state` trusts the **broker** for open positions (a position
  filled/closed while down is not resurrected) and the **DB** for membership; the Portfolio
  Manager `rebuild`s from that before resuming (tested).
- **Position cap / capital exhaustion**: skipped with an explicit reason; never exceeded.

## Tests (`tests/test_execution/`)
- **Live Portfolio Manager** — cap → `position_cap`; shared capital (not summed); release +
  compounding; `already_open` guard; risk-based sizing; sector cap; insufficient capital;
  reconciliation rebuild.
- **Scheduler** — spawn/retire/unchanged diff + no-op; advisory lock single-fire then re-acquire;
  reconciliation (broker-vs-DB); a full tick fires & diffs; a tick **skips** when the lock is held.

## Remaining integration wiring (the edges — next slice)
The pure core is done and tested; these connect it to running infra (need APScheduler/DB/broker,
so they land as a focused integration PR like the backtest glue):
1. **APScheduler driver** — an in-process scheduler that, per live dynamic deployment, calls
   `run_refresh_tick` on the cadence with a `PostgresAdvisoryLock(AsyncSessionLocal, key)` and
   real `on_spawn`/`on_retire` that persist `universe_members` transitions + warm indicators +
   square off via the OMS.
2. **`strategy_evaluator.py`** — drive once per active member; route each entry intent through
   `LivePortfolioManager.evaluate_entry` before emitting an order.
3. **`risk_manager.py` / `risk_execution_config_service.py`** — portfolio-scope fields
   (`max_concurrent_positions`, `sector_cap_pct`) with `rms_sources` provenance; per-member risk
   inherits from the portfolio parent.
4. **`app/api/v1/routes/execution.py`** — branch dynamic deployments to the portfolio-aware path.
5. **OMS fan-out** — idempotent client order ids per (deployment, member, intent).

## Status across all phases
- **A** (spec/resolver/provider) ✅ · **B** (portfolio backtest core) ✅ · **C** (survivorship) ✅
- **Glue** (engine `/run-portfolio-sync`, orchestrator, route branch, ingestion) ✅
- **D** (live core: model, Portfolio Manager, scheduler, reconciliation) ✅ — edges wiring remains.
- Out of scope (§19): cross-strategy netting, correlation sizing, options/futures universes,
  Redis/distributed queue.
