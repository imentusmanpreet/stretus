# Dynamic Universe — Phase C design note

Phase C of `docs/dynamic-universe-implementation.md` (§6, §7, §17): **survivorship-safe
point-in-time constituents**. Builds on Phases A/B. Makes a historical dynamic backtest read
the universe membership *as it stood at `asof`* — so a since-delisted name correctly appears
in a 2021 resolution and is correctly absent today (Invariant 7). The static path is untouched.

## What shipped

| Component | Change |
|---|---|
| `app/db/models/universe.py` *(new)* | `UniverseMembership` ORM model — `ai_strategy.universe_membership`: `(tenant_id, universe_key, symbol, valid_from, valid_to, source)`. Unique on `(tenant_id, universe_key, symbol, valid_from)` (idempotent ingest); index on `(universe_key, valid_from, valid_to)` (fast point-in-time lookup). Tenant-scoped (§15). |
| `migrations/versions/0007_universe_membership.py` *(new)* | Alembic migration (revision `c8d9e0f1a2b3`, off head `a7b8c9d0e1f2`). Additive, idempotent (`table_exists` guard), reversible. Single linear head confirmed. |
| `migrations/env.py` | Registers the new model for autogenerate. |
| `app/services/universe/membership.py` *(new)* | KB-free point-in-time resolution: the **pure** `members_on` rule, the injectable `MembershipStore` (with `InMemoryMembershipStore` for tests + `SqlMembershipStore` for prod), and `MembershipPoolProvider` — the resolver's `pool_provider` for index/sector/f_and_o. `source_universe_key` maps a `UniverseSource` → its `universe_key`. |
| `app/services/universe/resolver.py` | `survivorship_mode` now auto-detects (Invariant 7): a `pool_provider` advertising `point_in_time = True` (the membership store) ⇒ `"point_in_time"`; otherwise `"approximate"`. An explicit value still wins. Backward-compatible — Phase A/B callers (no provider / watchlist) still get `"approximate"`. |

## The membership rule (single source of truth)
"Members of `universe_key` on date D":
```
valid_from <= D AND (valid_to IS NULL OR valid_to > D)
```
Half-open intervals `[valid_from, valid_to)`: a name that leaves on date X is a member up to
but **not including** X. `valid_to IS NULL` ⇒ still current. `members_on` is a pure function
(no I/O), so survivorship safety is unit-tested without a database.

## How it plugs in
The resolver already injects `pool_provider` (Phase A). Phase C supplies a real one:
`MembershipPoolProvider(SqlMembershipStore(AsyncSessionLocal))`. The provider maps the source
to its `universe_key`, loads that key's intervals from the store, and applies `members_on` at
`asof`. Because it advertises `point_in_time = True`, the resolver stamps the snapshot's
`survivorship_mode` accordingly — no caller change.

## Tests (`tests/test_universe/test_membership.py`)
- **Survivorship** — DELISTED (left 2022-06-01) is in the 2021 resolution, absent in 2025;
  LATECOMER (joined 2023) is absent in 2021, present in 2025.
- **Half-open interval** — the name is a member the day before its exit date, not on it.
- **Source→key mapping** — index/sector use `name`; f_and_o uses `"f_and_o"`.
- **Provider** filters by `universe_key` + `asof`; advertises `point_in_time`.
- **Resolver integration** — membership provider ⇒ snapshot `survivorship_mode="point_in_time"`
  and the delisted name resolves into a historical snapshot; watchlist (no provider) ⇒
  `"approximate"`.

## Remaining work (later)
- An **ingestion job** to backfill `universe_membership` from a constituent-history source
  (§4.4) — the table is empty until populated; `SqlMembershipStore` reads whatever is there.
- The Phase-B **transport glue** (`/run_portfolio` endpoint + API branch) still applies; a
  dynamic backtest over a date range refreshes the resolver at each cadence point and now gets
  point-in-time membership for free once the store is populated.
- **KB-free** maintained (Invariant 11) — verified by the existing import-guard test, which
  scans the new module (it late-imports the DB layer inside methods, so package import stays
  light and the guard sees no `app.kb`/`app.planner`).

## Dependency note
Migrations need `alembic` (declared in `requirements.txt`); the model/provider tests need no DB
(injected in-memory store). `SqlMembershipStore` runs against `ai_strategy.universe_membership`
once migrated.
