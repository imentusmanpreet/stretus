# IMPLEMENTATION PROMPT — Dynamic Universe (Production-Grade, Direct `StrategySpec` Path)

> Hand-off spec. Self-contained. Read §0–§2 before writing any code. The dynamic-universe
> feature is **KB-free** (see Invariant 11) and lives entirely on the **direct
> `StrategySpec` path** — not the SDL/planner path, not the knowledge-base path.

---

## 0. Mission & Context

Add **dynamic universe** support to Stretus: a strategy that, instead of naming one
instrument, names a **rule for selecting instruments** which is re-resolved on a cadence
and trades a **time-varying set** of symbols under one shared capital pool.

Implement it on the **direct `StrategySpec` path**:
`app/strategy/spec.py` → `to_engine_yaml_dict()` → `quant_engine` for backtest, and
`app/services/execution/strategy_evaluator.py` for live.

Scale target: from today's ~150-symbol pool to **universes up to 5,000 symbols over
multi-year ranges**. Naïve "load all data then loop" is **forbidden** (§4). The feature
must be efficient, robust, observable, reproducible, backward-compatible, and **free of any
knowledge-base / catalog-signal dependency**.

**Canonical example** (use in docstrings, tests, reference case):
> *"Every morning one hour after open, trade the 10 most active NIFTY-500 stocks above their
> VWAP. Long on a break of the first-hour high. Risk 1% per trade, max 5 positions at once,
> square off at close."*

**Stress case:** *"Trade the top 10 crypto by volume"* against a 5,000-symbol pool over 2 years.

---

## 0.1 Codebase Map — the Direct Strategy Path (read this first)

This is the *only* path the feature touches. Know it cold before editing.

### Authoring → Spec → Validate
- **`app/services/ai/strategy_prompt.py`** — `build_strategy_system_prompt()` (~L161–186)
  assembles the LLM system prompt from the **live engine grammar**
  (`vocab.grammar_summary_for_prompt()`) + `StrategySpec.json_schema_for_llm()`. No canned
  examples, no KB. **This is where the `universe` block is taught.**
- **`app/strategy/generator.py`** — `generate_strategy(history, *, llm, system_prompt,
  max_repairs=2, max_output_tokens=8192, session_id)` → `tuple[StrategySpec | None,
  ValidationResult]`. LLM proposes JSON → `_parse_spec()` → `validate_spec()`; on failure it
  appends a repair message and retries. The **repair loop must cover universe errors.**
- **`app/strategy/spec.py`** — `StrategySpec` (Pydantic, `extra="forbid"`). `symbol: str`
  (L297). **`to_engine_yaml_dict()` (L385)** renders the inner `strategy:` mapping the engine
  loads, **formula mode only — explicitly no KB signal registry** (see the module docstring).
  Only the `"symbol"` line (L403) is instrument-specific; the rest of the body is
  symbol-agnostic. This is the "strategy template" the dynamic path stamps per member.
- **`app/strategy/validator.py`** — `validate_spec(...)` checks conditions compile against the
  **live engine** vocabulary (`app/strategy/vocab.py`). Reuse this for `screen` conditions.

### Backtest path
- **`app/api/v1/routes/backtest.py`** — fetches OHLCV for `spec.symbol` (single symbol, ~L111)
  via **`app/services/backtest/market_data.py`** (`StrategyMarketDataRequest`,
  `fetch_ohlcv_records()` — concurrent, chunked, cached), then calls the quant engine.
- **`quant_engine/main.py`** — `POST /run` (L177); `_execute_and_callback()` runs the backtest
  in a background executor; `_post_result()` posts results back with retry/backoff.
- **`quant_engine/engine/runner.py`** — `run_backtest(yaml_content, ohlcv_data, run_config,
  market_data_request, backtest_ref_id=None, reference_ohlcv=None, htf_ohlcv=None)` (L97).
  Steps: `_load_strategy_config_from_input()` → `load_ohlcv_data()` → optional
  `resample_ohlcv()` (intrabar) → indicators → `simulate_trades()` (~L389). **Single symbol
  throughout.**
- **`quant_engine/engine/loader.py`** — YAML → `StrategyConfig`; `cfg.symbol` (~L976). Must
  tolerate a `universe:` block in YAML for the dynamic path.
- **`quant_engine/engine/data.py`** — `load_ohlcv_data()` (L98) normalizes records → indexed
  DataFrame.
- **`quant_engine/engine/indicators.py`** — `add_all_indicators(df, config)`,
  `evaluate_indicator()` (L92). Vectorized, whole-column. **Reuse, do not fork.**
- **`quant_engine/engine/conditions.py`** — `compile_condition(str)` → compiled predicate with
  `.evaluate(df, i)`, `.indicator_refs`, `.raw`. The same compiler powers `screen` conditions.
- **`quant_engine/engine/simulator.py`** — `simulate_trades()` (L755); `Trade` dataclass
  (L60–84). The **per-symbol** entry/exit/SL/TP/trailing engine. The portfolio loop **wraps**
  this; if its per-bar logic isn't callable in isolation, do the **minimal** extraction with
  golden tests proving no behaviour change.

### Live path
- **`app/api/v1/routes/execution.py`** — `POST /strategy/evaluate/execute` (L33–72).
- **`app/services/execution/strategy_evaluator.py`** — `evaluate(req)` (L144): `_load_strategy`
  → `lookup_adapter_symbol` → `market_data_service.fetch_candles/fetch_ltp/fetch_circuit_limits`
  → `lookup_instrument_defaults` → exit phase → entry phase → entry gates. **Evaluates one
  symbol per call** — the live dynamic path drives it once per active member.
- **`app/services/execution/market_data_service.py`** — `fetch_candles()` (L82), cached by
  `(asset_class, symbol, timeframe)`; Upstox (equity) + Binance (crypto) clients.
- **`app/services/execution/risk_manager.py`** — per-strategy risk/sizing. Becomes
  portfolio-aware.
- **`app/services/execution/entry_gates.py`** — Phase-10 gates (backtest/live parity).

### Persistence & config
- **`app/db/models/strategy.py`** (`Strategy`, `Chat`, `Backtest`, `ChatMessage`),
  **`app/db/models/execution.py`** (`StrategyRiskConfig`, `InstrumentMetadata`,
  `ExecutionState`). Schema `ai_strategy`, Alembic in `migrations/`.
- **`app/services/execution/risk_execution_config_service.py`** —
  `RiskExecutionConfigSnapshot` + `rms_sources` provenance.
- **`app/core/config.py`** (settings), **`app/core/logging_setup.py`**,
  **`app/core/token_tracker.py`**.

### What we DROP (knowledge base — do not use)
The dynamic universe path must **not** import or depend on any of:
- `app/kb/**`, `stretus_knowledge_base/**`, `app/planner/**` (catalog signal picker, SDL).
- `app/services/discovery/scanner.py`'s `kb.stocks` coupling (`_select_universe`).
The current `scanner.py` resolves its candidate pool from `kb.stocks` — that is exactly the KB
dependency we are removing. We **reuse its proven async patterns** (concurrent fetch,
point-in-time evaluation, per-symbol failure tolerance, metric pre-computation) by **porting
them into a new KB-free `resolver.py`** whose pool comes from the universe-source registry
(§7), not the KB. `scanner.py` may remain for the legacy chat "pick-1" UX but is **not** on the
dynamic path.

---

## 1. Glossary
- **Universe rule** — declarative spec: source + screen + rank + take-N + eligibility + breadth
  gate + refresh cadence + max positions.
- **Resolver** — turns the rule into a ranked, eligible **set of symbols at a point in time** (a
  **snapshot**). KB-free.
- **Snapshot** — resolved membership at one `asof`, content-hashed and persisted.
- **Member** — a symbol currently in the resolved universe.
- **Portfolio Manager** — owns shared capital, the `max_positions` cap, sizing, and the single
  portfolio equity curve.
- **Strategy template** — the symbol-agnostic body from `to_engine_yaml_dict()`; stamped onto
  each member.
- **Tier A (screening)** — daily data, used only at the refresh cadence to rank candidates.
- **Tier B (execution)** — strategy-timeframe (or 1-min if intrabar) data, fetched lazily only
  for members, only for their membership windows.

---

## 2. Invariants (must hold in every phase)
1. **Reuse, don't rewrite.** A dynamic universe is the same template run per member under a
   portfolio layer. Do not fork `simulator.py`, `conditions.py`, `indicators.py`, or the body of
   `to_engine_yaml_dict()`.
2. **Static path untouched.** When `universe is None`, behaviour is byte-for-byte identical;
   all existing tests pass. Dynamic is additive and behind a feature flag.
3. **LLM-only fields.** The universe block is produced by the LLM via the system prompt — never
   regex-extracted (honors the project's LLM-only RMS rule).
4. **No look-ahead.** The resolver reads only bars with `timestamp ≤ asof`. Membership at T is a
   pure function of data available at T.
5. **Fail-closed eligibility.** Illiquid / halted / untradeable symbols are excluded regardless
   of rank. Stale or failed eligibility input excludes — never admits by default.
6. **Bounded memory.** Memory scales with active members (≤ `take`), not pool size or date
   range. Never materialize the full candidate × full-range high-frequency dataset.
7. **Survivorship honesty.** Until point-in-time constituents exist, backtests run
   `survivorship_mode="approximate"` and the payload says so explicitly.
8. **Reproducibility.** Every resolution is content-hashed and persisted. Same spec + same data
   window ⇒ identical snapshots ⇒ identical result.
9. **Shared capital + position cap live in the Portfolio Manager**, never in the per-symbol
   simulator.
10. **Backward-compatible payloads.** Dynamic responses extend, never break, existing
    single-symbol response schemas.
11. **KB-free.** No code on the dynamic path may import `app/kb`, `stretus_knowledge_base`, or
    `app/planner`. The candidate pool comes from the universe-source registry (§7), not
    `kb.stocks`. Screen conditions use the engine's formula compiler (already KB-free).
12. **Config-capped asset count.** A platform config limit
    (`DYNAMIC_UNIVERSE_MAX_ASSETS`, `DYNAMIC_UNIVERSE_MAX_POSITIONS`) hard-caps how many assets
    a dynamic strategy resolves and trades, regardless of the user's requested `take`. The cap
    is applied **after ranking** (top-N survive), is transparent (logged + surfaced + recorded
    in the snapshot), and applies identically in backtest and live (§7.1).

---

## 3. Architecture Overview

### 3.1 The template insight
`to_engine_yaml_dict()` already produces a body where only `symbol` is instrument-specific.
A dynamic-universe strategy = **the same template stamped onto each resolved member, run
together under one shared capital pool.** Everything else is plumbing.

### 3.2 The two-tier data model (load-bearing)
- **Tier A (screening):** runs only at the refresh cadence; daily bars + the screen's history
  window. Cheap, cacheable, precomputable.
- **Tier B (execution):** strategy-timeframe (or 1-min only on explicit intrabar request), only
  for members, only for membership windows, streamed and evicted.
This split is what makes 5,000-symbol universes feasible.

---

## 4. Data Architecture & Storage (production-grade)

### 4.1 Lazy, streaming `DataProvider`
A `DataProvider` abstraction (a `typing.Protocol`/ABC, so the store is swappable) that the
resolver and portfolio loop call on demand:
- **screening read** — daily bars for a candidate pool up to `asof` (cheap, cached, batched);
- **execution read** — strategy-TF (or 1-min) bars for a single symbol over a bounded window
  (fetched when a symbol becomes a member, evicted when it leaves and is flat);
- transparent caching + dedup so repeated backtests reuse data.
The "fetch all OHLCV and pass a dict into `run_backtest`" approach is **replaced** by this
provider for the dynamic path. The static path keeps its current loading.

### 4.2 Columnar OHLCV store
Beyond a few hundred symbols, broker-fetch + pandas does not scale. Stand up a local **columnar
OHLCV store** behind the `DataProvider` (start with **Parquet-on-disk**, upgrade path to
**ClickHouse** without touching callers — that is the point of the abstraction):
- range-sliced reads by `(symbol, timeframe, time range)`;
- daily + strategy-TF tiers ingested once and reused;
- 1-min ingested lazily, only for symbols/windows execution touches.
**Kafka is NOT required.**

### 4.3 Precomputed screening metrics (recommended)
A daily metrics cache (`relative_volume`, `rsi_14`, `atr_pct`, `distance_52w`, `ADV`, and the
new `pct_change`/`rel_strength`/`delivery`/`oi`) keyed `(symbol, date, metric, params_hash)`,
so the resolver reads precomputed values instead of recomputing per backtest.

### 4.4 Ingestion
Idempotent, resumable backfill + incremental top-up; bounded concurrency + backoff (port the
`scanner.py` concurrent-fetch pattern, KB-free). Respect broker/store rate limits.

### 4.5 Explicit scale guard
Before a dynamic backtest, compute the projected working set and **refuse or downgrade** runs
that would breach memory/time budgets (e.g. 1-min intrabar over 5,000 symbols × 2 years without
the store). Surface a clear, actionable error — never OOM.

---

## 5. Component Inventory

### Changed
| Component | Change |
|---|---|
| `app/strategy/spec.py` | Add universe models; `symbol` optional; add `universe`; one-of validator; branch in `to_engine_yaml_dict()` (body unchanged). |
| `app/core/config.py` | Add feature flag + asset-count caps `DYNAMIC_UNIVERSE_MAX_ASSETS` / `DYNAMIC_UNIVERSE_MAX_POSITIONS` (§7.1) + budget/scale-guard thresholds. |
| `app/strategy/validator.py` | Validate `screen` via the existing condition compiler; sanity-check `take`/`max_positions`. |
| `app/services/ai/strategy_prompt.py` | Teach the LLM to emit the universe block (LLM-only). |
| `app/strategy/generator.py` | Repair loop covers universe validation errors. |
| `quant_engine/engine/runner.py` | Add the portfolio backtest entry point wrapping per-symbol sim. |
| `quant_engine/engine/loader.py` | Tolerate a `universe:` block in YAML. |
| `app/services/execution/strategy_evaluator.py` | Drive per-member; route intents through the portfolio layer. |
| `app/services/execution/risk_manager.py` | Portfolio-aware: `max_positions`, shared capital, sector cap. |
| `app/services/execution/risk_execution_config_service.py` | Portfolio-scope fields + `rms_sources`. |
| `app/api/v1/routes/backtest.py`, `execution.py` | Branch dynamic → portfolio path. |

### New
| Component | Purpose |
|---|---|
| `app/services/universe/resolver.py` | KB-free: source→screen→eligibility→rank→take-N→breadth→snapshot. |
| `app/services/universe/sources.py` | Universe-source registry (constituents / exchange catalog / watchlist). |
| `app/services/universe/breadth.py` | Market-context/breadth engine (A/D, NH-NL, sector strength, market trend). |
| `app/services/data/provider.py` + store adapter(s) | Lazy, streaming, cached OHLCV access (Parquet→ClickHouse). |
| `quant_engine/engine/portfolio.py` | Backtest Portfolio Manager. |
| `app/services/execution/portfolio_manager.py` | Live Portfolio Manager. |
| `app/services/execution/universe_scheduler.py` | Live periodic re-resolution + spawn/retire. |
| Migrations | `universe_membership`, `universe_snapshots`, `universe_members`, daily-metrics cache. |
| Ingestion job | Populate/refresh the columnar store. |

> New code lives under a new `app/services/universe/` package (and `app/services/data/`) to keep
> it cleanly separate from the deprecated `app/services/discovery/` (KB) package.

---

## 6. Data Model / Schema (describe; implement per Alembic conventions)
- **`universe_membership`** — point-in-time constituents: `universe_id`, `instrument_id`,
  `valid_from`, `valid_to` (nullable). "Members on date D": `valid_from ≤ D AND (valid_to IS
  NULL OR valid_to > D)`.
- **`universe_snapshots`** — `strategy_id`/`run_id`, `asof_ts`, `members[]`, `ranking_metrics`,
  `snapshot_hash`, `survivorship_mode`.
- **`universe_members`** (live runtime) — `deployment_id`, `symbol`, `member_state`,
  `activated_at`, `status` (active/retiring/retired).
- **Daily metrics cache** — `(symbol, date, metric, params_hash) → value`.
- **Risk config extension** — `max_concurrent_positions`, `per_symbol_risk_pct`,
  `sector_cap_pct` with `rms_sources` provenance; per-member execution state under a portfolio
  parent.
All migrations idempotent + reversible; tenant-scoped; `ai_strategy` schema.

---

## 7. Universe Resolver Specification (KB-free)

Port the proven `scanner.py` patterns — concurrent fetch, point-in-time condition evaluation,
per-symbol failure tolerance, metric pre-computation — into `app/services/universe/resolver.py`
**without** the `kb.stocks` dependency.

Pipeline: **source → screen → eligibility (fail-closed) → rank → take-N (clamped to config cap,
§7.1) → breadth gate → hash → persist.**
- **Source (`sources.py`):** `index`/`sector`/`f_and_o` → `universe_membership` (point-in-time);
  `crypto_all` → exchange tradable-pairs catalog; `watchlist` → explicit list. No KB.
- **Screen:** boolean conditions in the **same engine grammar** as `entry_condition`
  (`conditions.compile_condition`); evaluated on daily (or `scan_timeframe`) bars up to `asof`.
- **Eligibility:** ADV floor (computed from OHLCV — no vendor), tradable, not-in-circuit;
  fail-closed.
- **Rank:** by `rank.by`/`order` (top-N): `rvol`, `pct_change` (gainers/losers), `rel_strength`,
  `momentum`, `atr_pct`, `distance_52w`, `delivery_volume*`, `oi_change*` (`*` need new feeds).
- **Asset-count cap:** after ranking, keep only `effective_take = min(spec.take,
  DYNAMIC_UNIVERSE_MAX_ASSETS)`; record `requested_take`/`effective_take`/`cap_reason` and warn
  when clamped (§7.1).
- **Breadth gate (`breadth.py`):** computed once per refresh over the Tier-A pool (A/D ratio,
  NH-NL ratio, sector strength, market trend); fail-closed.
- **Snapshot hash:** deterministic over `rule_hash + asof + sorted(members)` (reuse the SDL
  hashing style — port the hashing helper, not the SDL types).
- **Return:** members, per-symbol metrics, dropped-ineligible list, `survivorship_mode`.
- **Injectable `fetch_ohlcv`** (as `scanner.scan_universe` does) for testability.

---

## 7.1 Asset-Count Cap (config-driven hard platform limit)

The platform enforces a configurable ceiling on how many assets a dynamic-universe strategy may
actually select and trade, **independent of what the user requests**. If the user says
*"top 1000"* but the configured limit is 2, only **2** assets are used.

- **Config (`app/core/config.py`):**
  - `DYNAMIC_UNIVERSE_MAX_ASSETS` — hard cap on resolved members (effective `take`). Default
    conservative during rollout (e.g. `2`); raise via config as the system proves out — no code
    change.
  - `DYNAMIC_UNIVERSE_MAX_POSITIONS` — hard cap on concurrent positions (`max_positions`).
  - Platform-level for now; per-tenant / per-tier overrides are a follow-up (§19).
- **Spec keeps intent; platform clamps effect.** The LLM still writes the user's requested
  `take` into the `StrategySpec` (intent preserved — LLM-only rule honored). The **resolver**
  computes `effective_take = min(spec.take, DYNAMIC_UNIVERSE_MAX_ASSETS)` and keeps only that
  many. Likewise `effective_max_positions = min(spec.max_positions,
  DYNAMIC_UNIVERSE_MAX_POSITIONS, effective_take)`.
- **Clamp AFTER ranking, never before.** Screen + eligibility + rank run over the **full**
  eligible pool; the cap then selects the **top `effective_take`** by the ranking metric.
  (Capping before ranking would pick arbitrary names — forbidden.)
- **Transparent, never silently dropped.** The snapshot records `requested_take`,
  `effective_take`, and `cap_reason` (`"platform_limit" | "none"`); a `WARNING` is logged when
  clamped (`resolver|cap|requested=1000|effective=2|reason=platform_limit`); and the
  result/chat payload surfaces *"Requested 1000; platform limit applied → using 2."* The cap
  **clamps and proceeds** — it is not an error.
- **Validator:** flags a `spec.take`/`max_positions` above the configured ceiling with an
  **informational note** (not a hard reject) — the resolver clamps at runtime, so the limit can
  be changed via config without re-validating saved strategies.
- **Backtest + live parity:** the clamp lives in the resolver, which both paths call, so it is
  applied identically everywhere.

---

## 8. Backtest Design (lazy, streaming, portfolio-level)
Wraps — never replaces — per-symbol simulation:
1. Resolve candidate pool from `source` (point-in-time when available).
2. Walk a single unified clock, **streaming by time-chunk** for bounded memory.
3. At each refresh point, call the resolver using **only data ≤ that timestamp** (Tier A) →
   top-N snapshot; persist with hash.
4. **Diff membership.** Added: lazily load Tier-B data for the upcoming window (+ warm-up),
   initialize per-symbol state warmed from history. Removed: flatten (default, intraday) or
   hold-until-exit; **evict** their data once flat.
5. Per bar, evaluate each active member with the **existing** entry/exit/SL/TP/trailing/gates
   logic; emit intents.
6. **Portfolio Manager** arbitrates every intent: `max_positions`, shared capital, sector cap;
   size via `equal_weight` or `risk_based` (risk % of portfolio capital ÷ stop distance);
   record skips with reasons.
7. Produce **one** portfolio result: equity curve, Sharpe/Sortino/Calmar/maxDD/PnL, per-symbol
   attribution, all snapshots with hashes, `survivorship_mode`, skipped-entry counts,
   data-coverage summary.
Memory target: peak RAM scales with active members, independent of pool size and range.

---

## 9. Live Execution Design
1. **Universe Scheduler** (`universe_scheduler.py`): APScheduler in-process (no Celery/Kafka).
   Multi-replica safe via a **Postgres advisory lock** — exactly one replica fires each tick.
2. On each refresh: resolver on live data → snapshot (persisted + hashed). Diff vs
   `universe_members`.
3. Added → activate, warm indicators from recent history, create per-member sub-state.
4. Removed → square off (or hold-to-exit); mark retiring → retired.
5. Per bar/tick: drive `strategy_evaluator.evaluate()` once per active member; every intent
   passes through the portfolio-aware risk layer (above `risk_manager.py`) enforcing
   `max_positions` + shared capital.
6. Fan out bracket orders to existing Upstox/Binance adapters; idempotent client order IDs.
7. **Reconciliation:** on restart/failover, rebuild active membership + open positions from
   `universe_members` + broker state before resuming.

---

## 10. LLM Authoring
- `strategy_prompt.py`: when the user describes a **selection rule** ("most active", "top N
  by…", "above VWAP across NIFTY500") rather than a named instrument, emit a `universe` block
  and leave `symbol` null.
- Map intent → `rank.by` (most active→rvol/volume, biggest movers→pct_change, strongest→
  momentum/rel_strength, oversold→rsi, most volatile→atr_pct, near highs→distance_52w).
- Defaults: `take` from stated N (fallback 10); `max_positions` from stated cap (fallback =
  `take`).
- `screen` reuses the entry-condition grammar already injected.
- Repair loop surfaces and recovers from universe validation errors.

---

## 11. Risk & Provenance
- Portfolio-scope risk fields carry `rms_sources` provenance (`user|default|system_default`).
- Per-member risk inherits from the portfolio parent unless overridden.
- The Portfolio Manager is the **single authority** for capital, position count, and concurrency
  caps in both backtest and live (parity).

---

## 12. Engineering Standards (apply to ALL new and changed code)

### 12.1 Logging — beautiful, informative, structured, cheap
Match the existing house style and make logs a first-class debugging tool.
- **Two log channels, like the codebase already has:**
  - **Structured pipe-delimited logs** for machines/ops, mirroring `scanner.py`
    (`logger.info("resolver|begin|universe=%s|pool=%d|asof=%s", ...)`) and `runner.py`
    timing logs (`logger.info("⏱️ TIMING|step=resolve|duration=%.4fs|members=%d", dt, n)`).
  - **Human-readable `_emit(messages, "📊 …")` lines**, like `strategy_evaluator.py`, for the
    chat/UI trace where applicable.
- **Always use lazy `%`-formatting in logging calls** (`logger.info("...%s", x)`), never
  f-strings — args aren't rendered unless the level is enabled (performance + house style).
- **Every resolution logs:** `asof`, pool size, screened/eligible/taken counts, breadth verdict,
  `snapshot_hash`, `survivorship_mode`. **Every skipped entry logs** the reason
  (`position_cap` / `insufficient_capital` / `sector_cap`). **Every refresh diff logs** added /
  removed symbols.
- **Timing:** wrap each pipeline stage (resolve, screen, rank, Tier-B load, sim) with a
  `perf_counter` timing log, as `runner.py` does. Include row/symbol counts.
- **Use consistent prefixes** per module (`resolver|`, `portfolio|`, `scheduler|`, `provider|`)
  and a correlation id (`run_id`/`deployment_id`) on every line.
- **Levels:** `INFO` for lifecycle + counts, `DEBUG` for per-symbol detail, `WARNING` for
  fail-closed exclusions and degraded data, `ERROR` for refused runs / unrecoverable I/O. No
  secrets in logs.

### 12.2 Code quality & best practices
- **Type hints everywhere**; run `mypy`/`pyright` clean on new modules. Public functions have
  precise return types (`ResolvedUniverse`, not `dict`).
- **Pydantic v2 models with `ConfigDict(extra="forbid")`** for every new external/spec shape,
  matching `spec.py`. Frozen dataclasses for internal value objects (like `Candidate`).
- **Docstrings** on every module and public function: what, why, invariants honored, units, and
  failure behaviour — match the rich docstring style in `spec.py`/`scanner.py`/`types.py`.
- **Pure functions** for metric/screen/ranking math (no I/O), so they're trivially testable
  (mirror `scanner.py`'s `_safe_*` helpers).
- **Dependency injection / Protocols:** `DataProvider` and store adapters are interfaces;
  inject `fetch_ohlcv` for tests (as `scan_universe` does). No hidden globals.
- **No import cycles.** Keep the new `app/services/universe/` and `app/services/data/` packages
  free of heavy back-imports; late-import the engine in the resolver where needed (as
  `scanner.py` does for `engine.conditions`).
- **Constants over magic numbers** (warm-up windows, ADV window, budgets) in one place, surfaced
  via `app/core/config.py` settings where operationally tunable.
- **Small, single-responsibility functions and modules.** One concept per file.
- **Async/await** end-to-end on the I/O paths; never block the event loop.
- **Feature flag** (`app/core/config.py`) gates the whole dynamic path; default off.

### 12.3 Performance & scalability
- **Vectorized pandas** for indicators/metrics; never per-row Python loops in hot paths (reuse
  `indicators.add_all_indicators`).
- **Bounded concurrency:** `asyncio.gather` behind a `Semaphore` for fetches; tolerate
  per-symbol failures (port `scanner.py` behaviour). Respect rate limits with backoff.
- **Streaming + eviction:** never hold the full dataset; process by time-chunk; drop Tier-B
  frames for retired/flat members. Peak memory must be O(active members).
- **Cache aggressively, key precisely:** daily-metrics cache and DataProvider cache keyed by
  content; O(members) snapshot cost, O(pool) only for the cheap daily screen.
- **Avoid needless DataFrame copies**; use views/slices; pool/reuse buffers in the per-bar loop.

### 12.4 Maintainability & robustness
- **Errors are typed and fail-closed.** Define explicit exceptions
  (`ScaleGuardExceeded`, `UniverseSourceError`); never silently admit on uncertainty.
- **Backward compatibility:** dynamic response fields are additive; gate behind the flag.
- **Tests alongside code** (§16); golden snapshots for `to_engine_yaml_dict` and resolver
  output. Deterministic fixtures, no network in unit tests (inject `fetch_ohlcv`).
- **Observability built in, not bolted on** (§14): metrics + structured logs from day one.
- **Docs:** module-level design note in `docs/`, plus the operability runbook.

---

## 13. Failure Modes & Fail-Closed Behaviour
- **Resolver data gap** (symbol missing/stale): exclude, log `WARNING`, continue (preserve
  `scanner.py` per-symbol tolerance).
- **All-fetch-failure:** surface the root cause (network/auth), not "no matches" (preserve
  `scanner.py` error-surfacing).
- **Eligibility input stale:** exclude (fail-closed).
- **Live feed stale:** hold; place no new orders.
- **Scheduler missed/overlapping tick:** advisory lock + idempotent resolution; missed cadence
  resolves on next tick, logged.
- **Position cap / capital exhaustion:** skip with explicit reason; never exceed limits.
- **Scale-guard breach:** refuse with an actionable message.

---

## 14. Observability & Operability
- **Structured logs** (§12.1) for every resolution and skipped entry.
- **Metrics:** resolution latency, candidates scanned/passed/dropped, active member count,
  position-cap hit rate, data-cache hit rate, fetch failures, scheduler tick health.
- **Auditability:** snapshots + risk decisions persisted; a dynamic backtest is fully replayable
  from stored snapshots.
- **Runbook:** backfill the store; inspect/replay snapshots; pause a live dynamic deployment;
  advisory-lock behaviour under multi-replica.

---

## 15. Security & Multi-Tenancy
- All new tables carry tenant/owner scoping consistent with existing models; queries
  tenant-scoped.
- Store and caches partitioned/prefixed per tenant where applicable.
- No new component places orders or alters risk limits outside the existing
  authenticated/authorized path.

---

## 16. Testing & Validation Strategy (CI gates)
- **Backward compat:** all existing tests pass; static `to_engine_yaml_dict()` matches golden
  snapshots.
- **KB-free guard:** a test/lint asserts no dynamic-path module imports `app.kb`,
  `stretus_knowledge_base`, or `app.planner`.
- **Spec:** one-of validator (symbol XOR universe); dynamic spec round-trips; `json_schema_for_llm`
  includes `universe`.
- **Resolver determinism:** fixed fixtures → identical ranked members and snapshot hash.
- **Asset-count cap:** user `take=1000` with `DYNAMIC_UNIVERSE_MAX_ASSETS=2` → exactly 2
  members (the top-2 by rank), snapshot records `requested_take=1000`/`effective_take=2`, a
  clamp `WARNING` is logged, and the clamp is identical in backtest and live.
- **Look-ahead:** perturbing bars after `asof` changes nothing.
- **Survivorship:** seeded delisted name appears in a historical resolution, absent today.
- **Eligibility:** seeded illiquid/halted name dropped.
- **Portfolio cap:** N entries, `max_positions=k<N` → exactly k open, N−k skipped with reasons.
- **Capital sharing:** equity curve reflects the shared pool, not summed per-symbol.
- **Membership churn:** added/removed handled; removed flattened at refresh.
- **Memory/scale:** large-pool synthetic backtest stays within the RAM budget (assert peak
  working set bounded by members).
- **Live:** scheduler spawn/retire; advisory lock prevents double-fire; portfolio cap across
  members; restart reconciliation.

---

## 17. Phased Rollout
- **Phase A — Spec + Resolver + DataProvider interface.** Model change (flagged off), LLM
  prompt, KB-free resolver, lazy daily-screening reads. No execution changes.
- **Phase B — Portfolio backtest** with lazy/streaming Tier-B + Portfolio Manager + API branch +
  columnar store (Parquet first).
- **Phase C — Survivorship-safe constituents** (`universe_membership` + point-in-time source +
  survivorship test).
- **Phase D — Live** (scheduler, spawn/retire, portfolio-aware risk, OMS fan-out,
  reconciliation).
Each phase: feature-flagged, independently mergeable, one PR (or a small set), with code +
migrations + tests + a PR note listing which invariants (§2) it upholds and any caveats. Enable
for internal/canary strategies first.

---

## 18. Acceptance Criteria
- [ ] Static path byte-for-byte unchanged; all prior tests green.
- [ ] **No dynamic-path module imports the KB / planner packages** (enforced by test).
- [ ] Canonical example backtests end-to-end → one portfolio equity curve, snapshots hashed.
- [ ] 5,000-symbol / 2-year stress backtest runs within RAM and time budgets, never
      materializing the full high-frequency dataset.
- [ ] Look-ahead, determinism, survivorship-flag, eligibility, position-cap tests green.
- [ ] Live deployment resolves on cadence, spawns/retires members, enforces the portfolio cap,
      reconciles after restart.
- [ ] Config asset-count cap clamps resolved members **after ranking** (e.g. requested 1000 → 2),
      transparently logged/surfaced and recorded in the snapshot, identical in backtest and live.
- [ ] Scale-guard refuses infeasible runs with actionable errors.
- [ ] Structured + human logs and metrics present per §12.1/§14; runbook written.
- [ ] No Kafka/ClickHouse mandated beyond the columnar OHLCV store actually needed for scale.

---

## 19. Out of Scope (follow-ups)
Cross-strategy portfolio netting, correlation-based sizing beyond a simple sector cap,
options/futures universes, Redis distributed cache, distributed job queue, FPGA/HFT tier.

## 20. Deliverables
Per phase: implementation + migrations + tests + observability + docs (design note in `docs/` +
operability runbook). Final: end-to-end demo of the canonical and stress examples with the
reproducibility (snapshot-replay) proof.
