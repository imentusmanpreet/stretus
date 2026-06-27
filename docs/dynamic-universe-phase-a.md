# Dynamic Universe — Phase A design note

Phase A of `docs/dynamic-universe-implementation.md` (§17). It lays the **foundation**:
the spec model change, the config caps, the KB-free **resolver**, and the lazy
**DataProvider** interface — all **behind a feature flag, with the static single-symbol
path byte-for-byte unchanged** (Invariant 2). No execution code changes in this phase;
the portfolio backtest is Phase B, survivorship-safe constituents Phase C, live Phase D.

## What shipped

| Area | Change |
|---|---|
| `app/core/config.py` | Feature flag `dynamic_universe_enabled` (default **off**) + asset-count caps (`dynamic_universe_max_assets`/`_max_positions`, default **2**) + scale-guard thresholds (`_max_pool`, `_max_working_set`, `_adv_window`, `_warmup_bars`). |
| `app/strategy/spec.py` | New universe models (`UniverseSource`/`UniverseRank`/`UniverseEligibility`/`UniverseBreadthGate`/`UniverseRefresh`/`UniverseSpec`). `symbol` is now optional; added `universe`; a **one-of** validator (symbol XOR universe). `to_engine_yaml_dict()` delegates to a symbol-parametrised renderer so the static output is identical; `to_engine_template_dict(member_symbol)` stamps the same body per member (§3.1). |
| `app/strategy/validator.py` | `screen` conditions compile through the **same** engine compiler as `entry_condition` (refactored shared helper). `take`/`max_positions` over the platform cap → **informational note**, never a hard reject (§7.1). Source/scan-timeframe sanity checks; unwired rank-metric (delivery/oi) note. |
| `app/services/ai/strategy_prompt.py` | Teaches the LLM to emit a `universe` block (intent → `rank.by`, source mapping, defaults) and leave `symbol` null when the user names a **selection rule** (LLM-only — Invariant 3). |
| `app/services/universe/` *(new)* | KB-free package: `types` (value objects + `ResolvedUniverse` snapshot), `hashing` (rule + snapshot hash, ported SDL style), `metrics` (pure rvol/RSI/ATR%/pct_change/momentum/rel_strength/ADV), `sources` (source registry, injectable pool provider), `breadth` (A/D + NH-NL gate, fail-closed), `resolver` (the pipeline), `errors` (`UniverseSourceError`, `ScaleGuardExceeded`). |
| `app/services/data/` *(new)* | `DataProvider` Protocol (swappable store: injected fetcher → Parquet → ClickHouse) + `CachingFetchProvider` adapter with per-symbol `evict`. |

## Resolver pipeline (`resolve_universe`)

`source → screen → eligibility (fail-closed) → rank → take-N (clamped to cap) → breadth → hash`

- **No look-ahead** (Invariant 4): every fetch is bounded by `to=asof`; metrics read the
  last bar ≤ asof. Test: a post-`asof` spike bar changes neither membership nor hash.
- **Asset-count cap** (§7.1, Invariant 12): `effective_take = min(take, MAX_ASSETS)` applied
  **after** ranking; records `requested_take`/`effective_take`/`cap_reason`, logs a
  `WARNING`, and surfaces *"Requested N; platform limit applied → using M."*
- **Fail-closed** (Invariant 5): NaN/short eligibility input excludes; a failing or
  uncomputable breadth gate admits **no** members.
- **Reproducibility** (Invariant 8): `rule_hash` + `snapshot_hash` over sorted members.
- **Survivorship honesty** (Invariant 7): `survivorship_mode="approximate"` until Phase C.
- **KB-free** (Invariant 11): enforced by a static-import lint test over the new packages.

## Known Phase-A limitations (closed in later phases)
- Pool for `index`/`sector`/`f_and_o`/`crypto_all` comes from an **injected** `pool_provider`;
  the point-in-time `universe_membership` store is **Phase C**. Without a provider these
  sources raise `UniverseSourceError` (never a silent empty pool). `watchlist` works fully.
- Eligibility implements the **ADV liquidity floor** (computed from OHLCV, no vendor);
  tradable/circuit checks need feeds wired later. `delivery_volume`/`oi_change` ranking
  likewise await feeds (validator notes this).
- No portfolio backtest / live execution yet (Phases B/D). The columnar Parquet store sits
  behind the `DataProvider` seam and is introduced in Phase B without caller changes.

## Tests
`tests/test_universe/` — spec one-of + template render, resolver determinism/snapshot hash,
asset-count cap after ranking, no-look-ahead, ADV eligibility, screen filtering, breadth gate,
scale guard, source registry, validator cap-notes, and the **KB-free import guard**. The
existing strategy/discovery suites stay green (static path unchanged).

## Flag
Everything is gated by `dynamic_universe_enabled` (default off) and is additive; with the flag
off and `symbol` set, behaviour is identical to today.
