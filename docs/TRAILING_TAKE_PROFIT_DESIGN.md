# Trailing Take-Profit & Trailing Stop-Loss — Design & Implementation Plan

> Status: **Design (approved direction)** — pending OMS contract confirmation.
> Owners: Strategy / Execution / OMS teams.
> Goal: Introduce **trailing take-profit** (and align **trailing stop-loss**) across
> chat, backtest, and the live/paper order pipeline — efficient, optimised, and
> **backtest↔live accurate**.

---

## 1. Summary

A **trailing exit** is an exit level that follows the position's running extreme
(highest-high for longs, lowest-low for shorts), ratchets in one direction only
(never loosens), and fires when price reverses to touch it.

The key simplification: **trailing stop-loss and trailing take-profit are the same
primitive.** They differ only in (a) where they start, (b) when they activate, and
(c) the label we report. We build **one** trailing mechanism and reuse it.

### Division of responsibility (locked)

| Component | Responsibility |
|---|---|
| **Chat / Draft** | Let the user express trailing intent; persist it in the strategy draft & YAML. |
| **Backtest engine** | Compute trailing exits historically (source of truth for accuracy). Already has trailing SL; we add trailing TP. |
| **Eval engine** | **Generate orders only.** Attach the trailing spec to the order legs and return them in the API response. **It does NOT manage SL/TP/trailing exits.** |
| **OMS** *(external — NOT built by us)* | **Owns all price-based exits** (SL, TP, and trailing) for **both paper and live**. Monitors price, ratchets the trailing line, fires the exit, reports the fill. We only **define the contract** (§4) and hand it off. |

> **Decision:** Anything in the eval engine that checks SL/TP price hits is removed.
> The eval engine's job is: evaluate entry → build bracket order (with trailing spec)
> → return it. OMS handles the rest of the position lifecycle.

---

## 2. Core mechanism — the single trailing primitive

A trailing exit is fully defined by four parameters:

```jsonc
trailing = {
  "basis":      "percent" | "absolute",   // how distance is measured
  "distance":   2.0,                        // percent value, or absolute price units
  "activation": { "mode": "...", "value": ... },  // when trailing engages
  "anchor":     "running_extreme"           // peak-high (long) / trough-low (short)
}
```

### Semantics (canonical — backtest AND OMS must both implement THIS)

For a **LONG** position with entry fill price `E` and distance `d`:

1. Track running peak `P` = **max high** seen since entry (use **tick high** live;
   bar high in backtest). *Not* close / LTP.
2. Trailing stays **off** until activation is met:
   - `immediate` → on from entry.
   - `profit_pct = v` → on once `(price − E) / E × 100 ≥ v`.
   - `profit_absolute = v` → on once `price − E ≥ v`.
3. After activation, each tick/bar: `line = max(line, P − d)` — **ratchet**; the
   line never moves down.
4. Fire the exit when `price ≤ line`.

**SHORT** is the exact mirror: trough `P` = **min low**; `line = min(line, P + d)`;
fire when `price ≥ line`.

### Trailing SL vs Trailing TP — same engine, different framing

| | Trailing **Stop-Loss** | Trailing **Take-Profit** |
|---|---|---|
| Starts from | below entry (loss side) | in profit, after activation |
| Typical activation | immediate | after a meaningful gain (e.g. +5%) |
| Order type (live) | trigger / stop-market | trigger / stop-market (NOT a resting limit) |
| Reported exit reason | `TRAILING_STOP` | `TRAILING_TAKE_PROFIT` |
| Underlying math | identical | identical |

> **Important:** A trailing TP is **not** a resting LIMIT order — it fires on a
> pullback from the peak, which is mechanically a **trigger/stop** order sitting in
> profit territory. The TP leg's `order_type` changes accordingly when trailing.

---

## 3. Backtest ↔ Live parity (the accuracy contract)

This is the heart of "accurate". Rules:

1. **One definition.** The semantics in §2 are the single source of truth. The
   backtest implements them; the OMS contract (§4) is derived from them.
2. **Trail on extremes, not LTP.** Backtest tracks bar high/low; the OMS must track
   **tick high/low**. If the OMS trailed on LTP only, it would miss wick peaks the
   backtest captures and drift away from backtest results.
3. **Phase 1 = `percent` everywhere.** ATR / EMA / chandelier trailing require
   indicators the OMS cannot compute; they stay backtest-only (research) and are
   **not** sent live until/unless the OMS can support them. Percent keeps
   backtest == live trivially.
4. **Run backtest in 1-minute intrabar mode** for trailing strategies so its
   extremes are as fine-grained as practical. (Already the default for timeframes
   coarser than 1m.) Live ticks are finer than 1-min bars, so live is marginally
   more reactive — this is the closest achievable parity and is acceptable.

---

## 4. OMS Contract (hand this to the OMS team)

> **Scope note:** The OMS is **external — we do not build it.** This section is the
> spec we hand to the OMS team. Our deliverable is (a) this contract and (b) orders
> that carry the `trailing` block. Implementing the trailing behaviour itself is the
> OMS team's responsibility.

The eval engine attaches a `trailing` block to an order leg. The OMS must honour it.

### 4.1 New field on the order leg

```jsonc
"trailing": {
  "enabled": true,
  "basis": "percent",            // Phase 1: "percent" (optionally "absolute")
  "distance": 2.0,               // percent = 2% ; absolute = price units
  "activation": {
    "mode": "profit_pct",        // "immediate" | "profit_pct" | "profit_absolute"
    "value": 5.0                 // null when mode = "immediate"
  },
  "anchor": "running_extreme",   // peak-high (long) / trough-low (short)
  "fire": "market"               // exit order type when the line is touched
}
```

### 4.2 Behaviour the OMS must implement

> Given an order leg with `trailing.enabled = true`, for the parent position:
> 1. Entry fill price `E`, distance `d` (from `basis`/`distance`).
> 2. Track running extreme from **tick** high (long) / low (short) — **not LTP**.
> 3. Keep trailing **off** until `activation` is satisfied (measured from `E`).
> 4. After activation, each tick: ratchet the line — `max(P − d)` long /
>    `min(P + d)` short; never loosen.
> 5. When price reverses to the line, **fire** the leg (`fire = market` → market
>    exit), and cancel the sibling OCO leg.
> 6. Report the fill back: fill price, time, and `"trailing_exit": true` so we can
>    label the trade `TRAILING_STOP` / `TRAILING_TAKE_PROFIT`.

### 4.3 OMS questions to confirm (blocking Phase 0)

1. **Trail reference:** tick high/low. → **Recommended & required for parity.**
   Confirm OMS can track tick extremes.
2. **Fire type:** MARKET on touch (recommended for Phase 1 — simple, deterministic
   fill) vs trigger-limit (better price, risk of miss). → **Recommend MARKET.**
3. **Both paper and live** routed through OMS trailing. Confirm paper-mode OMS
   exists / will be built.

---

## 5. Data-model changes

### 5.1 `OrderLeg` — add trailing spec
`app/schemas/execution.py` (`OrderLeg`, ~line 363)
- New optional field `trailing: Optional[TrailingOrderSpec] = None`.
- New model `TrailingOrderSpec` matching §4.1.

### 5.2 `OpenPosition` — no SL/TP-management state needed
`app/schemas/execution.py` (`OpenPosition`, ~line 272)
- **No** peak/trail-level fields are added here — OMS owns that state, not us.
- (Optional) drop reliance on `stop_loss_price` / `take_profit_price` for exit
  decisions in the eval engine (see §6).

### 5.3 Backtest config — add trailing TP spec
`quant_engine/engine/loader.py` (`StrategyConfig`)
- New field `trailing_take_profit_spec: dict | None` parsed from YAML
  (`strategy.trailing_take_profit` / `risk.trailing_take_profit`), mirroring the
  existing `trailing_stop_spec`.

---

## 6. Eval-engine change — remove SL/TP management

`app/services/execution/strategy_evaluator.py`

Current `_run_exit_phase` (~line 520) checks, per position:
1. Stop-loss hit  ← **REMOVE** (OMS owns it)
2. Take-profit hit ← **REMOVE** (OMS owns it)
3. Exit signal (indicator/formula based)
4. Time-based exit (max holding)

**Plan:**
- Delete steps 1 & 2 (the `check_stop_loss` / `check_take_profit` branches at
  lines ~561–572).
- **Keep** steps 3 & 4 — exit-signal and time-based exits require strategy
  intelligence the OMS does not have, so the eval engine still emits these as exit
  instructions. *(Confirm this split — see Open Decisions.)*
- Entry phase unchanged, except the bracket it builds now carries the `trailing`
  spec on the relevant leg.

> Net effect: the eval engine becomes a **pure decision/order generator**. It does
> not track SL/TP price hits. OMS owns price-based exits end-to-end.

---

## 7. Phased implementation plan

Build order proves accuracy first (backtest), then lets users express it (chat),
then ships execution (orders → OMS).

### Phase 0 — Contract freeze (no code)
- Confirm OMS answers in §4.3.
- This document is the canonical spec.

### Phase 1 — Backtest (accuracy foundation) 🟩
| Task | File | Detail |
|---|---|---|
| Parse trailing-TP config | `quant_engine/engine/loader.py` | `StrategyConfig.trailing_take_profit_spec`; YAML parsing mirroring `trailing_stop_spec` |
| Trailing-TP math | `quant_engine/engine/simulator.py` (~`_compute_trailing_floor_long`, line 498) | Mirror for the profit side (long + short) per §2 semantics |
| Exit logic | `quant_engine/engine/simulator.py` (~line 1430 TP block) | Make TP trailing-aware: ratchet + activation; cover both intrabar (1m) and strategy-bar paths |
| New exit reason | `quant_engine/engine/simulator.py` (~line 1596) | `TRAILING_TAKE_PROFIT` + `exit_reason_detail` payload |
| Tests | `tests/test_quant_engine/` | Construct a path that trails up then pulls back; assert `exit_reason == "TRAILING_TAKE_PROFIT"` and parity of levels |

### Phase 2 — Chat / Draft / Validation 🟦 ✅ DONE
| Task | File | Status |
|---|---|---|
| Spec model | `app/strategy/spec.py` | ✅ `TrailingTakeProfit` model + `StrategySpec.trailing_take_profit` + render in `to_engine_yaml_dict()` |
| Vocabulary | `app/planner/engine_contract.py` + `app/strategy/{engine_bridge,vocab}.py` | ✅ `TRAILING_TAKE_PROFIT_TYPES` (Phase 1) + `trailing_take_profit_types()` bridge |
| Validation | `app/strategy/validator.py` | ✅ rejects non-engine-legal type; rejects trailing-stop + trailing-TP together (mirrors loader guard) |
| Draft persistence | `app/services/strategy/builder.py` | ✅ `trailing_take_profit_spec` field + `apply_signal_plan` + `merge_preview` + `to_draft_json` + `to_yaml_dict` |
| Direct flow | `app/services/chat/direct_strategy_flow.py` | ✅ spec → builder + spec → plan |
| LLM system prompt (direct path) | `app/services/ai/strategy_prompt.py` | ✅ **HARD RULE 5** teaches the trailing family (trailing_stop + trailing_take_profit): percent-only, distance_pct=give-back, activate_after_pct=profit threshold, and that BOTH may be set together (staged trail), avoiding a dominated/redundant pair. This is the prompt that drives the LLM-only `StrategySpec` generation (`use_direct_strategy_path`). |
| Schema descriptions | `app/strategy/spec.py` | ✅ `TrailingTakeProfit` fields carry LLM-facing descriptions (auto-included in `json_schema_for_llm()`) |
| Agent prompt | `app/services/agent/prompt.py` | ✅ conversational parse guidance |
| Tests | `tests/test_strategy/test_trailing_take_profit.py` | ✅ 12 tests (spec render, builder round-trip, validator, loader round-trip) |

**LLM-only path (primary):** `use_direct_strategy_path` → `direct_strategy_flow` → `generator.generate_strategy`
emits a `StrategySpec` from the system prompt above. The LLM now understands trailing take-profit end-to-end
(prompt rule + schema + descriptions → validator → builder → engine YAML → simulator). No regex involved.

**Note on regex:** `builder.extract_strategy_details` (regex RMS extractor) is still *physically* called at
`chat_service.py:3255`, but it writes to a throwaway `parsed_builder`; in the direct LLM path the spec is
authoritative, so it does not drive RMS. Deliberately NOT extended for trailing-TP (project direction is
LLM-only extraction). Full removal of the regex extractor is a separate cleanup, out of scope here.

### Phase 3 — Order schema + builder 🟩 ✅ DONE
| Task | File | Status |
|---|---|---|
| Leg field + model | `app/schemas/execution.py` | ✅ `TrailingOrderSpec` + `OrderLeg.trailing`. `SlTpConfig` carries `trailing_take_profit`/`trailing_stop`, allows `take_profit_pct=0`, validator: "static target OR trailing"; both trailing legs may coexist (dominated-line warning, not blocked) |
| Order build | `app/services/execution/trade_manager.py` | ✅ `build_bracket_order(trailing_take_profit=, trailing_stop=)` attaches the block; trailing-TP leg switches LIMIT→trigger (`TAKE_PROFIT` crypto / `SL-M` equity) + `trigger_price`; no-trailing path byte-for-byte legacy |
| Eval engine | `app/services/execution/strategy_evaluator.py` | ✅ reads trailing (`_parse_trailing_order_spec`), preserves `take_profit_pct=0`, passes specs into `build_bracket_order` |
| Tests | `tests/test_api/test_trailing_bracket_order.py` | ✅ 4 tests |

**Remaining hookup:** the persisted strategy_config the eval engine reads must include the
`trailing_take_profit` block (reads `raw["trailing_take_profit"]` / `sl_tp.trailing_take_profit`).
Verify the chat→strategy persistence writes it there; else add it in the assembler.

### Phase 4 — Eval engine slim-down 🟦
| Task | File | Detail |
|---|---|---|
| Remove SL/TP checks | `app/services/execution/strategy_evaluator.py` (`_run_exit_phase`, 561–572) | Delete SL & TP price-hit branches |
| Keep signal/time exits | same | Steps 3 & 4 remain *(pending confirmation)* |

### Phase 5 — Hand-off + reconciliation 🟨
*(OMS implementation is external — not our scope. Our work here is the hand-off and consuming what the OMS reports back.)*
| Task | Where | Detail |
|---|---|---|
| ~~Implement contract~~ | ~~OMS~~ | **External — OMS team, per §4. Not us.** |
| Hand off contract | — | Share §4 with the OMS team |
| Reconcile fills | execution ingest (ours) | Map OMS `trailing_exit: true` → `TRAILING_STOP` / `TRAILING_TAKE_PROFIT` on the trade record |
| Reporting (optional) | `quant_engine/engine/metrics.py` | Count trailing-TP exits in the breakdown |

---

## 8. Open decisions

1. **Eval-engine exit scope:** Confirm the eval engine **keeps** exit-signal and
   time-based exits (recommended — OMS can't evaluate indicator logic), while OMS
   takes SL/TP/trailing. *(Default assumed: keep.)*
2. **Trailing SL + Trailing TP together:** ✅ **ALLOWED.** A position may carry a
   trailing stop leg AND a trailing take-profit leg at once — they don't conflict.
   The engine (and the OMS) trail **both legs independently** and exit on whichever a
   reversal reaches first; **OCO** then cancels the sibling. With different
   distances/activation thresholds this yields a **staged trail** (e.g. a wide trail
   early, a tight trail once deep in profit). The one degenerate case — a *dominated*
   line that can never fire (always tighter AND activates no later than the other) —
   is surfaced to the user as a **non-blocking warning** (`dominated_trailing_line`),
   not rejected. **OMS impact:** the bracket may now carry a `trailing` block on BOTH
   the stop-loss leg and the take-profit leg; honour each independently under OCO.
3. **`absolute` basis:** Ship `percent` only in Phase 1; add `absolute` if needed.

---

## 9. Glossary

- **Ratchet** — the trailing line only ever tightens (moves toward profit); it never
  moves back.
- **Activation** — the profit threshold that must be reached before trailing engages.
- **Running extreme** — the highest high (long) / lowest low (short) since entry, the
  anchor the trailing line follows.
- **OCO** — one-cancels-other; when one bracket leg fills, its sibling is cancelled.
