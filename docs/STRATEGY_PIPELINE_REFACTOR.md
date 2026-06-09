# Strategy Pipeline Refactor — Design Doc

**Status:** Proposed (no code changed yet)
**Author:** audit + design pass, 2026-06
**Goal:** Make signal selection and parameters driven *only* by user intent, verified deterministically, with no hardcoded signal tables and no silent substitution. Fixes the failure families found in tester review.

---

## 1. Problem statement

Today a user prompt can produce a strategy that (a) drops the user's own signals, (b) invents signals/params the user never asked for, (c) is logically impossible, (d) disagrees with itself (wrapper symbol ≠ code symbol, market ≠ asset), (e) loses earlier edits across turns, and (f) emits values the backtest engine rejects. Root causes:

1. **Two competing pipelines.** An LLM "SDL selector" (modern, dynamic) and a legacy regex/keyword pipeline (`constraint_compiler.py`, `catalog_signal_picker.py`, `semantic_extractor.py`, builder regex). They are reconciled ad-hoc.
2. **The legacy pipeline is gated by an env flag** (`SDL_SELECTOR_ENABLED`). When off / on failure, `sdl_json` is `null` and the builder falls back to a hardcoded stop loss.
3. **Hardcoded keyword tables decide which signals survive** (`_subtract_unrequested_signals` + `_FAMILY_KEYWORDS`/`_SIGNAL_TO_FAMILY`). Incomplete tables → user signals dropped, unmapped concepts silently replaced with generic EMA/RSI.
4. **No honest verification gate.** A satisfiability checker exists but is unwired; contradictions and impossible inputs pass; provenance over-claims "user"; the planner emits anchors the engine rejects.
5. **Two stores for risk params drift** (`builder.take_profit` attr vs `risk_execution_config[...]`), and **state merge is replace-not-merge** across turns.

### Design goals
- **One pipeline:** LLM proposes the full strategy; deterministic code verifies and renders. No second judgment engine.
- **Catalog is the single source of signal knowledge** (data, not Python). Adding a signal never means editing compiler code.
- **Never drop a user signal by keyword heuristic.** Never silently substitute an unmapped concept — ask instead.
- **One source of truth** for params and for cross-turn state.
- **A hard verification gate** that blocks impossible / inconsistent strategies before the user sees them.
- **One shared contract** between planner and `quant_engine` (enums imported, not duplicated).

### Non-goals (this refactor)
- Adding new indicator math to `quant_engine`. (Catalog gaps like options-OI/gamma are tracked but out of scope; the gate will *honestly decline* them rather than fake them.)
- Rewriting the chat state machine wholesale. We fix specific seams.

---

## 2. Current architecture (as-is)

```
prompt ─► AgentRouter.decide (LLM tool choice)         app/services/agent/router.py
        ─► [Engine A] sdl_selector.compile_to_sdl       app/planner/sdl_selector.py   (LLM → SDL JSON)
              gated by SDL_SELECTOR_ENABLED (:598)
        ─► [Engine B] catalog_signal_picker / 
             constraint_compiler / semantic_extractor    app/planner/*.py            (regex/keyword)
        ─► builder.apply_signal_plan                     app/services/strategy/builder.py
        ─► compiler.compile_sdl → strategy_config        app/planner/compiler.py
        ─► quant_engine loader                           quant_engine/engine/loader.py
```

Key as-is facts (verified):
- `sdl_json` is `null` whenever `builder._sdl is None` — [builder.py:1714](../app/services/strategy/builder.py).
- Hardcoded default SLs are **inconsistent**: `0.25` ([builder.py:1074](../app/services/strategy/builder.py)), `1.5` ([compat.py:44](../app/kb/compat.py), [chat_service.py:4380](../app/services/chat/chat_service.py)), `2.0` ([builder.py:1109](../app/services/strategy/builder.py)), `0.25` ([risk_manager.py:49](../app/services/execution/risk_manager.py)).
- `_subtract_unrequested_signals` ([constraint_compiler.py:596](../app/planner/constraint_compiler.py)) deletes signals by keyword-family match.
- `_merge_sdl` is a full replace ([sdl_selector.py:581](../app/planner/sdl_selector.py)); `reconcile` downgrades unmentioned fields ([sdl_selector.py:717](../app/planner/sdl_selector.py)).
- Param double-store with `if attribute is None` mirror guard ([builder.py:2688](../app/services/strategy/builder.py), [chat_service.py:3327](../app/services/chat/chat_service.py)).
- Engine accepts only 4 SL anchors ([quant_engine/engine/loader.py:163](../quant_engine/engine/loader.py)); planner emits more (e.g. `vwap_deviation`).

---

## 3. Target architecture (to-be)

**Propose → Verify → Render.** One path, three stages, each with a single responsibility.

```
prompt + prior SDL (if modify)
        │
        ▼
[1] PROPOSE   sdl_selector  (LLM, the ONLY interpreter)
        │     → SDL draft + provenance + unmapped_details
        ▼
[2] VERIFY    strategy_gate  (NEW, deterministic, blocking)
        │     reconcile + satisfiability + contradiction + contract + sanity
        │     → either: clarifying questions back to user
        │        or: a verified, engine-safe SDL
        ▼
[3] RENDER    compiler → strategy_config → quant_engine
              (thin, generic; reads catalog metadata only)
```

- **No Engine B judgment.** `constraint_compiler`'s signal-dropping (`_subtract_unrequested_signals`) and keyword tables are deleted. The legacy picker/semantic-extractor become optional *fallback proposers* only behind a flag, never editing the verified SDL.
- **`sdl_json` is never null.** Remove the `SDL_SELECTOR_ENABLED` fork from the happy path; on LLM failure we return a clarify/error state, not a silent legacy strategy.
- **Catalog is authoritative.** The gate and compiler read signal facts (params, ranges, contradicts, comparison kind, engine anchor support) from cards, not from Python constants.

---

## 3a. Input model: infer, don't interrogate

The current flow treats six fields (symbol, timeframe, objective, sentiment, experience, goal) as a **mandatory collection gate** before planning — the source of the "Which Indian stock…" loops and the weather→stock dead-ends. New model: **only the symbol is conditionally required; everything else is inferred or defaulted, shown transparently, and editable — never a blocking question.**

### Per-field resolution ladder (applied by the LLM proposer + gate, never by hardcoded per-market defaults)
```
1. USER-STATED   → verbatim, provenance = "user"
2. INFERRED      → derived from what the user DID say, provenance = "inferred"
3. SMART DEFAULT → last resort, explainable, provenance = "default"
```

| Field | Required? | Resolution |
|---|---|---|
| **symbol** | only to *backtest* | infer from prompt examples; else ask once, gently; logic can be built first and symbol requested at backtest time |
| **asset_class / market** | n/a | **derived** from the resolved symbol (deterministic) and **overwrites** any LLM guess — never a default |
| **timeframe** | no | infer from strategy type (ORB→5m, positional/swing→1d, scalping→1m); else default; always editable |
| **sentiment / direction** | no | derive from the signals themselves ("sell when EMA crosses below" = short) |
| **objective** | no | derive from timeframe + language |
| **experience** | **no — never ask** | risk-conservatism knob only; default "intermediate", adjust caps silently |
| **goal** | no | free-form; tuning hint only |

Implementation notes:
- The SDL proposer already sees the whole prompt and already infers most of these. "Let the LLM decide" largely means **stop overriding its inference with hardcoded market defaults**, and **tag provenance honestly** so the UI can surface "I assumed 5m — change it?" non-blockingly.
- Replace the scattered `0.25 / 1.5 / 2.0` SL defaults and the per-market default tables in `app/kb/compat.py` with one resolver that emits `default`-tagged values the gate can surface for optional confirmation.
- Remove the upfront collection gate in the chat flow; plan as soon as there is enough to plan, asking at most one genuinely-needed question.

## 4. The extended signal card schema

Cards live in `app/kb/signals/*.yaml`. Today they have `formula`, `match_when`, `params`, `contradicts`. We add the metadata the gate/compiler need so behavior is data-driven:

```yaml
name: price_below_ema
formula: "CLOSE < EMA({window})"
direction: bearish              # for direction coherence checks
comparison:
  lhs: price                    # price | indicator
  rhs: ema                      # what the lhs is compared against
  op: lt                        # gt | lt | cross_up | cross_down
params:
  window: { type: int, default: 20, min: 2, max: 400 }   # ranges => validation, not silent defaults
contradicts: [price_above_ema, ema_cross_up]             # already exists; now ENFORCED
engine:
  kind: filter                  # trigger | filter | exit
  anchors_supported: []         # for risk cards: the exact engine anchor ids this maps to
provenance_keys: [legs.0.entry.filters.N]                # canonical dotted paths
```

**Good news — most of this already exists.** Cards today already carry `direction`, `kind`/`roles` (trigger/filter/exit), `contradicts`, `pairs_well_with`, `experience_fit`, `intent_fit`, and `match_when`+`extract_params`. The refactor only *adds* three things: `comparison{lhs,rhs,op}`, param **ranges** (min/max), and `engine.anchors_supported`. The loader (`app/kb/loader.py:134`) is Pydantic `extra="forbid"`, so the schema in `app/kb/schemas.py` is extended first, then the YAML fields.

Why this kills bugs:
- **"EMA above price" vs "price above EMA"** becomes a `comparison{lhs,rhs,op}` lookup, not LLM guesswork — and lets the gate *auto-repair* a mis-map deterministically (see §5).
- **Param ranges** mean RSI threshold 55/65 are valid (no more `{50,60}`-only dict); out-of-range values are *clamped to nearest valid with a non-blocking note*, never silently snapped and never a raw error.
- **`contradicts`** is finally enforced by the gate (and the ~20% of cards with empty lists — `volume_spike`, `adx_strong_trend` — get backfilled).
- **`engine.anchors_supported`** is the single place the planner learns what the engine accepts → no `vwap_deviation` reaching the engine.

### 4b. Card-correctness fixes (independent of schema work)
The audit found cards whose *formula is wrong for the name* — so even a correct LLM mapping yields wrong logic. These are fixed in the YAML directly:
- `macd_bullish_cross`/`macd_bearish_cross`: formula is `MACD > 0`/`< 0` (zero-line), not a signal-line crossover. Fix to `MACD_LINE` vs `MACD_SIGNAL` cross.
- `rsi_cross_up`/`rsi_cross_down`: level checks, not crossovers. Fix to `RSI > t AND PREV(RSI) <= t` (or rename to a regime card).
- `is_above_sma` vs `price_below_sma`: broken mirror/contradicts/mirrors_to refs and inconsistent prefix; normalize naming and refs.
- `vwap_zscore_*` (not a true z-score), `donchian_breakout_*` (regime/event mislabel), `high_delivery_volume` (duplicate of `volume_spike`), `rsi_oversold.pairs_well_with` → non-existent `volume_dry_up`.

### 4c. Coverage gaps (concepts with no card → today silently faked)
Add cards (or have the gate decline honestly when truly unsupported):
- `volume_above_average` (plain `VOL > SMA(VOL)`, no multiplier) and `volume_dry_up` (`VOL < SMA(VOL)`).
- `false_breakout`/liquidity-sweep (break level then reclaim).
- relative strength as an entry **trigger** (exists only as filter).
- explicit HTF/multi-timeframe confirmation.
- options/OI/gamma stay out of scope — the gate says so plainly instead of substituting EMA/RSI.

A one-time migration script backfills these fields for the 124 existing cards (most are mechanical from the existing `formula`).

---

## 5. The verification gate (NEW: `app/planner/strategy_gate.py`)

Single entry point, deterministic, runs after PROPOSE and before RENDER on **both** create and modify turns.

```python
def verify(sdl: SDL, *, symbol_ctx: SymbolContext, catalog: CatalogMenu) -> GateResult:
    # GateResult = (repaired SDL, notes[])  OR  one Clarification  OR  (rare) BlockingError
```

### Gate philosophy: repair-first, ask-rarely, never-error
The gate keeps bad strategies away from the user **by fixing or asking — never by erroring**. Response priority:
1. **Auto-repair, silently (default).** Deterministic fixes from card/symbol metadata: market/asset from symbol; anchor→engine-supported mapping; cosmetic param normalization (§5d); drop a duplicate restatement; fill inferred timeframe/direction. A mis-mapped signal is re-mapped **only when the proposer's intent stub proves it** (§5b.5); otherwise it becomes a note or clarification. User sees a correct strategy.
2. **Inline note, non-blocking.** "Set timeframe to 5m for this ORB setup; assumed intermediate experience — tap to change." Strategy still shown and usable.
3. **One gentle question, rare.** Only for genuine, costly ambiguity with no safe default ("both long and short — both legs, or one?"), pre-filled with a recommended answer so "yes" works.
4. **Never** raw engine errors, JSON blobs, "symbol not found" dead-ends, or "captured X of Y" walls.

True hard-stops are reserved for genuinely unrunnable-and-unrepairable cases (very rare) and are phrased as "here's what I *can* build instead." Impossible *user* asks (RSI>90∧<10, bullish+bearish, 100% risk / 0% drawdown) are reframed/capped with a note or one question — not errored.

Checks (each maps to tester bugs; "→" shows the default response = repair unless noted):

| Check | Implementation | Kills |
|---|---|---|
| **Market/symbol reconciliation** | force `context.market`, `universe.asset_class`, and (if user gave one) the chart timeframe to agree with the resolved symbol; if symbol absent/ambiguous → clarify | #8 wrapper≠code, crypto-for-INFY |
| **Satisfiability** | wire existing `check_entry_satisfiable` ([condition_satisfiability.py](../app/planner/condition_satisfiability.py)); 0 fires → block | #1 EMA9<>EMA21, RSI>90∧<10 |
| **Contradiction** | enforce each card's `contradicts`; also direction coherence (`direction` vs `sentiment` vs signal directions) | #1, MACD bull∧bear, short+bullish |
| **Engine contract** | validate SL/TP `type`/`anchor` against card `engine.anchors_supported` + a **shared enum** imported by `quant_engine`; unmappable → clarify | #13 vwap_deviation |
| **Param ranges** | every signal param within card min/max; else clarify with the offending value | hallucinated 1.5 multiplier, out-of-range windows |
| **Input sanity** | risk%≤cap, drawdown>0, win-rate plausible, holding-period≠timeframe, trade-type valid | #10, #11 |
| **Unmapped honesty** | if `unmapped_details` non-empty OR a concept had no card, **do not substitute** — return it as a clarification/limitation | #2, #3 order-blocks/RS/wick |
| **Provenance honesty** | downgrade any field not traceable to user words from "user" to "default"+clarification (inverse of today's over-claim) | #15 "100% captured" while wrong |

Output contract:
- **(repaired SDL, coverage report)** → proceed to RENDER; notes render as small non-blocking "ℹ️ assumed/adjusted" chips on the strategy card.
- **one clarification** → asked only when truly ambiguous, pre-filled with a recommended answer (reusing existing `clarifications_needed` UI); the in-progress strategy is still shown.
- **blocking error** → rare; phrased as "I can't build this exactly as stated — here's the closest I can do," never a raw error.

### 5b. Gate invariants (so the gate never becomes a second planner)
1. **Structure-only.** The gate acts *only* on structured artifacts — the SDL, provenance, card metadata, the engine contract, and the proposer's per-signal intent stub. It **must never re-read the user's prose.** Re-parsing text = becoming a second planner.
2. **Proposer emits an intent stub** per signal: `{user_span, intended: {lhs, rhs, op}}`. This is how the gate adjudicates mapping errors without reading prose.
3. **Allowed silent repairs** (deterministic, meaning-preserving): symbol→market/asset, alias resolution, engine-anchor normalization, provenance correction, duplicate cleanup, deterministic default fill.
4. **Never silent**: flipping bullish↔bearish, swapping one trading concept for another, inventing indicators, converting unsupported concepts to EMA/RSI. These are *unmapped* or a *clarification*, never a quiet substitution.
5. **Mapping-error repair is gated by proof.** `ema_above`→`price_below_ema` is silent-safe *only* when the intent stub's `intended.rhs` (price/close) disagrees with the card's `comparison.rhs` (EMA) **and** a card matching the intended comparison exists. Otherwise → note or clarification.
6. **Unadjudicable contradiction → bounded self-heal.** A logical contradiction the gate can't attribute to a specific field (e.g. `ema_above ∧ ema_cross_down` with no decisive intent stub) triggers **one** re-proposal pass (the contradiction is fed back to the proposer); if it's still inconsistent, ask **one** clarification. Never silently pick a side.

### 5c. Intent Coverage Report (gate output, not LLM self-report)
Computed **deterministically by the gate** from reconciled provenance — replaces today's misleading self-graded "100% captured". Each entry is `{path, value, source, evidence}`.
```json
{
  "captured":  [{"path":"risk.stop_loss.value","value":2,"evidence":"stop loss 2%"}],
  "inferred":  [{"path":"context.timeframe","value":"5m","reason":"ORB setup"}],
  "defaults":  [{"path":"experience","value":"intermediate"}],
  "repaired":  [{"path":"context.market","from":"crypto","to":"equity_cash","reason":"derived from INFY.NS"}],
  "unmapped":  [{"concept":"order blocks","note":"no catalog card"}]
}
```
- **Honest metric:** "fully captured" iff `unmapped` is empty AND meaningful `defaults` are confirmed.
- Triple-use: the user-facing **readback**, the **capture metric**, and the **regression-test oracle** (golden tests assert buckets per prompt — every tester bug becomes one assertion).

### 5d. Parameter normalization policy (never silently change a meaningful value)
- **Cosmetic / type normalization → silent:** `"2%"`→`2.0`, `"two percent"`→`2.0`, collapse a window to its natural int (`14.0`→`14`). Numeric meaning preserved.
- **Meaningful value (SL%, TP%, RSI threshold, windows, multipliers) → never silent:** out of bounds → **note + suggested value**, or **clarify** if the correction would materially shift the strategy. The user's value is never quietly overwritten.
- **Rounding only to a param's natural type, never to "nice" numbers** (`13.5→14` ok; `2.8%→3%` is a material change, not allowed).
- **Hard-impossible vs unusual:** outside a card's *hard* bound (RSI 150) → clarify with a suggestion; outside a *soft/typical* bound (SL 50%) → note, never clamp. Card ranges encode both; each param carries a `violation_action: silent | note | clarify` (meaningful params default to never-silent).

The gate is pure and unit-testable in isolation (no LLM, no DB).

---

## 6. Single source of truth for params

Collapse the two stores. Make `risk_execution_config` (or one chosen struct) the only home for SL/TP/RR; `builder.take_profit`/`stop_loss` become read-through properties, not independent fields.

- Delete the `if attribute is None` mirror guards ([builder.py:2688](../app/services/strategy/builder.py), [chat_service.py:3327](../app/services/chat/chat_service.py)).
- `to_draft_json` reads the single store ([builder.py:1718](../app/services/strategy/builder.py)).
- One default resolver replaces the scattered `0.25 / 1.5 / 2.0` literals; defaults are objective/experience-driven *and tagged as `default` in provenance* (so the gate asks to confirm them).

Kills #4 (edits don't apply), and the inconsistent-default confusion.

---

## 7. Merge / modify semantics

- `_merge_sdl` ([sdl_selector.py:581](../app/planner/sdl_selector.py)) becomes a **deep merge**: start from `existing`, overlay only fields the change-turn actually set. (LLM is prompted to emit a *patch* intent, but the merge no longer trusts it to re-emit everything.)
- `reconcile` ([sdl_selector.py:717](../app/planner/sdl_selector.py)) only **upgrades** provenance (default→user) and never downgrades a prior `user` field because it wasn't restated this turn; carry `existing.provenance` forward.
- `merge_preview` ([builder.py:1462](../app/services/strategy/builder.py)) **clears stale validation** when the symbol changed; never restores an error bound to the previous symbol.

Kills #5 (forgets earlier edits) and #6 (stale TATAMOTORS error).

---

## 8. File-by-file change list

Grouped by workstream; each ships independently.

### WS1 — State & edits (smallest, highest user-pain)
- `app/planner/sdl_selector.py`: rewrite `_merge_sdl` (deep merge); change `reconcile` call sites to pass prior provenance.
- `app/planner/provenance_reconciler.py`: upgrade-only semantics; accept `prior_field_sources`.
- `app/services/strategy/builder.py`: collapse param double-store; properties for `stop_loss`/`take_profit`; `merge_preview` stale-validation clear; single default resolver.
- `app/services/chat/chat_service.py`: remove duplicate mirror logic (~:3327); read single store.
- `app/services/chat/inputs_snapshot.py`: read single store consistently.

### WS2 — Verification gate
- **NEW** `app/planner/strategy_gate.py`: `verify()` + checks.
- **NEW** `app/planner/engine_contract.py`: shared SL/TP `type`/`anchor` enums.
- `quant_engine/engine/loader.py`: import anchors from `engine_contract` (single source).
- `app/planner/condition_satisfiability.py`: ensure callable in-process; called by gate.
- `app/services/chat/chat_service.py`: call gate after PROPOSE, before RENDER (create + modify); route clarifications/blocks to existing reply paths.
- `app/planner/compiler.py`: drop local `anchor_map` ([:239](../app/planner/compiler.py)); rely on gated SDL + contract.

### WS3 — De-hardcode signals + fix the catalog
- `app/planner/constraint_compiler.py`: **delete** `_subtract_unrequested_signals`, `_FAMILY_KEYWORDS`, `_SIGNAL_TO_FAMILY`, threshold→signal dicts, family frozensets. What remains: pure rendering of card formulas → `entry_condition`/`exit_condition` strings, reading params from the SDL/cards. (Likely shrinks the file by >50%.)
- `app/kb/schemas.py`: add `comparison`, param ranges, `engine.anchors_supported` (Pydantic models) — schema-first since loader is `extra="forbid"`.
- `app/kb/signals/*.yaml` + `app/kb/loader.py`: populate new fields; loader exposes them.
- **Card-correctness fixes (§4b):** `macd_*_cross`, `rsi_cross_*`, `is_above_sma`/`price_below_sma` mirror+refs, `vwap_zscore_*`, `donchian_breakout_*`, `high_delivery_volume`, `rsi_oversold.pairs_well_with`.
- **Coverage cards (§4c):** add `volume_above_average`, `volume_dry_up`, `false_breakout`, RS-as-trigger, HTF; backfill empty `contradicts` lists.
- **NEW** `scripts/backfill_signal_cards.py`: one-time metadata backfill + a validation test that every enabled card has the new fields and valid refs (catches broken `mirrors_to`/`contradicts`/`pairs_well_with`).
- `app/planner/catalog_schema.py`: surface new card fields in the menu shown to the LLM and to the gate.

### WS4 — Inputs (infer-don't-interrogate), routing, symbol, never-null
- **Remove the upfront 6-field collection gate** (§3a): plan as soon as plannable; experience never asked; symbol the only conditional ask.
- `app/services/strategy/builder.py` + `app/services/ai/user_input_interpreter.py`: resolution ladder (user→inferred→default) with honest provenance; derive asset/market from symbol and overwrite LLM guess; infer timeframe/direction/objective; symbol alias resolution (`HDFC`→`HDFCBANK.NS`) wired into chat path; multi-symbol → explicit choice (call/replace `_resolve_multi_symbol_list`).
- `app/services/agent/router.py` + `app/services/chat/strategy_flow.py`: off-topic intent → graceful answer instead of ask-symbol default.
- `app/planner/sdl_selector.py` / `sdl_flow.py`: remove `SDL_SELECTOR_ENABLED` from happy path so `sdl_json` is never null; failure → clarify, not legacy silent strategy.
- `app/services/ai/user_input_interpreter.py`: stop instructing the LLM to coerce to nearest legal value; emit raw + let the gate repair/clarify (no silent "2 day"→1d).

---

## 9. Sequencing & flags
- Land WS1 first (no behavior risk beyond edits actually applying). Add regression tests from the PDF.
- WS2 behind a `STRATEGY_GATE_ENABLED` flag during rollout; default on once green.
- WS3 depends on the card-schema migration (WS3 step 1) being merged first.
- WS4 last; mostly additive guards.
- Keep Engine B reachable behind a debug flag for one release as a safety net, then delete.

---

## 10. Test plan (each tester bug → a test)
Add `tests/test_planner/test_gate.py` and extend `tests/test_chat/`. Concrete cases mirrored from the review sheet:
- INFY "9 EMA above close" → gate blocks contradiction; offers `price_below_ema(9)`.
- "volume above 20 SMA" → no hidden 1.5; param within range or clarify.
- "long lower wick" → rejection-candle survives (not dropped).
- order-blocks / gamma / RS-vs-Nifty → unmapped clarification, **no** EMA substitution.
- "change TP to 2%" across turns → applies and persists.
- 3–4 sequential edits → all retained.
- TATAMOTORS→INFY → stale error cleared.
- "HDFC" → resolves HDFCBANK.NS.
- 5 assets → explicit choice, none silently dropped.
- "2 day" / "hold 10 years" / "Spot" → reject-and-ask.
- "RSI>90 AND RSI<10", "bullish and bearish", "100% risk/0% drawdown" → blocked with reason.
- `vwap_deviation` SL → mapped or clarified; never reaches engine.
- generic prompt → market/symbol/asset all agree (or asks for symbol).

## 11. Observability — beautiful, icon-led step logs

One structured, greppable log line per *meaningful* step, so a tester can read a whole turn top-to-bottom and see exactly what happened and why. Builds on the existing emoji style (`🧠 sdl_selector | CREATE …`) and unifies it across Propose→Verify→Render.

**Format:** `<icon> <stage> | sid=<8> t=<turn> | <event> | k=v k=v`

**Stage icons:** 🧠 propose · 🛡️ verify · 🏗️ render · 📊 backtest · 💬 reply
**Action icons (gate):** ✅ pass · 🔧 repair · 💡 inferred · ⚙️ default · ❓ clarify · ♻️ self-heal · ⛔ block (rare)
**Coverage icons:** ✅ captured · 💡 inferred · ⚙️ default · 🔧 repaired · ❓ unmapped

Example turn (INFY "9 EMA above close" prompt):
```
🧠 propose | sid=d0881d27 t=1 | sdl_drafted        | legs=1 signals=3 ms=820
🛡️ verify  | sid=d0881d27 t=1 | symbol_resolved    | in="INFY" -> INFY.NS asset=equity_cash
🔧 verify  | sid=d0881d27 t=1 | repair.market      | crypto -> equity_cash reason=from_symbol
🔧 verify  | sid=d0881d27 t=1 | repair.signal_map  | ema_above -> price_below_ema reason=intent_rhs=price
💡 verify  | sid=d0881d27 t=1 | infer.timeframe    | 5m reason=intraday_ema_setup
🛡️ verify  | sid=d0881d27 t=1 | satisfiable        | fires=7/200 ok=true
✅ verify  | sid=d0881d27 t=1 | gate_pass          | repaired=2 inferred=1 defaults=1 unmapped=0
🏗️ render  | sid=d0881d27 t=1 | compiled           | entry="EMA(9)<EMA(21) AND CLOSE<EMA(9) AND ..." 
💬 reply   | sid=d0881d27 t=1 | coverage           | captured=8 inferred=1 defaults=1 repaired=2 unmapped=0
```
Self-heal path:
```
⛔ verify  | sid=… t=1 | contradiction      | ema_above ∧ ema_cross_down unadjudicable
♻️ verify  | sid=… t=1 | self_heal.reproposed | attempt=1
✅ verify  | sid=… t=1 | gate_pass          | …
```

Implementation:
- **NEW** `app/core/step_logger.py`: `log_step(icon, stage, sid, turn, event, **kv)` — single formatter; one place to tune format/levels; `STEP_LOG_LEVEL` env to gate verbosity.
- Each stage calls it at its meaningful boundaries (propose drafted, every gate repair/infer/default/clarify, satisfiability result, gate pass/block, compile, coverage). No raw `print`/ad-hoc `logger.info` in the new path.
- The coverage report (§5c) is logged once per turn and is the same object surfaced to the readback — logs and UI never disagree.

## 11b. Risks
- **Card backfill correctness** — mitigate with a validation pass + golden tests per card.
- **Gate too strict** → over-asking. Tune: only clarify when genuinely ambiguous/impossible; sane defaults still allowed but tagged.
- **Hidden Engine-B dependencies** in chat flow — keep behind flag one release; monitor.
- **LLM patch-merge prompt** must still describe the change clearly even though merge no longer trusts full re-emission.
```
