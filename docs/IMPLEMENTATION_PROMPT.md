# Implementation Prompt — Strategy Pipeline Refactor

> Hand this to an engineer or a coding agent. It is self-contained but cross-references the design doc
> [`docs/STRATEGY_PIPELINE_REFACTOR.md`](./STRATEGY_PIPELINE_REFACTOR.md) for rationale. Read that doc first.

## 0. Mission

Rebuild the strategy-creation pipeline of the Stretus chatbot so that **the user's intent is the only thing that drives signals and parameters**. Replace the hardcoded keyword/regex engine with a single path: **Propose (LLM) → Verify (deterministic gate) → Render (compiler → quant_engine)**. The system must be dynamic, correct, robust, and smooth: it repairs or asks, it never dumps raw errors, never blocks unnecessarily, and never silently substitutes a concept or a meaningful parameter.

Ship in four ordered, independently-mergeable workstreams (WS1→WS4). Each WS has: objective, files, contracts, tasks, acceptance tests. Do not start a WS until the prior one is green, unless noted as parallel-safe.

## 1. Non-negotiable invariants (apply to every WS)

1. **One interpreter.** Only the LLM proposer reads the user's prose. The gate, compiler, and engine act **only** on structured artifacts (SDL, provenance, card metadata, engine contract, the proposer's per-signal intent stub). If you find yourself regex-parsing user text outside the proposer, stop — that's the bug we're removing.
2. **Repair-first, ask-rarely, never-error.** No raw engine errors, JSON blobs, "symbol not found" dead-ends, or "captured X of Y" walls reach the user. See design §5/§5b.
3. **Never silently change intent or a meaningful value.** Bullish↔bearish flips, concept swaps, inventing indicators, and changing SL%/TP%/thresholds/windows are forbidden as silent acts — they are repairs *with a visible note*, a clarification, or `unmapped`. Cosmetic normalization only is silent (design §5d).
4. **No new hardcoded signal tables.** Signal knowledge lives in YAML cards, read generically. Adding/altering a signal must never require editing planner Python.
5. **Every meaningful step is logged** via the icon-led step logger (design §11). Logs and the user readback are derived from the *same* coverage object.
6. **Infer, don't interrogate.** Only `symbol` is conditionally required (deferrable to backtest). Everything else is inferred or defaulted with honest provenance; experience is never asked (design §3a).

## 2. Shared building blocks (build these before WS1 uses them)

### 2a. Step logger — NEW `app/core/step_logger.py`
```python
def log_step(icon: str, stage: str, sid: str, turn: int, event: str, **kv) -> None:
    # format: "<icon> <stage> | sid=<sid8> t=<turn> | <event> | k=v k=v"
    # honor STEP_LOG_LEVEL env; one formatter, no f-string logging elsewhere in the new path.
```
Stage icons 🧠 propose · 🛡️ verify · 🏗️ render · 📊 backtest · 💬 reply. Action icons ✅🔧💡⚙️❓♻️⛔. Use exactly the format and examples in design §11.

### 2b. Engine contract — NEW `app/planner/engine_contract.py`
Single source of truth for what the engine accepts. Move the anchor set out of `quant_engine/engine/loader.py:163` here and **import it from both sides**:
```python
STOP_LOSS_ANCHORS = ("opening_range_high","opening_range_low","prev_n_bar_high","prev_n_bar_low")
STOP_LOSS_TYPES   = (...)   # percent, atr, points, structural...
TAKE_PROFIT_TYPES = (...)
# + a mapping table: planner-anchor -> engine-anchor (e.g. swing_low->prev_n_bar_low, vwap_deviation->NEEDS_CLARIFY)
```
`quant_engine/engine/loader.py` imports `STOP_LOSS_ANCHORS` instead of its local literal. This guarantees the planner can never emit an anchor the engine rejects (kills the `vwap_deviation` backtest error).

## 3. WS1 — State, edits, and single source of truth  *(do first; smallest, highest pain)*

**Objective:** user edits always apply and accumulate across turns; no stale state; one home for risk params.

**Files:** `app/planner/sdl_selector.py`, `app/planner/provenance_reconciler.py`, `app/services/strategy/builder.py`, `app/services/chat/chat_service.py`, `app/services/chat/inputs_snapshot.py`.

**Tasks:**
1. **Deep-merge modify.** Rewrite `_merge_sdl` ([sdl_selector.py:581](../app/planner/sdl_selector.py)) to start from `existing` and overlay only fields the change-turn set (recursive dict/list-by-key merge), instead of `SDL(**updated.model_dump())`. The LLM modify call should emit a *patch intent*; the merge must not depend on it re-emitting the whole SDL.
2. **Upgrade-only provenance.** `reconcile` ([sdl_selector.py:717](../app/planner/sdl_selector.py), [provenance_reconciler.py](../app/planner/provenance_reconciler.py)) accepts `prior_field_sources` and only promotes `default→inferred→user`; it must never demote a prior `user` field because it wasn't restated this turn.
3. **One risk store.** Make `builder.stop_loss`/`take_profit`/`risk_reward` read-through **properties** over a single dict (pick `risk_execution_config`). Delete the `if attribute is None` mirror guards at [builder.py:2688](../app/services/strategy/builder.py) and [chat_service.py:3327](../app/services/chat/chat_service.py). `to_draft_json` ([builder.py:1718](../app/services/strategy/builder.py)) and `inputs_snapshot.py` read the one store.
4. **Single default resolver.** Replace scattered `0.25 / 1.5 / 2.0` literals (builder.py:1074/1109, compat.py:44/50, chat_service.py:4380, risk_manager.py:49) with one function that returns a `default`-tagged value (objective/experience-aware). Defaults are surfaced for optional confirmation, not silently authoritative.
5. **Stale-validation clear.** In `merge_preview` ([builder.py:1462](../app/services/strategy/builder.py)), do not restore a symbol-bound validation error when `preview["symbol"]` differs from the incoming symbol.
6. **Log** each edit: `🔧 verify | … | edit.applied | field=risk.take_profit 1.0->2.0`.

**Acceptance (add `tests/test_chat/test_edits.py`):**
- "change take profit to 2%" → draft + card show 2.0, persists next turn.
- 4 sequential edits (entry sig, SL, TP, exit sig) → all four retained after turn 4.
- TATAMOTORS→INFY → no stale "TATAMOTORS not found".
- modify a param not mentioned this turn → its prior `user` provenance survives.

## 4. WS2 — The verification gate + intent coverage report

**Objective:** a deterministic, structure-only gate that repairs, asks once, or (rarely) blocks — and emits the honest coverage report. Depends on 2a/2b.

**Files:** NEW `app/planner/strategy_gate.py`; `app/planner/condition_satisfiability.py`; `app/planner/compiler.py`; `app/services/chat/chat_service.py`; `app/planner/sdl_selector.py` (intent stub).

**Contracts:**
```python
@dataclass
class CoverageReport:           # design §5c
    captured: list[dict]; inferred: list[dict]; defaults: list[dict]
    repaired: list[dict]; unmapped: list[dict]

@dataclass
class GateResult:
    sdl: SDL | None             # repaired SDL when ok
    coverage: CoverageReport
    clarification: dict | None  # one pre-filled question, or None
    blocked: str | None         # rare, human phrased

def verify(sdl: SDL, *, symbol_ctx: SymbolContext, catalog: CatalogMenu,
           prior_provenance: dict | None = None) -> GateResult: ...
```

**Tasks:**
1. **Proposer intent stub.** Extend the SDL proposer so each signal carries `intent={user_span, intended:{lhs,rhs,op}}`. This is the *only* evidence the gate uses to adjudicate mapping errors (design §5b.2/.5).
2. **Implement checks** in priority order, each emitting a step log and a coverage entry:
   - market/symbol reconciliation (derive market/asset from `symbol_ctx`; overwrite LLM guess) → `repaired`.
   - alias resolution wired via symbol_ctx → `repaired`/`captured`.
   - engine-contract normalization using `engine_contract` map; unmappable anchor → clarify.
   - satisfiability: wire `check_entry_satisfiable` ([condition_satisfiability.py](../app/planner/condition_satisfiability.py)); 0 fires → §5b.6 self-heal.
   - contradiction: enforce card `contradicts`; mapping-error repair only with intent-stub proof; else self-heal/clarify.
   - param policy (design §5d): cosmetic silent; meaningful → note/clarify; never overwrite silently.
   - input sanity: risk%≤cap, drawdown>0, plausible win-rate, holding-period≠timeframe, trade-type valid → reframe/cap with note or one clarify.
   - unmapped honesty: any `unmapped_details` or no-card concept → `unmapped` entry + note; **never** substitute.
   - provenance honesty: recompute coverage from reconciled provenance (do not trust LLM self-report).
3. **Bounded self-heal** (design §5b.6): on unadjudicable contradiction, re-call the proposer once with the contradiction described; if still inconsistent, return one `clarification`. Log `♻️ self_heal.reproposed attempt=1`.
4. **Wire into chat** ([chat_service.py](../app/services/chat/chat_service.py)) after PROPOSE, before RENDER, on **create and modify** turns. Route `clarification`→existing clarifications UI, `blocked`→friendly reply, ok→RENDER. Drop `compiler.py` local `anchor_map` ([:239](../app/planner/compiler.py)); rely on gated SDL + contract.
5. Behind flag `STRATEGY_GATE_ENABLED` (default on once green).

**Acceptance (`tests/test_planner/test_gate.py`):** one assertion per PDF bug, all checking the coverage buckets / repaired list / clarification — e.g. INFY "9 EMA above close" → repaired `ema_above→price_below_ema` *with intent proof* (and self-heal/clarify *without* it); `vwap_deviation` SL → repaired or clarified, never reaches engine; "RSI>90 AND RSI<10" / "bullish and bearish" / "100% risk, 0% drawdown" → reframed-with-note or one clarification, never raw error; generic prompt → market/symbol/asset agree.

## 5. WS3 — De-hardcode signals + fix the catalog  *(parallel-safe with WS1; needs schema step first)*

**Objective:** delete the keyword engine; make cards the single source of signal truth; fix card-level correctness bugs.

**Files:** `app/planner/constraint_compiler.py`, `app/kb/schemas.py`, `app/kb/loader.py`, `app/kb/signals/*.yaml`, `app/planner/catalog_schema.py`, NEW `scripts/backfill_signal_cards.py`.

**Tasks:**
1. **Schema first** ([schemas.py](../app/kb/schemas.py), Pydantic `extra="forbid"`): add `comparison{lhs,rhs,op}`, per-param `{type,default,min,max,violation_action}`, and `engine.anchors_supported`. Then populate cards.
2. **Delete the keyword engine** from [constraint_compiler.py](../app/planner/constraint_compiler.py): remove `_subtract_unrequested_signals`, `_FAMILY_KEYWORDS`, `_SIGNAL_TO_FAMILY`, `_RSI_*_THRESHOLD_SIGNALS`, family frozensets, and the `_*_RE` mapping regexes. Keep only generic formula rendering that reads params from the SDL/cards. **No signal is ever dropped by keyword heuristic.**
3. **Card-correctness fixes** (design §4b): `macd_*_cross` (use MACD-vs-signal cross, not `MACD>0`), `rsi_cross_*` (true crossover or rename), `is_above_sma`/`price_below_sma` mirror+contradicts+mirrors_to, `vwap_zscore_*`, `donchian_breakout_*` regime/event label, dedupe `high_delivery_volume`, fix `rsi_oversold.pairs_well_with`.
4. **Coverage cards** (design §4c): add `volume_above_average`, `volume_dry_up`, `false_breakout`, RS-as-trigger, explicit HTF; backfill empty `contradicts` (`volume_spike`, `adx_strong_trend`, …).
5. **`scripts/backfill_signal_cards.py`** + a CI test that every enabled card has the new fields and that all `mirrors_to`/`contradicts`/`pairs_well_with` refs resolve to real cards.
6. **`catalog_schema.py`**: expose new fields in the menu shown to the LLM and the gate.

**Acceptance:** "long lower wick" keeps the rejection-candle filter (not dropped); "MACD crosses above signal" renders a signal-line cross; order-blocks/gamma/RS → `unmapped` (no EMA substitution); ref-validation test passes for all 124 cards.

## 6. WS4 — Inputs (infer-don't-interrogate), routing, never-null  *(last)*

**Objective:** remove the upfront collection gate; smooth routing; `sdl_json` never null.

**Files:** `app/services/strategy/builder.py`, `app/services/ai/user_input_interpreter.py`, `app/services/agent/router.py`, `app/services/chat/strategy_flow.py`, `app/planner/sdl_selector.py`/`sdl_flow.py`.

**Tasks:**
1. Remove the 6-field collection gate; plan as soon as plannable; experience never asked (design §3a). Symbol the only conditional ask, deferrable to backtest.
2. Implement the **resolution ladder** with honest provenance; derive asset/market from symbol and overwrite LLM guess; infer timeframe/direction/objective; wire symbol alias (`HDFC→HDFCBANK.NS`) into the chat path; multi-symbol → one explicit choice (use `_resolve_multi_symbol_list` which is currently dead code).
3. Off-topic intent ("weather", "SOL benefits") → graceful answer, not the ask-symbol default ([strategy_flow.py:208](../app/services/chat/strategy_flow.py)).
4. Stop instructing the LLM to coerce to nearest legal value ([user_input_interpreter.py:575](../app/services/ai/user_input_interpreter.py)); emit raw, let the gate repair/clarify (no silent "2 day"→1d, no "Spot"→intraday).
5. Remove `SDL_SELECTOR_ENABLED` from the happy path so `sdl_json` is never null; failure → clarify, not a legacy silent strategy.

**Acceptance:** "HDFC"→HDFCBANK.NS; 5 assets→one choice, none dropped; "2 day"/"hold 10 years"/"Spot"→note-or-ask (not silent); "weather"→friendly reply; a full strategy prompt produces a strategy in one shot with at most one question.

## 7. Global definition of done

- All four WS acceptance suites green; the PDF tester sheet re-run shows each prior bug fixed or downgraded to a graceful note/question.
- A grep confirms no signal-name keyword tables remain in `app/planner/constraint_compiler.py`.
- `quant_engine` imports anchors from `engine_contract`; planner can't emit an anchor the engine rejects.
- Every turn produces the icon log trace and a single coverage object shared by logs + readback.
- Behind-flag legacy path kept one release, then deleted in a follow-up.

## 8. Guardrails — do NOT
- Re-introduce prose parsing outside the proposer.
- Make the gate pick between intents without proof, or silently change a meaningful parameter.
- Block, error, or loop the user when a repair or single question would do.
- Add per-signal logic in Python — extend the card schema instead.
