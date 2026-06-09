# Branch Changes — `fix/chat-signals-redesign`

A walkthrough of every active change on this branch. The goal of the branch
is to make the AI planner **faithful to the user's literal prompt**: stop
hallucinating defaults, stop swapping indicator families, and stop adding
filters the user never asked for.

> **Status:** 92 files changed (~6.5k insertions, ~1.9k deletions), all
> uncommitted on top of `main` at `fbe2b02`.

---

## TL;DR — what's new

1. **Catalog-driven signal picker** ([app/planner/catalog_signal_picker.py](app/planner/catalog_signal_picker.py)) — every signal YAML now declares the natural-language phrases that should select it via a new `match_when` block. The picker walks the catalog instead of relying on hardcoded Python mappings.
2. **Fidelity validator** ([app/planner/fidelity_validator.py](app/planner/fidelity_validator.py)) — after a strategy is assembled, this checks that every concrete thing the user said is reflected in the output (indicator, period, direction, RR, SL anchor) and surfaces mismatches as clarification questions.
3. **Original prompt preservation** — the user's literal first message now travels through the planner instead of the agent router's paraphrased `goal`.
4. **Extraction audit trail** ([app/core/extraction_audit.py](app/core/extraction_audit.py)) — every field extraction (who said it, with what confidence, from which span of text) is logged per session for UI provenance.
5. **Unsupported-indicator detector** ([app/planner/unsupported_indicators.py](app/planner/unsupported_indicators.py)) — flags Ichimoku, Heikin-Ashi, Fibonacci, ICT/SMC vocabulary, options Greeks etc. so the chat layer can suggest alternatives instead of silently dropping them.
6. **Eight new semantic fields** flow from extractor → planner so the planner picks signals from precise user phrasing (RSI thresholds, MACD states, VWAP relations, MA relations, regime preference, SL type hint, partial exits, exit-on-opposite).
7. **RMS source labelling** — every risk field is now tagged `user`, `semantic`, `system_default`, etc. so the chat layer can gate the planner until the user has actually supplied SL/TP/RR (enforces the `RMS must be user-given` rule).
8. **All 82 signal YAMLs updated** with `match_when` blocks via a one-shot migration script.

---

## Schema additions

### [app/kb/schemas.py](app/kb/schemas.py)

Added the catalog-recognition contract:

- `SignalMatchRole` — `"entry_trigger"` / `"entry_filter"` / `"exit_trigger"`.
- `MatchParamSpec` — how to pull a single param from a regex match (named or numbered group, default, type coercion).
- `MatchWhen` — the new field on every `SignalCard`. Carries:
  - `phrases` — list of Python regex patterns; any match selects the signal.
  - `acts_as` — the role this signal plays when matched.
  - `confidence` — base score for conflict resolution between competing signals.
  - `mirrors_to` — the opposite-direction signal (used when the user says "exit on opposite crossover").
  - `extract_params` — how to lift parameters (e.g. SMA window) from the captured regex groups.
  - `requires_keyword` / `forbids_keyword` — disambiguation guards.

### [app/kb/execution_schemas.py](app/kb/execution_schemas.py)

`SemanticInstructions` gained eight new fields surfaced by the extractor so the planner can act without re-parsing the prompt:

```
rsi_thresholds       # [{"op": "above", "value": 60}, ...]
macd_states          # ["histogram_positive", "bullish_cross"]
vwap_relations       # ["price_above_vwap"]
regime_preference    # "trending" | "ranging" | "volatile" | None
sl_type_hint         # "atr" | "structural" | "percent" | None
partial_exits        # [{"trigger": "rr_multiple", "value": 1.5, "size_pct": 50}]
ma_relations         # [{"family": "sma", "side": "above", "window": 20, "kind": "cross"}, ...]
exit_on_opposite     # True when user wrote "exit on opposite crossover"
```

### [app/kb/signals/_registry.yaml](app/kb/signals/_registry.yaml)

Three new enabled signals: `price_below_sma`, `price_cross_above_sma`, `price_cross_below_sma`. The YAML files for these are new and follow the standard signal-card layout plus the `match_when` block.

---

## Planner changes

### NEW — [app/planner/catalog_signal_picker.py](app/planner/catalog_signal_picker.py)

The Phase 14 replacement for the old "preset template + constraint patches" pipeline.

- `pick_plan_from_catalog()` — public entry. Walks every signal in the KB, runs its `match_when.phrases` against the user prompt, collects matches with role + confidence, resolves conflicts, builds the final signal plan.
- `_find_all_matches()` — phrase scanner with a **negation guard**: if a word like "avoid", "ignore", "skip", "don't" appears within 80 characters before a match (and no clause separator interrupts), the match is dropped. Stops "avoid gap-down" from being read as a gap-down entry trigger.
- `_phrase_confidence()` — biases scoring toward longer / more specific phrase matches so "macd line crosses above signal" beats a generic "macd".
- `_resolve_role_conflicts()` — exactly one `entry_trigger` per indicator family wins (highest confidence); losers are demoted to `entry_filter`. Filters across different families are kept. Exits are OR'd.
- `_apply_exit_on_opposite()` — when the entry-trigger card has a `mirrors_to` and the user asked for opposite-side exit, the mirror signal is added as `exit_trigger`.
- `auto_fill_missing_families()` — backstop: if the user named an indicator family (SMA, EMA, ADX, …) but no signal of that family ended up in the plan, look up the family's default regime signal and append it with the right period.
- `merge_picker_with_preset()` — three-mode merge with the legacy preset plan:
  - **Mode A** (picker has a high-confidence trigger): picker plan wins, preset SL/TP scaffolding preserved.
  - **Mode B** (picker only matched filters/exits): keep preset's trigger, swap in picker's filters and exits.
  - **Mode C** (picker confidence below `DEFAULT_CONFIDENCE_THRESHOLD = 0.75`): preset wins.

### NEW — [app/planner/fidelity_validator.py](app/planner/fidelity_validator.py)

Cheap (regex + dict lookups, no extra LLM call) post-assembly check that emits `FidelityFinding` objects with severity `critical` / `warning` / `info`. Checks:

1. **Family swap** — user asked for SMA but plan has only EMA (or vice-versa).
2. **Indicator missing** — user said "MACD" but no MACD signal made it in.
3. **Period mismatch** — user said "20 SMA" but plan uses SMA(50).
4. **Fabricated filters** — plan added filters on indicator families the user never mentioned.
5. **Direction mismatch** — user asked long-only but plan has bearish entries.
6. **RR not from user** — user wrote a R:R ratio in the prompt but the strategy is using a different number with `source != user|semantic`.
7. **TP-RR inconsistent** — user gave BOTH a TP% and an RR ratio that disagree under their SL.
8. **SL/TP system-default** — RMS contains `source == system_default` for stop/take values that should be user-given.

`format_user_message()` rolls all findings into a single human-readable message the chat layer asks the user to clarify.

### NEW — [app/planner/unsupported_indicators.py](app/planner/unsupported_indicators.py)

Thin keyword detector for indicators / concepts the KB doesn't cover: Ichimoku, Heikin-Ashi, Renko, pivot points, Fibonacci, Wyckoff, ICT/SMC vocabulary (order block, liquidity sweep, breaker block, inducement), options Greeks, India VIX gate, pair trading, expiry-day logic, calendar schedules. Each entry carries a `suggested_alternative` so the chat layer can propose the closest mapped signal.

### Updated — [app/planner/constraint_compiler.py](app/planner/constraint_compiler.py)

The biggest single diff (~958 lines added). Two new processing phases:

- **Phase 12 — consume semantic fields directly.** Eight `_apply_*` functions (`_apply_ma_relations`, `_apply_vwap_relations`, `_apply_rsi_thresholds`, `_apply_macd_states`, `_apply_exit_on_opposite`, `_apply_regime_preference`, `_apply_sl_type_hint`, `_apply_partial_exits`) convert the new `SemanticInstructions` fields into KB signal injections. Lookup tables (`_VWAP_RELATION_TO_SIGNAL`, `_MACD_STATE_TO_SIGNAL`, `_RSI_BULLISH_THRESHOLD_SIGNALS`, `_RSI_BEARISH_THRESHOLD_SIGNALS`, …) replace previously-hardcoded mapping branches.
- **Phase 13 — subtract unrequested signals.** `_subtract_unrequested_signals()` + `_signal_family_of()` strip filters and exits whose indicator family does not appear anywhere in the user's prompt. This is what stops preset templates from auto-adding e.g. an RSI filter when the user only talked about MACD.
- EMA-window resolution now refuses to guess a second window when the user gave one; it logs instead.

### Updated — [app/planner/semantic_extractor.py](app/planner/semantic_extractor.py)

~476 lines added.

- New regex banks: `RSI_THRESHOLD_PATTERNS`, `MACD_STATE_PATTERNS`, `VWAP_RELATION_PATTERNS`, `MA_RELATION_PATTERNS`, `OPPOSITE_EXIT_PATTERNS`, plus "R notation" capture (`1.5R`, `2R target`).
- Eight new extraction methods feed the new `SemanticInstructions` fields.
- `extract()` now accepts an optional `session_id` and writes every captured field to the new extraction audit so the UI can render provenance.

### Updated — [app/planner/advanced_planner.py](app/planner/advanced_planner.py)

Step 2 now reads from `builder.original_user_prompt` (the raw user text) and falls back to the paraphrased `builder.goal` only when the original was never captured. Source is logged.

### Updated — [app/planner/param_resolver.py](app/planner/param_resolver.py)

`resolve_sl_tp` now tags each TP as `user_rr` or `ai_default` and emits a `WARNING` log every time a TP is AI-defaulted (with `action_required=chat_layer_should_gate_planner_until_rms_complete`). This is the trail the chat layer reads to refuse to finalize a plan with a phantom take-profit.

### Updated — [app/planner/semantic_normalizer.py](app/planner/semantic_normalizer.py)

Simplified the risk-reward normalization. The previous version dropped any builder RR that matched a "common LLM-hallucination default" (2.5 or 3.0). That false-positived on legitimate user requests for 1:2.5. The new code trusts the `source` label outright and only logs when builder and extractor disagree.

---

## Chat / builder layer

### Updated — [app/services/strategy/builder.py](app/services/strategy/builder.py)

- New immutable field **`original_user_prompt`** on `StrategyBuilder` — captured on the first substantive user turn, never overwritten by subsequent agent paraphrases, and round-tripped through `to_draft_json` / `merge_preview` so it survives across requests.
- Rewritten R:R parsing with `_classify()` helper. For ambiguous "3:7" style ratios it looks within ±25 characters for a `reward` or `risk` keyword to decide which side is which. Handles bare `1.5R` and `RR of 2`.
- New / extended RMS captures:
  - `max_consecutive_losses` ("stop after 3 consecutive losses", "max 2 losing trades in a row").
  - `cooldown_bars_after_loss`, `cooldown_bars_after_profit`.
  - More flexible TP, daily-loss-cap, and per-trade-risk patterns (optional prepositions / adverbs).
- `max trades` now distinguishes session-level open-position cap (`max_trades`) from the consecutive-loss circuit breaker. Both `max_positions` (legacy alias) and `max_trades` are written.

### Updated — [app/services/chat/chat_service.py](app/services/chat/chat_service.py)

- New helper `_resolve_strategy_source_prompt()` — always prefer the builder's preserved `original_user_prompt` over the agent's `goal` summary when handing text to the planner / fidelity validator.
- Integrated the **catalog picker** flow: build picker plan → merge with preset → run `auto_fill_missing_families` → record audit lines on the draft.
- Integrated the **fidelity validator** at strategy assembly: every assembled strategy gets `fidelity_findings` and `fidelity_summary` attached for the UI; critical findings are queued as clarification questions before the strategy is confirmed.
- Bumped `source_prompt` length from 200 to 1000 characters when forwarding to extraction so longer prompts don't lose context.

### Updated — [app/services/execution/risk_execution_config_service.py](app/services/execution/risk_execution_config_service.py)

`sync_builder_risk_from_state()` now preserves or assigns a **source label** to every RMS field. Anything not explicitly tagged becomes `system_default`. This makes the `RMS must be user-given` rule enforceable: the fidelity validator can now see "your SL of 2% is a system default" and gate accordingly.

---

## Catalog migration — 82 signal YAMLs

Every file under [app/kb/signals/](app/kb/signals/) (and three new ones — `price_below_sma`, `price_cross_above_sma`, `price_cross_below_sma`) gained a `match_when` block. Example from [macd_bullish_cross.yaml](app/kb/signals/macd_bullish_cross.yaml):

```yaml
match_when:
  phrases:
  - macd\s+(?:line\s+)?(?:crosses?|crossed|cross)\s+above\s+signal
  - macd\s+(?:bullish|positive)\s+cross(?:over)?
  - macd\s+up\s+cross
  - macd\s+golden\s+cross
  acts_as: entry_trigger
  mirrors_to: macd_bearish_cross
```

The wider diff is mostly YAML re-formatting (inline → block style) from the migration script.

---

## Scripts (new, one-shot helpers)

- [scripts/migrate_signals_match_when.py](scripts/migrate_signals_match_when.py) — idempotent script that walked every signal YAML and added a `match_when` block. Phrase lists are derived from the signal name and category, mirror exits are linked.
- [scripts/add_missing_formulas.py](scripts/add_missing_formulas.py) — fills missing `formula:` display fields. Required so the constraint compiler can rebuild conditions for signals previously dropped (e.g. Supertrend).
- [scripts/test_catalog_picker.py](scripts/test_catalog_picker.py), [scripts/test_complex_breakout_prompt.py](scripts/test_complex_breakout_prompt.py), [scripts/test_four_user_prompts.py](scripts/test_four_user_prompts.py), [scripts/test_live_prompts.py](scripts/test_live_prompts.py), [scripts/test_six_prompts.py](scripts/test_six_prompts.py), [scripts/test_signal_consumption.py](scripts/test_signal_consumption.py) — ad-hoc smoke tests for the new pipeline (not pytest tests; standalone executable prompts).

---

## Core (cross-cutting)

### NEW — [app/core/extraction_audit.py](app/core/extraction_audit.py)

A framework-free, thread-safe, in-memory per-session audit store. Anyone who pulls a value out of the user's prompt calls:

```python
record_extraction(
    session_id, field="stop_loss_pct", new_value=2.0,
    source="user", confidence=0.95, evidence="my SL is 2%",
    extractor="builder.extract_rms",
)
```

Stores up to 500 events per session (deque-bounded). Conflicts (a new value differs from the previous latest value for the same field) are logged at `WARNING` so the chat layer can show a "I had 2%, you just said 3% — which one?" clarification. Exposed via `get_trail(session_id)` / `latest_for_field(...)` / `clear_session(...)`.

### Updated — [app/core/signal_performance_cache.json](app/core/signal_performance_cache.json)

Cache entries for the three new SMA signals were added (34 lines).

---

## Cache artifacts (untracked)

`quant_engine/cache/ohlcv_fetch/*.pkl.gz` — three new OHLCV pickle caches generated while running the test scripts. These are local-only artifacts, not intended to be committed.

---

## How the new pipeline runs end-to-end

1. **User sends a message.** Chat service captures it as `builder.original_user_prompt` (immutable, set once).
2. **Semantic extractor** parses the prompt — populating both the legacy fields and the new eight (`rsi_thresholds`, `macd_states`, …). Every extracted value is recorded in the extraction audit.
3. **Catalog picker** walks every KB signal, runs `match_when.phrases`, scores hits, resolves conflicts. If the picker confidence ≥ 0.75 it owns the plan; otherwise it merges as filters on top of the legacy preset plan, with `auto_fill_missing_families` as a backstop.
4. **Constraint compiler** runs Phases 12 + 13: injects signals from the new semantic fields and strips families the user never mentioned.
5. **Param resolver** computes SL/TP, tagging each value as `user_rr` or `ai_default` and warning the chat layer when the latter happens.
6. **Fidelity validator** compares the assembled strategy back against the original prompt. Critical findings (family swap, period mismatch, fabricated filter, RR not from user, SL/TP system-default) become clarification questions; the strategy is not confirmed until they are addressed.
7. **Unsupported-indicator detector** runs in parallel — any Ichimoku / ICT / pivot mention surfaces a "we don't support this yet, want the alternative?" prompt.

The net effect: the planner can no longer paraphrase the user, swap their indicator, or silently fill in a phantom take-profit.
