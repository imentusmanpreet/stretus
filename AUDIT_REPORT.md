# Stretus-AI Strategy Generation — Deep Audit Report

**Scope:** End-to-end audit of conversational strategy generation from chat input to YAML emission.
**Repo:** `/home/im3/Desktop/stretus/stretus-ai-v2/stretus-ai`
**Date:** 2026-05-27
**Author:** Senior engineering audit (read-only). No fixes proposed in this pass.

All citations are `file:line` against the repo HEAD on branch `main`.

---

## 1. Executive Summary (10 bullets, worst first)

1. **The market is hardcoded to `indian_stocks` at the instant a `StrategyBuilder` is constructed** ([builder.py:37](app/services/strategy/builder.py#L37), [builder.py:405](app/services/strategy/builder.py#L405)) and then re-asserted unconditionally on every turn by `extract_strategy_details()` ([builder.py:2272](app/services/strategy/builder.py#L2272)). No code path anywhere reassigns it to `crypto_spot` from a symbol like ETH/USDC. `_normalise_market(raw)` is a static method whose entire body is `return DEFAULT_MARKET` ([builder.py:1129-1130](app/services/strategy/builder.py#L1129-L1130)) — it discards its input. Bug A is a feature of the data model, not a bad branch.
2. **There is no symbol→market inference.** `resolve_supported_stock()` returns `{symbol, display_name, exchange, sector}` but deliberately **omits `asset_class`** ([compat.py:103-108](app/kb/compat.py#L103-L108)) even though the underlying `Stock` schema has it. The KB knows; the chat layer can't read it.
3. **`markets.yaml` contains only NSE and BSE** ([markets.yaml:1-19](app/kb/markets.yaml)). There is no crypto exchange entry, so even a "lookup" would fail. `compat.py:36-54` synthesises only `indian_stocks` and `indian_indices` market configs in `get_all_market_configs()`.
4. **"daily" is parsed into the timeframe regex but NOT into the objective regex.** `_TF_RE` at [builder.py:395-400](app/services/strategy/builder.py#L395-L400) includes `daily`, and `timeframes.yaml:37` maps `daily → 1d`. But the `intraday` regex at [builder.py:2308](app/services/strategy/builder.py#L2308) does **not** list `daily`. Result: "5% profit daily" sets `timeframe=1d` and `objective=None`, which then defaults to `intraday`-flavored execution downstream — the headline Bug B contradiction.
5. **No validator rejects intraday + 1d.** `validator.py` only checks direction/sentiment alignment and `tp >= sl` ([validator.py:18-59](app/planner/validator.py#L18-L59)). It runs on the post-pick `StrategyPlan`, not the `StrategyBuilder`. There is no cross-field check against `timeframes.yaml` buckets, no rule that `intraday` objective requires a sub-1h timeframe.
6. **Defaults are scattered across at least 7 files and conflict.** `stop_loss_pct` has at least four different defaults: `1.5%` ([compat.py:44](app/kb/compat.py#L44)), `0.25%` ([risk_tiers.yaml:23](app/kb/risk_tiers.yaml#L23) and [schemas.py:295](app/kb/schemas.py)), `2.0%` ([builder.py:1000](app/services/strategy/builder.py#L1000) and `app/db/models/execution.py:64`). `apply_defaults()` ([builder.py:961-965](app/services/strategy/builder.py#L961-L965)) chains them in order *risk_cfg → tier.base_sl_pct → market_cfg → 0.25*, so the value a user receives depends on which intermediate field happened to be set first — a single source of truth for SL/TP **does not exist**.
7. **The system mislabels user-provided values as defaults.** When the user says "5% profit daily", `take_profit_pct=5.0` is extracted correctly at [builder.py:2374-2383](app/services/strategy/builder.py#L2374-L2383). But because `stop_loss` was never given, [chat_service.py:3829-3893](app/services/chat/chat_service.py#L3829-L3893) sends the "use defaults" clarification prompt. If the user accepts, both `stop_loss_pct` AND `take_profit_pct` get re-tagged `"user_confirmed_default"` at [chat_service.py:3900-3903](app/services/chat/chat_service.py#L3900-L3903) — even though `tp_missing` should be False by line 3817. The hardcoded fallback `1.5` at [chat_service.py:3834](app/services/chat/chat_service.py#L3834) comes from `MARKET_CONFIG["indian_stocks"]["default_stop_loss"]` ([compat.py:44](app/kb/compat.py#L44)) — a synthesised legacy literal with no per-asset awareness.
8. **`semantic_extractor.py` is a 1,397-line god-object** containing ~20 distinct extractor methods, each backed by its own regex pattern table (HTF, SL, trailing, RR, ref-symbols, sessions, volume, momentum, candles, indicators, direction, gap, consecutive losses, cooldown, spread, confirmation, RSI thresholds, MACD states, VWAP relations, MA relations, opposite-exit). It violates SRP and is a maintenance time-bomb. Even so, **none of what it extracts feeds back into signal selection** — `catalog_signal_picker.pick_plan_from_catalog()` ([catalog_signal_picker.py:726-804](app/planner/catalog_signal_picker.py#L726-L804)) uses an entirely independent regex/phrase-matcher against signal-card `match_when.phrases`. Semantic intent is consulted only for *gating* signals (HTF rules, structural SL, ADX thresholds), not picking them.
9. **Signal selection is deterministic-on-text-features.** The `signal_performance_cache` ([core/signal_performance_cache.py](app/core/signal_performance_cache.py)) is called only from `param_resolver.resolve_params()` ([param_resolver.py:49-56](app/planner/param_resolver.py#L49-L56)), *after* signals are already chosen — it tweaks periods, not picks. There is no feedback loop from backtest performance into next-turn signal choice. Same prompt → same signals, forever.
10. **The response payload is built by 5 disjoint layers** (`response_composer`, `persona_responder`, `response_guard`, `strategy_flow.build_final_strategy_payload`, `chat_service.ChatMessage` assembly) with no shared schema. The `ChatMessage.strategy_json["context"]` wrapper carries `session_id`/`user_id`/`org_id`/`strategy_id`/`next_state`/`current_mode`/`yaml_path` — most of which are then **stripped by `app/api/.../chat.py` before return to the client** ([chat.py:166-174 filter logic]). The payload bloats, persists redundant fields to the DB, and then gets re-flattened at the API boundary. Three layers overlay `risk_and_execution` onto itself in `chat.py:317-320`.

---

## 2. End-to-End Trace — "I want to trade ETH/USDC and want 5% profit daily"

Walking the actual call graph. Every default that fires is called out.

| # | Layer | File:Line | Action | Default fired? |
|---|---|---|---|---|
| 1 | HTTP | `app/api/.../chat.py:405` | `send_message()` validates session, persists user message, enqueues `run_ai_processing(session_id, user_msg.id, body.content)` as background task. | — |
| 2 | Service entry | [chat_service.py:1871](app/services/chat/chat_service.py#L1871) | `run_ai_processing()` runs. Loads chat history; restores last draft. | — |
| 3 | Builder init | [builder.py:403-405](app/services/strategy/builder.py#L403-L405) | `StrategyBuilder()` constructed. **`self.market = DEFAULT_MARKET = "indian_stocks"`** before any input is inspected. | ✅ `market="indian_stocks"` |
| 4 | Restore draft | [chat_service.py:1926](app/services/chat/chat_service.py#L1926) | `builder.merge_preview(last_draft)` runs `_normalise_market(raw)` on the saved market, which **returns `DEFAULT_MARKET` regardless of input** ([builder.py:1129-1130](app/services/strategy/builder.py#L1129-L1130)). | ✅ Crypto state, if any, is erased here. |
| 5 | Risk hydrate | [chat_service.py:1943-1948](app/services/chat/chat_service.py#L1943-L1948) | `_hydrate_builder_risk_execution_config()` loads risk row from DB. | — |
| 6 | Preset detect | [chat_service.py:1962-1974](app/services/chat/chat_service.py#L1962-L1974) | `kb.detect_preset_in_text("I want to trade ETH/USDC and want 5% profit daily")` — keyword scan over preset YAML names. No preset matches "ETH/USDC" or "5% profit". | — |
| 7 | Agent router | [chat_service.py:2040-2054](app/services/chat/chat_service.py#L2040-L2054) | `AgentRouter().route_user_intent(...)` returns a dict including `intent` (likely `new_strategy`) and a `reply_text`. | — |
| 8 | Field extract | [chat_service.py:2535](app/services/chat/chat_service.py#L2535) → [builder.py:2271-2386](app/services/strategy/builder.py#L2271-L2386) | `extract_strategy_details(user_content, builder)` runs the regex barrage on the raw text. | |
| 8a |  | [builder.py:2272](app/services/strategy/builder.py#L2272) | First line: **`builder.market = DEFAULT_MARKET`** — re-asserts indian_stocks on every turn, even if a previous turn somehow set it. | ✅ Re-assertion |
| 8b |  | [builder.py:2275-2288](app/services/strategy/builder.py#L2275-L2288) | `_TF_RE.search()` matches **`daily`** (regex literal at [builder.py:397](app/services/strategy/builder.py#L397)). `normalise_timeframe("daily")` → `1d` via [timeframes.yaml:37](app/kb/timeframes.yaml#L37). `builder.timeframe = "1d"`. | — (parsed correctly) |
| 8c |  | [builder.py:2307-2320](app/services/strategy/builder.py#L2307-L2320) | Objective regex: keyword list is `(intraday\|day trade\|day trading\|scalp\|scalping\|quick profit\|quick profits\|same day\|today only\|short term\|btst)`. **"daily" is NOT in the list.** `builder.objective` stays `None`. | — (silent miss) |
| 8d |  | [builder.py:2366-2386](app/services/strategy/builder.py#L2366-L2386) | `_extract_rms_from_text()` regex correctly extracts `take_profit_pct = 5.0` with `source="user"` (matches `…(\d+)\s*%\s+…profit\b`). Mirrors to `builder.take_profit = 5.0`. **`stop_loss_pct` is NOT inferred** — no regex maps "profit" → SL. | `take_profit=5` (user); `stop_loss=None` |
| 9 | Symbol resolve | [chat_service.py:2715-2759](app/services/chat/chat_service.py#L2715-L2759) | `_extract_explicit_stock_query("ETH/USDC")` → calls `resolve_supported_stock()` ([compat.py:82-108](app/kb/compat.py#L82-L108)). If ETH/USDC is in the KB it returns a dict — **but the dict does not contain `asset_class`**, only `symbol/display_name/exchange/sector`. The chat layer assigns `builder.symbol` ([chat_service.py:2752]) but never updates `builder.market`. | ✅ Symbol set, market NOT updated. |
| 10 | RMS gate | [chat_service.py:3811-3822](app/services/chat/chat_service.py#L3811-L3822) | Computes `sl_missing = True` (stop_loss is None and not in rms_sources). Computes `tp_missing = False` (take_profit was extracted). | — |
| 11 | Default prompt | [chat_service.py:3829-3893](app/services/chat/chat_service.py#L3829-L3893) | Because `sl_missing == True`, sends user a "Before I plan the signals, I need your call on risk" message proposing `default_sl = 1.5` from `MARKET_CONFIG["indian_stocks"]["default_stop_loss"]` ([compat.py:44](app/kb/compat.py#L44)), `default_tp = 1.5 × 2.0 = 3.0` ([chat_service.py:3836-3837](app/services/chat/chat_service.py#L3836-L3837)). Note the proposed TP (3%) **conflicts with the user's already-extracted 5% target**. | ✅ Proposes `1.5/3.0`, ignoring user-stated 5%. |
| 12 | User accepts | [chat_service.py:3897-3905](app/services/chat/chat_service.py#L3897-L3905) | If user says "use defaults", both `stop_loss_pct` and `take_profit_pct` source tags become `"user_confirmed_default"`. The 5% already in `risk_execution_config` survives **only because `apply_defaults` honors `risk_cfg.get("take_profit_pct")` first** ([builder.py:967-972](app/services/strategy/builder.py#L967-L972)). But the audit/source field is now mislabeled. | ✅ Mislabel: 5% TP user value tagged "user_confirmed_default". |
| 13 | Defaults | [builder.py:955-994](app/services/strategy/builder.py#L955-L994) | `apply_defaults()` invoked. `stop_loss = max(risk_cfg or tier.base_sl_pct=0.25 or 1.5)`. The final SL becomes **0.25%** (tier base for default/intermediate, [risk_tiers.yaml:71](app/kb/risk_tiers.yaml#L71)), not the 1.5% the user was just shown. Timeframe stays `1d`. Daily-loss-cap defaults to 2.0% via `get_experience_risk` ([builder.py:979-982](app/services/strategy/builder.py#L979-L982)). `max_trade = "3 trading days"` because `objective` is None → falls into the `else` at [builder.py:992-993](app/services/strategy/builder.py#L992-L993). | ✅ SL=0.25 (tier), TP=5 (user), DLC=2.0, max_trade="3 trading days" |
| 14 | Plan signals | [chat_service.py:3994-4006](app/services/chat/chat_service.py#L3994-L4006) | `plan_signals_v2(builder, ...)` runs. Internally calls `pick_plan_from_catalog(prompt="I want to trade ETH/USDC and want 5% profit daily", kb=..., timeframe="1d", ...)`. The phrase matcher returns **no matches** ([catalog_signal_picker.py:740](app/planner/catalog_signal_picker.py#L740)). Falls back to preset detection, which finds nothing crypto-shaped. Raises `NoValidCandidate` OR returns an empty stack. | ✅ No signals derived from intent. |
| 15 | Semantic gate | [chat_service.py:4047-4070](app/services/chat/chat_service.py#L4047-L4070) | `SemanticExtractor().extract("…")` returns nearly empty (no HTF, no SL anchor, no indicators, no momentum). `ExecutionOrchestrator.apply_semantic_gates()` has nothing to overlay. | — |
| 16 | YAML emit | [builder.py:1517](app/services/strategy/builder.py#L1517) → [yaml_generator.py:26](app/services/strategy/yaml_generator.py#L26) | `to_yaml_dict()` calls `apply_defaults()` AGAIN (idempotent), then `generate_yaml()` writes to disk with `market: indian_stocks`, `exchange: NSE`, `symbol: ETH/USDC.NS`(?), `timeframe: 1d`, `stop_loss: 0.25`, `take_profit: 5.0`, `max_trade: "3 trading days"`. | ✅ Final YAML: crypto symbol with NSE suffix, daily candle, 20:1 RR. |
| 17 | Compose response | [chat_service.py:3449](app/services/chat/chat_service.py#L3449) | `strategy_market = builder.market or "indian_stocks"` — last belt-and-braces hardcode. Builds `ChatMessage` with `content/strategy_draft/strategy_json/strategy_id/...`. | ✅ Indian-equity label in chat reply. |

**Net result of the trace:** A crypto request becomes a daily-candle Indian-equity strategy with a 0.25% SL, a 5% TP (20:1 risk:reward, infeasible), no entry signals, with the user's 5% TP marked as `user_confirmed_default`. Every layer either silently dropped data or silently injected defaults that contradict earlier layers.

---

## 3. Root Cause per Headline Bug

### Bug A — Crypto request labeled as Indian market

**Symptom:** "ETH/USDC" produces `market_type: indian_equity` (or `NSE`) in the response and YAML.

**Code path:**
1. [builder.py:37](app/services/strategy/builder.py#L37) — `DEFAULT_MARKET = "indian_stocks"`.
2. [builder.py:405](app/services/strategy/builder.py#L405) — Constructor: `self.market: Optional[str] = DEFAULT_MARKET`.
3. [builder.py:1129-1130](app/services/strategy/builder.py#L1129-L1130) — `_normalise_market(raw)` static method whose entire body is `return DEFAULT_MARKET`. **It discards its argument.**
4. [builder.py:2272](app/services/strategy/builder.py#L2272) — `extract_strategy_details()` reasserts `builder.market = DEFAULT_MARKET` on **every** chat turn before any inference.
5. [compat.py:82-108](app/kb/compat.py#L82-L108) — `resolve_supported_stock()` returns `{symbol, display_name, exchange, sector}`. Note: **no `asset_class`**.
6. [chat_service.py:3449](app/services/chat/chat_service.py#L3449) — Belt-and-braces: `strategy_market = builder.market or "indian_stocks"`.
7. [markets.yaml:1-19](app/kb/markets.yaml) — File only defines NSE and BSE. No crypto exchanges.
8. [compat.py:36-54](app/kb/compat.py#L36-L54) — `get_all_market_configs()` synthesises only `indian_stocks` and `indian_indices`.

**Root cause:** The data model treats `indian_stocks` as a literal constant, not as one of multiple possible markets. There is **no symbol→market inference function anywhere in the codebase**. The KB *has* asset_class on the `Stock` model but the compat helper deliberately drops it before passing it to chat. Even if it were preserved, `_normalise_market()` is a noop and would discard it again. The mapping is "guesswork that always guesses the same thing."

**Blast radius:** Affects every non-Indian-equity request: crypto, US equities, forex, commodities. Affects exchange suffixing, market hours, holiday calendar, instrument-key resolution, and the persona-responder's market labels.

**Fix direction (one paragraph):** Make `market` derived state, not initial state. Remove the assignment at [builder.py:405] and [builder.py:2272]; introduce a `derive_market(symbol, kb) → MarketSpec` function that consults the `Stock.asset_class` already in the KB and an extended `markets.yaml` covering at least `crypto_spot`. `resolve_supported_stock` should return the full Stock or at minimum include `asset_class`. Delete `_normalise_market` entirely. Audit all `"indian_stocks"` literals and either remove them or replace with `MarketRegistry.lookup(builder.symbol)`.

---

### Bug B — Intraday + 1d timeframe contradiction

**Symptom:** "5% profit daily" produces `trade_type=intraday` AND `timeframe=1d`.

**Code path:**
1. [builder.py:395-400](app/services/strategy/builder.py#L395-L400) — `_TF_RE` regex includes the literal word `daily`.
2. [builder.py:2275-2288](app/services/strategy/builder.py#L2275-L2288) — Timeframe extraction matches "daily", calls `normalise_timeframe("daily")` → "1d" via [timeframes.yaml:37](app/kb/timeframes.yaml#L37) (`daily: 1d`).
3. [builder.py:2307-2320](app/services/strategy/builder.py#L2307-L2320) — Objective regex: `(intraday|day trade|day trading|scalp|scalping|quick profit|quick profits|same day|today only|short term|btst)`. **No "daily".** `builder.objective` stays `None`.
4. [builder.py:986-993](app/services/strategy/builder.py#L986-L993) — `apply_defaults()` only branches `intraday | positional | else`, so `objective=None` falls into the `else` branch, labeled `"3 trading days"` (not literally `intraday`, but interpreted as such by downstream display logic).
5. The "intraday" label that the user sees may come from one of these downstream sources:
   - `_max_trades_per_day()` ([builder.py:1010-1012](app/services/strategy/builder.py#L1010-L1012)) — returns 2 if `objective == "intraday"`, else 1. So a None objective ≠ intraday here.
   - Persona/composer prose that infers intraday from "daily" target — needs separate trace.
6. **Validation:** [validator.py:18-59](app/planner/validator.py#L18-L59) does **not** cross-check objective vs timeframe. [fidelity_validator.py:33-95](app/planner/fidelity_validator.py) checks indicator-family swaps and period mismatches only. [timeframes.yaml:5-9](app/kb/timeframes.yaml#L5-L9) defines `intraday: [10m,15m,30m]` and `positional: [1d]` buckets — **but no code consults these buckets for validation**.

**Root cause:** Two independent regex extractors race over the same word. The timeframe regex correctly maps "daily" → 1d; the objective regex doesn't recognize "daily" at all, leaving the field unset. A separate downstream component (likely persona_responder or response_composer) sees the user said "daily" and labels the *frequency* as intraday. The KB defines bucket constraints that no validator checks.

**Blast radius:** Any prompt using "daily", "every day", "per day" as a frequency on an intraday timeframe (or vice versa) corrupts the trade_type. Affects risk-tier selection, signal picking (since `intent.frequency` flows into `soft_ranker.pick`), and the user-visible explanation.

**Fix direction:** Treat **frequency** and **timeframe** and **objective** as three separate fields with explicit extraction. "Daily" as in "profit goal" is a *frequency*; "1d" is a *timeframe*; "intraday" is an *objective*. Then add a `validate_combo(objective, timeframe)` step that consults `timeframes.yaml` buckets and rejects `intraday + 1d` before it reaches the YAML.

---

### Bug C — Stop-loss / take-profit ignored on first turn

**Symptom:** User says "5% profit daily" on first turn; system asks them to confirm defaults (1.5% SL → 3% TP) and tags accepted values as `user_confirmed_default`.

**Code path:**
1. [builder.py:1721-1733](app/services/strategy/builder.py#L1721-L1733) — `_extract_rms_from_text()` correctly extracts `take_profit_pct = 5.0` with `source="user"` from "5% profit".
2. [builder.py:2374-2383](app/services/strategy/builder.py#L2374-L2383) — Stored at `risk_execution_config["take_profit_pct"] = 5.0`; mirrored to `builder.take_profit = 5.0`.
3. **Stop-loss is never inferred** — [builder.py:1708-1717](app/services/strategy/builder.py#L1708-L1717) only matches `stop[\s\-]*loss|stoploss|sl|stop` keywords; "profit" alone is not interpreted symmetrically.
4. [chat_service.py:3811-3822](app/services/chat/chat_service.py#L3811-L3822) — `sl_missing = True`; `tp_missing = False` because `builder.take_profit is not None` and `"take_profit_pct" in rms_sources` and `not has_rr`. Correct so far.
5. [chat_service.py:3829-3893](app/services/chat/chat_service.py#L3829-L3893) — Because `sl_missing OR tp_missing`, the assistant sends the clarification message **even though only SL is missing**. The message proposes both `default_sl = 1.5` and `default_tp = 3.0`, even though the user already stated TP = 5%. The proposal contradicts the user's input.
6. [chat_service.py:3834](app/services/chat/chat_service.py#L3834) — Default value `1.5` comes from `MARKET_CONFIG["indian_stocks"]["default_stop_loss"]` = 1.5 ([compat.py:44](app/kb/compat.py#L44)). It is a literal in compat.py, not user-tunable.
7. [chat_service.py:3897-3905](app/services/chat/chat_service.py#L3897-L3905) — On user acceptance, `if sl_missing: existing_sources["stop_loss_pct"] = "user_confirmed_default"` runs (correct). `if tp_missing: existing_sources["take_profit_pct"] = "user_confirmed_default"` runs only if `tp_missing` is True. **However**, since the user's take-profit value (5%) was already in `risk_execution_config`, `apply_defaults()` ([builder.py:967-972](app/services/strategy/builder.py#L967-L972)) honors it. So the 5% survives — but the system's prose has already lied to the user by proposing 3% as the default.
8. The `default_stop_loss = 1.5` literal originates in [compat.py:44](app/kb/compat.py#L44). Per the comment at [compat.py:36-40](app/kb/compat.py#L36-L40), this is a "sensible Indian-stocks default that match[es] what the legacy YAML used to ship with" — it has no per-asset, per-tier, or per-timeframe basis. It's there because the legacy file had it.

**Root cause:** Two related bugs.
- (a) The clarification gate at [chat_service.py:3829] fires on `sl_missing OR tp_missing`, then proposes defaults for **both** SL and TP without checking which of them is actually missing. It should propose only the missing field(s).
- (b) The "user_confirmed_default" tag is computed before `apply_defaults` runs and never inspects whether the value about to be applied is the user's or the system's. The audit trail conflates "user accepted default" with "user is OK with system defaults for whatever isn't already specified".

**Blast radius:** All first-turn risk-aware prompts. Every conversation where the user states one risk parameter but not its symmetric counterpart will receive a clarification that proposes contradictory numbers and pollutes the source-tracking audit.

**Fix direction:** Separate per-field "is missing" booleans from the message-composition logic. The clarification text should propose defaults **only for `sl_missing`** when `tp_missing` is False (and vice versa). The `"user_confirmed_default"` source should only be written when `apply_defaults` actually substitutes a value the user did not provide. The `1.5%` literal should be deleted from `compat.py:44` and replaced with `tier.base_sl_pct` (or the volatility-adjusted ATR baseline already in `param_resolver._BASELINE_ATR_PCT = 1.5` at [param_resolver.py:28](app/planner/param_resolver.py#L28)) — there's no reason for two different files to enshrine "1.5" independently.

---

### Bug D — Scattered / duplicated defaults

See §4 for the full inventory table. Headline conflicts:

- **stop_loss_pct**: `1.5` ([compat.py:44](app/kb/compat.py#L44)), `0.25` ([risk_tiers.yaml:23,39,55,71](app/kb/risk_tiers.yaml)), `2.0` (`app/db/models/execution.py:64`, [builder.py:1000](app/services/strategy/builder.py#L1000)), `0.25` ([builder.py:965](app/services/strategy/builder.py#L965) trailing fallback). Order of precedence in `apply_defaults`: `risk_cfg → tier.base_sl_pct → cfg.default_stop_loss → 0.25`. **The first-found-wins chain means whichever path populated the dict first determines the answer** — not a deliberate hierarchy.
- **take_profit_pct**: `3.0` ([compat.py:45](app/kb/compat.py#L45)), `4.0` ([compat.py:51](app/kb/compat.py#L51), indices), `0.5` ([risk_tiers.yaml](app/kb/risk_tiers.yaml#L24)), `5.0` (`app/db/models/execution.py:65`, [builder.py:971,1001](app/services/strategy/builder.py#L971)).
- **timeframe**: `"15m"` ([compat.py:46](app/kb/compat.py#L46)), `"1d"` ([builder.py:974](app/services/strategy/builder.py#L974)), `[1m,...,1d]` ([timeframes.yaml:2](app/kb/timeframes.yaml#L2)).
- **market**: `"indian_stocks"` literal in 5+ places ([builder.py:37,405,1130,2272](app/services/strategy/builder.py); [chat_service.py:3449](app/services/chat/chat_service.py#L3449)).
- **default_tp_rr**: Hardcoded `2.0` at [chat_service.py:3836](app/services/chat/chat_service.py#L3836). Nowhere else.
- **default_dlc**: Hardcoded `3.0` at [chat_service.py:3838](app/services/chat/chat_service.py#L3838). But `apply_defaults` uses `tier.daily_loss_cap_pct` (`1.0/2.0/3.0`) at [builder.py:980](app/services/strategy/builder.py#L980), clamped `[1.0, 5.0]` at [compat.py:79](app/kb/compat.py#L79). So `chat_service` shows 3.0 to the user, but `apply_defaults` may write a different value.

**Root cause:** No source-of-truth registry. The KB YAML files (`risk_tiers.yaml`, `timeframes.yaml`) were intended as the authoritative store, but `compat.py`, `chat_service.py`, `builder.py`, and the SQL model all carry their own literals that were never migrated.

**Blast radius:** The user never sees a consistent default. Logs report values that differ from what the YAML claims is the default. New developers don't know which file to edit.

**Fix direction:** Make `app/kb/risk_tiers.yaml` (plus a new `app/kb/defaults.yaml`) the single source. Delete `compat.get_all_market_configs()` once its callers migrate. Delete `app/db/models/execution.py` numeric defaults — execution-layer defaults should be set in code from KB on insert, not in the DDL.

---

### Bug E — Bloated chat response with duplicates

See §5 for the field-level table. Headline:

- `ChatMessage.strategy_json` is `{"context": {...}, "backtest_result": {...}}` — the `"context"` wrapper carries 8+ fields, several of which (`strategy_config`, `next_state`, `current_mode`, `yaml_path`) are then stripped by the API filter at `app/api/.../chat.py` before serving to the client. They are still persisted to the DB.
- `session_id` is stored both in `ChatMessage.session_id` (column) and `strategy_json["context"]["session_id"]`.
- `strategy_id` same: column + nested.
- `org_id`, `user_id` are stored in `context` but hardcoded/derivable and never re-read.
- `strategy_draft` carries `kb_query`, `kb_snippets` (internal planner debug) that the API also strips.
- `risk_and_execution` is overlaid 3× at `chat.py:317-320` (per agent E) — three different sources writing to the same keys in order.
- Compose flow has **5 layers**: `response_composer.compose_response` (returns markdown string only), `persona_responder.compose_milestone_response` (returns markdown string only), `response_guard.guard_dynamic_assistant_reply` (returns string or None), `strategy_flow.build_final_strategy_payload` (returns the `context` dict), and `chat_service.run_ai_processing` assembling the final `ChatMessage`. No layer owns the schema.

**Root cause:** Free-form `dict` rather than a typed model. `app/schemas/chat.py` `ChatMessageItem` declares `model_config = ConfigDict(extra="allow")` — explicit permission to carry anything. Each developer added a key in whichever layer was convenient at the time.

**Fix direction:** Define a typed `AssistantResponse` model with exactly the fields the client needs; fail-loud on extras; remove the `context` wrapper; remove keys already on `ChatMessage` columns.

---

### Bug F — Signal derivation logic

What `semantic_extractor.py` (1,397 lines) actually does:
- 20 distinct extractor methods, each backed by its own regex table. Catalogued in §6.
- Public entry `extract(prompt, *, session_id)` at [semantic_extractor.py:380-451](app/planner/semantic_extractor.py#L380-L451).
- **It does not pick signals.** It produces a `SemanticInstructions` record (HTF rules, structural SL, RR, ref-symbol conditions, session filters, volume/momentum filters, candle filter, MA/VWAP/MACD/RSI state hints, direction, gap filter, cooldown, etc.).

How `catalog_signal_picker.py` (804 lines) actually picks:
- Entry `pick_plan_from_catalog(...)` at [catalog_signal_picker.py:726-804](app/planner/catalog_signal_picker.py#L726-L804).
- Step 1: `_find_all_matches()` ([:240-298](app/planner/catalog_signal_picker.py#L240-L298)) iterates every signal card and regex-matches each `match_when.phrases` entry against the prompt. Confidence = `base + 0.2 * (span_len + phrase_len)/200` ([:119-128](app/planner/catalog_signal_picker.py#L119-L128)).
- Step 2: `_resolve_role_conflicts()` ([:301-375](app/planner/catalog_signal_picker.py#L301-L375)) keeps one entry_trigger per family.
- Step 3: `_apply_exit_on_opposite()` mirrors entry to exit if requested.
- Step 4: `auto_fill_missing_families()` ([:579-657](app/planner/catalog_signal_picker.py#L579-L657)) inserts defaults for indicator families the user *named* but the planner didn't pick.

How `param_resolver.py` (246 lines) sets indicator params:
- `resolve_params(card, *, timeframe, symbol, objective, ohlcv)` at [param_resolver.py:37-85](app/planner/param_resolver.py#L37-L85).
- Three-tier hierarchy: (1) `signal_performance_cache.get_best_params` (only fires with ≥10 historical trades), (2) `estimate_params_for_signal` from OHLCV (only with ≥20 bars), (3) `card.params_by_timeframe[tf]`.
- SL/TP resolved separately via `resolve_sl_tp` ([param_resolver.py:155-225](app/planner/param_resolver.py#L155-L225)) using `_BASELINE_ATR_PCT = 1.5`, regime multipliers `_REGIME_SL_MULT / _REGIME_TP_MULT` ([:140-151](app/planner/param_resolver.py#L140-L151)), and `parse_risk_reward` for user RR.

For **"I want to trade ETH/USDC for 5% daily profit"**:
- `semantic_extractor` extracts nothing (no HTF, no SL anchor, no RR, no indicators, no direction).
- `catalog_signal_picker._find_all_matches` returns `[]` — no signal card's `match_when.phrases` mentions ETH, USDC, or "5% profit".
- `auto_fill_missing_families` does nothing — the user named no indicator family.
- Pipeline falls back to preset detection at [pipeline.py:158-162](app/planner/pipeline.py#L158-L162); no preset matches crypto.
- Final result is either an empty plan, a `NoValidCandidate` exception, or — if a preset is fuzzily matched on some other keyword — an arbitrary preset's signal stack. **The reasoning is not defensible**: it is pure keyword luck.

**Feedback loops:** `signal_performance_cache` is read once at [param_resolver.py:49-56](app/planner/param_resolver.py#L49-L56), *after* signals are picked. It tunes period parameters, not signal choice. There is **no reranking based on past win-rate**, no online learning, no backtest-score gradient. The system is deterministic on text features.

---

### Bug G — Indicator selection intelligence

**Does the system distinguish trend / mean-reversion / breakout / momentum?** Partially.
- The KB has `signal_family` metadata per signal card (`SMA / EMA / RSI / MACD / SUPERTREND / STOCHASTIC / ...`) used at [fidelity_validator.py:55-73](app/planner/fidelity_validator.py#L55-L73).
- The `strategy_family_preserver.py` carries hardcoded `FAMILY_SPECS` with `contraindicated_signals` lists (e.g., ORB ↔ VWAP_RECLAIM mutual exclusion at lines 37-150). But this only fires if a family was already detected from user prompt; if no family detected, the contraindicated matrix is never consulted.

**Fallback chain when user provides zero indicator hints:**
1. `semantic_extractor.extract()` returns all-empty (lines 409-433).
2. `pick_plan_from_catalog` returns no matches.
3. `auto_fill_missing_families` does nothing (no mentions).
4. Pipeline falls back to `_resolve_preset` ([pipeline.py:158-162](app/planner/pipeline.py#L158-L162)) which tries explicit preset → semantic base_framework → goal-text keyword scan.
5. If none match: `NoValidCandidate` raised at [pipeline.py:563](app/planner/pipeline.py#L563).

There is **no "balanced default" strategy** for the zero-hint case.

**Conflict detection for incompatible indicators:**
- `fidelity_validator.py` checks: indicator-family swaps, period mismatches, fabricated filters, direction mismatch, RR mismatch, default-sourced SL/TP — but it does not flag "RSI overbought (mean-reversion) + breakout (trend-continuation)" as a logical conflict.
- The only place such semantics live is the hardcoded `FAMILY_SPECS.contraindicated_signals` matrix in `strategy_family_preserver.py`, and that requires the user to have explicitly named a family.

**Net assessment:** "Intelligence" is two regex passes (semantic_extractor for gates; catalog_signal_picker for picks) over a hand-tagged YAML catalog. Output quality is bounded by phrase-table coverage, which is brittle and uneven across signal cards. There is no scoring against the user's stated *objective* (only against `intent.style` from `intent_extractor` LLM call), and no semantic conflict check.

---

## 4. Defaults Inventory Table

| Field | Default | Defined in (file:line) | Type | Read by |
|---|---|---|---|---|
| `DEFAULT_MARKET` | `"indian_stocks"` | [builder.py:37](app/services/strategy/builder.py#L37) | hardcoded literal | builder init, `_normalise_market`, `extract_strategy_details` |
| `market` (init) | `DEFAULT_MARKET` | [builder.py:405](app/services/strategy/builder.py#L405) | constructor assignment | every turn |
| `market` (normaliser) | `DEFAULT_MARKET` | [builder.py:1130](app/services/strategy/builder.py#L1130) | static no-op | `merge_preview` |
| `market` (re-assert) | `DEFAULT_MARKET` | [builder.py:2272](app/services/strategy/builder.py#L2272) | every-turn override | `extract_strategy_details` |
| `market` (fallback) | `"indian_stocks"` | [chat_service.py:3449](app/services/chat/chat_service.py#L3449) | `or "indian_stocks"` | strategy persistence |
| `stop_loss_pct` (compat / market) | `1.5` | [compat.py:44](app/kb/compat.py#L44) | hardcoded in helper | `apply_defaults`, chat_service:3834 |
| `stop_loss_pct` (compat / indices) | `1.5` | [compat.py:50](app/kb/compat.py#L50) | hardcoded | indices market path |
| `stop_loss_pct` (schema) | `0.25` | `app/kb/schemas.py:295` | Pydantic default | RiskTier construction |
| `stop_loss_pct` (DB) | `2.0` | `app/db/models/execution.py:64` | SQL column default | DB insert if app omits |
| `stop_loss_pct` (apply_defaults fallback) | `0.25` | [builder.py:965](app/services/strategy/builder.py#L965) | inline literal | terminal fallback |
| `stop_loss_pct` (reward_factor fallback) | `2.0` | [builder.py:1000](app/services/strategy/builder.py#L1000) | inline literal | RR display only |
| `take_profit_pct` (compat / market) | `3.0` | [compat.py:45](app/kb/compat.py#L45) | hardcoded | apply_defaults |
| `take_profit_pct` (compat / indices) | `4.0` | [compat.py:51](app/kb/compat.py#L51) | hardcoded | indices path |
| `take_profit_pct` (schema) | `3.0` | `app/kb/schemas.py:296` | Pydantic default | RiskTier construction |
| `take_profit_pct` (DB) | `5.0` | `app/db/models/execution.py:65` | SQL column default | DB insert |
| `take_profit_pct` (apply_defaults fallback) | `5.0` | [builder.py:971](app/services/strategy/builder.py#L971) | inline literal | terminal fallback |
| `take_profit_pct` (reward_factor fallback) | `5.0` | [builder.py:1001](app/services/strategy/builder.py#L1001) | inline literal | RR display only |
| `default_tp_rr` (clarification message) | `2.0` | [chat_service.py:3836](app/services/chat/chat_service.py#L3836) | inline literal | user-facing prose |
| `default_dlc` (clarification message) | `3.0` | [chat_service.py:3838](app/services/chat/chat_service.py#L3838) | inline literal | user-facing prose |
| `default_timeframe` (compat / market) | `"15m"` | [compat.py:46](app/kb/compat.py#L46) | hardcoded | apply_defaults |
| `default_timeframe` (compat / indices) | `"15m"` | [compat.py:52](app/kb/compat.py#L52) | hardcoded | indices path |
| `timeframe` (apply_defaults fallback) | `"1d"` | [builder.py:974](app/services/strategy/builder.py#L974) | inline literal | terminal fallback |
| `timeframe` (strategy persist fallback) | `"1d"` | [chat_service.py:3450](app/services/chat/chat_service.py#L3450) | `or "1d"` | DB write |
| `supported timeframes` | `[1m,3m,5m,10m,15m,30m,1h,1d]` | [timeframes.yaml:2](app/kb/timeframes.yaml#L2) | YAML | planner |
| `SUPPORTED_USER_TIMEFRAMES` | `(1m,5m,10m,15m,30m,1h,1d)` | [builder.py:67-75](app/services/strategy/builder.py#L67-L75) | tuple literal | chat validation (note: excludes 3m which yaml supports) |
| `objective` (intraday max_trade label) | `"1 trading day"` | [builder.py:989](app/services/strategy/builder.py#L989) | inline literal | YAML output |
| `objective` (positional max_trade) | `"5 trading days"` | [builder.py:991](app/services/strategy/builder.py#L991) | inline literal | YAML output |
| `objective` (fallback / `None`) | `"3 trading days"` | [builder.py:993](app/services/strategy/builder.py#L993) | inline literal | YAML output |
| `max_trades_per_day` | `2 if intraday else 1` | [builder.py:1010-1012](app/services/strategy/builder.py#L1010-L1012) | hardcoded | risk_and_execution_summary |
| `daily_loss_cap_pct` (beginner) | `1.0` | [risk_tiers.yaml:21](app/kb/risk_tiers.yaml#L21) | YAML | apply_defaults |
| `daily_loss_cap_pct` (intermediate) | `2.0` | [risk_tiers.yaml:37](app/kb/risk_tiers.yaml#L37) | YAML | apply_defaults |
| `daily_loss_cap_pct` (expert) | `3.0` | [risk_tiers.yaml:53](app/kb/risk_tiers.yaml#L53) | YAML | apply_defaults |
| `daily_loss_cap_pct` (default tier) | `2.0` | [risk_tiers.yaml:69](app/kb/risk_tiers.yaml#L69) | YAML | apply_defaults |
| `daily_loss_cap` (`risk_summary` fallback) | `3.0` | [builder.py:1053](app/services/strategy/builder.py#L1053) | inline literal | UI display |
| `daily_loss_cap` (DB) | `3.0` | `app/db/models/execution.py:68` | SQL default | DB insert |
| `daily_loss_cap` (clamp bounds) | `(1.0, 5.0)` | [compat.py:77-79](app/kb/compat.py#L77-L79) | function return | apply_defaults |
| `per_trade_risk_pct` (beginner/inter/expert/default) | `1.0/2.0/3.0/2.0` | [risk_tiers.yaml:20,36,52,68](app/kb/risk_tiers.yaml) | YAML | apply_defaults |
| `per_trade_risk` (DB) | `2.0` | `app/db/models/execution.py:69` | SQL default | DB insert |
| `per_trade_risk` (fallback) | `2.0` | [builder.py:1008](app/services/strategy/builder.py#L1008) | inline | `_per_trade_risk_pct` |
| `base_sl_pct` (all tiers) | `0.25` | [risk_tiers.yaml:23,39,55,71](app/kb/risk_tiers.yaml) | YAML | apply_defaults via tier |
| `base_tp_pct` (all tiers) | `0.5` | [risk_tiers.yaml:24,40,56,72](app/kb/risk_tiers.yaml) | YAML | apply_defaults via tier |
| `max_open_positions` (b/i/e/d) | `2/3/5/3` | [risk_tiers.yaml:22,38,54,70](app/kb/risk_tiers.yaml) | YAML | apply_tier_execution_defaults |
| `max_consecutive_losses` (all) | `3` | [risk_tiers.yaml:25,41,57,73](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `cooldown_bars_after_loss` (b/i/e/d) | `3/1/0/1` | [risk_tiers.yaml:26,42,58,74](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `cooldown_bars_after_profit` (b/i/e/d) | `1/0/0/0` | [risk_tiers.yaml:27,43,59,75](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `max_spread_bps` (all) | `0` (disabled) | [risk_tiers.yaml:28,44,60,76](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `entry_confirmation_bars` (b/i/e/d) | `2/1/1/1` | [risk_tiers.yaml:29,45,61,77](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `max_capital_allocation_pct` (b/i/e/d) | `10/20/30/20` | [risk_tiers.yaml:30,46,62,78](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `position_sizing_mode` (b/i/e/d) | `fixed_fractional/risk_based/risk_based/risk_based` | [risk_tiers.yaml:31,47,63,79](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `gap_filter` (all) | `"none"` | [risk_tiers.yaml:32,48,64,80](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `min_trade_value` (b/i/e/d INR) | `5000/2000/1000/2000` | [risk_tiers.yaml:33,49,65,81](app/kb/risk_tiers.yaml) | YAML | tier execution |
| `min_trade_value` (DB) | `500.0` | `app/db/models/execution.py:167` | SQL default | DB insert |
| `cash_reserve_pct` | `0.1` | [param_resolver.py:236](app/planner/param_resolver.py#L236) | inline literal | param_resolver |
| `_BASELINE_ATR_PCT` | `1.5` | [param_resolver.py:28](app/planner/param_resolver.py#L28) | module constant | resolve_sl_tp |
| `_VOL_MULT_MIN / _VOL_MULT_MAX` | `0.5 / 3.0` | [param_resolver.py:29-30](app/planner/param_resolver.py#L29-L30) | module constants | resolve_sl_tp |
| `_REGIME_SL_MULT` | `{trending_up:1.1, trending_down:1.1, ranging:0.85, volatile:1.4}` | [param_resolver.py:140-143](app/planner/param_resolver.py#L140-L143) | dict literal | resolve_sl_tp |
| `_REGIME_TP_MULT` | `{trending_up:1.2, trending_down:1.2, ranging:0.9, volatile:1.0}` | [param_resolver.py:148-151](app/planner/param_resolver.py#L148-L151) | dict literal | resolve_sl_tp |
| `regime` (no OHLCV) | `"ranging"` | [pipeline.py:54](app/planner/pipeline.py) | inline literal | param_resolver fallback |
| `intent.hold_horizon / frequency / profit_size / style / risk_appetite` | `hours / medium / medium / trend / moderate` | [intent_taxonomy.yaml:12-16](app/kb/intent_taxonomy.yaml) | YAML | intent_extractor fallback |
| `RSI window` | `14` | `app/kb/signals/rsi_*.yaml` (params_by_timeframe) | YAML | param_resolver |
| `EMA short/long (intraday)` | `5 / 13` | `app/kb/signals/ema_above.yaml:36-46` | YAML | param_resolver |
| `EMA short/long (swing)` | `9 / 21` | `app/kb/signals/ema_above.yaml:48-51` | YAML | param_resolver |
| `ATR window` | `14` | `app/kb/signals/atr_*.yaml` | YAML | param_resolver |
| `_MIN_TRADES_FOR_CONFIDENCE` | `10` | `app/core/signal_performance_cache.py:133` | module constant | get_best_params |
| `markets.yaml` (exchanges) | `NSE (.NS) / BSE (.BO)` only | [markets.yaml:1-19](app/kb/markets.yaml) | YAML | format_symbol |

**Conflicts (same field, different values):**

| Field | Conflicting values | Where |
|---|---|---|
| `stop_loss_pct` | `1.5%`, `0.25%`, `2.0%` | compat.py:44 vs risk_tiers.yaml:23 vs execution.py:64 |
| `take_profit_pct` | `3.0% / 4.0% / 5.0% / 0.5%` | compat.py:45,51 vs schemas vs execution.py:65 vs risk_tiers (base_tp_pct) |
| `default_timeframe` | `"15m"` vs `"1d"` | compat.py:46 vs builder.py:974 |
| `default_dlc` (shown to user) | `3.0` (chat_service:3838) vs `2.0` (intermediate tier) vs `[1.0, 5.0]` (clamp) | three places |
| `default_stop_loss` (shown) | `1.5` (chat_service:3834 via compat) vs `0.25` (tier base actually written) | clarification UI lies to user vs apply_defaults reality |
| `min_trade_value` | `5000 / 2000 / 1000 / 500` | risk_tiers.yaml vs execution.py:167 |
| `supported timeframes` | `[1m,3m,5m,10m,15m,30m,1h,1d]` vs `(1m,5m,10m,15m,30m,1h,1d)` | timeframes.yaml:2 vs builder.py:67 (3m dropped in chat layer) |

---

## 5. Response Payload Audit

**Assembly layers (5):**
1. [response_composer.py:95](app/services/chat/response_composer.py#L95) `compose_response(code, **facts)` — returns a markdown string only.
2. [persona_responder.py:295](app/services/chat/persona_responder.py#L295) `compose_milestone_response(event, ctx, llm)` — returns a markdown string (LLM-varied or templated fallback).
3. [response_guard.py:52](app/services/chat/response_guard.py#L52) `guard_dynamic_assistant_reply(intent, reply_text)` — returns a cleaned string or None.
4. [strategy_flow.py:416](app/services/chat/strategy_flow.py#L416) `build_final_strategy_payload(...)` — returns `{"context": {...}, "success": bool, ...}`.
5. [chat_service.py:4346-4360](app/services/chat/chat_service.py) builds the `ChatMessage` ORM object.

**Duplicate fields:**

| Field | Locations |
|---|---|
| `session_id` | ChatMessage.session_id (column) + `strategy_json["context"]["session_id"]` (strategy_flow.py:430) |
| `strategy_id` | ChatMessage.strategy_id (column) + `strategy_json["context"]["strategy_id"]` (strategy_flow.py:438) |
| `user_id` | persisted via session + `strategy_json["context"]["user_id"]` (strategy_flow.py:432) |
| `org_id` | hardcoded `DEFAULT_ORG_ID` + `strategy_json["context"]["org_id"]` (strategy_flow.py:431) |
| `risk_and_execution` | 3× overlay at `app/api/.../chat.py:317-320` |
| `strategy_object` | inside `context` AND filtered/extracted at API layer |

**Always-null / stripped-by-API fields persisted to DB:**
- `strategy_draft["kb_query"]` and `kb_snippets` — planner debug; stripped by API
- `strategy_json["context"]["strategy_config"]` — stripped by API
- `strategy_json["context"]["next_state"]`, `current_mode`, `yaml_path` — stripped by API
- `strategy_draft["validation_flags"]` — set by persona_responder:254, stripped
- `strategy_draft["user_rejection"]`, `agent_decision` — set by chat_service tracking, not consumed

**Stringified duplicates:**
- Backtest metrics: structured dict in `strategy_json["backtest_result"]["metrics"]` AND rendered as markdown table inside `content`.
- Signals: `strategy_draft["entry_names"]/["exit_names"]` (display strings) AND `strategy_json["context"]["strategy_object"]["signals"]` (full objects).
- Risk params: `strategy_draft["sl_pct"]/["tp_pct"]` (floats) AND `strategy_object["risk_and_execution"]` (dict).

**Schema:** `ChatMessageItem` at `app/schemas/chat.py:103-120` declares `model_config = ConfigDict(extra="allow")` — explicit permission for arbitrary keys. `MessageResponse` at `:71-100` declares `strategy_preview` field that is **never populated by any code path**.

---

## 6. Signal & Indicator Selection Logic (Flowchart in Prose)

```
User text
   │
   ▼
intent_extractor.extract(goal, kb.intent_taxonomy)   ← only LLM call in planner
   │   returns Intent{style, frequency, hold_horizon, profit_size, risk_appetite}
   │
   ▼
semantic_extractor.extract(prompt)                   ← 20-extractor regex barrage
   │   returns SemanticInstructions{htf_rules, structural_sl, rr,
   │                                ref_symbols, sessions, volume_momentum,
   │                                candle, indicators, direction, gap_filter,
   │                                cooldown, spread, confirmation, rsi_thresholds,
   │                                macd_states, vwap_relations, ma_relations,
   │                                exit_on_opposite}
   │
   ▼
_resolve_preset(builder)                              ← preset.py:373-460
   │   tries: (a) explicit builder.strategy_preset
   │          (b) semantic_intent.base_framework
   │          (c) kb.detect_preset_in_text(goal)  ← keyword scan
   │
   ├─ preset found → _apply_preset → pin signal stack from preset YAML
   │
   └─ no preset → fall through to filter+rank
         │
         ▼
      pick_plan_from_catalog(prompt, kb, tf, sentiment, ...)
         │   step 1: _find_all_matches      ← regex over signal_card.match_when.phrases
         │   step 2: _resolve_role_conflicts ← keep 1 entry_trigger/family, dedup
         │   step 3: _build_signal           ← extract regex captures → params
         │   step 4: _apply_exit_on_opposite ← optional mirror
         │
         ├─ matches exist → confidence scoring (specificity bonus)
         └─ no matches    → auto_fill_missing_families
                              │
                              ├─ user named indicator family in prompt
                              │     → _pick_family_default_signal (6-step preference)
                              │
                              └─ user named nothing → return empty plan
                                    │
                                    └─ pipeline raises NoValidCandidate
   │
   ▼
hard_filters.filter_candidates    ← sentiment match, timeframe support, role
soft_ranker.pick(intent, experience, tf, avoid_family, pair_with)
   │   pick entry_trigger → loop entry_filters (avoid family dup, max 3) → exit_trigger
   │
   ▼
semantic_signal_composer.inject_entry_filters(semantic_intent_dict, ...)
   │   overlays HTF rules, ADX threshold, RS condition, volume filter
   │
   ▼
classify_regime(ohlcv)  ← quant_engine; default "ranging" if no data
   │
   ▼
For each picked signal: resolve_params(card, tf, symbol, objective, ohlcv)
   │   tier 1: signal_performance_cache.get_best_params (≥10 trades)
   │   tier 2: estimate_params_for_signal (≥20 bars)
   │   tier 3: card.params_by_timeframe[tf]
   │
   ▼
resolve_sl_tp(risk_tier, ohlcv, user_rr, regime)
   │   compute realized ATR%, vol_mult = ATR%/_BASELINE_ATR_PCT (clamp [0.5,3.0])
   │   sl = base_sl_pct × vol_mult × _REGIME_SL_MULT[regime]
   │   tp = sl × (user_rr or _REGIME_TP_MULT[regime])
   │
   ▼
StrategyPlan assembled → validate_plan(plan)
   │   checks: entry_trigger direction matches sentiment
   │           exit_trigger direction is opposite sentiment
   │           entry_filters all match sentiment
   │           no signal in two roles
   │           sl/tp > 0 and tp >= sl
   │
   ▼
builder.apply_signal_plan(plan) → builder.to_yaml_dict() → apply_defaults() → YAML on disk
```

**Honest assessment of "intelligence":**

- The pipeline has the *shape* of a planner: extract → match → rank → tune. But the matching layer (`catalog_signal_picker`) is pure regex-over-phrase-tables, independent of the (excellent) structured intent that `semantic_extractor` produces. If the user's wording doesn't match any signal card's hand-tagged phrase, the planner falls into preset detection, which is more keyword luck.
- The only ML/data-driven component is `signal_performance_cache` — and it informs **parameter tuning**, not signal selection.
- The only LLM call in the planner is `intent_extractor` for the soft Intent fields, which then feed `soft_ranker` (not the catalog picker).
- Conflict detection is hand-coded in `strategy_family_preserver` and only runs when a family is explicitly detected.
- Selection is **brittle keyword matching dressed up as a multi-stage planner**.

---

## 7. Validation Gap Matrix

| Combination that should be rejected | Validator? | Where |
|---|---|---|
| `intraday + timeframe=1d` | **No** | No code consults `timeframes.yaml` buckets. validator.py only checks direction. |
| `crypto symbol + market=indian_stocks (NSE)` | **No** | No symbol→market validation. `_normalise_market` is a no-op. |
| `take_profit ≤ stop_loss` | **Partial** | [validator.py:56-58](app/planner/validator.py#L56-L58): `if tp < sl: raise`. Misses `tp == sl`. Runs only on `StrategyPlan`, not on `builder.stop_loss / builder.take_profit`. |
| `tp_pct > 0 and sl_pct > 0` | Yes | [validator.py:54-55](app/planner/validator.py#L54-L55) |
| `entry_trigger.direction == sentiment` | Yes | [validator.py:22-26](app/planner/validator.py#L22-L26) |
| `exit_trigger.direction == opposite(sentiment)` | Yes | [validator.py:29-34](app/planner/validator.py#L29-L34) |
| signal appears in two roles | Yes | [validator.py:44-51](app/planner/validator.py#L44-L51) |
| `stop_loss > 100%` (sanity) | **No** | No upper bound anywhere. |
| `position_size > 100%` | **No** | `max_capital_allocation_pct < 100` is read but never enforced. |
| `daily target ≫ feasible on chosen timeframe` (e.g. 5% daily on 1m) | **No** | Daily target is not even a typed field; only inferred from prose. |
| `holding_period > timeframe granularity` (e.g. 1m candles + 5-day max_trade) | **No** | `max_trade` is a display string, not validated. |
| `frequency keyword vs trade_type` (`"daily"` → frequency or objective?) | **No** | Two regex extractors silently disagree. |
| trend signal + counter-trend signal in same plan | **Partial** | Only if `strategy_family_preserver.FAMILY_SPECS` matrix triggered AND family pre-detected from prompt. |
| `unsupported timeframe` (e.g., `2h`) | Yes | [builder.py:67-89](app/services/strategy/builder.py#L67-L89) `SUPPORTED_USER_TIMEFRAMES` + `unsupported_user_timeframe_validation_facts`. (But silently drops the supported `3m`.) |
| RR mismatch between user-stated and planner-emitted | Yes (warning) | [fidelity_validator.py](app/planner/fidelity_validator.py) lines 286-333 — produces warning, not rejection |
| SL/TP sourced from defaults when user specified | Yes (warning) | [fidelity_validator.py](app/planner/fidelity_validator.py) lines 354-377 — warning only |
| Fabricated entry filter not requested by user | Yes (warning) | [fidelity_validator.py](app/planner/fidelity_validator.py) lines 240-260 — warning only |

**Key gap:** `validate_plan` runs on the `StrategyPlan`, *after* the catalog picker. But the `StrategyBuilder` produces the YAML through `to_yaml_dict() → apply_defaults() → yaml_generator`, which **bypasses `validate_plan` entirely**. There is no validation gate between `builder.apply_defaults()` and `yaml_generator.generate_yaml()`.

---

## 8. Architectural Recommendations (ranked by leverage)

1. **Decompose `chat_service.py` (4,426 lines) into intent-routed handlers.** The current monolith mixes routing, extraction, validation, state restoration, RMS clarification, signal planning orchestration, persistence, and response composition in a single function (`run_ai_processing`). Extract one handler per agent_router intent (`new_strategy`, `collect_input`, `risk_execution_update`, `market_inquiry`, `user_rejection`, `pause_workflow`). Each handler becomes 100–300 lines, individually testable. **Highest leverage**: most bugs in this audit live inside or around `run_ai_processing`.
2. **Make `market` derived, not initial.** Remove `DEFAULT_MARKET` from `builder.py:37/405/2272/1130` and `chat_service.py:3449`. Introduce `derive_market(symbol, kb) → MarketSpec` and `_normalise_market` becomes an actual normaliser. Extend `markets.yaml` and `Stock.asset_class` plumbing to support crypto. **Closes Bug A entirely.**
3. **Collapse defaults to one layer.** Pick `app/kb/risk_tiers.yaml` as the single source of truth for risk/execution defaults; add `app/kb/markets.yaml`-driven market defaults. Delete `compat.get_all_market_configs` literals (1.5, 3.0, 4.0, "15m"), delete SQL column defaults in `app/db/models/execution.py`, delete inline literals in `builder.py:965-1001`. Add a CI lint that fails on `default_` literals outside `app/kb/`. **Closes Bugs C, D simultaneously.**
4. **Add a single `validate_strategy_yaml(builder)` gate before `yaml_generator.generate_yaml`.** It should consult `timeframes.yaml` buckets (intraday vs 1d), `markets.yaml` (symbol vs exchange), risk tier bounds (stop_loss in [0, X], take_profit ≥ stop_loss), and the contraindicated matrix unconditionally. **Closes most of §7.**
5. **Redraw the LLM/rules boundary.** Today, the LLM is used in two places (`intent_extractor`, `persona_responder`) for soft fields. The catalog picker is pure regex. The semantic extractor is 1,397 lines of regex tables. Move the *extraction* of fields (HTF, SL anchor, RR, indicators, direction) from regex to an LLM with a strict JSON schema; keep rules-based for *gating* (validator, family-conflict matrix, timeframe bucket). Rule-of-thumb: LLM for *understanding what the user said*, rules for *deciding whether it's allowed*.
6. **Type the chat response.** Replace `dict` + `ConfigDict(extra="allow")` with an explicit `AssistantResponse` Pydantic model. Remove the `context` wrapper. Remove keys already on `ChatMessage` columns. Drop API-stripped fields entirely. **Closes Bug E and prevents new fields from drifting in silently.**
7. **Split `semantic_extractor.py` (1,397 lines) into one module per extractor family.** `htf.py`, `stop_loss.py`, `risk_reward.py`, etc. Keep the facade class but move all 20 pattern tables out of one file. The single-file form encourages copy-pasta regex; modularising forces shared helpers.
8. **Wire the performance cache into selection, not just tuning.** `signal_performance_cache.get_best_params` informs periods but not picks. Add a `score_signal_by_history(card, symbol, timeframe)` that influences `soft_ranker`. Without it, the system is genuinely deterministic-on-text and cannot improve.
9. **Introduce a `Frequency` field distinct from `Timeframe` and `Objective`.** "Daily" as profit cadence ≠ "1d" as candle ≠ "intraday" as trade horizon. Three regex passes over three named fields with no shared keywords. **Closes Bug B.**
10. **Delete the lying "user_confirmed_default" tag.** Either compute it *after* `apply_defaults` actually substitutes a value, or remove it. The current implementation tags audit fields based on what the user was *about to be asked*, not what was *actually written*.

---

## 9. Open Questions for Product/PM

1. **What asset classes does the product actually support?** The KB plumbing has `crypto_spot` typed but no `markets.yaml` entry, no compat config, no example strategies in `app/kb/presets/`. Was crypto ever shipped, or only declared?
2. **What does "5% daily profit" mean to the user?** A daily target, a per-trade target, or a cumulative weekly? The clarification flow assumes per-trade (TP=5%); the user's phrasing implies cumulative-per-day. This requires a product decision before the regex can be fixed.
3. **Is `intraday + 1d` ever legitimate?** Some firms run "session-bounded daily" strategies (enter on the open, exit by close, but using the prior 1d candle for setup). If so, the validation rule needs nuance.
4. **How much LLM trust is acceptable?** The current system is mostly rules with two LLM calls. Is the product OK with full-LLM extraction (faster to extend, more fragile) or does it want to stay regex-heavy (slower to extend, more deterministic)?
5. **Are the `MARKET_CONFIG` 1.5%/3.0%/15m defaults business-meaningful, or just historical artifacts?** The comment in `compat.py:36-40` calls them "sensible defaults that match what the legacy YAML used to ship with" — i.e., legacy artifacts. Were they ever calibrated against backtests?
6. **What is the intended response payload?** The 5-layer assembly and the schema saying "extra=allow" suggest no one owns the contract. Does the frontend rely on the persisted bloat, or would a typed model break only the server-side?
7. **What should happen when the user gives zero indicator hints?** Today: `NoValidCandidate` exception. Should the system propose a default trend-follower? A clarifying question? Refuse?
8. **Are tests intentionally locking in the wrong RR behavior?** `test_chat_integration.py:84` literally prints `"TP: 5% hardcoded, NOT 1:3!"` as if documenting a known wart. Is that a "we'll fix it later" comment or accepted behavior?

---

*End of audit.*
