# SDL Strategy Engine V2 — Complete Guide

> A plain-English guide to how Stretus turns a user's chat message into a correct,
> testable trading strategy — capturing **every** detail, guessing nothing, and
> hiding nothing. Read top to bottom; no prior knowledge assumed.

---

## 1. What is this, in one breath?

A user types something like *"Buy ETH on the 15-minute chart when RSI drops below 30,
stop loss 2%, take profit 2:1."*

This system turns that sentence into a **structured, validated strategy** that the
backtester can run — and it shows the user, in plain English, exactly what it built,
what it assumed, and what it couldn't do. **Nothing is silently dropped or invented.**

**The one-line mental model:**

> An AI **fills out a form** (the "SDL ticket") by choosing items from a **fixed menu**
> (our signal catalog). We then **check the form**, **turn it into something the engine
> can run**, **show it back to the user**, and **backtest it**. The AI only fills the
> form — it never writes the trading math.

---

## 2. The core idea (and why it's safe)

Old way (brittle): we used regex + keyword templates to guess the strategy from the
sentence. It dropped details ("short leg"), swapped indicators (asked Bollinger, got
Keltner), and faked numbers (asked 20 EMA, got 34).

New way (this system):

```
The AI understands language → fills a typed form by picking from the catalog menu.
Everything AFTER the form is deterministic (same input → same output, always).
```

Why it's safe:
- **The AI picks menu items by name; it never writes a formula.** The math always comes
  from a catalog card the engine already trusts.
- **The form records WHERE each value came from** (the user, a default, or an inference).
- **Anything the user asked for that we can't do is written down** (not ignored).
- **The user sees and confirms the form** before any backtest runs.

---

## 3. The big rules (guardrails), in plain words

1. **AI runs only at design time, once per turn.** When you create a strategy (one AI
   call) or modify it (one AI call per change). **Never** while evaluating/backtesting —
   that stays 100% deterministic.
2. **Catalog-only.** The AI may only choose signals/options that already exist in our
   catalog. If it names something that doesn't exist, validation rejects it.
3. **One path.** Prompt → SDL → Validate → Compile → Artifact → Backtest. No second
   pipeline, no "try again with a different method," no hidden fallback.
4. **Don't silently drop. Don't silently guess.** Unsupported asks are written into
   `unmapped_details`. Missing values get a sensible default **plus** a question for the
   user.
5. **Reuse the existing engine.** We do not rebuild the backtester, indicators, catalog,
   or the discovery (asset-picker) scanner. We wire into them.

---

## 4. The complete flow (the whole journey)

```
  User prompt
      │
      ▼
┌─────────────────┐
│ 1. SELECTOR     │  AI reads the prompt + the catalog "menu",
│    (the AI)     │  fills the SDL form by choosing menu items.
└─────────────────┘
      │  produces
      ▼
┌─────────────────┐
│ 2. SDL TICKET   │  A typed form: universe, legs, risk, gates, htf,
│  (the form)     │  + provenance (who said what) + unmapped + clarifications.
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ 3. VALIDATOR    │  Checks: names exist? params in range? can it ever trade?
│  (the checker)  │  every leg has exit+stop? enough history? engine can run it?
└─────────────────┘
      │  if errors → shown in read-back, user edits (back to step 1/5)
      ▼
┌─────────────────┐
│ 4. COMPILER     │  Turns the form into the engine's existing input format
│  (the builder)  │  (a strategy config). Strips out AI/provenance stuff.
└─────────────────┘
      │  produces
      ▼
┌─────────────────┐
│ 5. ARTIFACT     │  Immutable, executable strategy (version-stamped).
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ 6. EVALUATION   │  Existing backtester runs it. For a dynamic universe, the
│  (the backtest) │  discovery scanner first PICKS the asset, then backtests it.
└─────────────────┘
      │  produces
      ▼
   Backtest result (trades + in-sample / out-of-sample / per-regime metrics)
      │
      ▼
┌─────────────────┐
│ 7. READ-BACK    │  Plain-English summary: Built / Assumed / Couldn't-do +
│  + MATCH %      │  "N of M requirements captured (X%)". User confirms or edits.
└─────────────────┘
```

Two side rails that keep it honest:
- **Provenance** travels with the form, so read-back and match% are exact.
- **Offline judge + shadow mode** (dev-time only) measure how faithful the AI is, and
  catch anything the AI forgot to write down.

---

## 5. Key concepts (explained like you're new)

### The SDL ticket (the form)
A typed document that fully describes a strategy. Think of it as a **filled-in order
form**. Its sections:
- **context** — market, timeframe, objective.
- **universe** — *which asset(s)*: either a named symbol (**static**) or a rule to pick
  one (**dynamic**).
- **legs** — the entry/exit logic. One leg = one direction (long or short). Two legs =
  long **and** short (never mashed into one).
- **risk** — stop loss, take profit, trailing, scale-outs, risk:reward, sizing.
- **gates** — optional filters that block entries (regime, volume, event, session…).
- **htf_rules** — higher-timeframe confirmation.
- **provenance** — the honesty section (below).

### The catalog (the menu)
Our library of ~119 pre-built, tested signals (`app/kb/signals/*.yaml`). Each is a "card"
with a name, the real formula (used by the engine), params, and the phrases that match it.
**The AI can only pick from this menu.** It cannot invent a dish.

### Provenance (the "who said what" section)
For every field in the form, we record its source:
- `user` — the user explicitly asked for it.
- `default` — the user didn't say, so we filled a sensible default.
- `inferred` — we deduced it (e.g., "intraday" implies a short timeframe).

This is the heart of the design: it makes the read-back honest and the match% exact.

### unmapped_details (the "couldn't do" list)
If the user asks for something with no menu match, we write it here (verbatim) with a
reason — never drop it. Kinds:
- `missing_card` — no such indicator/signal in the catalog.
- `missing_operator` — a comparison our language can't express.
- `missing_param` — a parameter we can't represent.
- `engine_capability_gap` — the engine can't run this combo yet.
- `unsupported_universe` — an asset-picking rule beyond the scanner.

### clarifications_needed (the "please confirm" list)
When we had to default a money-affecting value (stop, target, risk:reward), we add a
question so the user can correct it. We **never** silently default these.

### match % (how well we captured intent)
A simple, deterministic score (no AI):
```
match = built / (built + missed)
```
- **built** = atomic user **requirements** we captured correctly.
- **missed** = atomic user requirements we wrote into `unmapped_details`.
- **Counting unit:** one *atomic user clause* that maps to one SDL field or one catalog
  signal. (So "long when above BB **and** volume **and** above 200 EMA" = three.)
- Defaults/inferred never count for or against the score.

### static vs dynamic universe (the big asset idea)
- **Static** = the user named the asset. We trade exactly that one. *(single asset)*
- **Dynamic** = the user gave a **rule** for choosing the asset ("the NSE stock with the
  highest relative volume that's above VWAP"). Our existing **discovery scanner** picks
  the asset by evaluating catalog conditions across the universe, ranking, and tie-breaking.
  Works for **Indian equity (`equity_cash`)** and **crypto spot (`crypto_spot`)**.

---

## 6. Each component (what · input → output · file)

| Component | What it does | Input → Output | File |
|---|---|---|---|
| **SDL schema** | Defines the form (types + provenance) | — | `app/planner/sdl.py` |
| **Catalog menu** | Compact machine-readable list of all signals + options the AI may pick | catalog → menu | `app/planner/catalog_schema.py` |
| **Selector (AI)** | Fills the form from the prompt; also handles modify-turns | prompt + menu → SDL | `app/planner/sdl_selector.py` |
| **Validator** | Checks names, params, satisfiability, safety, feasibility, engine-capability | SDL → ValidationResult | `app/planner/sdl_validator.py` |
| **Compiler** | Turns the validated form into the engine's existing strategy config | SDL → Artifact | `app/planner/compiler.py` |
| **Read-back + match%** | Plain-English summary + score, deterministically | SDL → text + % | `app/planner/readback.py` |
| **Evaluation** | Runs the backtest (and resolves the asset for dynamic universes) | Artifact + data → backtest result | reuse `quant_engine` + `app/services/discovery/*` |
| **Offline judge + shadow** | Dev-time measurement of faithfulness; regex vs SDL comparison | prompts → scores | `tests/test_planner/*` |

**Reused, never rebuilt:** `quant_engine` (backtester, indicators, conditions), the signal
catalog, `param_resolver.py`, `condition_satisfiability.py`, `llm.py`, and the discovery
scanner.

---

## 7. Worked Example A — STATIC asset (full end-to-end)

**User types:**
> *"Buy ETH on the 15-minute chart when RSI drops below 30. Stop loss 2%, take profit 2:1."*

### Step 1 — Selector fills the SDL
```json
{
  "context": { "market": "crypto", "timeframe": "15m", "objective": "mean_reversion" },
  "universe": { "type": "static", "asset_class": "crypto_spot", "symbol": "ETH_USDC" },
  "legs": [
    { "direction": "long",
      "entry": { "trigger": { "name": "rsi_oversold", "params": { "window": 14, "threshold": 30 } },
                 "filters": [] },
      "exit":  { "triggers": [ { "name": "rsi_overbought", "params": {} } ] } }
  ],
  "risk": {
    "stop_loss":   { "type": "percent", "value": 2 },
    "take_profit": { "type": "rr", "ratio": 2 }
  },
  "provenance": {
    "field_sources": {
      "universe.symbol": "user",
      "legs.0.entry.trigger": "user",
      "risk.stop_loss": "user",
      "risk.take_profit": "user",
      "legs.0.exit": "inferred"
    },
    "unmapped_details": [],
    "clarifications_needed": []
  },
  "version": 1
}
```
Notes: "ETH" → normalized to `ETH_USDC`; "RSI drops below 30" → `rsi_oversold` card (the AI
understands "drops below" = below); the exit was **inferred** (mirror of the entry).

### Step 2 — Validator
- Referential: `rsi_oversold`, `rsi_overbought` exist ✅; `ETH_USDC` resolves in the KB ✅.
- Parameter: window 14, threshold 30 in range ✅.
- Satisfiability: `RSI(14) < 30` can fire ✅ (not a 0-trade rule).
- Safety: leg has an exit + a stop ✅.
- Feasibility: 15m history covers RSI(14) warmup ✅.
- Engine capability: nothing exotic ✅.
→ **ok = true.**

### Step 3 — Compiler → Artifact (the existing engine format)
```json
{
  "artifact_id": "a1b2…", "version": 1,
  "symbol": "ETH_USDC", "timeframe": "15m",
  "entry_condition": "RSI(14) < 30",
  "exit_condition":  "RSI(14) > 70",
  "stop_loss_pct": 2.0, "take_profit_pct": 4.0
}
```
(The conditions are **strings**; the engine compiles them at run time, exactly as today.)

### Step 4 — Read-back + match%
> **Built:** Long entry when RSI(14) < 30 · Stop loss 2% · Take profit 2:1.
> **Assumed:** Exit on RSI(14) > 70 (mirror of your entry) — OK?
> **Couldn't do:** nothing.
> **Captured: 4 of 4 requirements (100%).**

User confirms.

### Step 5 — Backtest
The existing engine runs it on ETH_USDC 15m history → returns trades + in-sample /
out-of-sample / per-regime metrics, stamped with **version 1**.

---

## 8. Worked Example B — DYNAMIC asset (full end-to-end)

**User types:**
> *"On the 15-minute chart, trade the NSE stock with the highest relative volume that is
> also above its VWAP. Go long on a breakout above the opening-range high. ATR stop."*

The user did **not** name a stock — they gave a **rule to pick one**. That's a dynamic
universe.

### Step 1 — Selector fills the SDL
```json
{
  "context": { "market": "indian_stocks", "timeframe": "15m", "objective": "breakout" },
  "universe": {
    "type": "dynamic",
    "asset_class": "equity_cash",
    "screen": [ "CLOSE > VWAP" ],
    "rank":  { "by": "rvol", "order": "desc" },
    "tie_break": "highest_rvol"
  },
  "legs": [
    { "direction": "long",
      "entry": { "trigger": { "name": "opening_range_breakout", "params": { "minutes": 15 } },
                 "filters": [] },
      "exit":  { "triggers": [] } }
  ],
  "risk": { "stop_loss": { "type": "atr", "multiple": 1.5, "window": 14 },
            "take_profit": { "type": "rr", "ratio": 2 } },
  "provenance": {
    "field_sources": {
      "universe": "user",
      "legs.0.entry.trigger": "user",
      "risk.stop_loss": "user",
      "risk.take_profit": "default"
    },
    "unmapped_details": [],
    "clarifications_needed": [
      { "field": "risk.take_profit", "question": "No target given — I used 2:1. OK?", "assumed_value": "2:1" }
    ]
  },
  "version": 1
}
```
Notes: "highest relative volume" → `rank.by: rvol`; "above its VWAP" → a screen condition
(`CLOSE > VWAP`); both are things the discovery scanner already understands. The target was
**defaulted** (user didn't give one) → a clarification was added.

### Step 2 — Validator
- Referential: `opening_range_breakout` exists ✅; screen condition `CLOSE > VWAP` compiles ✅;
  `rank.by: rvol` is a scanner-supported metric ✅; `asset_class: equity_cash` valid ✅.
- Satisfiability, safety, feasibility ✅.
- Engine capability: "pick ONE stock by rule" → supported by the scanner ✅.
  *(If the user had said "trade the top 20", that part would be flagged
  `engine_capability_gap` — the scanner picks one, not twenty.)*
→ **ok = true.**

### Step 3 — Compiler → Artifact
```json
{
  "artifact_id": "c3d4…", "version": 1,
  "universe": { "type": "dynamic", "asset_class": "equity_cash",
                "screen": ["CLOSE > VWAP"], "rank": {"by":"rvol","order":"desc"},
                "tie_break": "highest_rvol" },
  "timeframe": "15m",
  "entry_condition": "CLOSE > OPENING_RANGE_HIGH(15)",
  "exit_condition":  "",
  "stop_loss": {"type":"atr","multiple":1.5,"window":14},
  "take_profit_pct": null, "risk_reward": 2
}
```

### Step 4 — Read-back + match%
> **Built:** Universe = NSE stock with highest relative volume **and** above VWAP · Long on
> opening-range(15) breakout · ATR(14) × 1.5 stop.
> **Assumed:** Take profit 2:1 (you didn't specify) — OK?
> **Couldn't do:** nothing.
> **Captured: 4 of 4 requirements (100%).**

### Step 5 — Evaluation (asset is picked first, THEN backtested)
```
1) discovery scanner runs over equity_cash universe:
     • keep stocks where CLOSE > VWAP   (the screen)
     • rank survivors by relative volume (desc)
     • tie-break → pick the single top stock   e.g. "TATASTEEL.NS"
2) backtest the strategy on TATASTEEL.NS 15m history
3) return trades + in-sample / OOS / per-regime metrics, version-stamped
```

**Honest note:** today the scanner picks **one** asset (live/selection style). "Trade the
top 20 in parallel" and "point-in-time historical universe backtest" are **not** built — if
asked, they appear in `unmapped_details` as `engine_capability_gap`, never faked.

---

## 9. The edit / modify loop

A strategy is editable any time — before assembly, after assembly, after backtest — using
the **same** loop:

```
Edit
 ├─ structured edit (tap a field, set a value)         → no AI
 └─ chat edit ("change the stop to 1%", "add a short") → ONE modify-turn (AI)
        ▼
   merge into the current SDL → changed fields' source = "user"
        ▼
   version++ (parent_version set) → re-validate → re-compile
        ▼
   mark any prior backtest STALE  (UI shows "results out of date — re-run?")
        ▼
   read-back again
```

This is why "the AI runs once" really means **once per turn** — a chat edit is a new turn.

---

## 10. What's supported vs flagged (honest scope)

| Capability | Status |
|---|---|
| Single named asset (equity or crypto) | ✅ supported |
| Dynamic asset = pick ONE by rule (equity_cash + crypto_spot) | ✅ via discovery scanner |
| Long, short, **or** both (two legs) | ✅ supported |
| ATR / percent / structural stops, RR / ATR targets, trailing, scale-outs | ✅ in SDL (engine-capability checked) |
| Gates: regime, volume, event, session, relative-strength | ✅ in SDL |
| Multi-timeframe (HTF) confirmation | ✅ in SDL (capability checked) |
| **Multiple assets at once (static)** | ❌ out of scope (single only) |
| **Dynamic "top-N in parallel"** | ⚠️ flagged `engine_capability_gap` |
| **Point-in-time historical dynamic backtest** | ⚠️ flagged `engine_capability_gap` |
| **Novel conditions with no catalog match** (e.g. "BB width lowest of 20") | ⚠️ flagged `unmapped_details` |

The rule everywhere: **if we can't do it, we say so — we never fake it.**

---

## 11. Build phases (order of work)

Each phase = one PR; the test suite runs after each; pass/fail counts reported before/after.

0. **SDL ticket** — the typed form (`app/planner/sdl.py`).
1. **Catalog menu** — the machine-readable menu of signals/options.
2. **Selector** — AI fills the form (create + modify), cached, structured output.
3. **Validator** — referential, parameter, satisfiability, safety, feasibility, engine-capability.
4. **Compiler** — form → the existing engine's strategy config (immutable artifact).
5. **Read-back + match% + versioning** — the user's view + the score + the edit loop.
6. **Evaluation** — wire the artifact into the existing backtester (+ discovery for dynamic).
7. **Offline judge + shadow** — measure faithfulness, compare to the old regex flow, then retire regex.

---

## 12. File map

```
app/planner/
  sdl.py               # the SDL ticket schema (form + provenance)
  catalog_schema.py    # builds the "menu" the AI sees
  sdl_selector.py      # AI: prompt → SDL  (create + modify turns)
  sdl_validator.py     # checks the SDL, returns ValidationResult
  compiler.py          # SDL → existing engine artifact
  readback.py          # plain-English summary + match%
  (reused) param_resolver.py, condition_satisfiability.py, fidelity_validator.py

app/services/discovery/  # REUSED: dynamic-asset scanner (scan, rank, tie-break)
app/services/ai/llm.py   # REUSED: the LLM transport
quant_engine/            # REUSED: backtester, indicators, conditions
app/kb/signals/*.yaml    # the catalog (the menu source)
tests/test_planner/      # golden-SDL tests, measurement corpus, offline judge
```

---

## 13. Glossary (quick reference)

- **SDL** — Strategy Definition Language: the typed form describing a strategy.
- **Ticket** — a filled-in SDL instance.
- **Catalog / menu** — the fixed library of signals the AI may choose from.
- **Selector** — the one AI call that fills the form.
- **Provenance** — per-field record of `user | default | inferred`.
- **unmapped_details** — things the user asked for that we can't build (written down, not dropped).
- **clarifications_needed** — defaulted money values we ask the user to confirm.
- **match %** — `built / (built + missed)`, counted per atomic user clause.
- **static universe** — a named asset.
- **dynamic universe** — a rule that the discovery scanner uses to pick an asset.
- **artifact** — the immutable, executable strategy the engine runs.
- **read-back** — the plain-English summary the user confirms before backtest.

---

*This document describes the design. Implementation proceeds phase by phase (Section 11);
each phase ships with tests and reported pass/fail counts.*
