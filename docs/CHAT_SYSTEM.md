# How the Chat System Works — A Detailed, Plain-English Guide

This document explains, in the simplest possible terms, what happens inside Stretus AI
every time a user types a message in the chat. It walks through **every component**, the
**order things happen in**, and **how the LLM (AI brain) is actually used**.

If you read this top to bottom, you'll understand the whole journey: from a user typing
"I want a breakout strategy on TCS" → to a finished, backtested trading strategy.

---

## 1. The Big Picture (read this first)

Stretus AI is a **conversational trading-strategy builder**. Instead of filling out forms,
the user *talks* to an AI strategist. Through the conversation, the system gradually collects
everything it needs (which stock, which timeframe, what the user wants), then:

1. **Plans** which technical trading signals to use (e.g. RSI oversold, Bollinger squeeze).
2. **Assembles** a complete strategy with entry/exit rules and risk management.
3. **Backtests** it against historical market data to see if it would have made money.
4. **Explains** the results back in plain trader language.

The whole thing is a **back-and-forth dialogue**. The user can change their mind, ask
questions, or start over at any time — and the system keeps up.

### The mental model: a stateful conversation driven by an AI "router"

Think of it like a smart receptionist (the **Agent Router**) who:
- Reads every message the user sends.
- Looks at where the conversation currently is ("are we still collecting inputs? did we
  already build a strategy?").
- Decides **what to do next** (collect more info, plan signals, run a backtest, answer a
  question, etc.).
- Hands the work off to the right specialist department.

The "deciding what to do next" part is powered by the **LLM** (a large language model,
accessed through OpenRouter or a local Ollama model).

---

## 2. The Key Players (components at a glance)

| # | Component | File | What it does (one line) |
|---|-----------|------|--------------------------|
| 1 | **Chat API** | [app/api/v1/routes/chat.py](../app/api/v1/routes/chat.py) | The front door. Receives messages, returns answers. |
| 2 | **Orchestrator** | [app/services/chat/chat_service.py](../app/services/chat/chat_service.py) | The conductor. Runs the whole turn from start to finish. |
| 3 | **Agent Router** | [app/services/agent/router.py](../app/services/agent/router.py) | The decision-maker. Picks what action to take. |
| 4 | **LLM Service** | [app/services/ai/llm.py](../app/services/ai/llm.py) | The AI brain connector. Talks to OpenRouter/Ollama. |
| 5 | **Input Interpreter** | [app/services/ai/user_input_interpreter.py](../app/services/ai/user_input_interpreter.py) | The translator. Turns messy human text into clean data. |
| 6 | **Persona Responder** | [app/services/chat/persona_responder.py](../app/services/chat/persona_responder.py) | The writer. Produces nice, on-brand replies. |
| 7 | **SDL Selector** | [app/planner/sdl_selector.py](../app/planner/sdl_selector.py) | The planner. Turns ideas into a structured strategy spec. |
| 8 | **Signal Knowledge Base** | [app/kb/signals/](../app/kb/signals/) | The library of 125 trading signals the AI can pick from. |
| 9 | **YAML Generator** | [app/services/strategy/yaml_generator.py](../app/services/strategy/yaml_generator.py) | The packager. Writes the final strategy to a file. |
| 10 | **Backtest API + Engine** | [app/api/v1/routes/backtest.py](../app/api/v1/routes/backtest.py) | The tester. Runs the strategy on historical data. |
| 11 | **App Bootstrap** | [app/main.py](../app/main.py) | The wiring. Starts everything up. |

---

## 3. What Happens When a User Sends a Message (step by step)

This is the heart of the document. Let's follow a single message through the whole system.

### Step 0 — The user sends a message

The frontend ([chat_mvp.html](../chat_mvp.html)) makes an HTTP call:

```
POST /api/v1/strategy/chats/{session_id}/messages
Body: { "content": "I want a breakout strategy on TCS, 15 minute chart" }
```

### Step 1 — The front door receives it ([chat.py](../app/api/v1/routes/chat.py))

The function [`send_message()`](../app/api/v1/routes/chat.py#L405) does **four quick things**
and then gets out of the way:

1. **Checks the chat session exists** ([chat.py:421](../app/api/v1/routes/chat.py#L421)).
   If not, returns a `404`.
2. **Saves the user's message to the database immediately**
   ([chat.py:431](../app/api/v1/routes/chat.py#L431)) via `accept_message()`.
3. **Returns a `202 ACCEPTED` response right away**
   ([chat.py:443](../app/api/v1/routes/chat.py#L443)) — it does **not** wait for the AI.
4. **Fires a background task** ([chat.py:436](../app/api/v1/routes/chat.py#L436)):
   `run_ai_processing(session_id, message_id, content)`.

> **Why return immediately?** AI thinking + backtesting can take many seconds. If the API
> waited, the user's browser request would time out. Instead, the server says "got it,
> working on it" and processes in the background.

### Step 2 — The client polls for the answer

Because the answer isn't ready yet, the frontend repeatedly calls:

```
GET /api/v1/strategy/chats/{session_id}/messages
```

handled by [`get_messages()`](../app/api/v1/routes/chat.py#L459). Each message carries a
`status`: `processing`, `completed`, or `failed`. When the assistant's reply flips to
`completed`, the UI shows it. This is the **"poll-and-respond"** pattern.

### Step 3 — The conductor takes over ([chat_service.py](../app/services/chat/chat_service.py))

Meanwhile, in the background, [`run_ai_processing()`](../app/services/chat/chat_service.py#L2018)
runs the entire "turn." Here's what it does, in order:

#### 3a. Track the running task
It registers itself in `_active_generations`
([chat_service.py:2030](../app/services/chat/chat_service.py#L2030)) so the turn can be
**cancelled** if the user hits "pause/stop." (See `POST /chats/{id}/pause`.)

#### 3b. Load the conversation history & rebuild memory
The system is **stateless between turns** — it stores everything in the database and rebuilds
its working memory each time. It:
- Loads up to **80 previous messages** for context.
- Rebuilds the **`StrategyBuilder`** (the in-memory object holding the half-built strategy)
  from the last saved "draft" ([chat_service.py:2072](../app/services/chat/chat_service.py#L2072)).
- Recovers `previous_state` — where the conversation left off
  ([chat_service.py:2070](../app/services/chat/chat_service.py#L2070)). Possible states
  include `collect_user_input`, `plan_signals`, `assemble_strategy`, `backtest_confirmation`.

> **Key idea:** the `StrategyBuilder` is *ephemeral*. It's reconstructed fresh every turn
> from the database draft. The database is the source of truth, not server memory.

#### 3c. Hydrate risk & execution config
It loads the saved risk settings (stop-loss style, position sizing, etc.) so the strategy
stays consistent across turns.

#### 3d. Quick pre-scans of the message
Before involving the AI, cheap text checks run:
- Detect **strategy preset** keywords ("breakout", "mean reversion", …).
- Extract any **explicit stock lists** the user typed.
- Extract **parameter overrides** (e.g. "volume spike > 200%", "within 5% of 52-week high").

#### 3e. **Ask the AI what to do** → the Agent Router

This is the central decision point. The orchestrator calls
[`AgentRouter.decide()`](../app/services/agent/router.py#L164) (around
[chat_service.py:2190](../app/services/chat/chat_service.py#L2190)).

---

## 4. The Agent Router — The Decision-Maker ([router.py](../app/services/agent/router.py))

The router's job: **given the conversation so far + the new message, decide the single next
action.** It does this with the help of the LLM and "tool calling."

### How it works:

1. **Builds a compact state summary** ([router.py:176](../app/services/agent/router.py#L176)).
   It packs the important facts (current state, what's been collected, last backtest result)
   into a small JSON blob so the AI doesn't need the entire history.

2. **Builds a system prompt** ([router.py:184](../app/services/agent/router.py#L184)).
   This is the AI's "job description" — it explains the agent's purpose and the rules for
   choosing actions.

3. **Calls the LLM with a menu of "tools"** ([router.py:205](../app/services/agent/router.py#L205))
   via `self._llm.chat_with_tools(...)`. The "tools" are the allowed actions, such as:
   - `START_NEW_STRATEGY` — begin a fresh strategy
   - `MODIFY_STRATEGY_INPUTS` — change a value the user already gave
   - `PLAN_STRATEGY_SIGNALS` — pick the technical signals
   - `ASSEMBLE_STRATEGY` — build the full strategy object
   - `RUN_BACKTEST` — test it on historical data
   - `ASK_USER_FOR_CLARIFICATION` — ask a follow-up question
   - `RESPOND_TEXT` — just chat / answer a question

4. **The LLM responds by "calling a tool."** Instead of free text, the AI replies with a
   structured choice like `{"tool": "modify_strategy_inputs", "arguments": {...}}`.

5. **The router parses that into an `AgentDecision`**
   ([router.py:206](../app/services/agent/router.py#L206)) — a clean object holding
   `tool_name`, `parameters`, optional `assistant_text`, and a `source` tag.

6. **Fallback safety net** ([router.py:224](../app/services/agent/router.py#L224)): if the
   LLM fails or returns garbage, it falls back to an older rule-based router so the
   conversation never dies.

The `AgentDecision` is then converted into a "legacy route" — an `intent` plus extracted
fields ([router.py:32-155](../app/services/agent/router.py#L32-L155)) — that the rest of
the orchestrator already knows how to handle.

> **Tool calling in plain terms:** rather than letting the AI ramble in prose, we give it a
> fixed list of buttons it's allowed to press. It presses exactly one and fills in the
> required details. This makes the AI's output **predictable and machine-readable.**

---

## 5. The LLM Service — The AI Brain Connector ([llm.py](../app/services/ai/llm.py))

This is the layer that actually talks to a real AI model. Everything above just *asks* it
to; `LLMService` does the talking.

### What model and provider?

It supports three modes (configured by environment variables):
- **`openrouter`** — calls a cloud model through [OpenRouter.ai](https://openrouter.ai)
  (a router that gives access to many models via one API).
- **`ollama`** — calls a model running **locally** on the machine (private, no internet,
  no API key needed, e.g. `qwen2.5:7b`).
- **`auto`** — try OpenRouter first; if it's down/rate-limited, fall back to Ollama
  ([llm.py:700](../app/services/ai/llm.py#L700)).

The model name comes from env vars like `OPENROUTER_MODEL` / `OLLAMA_MODEL`.

### The two main ways code talks to the AI:

1. **`chat(messages)`** ([llm.py:136](../app/services/ai/llm.py#L136)) — send a
   conversation, get back **plain text**. Used for writing replies and interpreting input.

2. **`chat_with_tools(messages, tools)`** ([llm.py:161](../app/services/ai/llm.py#L161)) —
   send a conversation **plus a list of allowed tools**, get back the AI's **tool choice**.
   Used by the Agent Router and the SDL Selector.

### API key management & rotation (important reliability feature)

OpenRouter calls can fail because a key hits its **rate limit (429)**, runs out of
**credits (402)**, or is **unauthorized (401)**. To stay alive, the service keeps a **pool
of API keys** and **rotates to the next key** automatically when one fails
([llm.py:345-396](../app/services/ai/llm.py#L345-L396)). The current rotation position is
tracked in [openrouter_key_state.json](../app/services/ai/openrouter_key_state.json). Only
when *all* keys are exhausted does it raise an error.

Every call also records token usage via `track_tokens()`
([llm.py:277](../app/services/ai/llm.py#L277)) for cost/usage analytics.

---

## 6. The Input Interpreter — The Translator ([user_input_interpreter.py](../app/services/ai/user_input_interpreter.py))

Humans type messy things: "yeah TCS sounds good, intraday, I'm new to this." The interpreter
uses the LLM to **turn that mess into clean structured data.**

[`route_user_message()`](../app/services/ai/user_input_interpreter.py#L1) sends the message
with a detailed system prompt ([lines 528-669](../app/services/ai/user_input_interpreter.py#L528-L669))
that teaches the AI to output JSON like:

```json
{
  "intent": "collect_input",
  "confidence": "high",
  "stock_query": "TCS",
  "timeframe_input": "intraday",
  "experience": "beginner",
  "sentiment": null,
  "needs_clarification": false,
  "clarification_topic": null
}
```

It recognizes **11 intents** (collect_input, general_chat, clarification, confirmation,
run_backtest, modify_input, new_strategy, user_rejection, pause_workflow, stock_advice_request,
invalid_value) and maps casual phrasing to canonical values (e.g. "I'm new" → `beginner`,
"day trading" → `intraday`).

It also has **safety rails**: e.g. if the user asks "should I buy this stock?" it flags
`stock_advice_request` so the system can give a **SEBI-compliant** non-advisory response
(it builds strategies, it doesn't give investment advice). And if confidence is low with no
clear fields, it forces an "ambiguous → ask for clarification" path
([lines 748-753](../app/services/ai/user_input_interpreter.py#L748-L753)).

> **Why two AI passes?** The interpreter extracts *what the user said*; the agent router
> decides *what to do about it*. Separating these keeps each prompt simple and reliable.

---

## 7. Building the Strategy — From Idea to Spec

Once enough inputs are collected and the user confirms, the workflow moves through phases.

### 7a. Planning signals — the SDL Selector ([sdl_selector.py](../app/planner/sdl_selector.py))

**SDL = Strategy Definition Language** — a structured, machine-readable description of a
complete trading strategy (which signals, entry/exit conditions, stop-loss, take-profit,
position sizing, etc.).

[`compile_to_sdl(prompt)`](../app/planner/sdl_selector.py#L636) uses the LLM (with tool
calling, the `submit_sdl` tool) to convert the natural-language strategy idea into a strict
**SDL ticket**. Its system prompt ([lines 149-273](../app/planner/sdl_selector.py#L149-L273))
contains:
- The **catalog of available signals** (so the AI only picks ones that exist).
- The **allowed enum values** (e.g. direction = long/short).
- **Guardrails** — every requested behavior must map to a real SDL field; anything
  unsupported is recorded rather than silently dropped; defaults trigger a clarification
  rather than a silent guess.

There's also [`modify_sdl(existing, change)`](../app/planner/sdl_selector.py#L695) for
iterative edits ("make the stop-loss tighter"), which **versions** the SDL for an audit
trail. The LLM call retries up to 3 times if the JSON doesn't parse cleanly
([lines 305-330](../app/planner/sdl_selector.py#L305-L330)).

### 7b. The Signal Knowledge Base ([app/kb/signals/](../app/kb/signals/))

This folder holds **125 YAML files**, one per trading signal — e.g.
[bb_squeeze.yaml](../app/kb/signals/bb_squeeze.yaml),
[vwap_reclaim_bullish.yaml](../app/kb/signals/vwap_reclaim_bullish.yaml),
`rsi_oversold.yaml`, `ema_cross_up.yaml`. Each describes what the signal means and its
parameters. This is the **menu of building blocks** the AI is allowed to choose from when
planning a strategy. It's loaded into memory at startup
([main.py:82](../app/main.py#L82)).

### 7c. Packaging — the YAML Generator ([yaml_generator.py](../app/services/strategy/yaml_generator.py))

[`generate_yaml(builder)`](../app/services/strategy/yaml_generator.py#L1) takes the finished
`StrategyBuilder` and writes it out as a `.yaml` file on disk (default folder
`./data/strategies`, filename like `tcs_breakout_15m.yaml`). This YAML is the **portable
contract** the backtest engine reads.

---

## 8. Writing the Reply — The Persona Responder ([persona_responder.py](../app/services/chat/persona_responder.py))

When a milestone is reached, the system needs to *tell the user about it* in a polished way.
[`compose_milestone_response(event, context, llm)`](../app/services/chat/persona_responder.py#L295)
handles events like:

- `signal_plan_ready` → "## Signal Plan — bias, chosen signals, confirm?"
- `strategy_assembled` → "## Strategy Built" with an entry/exit/risk table.
- `backtest_complete` → metrics table + plain-English interpretation.
- `backtest_failed` → error explanation + suggestion.

It uses a **persona system prompt** ([lines 31-127](../app/services/chat/persona_responder.py#L31-L127))
defining "Stretus" — a direct, confident, no-filler AI trading strategist that always
replies in markdown and adjusts tone to the user's experience level.

**Crucially, it always has a fallback:** if the LLM is unavailable, it returns a hand-written
template ([line 297](../app/services/chat/persona_responder.py#L297)), so the user always
gets a valid response.

### Saving the reply
Back in the orchestrator, the assistant's message is saved to the database with
`status=completed`, along with the updated strategy draft (`to_draft_json()`) so the **next
turn can rebuild state.** The polling client (Step 2) then picks it up and displays it.

---

## 9. Backtesting — Does the Strategy Actually Work? ([backtest.py](../app/api/v1/routes/backtest.py))

When the user confirms they want to test the strategy, a backtest is triggered.

### Trigger
[`trigger_backtest()`](../app/api/v1/routes/backtest.py#L196):
1. Checks the strategy exists and is confirmed.
2. Creates a `Backtest` DB record with `status=running`.
3. Fires a background task `_call_quant_engine(...)`.
4. Returns `202 ACCEPTED` with a `backtest_id` (same async pattern as chat).

### Background work — [`_call_quant_engine()`](../app/api/v1/routes/backtest.py#L75)
1. Reads the strategy's YAML to learn what market data it needs.
2. Decides on **intrabar execution** — for non-1-minute strategies it also fetches 1-minute
   bars so fills, stop-losses and take-profits are simulated precisely
   ([lines 95-98](../app/api/v1/routes/backtest.py#L95-L98)).
3. **Fetches OHLCV** (Open/High/Low/Close/Volume) candle data — the main symbol plus any
   auxiliary data (a reference index, higher timeframes). This data is cached on disk (see
   the `quant_engine/cache/ohlcv_fetch/*.pkl.gz` files).
4. **Hands the job to the quant engine** (in [quant_engine/](../quant_engine/)) via
   `queue_quant_backtest(...)`, which runs the actual trade simulation.
5. If anything fails, marks the backtest `failed` with the reason.

### Getting results
The client polls [`GET /backtest/{backtest_id}`](../app/api/v1/routes/backtest.py#L258),
which returns `status` (running/complete/failed) and `result_json` (returns %, win rate,
number of trades, etc.). Those metrics then flow back into the chat via the persona
responder's `backtest_complete` message.

---

## 10. How It's All Wired Together ([main.py](../app/main.py))

On startup, [main.py](../app/main.py):
1. Creates needed folders (logs, data, strategies).
2. **Initializes the LLM service** and logs which provider/model is active
   ([lines 66-79](../app/main.py#L66-L79)).
3. **Loads the Knowledge Base** — signals, stocks, timeframes
   ([lines 82-84](../app/main.py#L82-L84)).
4. Seeds the default risk/execution config in the database.
5. **Registers all the API routes** under `/api/v1/strategy`
   ([lines 140-143](../app/main.py#L140-L143)) — chat, strategy, backtest, execution.

It also exposes health endpoints: `GET /health`, `/health/kb`, `/health/llm`,
`/health/llm/models` — handy for checking the AI provider and KB are loaded.

---

## 11. The Complete Journey (one diagram)

```
USER types: "breakout strategy on TCS, 15m"
        │
        ▼
POST /chats/{id}/messages  ──►  save to DB  ──►  return 202 instantly
        │                                              │
        │                                              └─► fire background task
        ▼
   run_ai_processing()  (the conductor)
        │
        ├─ load last 80 messages + rebuild StrategyBuilder from DB draft
        ├─ recover previous_state (collect_user_input / plan_signals / ...)
        ├─ quick text pre-scans (presets, stock lists, param overrides)
        │
        ├─ AgentRouter.decide()  ──►  LLMService.chat_with_tools()
        │        │                          │
        │        │                          └─► OpenRouter (key rotation) or Ollama
        │        │
        │        └─ AI picks ONE tool: collect / plan / assemble / backtest / ask / respond
        │
        ├─ (interpret message) user_input_interpreter → clean JSON {intent, fields...}
        │
        ├─ dispatch by intent:
        │     collect_input   → validate & store inputs in builder
        │     plan_signals    → SDL Selector (LLM + tool call) → pick from 125 KB signals
        │     assemble        → build full strategy + risk config
        │     run_backtest    → generate YAML → trigger backtest
        │     clarification   → ask a follow-up question
        │     general_chat    → just answer
        │
        ├─ persona_responder → write a polished markdown reply (with template fallback)
        │
        └─ save assistant message to DB (status=completed) + updated draft
                  │
                  ▼
USER's browser was polling  GET /chats/{id}/messages
        │
        └─► sees status=completed  ──►  renders the AI reply

IF BACKTEST:
POST /backtest → 202 + backtest_id → background: fetch OHLCV → quant_engine simulates
        │
        └─ client polls GET /backtest/{id} → metrics → shown in chat
```

---

## 12. Key Design Ideas to Remember

1. **Async + polling.** The API never makes the user wait. It accepts the message, returns
   `202`, works in the background, and the client polls for the result. Same for backtests.

2. **Database is the memory.** Nothing important lives only in server RAM. Each turn rebuilds
   the `StrategyBuilder` from the saved draft, so the system is resilient and restartable.

3. **The LLM is used in three distinct roles:**
   - **Interpret** the user's message into clean data (Input Interpreter).
   - **Decide** the next action via tool calling (Agent Router, SDL Selector).
   - **Write** the human-facing reply (Persona Responder).

4. **Tool calling makes the AI reliable.** By forcing the AI to pick from a fixed menu of
   tools (and fill in arguments), its output is structured and predictable instead of
   free-form prose.

5. **Everything has a fallback.** Multiple API keys (rotation), OpenRouter→Ollama fallback,
   LLM→legacy router fallback, LLM→template fallback. The conversation never hard-fails.

6. **Grounded in a knowledge base.** The AI can only build strategies from the **125 real
   signals** in [app/kb/signals/](../app/kb/signals/) — it can't invent indicators that the
   backtest engine can't run.

7. **Compliance-aware.** Stock-advice requests are intercepted and answered in a
   SEBI-compliant, non-advisory way — the product builds and tests strategies, it does not
   tell users what to buy.

---

## 13. Where to Look in the Code (quick reference)

| To understand... | Start here |
|------------------|-----------|
| The API endpoints | [chat.py: `send_message`](../app/api/v1/routes/chat.py#L405), [`get_messages`](../app/api/v1/routes/chat.py#L459) |
| The whole turn orchestration | [chat_service.py: `run_ai_processing`](../app/services/chat/chat_service.py#L2018) |
| How "what to do next" is decided | [router.py: `AgentRouter.decide`](../app/services/agent/router.py#L164) |
| How the AI is actually called | [llm.py: `chat`](../app/services/ai/llm.py#L136), [`chat_with_tools`](../app/services/ai/llm.py#L161) |
| API key rotation | [llm.py: `_openrouter_with_rotation`](../app/services/ai/llm.py#L345) |
| Turning human text into data | [user_input_interpreter.py: `route_user_message`](../app/services/ai/user_input_interpreter.py#L1) |
| Planning the strategy spec | [sdl_selector.py: `compile_to_sdl`](../app/planner/sdl_selector.py#L636) |
| Writing the reply | [persona_responder.py: `compose_milestone_response`](../app/services/chat/persona_responder.py#L295) |
| Running a backtest | [backtest.py: `trigger_backtest`](../app/api/v1/routes/backtest.py#L196) |
| App startup/wiring | [main.py](../app/main.py) |
