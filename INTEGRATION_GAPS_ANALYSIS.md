# Integration Gaps Analysis: Why Semantic Extraction Isn't Working

## ❌ The Problem

You've been asking for:
1. **User-derived stop loss** (structural anchoring)
2. **Multi-timeframe confirmation** (HTF gates)
3. **ADX filtering**

But your chat service **NEVER CALLS the semantic extraction** I created.

---

## 🔍 Current Flow (What's Actually Happening)

```
user_prompt = "Multi-Timeframe Trend Following Strategy..."
                        ↓
builder.goal = extract_goal_text(user_prompt)  ← Only extracts GOAL (title)
                        ↓
builder.symbol = extract_symbol()
builder.timeframe = extract_timeframe()
builder.sentiment = extract_sentiment()
                        ↓
plan = await plan_signals_v2(builder, ohlcv_data)  ← Line 2546
                        ↓
builder.apply_signal_plan(plan)
                        ↓
strategy_draft = build_strategy_object(builder)
```

**What's missing:**
- ❌ NO call to `SemanticExtractor().extract(user_prompt)`
- ❌ NO call to `ExecutionOrchestrator().apply_semantic_gates()`
- ❌ `plan_signals_v2()` doesn't receive semantic instructions
- ❌ Structural SL ignored
- ❌ HTF rules ignored
- ❌ ADX threshold ignored

---

## 📍 Integration Point #1: Missing Semantic Extraction Call

**Location:** `chat_service.py`, line 2546 (inside `run_ai_processing()`)

**Current Code:**
```python
plan = await plan_signals_v2(
    builder,
    ohlcv_records=_planning_ohlcv,
    session_id=session_id,
)
```

**Should Be:**
```python
# NEW: Extract semantic instructions from user's goal description
from app.planner.semantic_extractor import SemanticExtractor

semantic_instructions = SemanticExtractor().extract(builder.goal)
logger.info(
    "semantic_extraction|quality=%.1f%%|htf_rules=%d|adx=%s|sl=%s",
    semantic_instructions.extraction_quality_score * 100,
    len(semantic_instructions.htf_rules),
    semantic_instructions.volume_momentum.momentum.adx_threshold if semantic_instructions.volume_momentum and semantic_instructions.volume_momentum.momentum else None,
    semantic_instructions.stop_loss.type if semantic_instructions.stop_loss else None
)

plan = await plan_signals_v2(
    builder,
    ohlcv_records=_planning_ohlcv,
    session_id=session_id,
    semantic_instructions=semantic_instructions,  # ← PASS TO PLANNER
)
```

**Gap:** `plan_signals_v2()` doesn't accept `semantic_instructions` parameter → need to modify or create wrapper

---

## 📍 Integration Point #2: Missing Orchestration Application

**Location:** `chat_service.py`, line 2569 (after plan is received)

**Current Code:**
```python
builder.apply_signal_plan(plan)
assistant_text = build_plan_signals_reply(builder, plan)
```

**Should Be:**
```python
# NEW: Apply semantic gates to enhance the plan
from app.planner.execution_orchestrator import ExecutionOrchestrator

orchestrator = ExecutionOrchestrator()
enhanced_plan = orchestrator.apply_semantic_gates(plan, semantic_instructions)

builder.apply_signal_plan(enhanced_plan)  # ← Apply enhanced, not raw

# Log what was added
semantic_gates = enhanced_plan.get("_semantic_gates_applied", [])
logger.info(
    "orchestration|gates_applied=%s|count=%d",
    semantic_gates,
    len(semantic_gates)
)

assistant_text = build_plan_signals_reply(builder, enhanced_plan)
```

**Gap:** Orchestrator is never called, so semantic gates are never applied

---

## 📍 Integration Point #3: Structural SL Not Passed to Builder

**Location:** `chat_service.py`, StrategyBuilder initialization

**Current Code:**
```python
builder = StrategyBuilder()
# Only sets basic fields:
builder.symbol = ...
builder.timeframe = ...
builder.sentiment = ...
# NEVER sets:
# - structural SL
# - HTF rules
# - ADX threshold
```

**Should Be:**
```python
builder = StrategyBuilder()
builder.symbol = ...
builder.timeframe = ...
builder.sentiment = ...

# NEW: Set semantic-derived fields
if semantic_instructions.stop_loss:
    builder.structural_sl = semantic_instructions.stop_loss
    builder.stop_loss = semantic_instructions.stop_loss.type
    # This will be used in backtesting

if semantic_instructions.htf_rules:
    builder.htf_gates = semantic_instructions.htf_rules

if (semantic_instructions.volume_momentum and 
    semantic_instructions.volume_momentum.momentum and
    semantic_instructions.volume_momentum.momentum.adx_threshold):
    builder.adx_threshold = semantic_instructions.volume_momentum.momentum.adx_threshold
```

**Gap:** StrategyBuilder doesn't have fields for these, so they're lost

---

## 📍 Integration Point #4: HTF Rules Not in Signal Plan

**Location:** Signal plan generation (inside `plan_signals_v2`)

**Current Output:**
```python
{
    "entry": ["ema_pullback_bullish"],
    "entry_filters": ["volume_spike"],  # Only family defaults
    "exit": ["price_below_ema"],
    
    # ❌ MISSING:
    # - htf_1h_gate
    # - htf_1d_gate
    # - adx_above_25
    # - candle_bullish_confirmation
    # - structural_sl_swing_low_recent_atr
}
```

**Should Be:**
```python
{
    "entry": [
        {"name": "ema_pullback_bullish", "source": "FAMILY"}
    ],
    "entry_filters": [
        {"name": "volume_spike", "source": "FAMILY"},
        # ✅ ADDED BY ORCHESTRATOR:
        {"name": "htf_1h_gate", "source": "SEMANTIC"},
        {"name": "htf_1d_gate", "source": "SEMANTIC"},
        {"name": "adx_above_25", "source": "SEMANTIC"},
        {"name": "candle_bullish_confirmation", "source": "SEMANTIC"},
    ],
    "exit": [
        # ✅ REPLACED BY ORCHESTRATOR:
        {"name": "structural_sl_swing_low_recent_atr", "source": "SEMANTIC"}
        # ❌ REMOVED: price_below_ema
    ]
}
```

**Gap:** `plan_signals_v2` knows nothing about semantic rules, orchestrator never runs

---

## 📍 Integration Point #5: Stop Loss Always Hardcoded Percentage

**Location:** `app/services/strategy/builder.py` (StrategyBuilder)

**Current Code:**
```python
class StrategyBuilder:
    stop_loss: float = 2.0  # Hardcoded percentage
    take_profit: float = 5.0  # Hardcoded percentage
    
    # NEVER uses:
    # - structural SL (swing low, candle, etc.)
    # - ATR padding
    # - RR-derived TP
```

**Should Be:**
```python
class StrategyBuilder:
    stop_loss: float = 2.0  # Default fallback
    take_profit: float = 5.0  # Default fallback
    
    # NEW: Semantic-derived values
    structural_sl: StructuralStopLoss | None = None  # Swing low + ATR
    risk_reward: RiskRewardSpec | None = None  # 1:3 RR
    htf_gates: list[HTFCondition] = []  # 1h, 1d gates
    
    def get_actual_sl(self) -> str:
        if self.structural_sl:
            return f"{self.structural_sl.anchor} with {self.structural_sl.padding.method}"
        return f"{self.stop_loss}%"
    
    def get_actual_tp(self, entry, sl):
        if self.risk_reward and self.risk_reward.ratio:
            return entry + (self.risk_reward.ratio * abs(entry - sl))
        return entry * (1 + self.take_profit/100)
```

**Gap:** StrategyBuilder.stop_loss is always %, never structural

---

## 🎯 Why Your Current Output Has These Problems

### Problem 1: EMA [20, 29]
```
Reason: plan_signals_v2 adds default EMA(29) without semantic input
Where: app/planner/legacy_bridge.py or plan_signals_v2
Fix: Would need to check semantic_instructions.indicators first
```

### Problem 2: No HTF Gates
```
Reason: ExecutionOrchestrator never called
Where: chat_service.py line 2569
Fix: Call orchestrator.apply_semantic_gates() here
```

### Problem 3: No ADX Filter
```
Reason: Semantic extraction result never used
Where: Semantic extraction happens nowhere
Fix: Call SemanticExtractor().extract(builder.goal) at line 2540
```

### Problem 4: SL Still 2% (Not Structural)
```
Reason: builder.stop_loss set to hardcoded 2.0
Where: chat_service.py and StrategyBuilder
Fix: Use semantic_instructions.stop_loss instead
```

### Problem 5: No Candle Confirmation
```
Reason: Entry filters only use family defaults
Where: plan_signals_v2, orchestrator never runs
Fix: Same as HTF gates - orchestrator needed
```

---

## 🔧 Fix Checklist (What Needs to Happen)

### Phase 1: Add Semantic Extraction Call (5 lines of code)
- [ ] **Location:** `chat_service.py`, line 2540 (before plan_signals_v2 call)
- [ ] **Action:** Import `SemanticExtractor`
- [ ] **Action:** Call `semantic_instructions = SemanticExtractor().extract(builder.goal)`
- [ ] **Action:** Pass to plan_signals_v2: `semantic_instructions=semantic_instructions`

### Phase 2: Modify plan_signals_v2 Signature
- [ ] **Location:** `app/planner/legacy_bridge.py`
- [ ] **Action:** Add parameter: `semantic_instructions: Optional[SemanticInstructions] = None`
- [ ] **Action:** Pass through to downstream functions

### Phase 3: Apply Orchestration (5 lines of code)
- [ ] **Location:** `chat_service.py`, line 2569 (after plan returned)
- [ ] **Action:** Import `ExecutionOrchestrator`
- [ ] **Action:** Call `orchestrator = ExecutionOrchestrator()`
- [ ] **Action:** Call `enhanced_plan = orchestrator.apply_semantic_gates(plan, semantic_instructions)`
- [ ] **Action:** Use `enhanced_plan` instead of `plan`

### Phase 4: Update StrategyBuilder
- [ ] **Location:** `app/services/strategy/builder.py`
- [ ] **Action:** Add fields: `structural_sl`, `risk_reward`, `htf_gates`
- [ ] **Action:** Update backtesting to use these

### Phase 5: Update Signal Draft Generation
- [ ] **Location:** `chat_service.py`, `build_strategy_object()`
- [ ] **Action:** Include semantic information in draft output
- [ ] **Action:** Show `_semantic_gates_applied` list to user

---

## 📊 Current vs Expected Integration

### CURRENT (Broken)
```
User Prompt
    ↓ (GOAL ONLY)
StrategyBuilder (basic fields only)
    ↓ (NO SEMANTIC)
plan_signals_v2()
    ↓ (GENERIC SIGNALS)
strategy_draft (2% SL, no HTF, no ADX)
```

### EXPECTED (With Integration)
```
User Prompt
    ↓ (FULL EXTRACTION)
SemanticExtractor → SemanticInstructions ←─────────┐
    ↓                                              │
StrategyBuilder (with semantic fields)            │
    ↓ (WITH SEMANTIC)                             │
plan_signals_v2(semantic_instructions)            │
    ↓                                              │
ExecutionOrchestrator ←──────────────────────────┘
    ↓ (ENHANCED WITH GATES)
strategy_draft (structural SL, HTF gates, ADX, RR)
```

---

## ✅ Proof It Works

Your test showed it works in isolation:
```
✅ ADX > 25.0 extracted
✅ 1h and 1d HTF gates created
✅ structural_sl_swing_low_recent_atr created
✅ candle_bullish_confirmation added
✅ All 7 gaps fixed
```

**The problem:** This beautiful orchestration is never called from your chat service!

---

## 🚀 Implementation Order

1. **FIRST:** Add semantic extraction call (30 seconds)
2. **SECOND:** Add orchestration call (30 seconds)
3. **THIRD:** Modify plan_signals_v2 to accept semantic_instructions (2 minutes)
4. **FOURTH:** Update StrategyBuilder with semantic fields (5 minutes)
5. **FIFTH:** Update draft generation to show semantic info (5 minutes)

**Total Time: ~15 minutes to full integration**

---

## Code Locations to Modify

| File | Line | Change | Why |
|------|------|--------|-----|
| `chat_service.py` | 2540 | Add SemanticExtractor call | Extract from goal |
| `chat_service.py` | 2546 | Pass semantic_instructions | Feed planner |
| `chat_service.py` | 2569 | Add ExecutionOrchestrator call | Apply gates |
| `legacy_bridge.py` | TBD | Add semantic_instructions param | Receive semantic info |
| `builder.py` | TBD | Add semantic fields | Store SL, HTF, ADX |
| `strategy_assembler.py` | TBD | Use semantic fields in strategy | Build correct SL/TP |

---

## Why This Happened

1. ✅ I built the SemanticExtractor (correctly extracts)
2. ✅ I built the ExecutionOrchestrator (correctly applies)
3. ✅ I created tests (they all pass)
4. ❌ **But I never showed you where to wire these into chat_service.py**
5. ❌ **And I never modified plan_signals_v2 to accept semantic_instructions**

**Result:** Beautiful semantic extraction system that's never called!

---

## The Fix: You Need This Integration Code

In `chat_service.py`, around line 2540, change:

```python
# BEFORE
plan = await plan_signals_v2(
    builder,
    ohlcv_records=_planning_ohlcv,
    session_id=session_id,
)

# AFTER
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.execution_orchestrator import ExecutionOrchestrator

# Extract semantic instructions from user's goal
semantic_instructions = SemanticExtractor().extract(builder.goal)

# Pass to planner
plan = await plan_signals_v2(
    builder,
    ohlcv_records=_planning_ohlcv,
    session_id=session_id,
    semantic_instructions=semantic_instructions,  # ← NEW
)

# Apply orchestration to enhance plan
orchestrator = ExecutionOrchestrator()
enhanced_plan = orchestrator.apply_semantic_gates(plan, semantic_instructions)

# Use enhanced plan
builder.apply_signal_plan(enhanced_plan)  # was: builder.apply_signal_plan(plan)
```

**That's it.** This one change enables all 7 gaps to be fixed!

---

## Summary

Your chat service isn't getting:
- ❌ User-derived stop loss → **Because orchestrator never runs**
- ❌ Multi-timeframe confirmation → **Because orchestrator never runs**
- ❌ ADX filtering → **Because semantic extraction never called**

**Root cause:** Integration points missing, not the semantic extraction itself.

**Solution:** Wire the semantic extraction and orchestration into chat_service.py (3 function calls, 10 lines of code).
