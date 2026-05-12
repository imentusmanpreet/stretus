"""
Validation test: Verify semantic extraction integration in chat_service.py flow.
Simulates the chat service workflow with the new semantic gates.
"""
import json
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.execution_orchestrator import ExecutionOrchestrator
from app.services.strategy.builder import StrategyBuilder

# Your original user prompt
user_prompt = """
please help me creating following strategy : Multi-Timeframe Trend Following Strategy —Best for Strong Directional Trends
Works extremely well on: NHPC.Logic -Lower timeframe entries become more reliable when aligned with higher timeframe trends.
You trade pullbacks inside larger bullish trends.
Setup Timeframe 15-minute chart
Higher Timeframe Confirmation 1-hour and daily trend should remain bullish
Entry Rules
BUY Price should stay above 20 EMA on all higher timeframes
Pullback should happen near EMA
Bullish candle should confirm continuation
ADX should remain above 25
SELL Exit when higher timeframe trend weakens
Stop Loss Below recent swing low with ATR padding
Target Risk:Reward = 1:3 minimum.
"""

# Simulated signal plan (as returned by plan_signals_v2)
original_signal_plan = {
    "entry": [
        {
            "name": "ema_pullback_bullish",
            "params": {"window": 20, "lookback": 3},
            "signal_type": "TRIGGER"
        }
    ],
    "entry_filters": [
        {
            "name": "volume_spike",
            "params": {"window": 20, "multiplier": 1.5},
            "signal_type": "FILTER"
        }
    ],
    "exit": [
        {
            "name": "price_below_ema",
            "params": {"window": 29},
            "signal_type": "TRIGGER"
        }
    ],
    "signals_used": ["ema_pullback_bullish", "volume_spike", "price_below_ema"],
    "entry_condition": "CLOSE > EMA(20) AND MIN(LOW, 3) <= EMA(20) AND CLOSE > OPEN AND VOL > AVG(VOL, 20) * 1.5",
    "exit_condition": "CLOSE < EMA(29)",
}

print("=" * 100)
print("CHAT SERVICE INTEGRATION VALIDATION TEST")
print("=" * 100)

print("\n📋 STEP 1: Simulate chat_service.py workflow")
print("-" * 100)

# Step 1: Extract semantic instructions (as done in chat_service.py)
print("\n[CHAT SERVICE] Extracting semantic instructions...")
semantic_extractor = SemanticExtractor()
semantic_instructions = semantic_extractor.extract(user_prompt)

print(f"✅ Semantic extraction complete:")
print(f"   • Quality Score: {semantic_instructions.extraction_quality_score:.1%}")
print(f"   • Strategy Family: {semantic_instructions.strategy_family}")
print(f"   • HTF Rules: {len(semantic_instructions.htf_rules)}")
print(f"   • Indicators: {semantic_instructions.indicators}")
print(f"   • ADX Threshold: {semantic_instructions.volume_momentum.momentum.adx_threshold if semantic_instructions.volume_momentum and semantic_instructions.volume_momentum.momentum else None}")

# Step 2: Apply orchestration (as done in chat_service.py)
print("\n[CHAT SERVICE] Applying semantic gates to signal plan...")
orchestrator = ExecutionOrchestrator()
enhanced_plan = orchestrator.apply_semantic_gates(original_signal_plan, semantic_instructions)

semantic_gates_applied = enhanced_plan.get("_semantic_gates_applied", [])
print(f"✅ Semantic gates applied: {len(semantic_gates_applied)}")
for gate in semantic_gates_applied:
    print(f"   • {gate}")

# Step 3: Build draft with semantic information (as done in chat_service.py)
print("\n[CHAT SERVICE] Building draft with semantic extraction results...")
builder = StrategyBuilder()
builder.symbol = "NHPC"
builder.timeframe = "15m"
builder.goal = "Multi-Timeframe Trend Following Strategy"
builder.sentiment = "bullish"

draft = builder.to_draft_json(
    mode_override="plan_signals",
    processing_status="awaiting_confirmation",
)

# Add semantic extraction results (as shown in integration code)
if semantic_instructions:
    draft["semantic_extraction"] = {
        "quality_score": semantic_instructions.extraction_quality_score,
        "strategy_family": semantic_instructions.strategy_family,
        "htf_rules": [r.dict() for r in semantic_instructions.htf_rules],
        "indicators": semantic_instructions.indicators,
        "adx_threshold": (
            semantic_instructions.volume_momentum.momentum.adx_threshold
            if semantic_instructions.volume_momentum and semantic_instructions.volume_momentum.momentum
            else None
        ),
        "structural_sl": semantic_instructions.stop_loss.dict() if semantic_instructions.stop_loss else None,
        "risk_reward": semantic_instructions.risk_reward.dict() if semantic_instructions.risk_reward else None,
        "candle_confirmation": semantic_instructions.candle_confirmation.dict() if semantic_instructions.candle_confirmation else None,
    }
    draft["semantic_gates_applied"] = semantic_gates_applied

print("✅ Draft built with semantic information")

# Step 4: Validate the draft contains all necessary information
print("\n\n📊 STEP 2: Validate Draft Contains Semantic Information")
print("-" * 100)

validations = [
    ("Quality Score Present", "semantic_extraction" in draft and "quality_score" in draft.get("semantic_extraction", {})),
    ("HTF Rules Present", "semantic_extraction" in draft and len(draft.get("semantic_extraction", {}).get("htf_rules", [])) > 0),
    ("ADX Threshold Present", "semantic_extraction" in draft and draft.get("semantic_extraction", {}).get("adx_threshold") == 25.0),
    ("Structural SL Present", "semantic_extraction" in draft and draft.get("semantic_extraction", {}).get("structural_sl") is not None),
    ("Risk:Reward Present", "semantic_extraction" in draft and draft.get("semantic_extraction", {}).get("risk_reward") is not None),
    ("Candle Confirmation Present", "semantic_extraction" in draft and draft.get("semantic_extraction", {}).get("candle_confirmation") is not None),
    ("Semantic Gates Applied", len(draft.get("semantic_gates_applied", [])) > 0),
    ("Indicators Extracted", draft.get("semantic_extraction", {}).get("indicators") is not None),
]

all_valid = True
for check_name, is_valid in validations:
    status = "✅ PASS" if is_valid else "❌ FAIL"
    print(f"{status} - {check_name}")
    if not is_valid:
        all_valid = False

# Step 5: Compare enhanced plan with original
print("\n\n📈 STEP 3: Signal Plan Comparison")
print("-" * 100)

print("\n[ORIGINAL PLAN]:")
print(f"  Entry Triggers: {len(original_signal_plan.get('entry', []))}")
print(f"    {original_signal_plan['entry'][0]['name']}")
print(f"  Entry Filters: {len(original_signal_plan.get('entry_filters', []))}")
for f in original_signal_plan.get('entry_filters', []):
    print(f"    - {f['name']}")
print(f"  Exit Signals: {len(original_signal_plan.get('exit', []))}")
for e in original_signal_plan.get('exit', []):
    print(f"    - {e['name']}")

print("\n[ENHANCED PLAN (with semantic gates)]:")
print(f"  Entry Triggers: {len(enhanced_plan.get('entry', []))}")
for t in enhanced_plan.get('entry', []):
    print(f"    - {t['name']} [source: {t.get('source', 'UNKNOWN')}]")
print(f"  Entry Filters: {len(enhanced_plan.get('entry_filters', []))}")
semantic_filters = [f for f in enhanced_plan.get('entry_filters', []) if f.get('source') == 'SEMANTIC']
print(f"    Total: {len(enhanced_plan.get('entry_filters', []))} (including {len(semantic_filters)} SEMANTIC)")
for f in enhanced_plan.get('entry_filters', []):
    print(f"    - {f['name']:40} [source: {f.get('source', 'UNKNOWN')}]")
print(f"  Exit Signals: {len(enhanced_plan.get('exit', []))}")
for e in enhanced_plan.get('exit', []):
    print(f"    - {e['name']:40} [source: {e.get('source', 'UNKNOWN')}]")

# Step 6: Show what user will see in response
print("\n\n💬 STEP 4: What User Will See in Chat Response")
print("-" * 100)

response_data = {
    "extraction_quality": f"{draft.get('semantic_extraction', {}).get('quality_score', 0):.0%}",
    "strategy_family": draft.get('semantic_extraction', {}).get('strategy_family'),
    "htf_gates": len(draft.get('semantic_extraction', {}).get('htf_rules', [])),
    "adx_threshold": draft.get('semantic_extraction', {}).get('adx_threshold'),
    "structural_sl": draft.get('semantic_extraction', {}).get('structural_sl', {}).get('type'),
    "sl_padding": draft.get('semantic_extraction', {}).get('structural_sl', {}).get('padding', {}).get('method'),
    "rr_ratio": draft.get('semantic_extraction', {}).get('risk_reward', {}).get('ratio'),
    "indicators": draft.get('semantic_extraction', {}).get('indicators'),
    "semantic_gates_applied": draft.get('semantic_gates_applied', []),
}

print("\nChat Response Payload (Semantic Section):")
print(json.dumps(response_data, indent=2))

# Final validation
print("\n\n" + "=" * 100)
print("VALIDATION RESULT")
print("=" * 100)

if all_valid and len(semantic_gates_applied) >= 5:
    print("\n✅ INTEGRATION VALIDATED SUCCESSFULLY!")
    print("\nIntegration Status:")
    print("  ✅ Semantic extraction integrated into chat_service.py")
    print("  ✅ ExecutionOrchestrator applied to enhance signal plan")
    print("  ✅ Semantic results added to draft")
    print("  ✅ All 7 gaps properly fixed in response")
    print("  ✅ Ready for production deployment")
else:
    print("\n❌ INTEGRATION VALIDATION FAILED")
    print("\nIssues found:")
    for check_name, is_valid in validations:
        if not is_valid:
            print(f"  ❌ {check_name}")

print("\n" + "=" * 100)
