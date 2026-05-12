"""
Comprehensive test: Multi-Timeframe Trend Following Strategy
Tests all 7 gaps with the original user prompt.
"""
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.execution_orchestrator import ExecutionOrchestrator

# Original user prompt from the issue
prompt = """
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

# Simulated signal plan from planner (before orchestration)
original_plan = {
    "entry": [
        {
            "name": "ema_pullback_bullish",
            "params": {"window": 20, "lookback": 3},
            "signal_type": "TRIGGER",
            "source": "FAMILY"
        }
    ],
    "entry_filters": [
        {
            "name": "volume_spike",
            "params": {"window": 20, "multiplier": 1.5},
            "signal_type": "FILTER",
            "source": "FAMILY"
        }
    ],
    "exit": [
        {
            "name": "price_below_ema",
            "params": {"window": 29},
            "signal_type": "TRIGGER",
            "source": "FAMILY"
        }
    ]
}

print("=" * 100)
print("COMPREHENSIVE TEST: Multi-Timeframe Trend Following Strategy")
print("=" * 100)

# Extract semantics
extractor = SemanticExtractor()
instructions = extractor.extract(prompt)

print("\n📊 GAP ANALYSIS - BEFORE FIXES")
print("-" * 100)
print("\nGap #1: ADX Threshold Extraction")
if instructions.volume_momentum and instructions.volume_momentum.momentum:
    mom = instructions.volume_momentum.momentum
    if mom.adx_threshold:
        print(f"  ✅ FIXED: ADX > {mom.adx_threshold} extracted correctly")
    else:
        print(f"  ❌ NOT FIXED: ADX not extracted")
else:
    print(f"  ❌ NOT FIXED: Momentum filter missing")

print("\nGap #2: Multi-Timeframe HTF Confirmation")
if instructions.htf_rules:
    print(f"  ✅ FIXED: {len(instructions.htf_rules)} HTF rules extracted")
    for rule in instructions.htf_rules[:3]:  # Show first 3
        print(f"     • {rule.timeframe}: {rule.condition} (role: {rule.role})")
else:
    print(f"  ❌ NOT FIXED: No HTF rules")

print("\nGap #3: Structural SL with ATR Padding")
if instructions.stop_loss:
    sl = instructions.stop_loss
    print(f"  ✅ FIXED: {sl.type} with {sl.padding.method if sl.padding else 'no'} padding")
    if sl.padding and sl.padding.method == "atr":
        print(f"     • ATR Multiple: {sl.padding.atr_multiple}x")
else:
    print(f"  ❌ NOT FIXED: SL not extracted")

print("\nGap #4: Risk:Reward Specification")
if instructions.risk_reward:
    print(f"  ✅ FIXED: RR {instructions.risk_reward.type} @ 1:{instructions.risk_reward.ratio}")
else:
    print(f"  ❌ NOT FIXED: RR not extracted")

print("\nGap #5: Indicator Extraction (EMA Windows)")
if instructions.indicators:
    print(f"  ✅ FIXED: Indicators extracted: {instructions.indicators}")
    if 'EMA' in instructions.indicators and 29 not in instructions.indicators['EMA']:
        print(f"     • No mystery EMA(29)! ✅")
    else:
        print(f"     • Mystery EMA(29) still present ❌")
else:
    print(f"  ❌ NOT FIXED: No indicators extracted")

print("\nGap #6: Candle Confirmation")
if instructions.candle_confirmation:
    print(f"  ✅ FIXED: {instructions.candle_confirmation.filter_type} detected")
else:
    print(f"  ❌ NOT FIXED: Candle confirmation not extracted")

print("\nGap #7: Signal Origin Tracking (check in orchestrator output)")

# Apply orchestration
orchestrator = ExecutionOrchestrator()
enhanced_plan = orchestrator.apply_semantic_gates(original_plan, instructions)

print("\n\n📈 SIGNAL PLAN - AFTER ORCHESTRATION")
print("-" * 100)

print(f"\nEntry Triggers: {len(enhanced_plan.get('entry', []))}")
for sig in enhanced_plan.get('entry', []):
    print(f"  • {sig['name']:40} [source: {sig.get('source', '?')}]")

print(f"\nEntry Filters Added by Semantics:")
semantic_filters = [s for s in enhanced_plan.get('entry_filters', []) if s.get('source') == 'SEMANTIC']
print(f"  Count: {len(semantic_filters)}")
for sig in semantic_filters:
    print(f"  • {sig['name']:40} [source: {sig['source']}]")

print(f"\nExit Signals:")
for sig in enhanced_plan.get('exit', []):
    source = sig.get('source', 'UNKNOWN')
    print(f"  • {sig['name']:40} [source: {source}]")
    # Check if generic SL was replaced
    if 'price_below' in sig.get('name', ''):
        print(f"     ⚠️  Generic exit still present!")
    if 'structural_sl' in sig.get('name', ''):
        print(f"     ✅ Structural SL applied!")

print(f"\n📝 Semantic Gates Applied: {len(enhanced_plan.get('_semantic_gates_applied', []))}")
for gate in enhanced_plan.get('_semantic_gates_applied', []):
    print(f"  ✅ {gate}")

# RR Validation
print("\n\n💰 RISK:REWARD VALIDATION")
print("-" * 100)
entry = 1000
sl = 985  # 15 point SL (below swing low)
tp_correct = 1045  # 45 points for 1:3 RR

is_valid, actual_rr, reason = orchestrator.validate_rr(
    entry, sl, tp_correct, instructions.risk_reward
)
print(f"\nScenario: Entry={entry}, SL={sl}, TP={tp_correct}")
print(f"Required RR: 1:{instructions.risk_reward.ratio if instructions.risk_reward else '?'}")
print(f"Actual RR: 1:{actual_rr:.1f}")
print(f"Valid: {is_valid} - {reason}")

# Summary
print("\n\n" + "=" * 100)
print("SUMMARY: Gap Resolution Status")
print("=" * 100)

gaps_fixed = [
    ("Gap #1: ADX Threshold Extraction", instructions.volume_momentum and instructions.volume_momentum.momentum and instructions.volume_momentum.momentum.adx_threshold),
    ("Gap #2: Multi-Timeframe Confirmation", len(instructions.htf_rules) > 0 and len(semantic_filters) >= 2),
    ("Gap #3: Structural SL (ATR Padding)", instructions.stop_loss and instructions.stop_loss.padding and instructions.stop_loss.padding.method == 'atr'),
    ("Gap #4: RR Specification", instructions.risk_reward and instructions.risk_reward.ratio == 3.0),
    ("Gap #5: EMA Windows (no mystery 29)", 'EMA' in instructions.indicators and 20 in instructions.indicators['EMA'] and 29 not in instructions.indicators.get('EMA', [])),
    ("Gap #6: Candle Confirmation", instructions.candle_confirmation is not None),
    ("Gap #7: Signal Origin Tracking", any(s.get('source') == 'SEMANTIC' for s in enhanced_plan.get('entry_filters', []))),
]

all_fixed = True
for gap_name, is_fixed in gaps_fixed:
    status = "✅ FIXED" if is_fixed else "❌ PENDING"
    print(f"\n{gap_name}")
    print(f"   {status}")
    if not is_fixed:
        all_fixed = False

print("\n" + "=" * 100)
if all_fixed:
    print("🎉 ALL GAPS FIXED! Your chat service is now highly efficient!")
else:
    print("⚠️  Some gaps remain. Review above for details.")

print(f"\n📊 Extraction Quality Score: {instructions.extraction_quality_score:.1%}")
print("=" * 100)
