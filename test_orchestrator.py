"""Test the ExecutionOrchestrator with semantic instructions."""
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.execution_orchestrator import ExecutionOrchestrator

# Your original prompt
prompt = """
Multi-Timeframe Trend Following Strategy
Setup: 15-minute chart
Higher Timeframe Confirmation: 1-hour and daily trend should remain bullish
Entry Rules:
- Price should stay above 20 EMA on all higher timeframes
- Pullback should happen near EMA
- Bullish candle should confirm continuation
- ADX should remain above 25
Stop Loss: Below recent swing low with ATR padding
Risk:Reward: 1:3 minimum
"""

# Example signal plan (what the planner would generate)
sample_signal_plan = {
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

print("=" * 80)
print("EXECUTION ORCHESTRATOR TEST")
print("=" * 80)

# Extract semantic instructions
extractor = SemanticExtractor()
instructions = extractor.extract(prompt)

print(f"\n📊 SEMANTIC EXTRACTION:")
print(f"   Strategy Family: {instructions.strategy_family}")
print(f"   HTF Rules: {len(instructions.htf_rules)}")
print(f"   Stop Loss: {instructions.stop_loss.type if instructions.stop_loss else None}")
print(f"   ADX Threshold: {instructions.volume_momentum.momentum.adx_threshold if instructions.volume_momentum and instructions.volume_momentum.momentum else None}")

# Apply orchestration
orchestrator = ExecutionOrchestrator()
enhanced_plan = orchestrator.apply_semantic_gates(sample_signal_plan, instructions)

print(f"\n🎯 ENHANCED SIGNAL PLAN:")
print(f"\n   Entry Triggers: {len(enhanced_plan.get('entry', []))}")
for sig in enhanced_plan.get('entry', []):
    print(f"      - {sig['name']} (source: {sig.get('source', 'UNKNOWN')})")

print(f"\n   Entry Filters: {len(enhanced_plan.get('entry_filters', []))}")
for sig in enhanced_plan.get('entry_filters', []):
    print(f"      - {sig['name']} (source: {sig.get('source', 'UNKNOWN')})")
    if 'params' in sig:
        for k, v in sig['params'].items():
            print(f"        └─ {k}: {v}")

print(f"\n   Exit Signals: {len(enhanced_plan.get('exit', []))}")
for sig in enhanced_plan.get('exit', []):
    print(f"      - {sig['name']} (source: {sig.get('source', 'UNKNOWN')})")

print(f"\n📝 SEMANTIC GATES APPLIED:")
if '_semantic_gates_applied' in enhanced_plan:
    for gate in enhanced_plan['_semantic_gates_applied']:
        print(f"   ✅ {gate}")
else:
    print("   No gates applied")

# Test RR validation
print(f"\n✅ RR VALIDATION:")
entry = 1000
sl = 985  # 15 point SL
tp = 1030  # 30 point TP (2:1 actual)

is_valid, actual_rr, reason = orchestrator.validate_rr(
    entry, sl, tp, instructions.risk_reward
)
print(f"   Entry: {entry}, SL: {sl}, TP: {tp}")
print(f"   Required RR: 1:3")
print(f"   Actual RR: 1:{actual_rr:.1f}")
print(f"   Valid: {is_valid}")
print(f"   Reason: {reason}")

print("\n" + "=" * 80)
