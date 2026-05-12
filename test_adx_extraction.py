"""Quick test for ADX threshold extraction fix."""
from app.planner.semantic_extractor import SemanticExtractor

# Your original prompt
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

extractor = SemanticExtractor()
instructions = extractor.extract(prompt)

print("=" * 80)
print("SEMANTIC EXTRACTION TEST - ADX THRESHOLD")
print("=" * 80)

print(f"\n✅ Strategy Family: {instructions.strategy_family}")
print(f"✅ HTF Rules Extracted: {len(instructions.htf_rules)}")
print(f"✅ Stop Loss: {instructions.stop_loss}")
print(f"✅ Risk:Reward: {instructions.risk_reward}")
print(f"✅ Indicators: {instructions.indicators}")

# Check momentum/ADX extraction
if instructions.volume_momentum and instructions.volume_momentum.momentum:
    mom = instructions.volume_momentum.momentum
    print(f"\n✅ MOMENTUM FILTER DETECTED!")
    print(f"   - Filter Type: {mom.filter_type}")
    print(f"   - ADX Threshold: {mom.adx_threshold}")
    print(f"   - Description: {mom.description}")
    if mom.adx_threshold == 25.0:
        print(f"   ✅ ADX > 25 CORRECTLY EXTRACTED!")
    else:
        print(f"   ⚠️  ADX threshold might be incorrect")
else:
    print(f"\n❌ MOMENTUM FILTER NOT DETECTED - ADX extraction failed")

print(f"\n📊 Extraction Quality Score: {instructions.extraction_quality_score:.1%}")
print("=" * 80)
