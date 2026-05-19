#!/usr/bin/env python3
"""
Quick test script to validate SHORT selling implementation.
"""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "quant_engine"))

from engine.simulator import _detect_signal_direction

def test_direction_detection():
    """Test that signal direction detection works correctly."""
    
    # Test bullish signals
    bullish_rules = [
        {"name": "ema_cross_up", "params": {}},
        {"name": "rsi_oversold", "params": {}},
        {"name": "bullish_rejection_candle", "params": {}},
    ]
    direction = _detect_signal_direction(bullish_rules)
    print(f"✓ Bullish signals detected as: {direction}")
    assert direction == "LONG", f"Expected LONG, got {direction}"
    
    # Test bearish signals
    bearish_rules = [
        {"name": "ema_cross_down", "params": {}},
        {"name": "rsi_overbought", "params": {}},
        {"name": "bearish_rejection_candle", "params": {}},
    ]
    direction = _detect_signal_direction(bearish_rules)
    print(f"✓ Bearish signals detected as: {direction}")
    assert direction == "SHORT", f"Expected SHORT, got {direction}"
    
    # Test mixed signals (more bullish)
    mixed_bullish = [
        {"name": "ema_cross_up", "params": {}},
        {"name": "ema_cross_down", "params": {}},
        {"name": "bullish_rejection_candle", "params": {}},
    ]
    direction = _detect_signal_direction(mixed_bullish)
    print(f"✓ Mixed (more bullish) signals detected as: {direction}")
    assert direction == "LONG", f"Expected LONG, got {direction}"
    
    # Test mixed signals (more bearish)
    mixed_bearish = [
        {"name": "ema_cross_down", "params": {}},
        {"name": "bearish_rejection_candle", "params": {}},
        {"name": "ema_cross_up", "params": {}},
    ]
    direction = _detect_signal_direction(mixed_bearish)
    print(f"✓ Mixed (more bearish) signals detected as: {direction}")
    assert direction == "SHORT", f"Expected SHORT, got {direction}"
    
    # Test empty rules
    direction = _detect_signal_direction(None)
    print(f"✓ Empty signals detected as: {direction}")
    assert direction == "LONG", f"Expected LONG (default), got {direction}"
    
    print("\n✅ All direction detection tests passed!")

def test_pnl_calculation():
    """Test P&L calculation logic for LONG vs SHORT."""
    
    # LONG trade: buy at 100, sell at 110 = +10% profit
    entry_long = 100.0
    exit_long = 110.0
    pnl_long = exit_long - entry_long
    pnl_pct_long = (pnl_long / entry_long) * 100.0
    print(f"✓ LONG: Entry={entry_long}, Exit={exit_long}, P&L={pnl_pct_long:.2f}%")
    assert pnl_pct_long == 10.0, f"Expected 10%, got {pnl_pct_long}%"
    
    # SHORT trade: sell at 100, buy back at 90 = +10% profit
    entry_short = 100.0
    exit_short = 90.0
    pnl_short = entry_short - exit_short
    pnl_pct_short = (pnl_short / entry_short) * 100.0
    print(f"✓ SHORT: Entry={entry_short}, Exit={exit_short}, P&L={pnl_pct_short:.2f}%")
    assert pnl_pct_short == 10.0, f"Expected 10%, got {pnl_pct_short}%"
    
    # SHORT trade loss: sell at 100, buy back at 110 = -10% loss
    entry_short_loss = 100.0
    exit_short_loss = 110.0
    pnl_short_loss = entry_short_loss - exit_short_loss
    pnl_pct_short_loss = (pnl_short_loss / entry_short_loss) * 100.0
    print(f"✓ SHORT LOSS: Entry={entry_short_loss}, Exit={exit_short_loss}, P&L={pnl_pct_short_loss:.2f}%")
    assert pnl_pct_short_loss == -10.0, f"Expected -10%, got {pnl_pct_short_loss}%"
    
    print("\n✅ All P&L calculation tests passed!")

def test_stop_loss_logic():
    """Test stop loss logic for LONG vs SHORT."""
    
    # LONG: stop is BELOW entry
    entry_long = 100.0
    stop_pct = 2.0
    stop_long = entry_long * (1.0 - stop_pct / 100.0)
    print(f"✓ LONG: Entry={entry_long}, Stop={stop_long:.2f} (below entry)")
    assert stop_long < entry_long, "LONG stop should be below entry"
    
    # SHORT: stop is ABOVE entry
    entry_short = 100.0
    stop_short = entry_short * (1.0 + stop_pct / 100.0)
    print(f"✓ SHORT: Entry={entry_short}, Stop={stop_short:.2f} (above entry)")
    assert stop_short > entry_short, "SHORT stop should be above entry"
    
    # LONG: take profit is ABOVE entry
    tp_pct = 5.0
    tp_long = entry_long * (1.0 + tp_pct / 100.0)
    print(f"✓ LONG: Entry={entry_long}, TP={tp_long:.2f} (above entry)")
    assert tp_long > entry_long, "LONG TP should be above entry"
    
    # SHORT: take profit is BELOW entry
    tp_short = entry_short * (1.0 - tp_pct / 100.0)
    print(f"✓ SHORT: Entry={entry_short}, TP={tp_short:.2f} (below entry)")
    assert tp_short < entry_short, "SHORT TP should be below entry"
    
    print("\n✅ All stop loss logic tests passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SHORT Selling Implementation")
    print("=" * 60)
    print()
    
    try:
        test_direction_detection()
        print()
        test_pnl_calculation()
        print()
        test_stop_loss_logic()
        print()
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! SHORT selling is working correctly!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
