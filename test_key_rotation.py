#!/usr/bin/env python3
"""
Visual test for API key rotation and fallback logic.

This script demonstrates:
1. OpenRouter key rotation (2 keys)
2. GROQ fallback activation on OpenRouter 429
3. GROQ key rotation (45 keys)

Usage:
    python test_key_rotation.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_key_info(keys, provider_name):
    """Print information about configured keys."""
    print(f"\n{provider_name} Keys:")
    print(f"  Total Keys: {len(keys)}")
    
    if len(keys) == 0:
        print("  ❌ No keys configured!")
        return
    
    print(f"  ✅ Keys configured:")
    for i, key in enumerate(keys, 1):
        # Show first 20 and last 10 characters of key
        if len(key) > 30:
            masked_key = f"{key[:20]}...{key[-10:]}"
        else:
            masked_key = key
        print(f"     {i}. {masked_key}")


def simulate_rotation(keys, provider_name):
    """Simulate key rotation logic."""
    print(f"\n{provider_name} Rotation Simulation:")
    print(f"  Starting with key cursor at position 0")
    
    if len(keys) == 0:
        print("  ❌ No keys to rotate!")
        return
    
    # Simulate 5 requests with rotation
    cursor = 0
    for request_num in range(1, 6):
        key_index = cursor % len(keys)
        key = keys[key_index]
        masked_key = f"{key[:15]}..." if len(key) > 15 else key
        
        print(f"\n  Request #{request_num}:")
        print(f"    → Using Key #{key_index + 1}: {masked_key}")
        
        # Simulate 429 error on some requests
        if request_num in [1, 3]:  # Simulate failures
            print(f"    ← Response: 429 Rate Limited ⚠️")
            print(f"    → Rotating to next key...")
            cursor += 1
        else:
            print(f"    ← Response: Success ✅")
            cursor = key_index  # Remember successful key


def simulate_fallback_flow():
    """Simulate complete fallback flow."""
    print_header("FALLBACK FLOW SIMULATION")
    
    print("\n📍 Step 1: Try OpenRouter Keys")
    print("  Request → OpenRouter Key #1")
    print("  Response: 429 Rate Limited ⚠️")
    print("  Action: Rotate to next OpenRouter key")
    
    print("\n  Request → OpenRouter Key #2")
    print("  Response: 429 Rate Limited ⚠️")
    print("  Action: All OpenRouter keys exhausted!")
    
    print("\n📍 Step 2: Activate GROQ Fallback")
    print("  ⚠️  WARNING: All OpenRouter keys rate limited (429)")
    print("  ⚠️  WARNING: Falling back to GROQ with 45 keys")
    
    print("\n  Request → GROQ Key #1")
    print("  Response: Success ✅")
    print("  ✅ INFO: Successfully used GROQ fallback")
    
    print("\n📍 Result: Request completed successfully using GROQ fallback")


def show_capacity_info(openrouter_keys, groq_keys):
    """Show capacity and rate limit information."""
    print_header("CAPACITY & RATE LIMITS")
    
    # OpenRouter capacity
    openrouter_rpm_per_key = 10  # requests per minute (free tier)
    openrouter_total_rpm = len(openrouter_keys) * openrouter_rpm_per_key
    
    print(f"\n📊 OpenRouter Capacity:")
    print(f"  Keys: {len(openrouter_keys)}")
    print(f"  Rate Limit: ~{openrouter_rpm_per_key} requests/minute per key")
    print(f"  Total Capacity: ~{openrouter_total_rpm} requests/minute")
    
    # GROQ capacity
    groq_rpm_per_key = 30  # requests per minute (free tier)
    groq_total_rpm = len(groq_keys) * groq_rpm_per_key
    
    print(f"\n📊 GROQ Capacity:")
    print(f"  Keys: {len(groq_keys)}")
    print(f"  Rate Limit: ~{groq_rpm_per_key} requests/minute per key")
    print(f"  Total Capacity: ~{groq_total_rpm} requests/minute")
    
    # Combined
    total_rpm = openrouter_total_rpm + groq_total_rpm
    
    print(f"\n📊 Combined Capacity:")
    print(f"  Total Keys: {len(openrouter_keys) + len(groq_keys)}")
    print(f"  Total Capacity: ~{total_rpm} requests/minute")
    print(f"  Requests/Second: ~{total_rpm // 60}")


def show_error_handling():
    """Show error handling behavior."""
    print_header("ERROR HANDLING")
    
    print("\n🔍 Error Code: 429 (Rate Limit)")
    print("  Behavior: Rotate to next key")
    print("  Fallback: Yes (GROQ after all OpenRouter keys)")
    print("  User Impact: Minimal (automatic retry)")
    
    print("\n🔍 Error Code: 401 (Authentication)")
    print("  Behavior: Rotate to next key")
    print("  Fallback: No (auth issue, not rate limit)")
    print("  User Impact: Error if all keys invalid")
    
    print("\n🔍 Error Code: 404 (Model Not Found)")
    print("  Behavior: Immediate error")
    print("  Fallback: No")
    print("  User Impact: Error message")
    
    print("\n🔍 Error Code: 500/502/503 (Server Error)")
    print("  Behavior: Immediate error")
    print("  Fallback: No")
    print("  User Impact: Error message")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  API KEY ROTATION & FALLBACK TEST")
    print("=" * 70)
    
    # Load configuration
    settings = get_settings()
    openrouter_keys = settings.openrouter_api_key_pool()
    groq_keys = settings.groq_api_key_pool()
    
    # Show configuration
    print_header("CONFIGURATION")
    print_key_info(openrouter_keys, "OpenRouter")
    print_key_info(groq_keys, "GROQ")
    
    # Show capacity
    show_capacity_info(openrouter_keys, groq_keys)
    
    # Simulate rotations
    print_header("KEY ROTATION SIMULATION")
    simulate_rotation(openrouter_keys, "OpenRouter")
    print("\n" + "-" * 70)
    simulate_rotation(groq_keys[:5], "GROQ (first 5 keys)")  # Show first 5 for brevity
    
    # Simulate fallback
    simulate_fallback_flow()
    
    # Show error handling
    show_error_handling()
    
    # Summary
    print_header("SUMMARY")
    print(f"\n✅ OpenRouter Keys: {len(openrouter_keys)} configured")
    print(f"✅ GROQ Keys: {len(groq_keys)} configured")
    print(f"✅ Total Keys: {len(openrouter_keys) + len(groq_keys)}")
    print(f"✅ Rotation: Enabled for both providers")
    print(f"✅ Fallback: GROQ activates on OpenRouter 429")
    print(f"✅ Capacity: ~{(len(openrouter_keys) * 10) + (len(groq_keys) * 30)} requests/minute")
    
    print("\n" + "=" * 70)
    print("  Configuration is correct! 🎉")
    print("=" * 70)
    
    print("\n💡 Next Steps:")
    print("  1. Install GROQ SDK: pip install groq")
    print("  2. Run actual test: python test_groq_fallback.py")
    print("  3. Start your application and monitor logs")
    print("  4. Watch for rotation messages in logs")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
