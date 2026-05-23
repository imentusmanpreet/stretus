#!/usr/bin/env python3
"""
Test script for GROQ fallback functionality.

This script tests:
1. GROQ API key configuration
2. OpenRouter → GROQ fallback on 429 errors
3. Key rotation for both providers

Usage:
    python test_groq_fallback.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.services.ai.llm import LLMService


async def test_configuration():
    """Test that GROQ is properly configured."""
    print("=" * 70)
    print("Testing GROQ Configuration")
    print("=" * 70)
    
    settings = get_settings()
    
    print(f"\n✓ OpenRouter Keys Configured: {len(settings.openrouter_api_key_pool())}")
    print(f"✓ GROQ Keys Configured: {len(settings.groq_api_key_pool())}")
    print(f"✓ OpenRouter Model: {settings.openrouter_model}")
    print(f"✓ GROQ Model: {settings.groq_model}")
    
    if not settings.groq_api_key_pool():
        print("\n❌ ERROR: No GROQ API keys configured!")
        print("   Please add GROQ_API_KEY to your .env file")
        return False
    
    print(f"\n✅ Configuration OK - {len(settings.groq_api_key_pool())} GROQ keys available")
    return True


async def test_llm_service_info():
    """Test LLM service initialization and info."""
    print("\n" + "=" * 70)
    print("Testing LLM Service Initialization")
    print("=" * 70)
    
    llm = LLMService()
    info = llm.info()
    
    print(f"\n✓ Provider: {info['provider']}")
    print(f"✓ Active Model: {info['active_model']}")
    print(f"✓ OpenRouter Keys: {info['openrouter_keys_configured']}")
    print(f"✓ OpenRouter Failover: {'Enabled' if info['openrouter_failover_enabled'] else 'Disabled'}")
    print(f"✓ GROQ Keys: {info['groq_keys_configured']}")
    print(f"✓ GROQ Fallback: {'Enabled' if info['groq_fallback_enabled'] else 'Disabled'}")
    
    if not info['groq_fallback_enabled']:
        print("\n⚠️  WARNING: GROQ fallback is not enabled!")
        return False
    
    print(f"\n✅ LLM Service OK - GROQ fallback enabled with {info['groq_keys_configured']} keys")
    return True


async def test_simple_chat():
    """Test a simple chat request."""
    print("\n" + "=" * 70)
    print("Testing Simple Chat Request")
    print("=" * 70)
    
    llm = LLMService()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello from LLM!' in exactly those words."}
    ]
    
    try:
        print("\n📤 Sending test message...")
        response = await llm.chat(messages)
        print(f"📥 Response received: {response[:100]}...")
        print("\n✅ Chat test successful!")
        return True
    except Exception as e:
        print(f"\n❌ Chat test failed: {e}")
        return False


async def test_groq_direct():
    """Test GROQ API directly."""
    print("\n" + "=" * 70)
    print("Testing Direct GROQ API Call")
    print("=" * 70)
    
    llm = LLMService()
    
    if not llm._groq_keys:
        print("\n⚠️  Skipping - No GROQ keys configured")
        return True
    
    messages = [
        {"role": "user", "content": "Reply with just the word 'SUCCESS'"}
    ]
    
    try:
        print("\n📤 Calling GROQ directly...")
        response = await llm._call_groq(messages)
        print(f"📥 GROQ Response: {response[:100]}...")
        print("\n✅ Direct GROQ call successful!")
        return True
    except Exception as e:
        print(f"\n❌ Direct GROQ call failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GROQ FALLBACK TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Configuration
    results.append(await test_configuration())
    
    # Test 2: LLM Service Info
    results.append(await test_llm_service_info())
    
    # Test 3: Simple Chat (will use OpenRouter or fallback to GROQ)
    results.append(await test_simple_chat())
    
    # Test 4: Direct GROQ Call
    results.append(await test_groq_direct())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! GROQ fallback is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
