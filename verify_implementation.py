#!/usr/bin/env python3
"""
Verification script for OpenRouter key rotation implementation.
Run this to verify everything is working correctly.
"""

import sys
from pathlib import Path


def verify_env_file():
    """Verify .env file has OpenRouter keys."""
    print("1️⃣  Checking .env file...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("   ❌ .env file not found!")
        return False
    
    content = env_file.read_text()
    if "OPENROUTER_API_KEY=" not in content:
        print("   ❌ OPENROUTER_API_KEY not found in .env!")
        return False
    
    # Count keys
    for line in content.split('\n'):
        if line.startswith('OPENROUTER_API_KEY='):
            keys = line.split('=', 1)[1].split(',')
            keys = [k.strip() for k in keys if k.strip()]
            print(f"   ✅ Found {len(keys)} keys in .env")
            return True
    
    print("   ❌ Could not parse OPENROUTER_API_KEY")
    return False


def verify_json_file():
    """Verify JSON state file exists and has correct structure."""
    print("\n2️⃣  Checking JSON state file...")
    
    json_file = Path("app/services/ai/openrouter_key_state.json")
    if not json_file.exists():
        print("   ❌ JSON state file not found!")
        return False
    
    import json
    try:
        with open(json_file) as f:
            state = json.load(f)
        
        if 'active_key_index' not in state:
            print("   ❌ 'active_key_index' not found in JSON!")
            return False
        
        if 'keys' in state:
            print("   ⚠️  WARNING: 'keys' found in JSON (should only have index)")
            print("   💡 Keys should be in .env, not JSON")
        
        print(f"   ✅ JSON file OK (active_key_index: {state['active_key_index']})")
        return True
    
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")
        return False


def verify_key_loading():
    """Verify keys can be loaded from config."""
    print("\n3️⃣  Testing key loading from .env...")
    
    try:
        from app.core.config import get_settings
        
        settings = get_settings()
        keys = settings.openrouter_api_key_pool()
        
        if not keys:
            print("   ❌ No keys loaded from .env!")
            return False
        
        print(f"   ✅ Loaded {len(keys)} keys from .env")
        print(f"   📝 First key: {keys[0][:25]}...")
        print(f"   📝 Last key: {keys[-1][:25]}...")
        return True
    
    except Exception as e:
        print(f"   ❌ Error loading keys: {e}")
        return False


def verify_key_manager():
    """Verify key manager works correctly."""
    print("\n4️⃣  Testing OpenRouter Key Manager...")
    
    try:
        from app.core.config import get_settings
        from app.services.ai.openrouter_key_manager import get_openrouter_key_manager
        
        settings = get_settings()
        keys = settings.openrouter_api_key_pool()
        
        # Initialize manager
        manager = get_openrouter_key_manager(keys)
        
        # Get stats
        stats = manager.get_stats()
        print(f"   ✅ Manager initialized")
        print(f"   📊 Total keys: {stats['total_keys']}")
        print(f"   📊 Active index: {stats['active_key_index']}")
        print(f"   📊 Keys remaining: {stats['keys_remaining']}")
        
        # Get active key
        active_key = manager.get_active_key()
        print(f"   🔑 Active key: {active_key[:25]}...")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error with key manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_llm_service():
    """Verify LLM service can initialize."""
    print("\n5️⃣  Testing LLM Service...")
    
    try:
        from app.services.ai.llm import LLMService
        
        llm = LLMService()
        info = llm.info()
        
        print(f"   ✅ LLM Service initialized")
        print(f"   📊 Provider: {info['provider']}")
        print(f"   📊 Model: {info['openrouter_model']}")
        
        if 'openrouter_keys_remaining' in info:
            print(f"   📊 Keys remaining: {info['openrouter_keys_remaining']}")
            print(f"   📊 Active index: {info['openrouter_active_key_index']}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error with LLM service: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("OpenRouter Key Rotation - Implementation Verification")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("ENV File", verify_env_file()))
    results.append(("JSON File", verify_json_file()))
    results.append(("Key Loading", verify_key_loading()))
    results.append(("Key Manager", verify_key_manager()))
    results.append(("LLM Service", verify_llm_service()))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("🎉 All checks passed! Implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
