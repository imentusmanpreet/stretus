#!/usr/bin/env python3
"""
Setup script for OpenRouter API key rotation system.

This script helps you initialize the openrouter_key_state.json file
with your API keys.

Usage:
    python setup_openrouter_keys.py
"""

import json
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("OpenRouter API Key Setup")
    print("=" * 70)
    print()
    
    # Path to the state file
    state_file = Path("app/services/ai/openrouter_key_state.json")
    
    # Check if file already exists
    if state_file.exists():
        print(f"⚠️  State file already exists: {state_file}")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Aborted. No changes made.")
            return
        print()
    
    # Get API keys from user
    print("Enter your OpenRouter API keys (one per line).")
    print("Press Enter twice when done.")
    print()
    
    keys = []
    while True:
        key = input(f"Key {len(keys) + 1}: ").strip()
        if not key:
            if len(keys) == 0:
                print("❌ You must enter at least one API key.")
                continue
            else:
                break
        
        # Basic validation
        if not key.startswith("sk-or-v1-"):
            print("⚠️  Warning: Key doesn't start with 'sk-or-v1-'. Are you sure this is correct?")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                continue
        
        keys.append(key)
    
    print()
    print(f"✅ Collected {len(keys)} API keys")
    print()
    
    # Create state structure
    state = {
        "active_key_index": 0,
        "keys": keys
    }
    
    # Write to file
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✅ Successfully created: {state_file}")
        print()
        print("Next steps:")
        print("1. Verify the file was created correctly")
        print("2. Set LLM_PROVIDER=openrouter in your .env file")
        print("3. Restart your application")
        print()
        print("⚠️  IMPORTANT: Never commit this file to git!")
        print("   It's already in .gitignore, but double-check.")
        
    except Exception as e:
        print(f"❌ Error writing state file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
