#!/usr/bin/env python3
"""
Test script to verify token tracking implementation.

This script demonstrates how token tracking works:
1. Tracks tokens for each LLM API call
2. Shows individual token usage logs
3. Displays comprehensive summary at session end
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.token_tracker import track_tokens, log_token_summary, get_session_token_usage


async def test_token_tracking():
    """Test token tracking functionality."""
    
    print("=" * 80)
    print("TOKEN TRACKING TEST")
    print("=" * 80)
    print()
    
    # Simulate a session
    session_id = "test-session-123"
    
    print(f"📝 Starting session: {session_id}")
    print()
    
    # Simulate multiple API calls
    print("🔄 Simulating API calls...")
    print()
    
    # Call 1: OpenRouter
    track_tokens(
        session_id=session_id,
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct",
        total_tokens=1500,
        prompt_tokens=1000,
        completion_tokens=500,
    )
    
    # Call 2: Groq
    track_tokens(
        session_id=session_id,
        provider="groq",
        model="llama-3.3-70b-versatile",
        total_tokens=2000,
        prompt_tokens=1200,
        completion_tokens=800,
    )
    
    # Call 3: Another OpenRouter call
    track_tokens(
        session_id=session_id,
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct",
        total_tokens=1800,
        prompt_tokens=1100,
        completion_tokens=700,
    )
    
    # Call 4: Another Groq call
    track_tokens(
        session_id=session_id,
        provider="groq",
        model="llama-3.3-70b-versatile",
        total_tokens=2200,
        prompt_tokens=1300,
        completion_tokens=900,
    )
    
    print()
    print("✅ Simulated 4 API calls")
    print()
    
    # Get usage summary
    usage = get_session_token_usage(session_id)
    
    print("📊 Session Usage Summary:")
    print(f"  Total Tokens: {usage['total_tokens']:,}")
    print(f"  Prompt Tokens: {usage['prompt_tokens']:,}")
    print(f"  Completion Tokens: {usage['completion_tokens']:,}")
    print(f"  API Calls: {usage['call_count']}")
    print()
    
    print("By Provider:")
    for provider, tokens in usage['by_provider'].items():
        print(f"  {provider.upper()}: {tokens:,} tokens")
    print()
    
    print("By Model:")
    for model, tokens in usage['by_model'].items():
        print(f"  {model}: {tokens:,} tokens")
    print()
    
    # Log comprehensive summary
    print("=" * 80)
    print("LOGGING COMPREHENSIVE SUMMARY")
    print("=" * 80)
    print()
    
    log_token_summary(session_id, "backtest_complete")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_token_tracking())
