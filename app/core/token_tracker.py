"""
Token usage tracking for LLM API calls.

Tracks token consumption per session from start to backtest completion.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class TokenTracker:
    """
    Singleton token tracker for monitoring LLM API token usage per session.
    
    Tracks:
    - Total tokens consumed per session
    - Tokens per provider (Groq, OpenRouter, etc.)
    - Tokens per model
    - Token consumption timeline
    """
    
    _instance = None
    _session_tokens: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "by_provider": defaultdict(int),
        "by_model": defaultdict(int),
        "calls": [],
        "start_time": None,
        "last_update": None,
    })
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def track_usage(
        self,
        session_id: str,
        provider: str,
        model: str,
        total_tokens: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """
        Track token usage for a session.
        
        Args:
            session_id: Chat session ID
            provider: Provider name (groq, openrouter, etc.)
            model: Model name
            total_tokens: Total tokens consumed
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
        """
        if not session_id or total_tokens <= 0:
            return
        
        session_data = self._session_tokens[session_id]
        
        # Initialize start time on first call
        if session_data["start_time"] is None:
            session_data["start_time"] = datetime.now(timezone.utc)
        
        # Update totals
        session_data["total_tokens"] += total_tokens
        session_data["prompt_tokens"] += prompt_tokens
        session_data["completion_tokens"] += completion_tokens
        session_data["by_provider"][provider.lower()] += total_tokens
        session_data["by_model"][model] += total_tokens
        session_data["last_update"] = datetime.now(timezone.utc)
        
        # Record individual call
        session_data["calls"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
        
        # Log token usage
        logger.info(
            "🪙 TOKEN_USAGE|session_id=%s|provider=%s|model=%s|tokens=%d|"
            "session_total=%d|prompt=%d|completion=%d",
            session_id,
            provider,
            model,
            total_tokens,
            session_data["total_tokens"],
            prompt_tokens,
            completion_tokens,
        )
    
    def get_session_usage(self, session_id: str) -> dict[str, Any]:
        """Get token usage summary for a session."""
        if session_id not in self._session_tokens:
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "by_provider": {},
                "by_model": {},
                "call_count": 0,
            }
        
        session_data = self._session_tokens[session_id]
        return {
            "total_tokens": session_data["total_tokens"],
            "prompt_tokens": session_data["prompt_tokens"],
            "completion_tokens": session_data["completion_tokens"],
            "by_provider": dict(session_data["by_provider"]),
            "by_model": dict(session_data["by_model"]),
            "call_count": len(session_data["calls"]),
            "start_time": session_data["start_time"].isoformat() if session_data["start_time"] else None,
            "last_update": session_data["last_update"].isoformat() if session_data["last_update"] else None,
        }
    
    def log_session_summary(self, session_id: str, event: str = "summary") -> None:
        """Log a comprehensive token usage summary for a session."""
        usage = self.get_session_usage(session_id)
        
        if usage["total_tokens"] == 0:
            logger.info(
                "🪙 TOKEN_SUMMARY|session_id=%s|event=%s|total_tokens=0|no_api_calls",
                session_id,
                event,
            )
            return
        
        logger.info("=" * 100)
        logger.info("🪙 TOKEN USAGE SUMMARY - Session: %s", session_id)
        logger.info("=" * 100)
        logger.info("Event: %s", event)
        logger.info("Total API Calls: %d", usage["call_count"])
        logger.info("Total Tokens Consumed: %s", f"{usage['total_tokens']:,}")
        logger.info("  - Prompt Tokens: %s", f"{usage['prompt_tokens']:,}")
        logger.info("  - Completion Tokens: %s", f"{usage['completion_tokens']:,}")
        logger.info("-" * 100)
        
        if usage["by_provider"]:
            logger.info("By Provider:")
            for provider, tokens in sorted(usage["by_provider"].items(), key=lambda x: x[1], reverse=True):
                percentage = (tokens / usage["total_tokens"] * 100) if usage["total_tokens"] > 0 else 0
                logger.info("  - %s: %s tokens (%.1f%%)", provider.upper(), f"{tokens:,}", percentage)
        
        if usage["by_model"]:
            logger.info("By Model:")
            for model, tokens in sorted(usage["by_model"].items(), key=lambda x: x[1], reverse=True):
                percentage = (tokens / usage["total_tokens"] * 100) if usage["total_tokens"] > 0 else 0
                logger.info("  - %s: %s tokens (%.1f%%)", model, f"{tokens:,}", percentage)
        
        logger.info("=" * 100)
        
        # Also log structured format for parsing
        logger.info(
            "🪙 TOKEN_SUMMARY|session_id=%s|event=%s|total_tokens=%d|prompt_tokens=%d|"
            "completion_tokens=%d|call_count=%d|providers=%s|models=%s",
            session_id,
            event,
            usage["total_tokens"],
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["call_count"],
            ",".join(usage["by_provider"].keys()),
            ",".join(usage["by_model"].keys()),
        )
    
    def reset_session(self, session_id: str) -> None:
        """Reset token tracking for a session."""
        if session_id in self._session_tokens:
            del self._session_tokens[session_id]
            logger.info("🪙 TOKEN_TRACKER|session_id=%s|action=reset", session_id)
    
    def get_all_sessions(self) -> dict[str, dict[str, Any]]:
        """Get token usage for all tracked sessions."""
        return {
            session_id: self.get_session_usage(session_id)
            for session_id in self._session_tokens.keys()
        }


# Global singleton instance
_tracker = TokenTracker()


def track_tokens(
    session_id: str,
    provider: str,
    model: str,
    total_tokens: int,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """
    Track token usage for a session.
    
    Convenience function for the global tracker instance.
    """
    _tracker.track_usage(
        session_id=session_id,
        provider=provider,
        model=model,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def get_session_token_usage(session_id: str) -> dict[str, Any]:
    """Get token usage summary for a session."""
    return _tracker.get_session_usage(session_id)


def log_token_summary(session_id: str, event: str = "summary") -> None:
    """Log token usage summary for a session."""
    _tracker.log_session_summary(session_id, event)


def reset_session_tokens(session_id: str) -> None:
    """Reset token tracking for a session."""
    _tracker.reset_session(session_id)
