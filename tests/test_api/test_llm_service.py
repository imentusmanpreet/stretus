from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.services.ai import llm as llm_module


def test_settings_groq_api_key_pool_merges_and_dedupes_sources(monkeypatch):
    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_primary",
        groq_api_keys="gsk_secondary,\ngsk_third,gsk_primary",
    )
    monkeypatch.setattr(
        settings,
        "_settings_env_values",
        lambda: {
            "GROQ_API_KEY_2": "gsk_fourth",
            "GROQ_API_KEY_1": "gsk_primary",
            "GROQ_API_KEY_EXTRA": "gsk_fifth",
        },
    )

    assert settings.groq_api_key_pool() == [
        "gsk_primary",
        "gsk_secondary",
        "gsk_third",
        "gsk_fourth",
        "gsk_fifth",
    ]


def test_settings_groq_api_key_pool_supports_indexed_single_variable(monkeypatch):
    settings = Settings(
        _env_file=None,
        groq_api_key="3:gsk_third,1:gsk_first,[2]=gsk_second,4:gsk_first",
    )
    monkeypatch.setattr(settings, "_settings_env_values", lambda: {})

    assert settings.groq_api_key_pool() == [
        "gsk_first",
        "gsk_second",
        "gsk_third",
    ]


def test_llm_service_rotates_to_next_groq_key_on_rate_limit(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            effective_provider=lambda: "groq",
            groq_api_key="",
            groq_api_key_pool=lambda: ["gsk_one", "gsk_two"],
            groq_model="llama-3.3-70b-versatile",
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5:7b",
        ),
    )
    llm_module.LLMService._groq_key_cursor = 0
    service = llm_module.LLMService()

    attempts: list[str] = []

    async def fake_call_once(self, messages, api_key, model=None, **kwargs):
        attempts.append(api_key)
        if api_key == "gsk_one":
            raise AppError(429, "Rate limit exceeded.")
        return "ok-from-second-key"

    monkeypatch.setattr(llm_module.LLMService, "_call_groq_once", fake_call_once)

    result = asyncio.run(service.chat([{"role": "user", "content": "hi"}]))

    assert result == "ok-from-second-key"
    assert attempts == ["gsk_one", "gsk_two"]
    assert llm_module.LLMService._groq_key_cursor == 1


def test_llm_service_raises_when_all_groq_keys_are_rate_limited(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            effective_provider=lambda: "groq",
            groq_api_key="",
            groq_api_key_pool=lambda: ["gsk_one", "gsk_two"],
            groq_model="llama-3.3-70b-versatile",
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5:7b",
        ),
    )
    llm_module.LLMService._groq_key_cursor = 0
    service = llm_module.LLMService()

    async def fake_call_once(self, messages, api_key, model=None, **kwargs):
        raise AppError(429, "Rate limit exceeded.")

    monkeypatch.setattr(llm_module.LLMService, "_call_groq_once", fake_call_once)

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.chat([{"role": "user", "content": "hi"}]))

    assert exc_info.value.status_code == 429
    assert "All configured Groq API keys are currently rate limited" in exc_info.value.message


def test_llm_service_skips_invalid_groq_key_and_uses_next_key(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            effective_provider=lambda: "groq",
            groq_api_key="",
            groq_api_key_pool=lambda: ["gsk_invalid", "gsk_valid"],
            groq_model="llama-3.3-70b-versatile",
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5:7b",
        ),
    )
    llm_module.LLMService._groq_key_cursor = 0
    service = llm_module.LLMService()

    attempts: list[str] = []

    async def fake_call_once(self, messages, api_key, model=None, **kwargs):
        attempts.append(api_key)
        if api_key == "gsk_invalid":
            raise AppError(401, "Authentication failed.")
        return "ok-from-valid-key"

    monkeypatch.setattr(llm_module.LLMService, "_call_groq_once", fake_call_once)

    result = asyncio.run(service.chat([{"role": "user", "content": "hi"}]))

    assert result == "ok-from-valid-key"
    assert attempts == ["gsk_invalid", "gsk_valid"]
    assert llm_module.LLMService._groq_key_cursor == 1


def test_llm_service_chat_with_tools_rotates_groq_keys(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            effective_provider=lambda: "groq",
            groq_api_key="",
            groq_api_key_pool=lambda: ["gsk_one", "gsk_two"],
            groq_model="llama-3.3-70b-versatile",
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5:7b",
        ),
    )
    llm_module.LLMService._groq_key_cursor = 0
    service = llm_module.LLMService()
    attempts: list[str] = []

    async def fake_call_once(self, messages, tools, api_key, model=None, **kwargs):
        attempts.append(api_key)
        if api_key == "gsk_one":
            raise AppError(429, "Rate limit exceeded.")
        return {
            "content": "",
            "tool_calls": [{"name": "run_backtest", "arguments": {"session_id": "s1"}}],
        }

    monkeypatch.setattr(llm_module.LLMService, "_call_groq_tools_once", fake_call_once)

    result = asyncio.run(
        service.chat_with_tools(
            [{"role": "user", "content": "run"}],
            [{"type": "function", "function": {"name": "run_backtest", "parameters": {"type": "object"}}}],
        )
    )

    assert result["tool_calls"][0]["name"] == "run_backtest"
    assert attempts == ["gsk_one", "gsk_two"]
    assert llm_module.LLMService._groq_key_cursor == 1
