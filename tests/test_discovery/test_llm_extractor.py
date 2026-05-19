"""Phase 9l — LLM-based primitive extractor.

The regex extractor handles common trader phrasings deterministically;
the LLM extractor catches the long tail (any natural-language variant
the user might come up with). These tests pin the contract:

  • build_extraction_tool produces a JSONSchema with the primitive
    enum drawn from PRIMITIVES so adding a new primitive automatically
    extends the LLM's vocabulary.

  • extract_via_llm returns a list of {name, params} dicts that have
    been validated against the primitive library — unknown names
    dropped, non-numeric params dropped.

  • All LLM failure modes (no API key, network error, malformed tool
    output, missing tool call) return [] silently so the caller can
    fall back to the regex path without special-casing.

Tests mock LLMService — no real LLM is called from CI.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, patch

import pytest


if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")


from app.services.discovery.llm_extractor import (
    build_extraction_tool,
    extract_via_llm,
    validate_conditions,
)
from app.services.discovery.primitives import PRIMITIVES


# ── Tool definition derived from the primitive library ─────────────────────


def test_extraction_tool_lists_all_primitives_as_enum_values():
    """The tool's `name` field enum must equal the set of primitive
    names so the LLM can never invent a name we don't recognise."""
    tool = build_extraction_tool()
    fn = tool["function"]
    schema = fn["parameters"]
    enum_values = (
        schema["properties"]["conditions"]["items"]["properties"]["name"]["enum"]
    )
    assert set(enum_values) == set(PRIMITIVES.keys())


def test_extraction_tool_describes_each_primitive():
    """The tool description must include each primitive's text so the
    LLM has enough context to pick the right one."""
    tool = build_extraction_tool()
    description = tool["function"]["description"]
    for name in PRIMITIVES:
        assert name in description, f"primitive {name!r} missing from tool description"


def test_extraction_tool_marks_conditions_as_required():
    tool = build_extraction_tool()
    schema = tool["function"]["parameters"]
    assert "conditions" in schema["required"]
    assert schema["properties"]["conditions"]["type"] == "array"


# ── validate_conditions: clean LLM output ──────────────────────────────────


def test_validate_passes_well_formed_conditions():
    out = validate_conditions([
        {"name": "volume_spike", "params": {"multiplier": 1.5}},
        {"name": "above_vwap"},
    ])
    assert out == [
        {"name": "volume_spike", "params": {"multiplier": 1.5}},
        {"name": "above_vwap", "params": {}},
    ]


def test_validate_drops_unknown_primitive_names():
    """The LLM must never invent names. Validator strips them silently
    so a hallucination becomes a no-op rather than a crash."""
    out = validate_conditions([
        {"name": "volume_spike", "params": {"multiplier": 2.0}},
        {"name": "this_does_not_exist", "params": {}},
        {"name": "above_vwap"},
    ])
    names = [c["name"] for c in out]
    assert "this_does_not_exist" not in names
    assert "volume_spike" in names
    assert "above_vwap" in names


def test_validate_drops_non_numeric_params():
    """LLM might emit `{threshold: "high"}` — drop the bad value but
    keep the primitive (it'll just use defaults)."""
    out = validate_conditions([
        {"name": "rsi_above", "params": {"threshold": "overbought", "extra": 5}},
    ])
    assert out == [{"name": "rsi_above", "params": {"extra": 5.0}}]


def test_validate_handles_empty_or_malformed_input():
    assert validate_conditions([]) == []
    assert validate_conditions([None, "garbage", {"params": {}}, {"name": ""}]) == []
    assert validate_conditions([{"name": "volume_spike"}]) == [
        {"name": "volume_spike", "params": {}},
    ]


# ── extract_via_llm: success path ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_returns_validated_conditions_when_tool_called():
    """Happy path: LLM calls set_discovery_conditions with a primitive
    list. Validator passes through clean, returns the list."""
    fake_response = {
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "name": "set_discovery_conditions",
            "arguments": {
                "conditions": [
                    {"name": "volume_spike", "params": {"multiplier": 2.0}},
                    {"name": "rsi_above", "params": {"threshold": 70}},
                    {"name": "above_vwap"},
                ],
            },
        }],
    }

    fake_llm = type("FakeLLM", (), {
        "chat_with_tools": AsyncMock(return_value=fake_response),
    })()
    with patch("app.services.discovery.llm_extractor.LLMService", return_value=fake_llm):
        result = await extract_via_llm("volume 2x rsi > 70 above vwap")

    assert result == [
        {"name": "volume_spike", "params": {"multiplier": 2.0}},
        {"name": "rsi_above", "params": {"threshold": 70.0}},
        {"name": "above_vwap", "params": {}},
    ]


@pytest.mark.asyncio
async def test_extract_handles_tool_arguments_as_json_string():
    """Some providers serialise tool args as a JSON string. Validator
    must round-trip them back to a dict before passing through."""
    fake_response = {
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "name": "set_discovery_conditions",
            "arguments": '{"conditions":[{"name":"above_vwap"}]}',
        }],
    }
    fake_llm = type("FakeLLM", (), {
        "chat_with_tools": AsyncMock(return_value=fake_response),
    })()
    with patch("app.services.discovery.llm_extractor.LLMService", return_value=fake_llm):
        result = await extract_via_llm("price above vwap")
    assert result == [{"name": "above_vwap", "params": {}}]


# ── extract_via_llm: failure modes all return [] ───────────────────────────


@pytest.mark.asyncio
async def test_extract_returns_empty_when_message_is_blank():
    """Don't burn an LLM call on whitespace."""
    assert await extract_via_llm("") == []
    assert await extract_via_llm("   ") == []


@pytest.mark.asyncio
async def test_extract_returns_empty_when_llm_init_fails():
    """No API key configured / LLMService raises on construction."""
    with patch(
        "app.services.discovery.llm_extractor.LLMService",
        side_effect=Exception("no api key"),
    ):
        assert await extract_via_llm("volume 2x") == []


@pytest.mark.asyncio
async def test_extract_returns_empty_when_llm_call_raises():
    """Network error, rate limit, etc. — all silently return []."""
    fake_llm = type("FakeLLM", (), {
        "chat_with_tools": AsyncMock(side_effect=Exception("rate limit")),
    })()
    with patch("app.services.discovery.llm_extractor.LLMService", return_value=fake_llm):
        assert await extract_via_llm("volume 2x") == []


@pytest.mark.asyncio
async def test_extract_returns_empty_when_no_tool_call_in_response():
    """LLM replied with plain text instead of calling the tool."""
    fake_response = {"content": "I don't see any conditions here.", "tool_calls": []}
    fake_llm = type("FakeLLM", (), {
        "chat_with_tools": AsyncMock(return_value=fake_response),
    })()
    with patch("app.services.discovery.llm_extractor.LLMService", return_value=fake_llm):
        assert await extract_via_llm("hi") == []


@pytest.mark.asyncio
async def test_extract_returns_empty_when_conditions_field_missing():
    fake_response = {
        "content": "",
        "tool_calls": [{"id": "1", "name": "set_discovery_conditions",
                        "arguments": {"oops": "no conditions key"}}],
    }
    fake_llm = type("FakeLLM", (), {
        "chat_with_tools": AsyncMock(return_value=fake_response),
    })()
    with patch("app.services.discovery.llm_extractor.LLMService", return_value=fake_llm):
        assert await extract_via_llm("volume 2x") == []


@pytest.mark.asyncio
async def test_extract_strips_hallucinated_primitives_from_llm_output():
    """End-to-end defense: LLM emits a hallucinated name → silently
    dropped by the validator, only the real ones survive."""
    fake_response = {
        "content": "",
        "tool_calls": [{
            "id": "1",
            "name": "set_discovery_conditions",
            "arguments": {
                "conditions": [
                    {"name": "moon_phase_alignment", "params": {}},   # hallucinated
                    {"name": "volume_spike", "params": {"multiplier": 1.5}},
                ],
            },
        }],
    }
    fake_llm = type("FakeLLM", (), {
        "chat_with_tools": AsyncMock(return_value=fake_response),
    })()
    with patch("app.services.discovery.llm_extractor.LLMService", return_value=fake_llm):
        result = await extract_via_llm("volume 1.5x")
    assert result == [{"name": "volume_spike", "params": {"multiplier": 1.5}}]
