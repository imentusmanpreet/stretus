"""
Generator (propose + repair loop) tests — engine-independent.

The generator's responsibility is ORCHESTRATION: call the LLM, parse JSON, validate,
and repair on failure. We stub ``validate_spec`` so these run without TA-Lib; the real
validation is covered in test_validator.py.
"""
from __future__ import annotations

import json

from app.strategy import generator as gen
from app.strategy.validator import ValidationError, ValidationResult


class FakeLLM:
    """Returns scripted responses in order; an Exception value is raised when reached."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, messages, session_id=None, max_tokens=None, reasoning_effort=None):
        item = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


_VALID_SPEC_JSON = json.dumps(
    {
        "name": "INFY 15m momentum",
        "symbol": "INFY.NS",
        "market": "indian_stocks",
        "timeframe": "15m",
        "objective": "intraday",
        "direction": "long_only",
        "entry_condition": "CLOSE > EMA(20) AND RSI(14) > 60",
        "exit_condition": "RSI(14) < 50",
        "stop_loss": {"type": "percent", "value": 1.5, "source": "user"},
        "take_profit": {"type": "risk_reward", "value": 2.0, "source": "user"},
    }
)


def _ok(*_a, **_k):
    return ValidationResult(notes=[ValidationError("stop_loss", "assumed_stop_loss", "x", "warning")])


def _fail(*_a, **_k):
    return ValidationResult(errors=[ValidationError("entry_condition", "unknown_identifier", "FOO")])


async def test_valid_on_first_attempt(monkeypatch):
    monkeypatch.setattr(gen, "validate_spec", _ok)
    spec, result = await gen.generate_strategy(
        [{"role": "user", "content": "INFY momentum"}],
        llm=FakeLLM([_VALID_SPEC_JSON]), system_prompt="sys",
    )
    assert spec is not None and result.ok
    assert spec.symbol == "INFY.NS"
    assert result.notes  # assumed-value notes pass through for the chat layer


async def test_json_in_markdown_fence_is_parsed(monkeypatch):
    monkeypatch.setattr(gen, "validate_spec", _ok)
    fenced = f"Here you go:\n```json\n{_VALID_SPEC_JSON}\n```"
    spec, result = await gen.generate_strategy(
        [{"role": "user", "content": "x"}], llm=FakeLLM([fenced]), system_prompt="sys",
    )
    assert spec is not None and result.ok


async def test_invalid_json_then_repaired(monkeypatch):
    monkeypatch.setattr(gen, "validate_spec", _ok)
    llm = FakeLLM(["not json at all", _VALID_SPEC_JSON])
    spec, result = await gen.generate_strategy(
        [{"role": "user", "content": "x"}], llm=llm, system_prompt="sys", max_repairs=2,
    )
    assert spec is not None and result.ok
    assert llm.calls == 2  # one bad, one repaired


async def test_validation_errors_then_repaired(monkeypatch):
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        return _fail() if calls["n"] == 1 else _ok()

    monkeypatch.setattr(gen, "validate_spec", flaky)
    llm = FakeLLM([_VALID_SPEC_JSON, _VALID_SPEC_JSON])
    spec, result = await gen.generate_strategy(
        [{"role": "user", "content": "x"}], llm=llm, system_prompt="sys", max_repairs=2,
    )
    assert spec is not None and result.ok
    assert llm.calls == 2


async def test_unrepairable_returns_none_with_errors(monkeypatch):
    monkeypatch.setattr(gen, "validate_spec", _fail)
    spec, result = await gen.generate_strategy(
        [{"role": "user", "content": "x"}],
        llm=FakeLLM([_VALID_SPEC_JSON]), system_prompt="sys", max_repairs=1,
    )
    assert spec is None and not result.ok
    assert any(e.code == "unknown_identifier" for e in result.errors)


async def test_llm_failure_is_handled(monkeypatch):
    monkeypatch.setattr(gen, "validate_spec", _ok)
    spec, result = await gen.generate_strategy(
        [{"role": "user", "content": "x"}],
        llm=FakeLLM([RuntimeError("boom")]), system_prompt="sys",
    )
    assert spec is None
    assert any(e.code == "llm_call_failed" for e in result.errors)
