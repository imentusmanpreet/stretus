"""Verify the legacy_bridge produces a dict shape compatible with
builder.apply_signal_plan and the existing chat response composers."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.planner.intent_extractor import IntentExtractor
from app.planner.legacy_bridge import plan_signals_v2
from app.planner.pipeline import Pipeline


class StubLLM:
    def __init__(self, payload):
        self._payload = payload

    async def chat(self, messages):
        return json.dumps(self._payload)


def _builder(**overrides):
    base = dict(
        symbol="hdfc", timeframe="1m", sentiment="bullish",
        experience="beginner", objective="positional",
        goal="short profit and quick trades",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def stub_pipeline():
    return Pipeline(intent_extractor=IntentExtractor(llm_service=StubLLM({
        "hold_horizon": "minutes", "frequency": "high",
        "profit_size": "small", "style": "scalping",
        "risk_appetite": "conservative",
    })))


async def test_legacy_bridge_returns_legacy_signal_plan_shape(stub_pipeline):
    plan = await plan_signals_v2(_builder(), pipeline=stub_pipeline)

    # Required keys for builder.apply_signal_plan
    assert "entry" in plan
    assert "exit" in plan
    assert "entry_condition" in plan
    assert "exit_condition" in plan
    assert "signals_used" in plan
    assert "signals_available" in plan
    assert plan["signals_available"] >= 25

    # entry/exit are lists of dicts with name/params/timeframe/signal_type
    for item in plan["entry"]:
        assert {"name", "params", "timeframe", "signal_type"} <= set(item.keys())
    for item in plan["exit"]:
        assert {"name", "params", "timeframe", "signal_type"} <= set(item.keys())

    # Direction sanity (smoke check on top of validator)
    assert plan["entry"][0]["name"] in {c.name for c in __import__("app.kb", fromlist=["kb"]).kb.signals.values()}


async def test_legacy_bridge_carries_planner_metadata(stub_pipeline):
    plan = await plan_signals_v2(_builder(), pipeline=stub_pipeline)
    assert "_planner_trace" in plan
    assert isinstance(plan["_planner_trace"], dict)
    assert "_sl_pct" in plan and plan["_sl_pct"] > 0
    assert "_tp_pct" in plan and plan["_tp_pct"] >= plan["_sl_pct"]


async def test_legacy_bridge_renders_formulas_with_params(stub_pipeline):
    """Regression: condition strings must not contain unrendered {param} placeholders."""
    plan = await plan_signals_v2(_builder(), pipeline=stub_pipeline)
    for label in ("entry_condition", "exit_condition"):
        cond = plan.get(label) or ""
        assert "{" not in cond and "}" not in cond, (
            f"{label} contains unrendered placeholders: {cond!r}"
        )


async def test_new_stocks_resolve_via_aliases(stub_pipeline):
    """The 4 newly added stocks must resolve through aliases and produce a plan."""
    for query, expected in [
        ("suzlon", "SUZLON.NS"),
        ("nhpc", "NHPC.NS"),
        ("gmr", "GMRAIRPORT.NS"),
        ("vodafone", "IDEA.NS"),
    ]:
        plan = await plan_signals_v2(
            _builder(symbol=query, timeframe="15m"),
            pipeline=stub_pipeline,
        )
        assert plan.get("entry"), f"{query!r} produced empty entry"
        # The strategy_assembler picks up symbol from builder; what we want here
        # is that the bridge ran end-to-end without raising UnsupportedStock.
