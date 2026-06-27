"""Regression guard — the dynamic-universe wiring must NOT change the static single-symbol path.

The dynamic deliverables are strictly additive (Invariant 10). These tests pin the static
contract so any accidental drift (a new field leaking onto the static schemas, the static
response shape changing, the dynamic endpoint firing when the flag is off) fails loudly.
"""
from __future__ import annotations

import pytest

from app.schemas.execution import EvaluateExecuteResponse, StrategyConfigPayload


def test_static_response_shape_unchanged():
    # The exact field set the OMS consumes from /evaluate/execute. Adding/removing a field here
    # is a breaking change to the static path and must be deliberate.
    assert set(EvaluateExecuteResponse.model_fields) == {
        "status", "action", "symbol", "ltp", "mode", "bracket_order", "multi_tp_bracket",
        "exits", "risk_snapshot", "entry_detail", "messages", "error",
    }


def test_static_strategy_config_has_no_universe_field():
    # The static inline config is symbol-only; the dynamic `universe` rule never leaked onto it
    # (dynamic deployments route through the sibling endpoint, not this schema).
    fields = set(StrategyConfigPayload.model_fields)
    assert "universe" not in fields
    assert "symbol" in fields


def test_dynamic_universe_disabled_by_default():
    # The field DEFAULT is False (so static deployments are never routed into the dynamic path
    # unless the platform explicitly opts in). Asserting the model default keeps this test
    # independent of any ambient .env that may enable the flag locally.
    from app.core.config import Settings

    assert Settings.model_fields["dynamic_universe_enabled"].default is False


async def test_dynamic_branch_rejects_when_flag_off(monkeypatch):
    # A request with a `universe` block routes into the dynamic branch, which guards on the
    # feature flag BEFORE any DB/market access — so a disabled platform returns 403 without
    # touching the static path's dependencies (db=None proves it). We force the flag OFF so the
    # test holds regardless of the ambient .env.
    from types import SimpleNamespace

    import app.core.config as cfg
    from fastapi import HTTPException

    from app.api.v1.routes.execution import evaluate_execute
    from app.schemas.execution import (
        EntryExitBlock,
        EvaluateExecuteRequest,
        ExitBlock,
        SignalRule,
        SlTpConfig,
        UniverseBlock,
    )

    monkeypatch.setattr(cfg, "get_settings", lambda: SimpleNamespace(dynamic_universe_enabled=False))

    req = EvaluateExecuteRequest(
        strategy_config=StrategyConfigPayload(
            symbol="PLACEHOLDER", timeframe="5m",
            entry=EntryExitBlock(trigger=SignalRule(type="ema_crossover")),
            exit=ExitBlock(), sl_tp=SlTpConfig(stop_loss_pct=2.0, take_profit_pct=4.0),
        ),
        universe=UniverseBlock(deployment_id="dep1", members=["AAA"]),
    )
    with pytest.raises(HTTPException) as exc:
        await evaluate_execute(req, db=None)   # db never touched when flag is off
    assert exc.value.status_code == 403


async def test_rule_in_request_resolves_universe_live(monkeypatch):
    # A `universe.rule` (no members) makes the endpoint resolve the universe LIVE at request
    # time and trade the result — the self-contained stateless dynamic flow. We stub the live
    # resolver + the fan-out so this needs no network/DB and asserts the wiring/precedence.
    from types import SimpleNamespace

    import app.core.config as cfg
    import app.services.execution.universe_driver as drv
    import app.services.execution.universe_evaluator as uev
    from app.api.v1.routes.execution import evaluate_execute
    from app.schemas.execution import (
        EntryExitBlock,
        EvaluateExecuteRequest,
        ExitBlock,
        SignalRule,
        SlTpConfig,
        UniverseBlock,
    )
    from app.schemas.universe_execution import UniverseEvaluateResponse
    from app.strategy.spec import UniverseRank, UniverseSource, UniverseSpec

    monkeypatch.setattr(cfg, "get_settings", lambda: SimpleNamespace(
        dynamic_universe_enabled=True, dynamic_universe_max_positions=2, dynamic_universe_max_assets=2))

    class _Resolved:
        member_symbols = ["BTC_USDT", "ETH_USDT"]   # what "top-2 most active" resolved to NOW
        snapshot_hash = "abc123def456"

    async def _fake_resolve(rule, *, asof, settings=None, run_id=None):
        return _Resolved()
    monkeypatch.setattr(drv, "resolve_live_universe", _fake_resolve)

    captured: dict = {}

    async def _fake_eval(db, *, deployment_id, base_config, member_symbols, max_positions, **kw):
        captured["members"] = member_symbols
        captured["max_positions"] = max_positions
        return UniverseEvaluateResponse(deployment_id=deployment_id, max_positions=max_positions)
    monkeypatch.setattr(uev, "evaluate_universe_deployment", _fake_eval)

    req = EvaluateExecuteRequest(
        strategy_config=StrategyConfigPayload(
            symbol="PLACEHOLDER", timeframe="5m",
            entry=EntryExitBlock(trigger=SignalRule(type="condition", params={"formula": "close > 0"})),
            exit=ExitBlock(), sl_tp=SlTpConfig(stop_loss_pct=2.0, take_profit_pct=4.0),
        ),
        universe=UniverseBlock(
            deployment_id="test-1",
            rule=UniverseSpec(source=UniverseSource(kind="crypto_all"), rank=UniverseRank(by="rvol"), take=2),
            max_positions=5,   # clamped to the platform ceiling (2)
        ),
    )
    resp = await evaluate_execute(req, db=None)   # db untouched — rule path skips the table
    assert captured["members"] == ["BTC_USDT", "ETH_USDT"]   # resolved live, not from a table
    assert captured["max_positions"] == 2                    # clamped to DYNAMIC_UNIVERSE_MAX_POSITIONS
    assert resp.deployment_id == "test-1"


def test_universe_block_lives_on_request_not_on_strategy_config():
    # The dynamic inputs sit on the request envelope; the per-symbol StrategyConfigPayload stays
    # single-symbol (its regression above still holds).
    from app.schemas.execution import EvaluateExecuteRequest

    assert "universe" in EvaluateExecuteRequest.model_fields
    assert "universe" not in StrategyConfigPayload.model_fields
