"""Stateless OMS-driven live dynamic path: clock-quantized resolution + cache + fail-closed.

These pin the production design: membership is resolved as-of the refresh WINDOW (stable within
it, re-picks at the boundary), resolution is memoized per window, and a live request without OMS
state fails closed.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import app.services.execution.live_universe as lu
from app.strategy.spec import UniverseRank, UniverseRefresh, UniverseSource, UniverseSpec

UTC = timezone.utc


def _rule() -> UniverseSpec:
    # daily @ 09:20 IST → one re-pick window per trading day.
    return UniverseSpec(
        source=UniverseSource(kind="crypto_all"),
        rank=UniverseRank(by="rvol", order="desc"),
        take=2,
        refresh=UniverseRefresh(cadence="daily", at="00:00"),  # crypto → UTC midnight boundary
    )


class _Resolved:
    def __init__(self, members):
        self.member_symbols = members
        self.snapshot_hash = "hash" + "".join(members)


async def test_resolution_is_cached_within_a_window_and_repicks_across_windows(monkeypatch):
    lu.clear_cache()
    calls: list[str] = []

    async def _fake_resolve(rule, *, asof, settings=None, run_id=None):
        calls.append(asof.isoformat())
        return _Resolved(["BTC_USDT", "ETH_USDT"])

    monkeypatch.setattr(lu, "resolve_live_universe", _fake_resolve, raising=False)
    # Patch the late-imported symbol too (resolve_universe_windowed imports inside the function).
    import app.services.execution.universe_driver as drv
    monkeypatch.setattr(drv, "resolve_live_universe", _fake_resolve)

    rule = _rule()
    # Two calls on the SAME day (same crypto UTC window) → resolver invoked once.
    await lu.resolve_universe_windowed(rule, now=datetime(2026, 6, 20, 5, 0, tzinfo=UTC))
    await lu.resolve_universe_windowed(rule, now=datetime(2026, 6, 20, 18, 30, tzinfo=UTC))
    assert len(calls) == 1, "same window must hit the cache, not re-resolve"

    # Next day → new window → resolver invoked again (automatic re-pick).
    await lu.resolve_universe_windowed(rule, now=datetime(2026, 6, 21, 2, 0, tzinfo=UTC))
    assert len(calls) == 2, "a new refresh window must re-resolve"


async def test_cache_key_is_per_rule(monkeypatch):
    lu.clear_cache()
    calls = []

    async def _fake_resolve(rule, *, asof, settings=None, run_id=None):
        calls.append(rule.take)
        return _Resolved(["BTC_USDT"])

    import app.services.execution.universe_driver as drv
    monkeypatch.setattr(drv, "resolve_live_universe", _fake_resolve)

    r1 = _rule()
    r2 = _rule(); r2 = r2.model_copy(update={"take": 1})
    now = datetime(2026, 6, 20, 5, 0, tzinfo=UTC)
    await lu.resolve_universe_windowed(r1, now=now)
    await lu.resolve_universe_windowed(r2, now=now)   # different rule → different key
    assert len(calls) == 2


# ── live fail-closed when the OMS omits state ─────────────────────────────────
async def test_live_dynamic_fails_closed_without_oms_state(monkeypatch):
    from types import SimpleNamespace

    import app.core.config as cfg
    from app.api.v1.routes.execution import evaluate_execute
    from app.schemas.execution import (
        EntryExitBlock,
        EvaluateExecuteRequest,
        EvaluationMode,
        ExitBlock,
        SignalRule,
        SlTpConfig,
        StrategyConfigPayload,
        UniverseBlock,
    )

    monkeypatch.setattr(cfg, "get_settings", lambda: SimpleNamespace(
        dynamic_universe_enabled=True, dynamic_universe_max_positions=2, dynamic_universe_max_assets=2))

    req = EvaluateExecuteRequest(
        mode=EvaluationMode.live,
        strategy_config=StrategyConfigPayload(
            symbol="PLACEHOLDER", timeframe="5m", asset_class="crypto_spot",
            entry=EntryExitBlock(trigger=SignalRule(type="condition", params={"formula": "close > 0"})),
            exit=ExitBlock(), sl_tp=SlTpConfig(stop_loss_pct=2.0, take_profit_pct=4.0),
        ),
        universe=UniverseBlock(
            deployment_id="dep-live-1",
            rule=UniverseSpec(source=UniverseSource(kind="crypto_all"), rank=UniverseRank(by="rvol"), take=2),
            # NO total_capital, NO open_positions → must fail closed in live mode.
        ),
    )
    resp = await evaluate_execute(req, db=None)   # db untouched — fails before resolution
    assert resp.status == "error"
    assert "total_capital" in resp.error and "open_positions" in resp.error


async def test_paper_dynamic_defaults_state_when_omitted(monkeypatch):
    # Paper may default capital/positions (parity/testing); only the resolution is stubbed.
    from types import SimpleNamespace

    import app.core.config as cfg
    import app.services.execution.live_universe as live_uni
    import app.services.execution.universe_evaluator as uev
    from app.api.v1.routes.execution import evaluate_execute
    from app.schemas.execution import (
        EntryExitBlock,
        EvaluateExecuteRequest,
        EvaluationMode,
        ExitBlock,
        SignalRule,
        SlTpConfig,
        StrategyConfigPayload,
        UniverseBlock,
    )
    from app.schemas.universe_execution import UniverseEvaluateResponse

    monkeypatch.setattr(cfg, "get_settings", lambda: SimpleNamespace(
        dynamic_universe_enabled=True, dynamic_universe_max_positions=2, dynamic_universe_max_assets=2))

    async def _fake_windowed(rule, *, now, settings=None, run_id=None):
        return _Resolved(["BTC_USDT", "ETH_USDT"])
    monkeypatch.setattr(live_uni, "resolve_universe_windowed", _fake_windowed)

    captured = {}
    async def _fake_eval(db, *, deployment_id, base_config, member_symbols, total_capital, **kw):
        captured["members"] = member_symbols
        captured["total_capital"] = total_capital
        return UniverseEvaluateResponse(deployment_id=deployment_id, max_positions=kw["max_positions"])
    monkeypatch.setattr(uev, "evaluate_universe_deployment", _fake_eval)

    req = EvaluateExecuteRequest(
        mode=EvaluationMode.paper,
        strategy_config=StrategyConfigPayload(
            symbol="PLACEHOLDER", timeframe="5m", asset_class="crypto_spot",
            entry=EntryExitBlock(trigger=SignalRule(type="condition", params={"formula": "close > 0"})),
            exit=ExitBlock(), sl_tp=SlTpConfig(stop_loss_pct=2.0, take_profit_pct=4.0),
        ),
        universe=UniverseBlock(
            deployment_id="dep-paper-1",
            rule=UniverseSpec(source=UniverseSource(kind="crypto_all"), rank=UniverseRank(by="rvol"), take=2),
        ),
    )
    resp = await evaluate_execute(req, db=None)
    assert resp.deployment_id == "dep-paper-1"
    assert captured["members"] == ["BTC_USDT", "ETH_USDT"]
    assert captured["total_capital"] == 100_000.0   # paper default
