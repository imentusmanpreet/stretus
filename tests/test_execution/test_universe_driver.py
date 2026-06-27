"""Phase D — refresh orchestration gating + process driver (§9).

The calendar gate (cadence-not-due / market-closed) short-circuits BEFORE any DB access, so it
is testable with ``db=None``. The scheduler drivers (Manual / APScheduler) are covered for
registration/fire and the apscheduler-absent error path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.services.execution.market_calendar import IST
from app.services.execution.universe_driver import (
    APSchedulerDriver,
    ManualDriver,
    run_universe_refresh,
)
from app.services.execution.universe_scheduler import InMemoryAdvisoryLock
from app.strategy.spec import (
    UniverseRank,
    UniverseRefresh,
    UniverseSource,
    UniverseSpec,
)

UTC = timezone.utc


def _equity_universe() -> UniverseSpec:
    return UniverseSpec(
        source=UniverseSource(kind="index", name="NIFTY500"),
        rank=UniverseRank(by="volume"),
        refresh=UniverseRefresh(cadence="daily", at="10:15"),
    )


def _crypto_universe() -> UniverseSpec:
    return UniverseSpec(
        source=UniverseSource(kind="crypto_all"),
        rank=UniverseRank(by="volume"),
        refresh=UniverseRefresh(cadence="daily", at="00:00"),
    )


async def test_refresh_skips_when_market_closed_without_touching_db():
    InMemoryAdvisoryLock._held.clear()
    # Saturday 11:00 IST → NSE closed. should_fire is True (never fired) so we reach the
    # market-open gate, which short-circuits before any DB use (db=None proves it).
    asof = datetime(2025, 6, 21, 11, 0, tzinfo=IST)
    result = await run_universe_refresh(
        db=None, deployment_id="dep1", universe=_equity_universe(),
        asof=asof, lock=InMemoryAdvisoryLock(key=101), last_fired_at=None,
    )
    assert result.fired is False


async def test_refresh_skips_when_cadence_not_due():
    InMemoryAdvisoryLock._held.clear()
    # Crypto is always open, so only the cadence gate can stop it. Fired the 00:00 slot already;
    # asof is later the same day before the next slot → not due.
    last = datetime(2025, 6, 20, 0, 0, tzinfo=UTC)
    asof = datetime(2025, 6, 20, 6, 0, tzinfo=UTC)
    result = await run_universe_refresh(
        db=None, deployment_id="dep1", universe=_crypto_universe(),
        asof=asof, lock=InMemoryAdvisoryLock(key=102), last_fired_at=last,
    )
    assert result.fired is False


async def test_manual_driver_registers_and_fires():
    fired = []
    driver = ManualDriver()

    async def job():
        fired.append(True)

    driver.add_deployment("dep1", cadence="daily", job=job)
    await driver.fire("dep1")
    assert fired == [True]
    driver.remove_deployment("dep1")
    await driver.fire("dep1")          # removed → no-op
    assert fired == [True]


def test_apscheduler_driver_raises_clear_error_when_missing():
    pytest.importorskip  # noqa: B018 — keep import side-effect-free
    try:
        import apscheduler  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="apscheduler"):
            APSchedulerDriver()
    else:
        pytest.skip("apscheduler is installed; the missing-dep path is not exercised")
