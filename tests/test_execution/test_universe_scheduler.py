"""Phase D — universe scheduler core (§9): spawn/retire diff, advisory lock, reconciliation.

All pure / injectable — no APScheduler, DB, or broker. Covers the membership diff, the
single-fire advisory lock, reconciliation rebuild, and one full tick over injected deps.
"""
from __future__ import annotations

from app.services.execution.universe_scheduler import (
    InMemoryAdvisoryLock,
    UniverseRefreshResult,
    diff_membership,
    reconcile_state,
    run_refresh_tick,
)


def test_diff_membership_spawn_retire_unchanged():
    diff = diff_membership(active={"A", "B", "C"}, resolved=["B", "C", "D"])
    assert diff.to_spawn == ["D"]
    assert diff.to_retire == ["A"]
    assert diff.unchanged == ["B", "C"]
    assert diff.is_noop is False


def test_diff_membership_noop_when_identical():
    diff = diff_membership(active={"A", "B"}, resolved=["A", "B"])
    assert diff.is_noop is True


async def test_advisory_lock_single_fire():
    InMemoryAdvisoryLock._held.clear()
    a = InMemoryAdvisoryLock(key=42)
    b = InMemoryAdvisoryLock(key=42)
    assert await a.try_acquire() is True
    assert await b.try_acquire() is False   # second replica blocked
    await a.release()
    assert await b.try_acquire() is True     # freed → now available
    await b.release()


def test_reconcile_state_trusts_broker_for_positions_db_for_membership():
    rows = [
        {"symbol": "A", "status": "active", "allocated_capital": 30_000.0},
        {"symbol": "B", "status": "active", "allocated_capital": 20_000.0},
        {"symbol": "C", "status": "retiring", "allocated_capital": 10_000.0},
    ]
    # Broker actually holds A and C (B was filled/closed while we were down).
    broker = {"A": 30_000.0, "C": 10_000.0}
    state = reconcile_state(rows, broker)
    assert state.active_symbols == ["A", "B"]
    assert state.retiring_symbols == ["C"]
    # Only broker-confirmed positions are restored; B is NOT resurrected.
    assert set(state.open_positions) == {"A", "C"}


async def test_run_refresh_tick_fires_and_diffs():
    InMemoryAdvisoryLock._held.clear()
    spawned, retired = [], []

    class _Resolved:
        member_symbols = ["B", "C", "D"]
        snapshot_hash = "deadbeef0000"

    async def resolve():
        return _Resolved()

    async def on_spawn(symbols, snapshot):
        spawned.extend(symbols)

    async def on_retire(symbols, snapshot):
        retired.extend(symbols)

    result = await run_refresh_tick(
        deployment_id="dep1", active_members={"A", "B", "C"},
        resolve=resolve, lock=InMemoryAdvisoryLock(key=1),
        on_spawn=on_spawn, on_retire=on_retire,
    )
    assert result.fired is True
    assert result.spawned == ["D"] and result.retired == ["A"]
    assert spawned == ["D"] and retired == ["A"]


async def test_run_refresh_tick_skips_when_lock_held():
    InMemoryAdvisoryLock._held.clear()
    holder = InMemoryAdvisoryLock(key=7)
    await holder.try_acquire()   # another replica already holds it

    async def resolve():  # should never be called
        raise AssertionError("resolve must not run when the lock is held")

    async def noop(symbols, snapshot):
        pass

    result = await run_refresh_tick(
        deployment_id="dep1", active_members=set(),
        resolve=resolve, lock=InMemoryAdvisoryLock(key=7),
        on_spawn=noop, on_retire=noop,
    )
    assert result.fired is False
    await holder.release()
