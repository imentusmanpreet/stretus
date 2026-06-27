"""
app/services/execution/universe_scheduler.py — live periodic re-resolution + spawn/retire (§9).

A live dynamic deployment re-resolves its universe on a cadence (e.g. daily, one hour after
open). On each tick this module:
  1. acquires a **Postgres advisory lock** so exactly ONE replica fires the tick (no
     double-spawn under multi-replica — §9 step 1);
  2. resolves the universe on live data → a fresh snapshot;
  3. **diffs** the snapshot against the currently-active members → spawn / retire / unchanged;
  4. spawns new members (warm up, create sub-state) and retires departed ones (square off or
     hold-to-exit).

The load-bearing logic — the membership **diff**, the **reconciliation** rebuild, and the
advisory-lock contract — is pure / injectable so it is unit-testable with no scheduler, DB,
or broker. The thin APScheduler driver and the broker fan-out are wired at the edges (see the
design note); they call into the pure pieces here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MembershipDiff:
    """What changed between the active members and a fresh resolution (§9 steps 2–4)."""

    to_spawn: list[str]      # in the new snapshot, not currently active → activate
    to_retire: list[str]     # currently active, gone from the snapshot → square off / hold
    unchanged: list[str]     # still active, no action

    @property
    def is_noop(self) -> bool:
        return not self.to_spawn and not self.to_retire


def diff_membership(active: set[str], resolved: list[str]) -> MembershipDiff:
    """PURE spawn/retire diff. Deterministic (sorted) so a tick that changes nothing is a
    clean no-op — important for idempotency under retried/overlapping ticks (§13)."""
    resolved_set = set(resolved)
    return MembershipDiff(
        to_spawn=sorted(resolved_set - active),
        to_retire=sorted(active - resolved_set),
        unchanged=sorted(active & resolved_set),
    )


# ── Advisory lock (one replica fires each tick) ───────────────────────────────
@runtime_checkable
class AdvisoryLock(Protocol):
    """A best-effort, non-blocking lock. ``try_acquire`` returns False if held elsewhere."""

    async def try_acquire(self) -> bool: ...
    async def release(self) -> None: ...


class InMemoryAdvisoryLock:
    """Process-local advisory lock for tests / single-process dev (not multi-replica safe)."""

    _held: set[int] = set()

    def __init__(self, key: int) -> None:
        self._key = key
        self._acquired = False

    async def try_acquire(self) -> bool:
        if self._key in InMemoryAdvisoryLock._held:
            return False
        InMemoryAdvisoryLock._held.add(self._key)
        self._acquired = True
        return True

    async def release(self) -> None:
        if self._acquired:
            InMemoryAdvisoryLock._held.discard(self._key)
            self._acquired = False


class PostgresAdvisoryLock:
    """Multi-replica-safe advisory lock via Postgres ``pg_try_advisory_lock`` (§9 step 1).

    Session-level lock keyed by a stable 64-bit int (derive from the deployment id). Exactly
    one replica acquires it per tick; the others skip. The DB layer is imported lazily so this
    module stays import-light.
    """

    def __init__(self, session_factory, key: int) -> None:
        self._session_factory = session_factory
        self._key = key
        self._session = None

    async def try_acquire(self) -> bool:
        from sqlalchemy import text  # late import

        self._session = self._session_factory()
        await self._session.__aenter__()
        result = await self._session.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": self._key}
        )
        acquired = bool(result.scalar())
        if not acquired:
            await self._session.__aexit__(None, None, None)
            self._session = None
        return acquired

    async def release(self) -> None:
        if self._session is None:
            return
        from sqlalchemy import text  # late import

        try:
            await self._session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": self._key})
        finally:
            await self._session.__aexit__(None, None, None)
            self._session = None


# ── Reconciliation (restart/failover) ─────────────────────────────────────────
@dataclass(frozen=True)
class ReconciledState:
    """The live state rebuilt on restart, before resuming (§9 step 7)."""

    active_symbols: list[str]
    open_positions: dict[str, float]   # symbol → allocated capital (from broker/state)
    retiring_symbols: list[str]


def reconcile_state(
    member_rows: list[dict],
    broker_positions: dict[str, float],
) -> ReconciledState:
    """PURE reconciliation: rebuild active membership + open positions from persisted
    ``universe_members`` rows and the broker's reported positions.

    Each row is ``{"symbol", "status", "allocated_capital"}``. Active/retiring members come
    from the rows; open positions are the intersection with what the BROKER actually holds —
    so a position the broker no longer has (filled/cancelled while we were down) is not
    resurrected, and one the broker holds is restored. Fail-safe: trust the broker for
    positions, the DB for membership.
    """
    active = sorted(r["symbol"] for r in member_rows if r.get("status") == "active")
    retiring = sorted(r["symbol"] for r in member_rows if r.get("status") == "retiring")
    row_capital = {r["symbol"]: float(r.get("allocated_capital") or 0.0) for r in member_rows}
    open_positions = {
        sym: (row_capital.get(sym) or qty_capital)
        for sym, qty_capital in broker_positions.items()
    }
    return ReconciledState(
        active_symbols=active, open_positions=open_positions, retiring_symbols=retiring,
    )


# ── The tick (orchestration over injectable pieces) ───────────────────────────
@dataclass
class UniverseRefreshResult:
    """Outcome of one scheduler tick (for logging/metrics/tests)."""

    fired: bool                       # False ⇒ another replica held the lock
    diff: MembershipDiff | None = None
    snapshot_hash: str | None = None
    spawned: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)


async def run_refresh_tick(
    *,
    deployment_id: str,
    active_members: set[str],
    resolve,            # async () -> ResolvedUniverse (the live resolver call, injected)
    lock: AdvisoryLock,
    on_spawn,           # async (symbols, snapshot) -> None
    on_retire,          # async (symbols, snapshot) -> None
    on_resolved=None,   # async (snapshot) -> None — persist the per-resolution audit row (§14)
) -> UniverseRefreshResult:
    """One scheduler tick: lock → resolve → [audit] → diff → spawn/retire. Injectable deps, no
    globals.

    Returns immediately with ``fired=False`` if another replica holds the advisory lock
    (§9 step 1). Always releases the lock. ``on_resolved`` (optional) persists the snapshot
    audit row for EVERY fired resolution — even a no-op tick is recorded, so the audit trail is
    continuous. ``on_spawn``/``on_retire`` perform the membership side effects (warm-up +
    sub-state creation; square-off + status transition) through the live Portfolio Manager +
    OMS at the edges.
    """
    logger.info("🔒 LOCK | deployment=%s | acquiring single-resolver advisory lock…", deployment_id)
    if not await lock.try_acquire():
        logger.info("🔒 ⏭️  SKIP | deployment=%s | reason=lock_held (another replica is resolving)", deployment_id)
        return UniverseRefreshResult(fired=False)
    logger.info("🔒 ✅ LOCK ACQUIRED | deployment=%s — this replica owns the refresh", deployment_id)
    try:
        logger.info("🧭 RESOLVE | deployment=%s | resolving live membership…", deployment_id)
        resolved = await resolve()
        logger.info(
            "🧭 RESOLVED | deployment=%s | members=%d → %s | snapshot=%s",
            deployment_id, len(resolved.member_symbols), resolved.member_symbols,
            (resolved.snapshot_hash or "")[:12],
        )
        if on_resolved is not None:
            await on_resolved(resolved)

        diff = diff_membership(active_members, resolved.member_symbols)
        logger.info(
            "🔁 DIFF | deployment=%s | spawn=%s | retire=%s | unchanged=%d%s",
            deployment_id, diff.to_spawn or "—", diff.to_retire or "—", len(diff.unchanged),
            "  (no-op — membership steady)" if diff.is_noop else "",
        )
        if diff.to_spawn:
            logger.info("🔁 ➕ SPAWN | deployment=%s | admitting %s", deployment_id, diff.to_spawn)
            await on_spawn(diff.to_spawn, resolved)
        if diff.to_retire:
            logger.info("🔁 ➖ RETIRE | deployment=%s | dropping %s", deployment_id, diff.to_retire)
            await on_retire(diff.to_retire, resolved)
        logger.info(
            "🌌 ✅ REFRESH DONE | deployment=%s | spawned=%d retired=%d unchanged=%d | snapshot=%s",
            deployment_id, len(diff.to_spawn), len(diff.to_retire), len(diff.unchanged),
            (resolved.snapshot_hash or "")[:12],
        )
        return UniverseRefreshResult(
            fired=True, diff=diff, snapshot_hash=resolved.snapshot_hash,
            spawned=diff.to_spawn, retired=diff.to_retire,
        )
    finally:
        await lock.release()
        logger.debug("🔓 LOCK RELEASED | deployment=%s", deployment_id)
