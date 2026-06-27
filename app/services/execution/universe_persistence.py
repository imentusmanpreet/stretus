"""
app/services/execution/universe_persistence.py — DB side-effects for the universe scheduler.

The scheduler's *decision* logic (diff, reconcile, lock) lives in ``universe_scheduler.py`` and
is pure. This module is the thin, async DB edge that those decisions drive:

  * :func:`persist_snapshot`      — append one ``universe_snapshots`` audit row per resolution
                                    (the immutable "why does membership look like this", §14).
  * :func:`load_active_members`   — the current active/retiring rows for reconciliation/fan-out.
  * :func:`apply_membership_diff` — upsert spawned members → ``active`` and transition retired
                                    ones → ``retiring`` (hold) / ``retired`` (flatten), per the
                                    spec's ``removal_policy``.
  * :func:`set_member_allocation` — stamp the capital a Portfolio-Manager admission locked.

These live under ``execution/`` (NOT ``services/universe/``) so the KB-free invariant on the
universe package is untouched: this module is allowed to import ORM models and a session.
Each function takes an :class:`AsyncSession` and never commits — the caller owns the
transaction boundary (so a whole tick is one atomic unit).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.universe import UniverseMember, UniverseSnapshot

logger = logging.getLogger(__name__)


def _as_uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _parse_asof(asof: Any) -> datetime:
    """ResolvedUniverse.asof is an ISO-8601 string (trailing Z); the column is a timestamptz."""
    if isinstance(asof, datetime):
        return asof
    s = str(asof).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def persist_snapshot(
    db: AsyncSession,
    *,
    deployment_id: Any,
    resolved: Any,
    tenant_id: Any | None = None,
) -> UniverseSnapshot:
    """Append one audit row for ``resolved`` (a :class:`ResolvedUniverse`). Idempotent on
    ``(deployment_id, asof, snapshot_hash)`` — a retried tick that re-resolves the same instant
    to the same membership reuses the existing row instead of duplicating it (§13)."""
    dep = _as_uuid(deployment_id)
    s_hash = getattr(resolved, "snapshot_hash", None) or None
    asof_dt = _parse_asof(getattr(resolved, "asof", None))

    existing = (
        await db.execute(
            select(UniverseSnapshot).where(
                UniverseSnapshot.deployment_id == dep,
                UniverseSnapshot.asof == asof_dt,
                UniverseSnapshot.snapshot_hash == s_hash,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        logger.info(
            "🧾 universe_snapshot | deployment=%s | asof=%s hash=%s — already recorded, reusing (idempotent)",
            dep, asof_dt.isoformat(), (s_hash or "")[:12],
        )
        return existing

    funnel = {
        "pool_size": getattr(resolved, "pool_size", 0),
        "screened_count": getattr(resolved, "screened_count", 0),
        "eligible_count": getattr(resolved, "eligible_count", 0),
        "members": len(getattr(resolved, "member_symbols", []) or []),
        "dropped": len(getattr(resolved, "dropped", []) or []),
        "requested_take": getattr(resolved, "requested_take", None),
        "effective_take": getattr(resolved, "effective_take", None),
        "breadth_ok": getattr(resolved, "breadth_ok", True),
        "breadth_reason": getattr(resolved, "breadth_reason", "") or "",
    }
    row = UniverseSnapshot(
        deployment_id=dep,
        tenant_id=_as_uuid(tenant_id) if tenant_id is not None else None,
        asof=asof_dt,
        members=list(getattr(resolved, "member_symbols", []) or []),
        snapshot_hash=s_hash,
        rule_hash=getattr(resolved, "rule_hash", None) or None,
        funnel=funnel,
        cap_reason=getattr(resolved, "cap_reason", None) or None,
        survivorship_mode=getattr(resolved, "survivorship_mode", None) or None,
    )
    db.add(row)
    await db.flush()
    logger.info(
        "🧾 SNAPSHOT SAVED | deployment=%s | asof=%s | hash=%s\n"
        "   funnel : pool=%d → screened=%d → eligible=%d → members=%d  (dropped=%d)\n"
        "   members: %s%s",
        dep, asof_dt.isoformat(), (s_hash or "")[:12],
        funnel["pool_size"], funnel["screened_count"], funnel["eligible_count"],
        funnel["members"], funnel["dropped"], row.members,
        "" if funnel["breadth_ok"] else f"  | ⚠️ breadth_blocked: {funnel['breadth_reason']}",
    )
    return row


async def load_active_members(
    db: AsyncSession, *, deployment_id: Any, include_retiring: bool = True,
) -> list[UniverseMember]:
    """The deployment's live member rows that still hold/track a position (active[+retiring])."""
    dep = _as_uuid(deployment_id)
    statuses = ["active", "retiring"] if include_retiring else ["active"]
    rows = (
        await db.execute(
            select(UniverseMember).where(
                UniverseMember.deployment_id == dep,
                UniverseMember.status.in_(statuses),
            )
        )
    ).scalars().all()
    return list(rows)


async def apply_membership_diff(
    db: AsyncSession,
    *,
    deployment_id: Any,
    to_spawn: list[str],
    to_retire: list[str],
    snapshot_hash: str | None,
    removal_policy: str = "flatten",
    tenant_id: Any | None = None,
) -> dict[str, list[str]]:
    """Apply a membership diff to ``universe_members`` (no commit).

    Spawned symbols are upserted to ``active`` (a previously-retired row is reactivated, not
    duplicated — the ``(deployment, symbol)`` unique key). Retired symbols transition to
    ``retiring`` when the spec holds open positions until their own exit
    (``removal_policy='hold_until_exit'``) or straight to ``retired`` + flat when the policy is
    ``flatten``. Returns ``{"spawned", "retiring", "retired"}`` for logging/audit.
    """
    dep = _as_uuid(deployment_id)
    tid = _as_uuid(tenant_id) if tenant_id is not None else None
    now = datetime.now(timezone.utc)
    spawned: list[str] = []
    retiring: list[str] = []
    retired: list[str] = []

    existing_rows = (
        await db.execute(select(UniverseMember).where(UniverseMember.deployment_id == dep))
    ).scalars().all()
    by_symbol = {r.symbol: r for r in existing_rows}

    for sym in to_spawn:
        row = by_symbol.get(sym)
        if row is None:
            row = UniverseMember(
                deployment_id=dep, tenant_id=tid, symbol=sym, status="active",
                snapshot_hash=snapshot_hash, activated_at=now,
            )
            db.add(row)
            by_symbol[sym] = row
        else:
            row.status = "active"
            row.snapshot_hash = snapshot_hash
            row.retired_at = None
            row.activated_at = row.activated_at or now
        spawned.append(sym)

    hold = str(removal_policy or "flatten").strip().lower() == "hold_until_exit"
    for sym in to_retire:
        row = by_symbol.get(sym)
        if row is None:
            continue
        if hold and float(row.allocated_capital or 0) > 0:
            row.status = "retiring"   # keep tracking until its own exit squares it off
            retiring.append(sym)
        else:
            row.status = "retired"
            row.allocated_capital = 0
            row.retired_at = now
            retired.append(sym)

    await db.flush()
    logger.info(
        "🔁 MEMBERS UPDATED | deployment=%s | policy=%s | ➕ spawned=%s | ⏳ retiring=%s | ➖ retired=%s",
        dep, removal_policy, spawned or "—", retiring or "—", retired or "—",
    )
    return {"spawned": spawned, "retiring": retiring, "retired": retired}


async def set_member_allocation(
    db: AsyncSession, *, deployment_id: Any, symbol: str, allocated_capital: float,
    member_state: dict | None = None,
) -> None:
    """Stamp the capital a Portfolio-Manager admission locked for ``symbol`` (no commit)."""
    dep = _as_uuid(deployment_id)
    row = (
        await db.execute(
            select(UniverseMember).where(
                UniverseMember.deployment_id == dep, UniverseMember.symbol == symbol,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return
    row.allocated_capital = float(allocated_capital)
    if member_state is not None:
        row.member_state = {**(row.member_state or {}), **member_state}
    await db.flush()
    logger.info("💰 ALLOCATION | deployment=%s | %s ← capital=%.2f%s", dep, symbol,
                float(allocated_capital), f" state+={list(member_state)}" if member_state else "")
