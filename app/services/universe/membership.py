"""
app/services/universe/membership.py — survivorship-safe point-in-time constituents (Phase C).

Resolves the provider-backed universe sources (index / sector / f_and_o) from the
``ai_strategy.universe_membership`` store as it stood at ``asof`` — so a dynamic backtest is
survivorship-safe (Invariant 7): a since-delisted name appears in a historical resolution and
is absent today. This is the KB-free replacement for ``kb.stocks`` (Invariant 11): the pool
comes from recorded constituent history, never the knowledge base.

Design (one rule, no DB needed to test it):
  * :func:`members_on` is the PURE point-in-time membership rule over a list of intervals —
    ``valid_from <= asof AND (valid_to IS NULL OR valid_to > asof)`` — trivially unit-testable.
  * :class:`MembershipStore` is the injectable seam that yields a universe's intervals
    (``SqlMembershipStore`` in prod, ``InMemoryMembershipStore`` in tests — no network).
  * :class:`MembershipPoolProvider` is the resolver's ``pool_provider``: it maps a
    ``UniverseSource`` to its ``universe_key``, asks the store for that key's intervals, and
    applies :func:`members_on`. It advertises ``point_in_time = True`` so the resolver can
    stamp ``survivorship_mode="point_in_time"`` automatically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.strategy.spec import UniverseSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MembershipRow:
    """One membership interval of ``symbol`` in ``universe_key`` (half-open ``[from, to)``)."""

    universe_key: str
    symbol: str
    valid_from: datetime
    valid_to: datetime | None = None
    source: str | None = None


def normalize_universe_key(name: str) -> str:
    """Canonicalize a universe key so lookups are robust to user/LLM phrasing.

    ``"NIFTY 500"`` / ``"Nifty-500"`` / ``"nifty500"`` → ``"NIFTY500"``; ``"banking"`` →
    ``"BANKING"``; ``"f_and_o"`` → ``"F_AND_O"`` (underscores kept). Seeding and lookup BOTH
    run through this, so a seeded key always matches the resolved one regardless of spacing/case.
    """
    return (name or "").upper().replace(" ", "").replace("-", "").strip()


def source_universe_key(source: UniverseSource) -> str:
    """Map a ``UniverseSource`` to its normalized membership ``universe_key``.

    index/sector → the ``name`` (e.g. ``"NIFTY 500"`` → ``"NIFTY500"``); f_and_o →
    ``"F_AND_O"``. Watchlist/crypto sources are not membership-backed and never reach here.
    """
    return normalize_universe_key(source.name or source.kind)


def members_on(rows: list[MembershipRow], *, asof: datetime) -> list[str]:
    """The PURE point-in-time membership rule (Invariant 4 & 7).

    Returns the symbols whose interval contains ``asof`` —
    ``valid_from <= asof AND (valid_to IS NULL OR valid_to > asof)`` — deduped, order-stable.
    A name that left on date X is a member up to but NOT including X (half-open interval).
    """
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        if r.valid_from <= asof and (r.valid_to is None or r.valid_to > asof):
            if r.symbol not in seen:
                seen.add(r.symbol)
                out.append(r.symbol)
    return out


@runtime_checkable
class MembershipStore(Protocol):
    """Yields the membership intervals for a universe key (the store is swappable)."""

    async def fetch_intervals(
        self, universe_key: str, *, tenant_id: str | None = None
    ) -> list[MembershipRow]:
        ...


class InMemoryMembershipStore:
    """A :class:`MembershipStore` over an in-memory interval list — for tests (no DB)."""

    def __init__(self, rows: list[MembershipRow]) -> None:
        self._rows = list(rows)

    async def fetch_intervals(
        self, universe_key: str, *, tenant_id: str | None = None
    ) -> list[MembershipRow]:
        return [r for r in self._rows if r.universe_key == universe_key]


class SqlMembershipStore:
    """A :class:`MembershipStore` backed by ``ai_strategy.universe_membership``.

    Takes an async ``session_factory`` (e.g. ``app.db.session.AsyncSessionLocal``). The DB
    layer is imported lazily inside the query so importing this package stays light and
    free of heavy back-imports (no import cycle, §12.2).
    """

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def fetch_intervals(
        self, universe_key: str, *, tenant_id: str | None = None
    ) -> list[MembershipRow]:
        from sqlalchemy import select  # late import — keep package import-light

        from app.db.models.universe import UniverseMembership

        stmt = select(UniverseMembership).where(
            UniverseMembership.universe_key == universe_key
        )
        if tenant_id is not None:
            stmt = stmt.where(UniverseMembership.tenant_id == tenant_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            records = result.scalars().all()
        return [
            MembershipRow(
                universe_key=rec.universe_key,
                symbol=rec.symbol,
                valid_from=rec.valid_from,
                valid_to=rec.valid_to,
                source=rec.source,
            )
            for rec in records
        ]


class MembershipPoolProvider:
    """The resolver's KB-free, point-in-time ``pool_provider`` (§7 source registry).

    ``point_in_time = True`` lets :func:`resolve_universe` stamp the snapshot's
    ``survivorship_mode`` as ``"point_in_time"`` automatically (Invariant 7).
    """

    point_in_time: bool = True

    def __init__(self, store: MembershipStore, *, tenant_id: str | None = None) -> None:
        self._store = store
        self._tenant_id = tenant_id

    async def __call__(self, source: UniverseSource, asof: datetime) -> list[str]:
        key = source_universe_key(source)
        rows = await self._store.fetch_intervals(key, tenant_id=self._tenant_id)
        members = members_on(rows, asof=asof)
        logger.info(
            "membership|resolve|key=%s|asof=%s|intervals=%d|members=%d",
            key, asof.isoformat(), len(rows), len(members),
        )
        return members
