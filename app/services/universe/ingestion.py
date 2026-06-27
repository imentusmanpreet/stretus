"""
app/services/universe/ingestion.py — populate ai_strategy.universe_membership (§4.4).

The membership store (Phase C) is empty until a backfill feeds it constituent history. This
module provides that ingestion: idempotent, resumable, and KB-free (Invariant 11). It does
NOT derive membership from the knowledge base — it records intervals supplied by an external
constituent source (a vendor file, a periodic index snapshot, or an ops seed).

Two entry points:
  * :func:`diff_snapshot` — the PURE incremental rule: given the currently-open members and a
    fresh snapshot at ``asof``, compute which intervals to OPEN (newly added) and which to
    CLOSE (departed). This is how periodic snapshots build point-in-time history. Testable
    with no DB.
  * :class:`MembershipIngestor` — applies that to the DB: ``upsert`` (idempotent insert of
    explicit intervals) and ``apply_snapshot`` (open new + close departed from one snapshot).

CLI (ops backfill): ``python -m app.services.universe.ingestion --universe NIFTY500
--symbols TCS,INFY,HDFCBANK [--asof 2024-01-01]`` seeds/extends a universe from a symbol list.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from .membership import MembershipRow, SqlMembershipStore, members_on

logger = logging.getLogger(__name__)


def diff_snapshot(
    current_open: set[str],
    new_members: set[str],
    *,
    universe_key: str,
    asof: datetime,
    source: str | None = None,
) -> tuple[list[MembershipRow], list[str]]:
    """PURE incremental membership diff for one snapshot at ``asof``.

    Returns ``(to_open, to_close)``:
      * ``to_open`` — new ``MembershipRow``s (``valid_from=asof``, ``valid_to=None``) for
        symbols that are in the snapshot but not currently open;
      * ``to_close`` — symbols currently open but absent from the snapshot, whose open
        interval should get ``valid_to=asof`` (they left the universe).
    Symbols present in both are untouched (their open interval continues). Deterministic
    (sorted) so re-running the same snapshot is a no-op.
    """
    to_open = [
        MembershipRow(universe_key=universe_key, symbol=s, valid_from=asof,
                      valid_to=None, source=source)
        for s in sorted(new_members - current_open)
    ]
    to_close = sorted(current_open - new_members)
    return to_open, to_close


class MembershipIngestor:
    """Writes membership intervals to ``ai_strategy.universe_membership`` (idempotent).

    Takes an async ``session_factory`` (e.g. ``app.db.session.AsyncSessionLocal``). The DB
    layer is imported lazily so importing this package stays light (no import cycle).
    """

    def __init__(self, session_factory, *, tenant_id: str | None = None) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id
        self._store = SqlMembershipStore(session_factory)

    async def upsert(self, rows: list[MembershipRow], *, source: str | None = None) -> int:
        """Idempotently insert ``rows``. Re-inserting the same interval is a no-op.

        Conflict key is ``(tenant_id, universe_key, symbol, valid_from)`` — on conflict we
        refresh ``valid_to``/``source`` (so closing an interval later is an update, not a
        duplicate). Returns the number of rows written.
        """
        if not rows:
            return 0
        from sqlalchemy.dialects.postgresql import insert  # late import

        from app.db.models.universe import UniverseMembership

        values = [
            {
                "tenant_id": self._tenant_id,
                "universe_key": r.universe_key,
                "symbol": r.symbol,
                "valid_from": r.valid_from,
                "valid_to": r.valid_to,
                "source": r.source or source,
            }
            for r in rows
        ]
        stmt = insert(UniverseMembership).values(values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_universe_membership_interval",
            set_={"valid_to": stmt.excluded.valid_to, "source": stmt.excluded.source},
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()
        logger.info("ingestion|upsert|rows=%d", len(values))
        return len(values)

    async def _close_intervals(
        self, universe_key: str, symbols: list[str], asof: datetime
    ) -> int:
        """Set ``valid_to=asof`` on the open interval of each departed symbol."""
        if not symbols:
            return 0
        from sqlalchemy import and_, update  # late import

        from app.db.models.universe import UniverseMembership

        stmt = (
            update(UniverseMembership)
            .where(and_(
                UniverseMembership.universe_key == universe_key,
                UniverseMembership.symbol.in_(symbols),
                UniverseMembership.valid_to.is_(None),
            ))
            .values(valid_to=asof)
        )
        if self._tenant_id is not None:
            stmt = stmt.where(UniverseMembership.tenant_id == self._tenant_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            await session.commit()
        logger.info("ingestion|close|universe=%s|closed=%d", universe_key, len(symbols))
        return result.rowcount or 0

    async def apply_snapshot(
        self,
        universe_key: str,
        members: list[str],
        *,
        asof: datetime,
        source: str | None = None,
    ) -> dict[str, int]:
        """Apply one constituent snapshot: open newly-added intervals, close departed ones.

        Idempotent: re-applying the same snapshot at the same ``asof`` opens/closes nothing.
        This is the incremental backfill primitive — feed it periodic snapshots in date
        order to build point-in-time history (Invariant 7).
        """
        rows = await self._store.fetch_intervals(universe_key, tenant_id=self._tenant_id)
        current_open = {r.symbol for r in rows if r.valid_to is None}
        to_open, to_close = diff_snapshot(
            current_open, set(members), universe_key=universe_key, asof=asof, source=source,
        )
        opened = await self.upsert(to_open, source=source)
        closed = await self._close_intervals(universe_key, to_close, asof)
        logger.info(
            "ingestion|apply_snapshot|universe=%s|asof=%s|opened=%d|closed=%d|members=%d",
            universe_key, asof.isoformat(), opened, closed, len(members),
        )
        return {"opened": opened, "closed": closed}


async def _cli_main(args: argparse.Namespace) -> None:
    from app.db.session import AsyncSessionLocal  # late import (CLI only)

    asof = (
        datetime.fromisoformat(args.asof).replace(tzinfo=timezone.utc)
        if args.asof else datetime.now(timezone.utc)
    )
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    ingestor = MembershipIngestor(AsyncSessionLocal, tenant_id=args.tenant_id)
    counts = await ingestor.apply_snapshot(
        args.universe, symbols, asof=asof, source=args.source,
    )
    logger.info("ingestion|cli_done|universe=%s|%s", args.universe, counts)
    print(f"universe={args.universe} asof={asof.isoformat()} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed/extend universe_membership (KB-free).")
    parser.add_argument("--universe", required=True, help="universe_key, e.g. NIFTY500")
    parser.add_argument("--symbols", required=True, help="comma-separated member symbols")
    parser.add_argument("--asof", default=None, help="ISO date for this snapshot (default now)")
    parser.add_argument("--source", default="cli_seed", help="provenance label")
    parser.add_argument("--tenant-id", default=None, dest="tenant_id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_cli_main(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
