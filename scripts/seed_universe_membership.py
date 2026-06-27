#!/usr/bin/env python
"""
scripts/seed_universe_membership.py
───────────────────────────────────
Seed ``ai_strategy.universe_membership`` from the existing KB stock universe
(``app/kb/stocks/universe.csv``) so dynamic-universe INDEX/SECTOR strategies resolve to
real members immediately — without waiting for a point-in-time constituent feed.

This is an OPS / ingestion tool, NOT part of the dynamic-universe runtime: it lives OUTSIDE
``app/services/universe/`` precisely so it may read the KB while the resolver stays KB-free
(Invariant 11). The resolver only ever reads the table this script populates.

Each enabled equity is recorded as a CURRENT member (``valid_from`` in the past, ``valid_to``
NULL) under several keys so common prompts match:
  * ``NIFTY500`` and ``NIFTY50``  — "top N NIFTY 500 / NIFTY 50 stocks"
  * its sector (e.g. ``BANKING``) — "strongest banking stocks"

Re-running is idempotent (upsert on the unique interval). Caveat: this uses TODAY's list for
all history, so backtests seeded this way are ``survivorship_mode="approximate"`` until a real
point-in-time constituent feed replaces it.

Run:  docker exec stretus_api python scripts/seed_universe_membership.py
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.db.session import AsyncSessionLocal
from app.kb import kb
from app.services.universe.ingestion import MembershipIngestor
from app.services.universe.membership import MembershipRow, normalize_universe_key

# History start used for every seeded interval (well before any backtest window).
_VALID_FROM = datetime(2020, 1, 1, tzinfo=timezone.utc)
# Broad index keys every enabled equity is filed under.
_INDEX_KEYS = ("NIFTY500", "NIFTY50", "NIFTY")


def _enabled_equities() -> list:
    """Enabled equities from the KB (``.NS``/``.BO`` symbols; excludes crypto pairs)."""
    out = []
    for s in kb.stocks.values():
        if not getattr(s, "enabled", False):
            continue
        sym = str(getattr(s, "symbol", "") or "")
        if sym.endswith((".NS", ".BO")):
            out.append(s)
    return out


async def main() -> None:
    equities = _enabled_equities()
    if not equities:
        print("No enabled equities found in the KB — nothing to seed.")
        return

    rows: list[MembershipRow] = []
    for key in _INDEX_KEYS:
        nk = normalize_universe_key(key)
        for s in equities:
            rows.append(MembershipRow(
                universe_key=nk, symbol=s.symbol, valid_from=_VALID_FROM,
                valid_to=None, source="kb_seed",
            ))
    sectors: set[str] = set()
    for s in equities:
        sector = str(getattr(s, "sector", "") or "").strip()
        if sector:
            sectors.add(sector)
            rows.append(MembershipRow(
                universe_key=normalize_universe_key(sector), symbol=s.symbol,
                valid_from=_VALID_FROM, valid_to=None, source="kb_seed",
            ))

    ingestor = MembershipIngestor(AsyncSessionLocal)
    n = await ingestor.upsert(rows, source="kb_seed")
    print(
        f"✅ Seeded {n} membership rows from {len(equities)} equities | "
        f"index keys={_INDEX_KEYS} | sectors={sorted(sectors)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
