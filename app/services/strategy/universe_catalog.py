"""
app/services/strategy/universe_catalog.py — the master supported-asset catalog.

``universe.csv`` (the KB stock universe) is the SINGLE source of truth for every asset the
platform supports — Indian equities + crypto pairs, each tagged with ``asset_class`` and
``sector``. This module reads that catalog and turns a user's *selection rule* into the
candidate **pool** the dynamic-universe resolver ranks over. Nothing is hardcoded or labeled:
the file IS the list, and a user term is just a FILTER over it.

  "top 10 crypto …"                  → asset_class = crypto_spot           (all supported coins)
  "NIFTY 500 / Indian stocks / F&O"  → asset_class = equity_cash           (all supported equities)
  "banking stocks"                   → equity_cash AND sector = banking
  explicit list (watchlist)          → handled by the resolver directly

This lives OUTSIDE ``app/services/universe/`` on purpose: the resolver package stays
import-clean (KB-free), and the pool is **injected** into it — the resolver only ever
receives "here are the candidates", it never reads the catalog itself.
"""
from __future__ import annotations

import logging
from datetime import datetime

from app.kb import kb
from app.strategy.spec import UniverseSource

logger = logging.getLogger(__name__)

_EQUITY = "equity_cash"
_CRYPTO = "crypto_spot"


def _dedupe_crypto(symbols: list[str]) -> list[str]:
    """Collapse multi-quote duplicates to one row per base coin (prefer ``_USDT``).

    universe.csv lists e.g. both ``BTC_USDC`` and ``BTC_USDT``; for ranking we want one
    entry per coin so the pool isn't doubled.
    """
    by_base: dict[str, str] = {}
    for s in symbols:
        base = s.rsplit("_", 1)[0]
        if base not in by_base or s.endswith("_USDT"):
            by_base[base] = s
    return list(by_base.values())


def supported_symbols(*, asset_class: str | None = None, sector: str | None = None) -> list[str]:
    """Enabled symbols from the catalog, optionally filtered by asset class and/or sector."""
    out: list[str] = []
    for st in kb.stocks.values():
        if not getattr(st, "enabled", False):
            continue
        if asset_class and getattr(st, "asset_class", None) != asset_class:
            continue
        if sector and (getattr(st, "sector", "") or "").lower() != sector.lower():
            continue
        out.append(st.symbol)
    return out


def known_sectors() -> set[str]:
    """Lower-cased set of sectors present in the catalog (for matching 'banking' etc.)."""
    return {
        (getattr(st, "sector", "") or "").lower()
        for st in kb.stocks.values()
        if getattr(st, "enabled", False)
    }


def crypto_supported_bases() -> set[str]:
    """Base coins we support (e.g. {'BTC','ETH',…}) — for filtering a live exchange feed."""
    return {s.rsplit("_", 1)[0] for s in supported_symbols(asset_class=_CRYPTO)}


def pool_for_source(source: UniverseSource) -> list[str]:
    """The candidate pool for ``source`` drawn from the master catalog.

    crypto_all → supported coins (deduped). index/f_and_o → all supported equities. sector →
    equities in that sector when the name matches a known sector, else all equities (so an
    unrecognised index/sector term still resolves to the broad supported set, never empty).
    """
    if source.kind == "crypto_all":
        return _dedupe_crypto(supported_symbols(asset_class=_CRYPTO))
    name = (source.name or "").strip().lower()
    if source.kind == "sector" and name in known_sectors():
        return supported_symbols(asset_class=_EQUITY, sector=name)
    return supported_symbols(asset_class=_EQUITY)


def make_kb_pool_provider():
    """An injectable resolver ``pool_provider`` backed by the master catalog.

    Carries ``point_in_time = False`` so the resolver honestly stamps snapshots
    ``survivorship_mode="approximate"`` — universe.csv is *today's* supported list, not a
    point-in-time constituent history.
    """
    async def provider(source: UniverseSource, asof: datetime) -> list[str]:
        pool = pool_for_source(source)
        logger.info(
            "📚 catalog | pool from universe.csv | kind=%s name=%s → %d supported asset(s)",
            source.kind, source.name, len(pool),
        )
        return pool

    provider.point_in_time = False  # type: ignore[attr-defined]
    return provider
