"""
app/services/universe/sources.py — the KB-free universe-source registry.

Resolves a :class:`app.strategy.spec.UniverseSource` into a concrete candidate pool — a
list of symbols — at a point in time. This is the piece that REPLACES the scanner's
``kb.stocks`` coupling (Invariant 11): the pool comes from here, never the knowledge base.

Source kinds and their Phase-A backing:
  * ``watchlist`` — fully resolved here: the explicit ``symbols`` list (deduped, cleaned).
  * ``index`` / ``sector`` / ``f_and_o`` — point-in-time constituents that live in the
    ``universe_membership`` table (Phase C). Until that lands, the pool is supplied by an
    INJECTED ``pool_provider`` (so the resolver is testable now and the Parquet/DB store
    drops in without touching callers). With no provider, a clear :class:`UniverseSourceError`
    is raised — never a silent empty pool.
  * ``crypto_all`` — the exchange's tradable-pairs catalog, likewise via ``pool_provider``.

The injected ``pool_provider`` has signature ``(UniverseSource, asof: datetime) -> list[str]``
(sync or async). It exists for the same reason ``scan_universe`` injects ``fetch_ohlcv``:
deterministic tests with no network, and a swappable production backing (§12.2).
"""
from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Awaitable, Callable, Union

from app.strategy.spec import UniverseSource

from .errors import UniverseSourceError

logger = logging.getLogger(__name__)

# Injected pool provider: (source, asof) -> symbols (sync or async).
PoolProvider = Callable[[UniverseSource, datetime], Union[list[str], Awaitable[list[str]]]]

# Source kinds whose pool comes from point-in-time constituents / an exchange catalog,
# i.e. those that REQUIRE a pool_provider until the membership store lands (Phase C).
_PROVIDER_BACKED_KINDS: frozenset[str] = frozenset({"index", "sector", "f_and_o", "crypto_all"})


def _clean_symbols(symbols: list[str]) -> list[str]:
    """Strip/upper-case and dedupe while preserving order. Drops blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        if not isinstance(raw, str):
            continue
        sym = raw.strip()
        if not sym:
            continue
        key = sym.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(sym)
    return out


async def resolve_pool(
    source: UniverseSource,
    *,
    asof: datetime,
    pool_provider: PoolProvider | None = None,
) -> list[str]:
    """Resolve ``source`` to a candidate symbol pool at ``asof`` (KB-free).

    Raises :class:`UniverseSourceError` when a provider-backed source has no
    ``pool_provider`` injected, or when a provider returns nothing usable — surfacing the
    real cause rather than a misleading "no matches" (Invariant: error-surfacing).
    """
    if source.kind == "watchlist":
        pool = _clean_symbols(source.symbols)
        if not pool:
            raise UniverseSourceError(
                "watchlist source has no symbols — provide a non-empty `symbols` list."
            )
        logger.info("sources|watchlist|symbols=%d", len(pool))
        return pool

    if source.kind in _PROVIDER_BACKED_KINDS:
        if pool_provider is None:
            raise UniverseSourceError(
                f"source kind {source.kind!r} (name={source.name!r}) needs point-in-time "
                f"constituents, which require either the universe_membership store "
                f"(Phase C) or an injected pool_provider. None was available."
            )
        result = pool_provider(source, asof)
        if inspect.isawaitable(result):
            result = await result
        pool = _clean_symbols(list(result or []))
        if not pool:
            raise UniverseSourceError(
                f"pool_provider returned no symbols for source kind {source.kind!r} "
                f"(name={source.name!r}) at asof={asof.isoformat()}."
            )
        logger.info(
            "sources|%s|name=%s|symbols=%d", source.kind, source.name, len(pool)
        )
        return pool

    # Defensive: the spec Literal should make this unreachable.
    raise UniverseSourceError(f"unknown universe source kind: {source.kind!r}")
