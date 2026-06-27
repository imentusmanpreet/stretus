"""
app/services/execution/live_universe.py — clock-quantized, cached live universe resolution.

The external OMS calls the evaluate endpoint every strategy timeframe (e.g. every 5m) and knows
NOTHING about the universe *re-pick* cadence — so deciding "when do I re-pick the members?" is
OUR design choice. We make it **stateless** by quantizing the resolution instant to the refresh
window:

    refresh_asof = previous_refresh_at(now, rule.refresh, asset_class)   # e.g. "today 09:20 IST"
    members      = resolve_live_universe(rule, asof=refresh_asof)        # deterministic

Because the resolver is deterministic, every call inside a refresh window returns the SAME
membership (no churn), and the set re-picks automatically when the window rolls — with **no
stored "last refreshed" state** (the clock is the state) and **no coordination across replicas**
(they all compute the same ``refresh_asof``).

On top of that we memoize by ``(rule_hash, refresh_asof)`` in-process, so the (potentially
expensive) resolution runs ONCE per window instead of on every 5m call. The key changes when the
window rolls, so the cache self-invalidates.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.services.execution.market_calendar import previous_refresh_at
from app.services.universe.hashing import rule_hash as _rule_hash

logger = logging.getLogger(__name__)

# Bounded in-process cache of resolved universes, keyed by (rule_hash, refresh_asof_iso).
# Small + FIFO-evicted: one live entry per (deployment-rule × window) is all we ever reuse.
_CACHE: "OrderedDict[tuple[str, str], Any]" = OrderedDict()
_CACHE_MAX = 256


def _asset_class_for_source(source_kind: str) -> str:
    """Crypto sources reason in UTC/24-7; everything else is equity (NSE/IST)."""
    return "crypto" if source_kind == "crypto_all" else "equity"


def clear_cache() -> None:
    """Drop all memoized resolutions (tests / manual invalidation)."""
    _CACHE.clear()


async def resolve_universe_windowed(
    rule: Any,
    *,
    now: datetime | None = None,
    settings: Settings | None = None,
    run_id: str | None = None,
):
    """Resolve ``rule`` (a :class:`UniverseSpec`) as-of the CURRENT refresh window, memoized.

    Returns a :class:`ResolvedUniverse`. Stable within a window, re-picks at the boundary, and
    resolves at most once per ``(rule, window)`` thanks to the in-process cache.
    """
    from app.services.execution.universe_driver import resolve_live_universe

    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    asset_class = _asset_class_for_source(rule.source.kind)
    refresh_asof = previous_refresh_at(now, rule.refresh, asset_class)

    rhash = _rule_hash(rule.model_dump(mode="json"))
    key = (rhash, refresh_asof.isoformat())

    logger.info(
        "\n┌─🪟 LIVE UNIVERSE | window-quantized resolution ───────────────────────\n"
        "│  now         : %s\n"
        "│  cadence      : %s%s  (asset=%s)\n"
        "│  refresh_asof : %s   ← members are STABLE until the next boundary\n"
        "│  cache_key    : rule=%s window=%s\n"
        "└──────────────────────────────────────────────────────────────────────",
        now.isoformat(),
        rule.refresh.cadence, f" @ {rule.refresh.at}" if rule.refresh.at else "", asset_class,
        refresh_asof.isoformat(), rhash[:12], refresh_asof.isoformat(),
    )

    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)  # mark recently used
        logger.info(
            "🪟 ✅ CACHE HIT | reusing %d member(s) resolved for this window: %s "
            "(no re-rank — same window) | cache_size=%d",
            len(cached.member_symbols), cached.member_symbols, len(_CACHE),
        )
        return cached

    logger.info("🪟 ↻ CACHE MISS | resolving the universe live for this window (first call since boundary)…")
    resolved = await resolve_live_universe(rule, asof=refresh_asof, settings=settings, run_id=run_id)

    _CACHE[key] = resolved
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        evicted_key, _ = _CACHE.popitem(last=False)
        logger.debug("🪟 cache evict | oldest window dropped | key=%s", evicted_key)
    logger.info(
        "🪟 💾 CACHED | %d member(s) for window %s: %s | cache_size=%d "
        "(subsequent calls this window are free)",
        len(resolved.member_symbols), refresh_asof.isoformat(), resolved.member_symbols, len(_CACHE),
    )
    return resolved
