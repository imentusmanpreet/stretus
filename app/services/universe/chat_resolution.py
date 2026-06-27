"""
app/services/universe/chat_resolution.py — resolve a universe rule to member symbols for the
chat backtest path (§9 authoring → §8 backtest bridge).

The chat backtests one symbol per loop iteration; for a dynamic-universe strategy it must run
over the RESOLVED members instead of the placeholder. This turns ``builder.universe`` into a
concrete, capped, ranked list of real symbols the chat's existing per-symbol backtest can run.

Resolution per source (KB-free, Invariant 11):
  * ``crypto_all``  → the exchange's volume-ranked catalog (Binance 24h ticker) — cheap, one
    call, already ranked, so "most active / top-N by volume" is answered directly.
  * ``watchlist``   → the explicit symbols (capped).
  * ``index`` / ``sector`` / ``f_and_o`` → the full resolver over point-in-time
    ``universe_membership`` (needs the table seeded; equities are served by the backtest feed).

The platform asset-count cap (§7.1) is applied here so the chat never spawns more than allowed.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Candidate-pool bound for the equity resolver path (kept modest for interactive latency).
_EQUITY_POOL_FETCH_CONCURRENCY = 12


def _clean(symbols: list[Any]) -> list[str]:
    seen, out = set(), []
    for s in symbols or []:
        if isinstance(s, str) and s.strip() and s.strip().upper() not in seen:
            seen.add(s.strip().upper()); out.append(s.strip())
    return out


async def _backtest_fetch_ohlcv(symbol: str, interval: str, from_utc: str, to_utc: str):
    """Backtest market-data fetcher (same path the rest of the backtest uses)."""
    from app.services.backtest.market_data import (
        StrategyMarketDataRequest,
        fetch_ohlcv_records,
    )
    request = StrategyMarketDataRequest(
        yaml_path="", raw_symbol=symbol,
        symbol=symbol.replace(".NS", "").replace(".BO", ""),
        interval=interval, from_utc=from_utc, to_utc=to_utc,
    )
    return await fetch_ohlcv_records(request)


async def resolve_member_symbols(
    universe_dict: dict[str, Any],
    *,
    asof: datetime,
    settings: Settings | None = None,
    session_id: str | None = None,
) -> tuple[list[str], str | None]:
    """Resolve ``universe_dict`` (a ``UniverseSpec`` dump) to member symbols + a user note.

    Returns ``(members, note)``. ``note`` is a short user-facing line (e.g. the platform-cap
    message) or ``None``. ``members == []`` means nothing resolved (caller surfaces why).
    """
    from app.strategy.spec import UniverseSpec

    settings = settings or get_settings()
    # Defensive: keep only real UniverseSpec fields. Display-only keys (e.g. a `summary`
    # carried by older drafts, or the response-layer `effective_take`/`resolved`) would trip
    # the model's extra="forbid" — strip them so resolution never crashes on a polluted dict.
    _allowed = set(UniverseSpec.model_fields)
    universe_dict = {k: v for k, v in (universe_dict or {}).items() if k in _allowed}
    universe = UniverseSpec.model_validate(universe_dict)
    cap = settings.dynamic_universe_max_assets
    effective_take = min(universe.take, cap)
    note = None
    if effective_take < universe.take:
        note = f"Requested {universe.take}; platform limit applied → using {effective_take}."

    kind = universe.source.kind
    logger.info(
        "\n┌─🌌 UNIVERSE RESOLVER | session=%s ───────────────────────────────────\n"
        "│  rule      : source=%s  rank=%s(%s)  screen=%s\n"
        "│  take      : requested=%d  →  effective=%d  (platform cap=%d)\n"
        "└──────────────────────────────────────────────────────────────────────",
        session_id, kind, universe.rank.by, universe.rank.order, universe.screen or "(none)",
        universe.take, effective_take, cap,
    )

    # An explicit list is the pool itself — no ranking needed.
    if kind == "watchlist":
        members = _clean(universe.source.symbols)[:effective_take]
        logger.info(
            "🌌 resolver | watchlist | 🏆 MEMBERS (%d): %s%s",
            len(members), members, (" | ⚠️ " + note) if note else "",
        )
        return members, note

    # Crypto: rank via Binance's live 24h ticker (one call gives volume / %-change / high-low
    # for every pair). The backtest feed does NOT serve daily crypto bars, so the OHLCV
    # resolver can't be used here — but the ticker covers most-active / gainers-losers / most-
    # volatile accurately and instantly. Restricted to the SUPPORTED coins (universe.csv).
    if kind == "crypto_all":
        from app.services.strategy.universe_catalog import crypto_supported_bases
        from app.services.universe.crypto_source import rank_supported_crypto
        members, rank_note = await rank_supported_crypto(
            by=universe.rank.by, order=universe.rank.order, limit=effective_take,
            supported_bases=crypto_supported_bases(),
        )
        merged_note = " ".join(x for x in (note, rank_note) if x) or None
        logger.info(
            "🌌 resolver | crypto_all (Binance ticker, by %s) | 🏆 MEMBERS (%d): %s%s",
            universe.rank.by, len(members), members, (" | ⚠️ " + merged_note) if merged_note else "",
        )
        return members, merged_note

    # Equities → the FULL resolver over
    # the master catalog (universe.csv). It fetches recent OHLCV and ranks by the requested
    # metric — gainers, volatility, momentum, RSI, 52-week distance, relative strength, … —
    # so ANY user rule works, not just volume. Pool = whatever the catalog says we support.
    from app.services.strategy.universe_catalog import make_kb_pool_provider
    from app.services.universe.resolver import resolve_universe

    provider = make_kb_pool_provider()
    try:
        resolved = await resolve_universe(
            universe, asof=asof, fetch_ohlcv=_backtest_fetch_ohlcv,
            pool_provider=provider, settings=settings, run_id=session_id,
            fetch_concurrency=_EQUITY_POOL_FETCH_CONCURRENCY,
        )
    except Exception as exc:  # noqa: BLE001 — surface as "no members" with the cause logged
        logger.warning("🌌 resolver | %s | ❌ resolve_failed | err=%s", kind, exc)
        return [], f"Could not resolve the {kind} universe: {exc}"
    # Distinguish "feed is down" from "nothing matched" (§13): if we got no members AND most
    # of the pool failed to FETCH, surface the real cause instead of a misleading "no members".
    if not resolved.member_symbols and resolved.pool_size:
        fetch_errors = sum(1 for d in resolved.dropped if d.reason in ("fetch_error", "data_gap"))
        if fetch_errors >= resolved.pool_size:
            logger.warning(
                "🌌 resolver | %s | ⚠️ market-data feed unavailable (%d/%d candidates failed to fetch)",
                kind, fetch_errors, resolved.pool_size,
            )
            return [], (
                f"The market-data feed is currently unavailable — all {resolved.pool_size} "
                f"candidate(s) failed to fetch. Please retry in a moment."
            )
    logger.info(
        "🌌 resolver | %s (catalog pool, ranked by %s) | pool=%d screened=%d eligible=%d → "
        "🏆 MEMBERS (%d): %s | survivorship=%s",
        kind, universe.rank.by, resolved.pool_size, resolved.screened_count,
        resolved.eligible_count, len(resolved.member_symbols), resolved.member_symbols,
        resolved.survivorship_mode,
    )
    return resolved.member_symbols, resolved.cap_message() or note
