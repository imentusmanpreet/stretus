"""
app/services/universe/backtest_orchestrator.py — Tier-B execution-data loading (§8 step 4).

The bridge between a resolved universe (Tier A, the membership snapshot) and the portfolio
backtest (which needs each member's EXECUTION-tier OHLCV). It loads Tier-B bars lazily,
only for resolved members, only over their membership window (+ warm-up), with bounded
concurrency and per-symbol failure tolerance — so peak memory scales with active members,
not pool size or range (Invariant 6). The store is any :class:`DataProvider` (injected
fetcher → Parquet → ClickHouse), so this is swappable and testable without a network.

It deliberately stops at *preparing the inputs* for the engine's portfolio backtest
(`engine.runner.run_portfolio_backtest`): symbol → execution OHLCV records, plus membership
windows. Wiring those across the API↔engine service boundary (a `/run_portfolio` endpoint)
is the remaining Phase-B transport glue and lives outside this KB-free module.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.services.data.provider import CachingFetchProvider, DataProvider, FetchOhlcv
from app.strategy.spec import UniverseSpec

from . import sources
from .resolver import resolve_universe
from .types import ResolvedUniverse

logger = logging.getLogger(__name__)

_DEFAULT_LOAD_CONCURRENCY = 16


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def load_member_execution_frames(
    resolved: ResolvedUniverse,
    *,
    provider: DataProvider,
    timeframe: str,
    window_from: datetime,
    window_to: datetime,
    warmup_bars: int = 0,
    bar_minutes: int = 1,
    concurrency: int = _DEFAULT_LOAD_CONCURRENCY,
) -> dict[str, list[dict[str, Any]]]:
    """Lazily fetch execution-tier OHLCV for each resolved member (Tier B).

    Returns ``{symbol: records}`` for members with data; members whose fetch fails or
    returns nothing are omitted (per-symbol tolerance — they simply don't trade). The
    ``warmup_bars`` pad the window start so indicators are primed when a member activates.
    Bounded by ``concurrency`` to respect store/broker limits (§12.3).
    """
    members = resolved.member_symbols
    if not members:
        logger.info("orchestrator|no_members|asof=%s", resolved.asof)
        return {}

    warmup_from = window_from - timedelta(minutes=max(0, warmup_bars) * max(1, bar_minutes))
    from_iso, to_iso = _iso(warmup_from), _iso(window_to)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _load(symbol: str) -> tuple[str, list[dict[str, Any]] | None]:
        async with sem:
            frame = await provider.fetch_execution(
                symbol, timeframe=timeframe, from_iso=from_iso, to_iso=to_iso
            )
        if frame is None or frame.empty:
            return symbol, None
        # Normalize back to the records shape run_portfolio_backtest / run_backtest expect.
        records = [
            {"timestamp": ts.isoformat() + "Z", **{c: float(frame.at[ts, c]) for c in frame.columns}}
            for ts in frame.index
        ]
        return symbol, records

    results = await asyncio.gather(*[_load(s) for s in members])
    out: dict[str, list[dict[str, Any]]] = {s: recs for s, recs in results if recs}
    logger.info(
        "orchestrator|tier_b_loaded|members=%d|with_data=%d|tf=%s|from=%s|to=%s",
        len(members), len(out), timeframe, from_iso, to_iso,
    )
    return out


def membership_windows_from_snapshots(
    snapshots: list[ResolvedUniverse],
    *,
    horizon_to: datetime,
) -> dict[str, list[tuple[str, str]]]:
    """Build per-symbol [from, to] membership windows from a sequence of refresh snapshots.

    A member is "active" from the snapshot that first includes it until the snapshot that
    drops it (or ``horizon_to`` if it is never dropped). These windows feed
    ``run_portfolio_backtest`` so a symbol only trades while it is in the universe (§8 step 4).
    Snapshots must be passed in ascending ``asof`` order.
    """
    windows: dict[str, list[tuple[str, str]]] = {}
    open_since: dict[str, str] = {}
    ordered = sorted(snapshots, key=lambda s: s.asof)
    prev_members: set[str] = set()
    for snap in ordered:
        current = set(snap.member_symbols)
        for sym in current - prev_members:           # newly added → open a window
            open_since[sym] = snap.asof
        for sym in prev_members - current:            # dropped → close its window
            if sym in open_since:
                windows.setdefault(sym, []).append((open_since.pop(sym), snap.asof))
        prev_members = current
    horizon_iso = _iso(horizon_to)
    for sym, start in open_since.items():             # still active at the horizon
        windows.setdefault(sym, []).append((start, horizon_iso))
    return windows


async def run_dynamic_backtest(
    *,
    universe: UniverseSpec,
    template_yaml: str,
    window_from: datetime,
    window_to: datetime,
    timeframe: str,
    fetch_ohlcv: FetchOhlcv,
    market_data_request: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    asof: datetime | None = None,
    pool_provider: sources.PoolProvider | None = None,
    reference_symbol: str | None = None,
    provider: DataProvider | None = None,
    engine_call=None,
    settings: Settings | None = None,
    starting_capital: float = 100_000.0,
    sector_map: dict[str, str] | None = None,
    sector_cap_pct: float | None = None,
    warmup_bars: int = 0,
    backtest_ref_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """End-to-end dynamic-universe backtest bridge: resolve → Tier-B load → engine.

    Ties the KB-free resolver, the Tier-B execution-data loader, and the engine's
    ``/run-portfolio-sync`` into one call. The platform asset/position caps (§7.1) are
    applied by the resolver (members) and clamped here for ``max_positions``. ``engine_call``
    is injectable (defaults to the real HTTP client) so this is testable with no network;
    ``fetch_ohlcv`` and ``pool_provider`` are injectable too.

    Phase-B/C v1 resolves a SINGLE snapshot at ``asof`` (defaults to ``window_to``); a
    member is active across the whole window. Multi-cadence refresh (resolve per refresh
    point, then :func:`membership_windows_from_snapshots`) is the natural extension and
    reuses the helpers already here.
    """
    settings = settings or get_settings()
    asof = asof or window_to
    run_config = dict(run_config or {})

    # 1. Resolve membership (survivorship auto-detected from the pool_provider).
    resolved = await resolve_universe(
        universe, asof=asof, fetch_ohlcv=fetch_ohlcv, pool_provider=pool_provider,
        reference_symbol=reference_symbol, settings=settings, run_id=run_id,
    )
    if not resolved.members:
        logger.warning("orchestrator|dynamic_backtest|no_members|asof=%s", resolved.asof)
        return {
            "members": [], "metrics": {"total_trades": 0.0}, "equity_curve": [],
            "survivorship_mode": resolved.survivorship_mode,
            "snapshots": [resolved.model_dump(mode="json")],
            "cap_message": resolved.cap_message(),
            "backtest_ref_id": backtest_ref_id,
        }

    # 2. Tier-B lazy load (only members, only the window + warm-up).
    provider = provider or CachingFetchProvider(fetch_ohlcv)
    member_ohlcv = await load_member_execution_frames(
        resolved, provider=provider, timeframe=timeframe,
        window_from=window_from, window_to=window_to, warmup_bars=warmup_bars,
    )

    # 3. Effective concurrent-position cap (§7.1): intent ∧ platform ceiling ∧ #members.
    effective_max_positions = min(
        universe.effective_max_positions(cap=settings.dynamic_universe_max_positions),
        max(1, len(member_ohlcv)),
    )

    # 4. Hand off to the engine's portfolio backtest.
    if engine_call is None:
        from app.services.backtest.quant_engine_client import run_quant_portfolio_sync
        engine_call = run_quant_portfolio_sync

    payload = {
        "template_yaml": template_yaml,
        "member_ohlcv": member_ohlcv,
        "run_config": run_config,
        "market_data_request": market_data_request,
        "membership_windows": {s: [(_iso(window_from), _iso(window_to))] for s in member_ohlcv},
        "starting_capital": starting_capital,
        "max_positions": effective_max_positions,
        "sizing_mode": "risk_based" if universe.rank.by in ("atr_pct",) else "equal_weight",
        "risk_per_trade_pct": 1.0,
        "sector_map": sector_map,
        "sector_cap_pct": sector_cap_pct,
        "survivorship_mode": resolved.survivorship_mode,
        "snapshots": [resolved.model_dump(mode="json")],
        "backtest_ref_id": backtest_ref_id,
    }
    result = await engine_call(payload)
    # Surface the cap message + snapshot transparency on the result (Invariant 10, additive).
    if resolved.cap_message():
        result.setdefault("cap_message", resolved.cap_message())
    logger.info(
        "orchestrator|dynamic_backtest|done|members=%d|max_positions=%d|survivorship=%s",
        len(member_ohlcv), effective_max_positions, resolved.survivorship_mode,
    )
    return result
