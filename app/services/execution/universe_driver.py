"""
app/services/execution/universe_driver.py — the live refresh orchestration + process driver (§9).

Two layers, kept separate so the load-bearing logic stays testable without infrastructure:

  * :func:`run_universe_refresh` — ONE deployment's refresh tick end-to-end: gate on the market
    calendar (``should_fire`` + ``is_market_open``), resolve the live universe in the asset's
    timezone, persist the snapshot audit row, diff vs the persisted members, and reconcile
    ``universe_members`` (spawn → active, retire → retiring/retired per ``removal_policy``). It
    composes the pure pieces (``run_refresh_tick``, ``diff_membership``) with the DB edge
    (``universe_persistence``) and the advisory lock — no APScheduler dependency.

  * :class:`SchedulerDriver` / :class:`APSchedulerDriver` / :class:`ManualDriver` — the process
    driver that fires one job per deployment on its cadence. APScheduler is **not currently a
    dependency**; ``APSchedulerDriver`` imports it lazily and raises an actionable error if
    absent, while ``ManualDriver`` lets the core be driven (and tested) without any scheduler.

The live resolver wrapper :func:`resolve_live_universe` returns a uniform
:class:`ResolvedUniverse` for BOTH asset classes — crypto via the Binance 24h-ticker ranker,
equity via the full KB-free resolver — so the persistence + diff path is identical regardless
of source.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.execution.market_calendar import is_market_open, should_fire
from app.services.execution.universe_scheduler import (
    AdvisoryLock,
    UniverseRefreshResult,
    run_refresh_tick,
)

logger = logging.getLogger(__name__)


# ── Live resolver wrapper (uniform ResolvedUniverse for crypto + equity) ──────
def _asset_class_for_source(source_kind: str) -> str:
    """Crypto sources reason in UTC/24-7; everything else is equity (NSE/IST)."""
    return "crypto" if source_kind == "crypto_all" else "equity"


async def resolve_live_universe(
    universe: Any,
    *,
    asof: datetime,
    settings: Any | None = None,
    run_id: str | None = None,
):
    """Resolve ``universe`` (a :class:`UniverseSpec`) to a :class:`ResolvedUniverse` on LIVE data.

    crypto_all → rank via the Binance 24h ticker (one call), wrapped into a ResolvedUniverse so
    downstream audit/diff is uniform. index/sector/f_and_o/watchlist → the full KB-free resolver
    (``resolve_universe`` with the catalog pool provider). Reuses the EXACT primitives the chat
    backtest path uses (Invariant 1) — no resolution logic is reimplemented here.
    """
    from app.core.config import get_settings
    from app.services.universe.hashing import rule_hash, snapshot_hash
    from app.services.universe.types import ResolvedMember, ResolvedUniverse

    settings = settings or get_settings()
    kind = universe.source.kind
    asof_iso = asof.astimezone(asof.tzinfo).isoformat() if asof.tzinfo else asof.isoformat()
    logger.info("🧭 resolve_live_universe | kind=%s | rank=%s(%s) | asof=%s",
                kind, universe.rank.by, universe.rank.order, asof_iso)

    if kind == "crypto_all":
        from app.services.strategy.universe_catalog import crypto_supported_bases
        from app.services.universe.crypto_source import rank_supported_crypto

        cap = settings.dynamic_universe_max_assets
        effective_take = min(universe.take, cap)
        logger.info("🪙 crypto | ranking supported bases via Binance 24h ticker | take=%d (cap=%d)",
                    effective_take, cap)
        members, note = await rank_supported_crypto(
            by=universe.rank.by, order=universe.rank.order, limit=effective_take,
            supported_bases=crypto_supported_bases(),
        )
        logger.info("🪙 crypto | 🏆 MEMBERS (%d): %s%s", len(members), members,
                    f" | ⚠️ {note}" if note else "")
        r_hash = rule_hash(universe.model_dump(mode="json"))
        s_hash = snapshot_hash(rule_hash=r_hash, asof=asof_iso, members=members)
        return ResolvedUniverse(
            asof=asof_iso,
            members=[ResolvedMember(symbol=s, rank=i + 1, rank_value=0.0) for i, s in enumerate(members)],
            requested_take=universe.take,
            effective_take=effective_take,
            cap_reason="platform_limit" if effective_take < universe.take else "none",
            pool_size=len(members),
            screened_count=len(members),
            eligible_count=len(members),
            breadth_reason=note or "",
            survivorship_mode="approximate",
            rule_hash=r_hash,
            snapshot_hash=s_hash,
        )

    # Equities → the full resolver over the master catalog (KB-free pool provider).
    from app.services.strategy.universe_catalog import make_kb_pool_provider
    from app.services.universe.chat_resolution import _backtest_fetch_ohlcv
    from app.services.universe.resolver import resolve_universe

    return await resolve_universe(
        universe, asof=asof, fetch_ohlcv=_backtest_fetch_ohlcv,
        pool_provider=make_kb_pool_provider(), settings=settings, run_id=run_id,
    )


# ── One deployment's refresh tick (calendar-gated, persisted, reconciled) ─────
async def run_universe_refresh(
    db: AsyncSession,
    *,
    deployment_id: str,
    universe: Any,
    asof: datetime,
    lock: AdvisoryLock,
    last_fired_at: datetime | None = None,
    tenant_id: Any | None = None,
    resolve: Callable[[], Awaitable[Any]] | None = None,
    force: bool = False,
) -> UniverseRefreshResult:
    """Run one calendar-gated refresh tick for ``deployment_id`` and reconcile ``universe_members``.

    Gates (unless ``force``): the refresh cadence must be due (:func:`should_fire`) AND the
    asset's market must be open (:func:`is_market_open`) — an equity refresh never runs outside
    the NSE session. On firing: resolve → persist the snapshot audit row → diff vs the persisted
    active members → spawn/retire. ``resolve`` is injectable (defaults to the live resolver) so
    the orchestration is testable without market data.
    """
    from app.services.execution.universe_persistence import (
        apply_membership_diff,
        load_active_members,
        persist_snapshot,
    )

    asset_class = _asset_class_for_source(universe.source.kind)
    logger.info(
        "\n┌─🌌 UNIVERSE REFRESH | deployment=%s ──────────────────────────────────\n"
        "│  asof     : %s\n"
        "│  rule     : source=%s(%s)  rank=%s(%s)  take=%d  refresh=%s%s\n"
        "│  asset    : %s  (tz=%s)  removal_policy=%s%s\n"
        "└──────────────────────────────────────────────────────────────────────",
        deployment_id, asof.isoformat(),
        universe.source.kind, universe.source.name or "-",
        universe.rank.by, universe.rank.order, universe.take,
        universe.refresh.cadence, f" @ {universe.refresh.at}" if universe.refresh.at else "",
        asset_class, "UTC" if asset_class == "crypto" else "Asia/Kolkata",
        universe.removal_policy, "  (FORCED)" if force else "",
    )

    # ── Gate: cadence due AND market open (equity never refreshes outside the session) ──
    if not force:
        fire = should_fire(asof, universe.refresh, asset_class, last_fired_at)
        open_now = is_market_open(asof, asset_class)
        logger.info("🌌 GATE | should_fire=%s (last_fired=%s) | market_open=%s",
                    fire, last_fired_at.isoformat() if last_fired_at else "never", open_now)
        if not fire:
            logger.info("🌌 ⏭️  SKIP | deployment=%s | reason=cadence_not_due — next due later", deployment_id)
            return UniverseRefreshResult(fired=False)
        if not open_now:
            logger.info("🌌 ⏭️  SKIP | deployment=%s | reason=market_closed | asset=%s", deployment_id, asset_class)
            return UniverseRefreshResult(fired=False)
    logger.info("🌌 ✅ GATE PASSED — proceeding to resolve + reconcile")

    active_rows = await load_active_members(db, deployment_id=deployment_id, include_retiring=False)
    active_members = {r.symbol for r in active_rows}
    logger.info("🌌 CURRENT MEMBERS | active=%d → %s", len(active_members), sorted(active_members) or "—")

    async def _resolve():
        if resolve is not None:
            return await resolve()
        return await resolve_live_universe(universe, asof=asof, run_id=str(deployment_id))

    async def _on_resolved(resolved):
        await persist_snapshot(db, deployment_id=deployment_id, resolved=resolved, tenant_id=tenant_id)

    async def _on_spawn(symbols, resolved):
        await apply_membership_diff(
            db, deployment_id=deployment_id, to_spawn=symbols, to_retire=[],
            snapshot_hash=resolved.snapshot_hash, removal_policy=universe.removal_policy,
            tenant_id=tenant_id,
        )

    async def _on_retire(symbols, resolved):
        await apply_membership_diff(
            db, deployment_id=deployment_id, to_spawn=[], to_retire=symbols,
            snapshot_hash=resolved.snapshot_hash, removal_policy=universe.removal_policy,
            tenant_id=tenant_id,
        )

    return await run_refresh_tick(
        deployment_id=str(deployment_id),
        active_members=active_members,
        resolve=_resolve,
        lock=lock,
        on_resolved=_on_resolved,
        on_spawn=_on_spawn,
        on_retire=_on_retire,
    )


# ── Process driver (one job per deployment; APScheduler optional) ─────────────
@runtime_checkable
class SchedulerDriver(Protocol):
    """Schedules one recurring refresh job per deployment. Injectable so the core logic above
    never depends on a concrete scheduler (APScheduler is an optional dependency)."""

    def add_deployment(self, deployment_id: str, *, cadence: str, job: Callable[[], Awaitable[None]]) -> None: ...
    def remove_deployment(self, deployment_id: str) -> None: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...


class ManualDriver:
    """A no-scheduler driver: registers jobs and fires them on demand (tests / single-shot).

    Lets the whole refresh path run without APScheduler — call :meth:`fire` to trigger a
    deployment's job exactly once (e.g. from a cron entrypoint, a test, or a manual backfill).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Callable[[], Awaitable[None]]] = {}

    def add_deployment(self, deployment_id: str, *, cadence: str, job: Callable[[], Awaitable[None]]) -> None:
        self._jobs[deployment_id] = job
        logger.info("🗓️ manual_driver|registered|deployment=%s|cadence=%s", deployment_id, cadence)

    def remove_deployment(self, deployment_id: str) -> None:
        self._jobs.pop(deployment_id, None)

    async def fire(self, deployment_id: str) -> None:
        job = self._jobs.get(deployment_id)
        if job is not None:
            await job()

    def start(self) -> None:  # nothing to start — firing is manual
        pass

    def shutdown(self) -> None:
        self._jobs.clear()


class APSchedulerDriver:
    """APScheduler-backed driver: one ``AsyncIOScheduler`` cron/interval job per deployment.

    APScheduler is NOT currently installed (see project notes). This class imports it lazily and
    raises an actionable error if absent, so importing this module never fails and the core
    refresh logic stays usable via :class:`ManualDriver`. To enable: ``pip install apscheduler``
    and add it to the deps.
    """

    # Coarse cadence → APScheduler cron kwargs. The exact wall-clock time is enforced inside the
    # job by the market calendar (should_fire), so the cron only needs to tick often enough.
    _CADENCE_INTERVAL_MIN = {"daily": 60, "weekly": 60, "intraday": 5}

    def __init__(self) -> None:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "APSchedulerDriver requires apscheduler, which is not installed. "
                "Install it (`pip install apscheduler`) and add it to the deployment deps, or "
                "use ManualDriver to drive refreshes from an external scheduler/cron."
            ) from exc
        self._scheduler = AsyncIOScheduler()
        self._jobs: dict[str, Any] = {}

    def add_deployment(self, deployment_id: str, *, cadence: str, job: Callable[[], Awaitable[None]]) -> None:
        minutes = self._CADENCE_INTERVAL_MIN.get(cadence, 60)
        self._jobs[deployment_id] = self._scheduler.add_job(
            job, "interval", minutes=minutes, id=deployment_id, replace_existing=True,
        )
        logger.info("🗓️ apscheduler|registered|deployment=%s|every=%dmin", deployment_id, minutes)

    def remove_deployment(self, deployment_id: str) -> None:
        if deployment_id in self._jobs:
            self._scheduler.remove_job(deployment_id)
            self._jobs.pop(deployment_id, None)

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)
