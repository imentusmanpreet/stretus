"""
app/services/universe/resolver.py — the KB-free universe resolution pipeline.

Turns a :class:`app.strategy.spec.UniverseSpec` into a ranked, eligible, content-hashed
:class:`ResolvedUniverse` at a point in time. Ports ``scanner.py``'s proven async patterns —
concurrent bounded fetch, point-in-time condition evaluation, per-symbol failure tolerance,
metric pre-computation — WITHOUT the ``kb.stocks`` dependency (Invariant 11). The pool comes
from :mod:`sources`; screen conditions compile with the engine's own formula compiler.

Pipeline (§7):  source → screen → eligibility (fail-closed) → rank → take-N (clamped to the
platform cap, §7.1) → breadth gate (fail-closed) → hash → return.

Invariants honored here:
  * 4 (no look-ahead): every fetch is bounded by ``to=asof``; metrics read the last bar ≤ asof.
  * 5 (fail-closed): NaN/stale eligibility input EXCLUDES; a failing/uncomputable breadth gate
    admits NO members.
  * 6 (bounded memory): screening frames are daily and dropped after the pass; no high-frequency
    matrix is materialized.
  * 8 (reproducibility): the result carries ``rule_hash`` + ``snapshot_hash``.
  * 12 (config cap): ``effective_take = min(take, DYNAMIC_UNIVERSE_MAX_ASSETS)`` applied AFTER
    ranking, recorded transparently, with a WARNING when it clamps (§7.1).
``fetch_ohlcv`` and ``pool_provider`` are injectable for deterministic, network-free tests.
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from time import perf_counter

import pandas as pd

from app.core.config import Settings, get_settings
from app.services.data.provider import FetchOhlcv, records_to_frame
from app.strategy.spec import UniverseSpec

from . import sources
from .breadth import evaluate_breadth
from .errors import ScaleGuardExceeded
from .hashing import rule_hash, snapshot_hash
from .metrics import average_daily_value, rank_metric
from .types import (
    Candidate,
    Dropped,
    DroppedMember,
    ResolvedMember,
    ResolvedUniverse,
    SurvivorshipMode,
)

logger = logging.getLogger(__name__)

# Bounded concurrency for the screening fetch (port scanner's tolerance; respect feeds).
_DEFAULT_FETCH_CONCURRENCY = 20
# Calendar-day floor for the daily screening window (≥ 252 trading days for 52-week metrics),
# plus a buffer for non-trading days — mirrors scanner.py's lookback policy.
_SCREEN_LOOKBACK_DAYS_FLOOR = 280
_SCREEN_LOOKBACK_BUFFER_DAYS = 30


def _iso(dt: datetime) -> str:
    """UTC ISO-8601 with a trailing Z, matching the market-data fetchers."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _screen_window(universe: UniverseSpec, asof: datetime) -> tuple[str, str]:
    """(from_iso, to_iso) for the Tier-A screening fetch — bounded at ``asof`` (no look-ahead)."""
    rank_window = universe.rank.window or 0
    needed_days = max(_SCREEN_LOOKBACK_DAYS_FLOOR, rank_window)
    from_dt = asof - timedelta(days=needed_days + _SCREEN_LOOKBACK_BUFFER_DAYS)
    return _iso(from_dt), _iso(asof)


def _compile_screen(conditions: list[str]):
    """Compile screen conditions once (engine compiler) → (compiled, indicator_config).

    Late-imports the engine like ``scanner.py`` does, so this module loads without dragging
    the engine's heavy graph into import time. Returns ``None`` compiled-list on bad input
    (the validator should have rejected those already; this is a runtime backstop).
    """
    # Reuse engine_bridge so quant_engine/ is on sys.path in EVERY process (the API
    # container, not just the engine/tests) — a raw ``import engine`` fails otherwise.
    from app.strategy import engine_bridge
    compile_condition = engine_bridge.get_compile_condition()

    compiled = []
    indicator_config: dict[str, set[int]] = {}
    for cond in conditions:
        c = compile_condition(cond)
        if c is None:
            continue
        compiled.append(c)
        for ref in c.indicator_refs:
            indicator_config.setdefault(ref.name, set()).add(ref.period)
    return compiled, {n: sorted(p) for n, p in indicator_config.items()}


def _passes_screen(df: pd.DataFrame, compiled, indicator_config) -> bool:
    """Evaluate every compiled screen condition at the last bar (≤ asof). All must pass."""
    if not compiled:
        return True
    from app.strategy import engine_bridge
    engine_bridge.is_available()  # ensure quant_engine/ on sys.path (idempotent)
    from engine.indicators import add_all_indicators  # late import

    eval_df = add_all_indicators(df, indicator_config) if indicator_config else df
    last_i = len(eval_df) - 1
    if last_i < 0:
        return False
    for c in compiled:
        try:
            if not c.evaluate(eval_df, last_i):
                return False
        except Exception as exc:  # noqa: BLE001 — per-symbol tolerance
            logger.warning("resolver|screen_eval_failed|cond=%r|err=%s", c.raw, exc)
            return False
    return True


async def resolve_universe(
    universe: UniverseSpec,
    *,
    asof: datetime,
    fetch_ohlcv: FetchOhlcv,
    pool_provider: sources.PoolProvider | None = None,
    reference_symbol: str | None = None,
    settings: Settings | None = None,
    survivorship_mode: SurvivorshipMode | None = None,
    run_id: str | None = None,
    fetch_concurrency: int = _DEFAULT_FETCH_CONCURRENCY,
) -> ResolvedUniverse:
    """Resolve ``universe`` to a snapshot at ``asof``. See module docstring for the pipeline.

    Args:
      universe: the rule to resolve.
      asof: the point in time; only data ≤ asof is read (Invariant 4).
      fetch_ohlcv: injectable async ``(symbol, interval, from_iso, to_iso) -> records``.
      pool_provider: injectable resolver for index/sector/f_and_o/crypto_all pools (KB-free).
      reference_symbol: index used by the ``rel_strength`` ranking metric (optional).
      settings: platform config (caps, ADV window); defaults to ``get_settings()``.
      survivorship_mode: ``"point_in_time"`` or ``"approximate"``. When ``None`` (default) it
        is auto-detected (Invariant 7): a point-in-time ``pool_provider`` (one exposing a
        truthy ``point_in_time`` attribute, e.g. the membership store) ⇒ ``"point_in_time"``;
        otherwise ``"approximate"``. An explicit value always wins.
      run_id: correlation id stamped on every log line (§12.1).
    """
    settings = settings or get_settings()
    cid = run_id or "—"
    if survivorship_mode is None:
        survivorship_mode = (
            "point_in_time" if getattr(pool_provider, "point_in_time", False)
            else "approximate"
        )
    t_begin = perf_counter()
    scan_tf = (universe.scan_timeframe or "1d").strip().lower()
    from_iso, to_iso = _screen_window(universe, asof)
    asof_iso = _iso(asof)

    # ── 1. source → candidate pool (KB-free) ──────────────────────────────────
    t0 = perf_counter()
    pool = await sources.resolve_pool(universe.source, asof=asof, pool_provider=pool_provider)
    logger.info(
        "resolver|begin|run=%s|source=%s|name=%s|pool=%d|asof=%s|scan_tf=%s",
        cid, universe.source.kind, universe.source.name, len(pool), asof_iso, scan_tf,
    )

    # Scale guard (§4.5): refuse an oversized pool with an actionable message, never OOM.
    if len(pool) > settings.dynamic_universe_max_pool:
        raise ScaleGuardExceeded(
            f"candidate pool of {len(pool)} exceeds DYNAMIC_UNIVERSE_MAX_POOL="
            f"{settings.dynamic_universe_max_pool}. Narrow the source/screen or raise the limit."
        )

    # ── 2. concurrent, bounded, fault-tolerant screening fetch (Tier A) ───────
    sem = asyncio.Semaphore(max(1, fetch_concurrency))
    needs_reference = universe.rank.by == "rel_strength" and bool(reference_symbol)
    reference_df: pd.DataFrame | None = None

    async def _fetch(symbol: str) -> tuple[str, pd.DataFrame | None, bool]:
        """Returns (symbol, frame, fetch_errored)."""
        async with sem:
            try:
                records = await fetch_ohlcv(symbol, scan_tf, from_iso, to_iso)
            except Exception as exc:  # noqa: BLE001 — per-symbol tolerance (Invariant 13)
                logger.warning("resolver|fetch_failed|run=%s|symbol=%s|err=%s", cid, symbol, exc)
                return symbol, None, True
            return symbol, records_to_frame(records), False

    if needs_reference:
        ref_records = None
        try:
            ref_records = await fetch_ohlcv(reference_symbol, scan_tf, from_iso, to_iso)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolver|reference_fetch_failed|run=%s|symbol=%s|err=%s", cid, reference_symbol, exc)
        reference_df = records_to_frame(ref_records)

    fetched = await asyncio.gather(*[_fetch(s) for s in pool])
    t_fetch = perf_counter() - t0
    logger.info("⏱️ TIMING|step=screen_fetch|run=%s|duration=%.4fs|pool=%d", cid, t_fetch, len(pool))

    # ── 3. screen + metric pre-computation (pure O(N) over the tail) ───────────
    t0 = perf_counter()
    compiled, indicator_config = _compile_screen(universe.screen)
    adv_window = universe.eligibility.adv_window or settings.dynamic_universe_adv_window
    rank_window = universe.rank.window
    screen_frames: dict[str, pd.DataFrame] = {}  # for the breadth gate

    candidates: list[Candidate] = []
    dropped: list[Dropped] = []
    screened_count = 0
    for symbol, df, errored in fetched:
        if df is None:
            dropped.append(Dropped(symbol, "fetch_error" if errored else "data_gap"))
            continue
        if len(df) < 2:
            dropped.append(Dropped(symbol, "insufficient_data"))
            continue
        screen_frames[symbol] = df
        if not _passes_screen(df, compiled, indicator_config):
            dropped.append(Dropped(symbol, "screen_failed"))
            continue
        screened_count += 1
        metrics = {
            "rank_value": rank_metric(df, by=universe.rank.by, window=rank_window, reference=reference_df),
            "adv": average_daily_value(df, window=adv_window),
            "close": float(df["close"].iloc[-1]),
        }
        candidates.append(Candidate(symbol=symbol, metrics=metrics))

    # ── 4. eligibility (fail-closed) ──────────────────────────────────────────
    min_adv = universe.eligibility.min_adv
    eligible: list[Candidate] = []
    for cand in candidates:
        adv = cand.metrics.get("adv", float("nan"))
        if min_adv is not None and (math.isnan(adv) or adv < min_adv):
            dropped.append(Dropped(cand.symbol, "ineligible_adv"))
            continue
        eligible.append(cand)

    # ── 5. rank (drop unrankable, fail-closed), then 6. take-N with the cap ────
    rankable: list[Candidate] = []
    for cand in eligible:
        rv = cand.metrics.get("rank_value", float("nan"))
        if rv is None or math.isnan(rv):
            dropped.append(Dropped(cand.symbol, "rank_unavailable"))
            continue
        rankable.append(cand)
    rankable.sort(
        key=lambda c: c.metrics["rank_value"],
        reverse=(universe.rank.order == "desc"),
    )

    requested_take = universe.take
    effective_take = min(requested_take, settings.dynamic_universe_max_assets)
    cap_reason = "platform_limit" if effective_take < requested_take else "none"
    if cap_reason == "platform_limit":
        logger.warning(
            "resolver|cap|run=%s|requested=%d|effective=%d|reason=platform_limit",
            cid, requested_take, effective_take,
        )
    taken = rankable[:effective_take]
    t_rank = perf_counter() - t0
    logger.info(
        "⏱️ TIMING|step=screen_rank|run=%s|duration=%.4fs|screened=%d|eligible=%d|rankable=%d|taken=%d",
        cid, t_rank, screened_count, len(eligible), len(rankable), len(taken),
    )

    # ── 7. breadth gate (fail-closed): a failing gate admits NO members ────────
    breadth_ok, breadth_reason = evaluate_breadth(universe.breadth, screen_frames)
    if not breadth_ok:
        logger.warning("resolver|breadth_blocked|run=%s|%s", cid, breadth_reason)
        for cand in taken:
            dropped.append(Dropped(cand.symbol, "breadth_blocked"))
        members: list[ResolvedMember] = []
    else:
        members = [
            ResolvedMember(
                symbol=cand.symbol,
                rank=i + 1,
                rank_value=float(cand.metrics["rank_value"]),
                metrics={k: float(v) for k, v in cand.metrics.items() if not math.isnan(v)},
            )
            for i, cand in enumerate(taken)
        ]

    # ── 8. hash for reproducibility ───────────────────────────────────────────
    r_hash = rule_hash(universe.model_dump(mode="json"))
    s_hash = snapshot_hash(rule_hash=r_hash, asof=asof_iso, members=[m.symbol for m in members])

    resolved = ResolvedUniverse(
        asof=asof_iso,
        members=members,
        dropped=[DroppedMember(symbol=d.symbol, reason=d.reason) for d in dropped],
        requested_take=requested_take,
        effective_take=effective_take,
        cap_reason=cap_reason,
        pool_size=len(pool),
        screened_count=screened_count,
        eligible_count=len(eligible),
        breadth_ok=breadth_ok,
        breadth_reason=breadth_reason,
        survivorship_mode=survivorship_mode,
        rule_hash=r_hash,
        snapshot_hash=s_hash,
    )
    logger.info(
        "resolver|done|run=%s|asof=%s|pool=%d|screened=%d|eligible=%d|members=%d|"
        "breadth=%s|cap=%s|survivorship=%s|hash=%s|duration=%.4fs",
        cid, asof_iso, len(pool), screened_count, len(eligible), len(members),
        breadth_ok, cap_reason, survivorship_mode, s_hash[:12], perf_counter() - t_begin,
    )
    return resolved
