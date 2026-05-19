"""
app/services/discovery/scanner.py
─────────────────────────────────
Phase 9 — universe scanner.

Iterates the KB stock universe, fetches recent OHLCV for each candidate
(via the existing market-data path so caching + chunked fetches are reused),
evaluates the strategy preset's `discovery.conditions` against each
candidate's most-recent bar, and returns a ScanResult.

The scanner also pre-computes the metric values that downstream tie-break
methods need (relative_volume, distance_to_52w_high_pct, rsi_14, atr_14_pct,
close) so tie-breaking is a pure O(N) sort with no I/O.

NO SIDE EFFECTS — the scanner does not mutate the strategy/builder. The
caller (chat layer) decides what to do with the result.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any

import numpy as np
import pandas as pd

from app.kb import kb
from app.services.discovery.tie_break import available_tie_break_options
from app.services.discovery.types import (
    Candidate,
    DiscoveryConfig,
    DiscoveryStatus,
    ScanResult,
)

logger = logging.getLogger(__name__)


# Bars per calendar day for NSE — same numbers used by app.services.backtest.market_data,
# kept locally to avoid an import cycle (the scanner is called from chat layer
# which sometimes initialises before backtest service does).
_BARS_PER_CALENDAR_DAY: dict[str, float] = {
    "1m":  375.0, "5m":  75.0, "10m": 37.0, "15m": 25.0,
    "30m": 12.0, "1h":  6.25, "1d":  1.0,
}


def _required_bars_for_conditions(lookback_days: int, interval: str) -> int:
    """Conservative bar count needed to evaluate discovery conditions.
    Uses a 252-day floor since 52-week filters need a full year of data."""
    bars_per_day = _BARS_PER_CALENDAR_DAY.get(interval, 1.0)
    return max(int(lookback_days * bars_per_day), 60)


def _df_from_records(records: list[dict[str, Any]] | None) -> pd.DataFrame | None:
    """Convert market-data OHLCV records into the DataFrame shape the AST
    evaluator expects. Returns None when records are missing/empty."""
    if not records:
        return None
    df = pd.DataFrame.from_records(records)
    if "timestamp" not in df.columns:
        return None
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").tz_localize(None)
    keep_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if not keep_cols:
        return None
    return df[keep_cols].astype(float)


# ── Metric helpers (pure functions over the candidate's tail OHLCV) ─────────


def _safe_relative_volume(df: pd.DataFrame, window: int = 20) -> float:
    """VOL[-1] / AVG(VOL, window)[-1]. Returns NaN when warm-up incomplete."""
    if "volume" not in df.columns or len(df) < window + 1:
        return float("nan")
    avg = df["volume"].iloc[-(window + 1):-1].mean()
    if avg <= 0 or pd.isna(avg):
        return float("nan")
    return float(df["volume"].iloc[-1]) / float(avg)


def _safe_52w_distance_pct(df: pd.DataFrame, side: str, window: int = 252) -> float:
    """Percentage distance from `close` to the 52-week extremum. Side="high"
    returns (max_high - close) / close * 100 (always ≥ 0); side="low"
    returns (close - min_low) / close * 100 (always ≥ 0). NaN if warm-up
    incomplete."""
    if len(df) < window:
        return float("nan")
    close = float(df["close"].iloc[-1])
    if close <= 0:
        return float("nan")
    tail = df.tail(window)
    if side == "high":
        peak = float(tail["high"].max())
        return (peak - close) / close * 100.0
    if side == "low":
        trough = float(tail["low"].min())
        return (close - trough) / close * 100.0
    return float("nan")


def _safe_rsi(df: pd.DataFrame, window: int = 14) -> float:
    """RSI(14) at the last bar — Wilder smoothing, identical to engine.indicators.
    NaN when fewer than window+1 bars or all-up / all-down series."""
    if "close" not in df.columns or len(df) < window + 1:
        return float("nan")
    delta = df["close"].astype(float).diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    last_loss = float(avg_loss.iloc[-1])
    if last_loss == 0 or pd.isna(last_loss):
        return float("nan")
    rs = float(avg_gain.iloc[-1]) / last_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _safe_atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    """ATR(window) as a percentage of last close."""
    if len(df) < window + 1:
        return float("nan")
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean().iloc[-1]
    last_close = float(close.iloc[-1])
    if pd.isna(atr) or last_close <= 0:
        return float("nan")
    return float(atr) / last_close * 100.0


def _compute_candidate_metrics(
    df: pd.DataFrame,
    *,
    lookback_window_bars: int = 252,
) -> dict[str, float]:
    """Pre-compute every metric a tie-break method might want.

    Phase 9i — `lookback_window_bars` controls the window used for the
    high/low distance metrics. Defaults to 252 (≈ 1 trading year, the
    classic "52-week" window) but the orchestrator overrides via the
    discovery parameters so a user-typed "20-day high" produces a
    metric measured against the 20-day high rather than the 52-week
    high. The metric KEY name is preserved (distance_to_52w_high_pct)
    for backward compat with the tie-break methods that reference it.
    """
    last_close = float(df["close"].iloc[-1])
    window = max(int(lookback_window_bars), 2)
    return {
        "close":                      last_close,
        "relative_volume":            _safe_relative_volume(df),
        "distance_to_52w_high_pct":   _safe_52w_distance_pct(df, "high", window=window),
        "distance_to_52w_low_pct":    _safe_52w_distance_pct(df, "low",  window=window),
        "rsi_14":                     _safe_rsi(df),
        "atr_14_pct":                 _safe_atr_pct(df),
    }


# ── Condition evaluation per candidate ───────────────────────────────────────


def _evaluate_conditions_detailed(
    df: pd.DataFrame, conditions: list[str],
) -> tuple[bool, str | None]:
    """Evaluate all conditions at the last bar of df.

    Phase 9m — returns (passed, first_failing_condition). When passed
    is False, first_failing_condition is the AST string the scanner
    used (so the caller can aggregate fail counts per condition);
    when passed is True it's None.
    """
    if not conditions:
        return False, None
    from engine.conditions import compile_condition
    from engine.indicators import add_all_indicators

    indicator_config: dict[str, set[int]] = {}
    compiled_conds = []
    for cond in conditions:
        compiled = compile_condition(cond)
        if compiled is None:
            return False, cond
        compiled_conds.append(compiled)
        for ref in compiled.indicator_refs:
            indicator_config.setdefault(ref.name, set()).add(ref.period)

    enriched = {n: sorted(p) for n, p in indicator_config.items()}
    eval_df = add_all_indicators(df, enriched) if enriched else df
    last_i = len(eval_df) - 1
    if last_i < 0:
        return False, conditions[0]
    for compiled in compiled_conds:
        try:
            if not compiled.evaluate(eval_df, last_i):
                return False, compiled.raw
        except Exception as exc:
            logger.warning("scanner|condition_eval_failed|cond=%r|err=%s", compiled.raw, exc)
            return False, compiled.raw
    return True, None


def _evaluate_conditions(df: pd.DataFrame, conditions: list[str]) -> bool:
    """Backward-compat shim — returns True iff every condition passes."""
    passed, _ = _evaluate_conditions_detailed(df, conditions)
    return passed


# ── Universe selection ──────────────────────────────────────────────────────


def _select_universe(
    universe_key: str, universe_override: list[str] | None = None,
) -> list:
    """Resolve a discovery.universe key to a concrete list of Stock objects.

    When `universe_override` is non-empty, only enabled stocks whose KB
    symbol (case-insensitive, with or without the .NS suffix) appears
    in the list are kept. The override wins over universe_key so the
    user's "choose among these stocks: TCS, INFY" narrows the scan to
    exactly those symbols, regardless of the preset's default universe.
    Symbols in the override that don't exist in the KB are silently
    dropped — the scanner can only scan stocks it knows about.
    """
    base = [s for s in kb.stocks.values() if s.enabled]
    if universe_override:
        wanted: set[str] = set()
        for sym in universe_override:
            if not isinstance(sym, str):
                continue
            cleaned = sym.strip().upper()
            if not cleaned:
                continue
            wanted.add(cleaned)
            # Also allow the bare form so callers can pass either
            # "TCS" or "TCS.NS" and get the same result.
            bare = cleaned.split(".", 1)[0]
            if bare and bare != cleaned:
                wanted.add(bare)
        if wanted:
            filtered = [
                s for s in base
                if s.symbol.upper() in wanted
                or s.symbol.upper().split(".", 1)[0] in wanted
            ]
            if filtered:
                return filtered
            logger.warning(
                "scanner|universe_override_no_match|override=%s — falling back to full universe",
                universe_override,
            )
    if universe_key in (None, "", "kb_default"):
        return base
    # Future hook: sector subsets ("banking"), market-cap buckets, etc.
    # For now, unknown keys fall back to the full KB universe with a warning.
    logger.warning("scanner|unknown_universe_key=%s — falling back to kb_default", universe_key)
    return base


# ── Public scanner entry point ──────────────────────────────────────────────


async def scan_universe(
    discovery: DiscoveryConfig,
    *,
    interval: str,
    asof_utc: datetime | None = None,
    fetch_ohlcv: "Callable | None" = None,    # noqa: F821 — late-bound helper for testability
    universe_override: list[str] | None = None,
) -> ScanResult:
    """Run the full discovery scan.

    Args:
      discovery: the preset's DiscoveryConfig (must have conditions).
      interval:  the strategy's main timeframe (e.g. "5m"). Discovery
                 conditions evaluate on this interval UNLESS the preset
                 declares `scan_timeframe` (Phase 9c) — useful when the
                 conditions are inherently daily (e.g. "near 52-week high")
                 but the strategy itself trades intraday.
      asof_utc:  the as-of UTC datetime; defaults to "now". The scanner
                 fetches data ending at this point.
      fetch_ohlcv: async callable(symbol, interval, from_utc, to_utc) →
                 list[dict] OHLCV records. When None, defaults to the
                 normal market-data fetcher. Injectable for testing.

    Returns a ScanResult with `status` ∈ {"single", "none", "multiple"}.
    """
    if not discovery.has_anything_to_scan():
        raise ValueError(
            "scan_universe called with a discovery config that has no conditions."
        )

    fetcher = fetch_ohlcv if fetch_ohlcv is not None else _default_fetch_ohlcv
    # Phase 9c — preset can override the scan timeframe. e.g. an intraday
    # strategy whose discovery checks 52-week highs needs daily bars to
    # evaluate that condition correctly.
    scan_interval = (discovery.scan_timeframe or interval).strip().lower()
    asof = asof_utc or datetime.now(timezone.utc)
    asof_iso = asof.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    # Compute the from_utc to leave enough warm-up for 52-week filters.
    # We're conservative: pull `lookback_days` calendar days + a 30-day buffer
    # for non-trading days.
    lookback = max(discovery.lookback_days, 280) + 30
    from_dt = asof - timedelta(days=lookback)
    from_iso = from_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    universe = _select_universe(discovery.universe, universe_override)
    logger.info(
        "scanner|begin|universe=%s|symbols=%d|main_tf=%s|scan_tf=%s|asof=%s",
        discovery.universe, len(universe), interval, scan_interval, asof_iso,
    )

    # Phase 9i — read the active high/low window from the discovery
    # parameters so the metric computation (distance_to_52w_high_pct
    # etc.) uses the same window as the user-supplied condition.
    # Defaults to 252 if the preset doesn't declare one.
    try:
        metric_window = int(float(discovery.parameters.get("lookback_window_bars", 252)))
    except (TypeError, ValueError):
        metric_window = 252

    # Fetch all candidates concurrently; tolerate per-symbol failures so one
    # missing data feed doesn't sink the whole scan.
    tasks = [
        _fetch_and_evaluate(
            stock, scan_interval, from_iso, asof_iso,
            discovery.conditions, fetcher,
            metric_window=metric_window,
        )
        for stock in universe
    ]
    per_symbol_results = await asyncio.gather(*tasks, return_exceptions=False)

    candidates: list[Candidate] = []
    failed = 0
    fail_counts: dict[str, int] = {str(c): 0 for c in discovery.conditions}
    # Phase 9n — capture one example of each unique fetch-error
    # message so the chat layer can surface "InfraError: 503 Service
    # Unavailable" instead of the useless "loosen your conditions"
    # hint when every fetch fails for the same reason.
    fetch_errors: dict[str, str] = {}
    # Per-symbol transparency lists. The aggregate counters above
    # answer "how many?"; these answer "which ones?" so the no-match
    # message can show the user the universe breakdown rather than
    # leaving them guessing.
    scanned_symbols: list[str] = [stock.symbol for stock in universe]
    passed_symbols: list[str] = []
    failed_fetch_symbols: list[str] = []
    failed_condition_by_symbol: dict[str, str] = {}
    for stock, result in zip(universe, per_symbol_results):
        if result is None or not isinstance(result, dict):
            failed += 1
            failed_fetch_symbols.append(stock.symbol)
            continue
        kind = result.get("kind")
        if kind == "fetch_error":
            failed += 1
            failed_fetch_symbols.append(stock.symbol)
            err = str(result.get("error") or "unknown error")
            # Keep only one example per unique error string. Use the
            # symbol as the key for the first occurrence so we can
            # quote it back ("HDFCBANK.NS: 503 Service Unavailable").
            if err not in fetch_errors.values():
                fetch_errors[stock.symbol] = err
        elif kind == "insufficient_data":
            failed += 1
            failed_fetch_symbols.append(stock.symbol)
        elif kind == "pass":
            passed_symbols.append(stock.symbol)
            candidates.append(Candidate(
                symbol=stock.symbol,
                display_name=stock.display_name,
                sector=stock.sector,
                metrics=dict(result.get("metrics") or {}),
            ))
        elif kind == "fail":
            cond = str(result.get("failed_condition") or "")
            if cond:
                fail_counts[cond] = fail_counts.get(cond, 0) + 1
                failed_condition_by_symbol[stock.symbol] = cond

    options = (
        list(discovery.tie_break_options) if discovery.tie_break_options
        else available_tie_break_options()
    )

    status: DiscoveryStatus = (
        "single"   if len(candidates) == 1
        else "none" if len(candidates) == 0
        else "multiple"
    )
    logger.info(
        "scanner|done|status=%s|candidates=%s|scanned=%d|failed=%d",
        status, [c.symbol for c in candidates], len(universe), failed,
    )
    return ScanResult(
        status=status,
        candidates=candidates,
        tie_break_options=options if status == "multiple" else [],
        asof_iso=asof_iso,
        scanned_count=len(universe),
        failed_fetches=failed,
        fail_counts=fail_counts,
        fetch_errors=fetch_errors,
        scanned_symbols=scanned_symbols,
        passed_symbols=passed_symbols,
        failed_fetch_symbols=failed_fetch_symbols,
        failed_condition_by_symbol=failed_condition_by_symbol,
    )


async def _fetch_and_evaluate(
    stock,
    interval: str,
    from_iso: str,
    to_iso: str,
    conditions: list[str],
    fetcher,
    *,
    metric_window: int = 252,
) -> dict | None:
    """Fetch one candidate's OHLCV and evaluate discovery conditions.

    Phase 9n — returns a dict so the caller can disambiguate
    fetch-failures (exception during fetch), insufficient-data
    failures (fetch returned empty/short), and condition-failures
    (data fetched OK, but one of the conditions said no):

      • fetch failed       → {"kind": "fetch_error",
                              "error": "<exception message>"}
      • insufficient data  → {"kind": "insufficient_data"}
      • condition failed   → {"kind": "fail", "metrics": {...},
                              "failed_condition": "<AST string>"}
      • passed everything  → {"kind": "pass",   "metrics": {...}}

    Returning a dict instead of None lets the caller surface the
    actual exception message in the no-match reply so the user
    isn't left guessing "data fetch errors" with no other detail.
    """
    try:
        records = await fetcher(stock.symbol, interval, from_iso, to_iso)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("scanner|fetch_failed|symbol=%s|err=%s", stock.symbol, msg)
        return {"kind": "fetch_error", "error": msg[:240]}

    df = _df_from_records(records)
    if df is None or len(df) < 2:
        logger.info("scanner|insufficient_data|symbol=%s|rows=%s",
                    stock.symbol, 0 if df is None else len(df))
        return {"kind": "insufficient_data"}

    metrics = _compute_candidate_metrics(df, lookback_window_bars=metric_window)
    try:
        passed, first_failing = _evaluate_conditions_detailed(df, conditions)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("scanner|eval_failed|symbol=%s|err=%s", stock.symbol, msg)
        return {"kind": "fetch_error", "error": msg[:240]}

    if passed:
        return {"kind": "pass", "metrics": metrics}
    return {
        "kind": "fail", "metrics": metrics,
        "failed_condition": first_failing or "",
    }


async def _default_fetch_ohlcv(
    symbol: str,
    interval: str,
    from_utc: str,
    to_utc: str,
) -> list[dict[str, Any]]:
    """Bridge to the existing market-data fetcher. Late-imported so this
    module can be loaded without dragging in the backtest package's heavy
    dependency graph (which transitively pulls in the DB layer)."""
    from app.services.backtest.market_data import (
        StrategyMarketDataRequest,
        fetch_ohlcv_records,
    )

    request = StrategyMarketDataRequest(
        yaml_path="",
        raw_symbol=symbol,
        symbol=symbol.replace(".NS", "").replace(".BO", ""),
        interval=interval,
        from_utc=from_utc,
        to_utc=to_utc,
    )
    return await fetch_ohlcv_records(request)
