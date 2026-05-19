"""Phase 9 — scanner + orchestrator with mocked OHLCV fetching.

The scanner normally calls app.services.backtest.market_data.fetch_ohlcv_records.
We inject a stub fetcher so tests don't hit HTTP. Each candidate's "fetched"
OHLCV is constructed deterministically so we can predict pass/fail per
condition.
"""
from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

from app.kb import kb
from app.services.discovery.scanner import scan_universe
from app.services.discovery.types import DiscoveryConfig, TieBreakOption


def _bar(ts, op, hi, lo, cl, vol):
    return {"timestamp": ts.isoformat().replace("+00:00", "Z"),
            "open": op, "high": hi, "low": lo, "close": cl, "volume": vol}


def _series(start_close: float, n: int, *, end_volume_mult: float = 1.0, ramp_to_high: bool = False) -> list[dict]:
    """Build n daily bars ending today.

    Default shape: a saw-tooth that rises to a peak around bar 100 then
    falls back, leaving the LAST bar well below the 252-bar high (so
    'near 52w high' is False by default). `ramp_to_high=True` overrides the
    last bar with a fresh peak so the proximity condition fires.
    `end_volume_mult` scales the LAST bar's volume.
    """
    rows = []
    base_ts = pd.Timestamp("2026-05-12 09:30", tz="UTC")
    for i in range(n):
        ts = base_ts - pd.Timedelta(days=(n - 1 - i))
        # Saw-tooth: rise for first 100 bars, then decay back toward baseline.
        if i < 100:
            close = start_close + i * 0.5            # peaks near 150
        else:
            close = max(start_close, 150.0 - (i - 100) * 0.3)
        if ramp_to_high and i == n - 1:
            close = 200.0       # well above the saw-tooth peak → new 52w high
        vol = 100_000.0 if i < n - 1 else 100_000.0 * end_volume_mult
        rows.append(_bar(ts, close, close + 0.5, close - 0.5, close, vol))
    return rows


# ── Scanner with stubbed fetcher ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scanner_returns_single_when_only_one_matches():
    """Stub the fetcher so exactly ONE of the KB universe stocks has a
    today-volume 3× its 20-bar average. Conditions filter by volume spike."""
    target_symbol = "HDFCBANK.NS"

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        # Only the target gets a volume spike at the last bar.
        mult = 3.0 if symbol == target_symbol else 1.0
        return _series(start_close=100.0, n=300, end_volume_mult=mult)

    discovery = DiscoveryConfig(
        enabled=True,
        conditions=["VOL > AVG(VOL, 20) * 2.0"],
        tie_break_options=[
            TieBreakOption(method="highest_relative_volume", label="Highest relative volume"),
        ],
    )
    result = await scan_universe(discovery, interval="1d", fetch_ohlcv=stub_fetcher,
                                  asof_utc=datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc))
    assert result.status == "single"
    assert [c.symbol for c in result.candidates] == [target_symbol]
    # tie_break_options is empty when status != "multiple"
    assert result.tie_break_options == []


@pytest.mark.asyncio
async def test_scanner_returns_none_when_zero_match():
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _series(start_close=100.0, n=300, end_volume_mult=1.0)   # no spike

    discovery = DiscoveryConfig(
        enabled=True,
        conditions=["VOL > AVG(VOL, 20) * 2.0"],
    )
    result = await scan_universe(discovery, interval="1d", fetch_ohlcv=stub_fetcher)
    assert result.status == "none"
    assert result.candidates == []


@pytest.mark.asyncio
async def test_scanner_returns_multiple_when_many_match():
    """All universe stocks get a volume spike → all qualify → status=multiple."""
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _series(start_close=100.0, n=300, end_volume_mult=3.0)

    discovery = DiscoveryConfig(
        enabled=True,
        conditions=["VOL > AVG(VOL, 20) * 2.0"],
    )
    result = await scan_universe(discovery, interval="1d", fetch_ohlcv=stub_fetcher)
    assert result.status == "multiple"
    assert len(result.candidates) > 1
    # tie_break_options materialised from defaults (preset didn't declare any)
    assert len(result.tie_break_options) > 0


@pytest.mark.asyncio
async def test_scanner_tolerates_per_symbol_fetch_failures():
    """One symbol's fetcher raises — the scanner must still process the rest."""
    fail_symbol = next(iter(kb.stocks.keys()))

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        if symbol == fail_symbol:
            raise RuntimeError("simulated outage")
        return _series(start_close=100.0, n=300, end_volume_mult=3.0)

    discovery = DiscoveryConfig(enabled=True, conditions=["VOL > AVG(VOL, 20) * 2.0"])
    result = await scan_universe(discovery, interval="1d", fetch_ohlcv=stub_fetcher)
    assert result.failed_fetches >= 1
    # The successful symbols should still appear in the candidate list.
    assert fail_symbol not in {c.symbol for c in result.candidates}


@pytest.mark.asyncio
async def test_scanner_pre_computes_metrics_for_tie_break():
    """Each candidate must carry the metrics tie-break methods read off."""
    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        return _series(start_close=100.0, n=300, end_volume_mult=3.0)

    discovery = DiscoveryConfig(enabled=True, conditions=["VOL > AVG(VOL, 20) * 2.0"])
    result = await scan_universe(discovery, interval="1d", fetch_ohlcv=stub_fetcher)
    assert result.candidates
    c = result.candidates[0]
    # All canonical metric keys must be present (NaN is fine; we test for
    # KEYS not values so a tie-break ranker can rely on the schema).
    for key in ("relative_volume", "close", "distance_to_52w_high_pct",
                "distance_to_52w_low_pct", "rsi_14", "atr_14_pct"):
        assert key in c.metrics


@pytest.mark.asyncio
async def test_scanner_rejects_when_discovery_has_no_conditions():
    """A misconfigured DiscoveryConfig with no conditions must fail loudly."""
    discovery = DiscoveryConfig(enabled=True, conditions=[])
    with pytest.raises(ValueError, match="no conditions"):
        await scan_universe(discovery, interval="1d", fetch_ohlcv=None)


@pytest.mark.asyncio
async def test_scanner_evaluates_multiple_conditions_with_AND_semantics():
    """A candidate must pass ALL conditions; failing any one excludes it."""
    target_symbol = "HDFCBANK.NS"

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        # Only the target stock gets both the volume spike AND the 52w-high
        # ramp. Others get just the volume spike but stay far from their high.
        if symbol == target_symbol:
            return _series(start_close=100.0, n=300, end_volume_mult=3.0, ramp_to_high=True)
        return _series(start_close=100.0, n=300, end_volume_mult=3.0)

    discovery = DiscoveryConfig(
        enabled=True,
        conditions=[
            "VOL > AVG(VOL, 20) * 2.0",
            "CLOSE >= MAX(HIGH, 252) * 0.98",
        ],
    )
    result = await scan_universe(discovery, interval="1d", fetch_ohlcv=stub_fetcher)
    # Only the target passes both conditions.
    assert result.status == "single"
    assert result.candidates[0].symbol == target_symbol
