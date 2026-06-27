"""Phase B — Tier-B execution-data loading + membership windows (§8 step 4).

Uses the injectable CachingFetchProvider (no network). Confirms members' execution frames
load lazily and that membership windows are derived correctly from a refresh sequence.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.data.provider import CachingFetchProvider
from app.services.universe.backtest_orchestrator import (
    load_member_execution_frames,
    membership_windows_from_snapshots,
    run_dynamic_backtest,
)
from app.services.universe.membership import InMemoryMembershipStore, MembershipPoolProvider, MembershipRow
from app.services.universe.types import ResolvedMember, ResolvedUniverse
from app.strategy.spec import UniverseSpec

pytestmark = pytest.mark.asyncio

ASOF = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _resolved(symbols: list[str], asof: datetime = ASOF) -> ResolvedUniverse:
    return ResolvedUniverse(
        asof=asof.isoformat().replace("+00:00", "Z"),
        members=[ResolvedMember(symbol=s, rank=i + 1, rank_value=float(10 - i))
                 for i, s in enumerate(symbols)],
        requested_take=len(symbols), effective_take=len(symbols),
    )


async def test_load_member_execution_frames_only_for_members():
    pool = {
        "AAA": [{"timestamp": (ASOF - timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                 "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1} for i in range(10)],
        "BBB": [],  # no data → omitted
    }

    async def fetch(symbol, interval, from_iso, to_iso):
        return pool.get(symbol, [])

    provider = CachingFetchProvider(fetch)
    out = await load_member_execution_frames(
        _resolved(["AAA", "BBB"]), provider=provider, timeframe="1d",
        window_from=ASOF - timedelta(days=10), window_to=ASOF)
    assert "AAA" in out and "BBB" not in out
    assert len(out["AAA"]) > 0


async def test_membership_windows_open_and_close():
    t0 = ASOF
    t1 = ASOF + timedelta(days=7)
    t2 = ASOF + timedelta(days=14)
    horizon = ASOF + timedelta(days=21)
    snaps = [
        _resolved(["AAA", "BBB"], t0),   # AAA, BBB active
        _resolved(["AAA", "CCC"], t1),   # BBB dropped, CCC added
        _resolved(["AAA", "CCC"], t2),   # unchanged
    ]
    windows = membership_windows_from_snapshots(snaps, horizon_to=horizon)
    # BBB active t0→t1, then dropped.
    assert windows["BBB"] == [(snaps[0].asof, snaps[1].asof)]
    # AAA active the whole time → open until horizon.
    assert windows["AAA"][0][0] == snaps[0].asof
    assert windows["AAA"][-1][1] == horizon.isoformat().replace("+00:00", "Z")
    # CCC added at t1, still active → open until horizon.
    assert windows["CCC"][0][0] == snaps[1].asof


def _series_records(n: int = 300, base: float = 100.0) -> list[dict]:
    return [
        {"timestamp": (ASOF - timedelta(days=n - 1 - i)).isoformat().replace("+00:00", "Z"),
         "open": base, "high": base + 1, "low": base - 1, "close": base + i * 0.1, "volume": 1000}
        for i in range(n)
    ]


async def test_run_dynamic_backtest_resolves_loads_and_calls_engine():
    pool = {"AAA": _series_records(), "BBB": _series_records(base=200.0)}

    async def fetch(symbol, interval, from_iso, to_iso):
        return pool.get(symbol, [])

    captured = {}

    async def fake_engine(payload):
        captured.update(payload)
        return {"members": sorted(payload["member_ohlcv"].keys()),
                "metrics": {"total_trades": 0.0}, "equity_curve": [],
                "survivorship_mode": payload["survivorship_mode"]}

    spec = UniverseSpec(source={"kind": "watchlist", "symbols": ["AAA", "BBB"]},
                        rank={"by": "rvol"}, take=10, screen=["CLOSE > 0"])
    result = await run_dynamic_backtest(
        universe=spec,
        template_yaml="strategy:\n  symbol: PLACEHOLDER\n",
        window_from=ASOF - timedelta(days=200), window_to=ASOF, timeframe="1d",
        fetch_ohlcv=fetch,
        market_data_request={"symbol": "PLACEHOLDER", "interval": "1d",
                             "from_utc": "x", "to_utc": "y"},
        engine_call=fake_engine,
    )
    # The engine got the resolved members, membership windows, and survivorship mode.
    assert set(captured["member_ohlcv"].keys()) == {"AAA", "BBB"}
    assert set(captured["membership_windows"].keys()) == {"AAA", "BBB"}
    assert captured["survivorship_mode"] == "approximate"  # watchlist → approximate
    assert result["members"] == ["AAA", "BBB"]


async def test_run_dynamic_backtest_point_in_time_with_membership_provider():
    asof = datetime(2021, 1, 1, tzinfo=timezone.utc)

    def _hist(n=300):
        return [
            {"timestamp": (asof - timedelta(days=n - 1 - i)).isoformat().replace("+00:00", "Z"),
             "open": 100, "high": 101, "low": 99, "close": 100 + i * 0.1, "volume": 1000}
            for i in range(n)
        ]

    pool = {"PRESENT": _hist(), "DELISTED": _hist()}

    async def fetch(symbol, interval, from_iso, to_iso):
        return pool.get(symbol, [])

    async def fake_engine(payload):
        return {"members": sorted(payload["member_ohlcv"].keys()),
                "metrics": {"total_trades": 0.0},
                "survivorship_mode": payload["survivorship_mode"]}

    rows = [
        MembershipRow("NIFTY500", "PRESENT", datetime(2020, 1, 1, tzinfo=timezone.utc), None),
        MembershipRow("NIFTY500", "DELISTED", datetime(2020, 1, 1, tzinfo=timezone.utc),
                      datetime(2022, 6, 1, tzinfo=timezone.utc)),
    ]
    spec = UniverseSpec(source={"kind": "index", "name": "NIFTY500"},
                        rank={"by": "rvol"}, take=10, screen=["CLOSE > 0"])
    result = await run_dynamic_backtest(
        universe=spec, template_yaml="strategy:\n  symbol: X\n",
        window_from=asof - timedelta(days=200), window_to=asof, asof=asof, timeframe="1d",
        fetch_ohlcv=fetch,
        market_data_request={"symbol": "X", "interval": "1d", "from_utc": "x", "to_utc": "y"},
        pool_provider=MembershipPoolProvider(InMemoryMembershipStore(rows)),
        engine_call=fake_engine,
    )
    assert result["survivorship_mode"] == "point_in_time"
    assert "DELISTED" in result["members"]
