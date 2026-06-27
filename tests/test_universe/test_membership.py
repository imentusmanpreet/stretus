"""Phase C — survivorship-safe point-in-time constituents (§6, §7, Invariant 7).

The pure membership rule + the resolver integration, all DB-free via an injected store.
The headline test: a since-delisted name appears in a HISTORICAL resolution and is absent
TODAY — that is survivorship safety.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.universe.membership import (
    InMemoryMembershipStore,
    MembershipPoolProvider,
    MembershipRow,
    members_on,
    source_universe_key,
)
from app.services.universe.resolver import resolve_universe
from app.strategy.spec import UniverseSource, UniverseSpec


def _d(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# Seed: PRESENT is a current member; DELISTED left the index on 2022-06-01; LATECOMER joined 2023.
_ROWS = [
    MembershipRow("NIFTY500", "PRESENT", _d(2020, 1, 1), None),
    MembershipRow("NIFTY500", "DELISTED", _d(2020, 1, 1), _d(2022, 6, 1)),
    MembershipRow("NIFTY500", "LATECOMER", _d(2023, 1, 1), None),
    MembershipRow("BANKING", "HDFCBANK", _d(2020, 1, 1), None),
]


def _nifty500_rows() -> list[MembershipRow]:
    return [r for r in _ROWS if r.universe_key == "NIFTY500"]


def test_members_on_is_survivorship_safe():
    # 2021: PRESENT + DELISTED (DELISTED still a member); LATECOMER not yet joined.
    assert set(members_on(_nifty500_rows(), asof=_d(2021, 1, 1))) == {"PRESENT", "DELISTED"}
    # 2025: DELISTED gone, LATECOMER present.
    assert set(members_on(_nifty500_rows(), asof=_d(2025, 1, 1))) == {"PRESENT", "LATECOMER"}


def test_members_on_half_open_interval():
    # valid_to is EXCLUSIVE: on the exit date itself, the name is no longer a member.
    assert "DELISTED" not in members_on(_nifty500_rows(), asof=_d(2022, 6, 1))
    # one day before, it still is.
    assert "DELISTED" in members_on(_nifty500_rows(), asof=_d(2022, 5, 31))


def test_source_universe_key_mapping():
    # Keys are normalized (case/space-insensitive) so "NIFTY 500" == "NIFTY500" etc.
    assert source_universe_key(UniverseSource(kind="index", name="NIFTY 500")) == "NIFTY500"
    assert source_universe_key(UniverseSource(kind="index", name="NIFTY500")) == "NIFTY500"
    assert source_universe_key(UniverseSource(kind="sector", name="banking")) == "BANKING"
    assert source_universe_key(UniverseSource(kind="f_and_o")) == "F_AND_O"


async def test_provider_filters_by_universe_key_and_asof():
    prov = MembershipPoolProvider(InMemoryMembershipStore(_ROWS))
    assert prov.point_in_time is True
    nifty = await prov(UniverseSource(kind="index", name="NIFTY500"), _d(2021, 1, 1))
    assert set(nifty) == {"PRESENT", "DELISTED"}
    banking = await prov(UniverseSource(kind="sector", name="banking"), _d(2021, 1, 1))
    assert banking == ["HDFCBANK"]


async def test_resolver_stamps_point_in_time_survivorship_with_membership_provider():
    # A delisted name resolves into a historical snapshot, and the snapshot is marked
    # point_in_time automatically (auto-detected from the provider).
    asof = _d(2021, 1, 1)

    def _series(symbol):
        return [
            {"timestamp": (asof - timedelta(days=300 - i)).isoformat().replace("+00:00", "Z"),
             "open": 100, "high": 101, "low": 99, "close": 100 + i * 0.1, "volume": 1000}
            for i in range(300)
        ]

    pool = {"PRESENT": _series("PRESENT"), "DELISTED": _series("DELISTED")}

    async def fetch(symbol, interval, from_iso, to_iso):
        return pool.get(symbol, [])

    spec = UniverseSpec(source={"kind": "index", "name": "NIFTY500"},
                        rank={"by": "rvol"}, take=10, screen=["CLOSE > 0"])
    resolved = await resolve_universe(
        spec, asof=asof, fetch_ohlcv=fetch,
        pool_provider=MembershipPoolProvider(InMemoryMembershipStore(_nifty500_rows())),
    )
    assert resolved.survivorship_mode == "point_in_time"
    assert "DELISTED" in resolved.member_symbols  # present historically


async def test_resolver_defaults_to_approximate_without_point_in_time_provider():
    # A watchlist source (no membership provider) → approximate survivorship.
    asof = _d(2025, 1, 1)

    def _series():
        return [
            {"timestamp": (asof - timedelta(days=300 - i)).isoformat().replace("+00:00", "Z"),
             "open": 100, "high": 101, "low": 99, "close": 100 + i * 0.1, "volume": 1000}
            for i in range(300)
        ]

    pool = {"AAA": _series(), "BBB": _series()}

    async def fetch(symbol, interval, from_iso, to_iso):
        return pool.get(symbol, [])

    spec = UniverseSpec(source={"kind": "watchlist", "symbols": ["AAA", "BBB"]},
                        rank={"by": "rvol"}, take=10, screen=["CLOSE > 0"])
    resolved = await resolve_universe(spec, asof=asof, fetch_ohlcv=fetch)
    assert resolved.survivorship_mode == "approximate"
