"""
Phase-A resolver tests (KB-free), with an injected OHLCV fetcher — no network (§16).

Covers: determinism + snapshot hash, the asset-count cap applied AFTER ranking (§7.1),
no look-ahead (bars after asof never change the result), fail-closed eligibility (ADV floor),
screen filtering, the breadth gate, the scale guard, and the source registry.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import Settings
from app.services.universe.errors import ScaleGuardExceeded, UniverseSourceError
from app.services.universe.resolver import resolve_universe
from app.services.universe import sources
from app.strategy.spec import UniverseSource, UniverseSpec

ASOF = datetime(2025, 6, 1, tzinfo=timezone.utc)

pytestmark = pytest.mark.asyncio


def _rec(ts: datetime, close: float, vol: float) -> dict:
    return {
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "open": close, "high": close + 1, "low": max(close - 1, 0.1),
        "close": close, "volume": vol,
    }


def _series(symbol: str, *, base_vol: float, last_vol_mult: float = 1.5, n: int = 300,
            with_future_spike: bool = True) -> list[dict]:
    """n daily bars ending at ASOF, plus (optionally) one bar AFTER asof that a correct
    range-sliced read must ignore (look-ahead guard)."""
    recs = []
    for i in range(n):
        ts = ASOF - timedelta(days=n - 1 - i)
        vol = base_vol * (last_vol_mult if i == n - 1 else 1.0)
        recs.append(_rec(ts, close=100.0 + 0.1 * i, vol=vol))
    if with_future_spike:
        recs.append(_rec(ASOF + timedelta(days=1), close=9999.0, vol=base_vol * 1000))
    return recs


def _make_fetcher(pool: dict[str, list[dict]]):
    async def fetch(symbol, interval, from_iso, to_iso):
        to_dt = datetime.fromisoformat(to_iso.replace("Z", "+00:00"))
        recs = pool.get(symbol, [])
        # A correct store range-slices by time; this enforces the no-look-ahead contract.
        return [r for r in recs
                if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) <= to_dt]
    return fetch


async def _provider(pool: dict[str, list[dict]]):
    async def pp(source, asof):
        return list(pool.keys())
    return pp


def _settings(**over) -> Settings:
    base = dict(dynamic_universe_max_assets=2, dynamic_universe_max_positions=2,
                dynamic_universe_max_pool=5000)
    base.update(over)
    return Settings(**base)


def _spec(**over) -> UniverseSpec:
    u = dict(source={"kind": "index", "name": "NIFTY500"},
             rank={"by": "rvol", "order": "desc"}, take=10, screen=["CLOSE > 0"])
    u.update(over)
    return UniverseSpec(**u)


async def test_determinism_same_inputs_same_hash():
    pool = {f"S{i}": _series(f"S{i}", base_vol=1000 * (i + 1)) for i in range(5)}
    fetch, pp = _make_fetcher(pool), await _provider(pool)
    r1 = await resolve_universe(_spec(), asof=ASOF, fetch_ohlcv=fetch, pool_provider=pp, settings=_settings())
    r2 = await resolve_universe(_spec(), asof=ASOF, fetch_ohlcv=fetch, pool_provider=pp, settings=_settings())
    assert r1.snapshot_hash == r2.snapshot_hash
    assert r1.member_symbols == r2.member_symbols


async def test_asset_count_cap_applied_after_ranking():
    # Distinct rvol so ranking is unambiguous: higher index → higher last-bar volume mult.
    pool = {f"S{i}": _series(f"S{i}", base_vol=1000, last_vol_mult=1.0 + i) for i in range(6)}
    fetch, pp = _make_fetcher(pool), await _provider(pool)
    spec = _spec(take=1000)  # user asks for 1000
    r = await resolve_universe(spec, asof=ASOF, fetch_ohlcv=fetch, pool_provider=pp,
                               settings=_settings(dynamic_universe_max_assets=2))
    assert r.requested_take == 1000
    assert r.effective_take == 2
    assert r.cap_reason == "platform_limit"
    assert len(r.members) == 2
    # The TWO kept are the top-2 by rvol (S5, S4), proving the cap is AFTER ranking.
    assert r.member_symbols == ["S5", "S4"]
    assert r.cap_message() == "Requested 1000; platform limit applied → using 2."


async def test_no_look_ahead_future_bars_ignored():
    pool_a = {f"S{i}": _series(f"S{i}", base_vol=1000, last_vol_mult=1.0 + i, with_future_spike=True)
              for i in range(4)}
    pool_b = {f"S{i}": _series(f"S{i}", base_vol=1000, last_vol_mult=1.0 + i, with_future_spike=False)
              for i in range(4)}
    r_a = await resolve_universe(_spec(take=2), asof=ASOF, fetch_ohlcv=_make_fetcher(pool_a),
                                 pool_provider=await _provider(pool_a), settings=_settings())
    r_b = await resolve_universe(_spec(take=2), asof=ASOF, fetch_ohlcv=_make_fetcher(pool_b),
                                 pool_provider=await _provider(pool_b), settings=_settings())
    # The post-asof spike bar must not change membership OR the hash.
    assert r_a.snapshot_hash == r_b.snapshot_hash


async def test_eligibility_adv_floor_fail_closed():
    # S0 has tiny volume (below the floor); S1/S2 are liquid.
    pool = {
        "S0": _series("S0", base_vol=1.0, last_vol_mult=10.0),  # high rvol but illiquid
        "S1": _series("S1", base_vol=100000.0, last_vol_mult=1.2),
        "S2": _series("S2", base_vol=200000.0, last_vol_mult=1.1),
    }
    spec = _spec(take=5, eligibility={"min_adv": 1_000_000.0, "adv_window": 20})
    r = await resolve_universe(spec, asof=ASOF, fetch_ohlcv=_make_fetcher(pool),
                               pool_provider=await _provider(pool), settings=_settings(dynamic_universe_max_assets=5))
    assert "S0" not in r.member_symbols
    assert any(d.symbol == "S0" and d.reason == "ineligible_adv" for d in r.dropped)


async def test_screen_filters_pool():
    pool = {f"S{i}": _series(f"S{i}", base_vol=1000) for i in range(4)}
    # An impossible screen drops everyone.
    spec = _spec(take=5, screen=["CLOSE < 0"])
    r = await resolve_universe(spec, asof=ASOF, fetch_ohlcv=_make_fetcher(pool),
                               pool_provider=await _provider(pool), settings=_settings())
    assert r.members == []
    assert all(d.reason == "screen_failed" for d in r.dropped)


async def test_breadth_gate_blocks_when_failing():
    # All names falling → advance/decline ratio 0 → gate (min_ratio 1.0) fails → no members.
    pool = {}
    for i in range(4):
        recs = []
        for j in range(60):
            ts = ASOF - timedelta(days=60 - 1 - j)
            recs.append(_rec(ts, close=200.0 - j, vol=1000))  # strictly falling
        pool[f"S{i}"] = recs
    spec = _spec(take=5, screen=[], breadth={"metric": "advance_decline", "min_ratio": 1.0})
    r = await resolve_universe(spec, asof=ASOF, fetch_ohlcv=_make_fetcher(pool),
                               pool_provider=await _provider(pool), settings=_settings(dynamic_universe_max_assets=5))
    assert r.breadth_ok is False
    assert r.members == []
    assert any(d.reason == "breadth_blocked" for d in r.dropped)


async def test_scale_guard_refuses_oversized_pool():
    pool = {f"S{i}": [] for i in range(10)}
    spec = _spec()
    with pytest.raises(ScaleGuardExceeded):
        await resolve_universe(spec, asof=ASOF, fetch_ohlcv=_make_fetcher(pool),
                               pool_provider=await _provider(pool),
                               settings=_settings(dynamic_universe_max_pool=5))


async def test_watchlist_source_resolves_without_provider():
    pool = {"AAA": _series("AAA", base_vol=1000), "BBB": _series("BBB", base_vol=2000)}
    spec = _spec(source={"kind": "watchlist", "symbols": ["AAA", "BBB"]})
    r = await resolve_universe(spec, asof=ASOF, fetch_ohlcv=_make_fetcher(pool),
                               pool_provider=None, settings=_settings())
    assert set(r.member_symbols) <= {"AAA", "BBB"}


async def test_index_source_without_provider_raises():
    src = UniverseSource(kind="index", name="NIFTY500")
    with pytest.raises(UniverseSourceError):
        await sources.resolve_pool(src, asof=ASOF, pool_provider=None)
