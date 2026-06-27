"""Resilient + backtest-feed market-data clients (production data-source robustness).

The resilient chain must: serve the primary when it works, fall back on a RAISED failure, pass
through a legitimate ``None`` (no-circuit) without falling back, and re-raise when all sources
fail. The factory must honor EQUITY_MARKET_DATA_SOURCE.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.execution.market_data.base import MarketDataClient
from app.services.execution.market_data.resilient_client import ResilientMarketDataClient


class _Stub(MarketDataClient):
    def __init__(self, adapter_id, *, candles=None, ltp=None, circuit=None, raises=False):
        self.adapter_id = adapter_id
        self._candles, self._ltp, self._circuit, self._raises = candles, ltp, circuit, raises
        self.calls = 0

    async def fetch_candles(self, symbol, timeframe, lookback, adapter_symbol=None):
        self.calls += 1
        if self._raises:
            raise RuntimeError(f"{self.adapter_id} down")
        return self._candles

    async def fetch_ltp(self, symbol, adapter_symbol=None):
        self.calls += 1
        if self._raises:
            raise RuntimeError(f"{self.adapter_id} down")
        return self._ltp

    async def fetch_circuit_limits(self, symbol, adapter_symbol=None):
        self.calls += 1
        if self._raises:
            raise RuntimeError(f"{self.adapter_id} down")
        return self._circuit


def _df():
    return pd.DataFrame({"close": [100.0]})


async def test_primary_served_when_healthy_no_fallback_call():
    primary = _Stub("upstox_rest", ltp=101.0)
    fallback = _Stub("backtest_feed", ltp=99.0)
    client = ResilientMarketDataClient([primary, fallback])
    assert await client.fetch_ltp("X") == 101.0
    assert primary.calls == 1 and fallback.calls == 0   # fallback untouched


async def test_candles_fall_back_on_primary_failure():
    # Candles are closed bars → safe to fall back to the historical feed.
    primary = _Stub("upstox_rest", raises=True)
    fallback = _Stub("backtest_feed", candles=_df())
    client = ResilientMarketDataClient([primary, fallback])
    out = await client.fetch_candles("X", "5m", 10)
    assert out is not None
    assert primary.calls == 1 and fallback.calls == 1


async def test_ltp_does_NOT_fall_back_by_default_fails_closed():
    # LTP is the price we trade on — it must be live, never a stale historical close.
    primary = _Stub("upstox_rest", raises=True)
    fallback = _Stub("backtest_feed", ltp=99.0)
    client = ResilientMarketDataClient([primary, fallback])   # ltp_fallback defaults False
    with pytest.raises(RuntimeError):
        await client.fetch_ltp("X")
    assert fallback.calls == 0          # historical feed NEVER consulted for LTP


async def test_ltp_falls_back_only_when_explicitly_enabled():
    primary = _Stub("upstox_rest", raises=True)
    fallback = _Stub("backtest_feed", ltp=99.0)
    client = ResilientMarketDataClient([primary, fallback], ltp_fallback=True)
    assert await client.fetch_ltp("X") == 99.0


def test_adapter_id_notes_ltp_is_primary_only():
    client = ResilientMarketDataClient([_Stub("upstox_rest"), _Stub("backtest_feed")])
    assert "ltp=upstox_rest" in client.adapter_id


async def test_none_is_a_valid_result_not_a_failure():
    # fetch_circuit_limits returning None (venue exposes no bands) must NOT trigger a fallback.
    primary = _Stub("upstox_rest", circuit=None)
    fallback = _Stub("backtest_feed", circuit={"upper_circuit": 1.0, "lower_circuit": 0.5})
    client = ResilientMarketDataClient([primary, fallback])
    assert await client.fetch_circuit_limits("X") is None
    assert fallback.calls == 0                            # not consulted — None was valid


async def test_reraises_when_all_sources_fail():
    client = ResilientMarketDataClient([_Stub("a", raises=True), _Stub("b", raises=True)])
    with pytest.raises(RuntimeError):
        await client.fetch_candles("X", "5m", 10)


def test_adapter_id_describes_the_chain():
    client = ResilientMarketDataClient([_Stub("upstox_rest"), _Stub("backtest_feed")])
    assert client.adapter_id == "resilient(upstox_rest→backtest_feed; ltp=upstox_rest)"


def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        ResilientMarketDataClient([])


# ── factory honors the policy ─────────────────────────────────────────────────
@pytest.mark.parametrize("policy,expected", [
    # "upstox" = live broker feed, now reached through the stretus-backend gateway.
    ("upstox", "InternalMarketDataClient"),
    ("backtest_feed", "BacktestFeedClient"),
    ("resilient", "ResilientMarketDataClient"),
])
def test_factory_honors_equity_source_policy(monkeypatch, policy, expected):
    import app.services.execution.market_data.factory as fac
    import app.services.execution.market_data.internal_client as internal
    from app.schemas.execution import AssetClass

    # Live equity now flows through the gateway (InternalMarketDataClient), so the policy
    # object must carry the gateway settings the client reads at construction time.
    fake_settings = type("S", (), {
        "equity_market_data_source": policy,
        "historical_data_url": "http://gateway.test",
        "market_data_timeout_seconds": 5.0,
    })()
    monkeypatch.setattr(fac, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(internal, "get_settings", lambda: fake_settings)
    fac.reset_market_data_clients()
    client = fac.get_market_data_client(AssetClass.equity_cash)
    assert client.__class__.__name__ == expected
    fac.reset_market_data_clients()
