from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.services.backtest import market_data
from app.services.backtest.market_data import StrategyMarketDataRequest
from app.services.backtest.market_data_grpc import (
    bars_to_record_dicts,
    use_grpc_transport,
    validate_grpc_interval,
)


def test_use_grpc_transport_auto_with_target() -> None:
    cfg = Settings(market_data_fetch_transport="auto", market_data_grpc_target="mds:50057")
    assert use_grpc_transport(cfg) is True


def test_use_grpc_transport_auto_without_target() -> None:
    cfg = Settings(market_data_fetch_transport="auto", market_data_grpc_target="")
    assert use_grpc_transport(cfg) is False


def test_use_grpc_transport_http() -> None:
    cfg = Settings(market_data_fetch_transport="http", market_data_grpc_target="mds:50057")
    assert use_grpc_transport(cfg) is False


def test_validate_grpc_interval_rejects_10m() -> None:
    with pytest.raises(ValueError, match="not supported"):
        validate_grpc_interval("10m")


def test_bars_to_record_dicts() -> None:
    bar = MagicMock(ts="2024-01-02T09:15:00+05:30", open=1.0, high=2.0, low=0.5, close=1.5, volume=100)
    assert bars_to_record_dicts([bar]) == [
        {"ts": "2024-01-02T09:15:00+05:30", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100}
    ]


def test_fetch_ohlcv_records_uses_grpc_when_configured(monkeypatch) -> None:
    asyncio.run(_fetch_ohlcv_records_uses_grpc_when_configured(monkeypatch))


async def _fetch_ohlcv_records_uses_grpc_when_configured(monkeypatch) -> None:
    request = StrategyMarketDataRequest(
        yaml_path="",
        raw_symbol="TCS",
        symbol="TCS",
        interval="1d",
        from_utc="2024-01-01T00:00:00Z",
        to_utc="2024-01-15T00:00:00Z",
    )

    async def fake_chunk_grpc(*args, **kwargs):
        return [{"ts": "2024-01-02T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]

    class FakeChannel:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(market_data, "get_cached_records", lambda *a, **k: None)
    monkeypatch.setattr(market_data, "save_ohlcv_cache", lambda *a, **k: None)
    monkeypatch.setattr(market_data.settings, "market_data_fetch_transport", "grpc")
    monkeypatch.setattr(market_data.settings, "market_data_grpc_target", "marketdata-ingestion:50057")
    monkeypatch.setattr(market_data.settings, "backtest_fetch_chunk_days", 90)
    monkeypatch.setattr(market_data, "open_grpc_channel", lambda *a, **k: FakeChannel())
    monkeypatch.setattr(market_data, "fetch_ohlcv_chunk_grpc", fake_chunk_grpc)
    monkeypatch.setattr(
        "app.proto.gen.marketdata_pb2_grpc.MarketDataServiceStub",
        lambda channel: MagicMock(),
    )

    rows = await market_data.fetch_ohlcv_records(request)
    assert len(rows) == 1
    assert rows[0]["timestamp"] == "2024-01-02T00:00:00Z"
    assert rows[0]["close"] == 1.5
