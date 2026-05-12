from __future__ import annotations

import pytest

from app.services.backtest import market_data
from app.schemas.backtest import BacktestTriggerRequest
from app.services.backtest.market_data import (
    StrategyMarketDataRequest,
    _build_ohlcv_fetch_slots,
    _merge_ohlcv_slot_rows,
    extract_strategy_market_data_request,
    normalize_ohlcv_payload,
)
from app.services.strategy.builder import StrategyBuilder, extract_strategy_details
from quant_engine.engine.config import (
    BACKTEST_MARKET_DATA_FROM_UTC,
    BACKTEST_MARKET_DATA_TO_UTC,
)


def test_extract_strategy_market_data_request_enforces_configured_window(tmp_path):
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "strategy:",
                "  symbol: RELIANCE.NS",
                "  timeframe: 15m",
            ]
        ),
        encoding="utf-8",
    )

    request = extract_strategy_market_data_request(
        str(yaml_path),
        overrides=BacktestTriggerRequest(
            strategy_id="00000000-0000-0000-0000-000000000001",
            from_utc="2026-03-02T03:45:00Z",
            to_utc="2026-03-02T09:59:00Z",
            symbol="TCS",
            interval="5m",
        ),
    )

    assert request == StrategyMarketDataRequest(
        yaml_path=str(yaml_path),
        raw_symbol="TCS",
        symbol="TCS",
        interval="5m",
        from_utc=BACKTEST_MARKET_DATA_FROM_UTC,
        to_utc=BACKTEST_MARKET_DATA_TO_UTC,
    )


def test_extract_strategy_market_data_request_uses_configured_default_window(tmp_path):
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "strategy:",
                "  symbol: INFY.NS",
                "  timeframe: 15m",
            ]
        ),
        encoding="utf-8",
    )

    request = extract_strategy_market_data_request(
        str(yaml_path),
        overrides=BacktestTriggerRequest(strategy_id="00000000-0000-0000-0000-000000000001"),
    )

    assert request.symbol == "INFY"
    assert request.interval == "15m"
    assert request.from_utc == BACKTEST_MARKET_DATA_FROM_UTC
    assert request.to_utc == BACKTEST_MARKET_DATA_TO_UTC


def test_extract_strategy_market_data_request_rejects_inverted_window(tmp_path):
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "strategy:",
                "  symbol: TCS.NS",
                "  timeframe: 15m",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="from_utc must be earlier than to_utc"):
        extract_strategy_market_data_request(
            str(yaml_path),
            overrides=BacktestTriggerRequest(
                strategy_id="00000000-0000-0000-0000-000000000001",
                from_utc="2026-03-02T09:59:00Z",
                to_utc="2026-03-02T03:45:00Z",
            ),
        )


def test_extract_strategy_details_ignores_optional_input_phrases() -> None:
    builder = StrategyBuilder()

    extract_strategy_details("Daily loss cap: 2% Max trades: 3-5 per day", builder)

    assert builder.timeframe is None
    assert builder.daily_loss_cap is None
    assert builder.max_trade is None


def test_build_ohlcv_fetch_slots_uses_six_month_backtest_windows() -> None:
    slots = _build_ohlcv_fetch_slots(
        BACKTEST_MARKET_DATA_FROM_UTC,
        BACKTEST_MARKET_DATA_TO_UTC,
    )

    assert [(slot.index, slot.from_utc, slot.to_utc) for slot in slots] == [
        (1, "2024-01-01T00:00:00Z", "2024-06-30T23:59:59Z"),
        (2, "2024-07-01T00:00:00Z", "2024-12-31T23:59:59Z"),
        (3, "2025-01-01T00:00:00Z", "2025-06-30T23:59:59Z"),
        (4, "2025-07-01T00:00:00Z", "2025-12-31T23:59:59Z"),
        (5, "2026-01-01T00:00:00Z", "2026-03-31T23:59:59Z"),
    ]


def test_merge_ohlcv_slot_rows_sorts_and_deduplicates_timestamps() -> None:
    rows = _merge_ohlcv_slot_rows(
        [
            [
                {
                    "timestamp": "2024-07-01T03:45:00Z",
                    "open": 102.0,
                    "high": 105.0,
                    "low": 101.0,
                    "close": 104.0,
                    "volume": 2000.0,
                }
            ],
            [
                {
                    "timestamp": "2024-01-01T03:45:00Z",
                    "open": 100.0,
                    "high": 103.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1200.0,
                },
                {
                    "timestamp": "2024-07-01T03:45:00Z",
                    "open": 102.0,
                    "high": 106.0,
                    "low": 101.0,
                    "close": 105.0,
                    "volume": 2100.0,
                },
            ],
        ]
    )

    assert [row["timestamp"] for row in rows] == [
        "2024-01-01T03:45:00Z",
        "2024-07-01T03:45:00Z",
    ]
    assert rows[1]["close"] == 105.0


@pytest.mark.asyncio
async def test_fetch_ohlcv_records_continues_when_a_slot_fails(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint: str, params: dict):
            calls.append(dict(params))
            if "2024-07-01T00:00:00Z" <= params["from"] <= "2024-12-31T23:59:59Z":
                raise RuntimeError("temporary upstream failure")
            return FakeResponse(
                {
                    "data": [
                        {
                            "timestamp": params["from"],
                            "open": 100,
                            "high": 105,
                            "low": 99,
                            "close": 104,
                            "volume": 1500,
                        }
                    ]
                }
            )

    monkeypatch.setattr(market_data.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(market_data.settings, "historical_data_url", "http://data.test")
    monkeypatch.setattr(market_data, "OHLCV_SLOT_FETCH_DELAY_SECONDS", 0)

    rows = await market_data.fetch_ohlcv_records(
        StrategyMarketDataRequest(
            yaml_path="",
            raw_symbol="TCS.NS",
            symbol="TCS",
            interval="1d",
            from_utc=BACKTEST_MARKET_DATA_FROM_UTC,
            to_utc=BACKTEST_MARKET_DATA_TO_UTC,
        )
    )

    assert len(calls) == 11
    assert len(rows) == 4
    assert calls[0]["from"] == "2024-01-01T00:00:00Z"
    assert calls[-1]["to"] == "2026-03-31T23:59:59Z"
    assert all(not ("2024-07" <= row["timestamp"][:7] <= "2024-12") for row in rows)


@pytest.mark.asyncio
async def test_fetch_ohlcv_records_recovers_first_half_with_monthly_fallback(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint: str, params: dict):
            calls.append(dict(params))
            is_first_half_request = (
                params["from"] == "2024-01-01T00:00:00Z"
                and params["to"] == "2024-06-30T23:59:59Z"
            )
            if is_first_half_request:
                raise RuntimeError("range too large")
            return FakeResponse(
                {
                    "data": [
                        {
                            "timestamp": params["from"],
                            "open": 100,
                            "high": 105,
                            "low": 99,
                            "close": 104,
                            "volume": 1500,
                        }
                    ]
                }
            )

    monkeypatch.setattr(market_data.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(market_data.settings, "historical_data_url", "http://data.test")
    monkeypatch.setattr(market_data, "OHLCV_SLOT_FETCH_DELAY_SECONDS", 0)

    rows = await market_data.fetch_ohlcv_records(
        StrategyMarketDataRequest(
            yaml_path="",
            raw_symbol="TCS.NS",
            symbol="TCS",
            interval="1d",
            from_utc=BACKTEST_MARKET_DATA_FROM_UTC,
            to_utc=BACKTEST_MARKET_DATA_TO_UTC,
        )
    )

    assert len(calls) == 11
    assert len(rows) == 10
    assert [row["timestamp"] for row in rows[:6]] == [
        "2024-01-01T00:00:00Z",
        "2024-02-01T00:00:00Z",
        "2024-03-01T00:00:00Z",
        "2024-04-01T00:00:00Z",
        "2024-05-01T00:00:00Z",
        "2024-06-01T00:00:00Z",
    ]


def test_to_draft_json_omits_risk_and_execution_block() -> None:
    builder = StrategyBuilder()
    builder.symbol = "TCS.NS"
    builder.timeframe = "1m"
    builder.objective = "intraday"
    builder.sentiment = "bullish"
    builder.experience = "beginner"
    builder.stop_loss = 2.0
    builder.take_profit = 5.0

    draft = builder.to_draft_json(
        mode_override="backtest_confirmation",
        processing_status="awaiting_confirmation",
    )

    assert "daily_loss_cap_pct" not in draft
    assert "max_trade" not in draft
    assert "risk_and_execution" not in draft


def test_normalize_ohlcv_payload_sorts_and_casts_values():
    payload = {
        "data": [
            {
                "datetime": "2026-01-01T04:00:00Z",
                "open": "101",
                "high": "105",
                "low": "100",
                "close": "104",
                "volume": "1500",
            },
            {
                "datetime": "2026-01-01T03:45:00Z",
                "open": "100",
                "high": "103",
                "low": "99",
                "close": "101",
                "volume": "1200",
            },
        ]
    }

    rows = normalize_ohlcv_payload(payload)

    assert rows == [
        {
            "timestamp": "2026-01-01T03:45:00Z",
            "open": 100.0,
            "high": 103.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1200.0,
        },
        {
            "timestamp": "2026-01-01T04:00:00Z",
            "open": 101.0,
            "high": 105.0,
            "low": 100.0,
            "close": 104.0,
            "volume": 1500.0,
        },
    ]


def test_normalize_ohlcv_payload_rejects_invalid_high_low():
    payload = {
        "data": [
            {
                "datetime": "2026-01-01T03:45:00Z",
                "open": "100",
                "high": "98",
                "low": "99",
                "close": "101",
                "volume": "1200",
            },
        ]
    }

    with pytest.raises(ValueError, match="high lower than other price fields"):
        normalize_ohlcv_payload(payload)
