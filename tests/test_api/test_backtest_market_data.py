from __future__ import annotations

import pytest

from app.schemas.backtest import BacktestTriggerRequest
from app.services.backtest import market_data
from app.services.backtest.backtest_window import BacktestWindowError
from app.services.backtest.market_data import (
    StrategyMarketDataRequest,
    _build_ohlcv_fetch_slots,
    _dedup_and_sort_records,
    extract_strategy_market_data_request,
    normalize_ohlcv_payload,
)
from app.services.strategy.builder import StrategyBuilder, extract_strategy_details
from quant_engine.engine.config import (
    BACKTEST_MARKET_DATA_FROM_UTC,
    BACKTEST_MARKET_DATA_TO_UTC,
)


def test_extract_strategy_market_data_request_honors_user_window(tmp_path):
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
        from_utc="2026-03-02T03:45:00Z",
        to_utc="2026-03-02T09:59:00Z",
        user_specified_window=True,
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
    assert request.user_specified_window is False


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

    with pytest.raises(BacktestWindowError, match="earlier than the end"):
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
        "2026-03-31T23:59:59Z",
    )

    assert [(slot.index, slot.from_utc, slot.to_utc) for slot in slots] == [
        (1, "2024-01-01T00:00:00Z", "2024-06-30T23:59:59Z"),
        (2, "2024-07-01T00:00:00Z", "2024-12-31T23:59:59Z"),
        (3, "2025-01-01T00:00:00Z", "2025-06-30T23:59:59Z"),
        (4, "2025-07-01T00:00:00Z", "2025-12-31T23:59:59Z"),
        (5, "2026-01-01T00:00:00Z", "2026-03-31T23:59:59Z"),
    ]


def test_dedup_and_sort_records_sorts_and_deduplicates_timestamps() -> None:
    rows = _dedup_and_sort_records(
        [
            {
                "timestamp": "2024-07-01T03:45:00Z",
                "open": 102.0,
                "high": 105.0,
                "low": 101.0,
                "close": 104.0,
                "volume": 2000.0,
            },
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
        ]
    )

    assert [row["timestamp"] for row in rows] == [
        "2024-01-01T03:45:00Z",
        "2024-07-01T03:45:00Z",
    ]
    assert rows[1]["close"] == 104.0


@pytest.mark.asyncio
async def test_fetch_ohlcv_records_fetches_multiple_chunks(monkeypatch) -> None:
    calls: list[dict] = []
    fixed_chunks = [
        ("2024-01-01T00:00:00Z", "2024-03-31T23:59:59Z"),
        ("2024-04-01T00:00:00Z", "2024-06-30T23:59:59Z"),
        ("2024-07-01T00:00:00Z", "2024-09-30T23:59:59Z"),
        ("2024-10-01T00:00:00Z", "2024-12-31T23:59:59Z"),
    ]

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

    monkeypatch.setattr(market_data, "use_grpc_transport", lambda _cfg: False)
    monkeypatch.setattr(market_data, "get_cached_records", lambda *_a, **_k: None)
    monkeypatch.setattr(market_data, "_build_date_chunks", lambda *_a, **_k: fixed_chunks)
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.extend_fetch_start",
        lambda from_utc, **_kwargs: from_utc,
    )
    monkeypatch.setattr(market_data.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(market_data.settings, "historical_data_url", "http://data.test")

    rows = await market_data.fetch_ohlcv_records(
        StrategyMarketDataRequest(
            yaml_path="",
            raw_symbol="TCS.NS",
            symbol="TCS",
            interval="1d",
            from_utc="2024-01-01T00:00:00Z",
            to_utc="2024-12-31T23:59:59Z",
        )
    )

    assert len(calls) == 4
    assert len(rows) == 4
    assert calls[0]["from"] == "2024-01-01T00:00:00Z"
    assert calls[-1]["to"] == "2024-12-31T23:59:59Z"


def test_build_monthly_fallback_slots_splits_half_year_window() -> None:
    from app.services.backtest.market_data import OhlcvFetchSlot, _build_monthly_fallback_slots

    slot = OhlcvFetchSlot(
        index=1,
        from_utc="2024-01-01T00:00:00Z",
        to_utc="2024-06-30T23:59:59Z",
    )
    fallback_slots = _build_monthly_fallback_slots(slot)

    assert len(fallback_slots) == 6
    assert [slot.from_utc for slot in fallback_slots] == [
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
