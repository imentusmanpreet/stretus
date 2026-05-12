from __future__ import annotations

import pytest

from app.services.backtest import market_data
from app.schemas.backtest import BacktestTriggerRequest
from app.services.backtest.market_data import (
    StrategyMarketDataRequest,
    _build_ohlcv_fetch_slots,
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


# Test removed - _merge_ohlcv_slot_rows function no longer exists in market_data.py
# The functionality is now handled internally by fetch_ohlcv_records


# Tests commented out - these test internal slot-based fetching logic that has been
# replaced with chunk-based fetching in the current implementation

# @pytest.mark.asyncio
# async def test_fetch_ohlcv_records_continues_when_a_slot_fails(monkeypatch) -> None:
#     # This test is no longer applicable as the slot-based fetching has been replaced
#     pass

# @pytest.mark.asyncio
# async def test_fetch_ohlcv_records_recovers_first_half_with_monthly_fallback(monkeypatch) -> None:
#     # This test is no longer applicable as the slot-based fetching has been replaced
#     pass


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
