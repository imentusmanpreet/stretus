"""
Backtest-related helpers shared across API routes and chat orchestration.
"""

from app.services.backtest.market_data import (
    StrategyMarketDataRequest,
    extract_htf_market_data_requests,
    extract_reference_market_data_request,
    extract_strategy_market_data_request,
    fetch_auxiliary_ohlcv,
    fetch_ohlcv_records,
    normalize_ohlcv_payload,
)
from app.services.backtest.quant_engine_client import (
    queue_quant_backtest,
    run_quant_backtest_sync,
)
from app.services.backtest.result_store import (
    apply_sanitized_result_to_row,
    insert_chat_backtest_row,
    insert_failed_chat_backtest,
    normalize_backtest_metric_aliases,
    summarize_backtest_for_db,
)

__all__ = [
    "StrategyMarketDataRequest",
    "apply_sanitized_result_to_row",
    "extract_htf_market_data_requests",
    "extract_reference_market_data_request",
    "extract_strategy_market_data_request",
    "fetch_auxiliary_ohlcv",
    "fetch_ohlcv_records",
    "insert_chat_backtest_row",
    "insert_failed_chat_backtest",
    "normalize_backtest_metric_aliases",
    "normalize_ohlcv_payload",
    "queue_quant_backtest",
    "run_quant_backtest_sync",
    "summarize_backtest_for_db",
]
