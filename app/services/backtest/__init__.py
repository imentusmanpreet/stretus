"""
Backtest-related helpers shared across API routes and chat orchestration.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.backtest.market_data import (
        INTRABAR_EXECUTION_INTERVAL,
        StrategyMarketDataRequest,
        build_main_fetch_request,
        extract_htf_market_data_requests,
        extract_reference_market_data_request,
        extract_strategy_market_data_request,
        fetch_auxiliary_ohlcv,
        fetch_ohlcv_records,
        normalize_ohlcv_payload,
        resolve_intrabar_execution,
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
        sort_backtest_monthly_performance_desc,
        summarize_backtest_for_db,
    )

__all__ = [
    "INTRABAR_EXECUTION_INTERVAL",
    "StrategyMarketDataRequest",
    "apply_sanitized_result_to_row",
    "build_main_fetch_request",
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
    "resolve_intrabar_execution",
    "run_quant_backtest_sync",
    "sort_backtest_monthly_performance_desc",
    "summarize_backtest_for_db",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "INTRABAR_EXECUTION_INTERVAL": ("app.services.backtest.market_data", "INTRABAR_EXECUTION_INTERVAL"),
    "StrategyMarketDataRequest": ("app.services.backtest.market_data", "StrategyMarketDataRequest"),
    "apply_sanitized_result_to_row": ("app.services.backtest.result_store", "apply_sanitized_result_to_row"),
    "build_main_fetch_request": ("app.services.backtest.market_data", "build_main_fetch_request"),
    "extract_htf_market_data_requests": ("app.services.backtest.market_data", "extract_htf_market_data_requests"),
    "extract_reference_market_data_request": (
        "app.services.backtest.market_data",
        "extract_reference_market_data_request",
    ),
    "extract_strategy_market_data_request": (
        "app.services.backtest.market_data",
        "extract_strategy_market_data_request",
    ),
    "fetch_auxiliary_ohlcv": ("app.services.backtest.market_data", "fetch_auxiliary_ohlcv"),
    "fetch_ohlcv_records": ("app.services.backtest.market_data", "fetch_ohlcv_records"),
    "insert_chat_backtest_row": ("app.services.backtest.result_store", "insert_chat_backtest_row"),
    "insert_failed_chat_backtest": ("app.services.backtest.result_store", "insert_failed_chat_backtest"),
    "normalize_backtest_metric_aliases": (
        "app.services.backtest.result_store",
        "normalize_backtest_metric_aliases",
    ),
    "normalize_ohlcv_payload": ("app.services.backtest.market_data", "normalize_ohlcv_payload"),
    "queue_quant_backtest": ("app.services.backtest.quant_engine_client", "queue_quant_backtest"),
    "resolve_intrabar_execution": ("app.services.backtest.market_data", "resolve_intrabar_execution"),
    "run_quant_backtest_sync": ("app.services.backtest.quant_engine_client", "run_quant_backtest_sync"),
    "sort_backtest_monthly_performance_desc": (
        "app.services.backtest.result_store",
        "sort_backtest_monthly_performance_desc",
    ),
    "summarize_backtest_for_db": ("app.services.backtest.result_store", "summarize_backtest_for_db"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
