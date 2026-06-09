"""
app.services.execution.market_data
──────────────────────────────────
Asset-class-aware live market-data clients used by the execution service.

Public surface:

    from app.services.execution.market_data import (
        MarketDataClient,           # abstract interface
        UpstoxClient,               # equity_cash
        BinanceClient,              # crypto_spot
        get_market_data_client,     # factory
    )

The legacy ``app.services.execution.market_data_service`` module re-exports
``MarketDataService`` (the high-level cached wrapper). New code should depend
on these client classes directly only when it needs a non-cached transport.
"""
from __future__ import annotations

from app.services.execution.market_data.base import MarketDataClient
from app.services.execution.market_data.binance_client import BinanceClient
from app.services.execution.market_data.factory import get_market_data_client
from app.services.execution.market_data.upstox_client import UpstoxClient

__all__ = [
    "MarketDataClient",
    "UpstoxClient",
    "BinanceClient",
    "get_market_data_client",
]
