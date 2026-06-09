"""
app.services.execution.market_data.base
───────────────────────────────────────
Abstract market-data client interface implemented by per-venue clients
(``UpstoxClient`` for equity_cash, ``BinanceClient`` for crypto_spot).

All implementations must return data in the same shape so the upstream
``MarketDataService`` and ``StrategyEvaluator`` stay asset-class agnostic.

DataFrame contract for ``fetch_candles``
────────────────────────────────────────
Returns a ``pandas.DataFrame`` with:

  - Index:    UTC-aware DatetimeIndex, oldest → newest (no duplicates).
  - Columns:  ``Open``, ``High``, ``Low``, ``Close``, ``Volume`` (capitalised).
  - Length:   at most ``lookback`` rows (the most recent ``lookback`` bars).

Implementations must raise ``RuntimeError`` for transport/auth failures and
``ValueError`` when the venue returned an empty / unusable payload.
"""
from __future__ import annotations

import abc
from typing import Optional

import pandas as pd


class MarketDataClient(abc.ABC):
    """Per-venue live market-data client."""

    #: Human-readable id used in log lines (e.g. ``"upstox_rest"``, ``"binance_rest"``).
    adapter_id: str = "abstract"

    @abc.abstractmethod
    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        adapter_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the most recent ``lookback`` candles at ``timeframe`` resolution."""

    @abc.abstractmethod
    async def fetch_ltp(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> float:
        """Return the latest trade price as a positive float."""

    @abc.abstractmethod
    async def fetch_circuit_limits(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Return ``{"upper_circuit": float, "lower_circuit": float}`` when the
        venue exposes hard price bounds, otherwise ``None``.

        For crypto venues that have no circuit breakers (e.g. Binance spot)
        implementations may return a soft band derived from the 24h high/low.
        """
