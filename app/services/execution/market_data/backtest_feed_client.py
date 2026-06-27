"""
app.services.execution.market_data.backtest_feed_client
────────────────────────────────────────────────────────
A :class:`MarketDataClient` that sources candles from the SAME historical market-data service
the backtester uses (``HISTORICAL_DATA_URL`` / the gRPC market-data target), via
:func:`app.services.backtest.market_data.fetch_ohlcv_records`.

Why this exists: the live per-venue clients (Upstox/Binance) hit the broker API directly and
depend on a live access token. The backtest feed is the proven, always-available Indian/crypto
OHLCV source already used for backtesting — so this client gives the live evaluator a data
source with **full backtest/live parity** and **no broker-token dependency**. It is used:
  * as the resilient FALLBACK behind the live client (when the broker feed fails), and
  * as the SOLE equity source when ``EQUITY_MARKET_DATA_SOURCE=backtest_feed`` (paper/parity).

It is a *historical* feed: ``fetch_ltp`` returns the latest available close (a paper-grade price,
not a live tick) and ``fetch_circuit_limits`` returns ``None`` so the caller applies its
percentage-default band. For REAL ``live`` order pricing, the live broker client must be primary.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from app.services.execution.market_data.base import MarketDataClient

logger = logging.getLogger(__name__)


def _iso_z(dt: datetime) -> str:
    """UTC ISO-8601 with a trailing Z, the form the backtest fetchers expect."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class BacktestFeedClient(MarketDataClient):
    """Equity/crypto OHLCV sourced from the backtest historical market-data service."""

    adapter_id = "backtest_feed"

    def __init__(self, *, adapter_id: Optional[str] = None) -> None:
        if adapter_id:
            self.adapter_id = adapter_id

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        adapter_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        # Late imports: keep this module import-light and avoid pulling the backtest stack at
        # process start. These helpers size the calendar window for ``lookback`` bars at
        # ``timeframe`` exactly as the backtest path does, so warm-up is never short.
        from app.services.backtest.market_data import (
            StrategyMarketDataRequest,
            _bars_per_calendar_day,
            _interval_buffer_days,
            fetch_ohlcv_records,
        )
        from app.services.data.provider import records_to_frame

        now = datetime.now(timezone.utc)
        bars_per_day = _bars_per_calendar_day(timeframe) or 26.0
        # 1.7× covers weekends/holidays (non-trading calendar days); buffer adds warm-up slack.
        window_days = math.ceil((lookback / bars_per_day) * 1.7) + _interval_buffer_days(
            timeframe, daily_default=30
        )
        from_dt = now - timedelta(days=max(7, window_days))

        clean = symbol.replace(".NS", "").replace(".BO", "")
        request = StrategyMarketDataRequest(
            yaml_path="", raw_symbol=symbol, symbol=clean, interval=timeframe,
            from_utc=_iso_z(from_dt), to_utc=_iso_z(now), user_specified_window=True,
        )
        records = await fetch_ohlcv_records(request)
        df = records_to_frame(records)
        if df is None or df.empty:
            raise ValueError(
                f"backtest feed returned no candles for {symbol} ({timeframe})."
            )
        logger.info(
            "📡 backtest_feed candles | symbol=%s tf=%s rows=%d window=%dd last=%s",
            symbol, timeframe, len(df), window_days, df.index[-1],
        )
        return df.tail(lookback)

    async def fetch_ltp(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> float:
        """Latest available close as a paper-grade LTP (historical feed has no live tick)."""
        df = await self.fetch_candles(symbol, "1d", 3, adapter_symbol=adapter_symbol)
        col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            raise ValueError(f"backtest feed candles for {symbol} have no close column.")
        ltp = float(df[col].iloc[-1])
        if ltp <= 0:
            raise ValueError(f"backtest feed close ≤ 0 for {symbol}.")
        logger.info("📡 backtest_feed LTP (last close) | symbol=%s ltp=%.4f", symbol, ltp)
        return ltp

    async def fetch_circuit_limits(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> Optional[dict]:
        """No live circuit bands on a historical feed → caller applies its percentage default."""
        return None
