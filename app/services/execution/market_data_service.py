"""
app/services/execution/market_data_service.py
──────────────────────────────────────────────
All execution / order-evaluation market data comes from a single source:

  MARKET_DATA_URL  (default: https://api.upstox.com/v2)
  ──────────────────────────────────────────────────────
  Candles   GET {MARKET_DATA_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
  LTP       GET {MARKET_DATA_URL}/market-quote/ltp?instrument_key=...
  Circuit   GET {MARKET_DATA_URL}/market-quote/quotes?instrument_key=...

  Upstox historical-candle ONLY supports these intervals: 1minute, 30minute, day, week, month.
  For strategy timeframes in between (e.g. 15m, 5m, 1h) we fetch the nearest finer-grain
  Upstox interval and resample to the target timeframe using pandas.

    Strategy → Upstox fetch interval → resample rule
    ─────────────────────────────────────────────────
    1m        1minute                  passthrough
    3m        1minute                  3T
    5m        1minute                  5T
    10m       1minute                  10T
    15m       1minute                  15T
    30m       30minute                 passthrough
    45m       30minute                 45T (resampled from 30m)
    1h        30minute                 60T
    2h        30minute                 120T
    4h        30minute                 240T
    1d        day                      passthrough

The backtest service uses a SEPARATE variable (HISTORICAL_DATA_URL → ngrok tunnel).
This module never touches HISTORICAL_DATA_URL.

All responses are cached in-process (MarketDataCache).
On network errors the last cached value is returned as a fallback.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional
from urllib.parse import quote as url_quote

import httpx
import pandas as pd

from app.core.config import get_settings
from app.services.execution.execution_cache import MarketDataCache

logger = logging.getLogger(__name__)
settings = get_settings()

_cache = MarketDataCache()

# ── Timeframe helpers ─────────────────────────────────────────────────────────

_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "45m": 45, "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}

# Upstox /historical-candle accepts ONLY these intervals:
#   1minute  30minute  day  week  month
# For any strategy timeframe that falls between these, we fetch the finest
# available Upstox interval and resample to the target.
_FETCH_INTERVAL: dict[str, str] = {
    "1m":  "1minute",
    "3m":  "1minute",
    "5m":  "1minute",
    "10m": "1minute",
    "15m": "1minute",
    "30m": "30minute",
    "45m": "30minute",
    "1h":  "30minute",
    "2h":  "30minute",
    "4h":  "30minute",
    "1d":  "day",
}

# pandas resample rule to apply AFTER fetching (None = no resample needed)
_RESAMPLE_RULE: dict[str, Optional[str]] = {
    "1m":  None,
    "3m":  "3min",
    "5m":  "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": None,
    "45m": "45min",
    "1h":  "60min",
    "2h":  "120min",
    "4h":  "240min",
    "1d":  None,
}

# Upstox rejects oversized 1-minute historical windows. We still fetch today's
# intraday endpoint separately below, so these caps only bound the closed-session
# historical request.
_MAX_HISTORICAL_CALENDAR_DAYS_BY_FETCH_INTERVAL: dict[str, int] = {
    "1minute": 30,
    "30minute": 180,
}

_UPSTOX_INSTRUMENT_KEY_OVERRIDES: dict[str, str] = {
    "GMRAIRPORT": "NSE_EQ|INE776C01039",
    "IDEA": "NSE_EQ|INE669E01016",
    "NHPC": "NSE_EQ|INE848E01016",
    "SUZLON": "NSE_EQ|INE040H01021",
}


def _fetch_interval(timeframe: str) -> str:
    return _FETCH_INTERVAL.get(timeframe, "1minute")


def _resample_rule(timeframe: str) -> Optional[str]:
    return _RESAMPLE_RULE.get(timeframe)


def _upstox_instrument_key(symbol: str) -> str:
    """RELIANCE.NS → NSE_EQ|RELIANCE"""
    s = symbol.strip().upper()
    if ":" in s:
        _, s = s.split(":", 1)
    if s.endswith(".NS"):
        ticker = s[:-3]
        return _UPSTOX_INSTRUMENT_KEY_OVERRIDES.get(ticker, f"NSE_EQ|{ticker}")
    if s.endswith(".BO"):
        return f"BSE_EQ|{s[:-3]}"
    return _UPSTOX_INSTRUMENT_KEY_OVERRIDES.get(s, f"NSE_EQ|{s}")


def _date_window(lookback_candles: int, timeframe: str) -> tuple[str, str]:
    """
    Return (from_date, to_date) YYYY-MM-DD covering at least lookback_candles
    bars at the FETCH interval (finer-grain than strategy timeframe), with buffer.

    For intraday intervals we budget 375 market-minutes per trading day
    (6h 15m for NSE).  We double the window and add 14 days for weekends/holidays.
    """
    _fi_to_mins = {"1minute": 1, "30minute": 30, "day": 1440}
    fi = _fetch_interval(timeframe)
    fi_mins = _fi_to_mins.get(fi, 1)

    total_mins   = lookback_candles * fi_mins * 2           # 2× safety buffer
    market_mins_per_day = 375                                # NSE session length
    trading_days = max(2, (total_mins // market_mins_per_day) + 1)
    calendar_days = trading_days * 2 + 14                   # weekends + holidays
    max_calendar_days = _MAX_HISTORICAL_CALENDAR_DAYS_BY_FETCH_INTERVAL.get(fi)
    if max_calendar_days is not None:
        calendar_days = min(calendar_days, max_calendar_days)

    to_dt   = date.today()
    from_dt = to_dt - timedelta(days=calendar_days)
    return from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")


# ── MarketDataService ─────────────────────────────────────────────────────────

class MarketDataService:
    """
    Unified market data client for the execution service.

    Single source: MARKET_DATA_URL (Upstox v2 base URL).
    Instantiate once per request; module-level cache is shared.
    """

    def __init__(self) -> None:
        self._upstox = UpstoxClient()

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        db_instrument_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Upstox historical-candle API and resample
        to the strategy timeframe if needed.

        Falls back to stale cache on network error.
        """
        cached = _cache.get_candles(symbol, timeframe)
        if cached is not None:
            logger.debug("cache hit: candles %s/%s", symbol, timeframe)
            return cached

        try:
            df = await self._upstox.fetch_candles(
                symbol, timeframe, lookback, db_instrument_key
            )
            _cache.set_candles(symbol, timeframe, df)
            return df
        except Exception as exc:
            logger.warning(
                "fetch_candles %s/%s failed — %s. Checking stale cache.",
                symbol, timeframe, exc,
            )
            stale = _cache.get_candles(symbol, timeframe)
            if stale is not None:
                return stale
            raise RuntimeError(
                f"No market data available for {symbol} ({timeframe}): {exc}"
            ) from exc

    async def fetch_ltp(
        self,
        symbol: str,
        db_instrument_key: Optional[str] = None,
    ) -> float:
        """
        Fetch last traded price from Upstox v2 market-quote/ltp.
        Falls back to last candle close if Upstox is unavailable.
        """
        cached = _cache.get_ltp(symbol)
        if cached is not None:
            return cached

        try:
            ltp = await self._upstox.fetch_ltp(symbol, db_instrument_key)
            _cache.set_ltp(symbol, ltp)
            return ltp
        except Exception as exc:
            logger.warning(
                "Upstox LTP unavailable for %s — %s. Falling back to candle close.",
                symbol, exc,
            )

        for tf in ("1m", "5m", "15m", "30m", "1h", "1d"):
            df = _cache.get_candles(symbol, tf)
            if df is not None and not df.empty:
                ltp = float(df["Close"].iloc[-1])
                _cache.set_ltp(symbol, ltp)
                logger.info("LTP from candle close (%s) | %s = %.4f", tf, symbol, ltp)
                return ltp

        raise RuntimeError(
            f"Cannot determine LTP for {symbol}: Upstox unavailable and no candle cache."
        )

    async def fetch_circuit_limits(
        self,
        symbol: str,
        db_instrument_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Fetch circuit limits from Upstox v2 market-quote/quotes.
        Returns None on error — evaluator falls back to DB instrument_metadata columns.
        """
        cached = _cache.get_circuit(symbol)
        if cached is not None:
            return cached

        try:
            data = await self._upstox.fetch_circuit_limits(symbol, db_instrument_key)
            if data:
                _cache.set_circuit(symbol, data)
                logger.info(
                    "Circuit limits from Upstox | %s upper=%s lower=%s",
                    symbol, data.get("upper_circuit"), data.get("lower_circuit"),
                )
            return data
        except Exception as exc:
            logger.warning(
                "fetch_circuit_limits %s failed — %s. Evaluator will use DB values.",
                symbol, exc,
            )
            return None


# ── UpstoxClient ──────────────────────────────────────────────────────────────

class UpstoxClient:
    """
    Async wrapper around Upstox v2 API endpoints.

    Base URL: MARKET_DATA_URL (.env).
    Auth: Bearer UPSTOX_ACCESS_TOKEN.

    Instrument key resolution (priority):
      1. db_instrument_key from instrument_metadata table (e.g. NSE_EQ|INE002A01018)
      2. Best-effort fallback: NSE_EQ|SYMBOL (works for market-quote endpoints;
         historical-candle may also need the ISIN-based key for some symbols)
    """

    def __init__(self) -> None:
        self._base    = settings.market_data_url.rstrip("/")
        self._token   = settings.upstox_access_token
        self._timeout = settings.market_data_timeout_seconds

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _resolve_key(self, symbol: str, db_instrument_key: Optional[str] = None) -> str:
        return db_instrument_key if db_instrument_key else _upstox_instrument_key(symbol)

    # ── Candles ───────────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        db_instrument_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch candles using the nearest valid Upstox interval, then resample
        to the strategy timeframe.

        Combines two Upstox endpoints:
          1. /historical-candle  — previous days (closed sessions)
          2. /historical-candle/intraday — today's candles (live, current session)

        Upstox historical-candle valid intervals: 1minute, 30minute, day, week, month.
        Example: strategy is 15m → fetch 1minute candles → resample to 15T.
        """
        if not self._token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set.")

        instrument_key = self._resolve_key(symbol, db_instrument_key)
        fetch_iv       = _fetch_interval(timeframe)
        resample_rule  = _resample_rule(timeframe)

        fi_table = {"1minute": 1, "30minute": 30, "day": 1440}
        fi_mins  = fi_table.get(fetch_iv, 1)
        tf_mins  = _TIMEFRAME_MINUTES.get(timeframe, 15)
        ratio    = max(1, tf_mins // fi_mins)
        fetch_lookback = lookback * ratio

        encoded_key = url_quote(instrument_key, safe="")

        # ── 1. Historical candles (previous sessions) ─────────────────────────
        from_date, to_date = _date_window(fetch_lookback, timeframe)
        hist_url = (
            f"{self._base}/historical-candle"
            f"/{encoded_key}/{fetch_iv}/{to_date}/{from_date}"
        )
        logger.info(
            "Fetching historical candles | symbol=%s fetch_interval=%s "
            "target_timeframe=%s from=%s to=%s",
            symbol, fetch_iv, timeframe, from_date, to_date,
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            hist_resp = await client.get(hist_url, headers=self._headers())

        if hist_resp.status_code != 200:
            raise RuntimeError(
                f"Upstox historical-candle returned {hist_resp.status_code} for {symbol} "
                f"(key={instrument_key}, interval={fetch_iv}): {hist_resp.text[:500]}"
            )

        hist_candles = hist_resp.json().get("data", {}).get("candles", [])
        if not hist_candles:
            raise ValueError(
                f"Empty historical candles from Upstox for {symbol} ({timeframe})."
            )

        df_hist = _upstox_candles_to_df(hist_candles[::-1])

        # ── 2. Today's intraday candles (current session) ─────────────────────
        # Only fetch intraday for sub-daily timeframes — daily bars don't need it.
        df_intraday = pd.DataFrame()
        if fetch_iv != "day":
            intraday_url = (
                f"{self._base}/historical-candle/intraday"
                f"/{encoded_key}/{fetch_iv}"
            )
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    intra_resp = await client.get(intraday_url, headers=self._headers())

                if intra_resp.status_code == 200:
                    intra_candles = intra_resp.json().get("data", {}).get("candles", [])
                    if intra_candles:
                        df_intraday = _upstox_candles_to_df(intra_candles[::-1])
                        logger.info(
                            "Intraday candles | symbol=%s rows=%d last=%s",
                            symbol, len(df_intraday), df_intraday.index[-1],
                        )
                    else:
                        logger.debug("Intraday candles empty for %s (pre-market or holiday)", symbol)
                else:
                    logger.warning(
                        "Intraday candle fetch returned %s for %s — using historical only",
                        intra_resp.status_code, symbol,
                    )
            except Exception as exc:
                logger.warning("Intraday candle fetch failed for %s — %s. Using historical only.", symbol, exc)

        # ── 3. Merge, deduplicate, sort ───────────────────────────────────────
        if not df_intraday.empty:
            df = pd.concat([df_hist, df_intraday])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            logger.info(
                "Merged candles | symbol=%s hist=%d intraday=%d total=%d last=%s",
                symbol, len(df_hist), len(df_intraday), len(df), df.index[-1],
            )
        else:
            df = df_hist

        # ── 4. Resample to target timeframe if needed ─────────────────────────
        if resample_rule:
            df = _resample_df(df, resample_rule)
            logger.info(
                "Resampled → %d bars (%s) for %s  last=%s",
                len(df), timeframe, symbol, df.index[-1],
            )
        else:
            logger.info(
                "Candles OK | symbol=%s rows=%d (%s)  last=%s",
                symbol, len(df), timeframe, df.index[-1],
            )

        if df.empty:
            raise ValueError(f"Resampled DataFrame is empty for {symbol} ({timeframe}).")

        return df.tail(lookback)

    # ── LTP ───────────────────────────────────────────────────────────────────

    async def fetch_ltp(
        self,
        symbol: str,
        db_instrument_key: Optional[str] = None,
    ) -> float:
        """
        GET {MARKET_DATA_URL}/market-quote/ltp?instrument_key=NSE_EQ|...
        Response: { "status": "success", "data": { "NSE_EQ:RELIANCE": { "last_price": 2345.6 } } }
        """
        if not self._token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set.")

        instrument_key = self._resolve_key(symbol, db_instrument_key)
        url = f"{self._base}/market-quote/ltp"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                url,
                params={"instrument_key": instrument_key},
                headers=self._headers(),
            )
        resp.raise_for_status()

        payload = resp.json()
        data = payload.get("data", {})

        lookup_key = instrument_key.replace("|", ":")
        quote = data.get(lookup_key) or (next(iter(data.values()), {}) if data else {})

        ltp = quote.get("last_price")
        if not ltp:
            raise ValueError(
                f"No LTP in Upstox response for {symbol} (key={instrument_key}). "
                f"Market may be closed. Response: {payload}"
            )
        logger.info("Upstox LTP | %s (key=%s) = %.4f", symbol, instrument_key, float(ltp))
        return float(ltp)

    # ── Circuit limits ────────────────────────────────────────────────────────

    async def fetch_circuit_limits(
        self,
        symbol: str,
        db_instrument_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        GET {MARKET_DATA_URL}/market-quote/quotes?instrument_key=NSE_EQ|...
        Returns upper_circuit / lower_circuit or None when data is unavailable.
        """
        if not self._token:
            return None

        instrument_key = self._resolve_key(symbol, db_instrument_key)
        url = f"{self._base}/market-quote/quotes"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                url,
                params={"instrument_key": instrument_key},
                headers=self._headers(),
            )
        resp.raise_for_status()

        payload = resp.json()
        data = payload.get("data", {})

        lookup_key = instrument_key.replace("|", ":")
        quote = data.get(lookup_key) or (next(iter(data.values()), {}) if data else {})

        upper = quote.get("upper_circuit_limit") or quote.get("upper_circuit")
        lower = quote.get("lower_circuit_limit") or quote.get("lower_circuit")

        if upper is None and lower is None:
            return None

        return {
            "upper_circuit": float(upper) if upper is not None else None,
            "lower_circuit": float(lower) if lower is not None else None,
        }


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _upstox_candles_to_df(candles: list) -> pd.DataFrame:
    """
    Convert Upstox candle arrays [ts, open, high, low, close, volume, oi] → DataFrame.
    Index: UTC-aware DatetimeIndex.
    """
    rows = []
    for row in candles:
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            rows.append({
                "timestamp": row[0],
                "Open":   float(row[1]),
                "High":   float(row[2]),
                "Low":    float(row[3]),
                "Close":  float(row[4]),
                "Volume": float(row[5]),
            })

    if not rows:
        raise ValueError("No usable candle rows in Upstox response.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp").sort_index()
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def _resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1-minute (or 30-minute) OHLCV bars to a coarser timeframe.
    rule: pandas offset alias, e.g. '15min', '60min'.
    """
    resampled = df.resample(rule, closed="left", label="left").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })
    return resampled.dropna(subset=["Open", "High", "Low", "Close"])
