"""
app.services.execution.market_data.upstox_client
────────────────────────────────────────────────
Async wrapper around the Upstox v2 REST API used for Indian equity execution.

Behaviour is byte-for-byte preserved from the original implementation in
``app.services.execution.market_data_service`` — only the packaging changed.

Endpoints
─────────
  Candles  GET {MARKET_DATA_URL}/historical-candle/{instrument_key}/{interval}/{to}/{from}
  Intraday GET {MARKET_DATA_URL}/historical-candle/intraday/{instrument_key}/{interval}
  LTP      GET {MARKET_DATA_URL}/market-quote/ltp?instrument_key=...
  Circuit  GET {MARKET_DATA_URL}/market-quote/quotes?instrument_key=...

Upstox historical-candle ONLY supports these intervals:
  1minute, 30minute, day, week, month
For strategy timeframes in between (e.g. 15m, 5m, 1h) we fetch the nearest
finer-grain Upstox interval and resample to the target timeframe.

    Strategy → fetch interval → resample rule
    ─────────────────────────────────────────
    1m   1minute    passthrough
    3m   1minute    3min
    5m   1minute    5min
    10m  1minute    10min
    15m  1minute    15min
    30m  30minute   passthrough
    45m  30minute   45min
    1h   30minute   60min
    2h   30minute   120min
    4h   30minute   240min
    1d   day        passthrough
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional
from urllib.parse import quote as url_quote

import httpx
import pandas as pd

from app.core.config import get_settings
from app.services.execution.market_data.base import MarketDataClient

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Timeframe helpers ─────────────────────────────────────────────────────────

_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "45m": 45, "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}

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

# Upstox rejects oversized 1-minute historical windows.
_MAX_HISTORICAL_CALENDAR_DAYS_BY_FETCH_INTERVAL: dict[str, int] = {
    "1minute":  30,
    "30minute": 180,
}

# Hand-curated ISIN map: Upstox historical-candle rejects ``NSE_EQ|SYMBOL`` for
# some equities and requires the ISIN-based form.
_UPSTOX_INSTRUMENT_KEY_OVERRIDES: dict[str, str] = {
    "GMRAIRPORT": "NSE_EQ|INE776C01039",
    "IDEA":       "NSE_EQ|INE669E01016",
    "NHPC":       "NSE_EQ|INE848E01016",
    "SUZLON":     "NSE_EQ|INE040H01021",
}


def _fetch_interval(timeframe: str) -> str:
    return _FETCH_INTERVAL.get(timeframe, "1minute")


def _resample_rule(timeframe: str) -> Optional[str]:
    return _RESAMPLE_RULE.get(timeframe)


def _upstox_instrument_key(symbol: str) -> str:
    """Best-effort fallback key: 'RELIANCE.NS' → 'NSE_EQ|RELIANCE'."""
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
    bars at the FETCH interval. NSE-session sized (375 min/day) with weekend +
    holiday buffer.
    """
    _fi_to_mins = {"1minute": 1, "30minute": 30, "day": 1440}
    fi = _fetch_interval(timeframe)
    fi_mins = _fi_to_mins.get(fi, 1)

    total_mins   = lookback_candles * fi_mins * 2          # 2× safety buffer
    market_mins  = 375                                      # NSE session length
    trading_days = max(2, (total_mins // market_mins) + 1)
    calendar_days = trading_days * 2 + 14                   # weekends + holidays
    max_calendar_days = _MAX_HISTORICAL_CALENDAR_DAYS_BY_FETCH_INTERVAL.get(fi)
    if max_calendar_days is not None:
        calendar_days = min(calendar_days, max_calendar_days)

    to_dt   = date.today()
    from_dt = to_dt - timedelta(days=calendar_days)
    return from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _upstox_candles_to_df(candles: list) -> pd.DataFrame:
    """Convert Upstox candle arrays [ts, o, h, l, c, v, oi] → DataFrame."""
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
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df.columns = [c.lower() for c in df.columns]
    return df


def _resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample finer OHLCV bars to a coarser timeframe. Expects lowercase columns."""
    resampled = df.resample(rule, closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    return resampled.dropna(subset=["open", "high", "low", "close"])


# ── UpstoxClient ──────────────────────────────────────────────────────────────

class UpstoxClient(MarketDataClient):
    """
    Equity-cash market data via Upstox v2.

    Instrument key resolution (priority):
      1. adapter_symbol from ref_data.instrument_adapter_mappings
         (e.g. "NSE_EQ|INE002A01018")
      2. Best-effort fallback constructed from the ticker:
         RELIANCE.NS → NSE_EQ|RELIANCE
    """

    adapter_id = "upstox_rest"

    def __init__(self) -> None:
        self._base    = settings.market_data_url.rstrip("/")
        self._token   = settings.upstox_access_token
        self._timeout = settings.market_data_timeout_seconds

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _resolve_key(self, symbol: str, adapter_symbol: Optional[str] = None) -> str:
        return adapter_symbol if adapter_symbol else _upstox_instrument_key(symbol)

    # ── Candles ───────────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        adapter_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self._token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set.")

        instrument_key = self._resolve_key(symbol, adapter_symbol)
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
            "Upstox historical-candle | symbol=%s key=%s fetch_interval=%s "
            "target_timeframe=%s from=%s to=%s",
            symbol, instrument_key, fetch_iv, timeframe, from_date, to_date,
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

        # ── 2. Today's intraday candles ───────────────────────────────────────
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
                            "Upstox intraday | symbol=%s rows=%d last=%s",
                            symbol, len(df_intraday), df_intraday.index[-1],
                        )
                    else:
                        logger.debug(
                            "Upstox intraday empty for %s (pre-market or holiday)",
                            symbol,
                        )
                else:
                    logger.warning(
                        "Upstox intraday returned %s for %s — using historical only",
                        intra_resp.status_code, symbol,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Upstox intraday fetch failed for %s — %s. Using historical only.",
                    symbol, exc,
                )

        # ── 3. Merge, dedupe, sort ────────────────────────────────────────────
        if not df_intraday.empty:
            df = pd.concat([df_hist, df_intraday])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            logger.info(
                "Upstox merged candles | symbol=%s hist=%d intraday=%d total=%d last=%s",
                symbol, len(df_hist), len(df_intraday), len(df), df.index[-1],
            )
        else:
            df = df_hist

        # ── 4. Resample if needed ─────────────────────────────────────────────
        if resample_rule:
            df = _resample_df(df, resample_rule)
            logger.info(
                "Upstox resampled → %d bars (%s) for %s last=%s",
                len(df), timeframe, symbol, df.index[-1],
            )
        else:
            logger.info(
                "Upstox candles OK | symbol=%s rows=%d (%s) last=%s",
                symbol, len(df), timeframe, df.index[-1],
            )

        if df.empty:
            raise ValueError(f"Resampled DataFrame is empty for {symbol} ({timeframe}).")

        return df.tail(lookback)

    # ── LTP ───────────────────────────────────────────────────────────────────

    async def fetch_ltp(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> float:
        if not self._token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set.")

        instrument_key = self._resolve_key(symbol, adapter_symbol)
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
        logger.info(
            "Upstox LTP | symbol=%s key=%s ltp=%.4f",
            symbol, instrument_key, float(ltp),
        )
        return float(ltp)

    # ── Circuit limits ────────────────────────────────────────────────────────

    async def fetch_circuit_limits(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> Optional[dict]:
        if not self._token:
            return None

        instrument_key = self._resolve_key(symbol, adapter_symbol)
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
