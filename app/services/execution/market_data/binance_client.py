"""
app.services.execution.market_data.binance_client
─────────────────────────────────────────────────
Async wrapper around the Binance Spot public REST API used for crypto
strategy evaluation.

Scope: market-data reads only. No authentication is required for the public
endpoints below, so no API key / secret is stored or sent. Order submission
to Binance is a separate, OMS-owned concern and is intentionally NOT in
this module.

Endpoints
─────────
  Klines   GET {BINANCE_SPOT_URL}/api/v3/klines
             ?symbol={ADAPTER_SYMBOL}&interval={INTERVAL}&limit={LIMIT}

  LTP      GET {BINANCE_SPOT_URL}/api/v3/ticker/price?symbol={ADAPTER_SYMBOL}

  Bounds   GET {BINANCE_SPOT_URL}/api/v3/ticker/24hr?symbol={ADAPTER_SYMBOL}
             (used as a soft "circuit limit" band — Binance has no exchange-
              side circuit breakers, but the 24h high/low provides a sensible
              sanity guard against fat-finger fills.)

Timeframe mapping (Binance native, no resampling needed)
────────────────────────────────────────────────────────
  1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

Strategy timeframes that do not map natively (e.g. ``10m``, ``45m``) fall
through to the finer Binance interval and are resampled with the same pandas
rule the equity path uses.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
import pandas as pd

from app.core.config import get_settings
from app.services.execution.market_data.base import MarketDataClient

logger = logging.getLogger(__name__)
settings = get_settings()


# Native Binance Spot intervals (per https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data).
_BINANCE_NATIVE_INTERVALS: set[str] = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


# Map strategy-side timeframes to (fetch_interval, optional resample_rule).
# When ``resample_rule`` is None the Binance native interval is used directly.
_TIMEFRAME_PLAN: dict[str, tuple[str, Optional[str]]] = {
    "1m":  ("1m",  None),
    "3m":  ("3m",  None),
    "5m":  ("5m",  None),
    "10m": ("5m",  "10min"),     # 10m has no native interval → resample from 5m
    "15m": ("15m", None),
    "30m": ("30m", None),
    "45m": ("15m", "45min"),     # 45m has no native interval → resample from 15m
    "1h":  ("1h",  None),
    "2h":  ("2h",  None),
    "4h":  ("4h",  None),
    "1d":  ("1d",  None),
}

# Default cap for Binance's ``limit`` query param. The API allows up to 1500,
# but we keep a safety margin to avoid 429s for over-eager backtests.
_BINANCE_MAX_LIMIT = 1000

# Default public-host base URL. Pulled from settings.crypto_market_data_url if
# the operator wants to point at a regional mirror or a test cluster.
_DEFAULT_BASE_URL = "https://api.binance.com"


def _plan_for(timeframe: str) -> tuple[str, Optional[str]]:
    if timeframe in _BINANCE_NATIVE_INTERVALS:
        return timeframe, None
    if timeframe in _TIMEFRAME_PLAN:
        return _TIMEFRAME_PLAN[timeframe]
    raise ValueError(
        f"Unsupported Binance timeframe: {timeframe!r}. "
        f"Supported: {sorted(_BINANCE_NATIVE_INTERVALS)} (or one of {sorted(_TIMEFRAME_PLAN.keys())})."
    )


def _strip_canonical_crypto_suffix(symbol: str) -> str:
    """Best-effort fallback: extract a Binance-native symbol from any input shape.

      "BTC_USDT"                          → "BTCUSDT"
      "crypto_spot:BTC_USDT@BINANCE_SPOT" → "BTCUSDT"
      "BTC/USDT"                          → "BTCUSDT"
      "BTCUSDT"                           → "BTCUSDT" (passthrough)
    """
    s = symbol.strip().upper()
    if s.startswith("CRYPTO_SPOT:") and "@" in s:
        s = s.split(":", 1)[1].split("@", 1)[0]
    if "/" in s:
        s = s.replace("/", "")
    if "_" in s:
        s = s.replace("_", "")
    return s


def _binance_klines_to_df(klines: list) -> pd.DataFrame:
    """
    Convert Binance kline rows to the OHLCV DataFrame contract.

    Each kline is:
      [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
       num_trades, taker_buy_base_volume, taker_buy_quote_volume, ignored]

    We index on ``open_time`` (ms since epoch) converted to UTC.
    """
    rows = []
    for row in klines:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        rows.append({
            "timestamp": int(row[0]),
            "Open":   float(row[1]),
            "High":   float(row[2]),
            "Low":    float(row[3]),
            "Close":  float(row[4]),
            "Volume": float(row[5]),
        })

    if not rows:
        raise ValueError("No usable kline rows in Binance response.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, unit="ms", errors="coerce")
    df = df.set_index("timestamp").sort_index()
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def _resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate OHLCV bars to a coarser timeframe."""
    resampled = df.resample(rule, closed="left", label="left").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })
    return resampled.dropna(subset=["Open", "High", "Low", "Close"])


# ── BinanceClient ─────────────────────────────────────────────────────────────

class BinanceClient(MarketDataClient):
    """
    Crypto-spot market data via Binance Spot public REST.

    Adapter-symbol resolution (priority):
      1. adapter_symbol from ref_data.instrument_adapter_mappings
         (adapter_id = 'binance_rest'; e.g. "BTCUSDT")
      2. Best-effort fallback constructed from the canonical symbol
         (BTC_USDT → BTCUSDT). The fallback path emits a warning since
         mappings should always exist for production strategies.
    """

    adapter_id = "binance_rest"

    def __init__(self) -> None:
        self._base    = self._resolve_base_url()
        self._timeout = settings.market_data_timeout_seconds

    @staticmethod
    def _resolve_base_url() -> str:
        """
        Resolve the Binance Spot base URL.

        Reads ``settings.crypto_market_data_url`` if set, otherwise falls back
        to the public production endpoint. We never reuse ``market_data_url``
        because that one is Upstox-owned.
        """
        url = getattr(settings, "crypto_market_data_url", None) or _DEFAULT_BASE_URL
        return str(url).rstrip("/")

    # ── Symbol resolution ────────────────────────────────────────────────────

    def _resolve_symbol(self, symbol: str, adapter_symbol: Optional[str]) -> str:
        if adapter_symbol:
            return adapter_symbol.strip().upper()
        fallback = _strip_canonical_crypto_suffix(symbol)
        logger.warning(
            "Binance adapter_symbol not provided — falling back to %s for symbol=%s. "
            "Add a ref_data binance_rest mapping to silence this warning.",
            fallback, symbol,
        )
        return fallback

    # ── Candles ──────────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
        adapter_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        binance_symbol = self._resolve_symbol(symbol, adapter_symbol)
        fetch_interval, resample_rule = _plan_for(timeframe)

        # When we resample to a coarser bar we need that many input bars at
        # the finer interval, so multiply through. ``+ 2`` is a safety pad.
        if resample_rule:
            ratio = _ratio_between_intervals(fetch_interval, timeframe)
            fetch_limit = min(_BINANCE_MAX_LIMIT, max(lookback * ratio + 2, lookback))
        else:
            fetch_limit = min(_BINANCE_MAX_LIMIT, max(lookback + 2, lookback))

        url = f"{self._base}/api/v3/klines"
        params = {
            "symbol":   binance_symbol,
            "interval": fetch_interval,
            "limit":    int(fetch_limit),
        }
        logger.info(
            "Binance klines | symbol=%s adapter_symbol=%s interval=%s target=%s limit=%d",
            symbol, binance_symbol, fetch_interval, timeframe, fetch_limit,
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Binance klines returned {resp.status_code} for {symbol} "
                f"(adapter_symbol={binance_symbol}, interval={fetch_interval}): "
                f"{resp.text[:500]}"
            )

        payload = resp.json()
        if not isinstance(payload, list):
            raise ValueError(
                f"Unexpected Binance klines payload for {symbol}: {payload}"
            )

        df = _binance_klines_to_df(payload)

        if resample_rule:
            df = _resample_df(df, resample_rule)
            logger.info(
                "Binance resampled → %d bars (%s) for %s last=%s",
                len(df), timeframe, symbol, df.index[-1] if not df.empty else "n/a",
            )
        else:
            logger.info(
                "Binance candles OK | symbol=%s rows=%d (%s) last=%s",
                symbol, len(df), timeframe, df.index[-1] if not df.empty else "n/a",
            )

        if df.empty:
            raise ValueError(f"Binance returned no usable candles for {symbol} ({timeframe}).")

        return df.tail(lookback)

    # ── LTP ──────────────────────────────────────────────────────────────────

    async def fetch_ltp(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> float:
        binance_symbol = self._resolve_symbol(symbol, adapter_symbol)
        url = f"{self._base}/api/v3/ticker/price"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params={"symbol": binance_symbol})

        if resp.status_code != 200:
            raise RuntimeError(
                f"Binance ticker/price returned {resp.status_code} for {symbol} "
                f"(adapter_symbol={binance_symbol}): {resp.text[:500]}"
            )

        payload = resp.json()
        price_raw = payload.get("price")
        if price_raw is None:
            raise ValueError(
                f"No 'price' in Binance ticker response for {symbol}: {payload}"
            )

        try:
            price = float(price_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Binance returned non-numeric price for {symbol}: {price_raw!r}"
            ) from exc

        if price <= 0:
            raise ValueError(f"Binance returned non-positive price for {symbol}: {price}")

        logger.info(
            "Binance LTP | symbol=%s adapter_symbol=%s price=%.10f",
            symbol, binance_symbol, price,
        )
        return price

    # ── Soft circuit / sanity bounds ─────────────────────────────────────────

    async def fetch_circuit_limits(
        self,
        symbol: str,
        adapter_symbol: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Binance has no exchange-side circuit breakers. We use the 24h
        high / low as a soft sanity band so the same risk-manager code path
        that blocks near-circuit equity entries also catches obviously
        outlier crypto fills.

        Returns ``None`` (and lets the caller fall back to system_configs
        percentage bands) when the 24h ticker is unavailable.
        """
        binance_symbol = self._resolve_symbol(symbol, adapter_symbol)
        url = f"{self._base}/api/v3/ticker/24hr"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params={"symbol": binance_symbol})
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Binance 24hr ticker failed for %s — %s. Falling back to defaults.",
                symbol, exc,
            )
            return None

        payload = resp.json()
        try:
            upper = float(payload.get("highPrice"))
            lower = float(payload.get("lowPrice"))
        except (TypeError, ValueError):
            logger.warning(
                "Binance 24hr ticker payload missing prices for %s: %s",
                symbol, payload,
            )
            return None

        if upper <= 0 or lower <= 0 or upper < lower:
            return None

        logger.info(
            "Binance 24h band | symbol=%s upper=%.10f lower=%.10f",
            symbol, upper, lower,
        )
        return {"upper_circuit": upper, "lower_circuit": lower}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interval_to_minutes(interval: str) -> int:
    """Best-effort Binance interval → minutes mapping."""
    unit = interval[-1]
    try:
        n = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse Binance interval: {interval!r}") from exc
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 60 * 24
    if unit == "w":
        return n * 60 * 24 * 7
    if unit == "M":
        return n * 60 * 24 * 30      # nominal — 30-day months
    raise ValueError(f"Unrecognised Binance interval unit: {interval!r}")


def _ratio_between_intervals(fine: str, coarse: str) -> int:
    """Return how many ``fine`` bars make up one ``coarse`` bar."""
    fine_mins   = _interval_to_minutes(fine)
    coarse_mins = _interval_to_minutes(coarse)
    if fine_mins <= 0:
        return 1
    return max(1, coarse_mins // fine_mins)
