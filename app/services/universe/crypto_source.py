"""
app/services/universe/crypto_source.py — KB-free candidate pool for ``crypto_all`` (§7).

Resolves the ``crypto_all`` universe source to a concrete, volume-ranked list of tradable
spot pairs from the exchange's own catalog — NOT the knowledge base (Invariant 11).

Ticker data is fetched directly from Binance's public 24hr ticker endpoint (one call,
no authentication). Adding a new exchange requires changing only this module.

Returns canonical ``BASE_QUOTE`` symbols (e.g. ``"BTC_USDT"``) — the shape the backtest
market-data layer understands (it maps ``BTC_USDT → BTCUSDT`` for the venue). Injectable and
fault-tolerant: on any failure it returns ``[]`` so the caller fails closed (no silent bad pool).
"""
from __future__ import annotations

import logging

import httpx

_BINANCE_BASE_URL = "https://api.binance.com"
_BINANCE_TICKER_PATH = "/api/v3/ticker/24hr"

logger = logging.getLogger(__name__)

# Leveraged / wrapped tokens we never want in a "most active" spot universe.
_EXCLUDE_TOKENS = ("UP", "DOWN", "BULL", "BEAR")
_SUPPORTED_QUOTES = ("USDT", "USDC")
# Stablecoin / fiat BASES — high turnover but no momentum; a "most active crypto"
# universe should hold real assets, not stable-to-stable pairs (USDC/USDT, USD1/USDT…).
_STABLE_BASES = frozenset({
    "USDC", "USDT", "USD1", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "PYUSD",
    "EUR", "EURI", "GBP", "AEUR", "XUSD",
})


def _to_canonical(symbol: str, quote: str) -> str | None:
    """``"BTCUSDT"`` + quote ``"USDT"`` → ``"BTC_USDT"``; None when it doesn't fit or is a
    leveraged token / stable-to-stable pair (filtered out of a 'most active' universe)."""
    if not symbol.endswith(quote):
        return None
    base = symbol[: -len(quote)]
    if not base or base in _STABLE_BASES or any(tag in base for tag in _EXCLUDE_TOKENS):
        return None
    return f"{base}_{quote}"


# How each universe rank metric is computed from a single 24h ticker row. The backtest feed
# does NOT serve daily crypto bars, but Binance's ticker gives volume, 24h % change, and the
# 24h high/low for EVERY pair in one call — enough to rank "most active", "biggest gainers/
# losers", and "most volatile" accurately, with no per-coin OHLCV fetch. Metrics that need
# longer history (rsi/distance_52w/rel_strength) fall back to volume (noted to the user).
def _ticker_metric(row: dict, by: str) -> float | None:
    try:
        last = float(row.get("lastPrice") or 0.0)
        if by in ("volume", "rvol"):
            return float(row.get("quoteVolume") or 0.0)
        if by in ("pct_change", "momentum"):
            return float(row.get("priceChangePercent") or 0.0)
        if by == "atr_pct":
            hi, lo = float(row.get("highPrice") or 0.0), float(row.get("lowPrice") or 0.0)
            return (hi - lo) / last * 100.0 if last > 0 else None
    except (TypeError, ValueError):
        return None
    return None  # rsi / distance_52w / rel_strength — not derivable from the ticker


# Metrics the ticker can rank directly; anything else falls back to volume.
_TICKER_RANKABLE = frozenset({"volume", "rvol", "pct_change", "momentum", "atr_pct"})


async def _fetch_ticker_rows(quote: str, base_url: str | None = None) -> list[dict]:
    """Fetch all 24h ticker rows from Binance (one call)."""
    base = (base_url or _BINANCE_BASE_URL).rstrip("/")
    url = f"{base}{_BINANCE_TICKER_PATH}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        rows = resp.json()
    if not isinstance(rows, list):
        return []
    return rows


async def rank_supported_crypto(
    *, by: str, order: str = "desc", limit: int = 10,
    supported_bases: set[str] | None = None, quote: str = "USDT",
    base_url: str | None = None,
) -> tuple[list[str], str | None]:
    """Rank SUPPORTED crypto by ``by`` using Binance's 24h ticker (one call).

    Returns ``(members, note)``. ``by`` ∈ {volume, rvol, pct_change, momentum, atr_pct} is
    ranked directly; any other metric falls back to volume with a user-facing ``note``
    explaining why (the ticker has no long-history metrics for crypto). ``order`` desc/asc
    (e.g. biggest gainers vs losers). Members are canonical ``BASE_QUOTE`` symbols, restricted
    to ``supported_bases`` (the platform's tradable coins).
    """
    quote = (quote or "USDT").upper()
    if quote not in _SUPPORTED_QUOTES:
        quote = "USDT"
    note = None
    effective_by = by
    if by not in _TICKER_RANKABLE:
        note = f"Crypto has no live '{by}' feed; ranked by volume instead."
        effective_by = "volume"
    try:
        rows = await _fetch_ticker_rows(quote, base_url)
    except Exception as exc:  # noqa: BLE001 — fail-closed
        logger.warning("crypto_source|ticker_fetch_failed|err=%s", exc)
        return [], note
    if not rows:
        return [], note

    scored: list[tuple[float, str]] = []
    for row in rows:
        canonical = _to_canonical(str(row.get("symbol") or ""), quote)
        if canonical is None:
            continue
        if supported_bases is not None and canonical.rsplit("_", 1)[0] not in supported_bases:
            continue
        val = _ticker_metric(row, effective_by)
        if val is not None:
            scored.append((val, canonical))
    scored.sort(key=lambda t: t[0], reverse=(order != "asc"))
    members = [sym for _, sym in scored[: max(1, limit)]]
    logger.info(
        "📡 crypto rank | by=%s(%s) | supported_candidates=%d | 🏆 top %d: %s%s",
        by, order, len(scored), max(1, limit), members, (" | " + note) if note else "",
    )
    return members, note


async def fetch_binance_volume_ranked(
    *, limit: int = 30, quote: str = "USDT", base_url: str | None = None,
    supported_bases: set[str] | None = None,
) -> list[str]:
    """Top-``limit`` spot pairs in ``quote`` by 24h quote-volume, as canonical symbols.

    Fetches from Binance's public ticker API (one HTTP call). When ``supported_bases``
    is given, only pairs whose base coin is in that set are considered.
    Returns ``[]`` on any error (fail-closed).
    """
    quote = (quote or "USDT").upper()
    if quote not in _SUPPORTED_QUOTES:
        quote = "USDT"
    try:
        rows = await _fetch_ticker_rows(quote, base_url)
    except Exception as exc:  # noqa: BLE001 — fail-closed
        logger.warning("crypto_source|ticker_fetch_failed|err=%s", exc)
        return []
    if not rows:
        return []

    ranked: list[tuple[float, str]] = []
    for row in rows:
        sym = str(row.get("symbol") or "")
        canonical = _to_canonical(sym, quote)
        if canonical is None:
            continue
        if supported_bases is not None and canonical.rsplit("_", 1)[0] not in supported_bases:
            continue
        try:
            vol = float(row.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol > 0:
            ranked.append((vol, canonical))
    ranked.sort(key=lambda t: t[0], reverse=True)
    pool = [sym for _, sym in ranked[: max(1, limit)]]
    _top = [f"{sym}({vol/1e9:.2f}B)" for vol, sym in ranked[: max(1, limit)]]
    logger.info(
        "\n┌─📡 BINANCE CRYPTO POOL (%s) ────────────────────────────────────────\n"
        "│  fetched   : %d total pairs\n"
        "│  real coins: %d (after dropping stablecoin/leveraged)\n"
        "│  🏆 top %d by 24h volume: %s\n"
        "└──────────────────────────────────────────────────────────────────────",
        quote, len(rows), len(ranked), max(1, limit), ", ".join(_top),
    )
    return pool
