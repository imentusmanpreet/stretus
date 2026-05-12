"""
app/services/execution/ref_data_service.py
──────────────────────────────────────────
Read-only helpers that query the ref_data schema (owned by other services).

We use raw SQL via SQLAlchemy text() to avoid tight ORM coupling with
a schema that is maintained outside this codebase.

Lookup chain for Upstox instrument key:
  ref_data.equities           → match ticker (uppercase) → equity_id
  ref_data.equity_data_source_mappings → equity_id + Upstox provider_id → source_symbol
  ref_data.data_providers     → WHERE name = 'Upstox'

InstrumentDefaults (replaces ai_strategy.instrument_metadata):
  ref_data.system_configs keys:
    nse.tick_size_default     → tick_size    (e.g. 0.05)
    nse.lot_size_default      → lot_size     (e.g. 1)
    nse.upper_circuit_default → fraction of LTP (e.g. 0.20 = 20%)
    nse.lower_circuit_default → fraction of LTP (e.g. 0.20 = 20%)

  If live circuit prices are available (from Upstox API), they override
  the fraction-based defaults.

Symbol normalisation:
  "RELIANCE.NS" | "reliance.ns" | "Reliance" → ticker = "RELIANCE"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Provider name as stored in ref_data.data_providers
_UPSTOX_PROVIDER_NAME = "Upstox"

# Cached provider UUID (populated on first lookup per process lifetime)
_upstox_provider_id: Optional[str] = None


def _normalize_ticker(symbol: str) -> str:
    """
    Convert any incoming symbol format to the uppercase ticker
    stored in ref_data.equities.

    Examples:
      "RELIANCE.NS"  → "RELIANCE"
      "reliance.ns"  → "RELIANCE"
      "Reliance"     → "RELIANCE"
      "NSE:RELIANCE" → "RELIANCE"
      "RELIANCE"     → "RELIANCE"
    """
    s = symbol.strip().upper()
    if ":" in s:
        _, s = s.split(":", 1)
    for suffix in (".NS", ".BO", ".BSE", ".NSE"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


async def _get_upstox_provider_id(db: AsyncSession) -> Optional[str]:
    """Return the UUID of the Upstox data provider, cached for the process lifetime."""
    global _upstox_provider_id
    if _upstox_provider_id:
        return _upstox_provider_id

    row = await db.execute(
        text(
            "SELECT id FROM ref_data.data_providers "
            "WHERE name = :name AND is_active = true "
            "LIMIT 1"
        ),
        {"name": _UPSTOX_PROVIDER_NAME},
    )
    result = row.fetchone()
    if result:
        _upstox_provider_id = str(result[0])
        logger.debug("Upstox provider_id resolved: %s", _upstox_provider_id)
    return _upstox_provider_id


async def lookup_upstox_instrument_key(
    symbol: str,
    db: AsyncSession,
) -> Optional[str]:
    """
    Resolve the Upstox instrument key (e.g. "NSE_EQ|INE002A01018") for a symbol
    by querying ref_data.equities → ref_data.equity_data_source_mappings.

    Returns None if the symbol is not found or has no Upstox mapping.

    The symbol may arrive in any casing / with exchange suffixes:
      "RELIANCE.NS", "reliance", "Reliance.ns" → all resolve to ticker "RELIANCE".
    """
    ticker = _normalize_ticker(symbol)

    try:
        provider_id = await _get_upstox_provider_id(db)
        if not provider_id:
            logger.warning("Upstox provider not found in ref_data.data_providers.")
            return None

        row = await db.execute(
            text(
                """
                SELECT m.source_symbol
                FROM   ref_data.equities e
                JOIN   ref_data.equity_data_source_mappings m
                       ON m.equity_id = e.id AND m.is_active = true
                WHERE  e.ticker      = :ticker
                  AND  m.provider_id = :provider_id
                LIMIT 1
                """
            ),
            {"ticker": ticker, "provider_id": provider_id},
        )
        result = row.fetchone()

        if result:
            key = result[0]
            logger.info(
                "ref_data lookup | symbol=%s ticker=%s upstox_key=%s",
                symbol, ticker, key,
            )
            return key

        logger.warning(
            "No Upstox instrument key found in ref_data for symbol=%s (ticker=%s).",
            symbol, ticker,
        )
        return None

    except Exception as exc:
        logger.error(
            "ref_data lookup failed for symbol=%s: %s", symbol, exc, exc_info=True
        )
        return None


async def lookup_equity_metadata(
    symbol: str,
    db: AsyncSession,
) -> Optional[dict]:
    """
    Return basic equity metadata from ref_data for a symbol:
      ticker, isin, lot_size, segment

    Returns None if symbol not found.
    """
    ticker = _normalize_ticker(symbol)
    try:
        row = await db.execute(
            text(
                "SELECT ticker, isin, lot_size, segment "
                "FROM ref_data.equities "
                "WHERE ticker = :ticker AND is_active = true "
                "LIMIT 1"
            ),
            {"ticker": ticker},
        )
        result = row.fetchone()
        if result:
            return {
                "ticker":   result[0],
                "isin":     result[1],
                "lot_size": result[2],
                "segment":  result[3],
            }
        return None
    except Exception as exc:
        logger.error("ref_data equity lookup failed for %s: %s", symbol, exc)
        return None


# ── InstrumentDefaults (replaces ai_strategy.instrument_metadata) ─────────────

@dataclass
class InstrumentDefaults:
    """
    Runtime instrument parameters built from ref_data.system_configs.

    tick_size    — price rounding granularity (e.g. 0.05)
    lot_size     — minimum order quantity multiple (e.g. 1 for NSE cash)
    upper_circuit — absolute upper circuit price (LTP × (1 + pct)); None if unknown
    lower_circuit — absolute lower circuit price (LTP × (1 - pct)); None if unknown
    """
    tick_size: float
    lot_size: int
    upper_circuit: Optional[float]
    lower_circuit: Optional[float]


# Safe fallback values used when the DB call fails entirely
_FALLBACK_TICK_SIZE        = 0.05
_FALLBACK_LOT_SIZE         = 1
_FALLBACK_UPPER_CIRCUIT_PCT = 0.20
_FALLBACK_LOWER_CIRCUIT_PCT = 0.20

# Cached system config defaults (populated on first request, valid for process lifetime)
_cached_tick_size:         Optional[float] = None
_cached_lot_size:          Optional[int]   = None
_cached_upper_circuit_pct: Optional[float] = None
_cached_lower_circuit_pct: Optional[float] = None


async def _load_system_configs(db: AsyncSession) -> None:
    """Populate the module-level cache from ref_data.system_configs."""
    global _cached_tick_size, _cached_lot_size, _cached_upper_circuit_pct, _cached_lower_circuit_pct

    keys = (
        "nse.tick_size_default",
        "nse.lot_size_default",
        "nse.upper_circuit_default",
        "nse.lower_circuit_default",
    )
    try:
        rows = await db.execute(
            text(
                "SELECT key, value FROM ref_data.system_configs "
                "WHERE key = ANY(:keys)"
            ),
            {"keys": list(keys)},
        )
        mapping = {r[0]: r[1] for r in rows.fetchall()}

        _cached_tick_size         = float(mapping.get("nse.tick_size_default",    _FALLBACK_TICK_SIZE))
        _cached_lot_size          = int(float(mapping.get("nse.lot_size_default", _FALLBACK_LOT_SIZE)))
        _cached_upper_circuit_pct = float(mapping.get("nse.upper_circuit_default", _FALLBACK_UPPER_CIRCUIT_PCT))
        _cached_lower_circuit_pct = float(mapping.get("nse.lower_circuit_default", _FALLBACK_LOWER_CIRCUIT_PCT))

        logger.debug(
            "system_configs loaded | tick=%.4f lot=%d upper_pct=%.2f lower_pct=%.2f",
            _cached_tick_size, _cached_lot_size,
            _cached_upper_circuit_pct, _cached_lower_circuit_pct,
        )
    except Exception as exc:
        logger.error("Failed to load ref_data.system_configs: %s", exc)


async def lookup_instrument_defaults(
    ltp: float,
    db: AsyncSession,
    live_circuit_limits: Optional[dict] = None,
) -> InstrumentDefaults:
    """
    Build an InstrumentDefaults from ref_data.system_configs defaults.

    Circuit prices are computed as:
      upper_circuit = ltp * (1 + nse.upper_circuit_default)   e.g. ltp * 1.20
      lower_circuit = ltp * (1 - nse.lower_circuit_default)   e.g. ltp * 0.80

    If live_circuit_limits (from Upstox real-time quotes) are provided,
    those absolute prices take priority over the percentage-based defaults.

    Args:
        ltp:                 Last traded price (used to convert circuit % → price)
        db:                  Async DB session
        live_circuit_limits: Optional dict {"upper_circuit": float, "lower_circuit": float}
                             from Upstox market-quote/quotes (absolute prices)
    """
    global _cached_tick_size, _cached_lot_size, _cached_upper_circuit_pct, _cached_lower_circuit_pct

    # Load from DB if cache is empty
    if _cached_tick_size is None:
        await _load_system_configs(db)

    tick  = _cached_tick_size         or _FALLBACK_TICK_SIZE
    lot   = _cached_lot_size          or _FALLBACK_LOT_SIZE
    u_pct = _cached_upper_circuit_pct or _FALLBACK_UPPER_CIRCUIT_PCT
    l_pct = _cached_lower_circuit_pct or _FALLBACK_LOWER_CIRCUIT_PCT

    # Default circuit prices from percentage defaults
    upper: Optional[float] = ltp * (1 + u_pct)
    lower: Optional[float] = ltp * (1 - l_pct)

    # Override with live Upstox values if present
    if live_circuit_limits:
        live_upper = live_circuit_limits.get("upper_circuit")
        live_lower = live_circuit_limits.get("lower_circuit")
        if live_upper is not None:
            upper = float(live_upper)
        if live_lower is not None:
            lower = float(live_lower)

    logger.info(
        "InstrumentDefaults | tick=%.4f lot=%d upper_circuit=%.2f lower_circuit=%.2f",
        tick, lot, upper or 0, lower or 0,
    )

    return InstrumentDefaults(
        tick_size=tick,
        lot_size=lot,
        upper_circuit=upper,
        lower_circuit=lower,
    )
