"""
app/services/execution/ref_data_service.py
──────────────────────────────────────────
Read-only helpers that query the ref_data schema (owned by other services).

Multi-asset canonical-symbol model
──────────────────────────────────
ref_data.instruments stores rows in canonical form per asset class:

  equity_cash : "equity:RELIANCE@NSE"
              | "equity:RELIANCE@BSE"
  crypto_spot : "crypto_spot:BTC_USDT@BINANCE_SPOT"
              | "crypto_spot:ETH_USDC@BINANCE_SPOT"

Lookup chain (per adapter):
  symbol → canonical_symbol (deterministic from asset_class)
         → instruments.id
         → instrument_adapter_mappings.adapter_symbol
           (e.g. "NSE_EQ|INE002A01018" for upstox_rest,
                 "BTCUSDT"             for binance_rest)

For backward compatibility, the legacy public function
``lookup_upstox_instrument_key`` is preserved as a thin wrapper over
``lookup_adapter_symbol(..., asset_class=AssetClass.equity_cash)``.

system_configs defaults
───────────────────────
Per-asset-class defaults live in ref_data.system_configs:

  equity_cash → nse.tick_size_default, nse.lot_size_default,
                nse.upper_circuit_default, nse.lower_circuit_default
  crypto_spot → crypto_spot.tick_size_default, crypto_spot.lot_size_default,
                crypto_spot.upper_bound_pct,  crypto_spot.lower_bound_pct
                (Binance has no exchange-side circuit breakers; we apply a
                soft band around LTP using these percentages, configurable
                per-tenant.)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant_context import default_adapter_id_for_context
from app.schemas.execution import AssetClass

logger = logging.getLogger(__name__)

# ── Adapter ids (must match rows in ref_data.adapters) ────────────────────────
ADAPTER_UPSTOX_REST  = "upstox_rest"
ADAPTER_BINANCE_REST = "binance_rest"

# Default market-data adapter per asset class.
_DEFAULT_ADAPTER_BY_ASSET_CLASS: dict[AssetClass, str] = {
    AssetClass.equity_cash: ADAPTER_UPSTOX_REST,
    AssetClass.crypto_spot: ADAPTER_BINANCE_REST,
}


def default_adapter_id(asset_class: AssetClass) -> str:
    """Return the adapter_id for live market-data, respecting tenant context when set."""
    try:
        return default_adapter_id_for_context(asset_class)
    except KeyError as exc:
        raise ValueError(
            f"No default market-data adapter is registered for asset_class={asset_class!r}."
        ) from exc


# ── Symbol normalisation ──────────────────────────────────────────────────────

_VALID_CRYPTO_BASE_QUOTE_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")


def _normalize_equity_ticker(symbol: str) -> str:
    """
    Strip casing and exchange suffixes to recover the raw NSE/BSE ticker.

      "RELIANCE.NS"  → "RELIANCE"
      "reliance.ns"  → "RELIANCE"
      "Reliance"     → "RELIANCE"
      "NSE:RELIANCE" → "RELIANCE"
    """
    s = symbol.strip().upper()
    if ":" in s:
        _, s = s.split(":", 1)
    for suffix in (".NS", ".BO", ".BSE", ".NSE"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _equity_venue_from_symbol(symbol: str) -> str:
    """`.BO/.BSE` → BSE; everything else (incl. bare ticker) → NSE."""
    s = symbol.strip().upper()
    if s.endswith(".BO") or s.endswith(".BSE"):
        return "BSE"
    return "NSE"


def _normalize_crypto_pair(symbol: str) -> str:
    """
    Accept any of the common shapes and return the KB-canonical BASE_QUOTE form.

      "BTC_USDT"        → "BTC_USDT"
      "btc_usdt"        → "BTC_USDT"
      "BTCUSDT"         → raises ValueError (ambiguous — base/quote unknown)
      "crypto_spot:BTC_USDT@BINANCE_SPOT" → "BTC_USDT"
      "BTC/USDT"        → "BTC_USDT"
    """
    s = symbol.strip().upper()

    if s.startswith("CRYPTO_SPOT:") and "@" in s:
        body = s.split(":", 1)[1].split("@", 1)[0]
        s = body

    if "/" in s:
        s = s.replace("/", "_")

    if not _VALID_CRYPTO_BASE_QUOTE_RE.match(s):
        raise ValueError(
            f"Crypto symbol '{symbol}' is not in canonical BASE_QUOTE form "
            "(e.g. 'BTC_USDT'). Bare 'BTCUSDT' is ambiguous and rejected."
        )
    return s


def to_canonical_symbol(symbol: str, asset_class: AssetClass) -> str:
    """
    Convert a free-form input symbol into the canonical form stored in
    ``ref_data.instruments.canonical_symbol`` for the given asset class.

      equity_cash : "RELIANCE.NS"    → "equity:RELIANCE@NSE"
                    "RELIANCE.BO"    → "equity:RELIANCE@BSE"
                    "equity:RELIANCE@NSE" → idempotent
      crypto_spot : "BTC_USDT"       → "crypto_spot:BTC_USDT@BINANCE_SPOT"
                    "crypto_spot:BTC_USDT@BINANCE_SPOT" → idempotent
    """
    s = symbol.strip()

    if asset_class == AssetClass.equity_cash:
        if s.lower().startswith("equity:") and "@" in s:
            head, venue = s.split("@", 1)
            prefix, ticker = head.split(":", 1)
            return f"{prefix.lower()}:{ticker.upper()}@{venue.upper()}"
        return f"equity:{_normalize_equity_ticker(symbol)}@{_equity_venue_from_symbol(symbol)}"

    if asset_class == AssetClass.crypto_spot:
        pair = _normalize_crypto_pair(symbol)
        return f"crypto_spot:{pair}@BINANCE_SPOT"

    raise ValueError(f"Unsupported asset_class: {asset_class!r}")


# Legacy helper kept for back-compat. Use ``to_canonical_symbol`` going forward.
def _to_canonical(symbol: str) -> str:  # pragma: no cover - kept for back-compat
    return to_canonical_symbol(symbol, AssetClass.equity_cash)


# ── Generic adapter-symbol lookup ─────────────────────────────────────────────

async def lookup_adapter_symbol(
    symbol: str,
    asset_class: AssetClass,
    db: AsyncSession,
    *,
    adapter_id: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the *venue-native* symbol for a strategy symbol by joining
    ``ref_data.instruments`` and ``ref_data.instrument_adapter_mappings``.

    Parameters
    ----------
    symbol       : Strategy symbol in any of the accepted shapes (see
                   ``to_canonical_symbol``).
    asset_class  : Drives canonicalisation and the default adapter selection.
    db           : Async SQLAlchemy session.
    adapter_id   : Override the default adapter (useful when more than one
                   market-data source exists for an asset class).

    Returns
    -------
    The adapter-specific symbol (e.g. ``"NSE_EQ|INE002A01018"`` for Upstox,
    ``"BTCUSDT"`` for Binance), or ``None`` if no active mapping exists.

    Errors during DB I/O are swallowed and logged — callers fall back to
    asset-class-specific best-effort defaults.
    """
    effective_adapter = adapter_id or default_adapter_id(asset_class)

    try:
        canonical = to_canonical_symbol(symbol, asset_class)
    except ValueError as exc:
        logger.error(
            "ref_data canonicalisation failed | symbol=%s asset_class=%s: %s",
            symbol, asset_class.value, exc,
        )
        return None

    try:
        row = await db.execute(
            text(
                """
                SELECT iam.adapter_symbol
                FROM   ref_data.instruments i
                JOIN   ref_data.instrument_adapter_mappings iam
                       ON iam.instrument_id = i.id
                WHERE  i.canonical_symbol = :canonical
                  AND  i.asset_class_id   = :asset_class
                  AND  iam.adapter_id     = :adapter_id
                  AND  iam.is_active      = true
                  AND  i.is_active        = true
                LIMIT 1
                """
            ),
            {
                "canonical": canonical,
                "asset_class": asset_class.value,
                "adapter_id": effective_adapter,
            },
        )
        result = row.fetchone()

        if result:
            adapter_symbol = result[0]
            logger.info(
                "ref_data lookup OK | symbol=%s asset_class=%s canonical=%s "
                "adapter=%s adapter_symbol=%s",
                symbol, asset_class.value, canonical,
                effective_adapter, adapter_symbol,
            )
            return adapter_symbol

        logger.warning(
            "ref_data lookup miss | symbol=%s asset_class=%s canonical=%s adapter=%s",
            symbol, asset_class.value, canonical, effective_adapter,
        )
        return None

    except Exception as exc:  # noqa: BLE001 — defensive: never hard-fail the evaluator
        logger.error(
            "ref_data lookup failed | symbol=%s asset_class=%s adapter=%s: %s",
            symbol, asset_class.value, effective_adapter, exc, exc_info=True,
        )
        return None


# ── Back-compat shim: lookup_upstox_instrument_key ────────────────────────────

async def lookup_upstox_instrument_key(
    symbol: str,
    db: AsyncSession,
) -> Optional[str]:
    """
    Equity-only convenience wrapper retained for legacy callers.

    Equivalent to:
      lookup_adapter_symbol(symbol, AssetClass.equity_cash, db,
                            adapter_id=ADAPTER_UPSTOX_REST)

    New code should call ``lookup_adapter_symbol`` directly with an explicit
    ``asset_class`` so the crypto path resolves correctly.
    """
    return await lookup_adapter_symbol(
        symbol,
        AssetClass.equity_cash,
        db,
        adapter_id=ADAPTER_UPSTOX_REST,
    )


# ── Equity metadata (lot/segment/isin) ────────────────────────────────────────

async def lookup_equity_metadata(
    symbol: str,
    db: AsyncSession,
) -> Optional[dict]:
    """
    Return basic equity metadata from ref_data for a symbol:
      ticker, isin, lot_size, segment

    ISIN is derived from the Upstox adapter_symbol suffix
    ("NSE_EQ|INE002A01018" → "INE002A01018"). Returns None if not found.
    """
    canonical = to_canonical_symbol(symbol, AssetClass.equity_cash)
    ticker = _normalize_equity_ticker(symbol)
    try:
        row = await db.execute(
            text(
                """
                SELECT ie.lot_size,
                       ie.segment,
                       iam.adapter_symbol
                FROM   ref_data.instruments i
                JOIN   ref_data.instrument_equity ie
                       ON ie.instrument_id = i.id
                LEFT   JOIN ref_data.instrument_adapter_mappings iam
                       ON iam.instrument_id = i.id
                      AND iam.adapter_id    = :adapter_id
                      AND iam.is_active     = true
                WHERE  i.canonical_symbol = :canonical
                  AND  i.is_active        = true
                LIMIT 1
                """
            ),
            {"canonical": canonical, "adapter_id": ADAPTER_UPSTOX_REST},
        )
        result = row.fetchone()
        if not result:
            return None

        lot_size, segment, adapter_symbol = result[0], result[1], result[2]
        isin: Optional[str] = None
        if adapter_symbol and "|" in adapter_symbol:
            isin = adapter_symbol.split("|", 1)[1] or None

        return {
            "ticker":   ticker,
            "isin":     isin,
            "lot_size": lot_size,
            "segment":  segment,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("ref_data equity lookup failed for %s: %s", symbol, exc)
        return None


# ── Crypto instrument metadata (tick/qty step/min notional) ───────────────────

async def lookup_crypto_instrument(
    symbol: str,
    db: AsyncSession,
) -> Optional[dict]:
    """
    Return canonical crypto instrument numeric attributes:
      price_tick_size, qty_step_size, min_notional

    These are sourced from ``ref_data.instruments`` directly (no per-asset-class
    sub-table is required because crypto-specific fields already live on the
    parent row). Returns ``None`` if no row exists.
    """
    canonical = to_canonical_symbol(symbol, AssetClass.crypto_spot)
    try:
        row = await db.execute(
            text(
                """
                SELECT price_tick_size, qty_step_size, min_notional
                FROM   ref_data.instruments
                WHERE  canonical_symbol = :canonical
                  AND  asset_class_id   = 'crypto_spot'
                  AND  is_active        = true
                LIMIT 1
                """
            ),
            {"canonical": canonical},
        )
        result = row.fetchone()
        if not result:
            logger.warning(
                "ref_data crypto instrument miss | symbol=%s canonical=%s",
                symbol, canonical,
            )
            return None
        return {
            "price_tick_size": float(result[0]) if result[0] is not None else None,
            "qty_step_size":   float(result[1]) if result[1] is not None else None,
            "min_notional":    float(result[2]) if result[2] is not None else None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("ref_data crypto lookup failed for %s: %s", symbol, exc)
        return None


# ── InstrumentDefaults (asset-class-aware) ────────────────────────────────────

@dataclass
class InstrumentDefaults:
    """
    Runtime instrument parameters derived from ref_data.

    tick_size     — price rounding granularity (e.g. 0.05 for NSE equity,
                    0.01 for BTCUSDT)
    lot_size      — minimum order quantity multiple (1 for NSE cash; for
                    crypto we use 0.0 here and instead rely on ``qty_step_size``
                    in the crypto rounder)
    qty_step_size — fractional quantity step for crypto (None for equity)
    min_notional  — min trade value enforced by the venue (USDT for crypto;
                    None for equity, where ``min_trade_value`` from the risk
                    config plays the same role)
    upper_circuit — absolute upper bound price; None if unknown
    lower_circuit — absolute lower bound price; None if unknown
    """
    tick_size: float
    lot_size: int
    upper_circuit: Optional[float]
    lower_circuit: Optional[float]
    qty_step_size: Optional[float] = None
    min_notional:  Optional[float] = None


# ── Safe fallbacks (used when system_configs / instruments queries fail) ──────
#
# Equity (NSE cash)
_FALLBACK_TICK_SIZE_EQUITY        = 0.05
_FALLBACK_LOT_SIZE_EQUITY         = 1
_FALLBACK_UPPER_CIRCUIT_PCT       = 0.20
_FALLBACK_LOWER_CIRCUIT_PCT       = 0.20

# Crypto (Binance spot) — no exchange-side circuit breakers; we apply a soft
# 10% band around LTP as a sanity guard. Same band protects the equity path
# when Upstox circuit limits are unavailable.
_FALLBACK_TICK_SIZE_CRYPTO        = 0.01
_FALLBACK_QTY_STEP_CRYPTO         = 0.00001
_FALLBACK_MIN_NOTIONAL_CRYPTO     = 10.0   # USDT
_FALLBACK_CRYPTO_BAND_PCT         = 0.10   # ±10%


# system_configs keys by asset class.
_SYSTEM_CONFIG_KEYS_BY_ASSET_CLASS: dict[AssetClass, tuple[str, ...]] = {
    AssetClass.equity_cash: (
        "nse.tick_size_default",
        "nse.lot_size_default",
        "nse.upper_circuit_default",
        "nse.lower_circuit_default",
    ),
    AssetClass.crypto_spot: (
        "crypto_spot.tick_size_default",
        "crypto_spot.qty_step_default",
        "crypto_spot.min_notional_default",
        "crypto_spot.upper_bound_pct",
        "crypto_spot.lower_bound_pct",
    ),
}


# Per-asset-class cache. Each entry: {key: value}. Loaded lazily on first use.
_SYSTEM_CONFIG_CACHE: dict[AssetClass, dict[str, float]] = {}


async def _load_system_configs(
    db: AsyncSession,
    asset_class: AssetClass,
) -> dict[str, float]:
    """Populate (or refresh) the system_configs cache for an asset class."""
    keys = _SYSTEM_CONFIG_KEYS_BY_ASSET_CLASS.get(asset_class, ())
    if not keys:
        logger.warning(
            "No system_configs key list registered for asset_class=%s",
            asset_class.value,
        )
        _SYSTEM_CONFIG_CACHE[asset_class] = {}
        return _SYSTEM_CONFIG_CACHE[asset_class]

    try:
        rows = await db.execute(
            text(
                "SELECT key, value FROM ref_data.system_configs "
                "WHERE key = ANY(:keys)"
            ),
            {"keys": list(keys)},
        )
        mapping: dict[str, float] = {}
        for k, v in rows.fetchall():
            try:
                mapping[str(k)] = float(v)
            except (TypeError, ValueError):
                logger.warning(
                    "system_configs value not numeric — skipped | key=%s value=%r",
                    k, v,
                )
        _SYSTEM_CONFIG_CACHE[asset_class] = mapping
        logger.debug(
            "system_configs loaded | asset_class=%s entries=%d keys=%s",
            asset_class.value, len(mapping), list(mapping.keys()),
        )
        return mapping
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load ref_data.system_configs for asset_class=%s: %s",
            asset_class.value, exc,
        )
        _SYSTEM_CONFIG_CACHE[asset_class] = {}
        return _SYSTEM_CONFIG_CACHE[asset_class]


async def _system_configs(
    db: AsyncSession,
    asset_class: AssetClass,
) -> dict[str, float]:
    """Return the cached system_configs map for the asset class, loading if needed."""
    cached = _SYSTEM_CONFIG_CACHE.get(asset_class)
    if cached is None:
        return await _load_system_configs(db, asset_class)
    return cached


def reset_system_configs_cache() -> None:
    """Test-only helper: drop the system_configs cache so reloads pick up fresh rows."""
    _SYSTEM_CONFIG_CACHE.clear()


# ── lookup_instrument_defaults ────────────────────────────────────────────────

async def lookup_instrument_defaults(
    ltp: float,
    db: AsyncSession,
    live_circuit_limits: Optional[dict] = None,
    *,
    asset_class: AssetClass = AssetClass.equity_cash,
    symbol: Optional[str] = None,
) -> InstrumentDefaults:
    """
    Build asset-class-aware InstrumentDefaults.

    Parameters
    ----------
    ltp                 : Last traded price — used to convert percent bands
                          into absolute upper/lower circuit prices when no
                          live numbers are available.
    db                  : Async DB session.
    live_circuit_limits : Optional ``{"upper_circuit": float, "lower_circuit": float}``
                          from the venue. When provided, the absolute prices
                          override the percentage-based defaults.
    asset_class         : ``equity_cash`` or ``crypto_spot``.
    symbol              : Optional — when supplied for ``crypto_spot``, we
                          enrich the result with per-instrument
                          ``tick_size`` / ``qty_step_size`` / ``min_notional``
                          fetched from ``ref_data.instruments``.

    Returns
    -------
    InstrumentDefaults — never None; falls back to safe per-asset-class values
    on any failure.
    """
    cfg = await _system_configs(db, asset_class)

    if asset_class == AssetClass.equity_cash:
        tick  = cfg.get("nse.tick_size_default",     _FALLBACK_TICK_SIZE_EQUITY)
        lot   = int(cfg.get("nse.lot_size_default",  _FALLBACK_LOT_SIZE_EQUITY))
        u_pct = _percent_default(cfg, "nse.upper_circuit_default", _FALLBACK_UPPER_CIRCUIT_PCT)
        l_pct = _percent_default(cfg, "nse.lower_circuit_default", _FALLBACK_LOWER_CIRCUIT_PCT)

        upper: Optional[float] = ltp * (1 + u_pct)
        lower: Optional[float] = ltp * (1 - l_pct)

        if live_circuit_limits:
            live_upper = live_circuit_limits.get("upper_circuit")
            live_lower = live_circuit_limits.get("lower_circuit")
            if live_upper is not None:
                upper = float(live_upper)
            if live_lower is not None:
                lower = float(live_lower)

        logger.info(
            "InstrumentDefaults (equity_cash) | tick=%.4f lot=%d "
            "upper=%.2f lower=%.2f",
            tick, lot, upper or 0, lower or 0,
        )
        return InstrumentDefaults(
            tick_size=float(tick),
            lot_size=int(lot),
            upper_circuit=upper,
            lower_circuit=lower,
        )

    if asset_class == AssetClass.crypto_spot:
        tick_default          = cfg.get("crypto_spot.tick_size_default",  _FALLBACK_TICK_SIZE_CRYPTO)
        qty_step_default      = cfg.get("crypto_spot.qty_step_default",   _FALLBACK_QTY_STEP_CRYPTO)
        min_notional_default  = cfg.get("crypto_spot.min_notional_default", _FALLBACK_MIN_NOTIONAL_CRYPTO)
        upper_pct             = _percent_default(cfg, "crypto_spot.upper_bound_pct", _FALLBACK_CRYPTO_BAND_PCT)
        lower_pct             = _percent_default(cfg, "crypto_spot.lower_bound_pct", _FALLBACK_CRYPTO_BAND_PCT)

        # Per-instrument override from ref_data.instruments (preferred).
        instrument_row = (
            await lookup_crypto_instrument(symbol, db) if symbol else None
        )
        tick         = (instrument_row or {}).get("price_tick_size") or tick_default
        qty_step     = (instrument_row or {}).get("qty_step_size")   or qty_step_default
        min_notional = (instrument_row or {}).get("min_notional")    or min_notional_default

        upper: Optional[float] = ltp * (1 + upper_pct) if ltp > 0 else None
        lower: Optional[float] = ltp * (1 - lower_pct) if ltp > 0 else None

        if live_circuit_limits:
            live_upper = live_circuit_limits.get("upper_circuit")
            live_lower = live_circuit_limits.get("lower_circuit")
            if live_upper is not None:
                upper = float(live_upper)
            if live_lower is not None:
                lower = float(live_lower)

        logger.info(
            "InstrumentDefaults (crypto_spot) | symbol=%s tick=%.10g "
            "qty_step=%.10g min_notional=%.4f upper=%.4f lower=%.4f",
            symbol, tick, qty_step, min_notional,
            upper or 0.0, lower or 0.0,
        )
        return InstrumentDefaults(
            tick_size=float(tick),
            # For crypto we keep ``lot_size`` as a sentinel ``1`` (the risk
            # manager treats integer lot rounding as a no-op when qty_step
            # is set) and rely on ``qty_step_size`` for fractional rounding.
            lot_size=1,
            upper_circuit=upper,
            lower_circuit=lower,
            qty_step_size=float(qty_step),
            min_notional=float(min_notional),
        )

    raise ValueError(f"Unsupported asset_class for instrument defaults: {asset_class!r}")


def _percent_default(cfg: dict[str, float], key: str, fallback: float) -> float:
    """
    Read a percent default from system_configs.

    Some seeders persist these as bare percentages (``20`` meaning 20%) and
    others as fractions (``0.20``). We normalise both: anything > 1 is treated
    as a percent and divided by 100.
    """
    raw = cfg.get(key, fallback)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float(fallback)
    return v / 100.0 if v > 1.0 else v
