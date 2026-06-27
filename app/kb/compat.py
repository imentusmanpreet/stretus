"""
app/kb/compat.py — Compatibility helpers backed by the new KB.

These mirror the legacy app.core.signals_config_loader functions so files like
app/services/strategy/builder.py can drop the legacy import without changing
their call sites. Kept intentionally tiny — once builder.py is refactored to
use kb.* directly, this module can be deleted.
"""
from __future__ import annotations

from typing import Any

from app.kb import kb
from app.kb.schemas import Stock


AMBIGUOUS_STOCK_VALIDATION_CODE = "validation.ambiguous_stock"


def _stock_option(stock: Stock) -> dict[str, str]:
    return {
        "symbol": stock.symbol,
        "display_name": stock.display_name,
        "exchange": stock.exchange,
        "sector": stock.sector,
    }


def ambiguous_stock_validation_facts(query: str, matches: list[Stock] | tuple[Stock, ...]) -> dict[str, Any]:
    return {
        "stock_query": " ".join(str(query or "").split()).strip(),
        "stock_options": [_stock_option(stock) for stock in matches],
    }


def get_all_market_configs() -> dict[str, dict]:
    """Return a dict of {market_key: market_dict} compatible with the legacy
    MARKET_CONFIG used in StrategyBuilder. The new KB doesn't carry per-market
    SL/TP/timeframe defaults, so we synthesize sensible Indian-stocks defaults
    that match what the legacy YAML used to ship with."""
    return {
        "indian_stocks": {
            "suffix": ".NS",
            "default_stop_loss": 1.5,
            "default_take_profit": 3.0,
            "default_timeframe": "15m",
        },
        "indian_indices": {
            "suffix": "",
            "default_stop_loss": 1.5,
            "default_take_profit": 4.0,
            "default_timeframe": "15m",
        },
    }


def get_risk_defaults() -> dict:
    """Static defaults for the legacy risk-summary view."""
    return {
        "position_sizing": "Risk based",
        "execution_mode": "Backtest",
        "trading_window": "9:15 - 15:30",
        "risk_validation": "system risk guardrails",
    }


def get_experience_risk(experience: str) -> dict:
    """Return per-trade and daily-loss-cap percentages for an experience tier."""
    tier = kb.get_risk_tier(experience)
    return {
        "per_trade_risk_pct": tier.per_trade_risk_pct,
        "daily_loss_cap_pct": tier.daily_loss_cap_pct,
        "max_open_positions": tier.max_open_positions,
    }


def get_daily_loss_cap_bounds() -> tuple[float, float]:
    """Min/max clamp for the daily loss cap. Matches the legacy YAML defaults."""
    return 1.0, 5.0


async def resolve_supported_stock(query: str) -> dict | None:
    """KB-backed replacement for app.services.knowledge.stock_matcher.resolve_supported_stock.

    Synchronous in spirit, but kept async for drop-in compatibility with the
    legacy await call sites in chat_service.py.
    """
    if not query or not str(query).strip():
        return None
    resolution = kb.resolve_stock_query(str(query))
    if resolution.is_ambiguous:
        facts = ambiguous_stock_validation_facts(str(query), resolution.ambiguous_matches)
        return {
            "ambiguous": True,
            "validation_code": AMBIGUOUS_STOCK_VALIDATION_CODE,
            "validation_facts": facts,
            "matches": facts["stock_options"],
        }

    if resolution.suggestion is not None:
        # Fuzzy "did you mean" — surface as a single-option confirmation, reusing
        # the ambiguous picker so the user explicitly accepts the correction
        # instead of having a typo silently resolved to a possibly-wrong stock.
        facts = ambiguous_stock_validation_facts(str(query), (resolution.suggestion,))
        return {
            "ambiguous": True,
            "validation_code": AMBIGUOUS_STOCK_VALIDATION_CODE,
            "validation_facts": facts,
            "matches": facts["stock_options"],
        }

    stock = resolution.stock
    if stock is None:
        return None
    return {
        "symbol":       stock.symbol,
        "display_name": stock.display_name,
        "exchange":     stock.exchange,
        "sector":       stock.sector,
        # How the query matched. Callers that re-feed free-form chat into the
        # resolver (e.g. the chat salvage net) use this to adopt only STRONG
        # matches and ignore weak short-prefix hits like "it"->ITC / "go"->GODREJCP.
        "match_kind":   resolution.match_kind,
    }
