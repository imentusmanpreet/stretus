"""
app/kb/compat.py — Compatibility helpers backed by the new KB.

These mirror the legacy app.core.signals_config_loader functions so files like
app/services/strategy/builder.py can drop the legacy import without changing
their call sites. Kept intentionally tiny — once builder.py is refactored to
use kb.* directly, this module can be deleted.
"""
from __future__ import annotations

from app.kb import kb


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
    stock = kb.lookup_stock(str(query))
    if stock is None:
        return None
    return {
        "symbol":       stock.symbol,
        "display_name": stock.display_name,
        "exchange":     stock.exchange,
        "sector":       stock.sector,
    }
