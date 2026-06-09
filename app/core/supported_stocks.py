"""
Supported stock universe — derived from app.kb (single source of truth).
"""
from __future__ import annotations

import re
from typing import Any

from app.kb import kb
from app.services.chat.response_composer import build_unsupported_stock_message


def normalize_supported_stock_symbol(symbol: str | None) -> str:
    """Strip casing, exchange hints, and any non-alphanumeric character.

    Used for both set membership and user-input lookups so that "BTC/USDT",
    "BTC-USDT", "btc usdt", and the canonical "BTC_USDT" all collapse to
    the same key.
    """
    cleaned = str(symbol or "").upper().strip()
    if ":" in cleaned:
        _, cleaned = cleaned.split(":", 1)
    cleaned = re.sub(r"(\.NS|\.BO)$", "", cleaned)
    return re.sub(r"[^A-Z0-9]", "", cleaned)


def _bare_symbols() -> tuple[str, ...]:
    """Normalized lookup keys for every enabled instrument in the KB.

    Must use the same normalization as `normalize_supported_stock_symbol` so
    that user input ("BTC/USDT") matches the stored canonical ("BTC_USDT").
    """
    bares = sorted({
        normalize_supported_stock_symbol(s.symbol)
        for s in kb.stocks.values()
        if s.enabled
    })
    return tuple(bares)


def _display_list() -> str:
    """Categorized markdown listing of every enabled instrument.

    Format:
        **Indian Equities (N)**
        Name A, Name B, …

        **Crypto Pairs (M)**
        - USDT pairs: BTC / USDT, ETH / USDT, …
        - USDC pairs: BTC / USDC, ETH / USDC, …
    """
    enabled = [s for s in kb.stocks.values() if s.enabled]
    if not enabled:
        return ""

    equity = sorted(
        (s for s in enabled if s.asset_class == "equity_cash"),
        key=lambda s: s.display_name,
    )
    crypto = sorted(
        (s for s in enabled if s.asset_class == "crypto_spot"),
        key=lambda s: s.display_name,
    )

    sections: list[str] = []

    if equity:
        sections.append(
            f"**Indian Equities ({len(equity)})**\n"
            + ", ".join(s.display_name for s in equity)
        )

    if crypto:
        # Sub-group crypto by quote currency for readability.
        by_quote: dict[str, list[str]] = {}
        for s in crypto:
            by_quote.setdefault(s.quote_asset or "OTHER", []).append(s.display_name)
        lines = [f"**Crypto Pairs ({len(crypto)})**"]
        for quote in sorted(by_quote):
            lines.append(f"- {quote} pairs: " + ", ".join(by_quote[quote]))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


SUPPORTED_STOCK_SYMBOLS = _bare_symbols()
SUPPORTED_STOCK_SYMBOLS_SET = set(SUPPORTED_STOCK_SYMBOLS)
SUPPORTED_STOCKS_DISPLAY = _display_list()
UNSUPPORTED_STOCK_VALIDATION_CODE = "validation.unsupported_stock"
UNSUPPORTED_STOCK_MESSAGE = build_unsupported_stock_message(SUPPORTED_STOCKS_DISPLAY)


def unsupported_stock_validation_facts() -> dict[str, Any]:
    return {"supported_stocks_display": SUPPORTED_STOCKS_DISPLAY}


def unsupported_stock_validation() -> tuple[str, dict[str, Any]]:
    return UNSUPPORTED_STOCK_VALIDATION_CODE, unsupported_stock_validation_facts()


def is_supported_stock_symbol(symbol: str | None) -> bool:
    return normalize_supported_stock_symbol(symbol) in SUPPORTED_STOCK_SYMBOLS_SET


def reload_supported_stocks() -> None:
    """Re-read the universe — call after kb.reload()."""
    global SUPPORTED_STOCK_SYMBOLS, SUPPORTED_STOCK_SYMBOLS_SET
    global SUPPORTED_STOCKS_DISPLAY, UNSUPPORTED_STOCK_MESSAGE
    SUPPORTED_STOCK_SYMBOLS = _bare_symbols()
    SUPPORTED_STOCK_SYMBOLS_SET = set(SUPPORTED_STOCK_SYMBOLS)
    SUPPORTED_STOCKS_DISPLAY = _display_list()
    UNSUPPORTED_STOCK_MESSAGE = build_unsupported_stock_message(SUPPORTED_STOCKS_DISPLAY)
