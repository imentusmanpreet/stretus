"""
Supported stock universe — derived from app.kb (single source of truth).
"""
from __future__ import annotations

import re
from typing import Any

from app.kb import kb
from app.services.chat.response_composer import build_unsupported_stock_message


def _bare_symbols() -> tuple[str, ...]:
    """Bare ticker (no .NS / .BO suffix), uppercased, sorted."""
    bares = sorted({s.symbol.split(".", 1)[0].upper() for s in kb.stocks.values() if s.enabled})
    return tuple(bares)


def _display_list() -> str:
    enabled = sorted(kb.stocks.values(), key=lambda s: s.display_name)
    if len(enabled) <= 1:
        return enabled[0].display_name if enabled else ""
    return ", ".join(s.display_name for s in enabled[:-1]) + f", and {enabled[-1].display_name}"


SUPPORTED_STOCK_SYMBOLS = _bare_symbols()
SUPPORTED_STOCK_SYMBOLS_SET = set(SUPPORTED_STOCK_SYMBOLS)
SUPPORTED_STOCKS_DISPLAY = _display_list()
UNSUPPORTED_STOCK_VALIDATION_CODE = "validation.unsupported_stock"
UNSUPPORTED_STOCK_MESSAGE = build_unsupported_stock_message(SUPPORTED_STOCKS_DISPLAY)


def unsupported_stock_validation_facts() -> dict[str, Any]:
    return {"supported_stocks_display": SUPPORTED_STOCKS_DISPLAY}


def unsupported_stock_validation() -> tuple[str, dict[str, Any]]:
    return UNSUPPORTED_STOCK_VALIDATION_CODE, unsupported_stock_validation_facts()


def normalize_supported_stock_symbol(symbol: str | None) -> str:
    cleaned = str(symbol or "").upper().strip()
    if ":" in cleaned:
        _, cleaned = cleaned.split(":", 1)
    cleaned = re.sub(r"(\.NS|\.BO)$", "", cleaned)
    return re.sub(r"[^A-Z0-9]", "", cleaned)


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
