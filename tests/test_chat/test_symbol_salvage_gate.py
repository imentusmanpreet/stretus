"""
Regression tests for the chat "symbol salvage" gate.

Bug: a plain chat message (e.g. the greeting "hi") was being treated as a stock
query and produced an asset picker ("Found multiple matches for 'hi': hal /
hindalco / ..."). Root cause was the salvage net in chat_service re-feeding the
raw message into the resolver and adopting it on ANY hit — including a weak
ambiguous prefix ("hi" -> Hindustan*) or a loose single-prefix ("it" -> ITC,
"go" -> GODREJCP, "can" -> CANBK).

Robust fix (defense-in-depth, resolver matching behaviour unchanged):
  - the resolver now reports HOW it matched via `match_kind`;
  - the salvage net adopts a symbol only on a STRONG, unambiguous match
    (exact symbol/name or curated alias), and short-circuits greetings outright.

These tests pin all three pieces so the bug class cannot silently return.
"""
from __future__ import annotations

import pytest

from app.kb import kb
from app.kb.compat import resolve_supported_stock
from app.services.chat.chat_service import _is_greeting_message


# ── 1. Resolver exposes match_kind, without changing what it resolves ──────────

def test_match_kind_exact_for_symbol_and_name():
    assert kb.resolve_stock_query("TCS").match_kind == "exact"
    assert kb.resolve_stock_query("HDFCBANK.NS").match_kind == "exact"
    assert kb.resolve_stock_query("Reliance").match_kind == "exact"


def test_match_kind_alias():
    res = kb.resolve_stock_query("hul")
    assert res.stock is not None and res.stock.symbol == "HINDUNILVR.NS"
    assert res.match_kind == "alias"


def test_match_kind_prefix_is_weak_single_hit():
    # "go" is a loose prefix of exactly one symbol (GODREJCP) — the classic
    # false-positive shape. It must resolve, but be tagged the weak "prefix" tier.
    res = kb.resolve_stock_query("go")
    assert res.stock is not None
    assert res.match_kind == "prefix"


def test_match_kind_ambiguous_for_greeting_prefix():
    res = kb.resolve_stock_query("hi")
    assert res.is_ambiguous is True
    assert res.match_kind == "ambiguous"


def test_match_kind_empty_when_unknown():
    assert kb.resolve_stock_query("zzzzzzzz").match_kind == ""


# ── 2. compat forwards match_kind on strong resolutions ───────────────────────

@pytest.mark.asyncio
async def test_compat_includes_match_kind_for_strong_match():
    m = await resolve_supported_stock("tcs")
    assert m and m["symbol"] == "TCS.NS"
    assert m["match_kind"] == "exact"

    alias = await resolve_supported_stock("hul")
    assert alias and alias["symbol"] == "HINDUNILVR.NS"
    assert alias["match_kind"] == "alias"


@pytest.mark.asyncio
async def test_compat_weak_prefix_is_tagged_prefix_not_strong():
    m = await resolve_supported_stock("go")
    assert m and m.get("symbol")          # still resolves
    assert m["match_kind"] == "prefix"    # but flagged weak, so the gate rejects it


@pytest.mark.asyncio
async def test_compat_ambiguous_has_no_symbol():
    m = await resolve_supported_stock("hi")
    assert m and m.get("ambiguous") is True
    assert not m.get("symbol")


# ── 3. The salvage-gate contract (the actual bug protection) ───────────────────

async def _would_salvage(text: str) -> bool:
    """Mirror the exact predicate used by the chat_service salvage net so the
    rule is asserted directly: greetings are skipped, and a symbol is adopted
    only on an unambiguous STRONG (exact/alias) resolution."""
    if _is_greeting_message(text):
        return False
    m = await resolve_supported_stock(text)
    return bool(m and m.get("symbol") and m.get("match_kind") in {"exact", "alias"})


@pytest.mark.parametrize("text", [
    "hi", "hello", "hey", "namaste",        # greetings (the reported bug)
    "it", "can", "am", "at", "go",          # loose single-prefix false positives
    "in", "on", "hdfc", "btc", "sbi", "tata",  # ambiguous prefixes
])
@pytest.mark.asyncio
async def test_chat_message_is_not_salvaged_as_symbol(text):
    assert await _would_salvage(text) is False, f"{text!r} should NOT be salvaged"


@pytest.mark.parametrize("text", [
    "tcs", "itc", "reliance", "wipro", "ongc",  # exact symbol/name
    "hul",                                       # curated alias
])
@pytest.mark.asyncio
async def test_genuine_symbol_is_still_salvaged(text):
    # The salvage net must still recover a real symbol the router dropped.
    assert await _would_salvage(text) is True, f"{text!r} SHOULD be salvaged"
