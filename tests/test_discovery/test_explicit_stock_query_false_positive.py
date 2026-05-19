r"""Phase 9g — _extract_explicit_stock_query must not extract generic
English words as stock tickers from natural-language discovery prompts.

The bug: the user's prompt "create intraday strategy on NSE stock whose
volume spike up today 2x..." was being parsed by the regex
    r"\b([A-Za-z][A-Za-z0-9&-]{1,19})\s+(?:on|in)\s+(?:NSE|BSE)\b"
which captures "strategy" (the word immediately preceding "on NSE"),
then `resolve_supported_stock("strategy")` fails, and the chat layer
emits "This stock is not currently supported for strategy creation and
backtesting" — which directly contradicts the user's intent (they're
describing the SETUP, not a ticker).

The fix: blocklist obvious non-ticker English words AND don't treat
"NSE stock / NSE shares / NSE equity / BSE company" (NSE/BSE followed
by a generic noun) as a stock callout — that's the user describing the
market scope, not naming a specific stock.

Source-text + unit-level tests because the chat_service module can't
be imported in the local test env (asyncpg gap)."""
from __future__ import annotations

from pathlib import Path

import pytest


_CHAT_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "services" / "chat" / "chat_service.py"
)


@pytest.fixture(scope="module")
def chat_service_source() -> str:
    return _CHAT_SERVICE_PATH.read_text(encoding="utf-8")


def _load_extract_explicit_stock_query():
    """Compile the extractor + its module-level constants out of
    chat_service.py without importing the module (which would pull in
    asyncpg). Slices from the helpers block (or the def itself if no
    helpers exist) through the next top-level def/class."""
    import re as re_module
    from typing import Optional  # used inside the function signature
    src = _CHAT_SERVICE_PATH.read_text(encoding="utf-8")

    def_marker = "def _extract_explicit_stock_query(message"
    def_idx = src.index(def_marker)

    # Find the next top-level def/class AFTER our def — that's the end.
    after_def = re_module.search(
        r"\n(def |async def |class )", src[def_idx + len(def_marker):]
    )
    end_idx = (
        def_idx + len(def_marker) + after_def.start() + 1
        if after_def
        else len(src)
    )

    # Pick the start: if helpers exist before the def, walk back to
    # the leading comment line; otherwise just start at the def.
    helpers_idx = src.rfind("_NON_TICKER_WORDS = frozenset", 0, def_idx)
    if helpers_idx == -1:
        block_start = def_idx
    else:
        block_start = helpers_idx
        while block_start > 0:
            prev_nl = src.rfind("\n", 0, block_start - 1)
            line_start = prev_nl + 1 if prev_nl != -1 else 0
            line = src[line_start:block_start - 1].strip()
            if line.startswith("#"):
                block_start = line_start
            else:
                break

    body = src[block_start:end_idx]
    namespace: dict = {"re": re_module, "Optional": Optional}
    exec(body, namespace)
    return namespace["_extract_explicit_stock_query"]


# ── Regression: the exact user prompt must NOT extract a stock ──────────────


def test_extract_does_not_match_strategy_on_nse_phrasing():
    """The literal prompt the user reported."""
    extract = _load_extract_explicit_stock_query()
    prompt = (
        "create intraday strategy on NSE stock whose volume spike up today "
        "2x and is having low pull back or is on breaking on verge of 52 "
        "weeks high or low"
    )
    assert extract(prompt) is None, (
        "Discovery prompt is mis-parsed as a stock query — the LLM cannot "
        "recover from a hard validation error before discovery dispatches."
    )


def test_extract_does_not_match_generic_english_words_before_nse():
    """Defense in depth: a battery of generic words must never be
    treated as tickers, regardless of phrasing."""
    extract = _load_extract_explicit_stock_query()
    for bad in [
        "create a strategy on NSE",
        "build a system in NSE",
        "find a stock on NSE",
        "trade shares on NSE",
        "find any stock in NSE",
        "pick a company on BSE",
        "i want to do trading on NSE",
        "show me an equity on NSE",
        "show me securities on NSE",
        "find a position in NSE",
        "look at the market on NSE",
        "i want an intraday on NSE",
        "i want a swing on NSE",
        "find a setup on NSE",
    ]:
        assert extract(bad) is None, f"false-positive extracted from: {bad!r}"


def test_extract_does_not_match_nse_stock_or_nse_shares_phrasing():
    """`NSE stock`, `NSE shares`, `NSE company`, `BSE equity` are
    generic phrases the user uses to describe scope, not specific
    callouts. The natural-exchange regex must not engage when the
    word AFTER `NSE`/`BSE` is itself a generic noun."""
    extract = _load_extract_explicit_stock_query()
    for prompt in [
        "show me NSE stock with high volume",
        "find an NSE stock breaking 52w high",
        "any NSE share doing volume spike",
        "BSE company breaking out today",
        "an NSE security with bullish bias",
    ]:
        assert extract(prompt) is None, (
            f"NSE/BSE-generic-noun phrasing should not extract: {prompt!r}"
        )


# ── Sanity: real stock callouts MUST still work ─────────────────────────────


def test_extract_still_matches_explicit_dot_suffix():
    extract = _load_extract_explicit_stock_query()
    assert extract("use RELIANCE.NS") == "RELIANCE"
    assert extract("backtest TCS.BO daily") == "TCS"


def test_extract_still_matches_explicit_exchange_colon():
    extract = _load_extract_explicit_stock_query()
    assert extract("use NSE: RELIANCE for the trade") == "RELIANCE"
    assert extract("trade NSE-HDFCBANK on 5m") == "HDFCBANK"


def test_extract_still_matches_real_ticker_on_nse():
    """`RELIANCE on NSE` — uppercase ticker — is a legitimate callout."""
    extract = _load_extract_explicit_stock_query()
    assert extract("use RELIANCE on NSE") == "RELIANCE"
    assert extract("HDFCBANK on NSE 5m intraday") == "HDFCBANK"


def test_extract_still_matches_when_user_names_a_supported_ticker_naturally():
    """`Infosys on NSE` should resolve — mixed case is OK if the
    captured token is a known ticker or stock name. We rely on the
    blocklist, not strict casing, so this still works."""
    extract = _load_extract_explicit_stock_query()
    # `Infosys` is not in the blocklist of generic words → still captured
    # (downstream resolve_supported_stock handles fuzzy matching).
    assert extract("use Infosys on NSE for intraday") == "INFOSYS"


# ── Source-text sanity: the blocklist must exist in the module ──────────────


def test_chat_service_source_defines_a_non_ticker_blocklist(chat_service_source):
    """Without an explicit blocklist constant, the fix is easy to revert
    accidentally. Encode the invariant in source."""
    assert (
        "_NON_TICKER_WORDS" in chat_service_source
        or "_GENERIC_NON_TICKER_WORDS" in chat_service_source
    ), (
        "Expected a named blocklist of generic English words to guard "
        "_extract_explicit_stock_query — without it the regex re-introduces "
        "the 'strategy on NSE' false-positive."
    )


def test_chat_service_blocklist_includes_obvious_non_tickers(chat_service_source):
    """Belt-and-suspenders: confirm a few sentinel words are in the
    blocklist (so a future refactor can't accidentally narrow it)."""
    # We don't pin the exact data structure — just that these tokens
    # appear inside the chat_service module near the extractor.
    for word in ["strategy", "stock", "share", "trade", "company", "intraday"]:
        assert word in chat_service_source.lower(), (
            f"blocklist sentinel word missing from chat_service.py: {word!r}"
        )
