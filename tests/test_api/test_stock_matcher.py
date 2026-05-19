from __future__ import annotations

from app.core.supported_stocks import SUPPORTED_STOCKS_DISPLAY, UNSUPPORTED_STOCK_MESSAGE, is_supported_stock_symbol
from app.services.knowledge.stock_matcher import _direct_supported_stock_match


def test_resolve_supported_stock_direct_symbol_match_for_supported_names():
    for query, expected_symbol in (
        ("TCS", "TCS"),
        ("HDFCBANK", "HDFCBANK"),
        ("HDFC Bank", "HDFCBANK"),
        ("INFY", "INFY"),
        ("ADANIENT", "ADANIENT"),
        ("RELIANCE", "RELIANCE"),
        ("NHPC", "NHPC"),
        ("SUZLON", "SUZLON"),
        ("GMRAIRPORT", "GMRAIRPORT"),
        ("IDEA", "IDEA"),
        ("Adani Enterprises", "ADANIENT"),
        ("Suzlon Energy", "SUZLON"),
        ("GMR Airports", "GMRAIRPORT"),
        ("Vodafone Idea", "IDEA"),
    ):
        result = _direct_supported_stock_match(query)

        assert result is not None
        assert result["symbol"] == expected_symbol


def test_direct_supported_stock_match_flags_ambiguous_prefixes():
    result = _direct_supported_stock_match("hdfc")

    assert result is not None
    assert result["ambiguous"] is True
    assert {match["symbol"] for match in result["matches"]} == {
        "HDFCAMC.NS",
        "HDFCBANK.NS",
        "HDFCLIFE.NS",
    }
    assert result["validation_code"] == "validation.ambiguous_stock"


def test_direct_supported_stock_match_accepts_exchange_suffix_inputs():
    for query, expected_symbol in (
        ("TCS.NS", "TCS"),
        ("INFY.NS", "INFY"),
        ("RELIANCE.BO", "RELIANCE"),
        ("NHPC.NS", "NHPC"),
        ("SUZLON.NS", "SUZLON"),
        ("GMRAIRPORT.NS", "GMRAIRPORT"),
        ("IDEA.NS", "IDEA"),
    ):
        result = _direct_supported_stock_match(query)

        assert result is not None
        assert result["symbol"] == expected_symbol


def test_supported_stock_message_matches_requested_copy():
    assert UNSUPPORTED_STOCK_MESSAGE == (
        "This stock is not currently supported for strategy creation and backtesting.\n\n"
        f"Currently supported stocks are: {SUPPORTED_STOCKS_DISPLAY}. "
        "Please select one of these to continue."
    )


def test_is_supported_stock_symbol_handles_exchange_suffixes():
    assert is_supported_stock_symbol("TCS.NS") is True
    assert is_supported_stock_symbol("RELIANCE.BO") is True
    assert is_supported_stock_symbol("NHPC.NS") is True
    assert is_supported_stock_symbol("SUZLON.NS") is True
    assert is_supported_stock_symbol("GMRAIRPORT.NS") is True
    assert is_supported_stock_symbol("IDEA.NS") is True
    assert is_supported_stock_symbol("COALINDIA.NS") is True
    assert is_supported_stock_symbol("NOTREAL.NS") is False
