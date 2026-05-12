from __future__ import annotations

from app.services.execution.market_data_service import _upstox_instrument_key


def test_upstox_instrument_key_uses_known_nse_instrument_keys_for_added_stocks() -> None:
    assert _upstox_instrument_key("NHPC.NS") == "NSE_EQ|INE848E01016"
    assert _upstox_instrument_key("SUZLON.NS") == "NSE_EQ|INE040H01021"
    assert _upstox_instrument_key("GMRAIRPORT.NS") == "NSE_EQ|INE776C01039"
    assert _upstox_instrument_key("IDEA.NS") == "NSE_EQ|INE669E01016"


def test_upstox_instrument_key_keeps_existing_fallback_for_other_symbols() -> None:
    assert _upstox_instrument_key("RELIANCE.NS") == "NSE_EQ|RELIANCE"

