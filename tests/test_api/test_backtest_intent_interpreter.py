from __future__ import annotations

from app.services.ai.user_input_interpreter import (
    _clean_backtest_symbols,
    _normalise_intent,
)


def test_normalise_run_backtest_intent():
    assert _normalise_intent("run_backtest") == "run_backtest"
    assert _normalise_intent("backtest") == "run_backtest"
    assert _normalise_intent("execute_backtest") == "run_backtest"


def test_normalise_confirmation_stays_confirmation():
    assert _normalise_intent("confirmation") == "confirmation"
    assert _normalise_intent("confirm") == "confirmation"


def test_mocked_llm_run_backtest_payload_fields():
    """Simulate post-LLM normalization for an explicit backtest request."""
    payload = {
        "intent": "run_backtest",
        "backtest_from_utc": "2024-01-01T00:00:00Z",
        "backtest_to_utc": "2024-02-01T23:59:59Z",
        "is_confirmation": True,
    }
    assert _normalise_intent(payload["intent"]) == "run_backtest"
    assert payload["backtest_from_utc"].startswith("2024-01-01")
    assert _normalise_intent("confirmation") != "run_backtest"


def test_mocked_llm_generic_confirmation_not_run_backtest():
    payload = {"intent": "confirmation", "is_confirmation": True}
    assert _normalise_intent(payload["intent"]) == "confirmation"
    assert payload.get("backtest_from_utc") is None


# ── Multi-asset backtest symbol extraction (legacy interpreter path) ──────────

def test_clean_backtest_symbols_from_json_array():
    assert _clean_backtest_symbols(["ETH_USDC", "BTC_USDC"]) == ["ETH_USDC", "BTC_USDC"]


def test_clean_backtest_symbols_from_comma_string():
    # The model sometimes returns a single delimited string instead of an array.
    assert _clean_backtest_symbols("ETH_USDC, BTC_USDC") == ["ETH_USDC", "BTC_USDC"]
    assert _clean_backtest_symbols("eth_usdc  btc_usdc") == ["ETH_USDC", "BTC_USDC"]


def test_clean_backtest_symbols_dedupes_and_uppercases():
    assert _clean_backtest_symbols(["btc_usdc", "BTC_USDC", " eth_usdc "]) == [
        "BTC_USDC",
        "ETH_USDC",
    ]


def test_clean_backtest_symbols_empty_and_none_return_none():
    assert _clean_backtest_symbols(None) is None
    assert _clean_backtest_symbols("") is None
    assert _clean_backtest_symbols([]) is None
    assert _clean_backtest_symbols([" ", ""]) is None
    assert _clean_backtest_symbols(123) is None
