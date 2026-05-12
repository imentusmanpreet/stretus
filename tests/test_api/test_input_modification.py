from __future__ import annotations

from app.services.chat.chat_service import (
    _detect_input_modification_request,
    _extract_input_modification_fields,
    _is_input_modification_selection_only,
)


def test_detect_input_modification_request_for_generic_edit_intent() -> None:
    assert _detect_input_modification_request("I want to modify inputs")
    assert _detect_input_modification_request("Can I update the strategy details?")


def test_extract_input_modification_fields_supports_core_aliases() -> None:
    assert _extract_input_modification_fields("Change timeframe and market view") == [
        "timeframe",
        "sentiment",
    ]
    assert _extract_input_modification_fields("Edit stock, trade type, and trading goal") == [
        "symbol",
        "objective",
        "goal",
    ]


def test_input_modification_selection_only_distinguishes_field_from_new_value() -> None:
    assert _is_input_modification_selection_only("change timeframe", ["timeframe"])
    assert not _is_input_modification_selection_only("change timeframe to 5m", ["timeframe"])

