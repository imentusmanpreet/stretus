"""A vague "stop loss" mention must default to a PERCENT stop, not a fabricated
ATR one. Covers `_semantic_sl_to_loader_spec`: it returns None when the semantic
stop carries no recognisable basis (no structure, no ATR), so the caller falls
back to the plain percent `stop_loss_pct`. Explicit ATR / structural mentions
still map through unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.planner.constraint_compiler import _semantic_sl_to_loader_spec


def _sl(*, padding=None, anchor="", description="", type=""):  # noqa: A002 - mirror attr names
    return SimpleNamespace(padding=padding, anchor=anchor, description=description, type=type)


def test_vague_stop_returns_none_so_percent_is_used():
    # "Initial stop loss" with no basis → no typed spec → percent fallback.
    assert _semantic_sl_to_loader_spec(_sl(description="stop loss")) is None
    assert _semantic_sl_to_loader_spec(_sl()) is None


def test_explicit_atr_mention_still_maps_to_atr():
    spec = _semantic_sl_to_loader_spec(_sl(description="use an ATR stop"))
    assert spec == {"type": "atr", "multiplier": 1.5, "window": 14}


def test_atr_padding_still_maps_to_atr():
    padding = SimpleNamespace(method="atr", atr_multiple=2.0, percent=None)
    spec = _semantic_sl_to_loader_spec(_sl(padding=padding))
    assert spec == {"type": "atr", "multiplier": 2.0, "window": 14}


def test_structural_anchor_still_maps_to_structural():
    spec = _semantic_sl_to_loader_spec(_sl(anchor="opening_range_low"))
    assert spec["type"] == "structural"
    assert spec["anchor"] == "opening_range_low"
    assert spec["opening_bars"] == 3
