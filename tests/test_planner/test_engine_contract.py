"""
tests/test_planner/test_engine_contract.py — the shared engine contract (§2b).

The planner-side contract MUST stay in lock-step with the engine loader's local
fallback, so the "planner can never emit an anchor the engine rejects" guarantee
holds in BOTH the co-located and standalone deployments.
"""
from app.planner import engine_contract as ec


def test_anchors_in_sync_with_engine_loader():
    from engine import loader  # quant_engine on sys.path via tests/conftest.py
    assert set(ec.STOP_LOSS_ANCHORS) == set(loader._STOP_LOSS_ANCHORS)
    assert set(ec.STOP_LOSS_TYPES) == set(loader._STOP_LOSS_TYPES)
    assert set(ec.TRAILING_TYPES) == set(loader._TRAILING_TYPES)
    assert set(ec.TRAILING_TAKE_PROFIT_TYPES) == set(loader._TRAILING_TAKE_PROFIT_TYPES)


def test_unmappable_stops_return_needs_clarify():
    assert ec.map_stop_loss_type("vwap_deviation") == ec.NEEDS_CLARIFY
    assert ec.map_stop_loss_type("points") == ec.NEEDS_CLARIFY
    assert ec.map_stop_loss_type("nonsense") == ec.NEEDS_CLARIFY
    assert ec.map_take_profit_type("points") == ec.NEEDS_CLARIFY


def test_structural_anchors_map_to_engine_legal_ids():
    for planner_type in ("swing_low", "candle_low", "orb_low",
                         "swing_high", "candle_high", "orb_high"):
        mapped = ec.map_stop_loss_type(planner_type)
        assert isinstance(mapped, dict) and mapped["type"] == "structural"
        assert mapped["anchor"] in ec.STOP_LOSS_ANCHORS


def test_opposite_side_needs_direction():
    assert ec.map_stop_loss_type("opposite_side", "long")["anchor"] == "prev_n_bar_low"
    assert ec.map_stop_loss_type("opposite_side", "short")["anchor"] == "prev_n_bar_high"
    assert ec.map_stop_loss_type("opposite_side", None) == ec.NEEDS_CLARIFY


def test_direct_types_passthrough():
    assert ec.map_stop_loss_type("percent") == {"type": "percent"}
    assert ec.map_stop_loss_type("atr") == {"type": "atr"}
