"""Regression tests: the SDL parser must never crash on advisory/enum slips the
LLM commonly makes (it used to drop the whole request into the legacy pipeline)."""
import json

from app.planner.sdl_selector import _parse_sdl_json


def _base() -> dict:
    return {
        "context": {"market": "crypto", "timeframe": "15m", "objective": "mean_reversion"},
        "universe": {"type": "static", "asset_class": "crypto_spot", "symbol": "BTCUSDT"},
        "legs": [{
            "direction": "long",
            "entry": {"trigger": {"name": "rsi_below_50", "params": {"window": 14, "threshold": 50}},
                      "filters": []},
            "exit": {"triggers": [], "filters": []},
        }],
        "risk": {"stop_loss": {"type": "percent", "value": 2.0},
                 "take_profit": {"type": "rr", "ratio": 2.0}},
    }


def test_numeric_intent_rhs_does_not_crash():
    d = _base()
    d["legs"][0]["entry"]["trigger"]["intent"] = {
        "user_span": "RSI < 50", "intended": {"lhs": "rsi", "rhs": 50, "op": "lt"}}
    sdl = _parse_sdl_json(json.dumps(d))
    assert sdl.legs[0].entry.trigger.intent.intended.rhs == "50"


def test_htf_role_filter_coerced_to_optional_filter():
    d = _base()
    d["htf_rules"] = [{"timeframe": "1h", "condition": "CLOSE > EMA(50)", "role": "filter"}]
    sdl = _parse_sdl_json(json.dumps(d))
    assert sdl.htf_rules[0].role == "optional_filter"


def test_htf_role_unknown_coerced_to_confirmation():
    d = _base()
    d["htf_rules"] = [{"timeframe": "1h", "condition": "CLOSE > EMA(50)", "role": "long"}]
    sdl = _parse_sdl_json(json.dumps(d))
    assert sdl.htf_rules[0].role == "confirmation"


def test_htf_role_gate_coerced_to_gating():
    d = _base()
    d["htf_rules"] = [{"timeframe": "1d", "condition": "CLOSE > EMA(200)", "role": "gate"}]
    sdl = _parse_sdl_json(json.dumps(d))
    assert sdl.htf_rules[0].role == "gating"
