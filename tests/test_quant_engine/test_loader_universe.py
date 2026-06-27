"""Phase B — loader tolerates a top-level `universe:` block (§5) and exposes it.

The strategy body is the symbol-agnostic template; the `universe:` sibling rides alongside.
The loader must load the strategy cleanly (ignoring `universe:`) and `extract_universe_block`
must return it so the orchestrator can detect dynamic mode. Static YAML is unaffected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from engine.loader import extract_universe_block, load_strategy_from_content

_STRATEGY = {
    "name": "t", "symbol": "TCS", "market": "indian_stocks", "timeframe": "5m",
    "objective": "intraday", "direction": "long_only",
    "entry": {"condition": "CLOSE > EMA(20)"},
    "risk_management": {"stop_loss_percent": 2, "take_profit_percent": 4},
    "entry_evaluation_mode": "formula", "exit_evaluation_mode": "formula",
}
_UNIVERSE = {"source": {"kind": "index", "name": "NIFTY500"},
             "rank": {"by": "rvol"}, "take": 10}


def test_loader_ignores_universe_sibling():
    y = yaml.safe_dump({"strategy": _STRATEGY, "universe": _UNIVERSE})
    cfg = load_strategy_from_content(y)
    assert cfg.symbol == "TCS"


def test_extract_universe_block_returns_block_when_present():
    y = yaml.safe_dump({"strategy": _STRATEGY, "universe": _UNIVERSE})
    block = extract_universe_block(y)
    assert block is not None
    assert block["source"]["name"] == "NIFTY500"


def test_extract_universe_block_none_for_static_yaml():
    y = yaml.safe_dump({"strategy": _STRATEGY})
    assert extract_universe_block(y) is None


def test_extract_universe_block_tolerates_malformed():
    assert extract_universe_block("not: [valid: yaml") is None
    assert extract_universe_block("") is None
