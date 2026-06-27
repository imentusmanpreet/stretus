"""Dynamic-universe YAML rendering: a universe spec writes strategy-template + universe block.

Without this, write_spec_yaml would call to_engine_yaml_dict() (which raises for a no-symbol
spec). Confirms the file the backtest route reads carries the `universe:` sibling it detects.
"""
from __future__ import annotations

import os
import tempfile

import yaml

from app.strategy.spec import StrategySpec
from app.strategy.yaml_writer import write_spec_yaml


def _dynamic_spec() -> StrategySpec:
    return StrategySpec(
        name="top crypto by volume", market="crypto", timeframe="1h",
        objective="intraday", direction="long_only",
        entry_condition="CLOSE > VWAP",
        stop_loss={"type": "percent", "value": 2.0, "source": "user"},
        take_profit={"type": "percent", "value": 4.0, "source": "user"},
        universe={"source": {"kind": "crypto_all"}, "rank": {"by": "volume", "order": "desc"},
                  "take": 10, "max_positions": 5, "screen": ["CLOSE > VWAP"]},
    )


def test_dynamic_spec_writes_strategy_and_universe_blocks(monkeypatch):
    monkeypatch.setenv("STRATEGY_FOLDER", tempfile.mkdtemp())
    path = write_spec_yaml(_dynamic_spec())
    doc = yaml.safe_load(open(path))
    assert set(doc.keys()) == {"strategy", "universe"}
    # strategy body is a valid single-symbol template with a placeholder instrument
    assert doc["strategy"]["symbol"] == "UNIVERSE_MEMBER"
    assert doc["strategy"]["entry"]["condition"] == "CLOSE > VWAP"
    # universe block carries the rule the backtest route / resolver consume
    assert doc["universe"]["source"]["kind"] == "crypto_all"
    assert doc["universe"]["rank"]["by"] == "volume"
    assert doc["universe"]["take"] == 10
    assert os.path.basename(path).startswith("universe_")
