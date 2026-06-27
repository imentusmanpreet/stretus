"""Phase B/C glue — the backtest route's dynamic-universe detection helpers.

The route branches to the portfolio path when a strategy YAML carries a top-level
``universe:`` block. These cover the pure detection + ISO parsing; the orchestration and
engine call are covered in tests/test_universe and tests/test_quant_engine.
"""
from __future__ import annotations

import sys
import types
from datetime import timezone

import yaml

# The route imports the backtest service graph; asyncpg is stubbed in some suites.
if "asyncpg" not in sys.modules:
    try:
        import asyncpg  # noqa: F401
    except Exception:  # pragma: no cover
        sys.modules["asyncpg"] = types.ModuleType("asyncpg")

import app.api.v1.routes.backtest as bt


def test_extract_universe_block_detects_dynamic():
    y = yaml.safe_dump({
        "strategy": {"symbol": "PLACEHOLDER", "entry": {"condition": "CLOSE > 0"}},
        "universe": {"source": {"kind": "index", "name": "NIFTY500"}, "rank": {"by": "rvol"}},
    })
    block = bt._extract_universe_block(y)
    assert block is not None and block["source"]["name"] == "NIFTY500"


def test_extract_universe_block_none_for_static():
    y = yaml.safe_dump({"strategy": {"symbol": "TCS", "entry": {"condition": "CLOSE > 0"}}})
    assert bt._extract_universe_block(y) is None


def test_extract_universe_block_tolerates_malformed():
    assert bt._extract_universe_block("not: [valid") is None
    assert bt._extract_universe_block("") is None


def test_parse_iso_adds_utc_when_naive():
    assert bt._parse_iso("2024-01-01T00:00:00Z").tzinfo is not None
    assert bt._parse_iso("2024-01-01T00:00:00").tzinfo == timezone.utc
