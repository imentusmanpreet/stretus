"""Phase B/C glue — the engine /run-portfolio-sync endpoint (dynamic portfolio backtest).

Drives the real engine app via FastAPI's TestClient over synthetic member OHLCV and asserts
the portfolio result shape, member pass-through, and the empty-members guard.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

try:
    from fastapi.testclient import TestClient
    import main as engine_main  # quant_engine/main.py
    _CLIENT_OK = True
except Exception:  # pragma: no cover - missing fastapi/uvicorn in some envs
    _CLIENT_OK = False

pytestmark = pytest.mark.skipif(not _CLIENT_OK, reason="fastapi/uvicorn not importable")

_FROM = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _ohlcv(n: int = 160, *, base: float = 100.0, amp: float = 8.0) -> list[dict]:
    out = []
    for i in range(n):
        close = base + amp * math.sin(i / 6.0) + i * 0.05
        out.append({
            "timestamp": (_FROM + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
            "open": close - 0.5, "high": close + 1, "low": close - 1,
            "close": close, "volume": 100000 + (i % 7) * 1000,
        })
    return out


def _template() -> str:
    return yaml.safe_dump({"strategy": {
        "name": "dyn", "symbol": "PLACEHOLDER", "market": "indian_stocks", "timeframe": "1d",
        "objective": "positional", "direction": "long_only",
        "entry": {"condition": "CLOSE > EMA(10)"}, "exit": {"condition": "CLOSE < EMA(10)"},
        "risk_management": {"stop_loss_percent": 3, "take_profit_percent": 6},
        "entry_evaluation_mode": "formula", "exit_evaluation_mode": "formula",
    }}, sort_keys=False)


def _body() -> dict:
    return {
        "template_yaml": _template(),
        "member_ohlcv": {"AAA": _ohlcv(), "BBB": _ohlcv(base=200.0, amp=5.0)},
        "run_config": {"starting_balance": 100000.0, "objective": "positional"},
        "market_data_request": {"symbol": "PLACEHOLDER", "interval": "1d",
                                "from_utc": _FROM.isoformat().replace("+00:00", "Z"),
                                "to_utc": (_FROM + timedelta(days=200)).isoformat().replace("+00:00", "Z")},
        "starting_capital": 100000.0, "max_positions": 2, "survivorship_mode": "point_in_time",
    }


def test_run_portfolio_sync_returns_portfolio_result():
    client = TestClient(engine_main.app)
    r = client.post("/run-portfolio-sync", json=_body())
    assert r.status_code == 200
    d = r.json()
    assert d["members"] == ["AAA", "BBB"]
    assert d["survivorship_mode"] == "point_in_time"
    assert "equity_curve" in d and "metrics" in d and "per_symbol_pnl" in d
    assert d["max_positions"] == 2


def test_run_portfolio_sync_rejects_empty_members():
    client = TestClient(engine_main.app)
    body = _body()
    body["member_ohlcv"] = {}
    r = client.post("/run-portfolio-sync", json=body)
    assert r.status_code == 400
