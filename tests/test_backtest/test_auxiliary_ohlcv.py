"""Phase 7 — auxiliary OHLCV fetch helpers (reference + HTF) and end-to-end
contract that the engine /run-sync endpoint accepts the new payload fields.

Stubs `asyncpg` in sys.modules because the app DB layer transitively imports
it; we don't actually need a database for these tests, but we need the import
graph to load. This is a workaround for a pre-existing local-env gap, NOT
something Phase 7 introduced.
"""
from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# ─── Stub asyncpg so app.db.session can be imported ───────────────────────────
# This must happen BEFORE the first `from app.services.backtest.*` import.
if "asyncpg" not in sys.modules:
    asyncpg_stub = types.ModuleType("asyncpg")
    asyncpg_stub.Connection = type("Connection", (), {})  # placeholder
    sys.modules["asyncpg"] = asyncpg_stub

from app.services.backtest.market_data import (  # noqa: E402
    StrategyMarketDataRequest,
    _normalize_reference_symbol_for_market_data,
    extract_htf_market_data_requests,
    extract_reference_market_data_request,
    fetch_auxiliary_ohlcv,
)


# ── Reference symbol normalisation ──────────────────────────────────────────


def test_reference_symbol_normalisation_strips_caret():
    assert _normalize_reference_symbol_for_market_data("^NSEI") == "NSEI"
    assert _normalize_reference_symbol_for_market_data("^nsebank") == "NSEBANK"


def test_reference_symbol_normalisation_passthrough_for_plain_ticker():
    assert _normalize_reference_symbol_for_market_data("RELIANCE") == "RELIANCE"


def test_reference_symbol_normalisation_returns_none_for_blank():
    assert _normalize_reference_symbol_for_market_data(None) is None
    assert _normalize_reference_symbol_for_market_data("") is None
    assert _normalize_reference_symbol_for_market_data("   ") is None
    assert _normalize_reference_symbol_for_market_data("^") is None


# ── extract_reference_market_data_request ────────────────────────────────────


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "strategy.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _main_request(yaml_path: Path) -> StrategyMarketDataRequest:
    return StrategyMarketDataRequest(
        yaml_path=str(yaml_path),
        raw_symbol="HDFCBANK.NS",
        symbol="HDFCBANK",
        interval="15m",
        from_utc="2026-01-01T00:00:00Z",
        to_utc="2026-02-01T00:00:00Z",
    )


def test_extract_reference_market_data_request_returns_none_when_absent(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry:
            condition: "CLOSE > 0"
          risk_management:
            stop_loss_percent: 1.0
            take_profit_percent: 2.0
    """)
    main = _main_request(yaml_path)
    assert extract_reference_market_data_request(main, str(yaml_path)) is None


def test_extract_reference_market_data_request_builds_subrequest(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          reference_symbol: "^NSEI"
          entry:
            condition: "RS(20) > 1"
          risk_management:
            stop_loss_percent: 1.0
            take_profit_percent: 2.0
    """)
    main = _main_request(yaml_path)
    sub = extract_reference_market_data_request(main, str(yaml_path))
    assert sub is not None
    assert sub.symbol == "NSEI"             # caret stripped
    assert sub.interval == main.interval    # shares main interval + window
    assert sub.from_utc == main.from_utc
    assert sub.to_utc == main.to_utc


def test_extract_reference_accepts_alias_benchmark_symbol(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          benchmark_symbol: "NIFTY"
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
    """)
    main = _main_request(yaml_path)
    sub = extract_reference_market_data_request(main, str(yaml_path))
    assert sub is not None
    assert sub.symbol == "NIFTY"


# ── extract_htf_market_data_requests ─────────────────────────────────────────


def test_extract_htf_returns_empty_when_block_absent(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
    """)
    main = _main_request(yaml_path)
    assert extract_htf_market_data_requests(main, str(yaml_path)) == []


def test_extract_htf_builds_one_per_distinct_timeframe(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
          htf:
            - timeframe: "1h"
              condition: "CLOSE > EMA(20)"
            - timeframe: "1d"
              condition: "CLOSE > EMA(50)"
    """)
    main = _main_request(yaml_path)
    subs = extract_htf_market_data_requests(main, str(yaml_path))
    assert [s.interval for s in subs] == ["1h", "1d"]
    assert all(s.symbol == main.symbol for s in subs)
    assert all(s.from_utc == main.from_utc for s in subs)


def test_extract_htf_skips_when_htf_matches_main_timeframe(tmp_path):
    """An HTF that's the same as the main TF would just refetch the main
    series; skip silently."""
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
          htf:
            - timeframe: "15m"
              condition: "CLOSE > 0"
            - timeframe: "1d"
              condition: "CLOSE > 0"
    """)
    main = _main_request(yaml_path)
    subs = extract_htf_market_data_requests(main, str(yaml_path))
    assert [s.interval for s in subs] == ["1d"]


def test_extract_htf_rejects_non_list(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
          htf: "not a list"
    """)
    main = _main_request(yaml_path)
    with pytest.raises(ValueError, match="must be a list"):
        extract_htf_market_data_requests(main, str(yaml_path))


def test_extract_htf_rejects_missing_timeframe(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
          htf:
            - condition: "CLOSE > 0"
    """)
    main = _main_request(yaml_path)
    with pytest.raises(ValueError, match="missing 'timeframe'"):
        extract_htf_market_data_requests(main, str(yaml_path))


# ── fetch_auxiliary_ohlcv (mocks fetch_ohlcv_records) ────────────────────────


@pytest.mark.asyncio
async def test_fetch_auxiliary_ohlcv_returns_none_pair_when_strategy_uses_neither(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          entry: { condition: "CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
    """)
    main = _main_request(yaml_path)
    ref, htf = await fetch_auxiliary_ohlcv(main)
    assert ref is None
    assert htf is None


@pytest.mark.asyncio
async def test_fetch_auxiliary_ohlcv_fetches_reference_only(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          reference_symbol: "^NSEI"
          entry: { condition: "REF_CLOSE > 0" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
    """)
    main = _main_request(yaml_path)

    fake_rows = [{"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]
    with patch("app.services.backtest.market_data.fetch_ohlcv_records",
               new=AsyncMock(return_value=fake_rows)) as mock_fetch:
        ref, htf = await fetch_auxiliary_ohlcv(main)
    assert ref == fake_rows
    assert htf is None
    assert mock_fetch.call_count == 1


@pytest.mark.asyncio
async def test_fetch_auxiliary_ohlcv_fetches_reference_and_multiple_htf(tmp_path):
    yaml_path = _write_yaml(tmp_path, """
        strategy:
          name: t
          symbol: HDFCBANK.NS
          market: indian_stocks
          timeframe: 15m
          reference_symbol: "^NSEI"
          entry: { condition: "RS(20) > 1" }
          risk_management: { stop_loss_percent: 1.0, take_profit_percent: 2.0 }
          htf:
            - timeframe: "1h"
              condition: "EMA(20) > EMA(50)"
            - timeframe: "1d"
              condition: "CLOSE > EMA(50)"
    """)
    main = _main_request(yaml_path)

    # Different return per call lets us assert routing of results into ref vs htf.
    call_seq = [
        [{"id": "ref"}],
        [{"id": "1h"}],
        [{"id": "1d"}],
    ]
    mock_fetch = AsyncMock(side_effect=call_seq)
    with patch("app.services.backtest.market_data.fetch_ohlcv_records", new=mock_fetch):
        ref, htf = await fetch_auxiliary_ohlcv(main)
    assert ref == [{"id": "ref"}]
    assert htf == {"1h": [{"id": "1h"}], "1d": [{"id": "1d"}]}
    assert mock_fetch.call_count == 3


# ── End-to-end contract: engine endpoints accept the new fields ──────────────


def test_engine_run_request_schema_accepts_aux_payloads():
    """The engine's RunRequest pydantic model must accept reference_ohlcv +
    htf_ohlcv as optional fields. This guards the wire-format contract that
    the API layer depends on."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant_engine"))
    from main import RunRequest, RunSyncRequest

    payload = {
        "backtest_id": "b1", "strategy_id": "s1",
        "yaml_content": "strategy: { name: x, symbol: T, market: indian_stocks, timeframe: 15m, entry: {condition: 'CLOSE > 0'}, risk_management: {stop_loss_percent: 1.0, take_profit_percent: 2.0} }",
        "ohlcv_data": [{"t": 0}],
        "market_data_request": {"symbol": "T", "interval": "15m", "from_utc": "2026-01-01T00:00:00Z", "to_utc": "2026-02-01T00:00:00Z"},
        "reference_ohlcv": [{"t": 0}],
        "htf_ohlcv": {"1h": [{"t": 0}], "1d": [{"t": 0}]},
    }
    req = RunRequest(**payload)
    assert req.reference_ohlcv == [{"t": 0}]
    assert set(req.htf_ohlcv.keys()) == {"1h", "1d"}

    sync_payload = {k: v for k, v in payload.items() if k not in {"backtest_id", "strategy_id"}}
    sync_payload["backtest_ref_id"] = "ref-1"
    sync_req = RunSyncRequest(**sync_payload)
    assert sync_req.reference_ohlcv == [{"t": 0}]
    assert sync_req.htf_ohlcv == {"1h": [{"t": 0}], "1d": [{"t": 0}]}


def test_engine_run_request_schema_aux_fields_default_to_none():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant_engine"))
    from main import RunRequest

    req = RunRequest(
        backtest_id="b", strategy_id="s",
        yaml_content="x", ohlcv_data=[{"t": 0}],
        market_data_request={"symbol": "T", "interval": "15m", "from_utc": "2026-01-01T00:00:00Z", "to_utc": "2026-02-01T00:00:00Z"},
    )
    assert req.reference_ohlcv is None
    assert req.htf_ohlcv is None
