"""Multi-asset backtest: response-wrapper shape, per-asset summary mapping,
comparison-reply rendering, and single-asset back-compat.

These are pure unit tests — no DB, network or LLM. They exercise the schema
(MultiAssetBacktestResult / AssetBacktestSummary), the summary-row mapping
helper, and the reply builder.
"""
from __future__ import annotations

from app.schemas.backtest import (
    AssetBacktestSummary,
    BacktestResultPayload,
    MultiAssetBacktestResult,
)
from app.services.chat.chat_service import _summary_row_from_result
from app.services.chat.strategy_flow import (
    _build_multi_asset_reply,
    build_backtest_result_reply,
)
from app.services.strategy.builder import StrategyBuilder


def _result(symbol: str, *, net_return: float, passed: bool = True) -> dict:
    """A minimal single-asset backtest result in the existing format."""
    return {
        "backtest_ref_id": f"ref-{symbol}",
        "strategy_name": "Demo",
        "backtest_date_range": {"from": "2024-01-01", "to": "2024-12-31", "num_days": 365},
        "config": {"symbol": symbol, "interval": "1d", "from_utc": "2024-01-01",
                   "to_utc": "2024-12-31", "starting_balance": 10000.0,
                   "slippage_bps": 5.0, "commission_bps": 2.0},
        "metrics": {
            "num_days": 365,
            "starting_balance": 10000.0,
            "ending_balance": 10000.0 * (1 + net_return / 100.0),
            "net_return_pct": net_return,
            "total_return_pct": net_return,
            "annual_return": net_return * 0.9,
            "volatility_pct": 14.2,
            "max_drawdown": -8.5,
            "win_rate": 55.0,
            "total_trades": 12,
        },
        "pass": passed,
        "failure_reason": "",
    }


# ── summary-row mapping ───────────────────────────────────────────────────────

def test_summary_row_maps_the_four_metrics():
    row = _summary_row_from_result("SBIN", _result("SBIN", net_return=12.5))
    assert row.symbol == "SBIN"
    assert row.backtest_ref_id == "ref-SBIN"
    assert row.total_return_pct == 12.5          # net_return_pct
    assert row.annual_return == 12.5 * 0.9
    assert row.volatility_pct == 14.2
    assert row.max_drawdown == -8.5
    assert row.pass_ is True


def test_summary_row_symbol_falls_back_to_config():
    # symbol=None (single-asset path) → take it from the run config.
    row = _summary_row_from_result(None, _result("TCS", net_return=3.0))
    assert row.symbol == "TCS"


def test_summary_row_total_return_falls_back_to_total_return_pct():
    res = _result("X", net_return=0.0)
    del res["metrics"]["net_return_pct"]
    res["metrics"]["total_return_pct"] = 7.7
    row = _summary_row_from_result("X", res)
    assert row.total_return_pct == 7.7


# ── wrapper schema ────────────────────────────────────────────────────────────

def test_wrapper_holds_array_plus_summary_and_aliases_pass():
    results = [_result("SBIN", net_return=12.5), _result("TCS", net_return=-3.0, passed=False)]
    summary = [_summary_row_from_result(r["config"]["symbol"], r) for r in results]
    wrapper = MultiAssetBacktestResult(
        strategy_name="Demo",
        backtest_date_range=results[0]["backtest_date_range"],
        num_assets=len(results),
        results=results,
        summary=summary,
    )
    assert wrapper.num_assets == 2
    assert len(wrapper.results) == 2
    # Each element is the unchanged single-asset payload.
    assert all(isinstance(r, BacktestResultPayload) for r in wrapper.results)

    dumped = wrapper.model_dump(by_alias=True)
    assert [r["pass"] for r in dumped["results"]] == [True, False]
    assert [s["pass"] for s in dumped["summary"]] == [True, False]
    assert {s["symbol"] for s in dumped["summary"]} == {"SBIN", "TCS"}


def test_single_asset_wrapper_is_length_one():
    results = [_result("SBIN", net_return=12.5)]
    wrapper = MultiAssetBacktestResult(
        num_assets=1, results=results,
        summary=[_summary_row_from_result("SBIN", results[0])],
    )
    assert wrapper.num_assets == 1
    assert len(wrapper.results) == 1


# ── comparison reply ──────────────────────────────────────────────────────────

def test_multi_asset_reply_renders_comparison_table():
    results = [_result("SBIN", net_return=12.5), _result("TCS", net_return=-3.0, passed=False)]
    wrapper = MultiAssetBacktestResult(
        num_assets=2, results=results,
        summary=[_summary_row_from_result(r["config"]["symbol"], r) for r in results],
    )
    reply = _build_multi_asset_reply(wrapper)
    assert "SBIN" in reply and "TCS" in reply
    for header in ("Total Return", "Annualized", "Volatility", "Max Drawdown"):
        assert header in reply
    assert "✅ Pass" in reply and "⚠️ Fail" in reply


def test_multi_asset_reply_renders_failed_asset_row():
    ok = _result("SBIN", net_return=12.5)
    summary = [
        _summary_row_from_result("SBIN", ok),
        AssetBacktestSummary(symbol="BADX", backtest_ref_id="",
                             failure_reason="no market data", **{"pass": False}),
    ]
    wrapper = MultiAssetBacktestResult(num_assets=1, results=[ok], summary=summary)
    reply = _build_multi_asset_reply(wrapper)
    assert "BADX" in reply
    assert "no market data" in reply


# ── single-asset back-compat through the public reply builder ─────────────────

def test_public_reply_uses_table_only_for_multiple_assets():
    builder = StrategyBuilder()
    results = [_result("SBIN", net_return=12.5), _result("TCS", net_return=-3.0)]
    multi = MultiAssetBacktestResult(
        num_assets=2, results=results,
        summary=[_summary_row_from_result(r["config"]["symbol"], r) for r in results],
    )
    table_reply = build_backtest_result_reply(builder, results[0], multi_result=multi)
    assert "| Asset |" in table_reply

    # One asset → no comparison table; falls back to the existing single reply.
    single = MultiAssetBacktestResult(
        num_assets=1, results=[results[0]],
        summary=[_summary_row_from_result("SBIN", results[0])],
    )
    single_reply = build_backtest_result_reply(builder, results[0], multi_result=single)
    assert "| Asset |" not in single_reply

    # No wrapper at all → also the single reply (full back-compat).
    legacy_reply = build_backtest_result_reply(builder, results[0])
    assert "| Asset |" not in legacy_reply
