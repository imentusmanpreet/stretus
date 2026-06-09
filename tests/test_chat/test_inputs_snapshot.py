from __future__ import annotations

from app.services.chat.inputs_snapshot import (
    append_inputs_snapshot,
    build_inputs_snapshot,
    build_inputs_snapshot_from_draft,
    format_inputs_summary_markdown,
)
from app.services.strategy.builder import StrategyBuilder


def test_build_inputs_snapshot_from_draft_only_populated_fields():
    draft = {
        "mode": "collect_user_input",
        "symbol": "TCS",
        "timeframe": "15m",
        "goal": None,
    }
    snapshot = build_inputs_snapshot_from_draft(draft, missing=["goal"])
    labels = [row["label"] for row in snapshot["summary_rows"]]

    assert labels == ["Stock", "Timeframe"]
    assert snapshot["missing_fields"] == ["goal"]
    assert snapshot["details"]["symbol"] == "TCS"
    assert snapshot["details"]["timeframe"] == "15m"


def test_details_contains_full_strategy_draft():
    draft = {
        "symbol": "TCS",
        "timeframe": "15m",
        "stop_loss_pct": 2.0,
        "risk_execution_config": {"stop_loss_pct": 2.0, "rms_sources": {}},
        "agent_decision": {"tool": "collect_input"},
    }
    snapshot = build_inputs_snapshot_from_draft(draft)
    details = snapshot["details"]
    assert details["symbol"] == "TCS"
    assert details["stop_loss_pct"] == 2.0
    assert details["risk_execution_config"]["stop_loss_pct"] == 2.0
    assert details["agent_decision"]["tool"] == "collect_input"
    assert "inputs_snapshot" not in details


def test_build_inputs_snapshot_from_draft_includes_backtest_window():
    draft = {
        "symbol": "IDEA",
        "timeframe": "1m",
        "entry_condition": "CLOSE > EMA(20)",
        "stop_loss_pct": 1.5,
        "backtest_window": {
            "from_utc": "2025-08-01T00:00:00Z",
            "to_utc": "2025-09-15T23:59:59Z",
        },
    }
    snapshot = build_inputs_snapshot_from_draft(draft)
    labels = [row["label"] for row in snapshot["summary_rows"]]

    assert "Backtest range" in labels
    assert "2025-08-01" in next(r["value"] for r in snapshot["summary_rows"] if r["label"] == "Backtest range")


def test_builder_wrapper_matches_draft():
    builder = StrategyBuilder()
    builder.symbol = "TCS.NS"
    builder.timeframe = "15m"
    builder.goal = "trend"

    from_draft = build_inputs_snapshot_from_draft(builder.to_draft_json())
    from_builder = build_inputs_snapshot(builder)

    assert from_draft["summary_rows"] == from_builder["summary_rows"]


def test_append_merges_draft_extras_with_builder():
    builder = StrategyBuilder()
    builder.symbol = "TCS.NS"
    builder.timeframe = "15m"
    stale_draft = {
        "symbol": "OLD",
        "backtest_window": {"from_utc": "2025-01-01T00:00:00Z", "to_utc": "2025-02-01T00:00:00Z"},
        "kb_signals_used": ["ema_above"],
    }
    out, merged = append_inputs_snapshot("Done.", builder, state="backtest_complete", draft=stale_draft)
    stock_row = next(r for r in merged["inputs_snapshot"]["summary_rows"] if r["label"] == "Stock")
    assert stock_row["value"] == "TCS.NS"
    assert "Backtest range" in [r["label"] for r in merged["inputs_snapshot"]["summary_rows"]]
    assert merged["kb_signals_used"] == ["ema_above"]
    assert "### Stored inputs" in out


def test_summary_hides_tier_default_sl_tp():
    draft = {
        "symbol": "ETH_USDT",
        "timeframe": "5m",
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
        "risk_execution_config": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
            "rms_sources": {
                "stop_loss_pct": "system_default",
                "take_profit_pct": "system_default",
            },
        },
    }
    snapshot = build_inputs_snapshot_from_draft(draft)
    labels = [row["label"] for row in snapshot["summary_rows"]]
    assert "Stop loss" not in labels
    assert "Take profit" not in labels


def test_summary_shows_user_sl_tp():
    draft = {
        "symbol": "ETH_USDT",
        "timeframe": "5m",
        "stop_loss_pct": 5.0,
        "take_profit_pct": 0.4,
        "risk_execution_config": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": 0.4,
            "rms_sources": {
                "stop_loss_pct": "user",
                "take_profit_pct": "user",
            },
        },
    }
    snapshot = build_inputs_snapshot_from_draft(draft)
    labels = [row["label"] for row in snapshot["summary_rows"]]
    assert "Stop loss" in labels
    assert "Take profit" in labels


def test_append_adds_snapshot_even_for_strategy_setup_heading():
    builder = StrategyBuilder()
    builder.symbol = "TCS.NS"
    builder.timeframe = "15m"
    text = "## Strategy Setup ù TCS\n\nSee stored inputs below."
    out, draft = append_inputs_snapshot(text, builder, state="collect_user_input", draft={})
    assert "### Stored inputs" in out
    assert draft["inputs_snapshot"]["summary_rows"]
