"""Unit tests for backtest DB summary helpers (no external deps)."""

from app.services.backtest.result_store import summarize_backtest_for_db


def test_summarize_backtest_for_db_strips_trade_list() -> None:
    raw = {
        "backtest_ref_id": "ref-1",
        "metrics": {
            "total_trades": 2,
            "backtest_trades": [{"x": 1}, {"x": 2}],
        },
        "pass": False,
    }
    out = summarize_backtest_for_db(raw)
    assert isinstance(out, dict)
    assert "backtest_trades" not in out.get("metrics", {})
    assert out["metrics"]["total_trades"] == 2
    # Original dict unchanged
    assert len(raw["metrics"]["backtest_trades"]) == 2


def test_summarize_backtest_for_db_backfills_trade_activity_aliases() -> None:
    raw = {
        "backtest_ref_id": "ref-1",
        "metrics": {
            "total_trades": 2,
            "backtest_trades": [
                {"outcome_pct": 2.0, "holding_duration_days": 1.0},
                {"outcome_pct": -1.0, "holding_duration_days": 3.0},
            ],
        },
        "pass": False,
    }

    out = summarize_backtest_for_db(raw)
    metrics = out["metrics"]

    assert "backtest_trades" not in metrics
    assert metrics["average_outcome_per_trade"] == 0.5
    assert metrics["average_holding_duration"] == 2.0
    assert "expectancy_pct" not in metrics
    assert "average_outcome_per_trade_pct" not in metrics
    assert "avg_trade_duration_days" not in metrics
    assert "average_holding_duration_days" not in metrics


def test_summarize_backtest_for_db_uses_trade_dates_for_duration_precision() -> None:
    raw = {
        "backtest_ref_id": "ref-1",
        "metrics": {
            "total_trades": 1,
            "average_holding_duration": 0.0,
            "backtest_trades": [
                {
                    "entry_date": "2026-01-01T03:45:00Z",
                    "exit_date": "2026-01-01T04:00:00Z",
                    "holding_duration_days": 0.01,
                },
            ],
        },
        "pass": False,
    }

    out = summarize_backtest_for_db(raw)
    metrics = out["metrics"]

    assert metrics["average_holding_duration"] == 0.0104
    assert "avg_trade_duration_days" not in metrics
