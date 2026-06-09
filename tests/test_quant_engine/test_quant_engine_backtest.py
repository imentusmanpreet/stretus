from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from app.services.strategy.builder import StrategyBuilder
from engine.assessment import build_assessment
from engine.conditions import evaluate_condition
from engine.config import (
    BACKTEST_MARKET_DATA_FROM_UTC,
    BACKTEST_MARKET_DATA_TO_UTC,
    PASS_FAIL_THRESHOLDS,
)
from engine.metrics import build_backtest_result
from engine.runner import run_backtest
from engine.simulator import Trade, simulate_trades


def _default_run_config() -> dict:
    return {
        "starting_balance": 10000.0,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "max_holding_candles": None,
    }


def _default_market_request() -> dict:
    return {
        "symbol": "RELIANCE",
        "interval": "15m",
        "from_utc": BACKTEST_MARKET_DATA_FROM_UTC,
        "to_utc": BACKTEST_MARKET_DATA_TO_UTC,
    }


def test_run_backtest_returns_expected_result_shape(tmp_path):
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "strategy:",
                '  name: "RELIANCE 15m Strategy"',
                "  symbol: RELIANCE.NS",
                "  market: indian_stocks",
                "  timeframe: 15m",
                "  entry:",
                '    condition: "CLOSE > OPEN"',
                "  exit:",
                '    condition: "PROFIT >= TAKE_PROFIT_TARGET"',
                "  indicators: {}",
                "  risk_management:",
                "    stop_loss_percent: 1.0",
                "    take_profit_percent: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    ohlcv_data = []
    for index in range(60):
        base = 100 + index
        ohlcv_data.append(
            {
                "timestamp": (
                    pd.Timestamp("2026-01-01T03:45:00Z") + pd.Timedelta(minutes=15 * index)
                ).isoformat().replace("+00:00", "Z"),
                "open": float(base),
                "high": float(base + 2),
                "low": float(base - 0.5),
                "close": float(base + 1),
                "volume": float(1000 + (index * 10)),
            }
        )

    result = run_backtest(
        str(yaml_path),
        ohlcv_data,
        _default_run_config(),
        _default_market_request(),
        "test-backtest-ref",
    )

    assert result["backtest_ref_id"] == "test-backtest-ref"
    assert result["backtest_date_range"] == {
        "from": "2024-01-01",
        "to": "2026-03-31",
        "num_days": 821,
    }
    assert set(result.keys()) == {
        "backtest_ref_id",
        "strategy_name",
        "backtest_date_range",
        "metrics",
        "assessment",
        "monthly_performance",
        "monthly_statistics",
        "market_phase_analysis",
        "failure_reason",
        "pass",
        "config",
        "strategy_type",
        "diagnostic_summary",
    }
    assert result["metrics"]["starting_balance"] == 10000.0
    assert result["metrics"]["total_trades"] >= PASS_FAIL_THRESHOLDS["positional"]["min_trades"]
    assert "total_return_pct" in result["metrics"]
    assert "volatility_pct" in result["metrics"]
    assert "profit_factor" in result["metrics"]
    assert "worst_drawdown_start_date" in result["metrics"]
    assert len(result["metrics"]["backtest_trades"]) == result["metrics"]["total_trades"]
    assert result["metrics"]["backtest_trades"][0]["instrument"] == "RELIANCE.NS"
    assert set(result["assessment"].keys()) == {
        "overall_grade",
        "return_potential",
        "risk_profile",
        "drawdown_tolerance_required",
        "recommended_for",
        "notes",
    }
    assert result["metrics"]["win_rate"] >= PASS_FAIL_THRESHOLDS["positional"]["min_win_rate_pct"]
    assert result["pass"] is True


def test_run_backtest_rejects_data_outside_configured_window(tmp_path):
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "strategy:",
                '  name: "RELIANCE Daily Strategy"',
                "  symbol: RELIANCE.NS",
                "  market: indian_stocks",
                "  timeframe: 1d",
                "  entry:",
                '    condition: "CLOSE > OPEN"',
                "  exit:",
                '    condition: "PROFIT >= TAKE_PROFIT_TARGET"',
                "  indicators: {}",
                "  risk_management:",
                "    stop_loss_percent: 1.0",
                "    take_profit_percent: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    ohlcv_data = []
    for index in range(30):
        base = 100 + index
        ohlcv_data.append(
            {
                "timestamp": (
                    pd.Timestamp("2026-04-01T00:00:00Z") + pd.Timedelta(days=index)
                ).isoformat().replace("+00:00", "Z"),
                "open": float(base),
                "high": float(base + 2),
                "low": float(base - 0.5),
                "close": float(base + 1),
                "volume": float(1000 + (index * 10)),
            }
        )

    with pytest.raises(ValueError, match="No OHLCV data is available"):
        run_backtest(
            str(yaml_path),
            ohlcv_data,
            _default_run_config(),
            {
                "symbol": "RELIANCE",
                "interval": "1d",
                "from_utc": "2026-01-01T00:00:00Z",
                "to_utc": "2026-01-30T00:00:00Z",
            },
            "out-of-window-backtest-ref",
        )


def test_build_backtest_result_sets_failure_reason_below_threshold():
    df = pd.DataFrame(
        {
            "open": list(range(100, 140)),
            "high": list(range(101, 141)),
            "low": list(range(99, 139)),
            "close": list(range(100, 140)),
            "volume": [1000] * 40,
        },
        index=pd.date_range("2026-01-01", periods=40, freq="D"),
    )
    trades = []
    for index in range(20):
        entry_ts = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        exit_ts = entry_ts + pd.Timedelta(days=1)
        entry_price = 100.0 + index
        is_win = index < 7
        exit_price = entry_price * (1.01 if is_win else 0.99)
        trades.append(
            Trade(
                entry_date=str(entry_ts),
                exit_date=str(exit_ts),
                symbol="RELIANCE.NS",
                side="LONG",
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=1.0 if is_win else -1.0,
                pnl_abs=0.01 if is_win else -0.01,
                pnl_inr=exit_price - entry_price,
                exit_reason="TAKE_PROFIT" if is_win else "STOP_LOSS",
                holding_candles=1,
            )
        )

    result = build_backtest_result(
        trades=trades,
        df=df,
        backtest_ref_id="threshold-failure-ref",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-02-09T00:00:00Z",
    )

    assert result["pass"] is False
    assert "below the required" in result["failure_reason"]
    assert result["metrics"]["total_trades"] == 20


def test_build_backtest_result_includes_average_trade_activity_aliases():
    df = pd.DataFrame(
        {
            "open": list(range(100, 110)),
            "high": list(range(101, 111)),
            "low": list(range(99, 109)),
            "close": list(range(100, 110)),
            "volume": [1000] * 10,
        },
        index=pd.date_range("2026-01-01", periods=10, freq="D"),
    )
    trades = [
        Trade(
            entry_date="2026-01-01T00:00:00Z",
            exit_date="2026-01-02T00:00:00Z",
            symbol="RELIANCE.NS",
            side="LONG",
            entry_price=100.0,
            exit_price=102.0,
            pnl_pct=2.0,
            pnl_abs=0.02,
            pnl_inr=2.0,
            exit_reason="TAKE_PROFIT",
            holding_candles=1,
        ),
        Trade(
            entry_date="2026-01-03T00:00:00Z",
            exit_date="2026-01-06T00:00:00Z",
            symbol="RELIANCE.NS",
            side="LONG",
            entry_price=100.0,
            exit_price=99.0,
            pnl_pct=-1.0,
            pnl_abs=-0.01,
            pnl_inr=-1.0,
            exit_reason="STOP_LOSS",
            holding_candles=3,
        ),
    ]

    result = build_backtest_result(
        trades=trades,
        df=df,
        backtest_ref_id="trade-activity-alias-ref",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-10T00:00:00Z",
    )

    metrics = result["metrics"]
    assert metrics["average_outcome_per_trade"] == 0.5
    assert metrics["average_holding_duration"] == 2.0
    assert "expectancy_pct" not in metrics
    assert "average_outcome_per_trade_pct" not in metrics
    assert "avg_trade_duration_days" not in metrics
    assert "average_holding_duration_days" not in metrics


def test_build_backtest_result_keeps_intraday_holding_duration_precision():
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-01T03:45:00Z", periods=2, freq="15min")
        .tz_convert("UTC")
        .tz_localize(None),
    )
    trades = [
        Trade(
            entry_date="2026-01-01T03:45:00Z",
            exit_date="2026-01-01T04:00:00Z",
            symbol="RELIANCE.NS",
            side="LONG",
            entry_price=100.0,
            exit_price=101.0,
            pnl_pct=1.0,
            pnl_abs=0.01,
            pnl_inr=1.0,
            exit_reason="EXIT_SIGNAL",
            holding_candles=1,
        )
    ]

    result = build_backtest_result(
        trades=trades,
        df=df,
        backtest_ref_id="intraday-duration-ref",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T23:59:59Z",
    )

    assert result["metrics"]["average_holding_duration"] == 0.0104
    assert "avg_trade_duration_days" not in result["metrics"]


def test_build_backtest_result_requires_minimum_trade_count():
    df = pd.DataFrame(
        {
            "open": list(range(100, 115)),
            "high": list(range(101, 116)),
            "low": list(range(99, 114)),
            "close": list(range(100, 115)),
            "volume": [1000] * 15,
        },
        index=pd.date_range("2026-01-01", periods=15, freq="D"),
    )

    trades = [
        Trade(
            entry_date="2026-01-01T00:00:00Z",
            exit_date="2026-01-02T00:00:00Z",
            symbol="RELIANCE.NS",
            side="LONG",
            entry_price=100.0,
            exit_price=101.0,
            pnl_pct=1.0,
            pnl_abs=0.01,
            pnl_inr=1.0,
            exit_reason="TAKE_PROFIT",
            holding_candles=1,
        )
        for _ in range(4)
    ]

    result = build_backtest_result(
        trades=trades,
        df=df,
        backtest_ref_id="too-few-trades",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-15T00:00:00Z",
    )

    assert result["metrics"]["total_trades"] == 4
    assert result["pass"] is True
    assert result["failure_reason"] == ""


def test_build_backtest_result_explains_no_trade_failure():
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )

    result = build_backtest_result(
        trades=[],
        df=df,
        backtest_ref_id="no-trades-ref",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-05T00:00:00Z",
    )

    assert result["pass"] is False
    assert result["metrics"]["total_trades"] == 0
    assert result["assessment"]["overall_grade"] == "D"
    assert result["assessment"]["recommended_for"] == "Not Recommended"
    assert "no trades were executed" in result["failure_reason"].lower()


def test_build_assessment_matches_expected_threshold_labels():
    assessment = build_assessment(
        {
            "total_trades": 169,
            "total_return_pct": 21.16,
            "sharpe_ratio": 1.55,
            "max_drawdown": 15.62,
            "volatility_pct": 6.61,
            "var_95_pct": -2.5,
            "recovery_time_days": 45,
            "trades_per_month": 56.33,
            "profit_factor": 1.48,
            "win_rate": 73.23,
        }
    )

    assert assessment["overall_grade"] == "B+"
    assert assessment["return_potential"] == "Strong"
    assert assessment["risk_profile"] == "Aggressive"
    assert assessment["drawdown_tolerance_required"] == "High"
    assert assessment["recommended_for"] == "Experienced Traders"
    assert assessment["notes"]


def test_formula_renderer_maps_opening_range_breakout_to_session_high():
    """The opening_range_breakout signal must render as OPENING_RANGE_HIGH(...),
    not the broken legacy 'CLOSE > HIGH'."""
    from app.planner.formulas import render_formula

    formula = render_formula("opening_range_breakout", {"opening_bars": 4})
    assert formula is not None
    assert "OPENING_RANGE_HIGH" in formula
    assert formula != "CLOSE > HIGH"


def test_builder_normalizes_legacy_opening_range_breakout_preview():
    builder = StrategyBuilder()
    builder.merge_preview(
        {
            "entry_condition": "CLOSE > HIGH AND CLOSE > EMA(20)",
            "signal_plan": {
                "entry": [
                    {
                        "name": "opening_range_breakout",
                        "params": {"opening_bars": 4},
                        "timeframe": "15m",
                        "signal_type": "TRIGGER",
                    },
                    {
                        "name": "price_above_ema",
                        "params": {"window": 20},
                        "timeframe": "15m",
                        "signal_type": "FILTER",
                    },
                ],
                "exit": [],
            },
        }
    )

    assert builder.entry_condition == "CLOSE > OPENING_RANGE_HIGH(4) AND CLOSE > EMA(20)"


def test_evaluate_condition_supports_opening_range_high_per_session():
    timestamps = []
    records = []
    base_days = [
        pd.Timestamp("2026-01-01T03:45:00Z"),
        pd.Timestamp("2026-01-02T03:45:00Z"),
    ]

    for base_day in base_days:
        day_rows = [
            (100.0, 101.0, 99.5, 100.5),
            (100.5, 102.0, 100.0, 101.5),
            (101.5, 103.0, 101.0, 102.5),
            (102.5, 104.0, 102.0, 103.5),
            (104.0, 106.0, 103.8, 105.5),
            (105.5, 107.0, 105.0, 106.5),
        ]
        for offset, (open_, high, low, close) in enumerate(day_rows):
            timestamps.append(base_day + pd.Timedelta(minutes=15 * offset))
            records.append(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                }
            )

    df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps).tz_convert("UTC").tz_localize(None))

    assert evaluate_condition("CLOSE > OPENING_RANGE_HIGH(4)", df, 3) is False
    assert evaluate_condition("CLOSE > OPENING_RANGE_HIGH(4)", df, 4) is True
    assert evaluate_condition("CLOSE > OPENING_RANGE_HIGH(4)", df, 9) is False
    assert evaluate_condition("CLOSE > OPENING_RANGE_HIGH(4)", df, 10) is True


def test_evaluate_condition_blocks_code_execution_payload():
    timestamps = pd.date_range("2026-01-01T03:45:00Z", periods=5, freq="15min")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 103.0, 104.0],
            "high": [101.0, 102.0, 104.0, 106.0, 107.0],
            "low": [99.0, 99.5, 100.5, 102.5, 103.0],
            "close": [100.0, 101.5, 103.5, 105.5, 106.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
        index=pd.DatetimeIndex(timestamps).tz_convert("UTC").tz_localize(None),
    )

    assert evaluate_condition("__import__('os').system('echo hacked')", df, 3) is False


def test_simulate_trades_honors_exit_signal():
    timestamps = pd.date_range("2026-01-01T03:45:00Z", periods=25, freq="15min")
    df = pd.DataFrame(
        {
            "open": [100 + i for i in range(25)],
            "high": [101 + i for i in range(25)],
            "low": [99 + i for i in range(25)],
            "close": [100.5 + i for i in range(25)],
            "volume": [1000.0] * 25,
        },
        index=pd.DatetimeIndex(timestamps).tz_convert("UTC").tz_localize(None),
    )

    trades, _ = simulate_trades(
        df=df,
        symbol="RELIANCE.NS",
        entry_condition="CLOSE > OPEN",
        exit_condition="PROFIT >= 1",
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        max_holding_candles=None,
    )

    assert trades
    assert trades[0].exit_reason == "EXIT_SIGNAL"


def test_simulate_trades_uses_pessimistic_same_bar_resolution():
    timestamps = pd.date_range("2026-01-01T03:45:00Z", periods=25, freq="15min")
    df = pd.DataFrame(
        {
            "open": [100.0] + [101.0] * 24,
            "high": [101.0, 103.0] + [101.0] * 23,
            "low": [99.0, 97.0] + [100.0] * 23,
            "close": [100.5, 100.0] + [100.5] * 23,
            "volume": [1000.0] * 25,
        },
        index=pd.DatetimeIndex(timestamps).tz_convert("UTC").tz_localize(None),
    )

    trades, _ = simulate_trades(
        df=df,
        symbol="RELIANCE.NS",
        entry_condition="CLOSE > OPEN",
        exit_condition="PROFIT >= 99",
        stop_loss_pct=1.0,
        take_profit_pct=1.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        max_holding_candles=None,
    )

    assert trades
    assert trades[0].exit_reason == "STOP_LOSS_AND_TAKE_PROFIT_SAME_BAR"
    assert trades[0].pnl_pct < 0


def test_simulate_trades_intraday_forces_session_end_exit_same_day():
    timestamps = pd.DatetimeIndex(
        [
            "2026-01-01T03:45:00Z",
            "2026-01-01T04:45:00Z",
            "2026-01-01T05:45:00Z",
            "2026-01-02T03:45:00Z",
            "2026-01-02T04:45:00Z",
            "2026-01-02T05:45:00Z",
        ]
    ).tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.4] * 6,
            "low": [99.6] * 6,
            "close": [100.1] * 6,
            "volume": [1000.0] * 6,
        },
        index=timestamps,
    )

    trades, _ = simulate_trades(
        df=df,
        symbol="RELIANCE.NS",
        entry_condition="CLOSE > OPEN",
        exit_condition="PROFIT >= 99",
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        max_holding_candles=99,
        objective="intraday",
    )

    assert trades
    assert all(trade.exit_reason == "SESSION_END" for trade in trades)
    assert all(
        pd.Timestamp(trade.entry_date).date() == pd.Timestamp(trade.exit_date).date()
        for trade in trades
    )


def test_simulate_trades_intraday_skips_last_bar_entry_that_would_carry_overnight():
    timestamps = pd.DatetimeIndex(
        [
            "2026-01-01T03:45:00Z",
            "2026-01-01T04:45:00Z",
            "2026-01-02T03:45:00Z",
            "2026-01-02T04:45:00Z",
        ]
    ).tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.2, 101.2, 100.2, 100.2],
            "low": [99.8, 99.8, 99.8, 99.8],
            "close": [99.9, 101.0, 99.9, 99.9],
            "volume": [1000.0] * 4,
        },
        index=timestamps,
    )

    trades, diagnostics = simulate_trades(
        df=df,
        symbol="RELIANCE.NS",
        entry_condition="CLOSE > OPEN",
        exit_condition="PROFIT >= 99",
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        max_holding_candles=99,
        objective="intraday",
    )

    assert any(diag["entry_signal"] for diag in diagnostics)
    assert trades == []


def test_trade_carries_structured_entry_and_exit_reason():
    """Each trade must explain WHY it opened and closed: the matched bar, the
    indicator values, the fill price + costs, and the exit trigger."""
    from engine.conditions import compile_condition

    timestamps = pd.date_range("2026-01-01T03:45:00Z", periods=30, freq="15min").tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame(
        {
            "open":   [100 + i for i in range(30)],
            "high":   [101 + i for i in range(30)],
            "low":    [99 + i for i in range(30)],
            "close":  [100.5 + i for i in range(30)],
            "volume": [1000.0] * 30,
        },
        index=timestamps,
    )
    # Precompute EMA_5 with pandas so the entry can match a real indicator value
    # without depending on the TA-Lib-backed indicator pipeline.
    df["EMA_5"] = df["close"].ewm(span=5, adjust=False).mean()

    condition = "CLOSE > OPEN AND EMA(5) > 0"
    trades, _ = simulate_trades(
        df=df,
        symbol="RELIANCE.NS",
        entry_condition=condition,
        exit_condition="PROFIT >= 1",
        compiled_entry=compile_condition(condition),
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        slippage_bps=5.0,
        commission_bps=2.0,
        warm_up_candles=5,
        max_holding_candles=None,
    )

    assert trades
    trade = trades[0]

    # ── Entry reason ──────────────────────────────────────────────────────────
    entry = trade.entry_reason
    assert entry is not None
    assert "Entered LONG" in entry["summary"]
    assert entry["evaluation_mode"] == "formula"
    assert entry["condition"] == condition
    signal_idx = entry["signal_bar"]["index"]
    assert signal_idx >= 5                                   # past warm-up
    assert entry["signal_bar"]["close"] is not None
    # The indicator that made the formula true is captured with its value.
    assert entry["indicators"]["EMA_5"] is not None
    # Fill happens on the bar *after* the signal, at that bar's open + costs.
    assert entry["entry_bar"]["index"] == signal_idx + 1
    assert entry["fill"]["effective_price"] == pytest.approx(trade.entry_price, rel=1e-3)
    assert entry["fill"]["raw_price"] < entry["fill"]["effective_price"]  # buy-side costs add
    assert entry["initial_stop_price"] is not None

    # ── Exit reason ───────────────────────────────────────────────────────────
    exit_detail = trade.exit_reason_detail
    assert exit_detail is not None
    assert exit_detail["code"] == trade.exit_reason
    assert exit_detail["trigger"]["type"] == "exit_signal"
    assert exit_detail["fill"]["effective_price"] == pytest.approx(trade.exit_price, rel=1e-3)
    assert exit_detail["pnl_pct"] == pytest.approx(trade.pnl_pct, rel=1e-3)
    assert "Exited LONG" in exit_detail["summary"]

    # ── Survives serialization + Pydantic validation ──────────────────────────
    from app.schemas.backtest import BacktestTrade
    from engine.metrics import _serialize_backtest_trades

    entry_ts = pd.to_datetime([t.entry_date for t in trades]).tz_localize(None)
    exit_ts = pd.to_datetime([t.exit_date for t in trades]).tz_localize(None)
    serialized = _serialize_backtest_trades(trades, entry_ts, exit_ts, df)
    model = BacktestTrade(**serialized[0])
    assert model.entry_reason.indicators["EMA_5"] is not None
    assert model.exit_reason_detail.code == model.exit_reason


def test_exit_reason_detail_describes_stop_loss_trigger():
    """A stop-loss exit must record the stop price, the breaching bar, and a
    plain-language explanation."""
    timestamps = pd.date_range("2026-01-01T03:45:00Z", periods=25, freq="15min").tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame(
        {
            "open":   [100.0] + [101.0] * 24,
            "high":   [101.0, 101.5] + [101.0] * 23,
            "low":    [99.0, 95.0] + [100.0] * 23,   # bar #1 low=95 breaches the stop
            "close":  [100.5, 100.0] + [100.5] * 23,
            "volume": [1000.0] * 25,
        },
        index=timestamps,
    )

    trades, _ = simulate_trades(
        df=df,
        symbol="RELIANCE.NS",
        entry_condition="CLOSE > OPEN",
        exit_condition="PROFIT >= 99",
        stop_loss_pct=1.0,
        take_profit_pct=50.0,        # high enough that only the stop fires
        slippage_bps=0.0,
        commission_bps=0.0,
        max_holding_candles=None,
    )

    assert trades
    trade = trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    detail = trade.exit_reason_detail
    assert detail["trigger"]["type"] == "stop_loss"
    assert detail["trigger"]["stop_price"] is not None
    assert detail["trigger"]["bar_low"] == 95.0
    assert detail["exit_bar"]["index"] == 1
    assert "stop loss" in detail["summary"].lower()
