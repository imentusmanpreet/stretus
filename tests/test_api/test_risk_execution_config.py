from __future__ import annotations

from app.services.execution.risk_execution_config_service import (
    RiskExecutionConfigSnapshot,
    build_risk_execution_response,
    compose_risk_execution_values,
)
from app.services.strategy.builder import StrategyBuilder


def test_compose_risk_execution_values_derives_risk_reward() -> None:
    values = compose_risk_execution_values(
        max_trades=2,
        daily_loss_cap=3.0,
        execution_mode="Backtest",
        per_trade_risk=2.0,
        trading_window="9:15 - 15:30",
        position_sizing="Risk based",
        risk_validation="Pass",
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        minimum_trade_value=500.0,
    )

    assert values["risk_reward"] == 2.5


def test_builder_uses_db_backed_risk_execution_defaults() -> None:
    builder = StrategyBuilder()
    builder.objective = "intraday"
    builder.experience = "intermediate"
    builder.user_input_confirmed = True
    builder.symbol = "TCS.NS"
    builder.timeframe = "15m"
    builder.sentiment = "bullish"
    builder.goal = "Steady profit"
    builder.signal_plan = {"entry": [], "exit": []}
    builder.set_risk_execution_config(
        {
            "max_trades": 4,
            "risk_reward": 3.0,
            "daily_loss_cap": 1.5,
            "execution_mode": "Paper",
            "per_trade_risk": 1.2,
            "trading_window": "10:00 AM to 2:00 PM IST",
            "position_sizing": "Fixed size",
            "risk_validation": "Validated from DB",
            "stop_loss_pct": 1.8,
            "take_profit_pct": 4.2,
            "minimum_trade_value": 7500.0,
        }
    )

    builder.apply_defaults()
    summary = builder.risk_and_execution_summary(mode="assemble_strategy")

    assert builder.stop_loss == 1.8
    assert builder.take_profit == 4.2
    assert builder.daily_loss_cap == 1.5
    assert summary == {
        "daily_loss_cap": "1.5% of capital per day",
        "position_sizing": "Fixed size",
        "execution_mode": "Paper",
        "trading_window": "10:00 AM to 2:00 PM IST",
        "risk_validation": "Validated from DB",
        "per_trade_risk": "1.2% of capital per trade",
        "risk_reward": "3:1",
        "max_trades": "4 trades per day",
    }


def test_build_risk_execution_response_returns_postgres_snapshot_values_for_api() -> None:
    snapshot = RiskExecutionConfigSnapshot(
        config_scope="global",
        scope_id="global-default",
        session_id=None,
        strategy_id=None,
        max_trades=2,
        risk_reward=2.5,
        daily_loss_cap=3.0,
        execution_mode="Backtest",
        per_trade_risk=2.0,
        trading_window="9:15 - 15:30",
        position_sizing="Risk based",
        risk_validation="system risk guardials",
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        minimum_trade_value=500.0,
    )

    assert build_risk_execution_response(snapshot) == {
        "max_trades": 2.0,
        "risk_reward": 2.5,
        "stop_loss_pct": 2.0,
        "daily_loss_cap": 3.0,
        "execution_mode": "Backtest",
        "per_trade_risk": 2.0,
        "trading_window": "9:15 - 15:30",
        "position_sizing": "Risk based",
        "risk_validation": "system risk guardials",
        "take_profit_pct": 5.0,
        "minimum_trade_value": 500.0,
    }
