from __future__ import annotations

from datetime import datetime, timezone
import importlib
import sys
from types import SimpleNamespace
import types

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from app.core.errors import AppError, install_exception_handlers, normalize_exception
from app.services.execution.risk_execution_config_service import RiskExecutionConfigSnapshot

if "ta" not in sys.modules:
    sys.modules["ta"] = types.ModuleType("ta")

chat_route = importlib.import_module("app.api.v1.routes.chat")


def _build_test_app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/missing")
    async def missing() -> None:
        raise HTTPException(status_code=404, detail="Session 'abc' not found.")

    @app.get("/rate-limit")
    async def rate_limit() -> None:
        raise AppError(
            429,
            (
                "Rate limit exceeded. You have reached your API usage limit. "
                "Please retry after some time."
            ),
        )

    @app.get("/validation")
    async def validation(limit: int) -> dict[str, int]:
        return {"limit": limit}

    return app


def test_http_exception_returns_standard_error_envelope() -> None:
    client = TestClient(_build_test_app())

    response = client.get("/missing")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": 404,
            "message": "Session 'abc' not found.",
        }
    }


def test_app_error_returns_requested_rate_limit_envelope() -> None:
    client = TestClient(_build_test_app())

    response = client.get("/rate-limit")

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "code": 429,
            "message": (
                "Rate limit exceeded. You have reached your API usage limit. "
                "Please retry after some time."
            ),
        }
    }


def test_validation_errors_also_return_error_code_and_message() -> None:
    client = TestClient(_build_test_app())

    response = client.get("/validation", params={"limit": "oops"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == 422
    assert payload["error"]["message"].startswith("Validation failed.")


def test_history_message_payload_exposes_structured_error() -> None:
    now = datetime.now(timezone.utc)
    message = SimpleNamespace(
        id="msg-1",
        role=SimpleNamespace(value="assistant"),
        strategy_draft=None,
        strategy_json={
            "error": {
                "code": 429,
                "message": (
                    "Rate limit exceeded. You have reached your API usage limit. "
                    "Please retry after some time."
                ),
            }
        },
        content="Rate limit exceeded. You have reached your API usage limit. Please retry after some time.",
        status=SimpleNamespace(value="failed"),
        error_message="Rate limit exceeded. You have reached your API usage limit. Please retry after some time.",
        created_at=now,
        updated_at=now,
    )

    payload = chat_route._history_message_payload(message)

    assert payload["error"] == {
        "code": 429,
        "message": (
            "Rate limit exceeded. You have reached your API usage limit. "
            "Please retry after some time."
        ),
    }
    assert payload["error_message"] == (
        "Rate limit exceeded. You have reached your API usage limit. "
        "Please retry after some time."
    )
    assert "strategy_json" not in payload


def test_permission_error_returns_specific_strategy_file_message() -> None:
    app_error = normalize_exception(PermissionError("[Errno 13] Permission denied"))

    assert app_error.status_code == 500
    assert app_error.message == (
        "The server cannot write strategy files right now. "
        "Please fix the strategies folder permissions and retry."
    )


def test_public_strategy_draft_hides_risk_and_execution_for_any_mode() -> None:
    payload = chat_route._public_strategy_draft(
        {
            "mode": "awaiting_confirmation",
            "symbol": "TCS.NS",
            "risk_and_execution": {"max_trades": "2 trades per day"},
        }
    )

    assert payload == {
        "mode": "awaiting_confirmation",
        "symbol": "TCS.NS",
    }


def test_history_message_payload_hides_strategy_draft_after_backtest_complete() -> None:
    now = datetime.now(timezone.utc)
    message = SimpleNamespace(
        id="msg-2",
        role=SimpleNamespace(value="assistant"),
        strategy_draft={
            "mode": "backtest_complete",
            "symbol": "HDFCBANK.NS",
            "timeframe": "1m",
            "signal_plan": {"entry": ["rsi_below_50"]},
        },
        strategy_json={"backtest_result": {"metrics": {"total_return_pct": 1.2}}},
        content="Backtest complete.",
        status=SimpleNamespace(value="completed"),
        error_message=None,
        created_at=now,
        updated_at=now,
    )

    payload = chat_route._history_message_payload(message)

    assert payload["state"] == "backtest_complete"
    assert payload["strategy_draft"] is None
    assert payload["backtest_result"]["metrics"]["total_return_pct"] == 1.2


@pytest.mark.asyncio
async def test_history_message_payload_hydrates_runtime_risk_execution_from_db(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    message = SimpleNamespace(
        id="msg-2",
        role=SimpleNamespace(value="assistant"),
        strategy_draft=None,
        strategy_json={
            "context": {
                "strategy_object": {
                    "risk_and_execution": {
                        "per_trade_risk": "1.0% of capital per trade",
                        "trading_window": "old-window",
                    }
                }
            }
        },
        content="Strategy ready.",
        status=SimpleNamespace(value="completed"),
        error_message=None,
        created_at=now,
        updated_at=now,
        strategy_id="strategy-1",
    )

    async def _fake_resolve_active_risk_execution_config(db, *, session_id=None, strategy_id=None):
        assert session_id == "session-1"
        assert strategy_id == "strategy-1"
        return RiskExecutionConfigSnapshot(
            config_scope="strategy",
            scope_id="strategy-1",
            session_id="session-1",
            strategy_id="strategy-1",
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

    monkeypatch.setattr(
        chat_route,
        "resolve_active_risk_execution_config",
        _fake_resolve_active_risk_execution_config,
    )

    payload = await chat_route._history_message_payload_with_runtime_risk_execution(
        db=None,
        session_id="session-1",
        message=message,
        snapshot_cache={},
    )

    risk = payload["strategy_json"]["context"]["strategy_object"]["risk_and_execution"]
    # Assembled values win; runtime config only backfills missing keys.
    assert risk["per_trade_risk"] == "1.0% of capital per trade"
    assert risk["trading_window"] == "old-window"
    assert risk["stop_loss_pct"] == 2.0
    assert risk["take_profit_pct"] == 5.0
    assert risk["max_trades"] == 2.0
