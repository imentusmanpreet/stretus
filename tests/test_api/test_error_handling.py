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


def test_lean_strategy_view_keeps_meaningful_drops_internal() -> None:
    view = chat_route._lean_strategy_view(
        {
            "mode": "awaiting_confirmation",
            "symbol": "TCS.NS",
            "timeframe": "5m",
            "entry_condition": "CLOSE > EMA(20)",
            # percent stop → already conveyed by stop_loss_pct, must NOT be duplicated
            "stop_loss_spec": {"type": "percent", "pct": 1.5},
            "take_profit_pct": 0.0,
            "trailing_take_profit_spec": {"type": "percent", "distance_pct": 0.5, "activate_after_pct": 1.0},
            "risk_execution_config": {"max_trades": 2, "trading_window": "24/7", "stop_loss_pct": 1.5},
            # internal / duplicated noise that must be dropped:
            "inputs_snapshot": {"details": {"x": 1}, "summary_rows": [1, 2]},
            "agent_decision": {"tool": "plan_strategy_signals"},
            "signal_plan": {"entry_condition": "CLOSE > EMA(20)"},
        }
    )

    # Top-level is only the focused objects; strategy holds LOGIC only (no risk).
    assert set(view) <= {"strategy", "risk_execution_config", "gates", "review"}
    assert "risk" not in view["strategy"]
    assert view["strategy"]["symbol"] == "TCS.NS"
    assert view["strategy"]["entry_condition"] == "CLOSE > EMA(20)"
    # ALL risk is merged into risk_execution_config.
    rec = view["risk_execution_config"]
    assert rec["trailing_take_profit"]["distance_pct"] == 0.5
    assert rec["trading_window"] == "24/7"
    # A percent stop is NOT duplicated as a typed spec (stop_loss_pct already has it).
    assert "stop_loss" not in rec
    assert rec["stop_loss_pct"] == 1.5
    # Internal / duplicated fields are gone everywhere.
    import json as _json
    blob = _json.dumps(view)
    for noise in ("inputs_snapshot", "agent_decision", "signal_plan", "mode"):
        assert noise not in blob


def test_lean_strategy_view_keeps_typed_atr_stop_in_rec() -> None:
    """A non-percent (ATR/structural) stop IS surfaced as a typed `stop_loss` inside
    risk_execution_config, because it carries info beyond the flat stop_loss_pct."""
    view = chat_route._lean_strategy_view(
        {
            "symbol": "BTC_USDT",
            "stop_loss_spec": {"type": "atr", "window": 14, "multiplier": 1.5},
            "risk_execution_config": {"stop_loss_pct": 2.0},
        }
    )
    rec = view["risk_execution_config"]
    assert rec["stop_loss"] == {"type": "atr", "window": 14, "multiplier": 1.5}
    assert rec["stop_loss_pct"] == 2.0  # fallback percent still present


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
    # The raw strategy_draft blob is gone; the meaningful strategy is grouped under
    # a single `strategy_json` object, with backtest_result lifted to top level.
    assert "strategy_draft" not in payload
    assert payload["strategy_json"]["strategy"]["symbol"] == "HDFCBANK.NS"
    assert payload["backtest_result"]["metrics"]["total_return_pct"] == 1.2


@pytest.mark.asyncio
async def test_history_message_payload_hydrates_runtime_risk_execution_from_db(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    message = SimpleNamespace(
        id="msg-2",
        role=SimpleNamespace(value="assistant"),
        # Lean strategy_json is built from the draft; its risk_execution_config
        # carries a user value (preserved) and is missing others (DB backfills).
        strategy_draft={
            "symbol": "INFY.NS",
            "risk_execution_config": {
                "per_trade_risk": 1.0,
                "trading_window": "old-window",
                "rms_sources": {"per_trade_risk": "user"},
            },
        },
        strategy_json=None,
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

    rec = payload["strategy_json"]["risk_execution_config"]
    # Draft (user/assembled) values win; runtime DB config only backfills missing keys.
    assert rec["per_trade_risk"] == 1.0
    assert rec["trading_window"] == "old-window"
    assert rec["stop_loss_pct"] == 2.0
    assert rec["take_profit_pct"] == 5.0
    assert rec["max_trades"] == 2
    assert rec["rms_sources"]["per_trade_risk"] == "user"  # provenance preserved
