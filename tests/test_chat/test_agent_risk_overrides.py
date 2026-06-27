from __future__ import annotations

from app.services.chat.chat_service import (
    _apply_agent_risk_param_overrides,
    _parse_rms_agent_value,
)
from app.services.chat.inputs_snapshot import build_inputs_snapshot_from_draft
from app.services.strategy.builder import StrategyBuilder, _extract_rms_from_text


def test_extract_rms_sl_tp_with_to_keyword():
    rms = _extract_rms_from_text("change sl to 5% and tp to 0.4%")
    assert rms["stop_loss_pct"]["value"] == 5.0
    assert rms["take_profit_pct"]["value"] == 0.4


def test_parse_rms_agent_value_json_string():
    val, source = _parse_rms_agent_value('{"value": 5, "source": "user"}')
    assert val == 5.0
    assert source == "user"


def test_apply_agent_risk_param_overrides_updates_builder_and_draft():
    builder = StrategyBuilder()
    builder.risk_execution_config = {
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
        "rms_sources": {"stop_loss_pct": "system_default", "take_profit_pct": "system_default"},
    }
    params = {
        "stop_loss": '{"value": 5, "source": "user"}',
        "take_profit": '{"value": 0.4, "source": "user"}',
        "preserve_unmentioned_fields": "true",
    }
    assert _apply_agent_risk_param_overrides(builder, params) is True
    assert builder.stop_loss == 5.0
    assert builder.take_profit == 0.4
    draft = builder.to_draft_json()
    assert draft["stop_loss_pct"] == 5.0
    assert draft["take_profit_pct"] == 0.4
    snapshot = build_inputs_snapshot_from_draft(draft)
    sl_row = next(r for r in snapshot["summary_rows"] if r["label"] == "Stop loss")
    tp_row = next(r for r in snapshot["summary_rows"] if r["label"] == "Take profit")
    assert sl_row["value"] == "5%"
    assert tp_row["value"] == "0.4%"


def test_snapshot_reads_agent_decision_when_draft_pct_null():
    draft = {
        "symbol": "AAVE_USDT",
        "stop_loss_pct": None,
        "take_profit_pct": None,
        "risk_execution_config": {"stop_loss_pct": 2.0, "take_profit_pct": 5.0},
        "agent_decision": {
            "tool": "modify_strategy_inputs",
            "parameters": {
                "stop_loss": '{"value": 5, "source": "user"}',
                "take_profit": '{"value": 0.4, "source": "user"}',
            },
        },
    }
    snapshot = build_inputs_snapshot_from_draft(draft)
    sl_row = next(r for r in snapshot["summary_rows"] if r["label"] == "Stop loss")
    tp_row = next(r for r in snapshot["summary_rows"] if r["label"] == "Take profit")
    assert sl_row["value"] == "5%"
    assert tp_row["value"] == "0.4%"
