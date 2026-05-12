"""Tool catalog for the behavior-driven algo trading agent."""
from __future__ import annotations

from enum import Enum
from typing import Any


class AgentToolName(str, Enum):
    MODIFY_STRATEGY_INPUTS = "modify_strategy_inputs"
    ASK_USER_FOR_CLARIFICATION = "ask_user_for_clarification"
    PLAN_STRATEGY_SIGNALS = "plan_strategy_signals"
    ASSEMBLE_STRATEGY = "assemble_strategy"
    RUN_BACKTEST = "run_backtest"
    FETCH_MARKET_DATA = "fetch_market_data"
    GET_BACKTEST_RESULT = "get_backtest_result"
    MARK_STRATEGY_REJECTED = "mark_strategy_rejected"
    UPDATE_RISK_EXECUTION_CONFIG = "update_risk_execution_config"
    START_NEW_STRATEGY = "start_new_strategy"
    PAUSE_WORKFLOW = "pause_workflow"
    RESPOND_WITH_SAFETY_BOUNDARY = "respond_with_safety_boundary"
    RESPOND_TEXT = "respond_text"


_CORE_INPUT_PROPERTIES: dict[str, Any] = {
    "symbol": {
        "type": "string",
        "description": "Indian stock/company query or exchange-qualified symbol, for example INFY.NS.",
    },
    "timeframe": {
        "type": "string",
        "enum": ["1m", "5m", "10m", "15m", "30m", "1h", "1d"],
    },
    "objective": {"type": "string", "enum": ["intraday", "positional"]},
    "sentiment": {"type": "string", "enum": ["bullish", "bearish"]},
    "experience": {"type": "string", "enum": ["beginner", "intermediate", "expert"]},
    "goal": {
        "type": "string",
        "description": "Short natural-language trading objective, e.g. breakout with controlled risk.",
    },
}


def _tool(name: AgentToolName, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name.value,
            "description": description,
            "parameters": parameters,
        },
    }


AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        AgentToolName.MODIFY_STRATEGY_INPUTS,
        "Create or update strategy inputs. Use whenever the user provides or changes stock, timeframe, objective, sentiment, experience, or goal.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                **_CORE_INPUT_PROPERTIES,
                "preserve_unmentioned_fields": {"type": "boolean", "default": True},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.ASK_USER_FOR_CLARIFICATION,
        "Ask for one missing, ambiguous, or approval-gated decision.",
        {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "reason": {"type": "string"},
                "field": {
                    "type": "string",
                    "enum": ["symbol", "timeframe", "objective", "sentiment", "experience", "goal", "approval", "direction", "other"],
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.PLAN_STRATEGY_SIGNALS,
        "Run the KB-backed strategy planner after all six user inputs are complete and confirmed.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "use_market_data_calibration": {"type": "boolean", "default": True},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.ASSEMBLE_STRATEGY,
        "Assemble and persist the strategy JSON/YAML after the user approves the signal plan.",
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.RUN_BACKTEST,
        "Run a backtest only after the strategy is assembled and the user explicitly approves execution.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "strategy_id": {"type": "string"},
                "starting_balance": {"type": "number"},
                "slippage_bps": {"type": "number"},
                "commission_bps": {"type": "number"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.FETCH_MARKET_DATA,
        "Fetch OHLCV candles for market inquiry, planning, or backtesting.",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {"type": "string"},
                "from_utc": {"type": "string"},
                "to_utc": {"type": "string"},
                "purpose": {"type": "string", "enum": ["planning", "backtest", "market_inquiry"]},
            },
            "required": ["symbol", "interval", "purpose"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.GET_BACKTEST_RESULT,
        "Retrieve and explain the latest or a specific backtest result.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "backtest_id": {"type": "string"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.MARK_STRATEGY_REJECTED,
        "Record user dissatisfaction and halt automatic planning, assembling, or backtesting.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "reason": {"type": "string"},
                "question": {
                    "type": "string",
                    "description": "AI-written user-facing response after halting automation. The backend displays this text directly.",
                },
                "rejected_artifact": {
                    "type": "string",
                    "enum": ["inputs", "signal_plan", "strategy", "backtest", "market_view", "unknown"],
                },
            },
            "required": ["session_id", "reason", "question"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.UPDATE_RISK_EXECUTION_CONFIG,
        "Update risk and execution settings such as stop loss, target, daily loss cap, per-trade risk, or max trades.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "strategy_id": {"type": "string"},
                "stop_loss_pct": {"type": "number"},
                "take_profit_pct": {"type": "number"},
                "daily_loss_cap": {"type": "number"},
                "per_trade_risk": {"type": "number"},
                "max_trades": {"type": "integer"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.START_NEW_STRATEGY,
        "Start a separate fresh strategy and reset the current working strategy inputs.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                **_CORE_INPUT_PROPERTIES,
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.PAUSE_WORKFLOW,
        "Pause or cancel the current workflow without advancing to planning, assembly, or backtesting.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.RESPOND_WITH_SAFETY_BOUNDARY,
        "Respond to stock tips, buy/sell advice, or recommendation requests with the regulated safety boundary.",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        AgentToolName.RESPOND_TEXT,
        "Send a short conversational or educational reply when no backend action should run.",
        {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    ),
]

AGENT_TOOL_NAMES = {tool.value for tool in AgentToolName}
