"""
app/planner/strategy_assembler.py — Build the legacy-shaped strategy_object,
strategy_config, and retrieval_meta dicts from a populated StrategyBuilder.

This replaces app.services.knowledge.retriever.build_strategy_object /
build_strategy_config / build_retrieval_meta. The output dict shapes are
unchanged so chat_service.py and the response composers don't need to know
the planner was rewritten.
"""
from __future__ import annotations

import copy
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.kb import kb
from app.services.execution.risk_execution_config_service import (
    build_risk_and_execution_from_builder,
)


# ── Public helpers ────────────────────────────────────────────────────────────


def build_strategy_object(builder: Any) -> dict:
    """Build {strategy_object: {...}} from a StrategyBuilder whose signal_plan
    is already populated. The signal_plan must be in the legacy shape
    (entry/exit/signals_used/etc.) — produced by either the new planner
    bridge or the legacy retriever."""
    builder.apply_defaults()  # DB risk_execution_config + risk_tiers.yaml for unset fields
    plan = builder.signal_plan or {}

    entry_condition = builder.entry_condition or plan.get("entry_condition")
    exit_condition  = builder.exit_condition  or plan.get("exit_condition")

    reward_factor = None
    if builder.stop_loss and builder.take_profit and builder.stop_loss > 0:
        reward_factor = round(float(builder.take_profit) / float(builder.stop_loss), 2)

    asset = builder.format_symbol() or builder.symbol or "strategy"
    strategy_name = f"{re.sub(r'[^a-zA-Z0-9]+', '_', asset).lower()}_strategy"
    timeframe = builder.timeframe or "1d"

    signals_used = [
        str(name)
        for name in plan.get("signals_used", [])
        if str(name).strip()
    ] or [
        str(signal.get("name"))
        for signal in plan.get("entry", []) + plan.get("exit", [])
        if isinstance(signal, dict) and signal.get("name")
    ]
    raw_signals_available = plan.get("signals_available")
    try:
        signals_available = int(raw_signals_available)
    except (TypeError, ValueError):
        signals_available = len(kb.signals)

    indicators = _extract_indicators(entry_condition, exit_condition)

    risk_and_execution = build_risk_and_execution_from_builder(builder)
    execution_controls = _build_execution_controls(builder)

    strategy_object = {
        "ai_strategy_id":  str(uuid.uuid4()),
        "name":            strategy_name,
        "status":          "draft",
        "asset":           asset,
        "meta": {
            "market":               builder.market,
            "objective":            builder.objective,
            "goal":                 getattr(builder, "goal", None),
            "sentiment":            builder.sentiment,
            "timeframe":            timeframe,
            "experience":           builder.experience,
            "indicators":           indicators,
            "kb_signals_available": signals_available,
            "kb_signals_used":      signals_used,
            "strategy_preset":      getattr(builder, "strategy_preset", None),
        },
        "entry_condition":     entry_condition,
        "exit_condition":      exit_condition,
        "entry":               copy.deepcopy(plan.get("entry", [])),
        "exit":                copy.deepcopy(plan.get("exit", [])),
        "risk_and_execution":  risk_and_execution,
        "execution_controls":  execution_controls,
        "reward_factor":       reward_factor,
    }

    for item in strategy_object["entry"] + strategy_object["exit"]:
        item["timeframe"] = timeframe

    return {"strategy_object": strategy_object}


def build_strategy_config(builder: Any, strategy_payload: dict) -> dict:
    """Build the strategy_config dict (used by the YAML generator + UI)."""
    strategy_object = strategy_payload.get("strategy_object", {})
    return {
        "exit":          copy.deepcopy(strategy_object.get("exit", [])),
        "name":          strategy_object.get("name", "strategy"),
        "asset":         strategy_object.get("asset", builder.format_symbol()),
        "entry":         copy.deepcopy(strategy_object.get("entry", [])),
        "goal":          getattr(builder, "goal", None),
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "reward_factor": strategy_object.get("reward_factor"),
    }


def build_retrieval_meta(strategy_payload: dict) -> dict:
    """Lightweight summary embedded in chat responses."""
    strategy_object = strategy_payload.get("strategy_object", {})
    return {
        "signals_available": len(kb.signals),
        "signals_used": [
            signal["name"]
            for signal in strategy_object.get("entry", []) + strategy_object.get("exit", [])
        ],
    }


def enrich_plan_with_ohlcv(
    plan: dict,
    ohlcv_records: list[dict],
    objective: str,
    timeframe: str,
) -> dict:
    """Re-estimate signal parameters from OHLCV statistics and re-render the
    entry/exit condition strings. Returns the plan unchanged if there are no
    records to work with."""
    if not ohlcv_records or not plan:
        return plan
    from app.core.signal_param_estimator import enrich_plan_params, ohlcv_to_df
    df = ohlcv_to_df(ohlcv_records)
    return enrich_plan_params(plan, df, objective, timeframe)


# ── Internal ──────────────────────────────────────────────────────────────────


def _build_execution_controls(builder: Any) -> dict[str, Any]:
    """Mirror draft-level execution parameters on the persisted strategy_object."""
    controls: dict[str, Any] = {}

    for field in (
        "direction",
        "entry_window_start",
        "entry_window_end",
        "gap_filter",
        "position_sizing_mode",
    ):
        val = getattr(builder, field, None)
        if val is not None and str(val).strip():
            controls[field] = val

    for field in (
        "max_consecutive_losses",
        "cooldown_bars_after_loss",
        "cooldown_bars_after_profit",
        "entry_confirmation_bars",
    ):
        val = getattr(builder, field, None)
        if val is not None:
            controls[field] = int(val)

    for field in (
        "max_spread_bps",
        "gap_threshold_pct",
        "rsi_entry_band_min",
        "rsi_entry_band_max",
        "volume_ratio_threshold",
        "max_capital_allocation_pct",
    ):
        val = getattr(builder, field, None)
        if val is not None:
            controls[field] = float(val)

    if getattr(builder, "stop_loss_spec", None):
        controls["stop_loss_spec"] = dict(builder.stop_loss_spec)
    if getattr(builder, "trailing_stop_spec", None):
        controls["trailing_stop_spec"] = dict(builder.trailing_stop_spec)
    if getattr(builder, "time_exit", None):
        controls["time_exit"] = dict(builder.time_exit)
    if getattr(builder, "reference_symbol", None):
        controls["reference_symbol"] = builder.reference_symbol
    if getattr(builder, "htf_rules", None):
        controls["htf_rules"] = list(builder.htf_rules)

    rsi_min = controls.get("rsi_entry_band_min")
    rsi_max = controls.get("rsi_entry_band_max")
    if rsi_min is not None or rsi_max is not None:
        controls["rsi_entry_band"] = {
            k: v
            for k, v in {"min": rsi_min, "max": rsi_max}.items()
            if v is not None
        }

    return controls


def _extract_indicators(entry_condition: str | None, exit_condition: str | None) -> dict[str, list[int]]:
    blob = f"{entry_condition or ''} {exit_condition or ''}"
    indicators: dict[str, list[int]] = {}
    for indicator in ("SMA", "EMA", "RSI", "ADX", "BB_UPPER", "BB_LOWER"):
        matches = re.findall(rf"{indicator}\((\d+)\)", blob, re.IGNORECASE)
        if matches:
            indicators[indicator] = sorted(set(map(int, matches)))
    if "MACD" in blob.upper():
        indicators["MACD"] = []
    if "VWAP" in blob.upper():
        indicators["VWAP"] = []
    return indicators
