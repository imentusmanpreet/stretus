"""
app/planner/constraint_compiler.py — Apply user/semantic constraints to executable signal plans.

Runs AFTER preset hydration and ExecutionOrchestrator gates, BEFORE builder.apply_signal_plan.
Mutates entry/exit signal lists and rebuilds entry_condition / exit_condition strings so
metadata (RSI band, EMA windows, breakout rules) becomes executable logic.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any

from app.kb.execution_schemas import SemanticInstructions
from app.planner.formulas import render_formula

logger = logging.getLogger(__name__)

# Below this score, auto-confirming a signal plan is unsafe.
SEMANTIC_QUALITY_REVIEW_THRESHOLD = 0.35

_RSI_BAND_RE = re.compile(
    r"rsi\s+(?:between|from)\s+(\d+\.?\d*)\s+(?:and|to|-)\s+(\d+\.?\d*)",
    re.IGNORECASE,
)
_EMA_CROSS_RE = re.compile(
    r"(\d+)\s*ema\s*(?:>|above|over|≥|>=)\s*(\d+)\s*ema",
    re.IGNORECASE,
)
_PREV_TF_HIGH_RE = re.compile(
    r"(?:break(?:out|s)?\s+(?:above|over)|above)\s+(?:the\s+)?(?:previous|prior|last)\s+"
    r"(\d+)[\s-]*(?:min(?:ute)?s?|m)\s+(?:high|candle\s+high|bar\s+high)",
    re.IGNORECASE,
)
_EMA_SLOPE_BULL_RE = re.compile(
    r"ema\s*(\d+)\s+(?:slope|trend)\s+(?:positive|rising|bullish|upward)",
    re.IGNORECASE,
)
_RISING_EMA_RE = re.compile(
    r"(?:rising|positive|bullish|upward)\s+(\d+)\s*ema"
    r"|(\d+)\s*ema\s+(?:is\s+)?(?:rising|positive|bullish|(?:slope\s+)?positive)",
    re.IGNORECASE,
)
_ATR_STOP_RE = re.compile(
    r"(\d+\.?\d*)\s*(?:x\s*)?atr\s+(?:stop|sl)|atr\s+stop(?:\s+loss)?(?:\s+at)?\s*(\d+\.?\d*)?",
    re.IGNORECASE,
)

_EMA_CROSS_SIGNALS = frozenset({
    "ema_cross_up", "ema_cross_down", "ema_above", "ema_below",
    "price_above_ema", "price_below_ema",
})
_RSI_GENERIC_SIGNALS = frozenset({
    "rsi_above_50", "rsi_above_60", "rsi_below_50", "rsi_below_40",
    "rsi_above", "rsi_below",
})
_VOLUME_SIGNALS = frozenset({"volume_spike", "relative_volume_spike"})


def needs_manual_review(quality_score: float) -> bool:
    """True when semantic extraction ran but confidence is too low to auto-finalize."""
    return 0.0 < quality_score < SEMANTIC_QUALITY_REVIEW_THRESHOLD


def apply_semantic_constraints(
    signal_plan: dict[str, Any],
    builder: Any,
    semantic_instructions: SemanticInstructions | None = None,
    source_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Deep-copy *signal_plan*, apply builder + semantic overrides, rebuild conditions.

    Order (user wins over preset):
      preset plan → semantic orchestrator → this compiler → apply_signal_plan
    """
    if not signal_plan:
        return signal_plan

    plan = copy.deepcopy(signal_plan)
    audit: list[str] = list(plan.get("_constraint_compiler_applied") or [])
    prompt = (source_prompt or getattr(builder, "goal", None) or "").strip()

    ema_windows = _resolve_ema_windows(builder, semantic_instructions, prompt)
    if ema_windows:
        fast, slow = ema_windows
        if _ensure_ema_cross_signal(plan, fast, slow, builder):
            audit.append(f"ema_cross:{fast}/{slow}")
        _patch_ema_signals(plan, fast, slow)

    if _apply_rsi_band(plan, builder):
        audit.append(
            f"rsi_band:{getattr(builder, 'rsi_entry_band_min', None)}"
            f"-{getattr(builder, 'rsi_entry_band_max', None)}"
        )

    if _apply_volume_ratio(plan, builder, semantic_instructions, prompt):
        audit.append(f"volume_ratio:{getattr(builder, 'volume_ratio_threshold', None)}")

    if _apply_breakout_rule(plan, builder, prompt):
        audit.append("breakout:prev_tf_high")

    if _apply_ema_slope_filter(plan, prompt):
        audit.append("ema_slope:positive")

    _apply_sl_tp_from_plan(plan, builder, semantic_instructions)
    _apply_time_exit(plan, builder, semantic_instructions, prompt)

    _hydrate_builder_metadata(builder, semantic_instructions, plan)
    _rebuild_conditions(plan)

    if _compile_exit_rules(plan, builder, semantic_instructions, prompt):
        audit.append("exit:compiled")

    _recompute_signals_used(plan)

    if audit:
        plan["_constraint_compiler_applied"] = audit
        logger.info(
            "constraint_compiler|applied|count=%d|changes=%s",
            len(audit),
            ", ".join(audit),
        )
    return plan


# ── EMA ───────────────────────────────────────────────────────────────────────


def _resolve_ema_windows(
    builder: Any,
    semantic: SemanticInstructions | None,
    prompt: str,
) -> tuple[int, int] | None:
    """Return (fast, slow) EMA periods from explicit user prose."""
    # 1. "9 EMA > 21 EMA" cross pattern (highest precision)
    m = _EMA_CROSS_RE.search(prompt)
    if m:
        fast, slow = int(m.group(1)), int(m.group(2))
        if fast < slow:
            return fast, slow
        return min(fast, slow), max(fast, slow)

    # 2. SemanticInstructions.indicators
    if semantic and semantic.indicators.get("EMA"):
        windows = sorted(set(int(w) for w in semantic.indicators["EMA"] if w))
        if len(windows) >= 2:
            return windows[0], windows[-1]
        if len(windows) == 1:
            w = windows[0]
            return (w, w * 2) if w <= 50 else (w // 2, w)

    # 3. Canonical semantic_intent on builder
    intent = getattr(builder, "semantic_intent", None) or {}
    if isinstance(intent, dict):
        ind = intent.get("indicators") or {}
        ema = ind.get("EMA") if isinstance(ind, dict) else None
        if isinstance(ema, list) and len(ema) >= 2:
            windows = sorted(int(x) for x in ema)
            return windows[0], windows[-1]

    return None


def _ensure_ema_cross_signal(
    plan: dict[str, Any],
    fast: int,
    slow: int,
    builder: Any,
) -> bool:
    """Insert ema_cross_up when user specified EMA windows but preset omitted EMA."""
    entry = plan.setdefault("entry", [])
    names = {(s.get("name") or "").lower() for s in entry if isinstance(s, dict)}
    if any(n in _EMA_CROSS_SIGNALS for n in names):
        return False
    entry.append({
        "name": "ema_cross_up",
        "params": {"window_fast": fast, "window_slow": slow},
        "signal_type": "FILTER",
        "source": "SEMANTIC",
    })
    # Persist on builder semantic intent indicators for downstream YAML
    intent = getattr(builder, "semantic_intent", None) or {}
    if isinstance(intent, dict):
        ind = dict(intent.get("indicators") or {})
        ind["EMA"] = [fast, slow]
        intent["indicators"] = ind
        builder.semantic_intent = intent
    return True


def _patch_ema_signals(plan: dict[str, Any], fast: int, slow: int) -> bool:
    changed = False
    for bucket in ("entry", "entry_filters"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict):
                continue
            name = (sig.get("name") or "").lower()
            if name not in _EMA_CROSS_SIGNALS:
                continue
            params = dict(sig.get("params") or {})
            if "window_fast" in params or name.startswith("ema_cross") or name in ("ema_above", "ema_below"):
                params["window_fast"] = fast
                params["window_slow"] = slow
                changed = True
            elif "period" in params or name.startswith("price_"):
                params["period"] = fast
                changed = True
            else:
                params["window_fast"] = fast
                params["window_slow"] = slow
                changed = True
            sig["params"] = params
    return changed


# ── RSI band ──────────────────────────────────────────────────────────────────


def _apply_rsi_band(plan: dict[str, Any], builder: Any) -> bool:
    rmin = getattr(builder, "rsi_entry_band_min", None)
    rmax = getattr(builder, "rsi_entry_band_max", None)
    if rmin is None and rmax is None:
        return False

    lo = float(rmin) if rmin is not None else 0.0
    hi = float(rmax) if rmax is not None else 100.0
    window = 14

    # Drop generic RSI threshold signals that contradict the band
    for bucket in ("entry", "entry_filters"):
        if bucket not in plan:
            continue
        plan[bucket] = [
            s for s in plan[bucket]
            if isinstance(s, dict)
            and (s.get("name") or "").lower() not in _RSI_GENERIC_SIGNALS
        ]

    band_clause = f"RSI({int(window)}) >= {lo:g} AND RSI({int(window)}) <= {hi:g}"
    extra = plan.setdefault("_entry_constraint_clauses", [])
    if band_clause not in extra:
        extra.append(band_clause)
    return True


# ── Volume ratio ──────────────────────────────────────────────────────────────


def _apply_volume_ratio(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None = None,
    prompt: str = "",
) -> bool:
    ratio = getattr(builder, "volume_ratio_threshold", None)
    if ratio is None and semantic and semantic.volume_ratio_threshold is not None:
        ratio = semantic.volume_ratio_threshold
        builder.volume_ratio_threshold = float(ratio)
    if ratio is None:
        m = re.search(
            r"volume\s+(?:at\s+least\s+|≥|>|above\s+)?(\d+\.?\d*)(?:x|×)",
            prompt,
            re.IGNORECASE,
        )
        if m:
            ratio = float(m.group(1))
            builder.volume_ratio_threshold = ratio
    if ratio is None:
        return False
    changed = False
    for bucket in ("entry", "entry_filters"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict):
                continue
            if (sig.get("name") or "").lower() not in _VOLUME_SIGNALS:
                continue
            params = dict(sig.get("params") or {})
            params["multiplier"] = float(ratio)
            sig["params"] = params
            changed = True
    return changed


# ── Breakout ──────────────────────────────────────────────────────────────────


def _apply_breakout_rule(
    plan: dict[str, Any],
    builder: Any,
    prompt: str,
) -> bool:
    m = _PREV_TF_HIGH_RE.search(prompt)
    if not m and not re.search(
        r"break(?:out|s)?\s+(?:above|over)\s+(?:previous|prior|last)\s+(?:high|bar)",
        prompt,
        re.IGNORECASE,
    ):
        return False

    tf_minutes = int(m.group(1)) if m else 15
    chart_tf = (getattr(builder, "timeframe", None) or "15m").lower().strip()
    window = _bars_for_timeframe(tf_minutes, chart_tf)

    entry = plan.setdefault("entry", [])
    names = {(s.get("name") or "").lower() for s in entry if isinstance(s, dict)}

    # Prefer Donchian-style breakout; demote generic ORB if present
    if "opening_range_breakout" in names and "n_bar_high_breakout" not in names:
        entry[:] = [
            s for s in entry
            if (s.get("name") or "").lower() != "opening_range_breakout"
        ]
        names.discard("opening_range_breakout")

    if "n_bar_high_breakout" not in names:
        entry.insert(0, {
            "name": "n_bar_high_breakout",
            "params": {"window": max(1, window)},
            "signal_type": "TRIGGER",
            "source": "SEMANTIC",
        })
        return True
    # Update window on existing breakout signal
    for sig in entry:
        if (sig.get("name") or "").lower() == "n_bar_high_breakout":
            sig.setdefault("params", {})["window"] = max(1, window)
            return True
    return False


def _apply_ema_slope_filter(plan: dict[str, Any], prompt: str) -> bool:
    """Add ema_sloping_up when user asks for positive EMA slope."""
    window: int | None = None
    m = _EMA_SLOPE_BULL_RE.search(prompt)
    if m:
        window = int(m.group(1))
    else:
        rm = _RISING_EMA_RE.search(prompt)
        if rm:
            window = int(rm.group(1) or rm.group(2) or 21)
    if window is None:
        return False
    entry = plan.setdefault("entry", [])
    names = {(s.get("name") or "").lower() for s in entry if isinstance(s, dict)}
    if "ema_sloping_up" in names:
        return False
    entry.append({
        "name": "ema_sloping_up",
        "params": {"window": window, "lookback": 3},
        "signal_type": "FILTER",
        "source": "SEMANTIC",
    })
    return True


def _bars_for_timeframe(target_minutes: int, chart_tf: str) -> int:
    tf_map = {"1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "1d": 1440}
    chart_m = tf_map.get(chart_tf, 15)
    if chart_m <= 0:
        return max(1, target_minutes // 15)
    return max(1, round(target_minutes / chart_m))


# ── SL / TP / time exit ───────────────────────────────────────────────────────


def _apply_sl_tp_from_plan(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
) -> None:
    """Sync planner SL/TP onto builder; honor semantic RR when present."""
    sl = plan.get("_sl_pct")
    tp = plan.get("_tp_pct")
    if sl is not None and getattr(builder, "stop_loss", None) is None:
        builder.stop_loss = float(sl)
    if tp is not None and getattr(builder, "take_profit", None) is None:
        builder.take_profit = float(tp)

    rr_ratio = None
    if semantic and semantic.risk_reward and semantic.risk_reward.ratio:
        rr_ratio = float(semantic.risk_reward.ratio)
    elif plan.get("_semantic_risk_reward", {}).get("ratio"):
        rr_ratio = float(plan["_semantic_risk_reward"]["ratio"])

    if rr_ratio and rr_ratio > 0 and getattr(builder, "stop_loss", None):
        builder.take_profit = round(float(builder.stop_loss) * rr_ratio, 4)
        rms = dict(getattr(builder, "risk_execution_config", None) or {})
        sources = dict(rms.get("rms_sources", {}))
        if sources.get("take_profit_pct") != "user":
            rms["take_profit_pct"] = builder.take_profit
            rms["risk_reward"] = rr_ratio
            sources["take_profit_pct"] = "semantic"
            sources["risk_reward"] = "semantic"
            rms["rms_sources"] = sources
            builder.risk_execution_config = rms


def _apply_time_exit(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
    prompt: str,
) -> None:
    """Square-off / mandatory exit time from prose or entry window end."""
    exit_time = None
    if re.search(
        r"(?:square[\s-]?off|mandatory\s+exit|force\s+exit|exit\s+all)\s+(?:at|by)\s+(\d{1,2}:\d{2})",
        prompt,
        re.IGNORECASE,
    ):
        m = re.search(
            r"(?:square[\s-]?off|mandatory\s+exit|force\s+exit|exit\s+all)\s+(?:at|by)\s+(\d{1,2}:\d{2})",
            prompt,
            re.IGNORECASE,
        )
        if m:
            exit_time = m.group(1)
    if not exit_time:
        exit_time = getattr(builder, "entry_window_end", None)

    if exit_time and not plan.get("_time_exit"):
        plan["_time_exit"] = {
            "exit_time": str(exit_time).strip(),
            "timezone": "Asia/Kolkata",
        }
        builder.time_exit = dict(plan["_time_exit"])


# ── Rebuild condition strings ─────────────────────────────────────────────────


def _rebuild_conditions(plan: dict[str, Any]) -> None:
    """Recompute entry_condition / exit_condition from (possibly mutated) signals."""
    entry_parts: list[str] = []
    for bucket in ("entry", "entry_filters"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict) or not sig.get("name"):
                continue
            part = render_formula(sig["name"], dict(sig.get("params") or {}))
            if part:
                entry_parts.append(part)

    for clause in plan.get("_entry_constraint_clauses") or []:
        if clause and clause not in entry_parts:
            entry_parts.append(clause)

    if entry_parts:
        plan["entry_condition"] = " AND ".join(entry_parts)

    exit_parts: list[str] = []
    for sig in plan.get("exit") or []:
        if not isinstance(sig, dict) or not sig.get("name"):
            continue
        part = render_formula(sig["name"], dict(sig.get("params") or {}))
        if part:
            exit_parts.append(part)

    if exit_parts:
        plan["exit_condition"] = " OR ".join(exit_parts)


def _recompute_signals_used(plan: dict[str, Any]) -> None:
    """Refresh signals_used after semantic rewrites (ORB → n_bar_high_breakout, etc.)."""
    names: list[str] = []
    seen: set[str] = set()
    for bucket in ("entry", "entry_filters", "exit"):
        for sig in plan.get(bucket) or []:
            if not isinstance(sig, dict):
                continue
            name = (sig.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    if names:
        plan["signals_used"] = names


def _hydrate_builder_metadata(
    builder: Any,
    semantic: SemanticInstructions | None,
    plan: dict[str, Any],
) -> None:
    """Sync compiled values back onto builder so root/draft JSON stops showing nulls."""
    if getattr(builder, "volume_ratio_threshold", None) is None:
        for bucket in ("entry", "entry_filters"):
            for sig in plan.get(bucket) or []:
                if not isinstance(sig, dict):
                    continue
                if (sig.get("name") or "").lower() not in _VOLUME_SIGNALS:
                    continue
                mult = (sig.get("params") or {}).get("multiplier")
                if mult is not None:
                    builder.volume_ratio_threshold = float(mult)
                    break
        if builder.volume_ratio_threshold is None and semantic and semantic.volume_ratio_threshold:
            builder.volume_ratio_threshold = float(semantic.volume_ratio_threshold)

    if plan.get("_stop_loss_spec") and not getattr(builder, "stop_loss_spec", None):
        builder.stop_loss_spec = dict(plan["_stop_loss_spec"])
    if plan.get("_trailing_stop_spec") and not getattr(builder, "trailing_stop_spec", None):
        builder.trailing_stop_spec = dict(plan["_trailing_stop_spec"])
    if plan.get("_time_exit") and not getattr(builder, "time_exit", None):
        builder.time_exit = dict(plan["_time_exit"])


def _semantic_sl_to_loader_spec(sl: Any) -> dict[str, Any]:
    """Map SemanticInstructions.stop_loss to quant_engine loader shape."""
    padding = sl.padding
    if padding and padding.method == "atr":
        return {
            "type": "atr",
            "multiplier": float(padding.atr_multiple or 1.5),
            "window": 14,
        }
    anchor = (sl.anchor or "").strip().lower()
    if anchor in {
        "swing_low_recent", "opening_range_low", "opening_range_high",
        "candle_low", "reclaim_candle", "vwap_deviation", "opposite_side",
        "prev_n_bar_low", "prev_n_bar_high",
    }:
        spec: dict[str, Any] = {"type": "structural", "anchor": anchor}
        if "opening_range" in anchor:
            spec["opening_bars"] = 3
        else:
            spec["window"] = 5
        if padding and padding.method == "percent" and padding.percent:
            spec["padding_pct"] = float(padding.percent)
        return spec
    # Plain ATR stop phrasing without structural anchor
    if re.search(r"atr", (sl.description or sl.type or ""), re.IGNORECASE):
        mult = 1.5
        if padding and padding.atr_multiple:
            mult = float(padding.atr_multiple)
        return {"type": "atr", "multiplier": mult, "window": 14}
    return {"type": "atr", "multiplier": 1.5, "window": 14}


def _semantic_ts_to_loader_spec(ts: Any) -> dict[str, Any]:
    """Map TrailingStopConfig to quant_engine loader shape."""
    raw_type = (ts.type or "").lower()
    spec: dict[str, Any] = {
        "activate_after_pct": float(ts.activate_after_pct or 0.0),
    }
    if raw_type in ("ema_based", "ema"):
        return {
            **spec,
            "type": "ema",
            "window": int(ts.ema_period or 9),
        }
    if raw_type in ("atr_based", "atr", "chandelier"):
        return {
            **spec,
            "type": "atr",
            "multiplier": float(ts.trail_distance_atr_multiple or 2.0),
            "window": 14,
        }
    if raw_type == "percent_based":
        return {
            **spec,
            "type": "percent",
            "distance_pct": float(ts.trail_distance_percent or 1.0),
        }
    return {**spec, "type": "ema", "window": int(ts.ema_period or 9)}


def _compile_exit_rules(
    plan: dict[str, Any],
    builder: Any,
    semantic: SemanticInstructions | None,
    prompt: str,
) -> bool:
    """Materialize ATR/structural SL, trailing stop, RR, and time exit on plan + builder."""
    changed = False
    sl_spec: dict[str, Any] | None = None
    ts_spec: dict[str, Any] | None = None

    if getattr(builder, "stop_loss_spec", None):
        sl_spec = dict(builder.stop_loss_spec)
    elif semantic and semantic.stop_loss:
        sl_spec = _semantic_sl_to_loader_spec(semantic.stop_loss)
    elif _ATR_STOP_RE.search(prompt):
        m = _ATR_STOP_RE.search(prompt)
        mult = 1.5
        if m:
            mult = float(m.group(1) or m.group(2) or 1.5)
        sl_spec = {"type": "atr", "multiplier": mult, "window": 14}

    if sl_spec:
        sl_spec.setdefault("source", "semantic")
        plan["_stop_loss_spec"] = sl_spec
        builder.stop_loss_spec = dict(sl_spec)
        changed = True

    if getattr(builder, "trailing_stop_spec", None):
        ts_spec = dict(builder.trailing_stop_spec)
    elif semantic and semantic.trailing_stop and semantic.trailing_stop.enabled:
        ts_spec = _semantic_ts_to_loader_spec(semantic.trailing_stop)
    elif re.search(r"trailing\s+stop|trail\s+(?:sl|stop)|ema\s+trail", prompt, re.IGNORECASE):
        ema_m = re.search(r"ema\s*(\d+)", prompt, re.IGNORECASE)
        ts_spec = {
            "type": "ema",
            "window": int(ema_m.group(1)) if ema_m else 9,
            "activate_after_pct": 0.0,
            "source": "semantic",
        }

    if ts_spec:
        ts_spec.setdefault("source", "semantic")
        plan["_trailing_stop_spec"] = ts_spec
        builder.trailing_stop_spec = dict(ts_spec)
        changed = True

    # Rich exit: percent SL/TP engine path + optional signal exits (not VWAP-only)
    exit_parts: list[str] = []
    if sl_spec or ts_spec or plan.get("_time_exit"):
        exit_parts.append(
            "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
        )
    for sig in plan.get("exit") or []:
        if not isinstance(sig, dict):
            continue
        name = (sig.get("name") or "").lower()
        # VWAP flip alone is too weak when user asked for ATR/trailing/time exits
        if sl_spec and name in ("vwap_bearish", "price_below_vwap", "below_vwap"):
            continue
        part = render_formula(sig.get("name", ""), dict(sig.get("params") or {}))
        if part and part not in exit_parts:
            exit_parts.append(part)

    if exit_parts:
        plan["exit_condition"] = " OR ".join(exit_parts)
        changed = True

    plan["risk_model"] = {
        "stop_loss": sl_spec,
        "trailing_stop": ts_spec,
        "risk_reward": (
            semantic.risk_reward.ratio
            if semantic and semantic.risk_reward and semantic.risk_reward.ratio
            else (plan.get("_semantic_risk_reward") or {}).get("ratio")
        ),
        "time_exit": plan.get("_time_exit"),
    }
    return changed
