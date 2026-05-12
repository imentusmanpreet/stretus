"""
app/planner/trade_management.py — Phase 3 trade management rule pack.

Nine independently-configurable rules that sit alongside the Phase-2 signal
filter framework. Each rule has:

  • enabled    — toggleable per strategy.
  • params     — adapts to whatever the user said (no hardcoded numbers when
                  the prompt mentions one).
  • source     — "prompt" if a phrase or value came from the user, "default"
                  otherwise.
  • required   — True for rules the user MUST decide (per the spec:
                  max_daily_trades, position_sizing, slippage). When the
                  user hasn't stated a value, find_missing_trade_management()
                  surfaces these in the missing-critical pipeline.

The rules:

   9. max_daily_trades         — overtrading protection.
  10. daily_loss_limit         — consecutive-loss + cumulative-loss halt.
  11. no_reentry_at_level      — block re-entry near a stopped-out level.
  12. position_sizing          — risk per trade (% of capital or fixed lot).
  13. partial_profit_booking   — scale out at T1, trail the rest.
  14. trailing_stop_loss       — move SL behind price as trade runs.
  15. manual_exit              — bars-since-entry + setup-invalidation rules.
  16. setup_grading            — confidence score from confirmation count.
  17. slippage                 — max acceptable slippage before trade invalid.

apply_trade_management_to_plan() writes everything to plan["_trade_management"]
so the engine / runner / order manager can act on it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.kb.execution_schemas import SemanticInstructions

logger = logging.getLogger(__name__)


# ── Spec data classes ────────────────────────────────────────────────────────


@dataclass
class TradeMgmtRuleSpec:
    name:        str
    enabled:     bool = False
    params:      dict[str, Any] = field(default_factory=dict)
    source:      str = "default"            # "prompt" | "default" | "user_disabled"
    required:    bool = False                # if True and source=="default", flag as missing-critical
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":        self.name,
            "enabled":     self.enabled,
            "params":      dict(self.params),
            "source":      self.source,
            "required":    self.required,
            "description": self.description,
        }


@dataclass
class TradeManagementConfig:
    max_daily_trades:        TradeMgmtRuleSpec
    daily_loss_limit:        TradeMgmtRuleSpec
    no_reentry_at_level:     TradeMgmtRuleSpec
    position_sizing:         TradeMgmtRuleSpec
    partial_profit_booking:  TradeMgmtRuleSpec
    trailing_stop_loss:      TradeMgmtRuleSpec
    manual_exit:             TradeMgmtRuleSpec
    setup_grading:           TradeMgmtRuleSpec
    slippage:                TradeMgmtRuleSpec

    def all(self) -> list[TradeMgmtRuleSpec]:
        return [
            self.max_daily_trades,
            self.daily_loss_limit,
            self.no_reentry_at_level,
            self.position_sizing,
            self.partial_profit_booking,
            self.trailing_stop_loss,
            self.manual_exit,
            self.setup_grading,
            self.slippage,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {r.name: r.to_dict() for r in self.all()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TradeManagementConfig":
        data = data or {}

        def _load(key: str, required: bool = False) -> TradeMgmtRuleSpec:
            raw = data.get(key) or {}
            return TradeMgmtRuleSpec(
                name=key,
                enabled=bool(raw.get("enabled", False)),
                params=dict(raw.get("params") or {}),
                source=str(raw.get("source") or "default"),
                required=bool(raw.get("required", required)),
                description=str(raw.get("description") or ""),
            )

        return cls(
            max_daily_trades       = _load("max_daily_trades",        required=True),
            daily_loss_limit       = _load("daily_loss_limit",        required=True),
            no_reentry_at_level    = _load("no_reentry_at_level"),
            position_sizing        = _load("position_sizing",         required=True),
            partial_profit_booking = _load("partial_profit_booking"),
            trailing_stop_loss     = _load("trailing_stop_loss"),
            manual_exit            = _load("manual_exit"),
            setup_grading          = _load("setup_grading"),
            slippage               = _load("slippage",                required=True),
        )


# ── Detection: turn prompt phrases into rule toggles + params ────────────────


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


_RULE_TRIGGERS: dict[str, list[tuple[str, Callable[[re.Match], dict] | None]]] = {
    "max_daily_trades": [
        (r"(?:max(?:imum)?|up to|at most|limit\s+to)\s+(?P<n>\d+)\s+trades?\s+(?:per\s+day|/day|daily)",
            lambda m: {"max_trades": int(m.group("n"))}),
        (r"(?P<n>\d+)\s+trades?\s+(?:per\s+day|/day|daily)\s+max",
            lambda m: {"max_trades": int(m.group("n"))}),
        (r"no\s+more\s+than\s+(?P<n>\d+)\s+trades?",
            lambda m: {"max_trades": int(m.group("n"))}),
        (r"overtrading\s+protection", None),
    ],
    "daily_loss_limit": [
        # "daily loss 2%", "daily loss limit 2%", "daily loss cap of 2%"
        (r"(?:daily|day)\s+loss\s+(?:limit|cap\s+)?(?:of\s+|at\s+)?(?P<pct>\d+(?:\.\d+)?)\s*%",
            lambda m: {"max_daily_loss_pct": float(m.group("pct"))}),
        # "stop trading after 3 losses" / "halt after 3 consecutive losses"
        (r"(?:stop|halt)\s+(?:trading|signals)\s+after\s+(?P<n>\d+)\s+(?:consecutive\s+)?losses",
            lambda m: {"max_consecutive_losses": int(m.group("n"))}),
        (r"(?:after\s+)?(?P<n>\d+)\s+(?:consecutive\s+)?losses?\s+in\s+a\s+row",
            lambda m: {"max_consecutive_losses": int(m.group("n"))}),
        (r"halt\s+(?:trading|signals)\s+(?:at|after)\s+(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?:loss|drawdown)",
            lambda m: {"max_daily_loss_pct": float(m.group("pct"))}),
    ],
    "no_reentry_at_level": [
        (r"no\s+re[- ]?entry\s+at\s+(?:the\s+)?same\s+(?:level|price)", None),
        (r"don'?t\s+re[- ]?enter\s+(?:at|near)\s+(?:the\s+)?same\s+price", None),
        (r"avoid\s+re[- ]?entry.*?same\s+level", None),
        (r"require\s+(?:a\s+)?fresh\s+setup\s+after\s+stop", None),
        (r"wait\s+for\s+(?:a\s+)?fresh\s+setup", None),
    ],
    "position_sizing": [
        (r"risk\s+(?P<pct>\d+(?:\.\d+)?)\s*%\s+(?:per\s+trade|of\s+capital)",
            lambda m: {"method": "percent_risk", "risk_per_trade_pct": float(m.group("pct"))}),
        (r"(?P<pct>\d+(?:\.\d+)?)\s*%\s+risk\s+per\s+trade",
            lambda m: {"method": "percent_risk", "risk_per_trade_pct": float(m.group("pct"))}),
        (r"(?P<lot>\d+)\s+(?:lot|lots|share|shares)\s+(?:fixed|per\s+trade)",
            lambda m: {"method": "fixed_lot", "lot_size": int(m.group("lot"))}),
        (r"fixed\s+lot\s+(?:size\s+)?(?:of\s+)?(?P<lot>\d+)",
            lambda m: {"method": "fixed_lot", "lot_size": int(m.group("lot"))}),
        (r"position\s+siz(?:e|ing)\s+based\s+on\s+stop\s+loss", None),
        (r"kelly\s+criterion", lambda m: {"method": "kelly"}),
    ],
    "partial_profit_booking": [
        (r"(?:partial|scale)\s+(?:profit|out|exit)\s+(?:at\s+)?(?:T1|first\s+target|target\s+1)", None),
        (r"book\s+(?P<pct>\d+)\s*%\s+at\s+(?:T1|target\s*1|first\s+target)",
            lambda m: {"first_target_pct_close": int(m.group("pct"))}),
        (r"close\s+(?:half|50\s*%)\s+at\s+(?:T1|target\s*1|first\s+target)",
            lambda m: {"first_target_pct_close": 50}),
        (r"partial\s+booking", None),
        (r"trail\s+(?:the\s+)?(?:rest|remainder|remaining)", None),
    ],
    "trailing_stop_loss": [
        (r"trail(?:ing)?\s+stop", None),
        (r"trail\s+(?:the\s+)?stop\s+(?:loss)?", None),
        (r"trail\s+sl\b", None),
        (r"chandelier\s+(?:exit|stop)", lambda m: {"type": "chandelier"}),
        (r"ema\s+trailing", lambda m: {"type": "ema_based"}),
        (r"atr\s+trailing", lambda m: {"type": "atr_based"}),
        (r"move\s+sl\s+to\s+break\s*even\s+(?:at|after)\s+(?P<r>\d+(?:\.\d+)?)\s*(?:R|×|x)?",
            lambda m: {"breakeven_after_r": float(m.group("r"))}),
    ],
    "manual_exit": [
        (r"exit\s+(?:if\s+)?(?:trade\s+)?(?:not\s+moving|stagnant|no\s+progress)\s+(?:within\s+|in\s+|after\s+)?(?P<bars>\d+)\s+(?:candles?|bars?)",
            lambda m: {"max_bars_to_progress": int(m.group("bars"))}),
        (r"(?:close|exit)\s+(?:the\s+)?trade\s+if\s+setup\s+(?:invalidated|broken|no longer valid)",
            lambda m: {"exit_on_setup_invalidation": True}),
        (r"time\s+stop\s+(?:at|after)\s+(?P<bars>\d+)\s+(?:bars?|candles?)",
            lambda m: {"max_bars_in_trade": int(m.group("bars"))}),
        (r"manual\s+exit\s+rules?", None),
    ],
    "setup_grading": [
        (r"(?:grade|score|rank)\s+(?:each\s+)?setup", None),
        (r"setup\s+(?:grading|scoring|confidence)", None),
        (r"more\s+confirmations\s+(?:get|=)\s+(?:higher|full)\s+(?:grade|size)", None),
        (r"weaker\s+setups?\s+(?:get|=)\s+(?:smaller|reduced|skipped)", None),
        (r"a\+\s*setup|a\s+plus\s+setup", None),
    ],
    "slippage": [
        # "slippage of 0.2%", "slippage tolerance 0.2%", "slippage <= 0.2%"
        (r"slippage\s+(?:of\s+|tolerance\s+(?:of\s+)?|<=?\s*|max\s+)?(?P<pct>\d+(?:\.\d+)?)\s*%",
            lambda m: {"max_slippage_pct": float(m.group("pct"))}),
        # "max 0.2% slippage" / "maximum 0.2% slippage" — number BEFORE the word.
        (r"max(?:imum)?\s+(?P<pct>\d+(?:\.\d+)?)\s*%\s+slippage",
            lambda m: {"max_slippage_pct": float(m.group("pct"))}),
        (r"(?P<pct>\d+(?:\.\d+)?)\s*%\s+(?:max(?:imum)?\s+)?slippage",
            lambda m: {"max_slippage_pct": float(m.group("pct"))}),
        (r"max(?:imum)?\s+slippage\s+(?:of\s+)?(?P<bps>\d+)\s*(?:bps|basis\s+points?)",
            lambda m: {"max_slippage_pct": float(m.group("bps")) / 100.0}),
        (r"invalidate\s+(?:trade|signal)\s+(?:if|when)\s+slippage\s+exceeds\s+(?P<pct>\d+(?:\.\d+)?)\s*%",
            lambda m: {"max_slippage_pct": float(m.group("pct")), "invalidate_on_breach": True}),
        (r"slippage\s+(?:handling|model)", None),
    ],
}


def build_trade_management_config(
    prompt_text: str,
    instructions: SemanticInstructions | None,
    builder: Any,
    *,
    existing: TradeManagementConfig | None = None,
) -> TradeManagementConfig:
    """Build a fully-populated TradeManagementConfig by combining:

      • Any toggles the user already set (from `existing` or
        builder.trade_management) so prior turns are preserved.
      • Phrase detection in `prompt_text` (regex triggers above).
      • Side-channel cross-references from `instructions` (e.g. trailing
        stop info already extracted by SemanticExtractor).
      • Builder-level config (risk_execution_config carries daily_loss_cap /
        max_trades / per_trade_risk if they were set elsewhere).
    """
    cfg = existing or _default_config()
    haystack = re.sub(r"\s+", " ", (prompt_text or "")).lower()

    for rule_name, triggers in _RULE_TRIGGERS.items():
        spec = getattr(cfg, rule_name)
        for pattern, extractor in triggers:
            match = re.search(pattern, haystack)
            if not match:
                continue
            spec.enabled = True
            spec.source  = "prompt"
            if extractor:
                try:
                    spec.params.update(extractor(match) or {})
                except Exception:
                    logger.debug(
                        "trade_management|param_extractor_failed|rule=%s",
                        rule_name,
                        exc_info=True,
                    )
            break

    _cross_reference_with_extraction(cfg, instructions)
    _cross_reference_with_builder(cfg, builder)
    _apply_descriptions(cfg)
    return cfg


def _default_config() -> TradeManagementConfig:
    return TradeManagementConfig(
        max_daily_trades       = TradeMgmtRuleSpec(name="max_daily_trades",       required=True),
        daily_loss_limit       = TradeMgmtRuleSpec(name="daily_loss_limit",       required=True),
        no_reentry_at_level    = TradeMgmtRuleSpec(name="no_reentry_at_level"),
        position_sizing        = TradeMgmtRuleSpec(name="position_sizing",        required=True),
        partial_profit_booking = TradeMgmtRuleSpec(name="partial_profit_booking"),
        trailing_stop_loss     = TradeMgmtRuleSpec(name="trailing_stop_loss"),
        manual_exit            = TradeMgmtRuleSpec(name="manual_exit"),
        setup_grading          = TradeMgmtRuleSpec(name="setup_grading"),
        slippage               = TradeMgmtRuleSpec(name="slippage",               required=True),
    )


def _cross_reference_with_extraction(
    cfg: TradeManagementConfig,
    instructions: SemanticInstructions | None,
) -> None:
    if instructions is None:
        return
    # Trailing stop info captured by the SemanticExtractor.
    if instructions.trailing_stop and getattr(instructions.trailing_stop, "enabled", False):
        ts = instructions.trailing_stop
        cfg.trailing_stop_loss.enabled = True
        cfg.trailing_stop_loss.source  = cfg.trailing_stop_loss.source or "prompt"
        if ts.type:
            cfg.trailing_stop_loss.params.setdefault("type", ts.type)
        if getattr(ts, "ema_period", None):
            cfg.trailing_stop_loss.params.setdefault("ema_period", ts.ema_period)
        if getattr(ts, "atr_multiple", None):
            cfg.trailing_stop_loss.params.setdefault("atr_multiple", ts.atr_multiple)
        if getattr(ts, "activate_after_pct", None):
            cfg.trailing_stop_loss.params.setdefault("activate_after_pct", ts.activate_after_pct)


def _cross_reference_with_builder(cfg: TradeManagementConfig, builder: Any) -> None:
    risk_cfg = getattr(builder, "risk_execution_config", None) or {}
    # max_daily_trades inherits from risk_execution_config when set.
    if risk_cfg.get("max_trades") is not None and not cfg.max_daily_trades.enabled:
        cfg.max_daily_trades.enabled = True
        cfg.max_daily_trades.source  = "prompt"
        cfg.max_daily_trades.params.setdefault("max_trades", int(risk_cfg["max_trades"]))
    # daily_loss inherits from risk_execution_config / builder.daily_loss_cap.
    if risk_cfg.get("daily_loss_cap") is not None and not cfg.daily_loss_limit.enabled:
        cfg.daily_loss_limit.enabled = True
        cfg.daily_loss_limit.source  = "prompt"
        cfg.daily_loss_limit.params.setdefault("max_daily_loss_pct", float(risk_cfg["daily_loss_cap"]))
    elif getattr(builder, "daily_loss_cap", None) is not None and not cfg.daily_loss_limit.enabled:
        cfg.daily_loss_limit.enabled = True
        cfg.daily_loss_limit.source  = "prompt"
        cfg.daily_loss_limit.params.setdefault("max_daily_loss_pct", float(builder.daily_loss_cap))
    # Position sizing inherits per_trade_risk if present.
    if risk_cfg.get("per_trade_risk") is not None and not cfg.position_sizing.enabled:
        cfg.position_sizing.enabled = True
        cfg.position_sizing.source  = "prompt"
        cfg.position_sizing.params.setdefault("method", "percent_risk")
        cfg.position_sizing.params.setdefault(
            "risk_per_trade_pct", float(risk_cfg["per_trade_risk"]),
        )


def _apply_descriptions(cfg: TradeManagementConfig) -> None:
    for spec in cfg.all():
        spec.description = _describe(spec)


def _describe(spec: TradeMgmtRuleSpec) -> str:
    if not spec.enabled:
        return ""
    p = spec.params
    if spec.name == "max_daily_trades":
        n = p.get("max_trades")
        return f"Stop generating new signals once {n} trades have been taken today." if n else \
               "Stop generating new signals once the daily trade cap is reached."
    if spec.name == "daily_loss_limit":
        bits = []
        if p.get("max_daily_loss_pct") is not None:
            bits.append(f"halt at {p['max_daily_loss_pct']}% daily loss")
        if p.get("max_consecutive_losses") is not None:
            bits.append(f"halt after {p['max_consecutive_losses']} consecutive losses")
        return ("Halt trading when: " + ", ".join(bits) + ".") if bits else "Halt trading on daily-loss breach."
    if spec.name == "no_reentry_at_level":
        radius = p.get("level_radius_pct", 0.2)
        return f"Reject new entries within {radius}% of any price level where a stop loss was hit today."
    if spec.name == "position_sizing":
        method = p.get("method", "percent_risk")
        if method == "percent_risk":
            r = p.get("risk_per_trade_pct")
            return f"Risk {r}% of capital per trade; size = (risk amount) / (stop distance)." if r \
                   else "Percent-risk sizing — value needed from you."
        if method == "fixed_lot":
            return f"Fixed lot size of {p.get('lot_size')} units per trade."
        return f"Sizing method: {method}."
    if spec.name == "partial_profit_booking":
        first = p.get("first_target_pct_close", 50)
        return f"Close {first}% of the position at the first target, trail the remainder."
    if spec.name == "trailing_stop_loss":
        ttype = p.get("type", "atr_based")
        details = []
        if p.get("ema_period"):
            details.append(f"EMA({p['ema_period']})")
        if p.get("atr_multiple"):
            details.append(f"{p['atr_multiple']}×ATR")
        if p.get("activate_after_pct"):
            details.append(f"after {p['activate_after_pct']}% profit")
        return f"Trailing stop ({ttype})" + (": " + ", ".join(details) if details else ".")
    if spec.name == "manual_exit":
        bits = []
        if p.get("max_bars_to_progress"):
            bits.append(f"close if trade hasn't progressed within {p['max_bars_to_progress']} bars")
        if p.get("max_bars_in_trade"):
            bits.append(f"time-stop after {p['max_bars_in_trade']} bars")
        if p.get("exit_on_setup_invalidation"):
            bits.append("close when the entry setup is no longer valid")
        return "; ".join(bits) if bits else "Manual / discretionary exit rule enabled."
    if spec.name == "setup_grading":
        return ("Grade each setup by counting active confirmations; A-grade trades take full size, "
                "B-grade trades take reduced size, C-grade trades are skipped.")
    if spec.name == "slippage":
        bits = []
        if p.get("max_slippage_pct") is not None:
            bits.append(f"max {p['max_slippage_pct']}% slippage")
        if p.get("invalidate_on_breach"):
            bits.append("invalidate trade if breached")
        return "; ".join(bits) if bits else "Slippage handling enabled."
    return ""


# ── Missing-critical surfacing ────────────────────────────────────────────────


def find_missing_trade_management(cfg: TradeManagementConfig) -> list[dict[str, str]]:
    """Return the list of required rules the user didn't specify. The chat
    flow surfaces these in workflow.missing_critical_inputs so the user
    is asked rather than defaulted on the must-decide rules per the spec."""
    missing: list[dict[str, str]] = []
    questions = {
        "max_daily_trades": (
            "max daily trades",
            "What is the maximum number of trades per day? (e.g. 'max 3 trades per day')",
        ),
        "daily_loss_limit": (
            "daily loss limit",
            "When should the system stop trading for the day? "
            "(e.g. '2% daily loss cap', 'halt after 3 consecutive losses')",
        ),
        "position_sizing": (
            "position sizing rule",
            "How should position size be decided? "
            "(e.g. 'risk 1% per trade', 'fixed lot size 75')",
        ),
        "slippage": (
            "slippage tolerance",
            "What is the maximum acceptable slippage before the trade becomes invalid? "
            "(e.g. 'max 0.2% slippage')",
        ),
    }
    for rule_name, (label, question) in questions.items():
        spec = getattr(cfg, rule_name)
        if spec.required and not spec.enabled:
            missing.append({
                "field":    f"tm.{rule_name}",
                "label":    label,
                "question": question,
            })
    return missing


# ── Plan integration ─────────────────────────────────────────────────────────


def apply_trade_management_to_plan(
    plan: dict[str, Any],
    cfg: TradeManagementConfig,
) -> dict[str, Any]:
    """Attach the full trade-management config to the plan under
    `_trade_management`. Also writes specific side-channel slots the engine
    already understands (trailing stop, daily loss cap, max trades) so
    nothing depends on engine-side reading the new block."""
    audit: dict[str, Any] = {"side_channels": [], "engine_overrides": []}
    plan["_trade_management"] = cfg.to_dict()
    audit["side_channels"].append("trade_management")

    # Backwards compatibility — drop values into engine-known slots.
    tsl = cfg.trailing_stop_loss
    if tsl.enabled and not plan.get("_trailing_stop_spec"):
        plan["_trailing_stop_spec"] = {
            "type":               tsl.params.get("type", "atr_based"),
            "ema_period":         tsl.params.get("ema_period"),
            "atr_multiple":       tsl.params.get("atr_multiple"),
            "activate_after_pct": tsl.params.get("activate_after_pct"),
        }
        audit["engine_overrides"].append("trailing_stop_spec")

    mdt = cfg.max_daily_trades
    if mdt.enabled and mdt.params.get("max_trades") is not None:
        plan.setdefault("_risk_overrides", {})["max_trades"] = int(mdt.params["max_trades"])
        audit["engine_overrides"].append("max_trades")

    dll = cfg.daily_loss_limit
    if dll.enabled and dll.params.get("max_daily_loss_pct") is not None:
        plan.setdefault("_risk_overrides", {})["daily_loss_cap"] = float(dll.params["max_daily_loss_pct"])
        audit["engine_overrides"].append("daily_loss_cap")

    ps = cfg.position_sizing
    if ps.enabled and ps.params.get("risk_per_trade_pct") is not None:
        plan.setdefault("_risk_overrides", {})["per_trade_risk"] = float(ps.params["risk_per_trade_pct"])
        audit["engine_overrides"].append("per_trade_risk")

    if audit["engine_overrides"]:
        logger.info(
            "trade_management|event=plan_integrated|engine_overrides=%s",
            audit["engine_overrides"],
        )
    return audit


# ── Summary rendering ────────────────────────────────────────────────────────


_LABELS = {
    "max_daily_trades":       "Max daily trades (rule 9)",
    "daily_loss_limit":       "Daily loss / consecutive-loss halt (rule 10)",
    "no_reentry_at_level":    "No re-entry at same level (rule 11)",
    "position_sizing":        "Position sizing (rule 12)",
    "partial_profit_booking": "Partial profit booking (rule 13)",
    "trailing_stop_loss":     "Trailing stop loss (rule 14)",
    "manual_exit":            "Manual / time-based exit (rule 15)",
    "setup_grading":          "Setup grading (rule 16)",
    "slippage":               "Slippage tolerance (rule 17)",
}


def render_trade_management_for_summary(cfg: TradeManagementConfig) -> list[str]:
    lines: list[str] = ["── Trade management rules ──"]
    for spec in cfg.all():
        status = "ON" if spec.enabled else "off"
        if spec.required and not spec.enabled:
            status = "NEEDS YOUR INPUT"
        elif spec.enabled and spec.source == "prompt":
            status = "ON (from your prompt)"
        label = _LABELS.get(spec.name, spec.name)
        lines.append(f"{label}: {status}")
        if spec.enabled and spec.description:
            lines.append(f"    └ {spec.description}")
    lines.append(
        "Rules marked NEEDS YOUR INPUT must be supplied before I can build the "
        "strategy — reply with the value (e.g. 'max 3 trades per day') and I "
        "will update the summary."
    )
    return lines


__all__ = [
    "TradeManagementConfig",
    "TradeMgmtRuleSpec",
    "apply_trade_management_to_plan",
    "build_trade_management_config",
    "find_missing_trade_management",
    "render_trade_management_for_summary",
]
