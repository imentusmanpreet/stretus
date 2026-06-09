"""
app/services/strategy/risk_defaults.py — the ONE resolver for default risk
parameters (WS1 §6 / design §3a).

Before this module the SL/TP default literals were scattered and inconsistent:
`0.25` (builder.apply_defaults), `2.0` (builder._reward_factor), `1.5`
(compat market config, chat_service), `0.25` (risk_manager). A user who never
stated a stop saw different numbers depending on which code path rendered them.

This module is the single home for those defaults. Each call returns a
`DefaultValue` carrying both the number AND its provenance source ("default"),
so callers can:
  * tag `risk_execution_config["rms_sources"]` honestly, and
  * surface the value for optional confirmation instead of treating it as
    silently authoritative (design §3a / §5d).

Resolution is objective/experience-aware: the experience tier's `base_sl_pct`
wins when set; otherwise an objective-shaped fallback is used (intraday tighter,
positional wider). TP defaults to SL × the default risk:reward.
"""
from __future__ import annotations

from dataclasses import dataclass

# Default risk:reward used to derive a take-profit when only the stop is known.
DEFAULT_RISK_REWARD: float = 2.0

# Objective-shaped stop-loss fallbacks (percent), used only when the experience
# tier carries no base_sl_pct. Intraday stops are tight; positional wider.
_OBJECTIVE_SL_PCT: dict[str, float] = {
    "intraday": 1.0,
    "positional": 2.0,
    "swing": 2.0,
}
_FALLBACK_SL_PCT: float = 1.5


@dataclass(frozen=True)
class DefaultValue:
    """A defaulted value plus honest provenance."""

    value: float
    source: str = "default"
    reason: str = ""


def _tier_base(experience: str | None, attr: str) -> float | None:
    """Return a truthy tier base value (base_sl_pct / base_tp_pct) or None."""
    try:
        from app.kb import kb
        tier = kb.get_risk_tier(experience)
    except Exception:
        return None
    val = getattr(tier, attr, None)
    return float(val) if val else None


def resolve_default_stop_loss(
    objective: str | None = None,
    experience: str | None = None,
) -> DefaultValue:
    """Default stop-loss percent for a user who didn't state one."""
    base = _tier_base(experience, "base_sl_pct")
    if base is not None:
        return DefaultValue(base, reason=f"tier:{(experience or 'intermediate')}")
    obj = (objective or "").lower().strip()
    val = _OBJECTIVE_SL_PCT.get(obj, _FALLBACK_SL_PCT)
    return DefaultValue(val, reason=f"objective:{obj or 'default'}")


def resolve_default_take_profit(
    objective: str | None = None,
    experience: str | None = None,
    *,
    stop_loss: float | None = None,
    risk_reward: float = DEFAULT_RISK_REWARD,
) -> DefaultValue:
    """Default take-profit percent. Prefers the tier base; otherwise derives
    from the (resolved) stop-loss × the default risk:reward."""
    base = _tier_base(experience, "base_tp_pct")
    if base is not None:
        return DefaultValue(base, reason=f"tier:{(experience or 'intermediate')}")
    sl = stop_loss if stop_loss is not None else resolve_default_stop_loss(objective, experience).value
    return DefaultValue(round(sl * risk_reward, 2), reason=f"rr:{risk_reward:g}")
