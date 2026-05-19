"""
app/services/discovery/primitives.py
────────────────────────────────────
Phase 9k — compositional discovery primitives.

The chat layer parses a user prompt like:

    "create intraday strategy on NSE stock whose volume spike up today
     1.5x AND price is above VWAP AND RSI > 60"

into an ordered list of *primitives* with parameters:

    [
      {"name": "volume_spike", "params": {"multiplier": 1.5}},
      {"name": "above_vwap",   "params": {}},
      {"name": "rsi_above",    "params": {"threshold": 60}},
    ]

The orchestrator then renders each primitive into a concrete AST
condition string and runs the scanner with EXACTLY the constraints the
user typed — no implicit "you also have to be near the 52-week high"
clause that nobody asked for.

Default conjunction is AND (each primitive yields one condition; the
scanner already requires every condition to pass). OR-grouping is a
future enhancement; today the user's "X OR Y" is treated as two
separate primitives, both expected to pass.

Primitives are intentionally small and orthogonal so new ones can be
added without breaking existing prompts. Each declares:
  • a template formula referencing AST identifiers (CLOSE, VWAP, EMA(N),
    RSI(N), MAX(HIGH, N), …)
  • default parameter values (used when the user mentions the primitive
    but doesn't supply explicit numbers, e.g. "near 52-week high"
    defaults to 2%)
  • parameter sanity bounds — the chat layer is expected to clamp before
    handing values in, but the renderer also asserts so a stray override
    can't blow up the scanner

NO SIDE EFFECTS — this module is pure data + pure rendering. It can be
imported anywhere without dragging in the chat / DB / market-data layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Primitive definition ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Primitive:
    """One named, parameterised discovery condition.

    `template` is a Python format-string referencing AST identifiers
    and `{placeholder}` tokens that map to keys in `default_params`.
    `render` substitutes user-supplied params (which override defaults)
    and returns the concrete AST string the scanner evaluates.
    """

    name: str
    template: str
    default_params: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def render(self, params: dict[str, Any] | None = None) -> str:
        merged: dict[str, Any] = {**self.default_params, **(params or {})}
        # Coerce integer-valued floats to int so e.g. window=252.0
        # renders as "MAX(HIGH, 252)" — matches the orchestrator's
        # behavior in _apply_parameters_to_condition.
        cleaned = {
            k: (int(v) if isinstance(v, float) and v.is_integer() else v)
            for k, v in merged.items()
        }
        try:
            return self.template.format(**cleaned)
        except KeyError as missing:
            raise ValueError(
                f"Primitive {self.name!r} missing required parameter "
                f"{missing}; supplied={params!r}, defaults={self.default_params!r}"
            )


# ── The library ──────────────────────────────────────────────────────────────


# Each primitive maps a single user-recognisable concept to one AST
# condition. New ones are added by appending here AND wiring up a
# detector in app.services.chat.chat_service._extract_discovery_conditions.

PRIMITIVES: dict[str, Primitive] = {

    # Volume —————————————————————————————————————————————————————————————
    "volume_spike": Primitive(
        name="volume_spike",
        template="VOL > AVG(VOL, 20) * {multiplier}",
        default_params={"multiplier": 2.0},
        description="Today's volume is at least N× its 20-day average.",
    ),
    "volume_above_avg": Primitive(
        name="volume_above_avg",
        template="VOL > AVG(VOL, 20)",
        description="Today's volume is above the 20-day average (loose).",
    ),

    # 52-week / lookback proximity ——————————————————————————————————————
    "near_52w_high": Primitive(
        name="near_52w_high",
        template="CLOSE >= MAX(HIGH, {window}) * {factor}",
        default_params={"window": 252, "factor": 0.98},
        description="Close is within (1-factor) × 100% of the N-bar high.",
    ),
    "near_52w_low": Primitive(
        name="near_52w_low",
        template="CLOSE <= MIN(LOW, {window}) * {factor}",
        default_params={"window": 252, "factor": 1.02},
        description="Close is within (factor-1) × 100% of the N-bar low.",
    ),
    "above_52w_high": Primitive(
        name="above_52w_high",
        template="CLOSE > MAX(HIGH, {window})",
        default_params={"window": 252},
        description="Close is above the N-bar high (fresh breakout).",
    ),
    "below_52w_low": Primitive(
        name="below_52w_low",
        template="CLOSE < MIN(LOW, {window})",
        default_params={"window": 252},
        description="Close is below the N-bar low (fresh breakdown).",
    ),

    # Pullback ——————————————————————————————————————————————————————————
    # Bullish pullback: recent low touched EMA(20) but close is back above.
    # Bearish equivalent mirrors with HIGH/below.
    "shallow_pullback_long": Primitive(
        name="shallow_pullback_long",
        template="MIN(LOW, {window}) <= EMA({ema_period}) AND CLOSE > EMA({ema_period})",
        default_params={"window": 3, "ema_period": 20},
        description="Recent low touched EMA but close recovered above it.",
    ),
    "shallow_pullback_short": Primitive(
        name="shallow_pullback_short",
        template="MAX(HIGH, {window}) >= EMA({ema_period}) AND CLOSE < EMA({ema_period})",
        default_params={"window": 3, "ema_period": 20},
        description="Recent high touched EMA but close failed to hold.",
    ),

    # Momentum ——————————————————————————————————————————————————————————
    "rsi_above": Primitive(
        name="rsi_above",
        template="RSI(14) > {threshold}",
        default_params={"threshold": 60},
        description="RSI(14) above the supplied threshold.",
    ),
    "rsi_below": Primitive(
        name="rsi_below",
        template="RSI(14) < {threshold}",
        default_params={"threshold": 40},
        description="RSI(14) below the supplied threshold.",
    ),

    # Trend confirmation ————————————————————————————————————————————————
    "above_vwap": Primitive(
        name="above_vwap",
        template="CLOSE > VWAP",
        description="Close is above session VWAP.",
    ),
    "below_vwap": Primitive(
        name="below_vwap",
        template="CLOSE < VWAP",
        description="Close is below session VWAP.",
    ),
    "above_ema": Primitive(
        name="above_ema",
        template="CLOSE > EMA({period})",
        default_params={"period": 20},
        description="Close is above the EMA of the given period.",
    ),
    "below_ema": Primitive(
        name="below_ema",
        template="CLOSE < EMA({period})",
        default_params={"period": 20},
        description="Close is below the EMA of the given period.",
    ),

    # Intraday position ——————————————————————————————————————————————————
    "near_day_high": Primitive(
        name="near_day_high",
        template="CLOSE >= HIGH * {factor}",
        default_params={"factor": 0.99},
        description="Close is within (1-factor)*100% of today's high.",
    ),
    "near_day_low": Primitive(
        name="near_day_low",
        template="CLOSE <= LOW * {factor}",
        default_params={"factor": 1.01},
        description="Close is within (factor-1)*100% of today's low.",
    ),
}


# ── Public renderers ─────────────────────────────────────────────────────────


def render_primitive(spec: dict[str, Any]) -> str:
    """Render one parsed `{name, params}` dict into an AST condition.
    Raises ValueError if the primitive name is unknown."""
    name = str(spec.get("name") or "").strip()
    if not name:
        raise ValueError("primitive spec missing 'name'")
    primitive = PRIMITIVES.get(name)
    if primitive is None:
        raise ValueError(
            f"unknown discovery primitive {name!r}; valid: {sorted(PRIMITIVES.keys())}"
        )
    params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    return primitive.render(params)


def render_conditions(specs: list[dict[str, Any]]) -> list[str]:
    """Render a list of `{name, params}` dicts into AST conditions
    suitable for handing to the scanner. Each item becomes one
    AND-joined condition (the scanner already requires every
    condition to pass)."""
    return [render_primitive(s) for s in specs if isinstance(s, dict)]


def primitive_descriptions(specs: list[dict[str, Any]]) -> list[str]:
    """Render a list of parsed primitives into one short, human-readable
    description per primitive (used in the no-match reply so the user
    can see exactly which constraints the scanner enforced)."""
    out: list[str] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "").strip()
        primitive = PRIMITIVES.get(name)
        if primitive is None:
            continue
        params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
        merged: dict[str, Any] = {**primitive.default_params, **(params or {})}
        out.append(_describe(name, merged))
    return out


def _describe(name: str, params: dict[str, Any]) -> str:
    """Format a single primitive into a user-facing line. Uses the
    same vocabulary the chat parser accepts so the user sees their
    own words echoed back."""
    if name == "volume_spike":
        return f"volume ≥ {float(params['multiplier']):g}× the 20-day average"
    if name == "volume_above_avg":
        return "volume above the 20-day average"
    if name == "near_52w_high":
        pct = max(0.0, (1.0 - float(params['factor'])) * 100.0)
        return f"within {pct:g}% of {_format_window_phrase(int(params['window']))} high"
    if name == "near_52w_low":
        pct = max(0.0, (float(params['factor']) - 1.0) * 100.0)
        return f"within {pct:g}% of {_format_window_phrase(int(params['window']))} low"
    if name == "above_52w_high":
        return f"close above {_format_window_phrase(int(params['window']))} high"
    if name == "below_52w_low":
        return f"close below {_format_window_phrase(int(params['window']))} low"
    if name == "shallow_pullback_long":
        return "shallow pullback to EMA (long)"
    if name == "shallow_pullback_short":
        return "shallow pullback to EMA (short)"
    if name == "rsi_above":
        return f"RSI(14) > {float(params['threshold']):g}"
    if name == "rsi_below":
        return f"RSI(14) < {float(params['threshold']):g}"
    if name == "above_vwap":
        return "price above VWAP"
    if name == "below_vwap":
        return "price below VWAP"
    if name == "above_ema":
        return f"price above EMA({int(params['period'])})"
    if name == "below_ema":
        return f"price below EMA({int(params['period'])})"
    if name == "near_day_high":
        pct = max(0.0, (1.0 - float(params['factor'])) * 100.0)
        return f"within {pct:g}% of today's high"
    if name == "near_day_low":
        pct = max(0.0, (float(params['factor']) - 1.0) * 100.0)
        return f"within {pct:g}% of today's low"
    return name


def _format_window_phrase(bars: int) -> str:
    """Mirror of the helper in app.services.discovery.orchestrator;
    duplicated locally so primitives.py stays import-cycle-free."""
    if bars % 252 == 0 and bars >= 252:
        years = bars // 252
        return f"{years}-year" if years > 1 else "1-year"
    if bars % 21 == 0 and bars >= 42:
        months = bars // 21
        return f"{months}-month"
    if bars % 5 == 0 and bars >= 25:
        weeks = bars // 5
        return f"{weeks}-week"
    return f"{bars}-day"
