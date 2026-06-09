"""
app/planner/compiler.py — SDL → immutable Strategy Artifact.

Accepts a VALIDATED SDL (validation must pass before calling compile_sdl).
Emits a StrategyArtifact that IS the existing engine input contract —
NOT a new execution graph. The artifact wraps the StrategyConfig fields in
a pydantic model for serialisability, and carries metadata (artifact_id,
version, content_hash).

Responsibilities
────────────────
* Resolve catalog card formulas + params → condition STRINGS.
* Combine multiple conditions per leg: entry = trigger AND filters;
  exit = trigger_1 OR trigger_2 (multiple exit triggers).
* Map SDL risk → engine stop_loss_spec / trailing_stop_spec / take_profit.
* Map SDL gates → engine gate fields.
* Map SDL htf_rules → engine htf_rules list.
* For dynamic universe: carry the discovery_config the scanner consumes.

DOES NOT TOUCH: provenance, clarifications, read-back, LLM fields.
DOES NOT RE-COMPILE CONDITIONS: emits strings; the engine compiles them at run time.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

# ── IST offset for session gate UTC conversion ────────────────────────────────
_IST_OFFSET_MINUTES = 330   # IST = UTC+5:30


def _local_hhmm_to_utc_minutes(hhmm: str, tz: str = "IST") -> int | None:
    """Convert an HH:MM local string to UTC minutes-of-day."""
    m = re.match(r"^(\d{1,2}):(\d{2})$", str(hhmm or "").strip())
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        return None
    if tz.upper() in ("IST", "ASIA/KOLKATA", "ASIA/CALCUTTA"):
        offset = _IST_OFFSET_MINUTES
    elif tz.upper() == "UTC":
        offset = 0
    else:
        offset = _IST_OFFSET_MINUTES  # conservative fallback
    return ((hh * 60 + mm) - offset) % (24 * 60)


# ── Strategy Artifact ─────────────────────────────────────────────────────────

class StrategyArtifact(BaseModel):
    """Immutable, executable strategy — the existing engine's input contract.

    Fields mirror StrategyConfig (quant_engine/engine/loader.py) so the engine
    can consume this directly.  The `discovery_config` block is carried for
    dynamic universes; the evaluation layer resolves the symbol before passing
    to the backtester.

    NO provenance / clarifications / LLM fields here.
    """

    # ── Artifact metadata ─────────────────────────────────────────────────────
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1                       # inherited from the SDL version
    content_hash: str = ""                 # SDL content_hash at compile time

    # ── Core engine fields (StrategyConfig contract) ──────────────────────────
    name: str = ""
    symbol: str = ""                       # resolved; empty for dynamic (scanner fills it)
    market: str = ""
    timeframe: str = ""
    objective: str = "intraday"
    direction: str = "long_only"           # "long_only" | "short_only" | "both"

    # Condition strings — compiled to AST by the engine at run time
    entry_condition: str = ""
    exit_condition: str = ""
    # Phase 12 — SHORT leg conditions for a two-sided ("both") strategy. The
    # engine runs a separate short pass on these; entry_condition/exit_condition
    # then describe the LONG leg only. Empty for single-direction strategies.
    short_entry_condition: str = ""
    short_exit_condition: str = ""

    # Legacy float risk fields (required by StrategyConfig)
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0

    # Rich risk specs (override legacy floats when present)
    stop_loss_spec: dict[str, Any] | None = None
    trailing_stop_spec: dict[str, Any] | None = None

    # HTF rules
    htf_rules: list[dict[str, Any]] = Field(default_factory=list)

    # ── Gate fields ────────────────────────────────────────────────────────────
    vol_filter_metric: str | None = None
    vol_filter_window: int = 14
    vol_filter_min: float | None = None
    vol_filter_max: float | None = None

    regime_filter_allowed: list[str] | None = None

    rs_filter_window: int | None = None
    rs_filter_min_ratio: float = 1.0
    reference_symbol: str | None = None

    event_skip_dates: list[str] | None = None

    entry_window_start_utc: int | None = None
    entry_window_end_utc: int | None = None

    volume_ratio_threshold: float | None = None

    # ── Discovery config (dynamic universe only) ───────────────────────────────
    discovery_config: dict[str, Any] | None = None

    def to_strategy_config_dict(self) -> dict[str, Any]:
        """Return a dict that StrategyConfig(**d) can consume directly."""
        d: dict[str, Any] = {
            "name":            self.name,
            "symbol":          self.symbol,
            "market":          self.market,
            "timeframe":       self.timeframe,
            "objective":       self.objective,
            "entry_condition": self.entry_condition,
            "exit_condition":  self.exit_condition,
            "short_entry_condition": self.short_entry_condition,
            "short_exit_condition":  self.short_exit_condition,
            "indicators":      {},
            "stop_loss":       self.stop_loss_pct,
            "take_profit":     self.take_profit_pct,
            "daily_loss_cap_pct": 5.0,
            "max_trades_per_day": 0,
            "max_holding_candles": None,
            "direction":       self.direction,
        }
        if self.stop_loss_spec:
            d["stop_loss_spec"] = self.stop_loss_spec
        if self.trailing_stop_spec:
            d["trailing_stop_spec"] = self.trailing_stop_spec
        if self.htf_rules:
            d["htf_rules"] = self.htf_rules
        if self.vol_filter_metric:
            d["vol_filter_metric"] = self.vol_filter_metric
            d["vol_filter_window"] = self.vol_filter_window
            d["vol_filter_min"]    = self.vol_filter_min
            d["vol_filter_max"]    = self.vol_filter_max
        if self.regime_filter_allowed is not None:
            d["regime_filter_allowed"] = tuple(self.regime_filter_allowed)
        if self.rs_filter_window is not None:
            d["rs_filter_window"]   = self.rs_filter_window
            d["rs_filter_min_ratio"] = self.rs_filter_min_ratio
            d["reference_symbol"]   = self.reference_symbol
        if self.event_skip_dates is not None:
            d["event_skip_dates"] = tuple(self.event_skip_dates)
        if self.entry_window_start_utc is not None:
            d["entry_window_start_utc"] = self.entry_window_start_utc
        if self.entry_window_end_utc is not None:
            d["entry_window_end_utc"] = self.entry_window_end_utc
        if self.volume_ratio_threshold is not None:
            d["volume_ratio_threshold"] = self.volume_ratio_threshold
        return d


# ── Condition string builder ──────────────────────────────────────────────────

def _resolve_condition(ref, kb) -> str | None:
    """Format a catalog card's formula with the ref's params.

    Returns None if the card is unknown or the formula can't be formatted.
    """
    card = kb.signals.get(ref.name)
    if card is None or not card.formula:
        return None

    # Merge card defaults + user params (user takes precedence)
    by_tf = card.params_by_timeframe or {}
    defaults: dict = {}
    if "15m" in by_tf:
        defaults = dict(by_tf["15m"])
    elif by_tf:
        defaults = dict(next(iter(by_tf.values())))
    merged = {**defaults, **(ref.params or {})}

    try:
        return card.formula.format_map(merged)
    except (KeyError, ValueError):
        return None


def _build_entry_condition(leg, kb) -> str:
    """Build entry_condition: trigger AND filter_1 AND filter_2 ..."""
    parts: list[str] = []

    trigger_cond = _resolve_condition(leg.entry.trigger, kb)
    if trigger_cond:
        parts.append(trigger_cond)

    for flt in leg.entry.filters or []:
        cond = _resolve_condition(flt, kb)
        if cond:
            parts.append(cond)

    return " AND ".join(parts) if parts else ""


def _build_exit_condition(leg, kb) -> str:
    """Build exit_condition: trigger_1 OR trigger_2 ..."""
    parts: list[str] = []
    for trig in leg.exit.triggers or []:
        cond = _resolve_condition(trig, kb)
        if cond:
            parts.append(cond)
    return " OR ".join(parts) if parts else ""


# ── Risk mapping ──────────────────────────────────────────────────────────────

def _map_stop_loss(risk) -> tuple[float, dict | None]:
    """Returns (stop_loss_pct, stop_loss_spec)."""
    sl = risk.stop_loss
    if sl is None:
        return 2.0, None

    if sl.type == "percent":
        pct = sl.value or 2.0
        return pct, {"type": "percent", "pct": pct}

    if sl.type == "atr":
        mult = sl.multiple or 1.5
        win  = sl.window or 14
        return 2.0, {"type": "atr", "multiplier": mult, "window": win}

    # Structural anchors — trust the GATED SDL. The gate (strategy_gate.verify)
    # normalises every structural stop to an engine-legal anchor via the shared
    # engine_contract, so by render time sl.anchor is already engine-safe. We no
    # longer keep a local anchor_map with a silent `opening_range_low` default
    # (that turned vwap_deviation/unknown stops into ORB stops behind the user's
    # back). Resolve via the contract here too, defensively, in case render is
    # reached without the gate (flag off / direct call).
    from app.planner import engine_contract as _ec
    mapped = _ec.map_stop_loss_type(sl.type)
    if isinstance(mapped, dict) and mapped.get("type") == "structural":
        anchor = mapped["anchor"]
    elif sl.anchor in _ec.STOP_LOSS_ANCHORS:
        anchor = sl.anchor
    else:
        # Unmappable and ungated → safe percent fallback (the gate would have
        # turned this into a clarification; never silently invent a structure).
        return (sl.value or 2.0), {"type": "percent", "pct": sl.value or 2.0}
    spec: dict[str, Any] = {"type": "structural", "anchor": anchor}
    if "opening" in anchor:
        spec["opening_bars"] = 3
    else:
        spec["window"] = sl.window or 5
    if sl.padding_atr:
        spec["padding_pct"] = sl.padding_atr
    return 2.0, spec


def _map_take_profit(risk, stop_loss_pct: float) -> float:
    """Returns take_profit_pct."""
    tp = risk.take_profit
    if tp is None:
        return stop_loss_pct * 2.0  # default 2:1

    if tp.type == "percent":
        return tp.value or (stop_loss_pct * 2.0)
    if tp.type == "rr":
        ratio = tp.ratio or 2.0
        return stop_loss_pct * ratio
    if tp.type == "atr":
        return stop_loss_pct * (tp.multiple or 2.0)  # approximation
    return stop_loss_pct * 2.0


def _map_trailing(risk) -> dict | None:
    tr = risk.trailing
    if tr is None or not tr.enabled:
        return None
    spec: dict[str, Any] = {"type": tr.type or "percent"}
    if tr.activate_after_pct:
        spec["activate_after_pct"] = tr.activate_after_pct
    if tr.type == "atr_based":
        spec["type"] = "atr"
        spec["multiplier"] = tr.distance_atr or 2.0
        spec["window"] = 14
    elif tr.type == "percent_based":
        spec["type"] = "percent"
        spec["distance_pct"] = tr.distance_pct or 1.0
    elif tr.type == "ema_based":
        spec["type"] = "ema"
        spec["window"] = tr.ema_period or 20
    elif tr.type == "chandelier":
        spec["multiplier"] = tr.distance_atr or 3.0
        spec["window"] = 14
    return spec


# ── Gate mapping ──────────────────────────────────────────────────────────────

def _map_gates(gates, artifact: StrategyArtifact) -> None:
    """Populate gate fields on the artifact from the SDL GatesSpec."""
    if not gates:
        return

    if gates.regime:
        artifact.regime_filter_allowed = list(gates.regime.allowed)

    if gates.volatility:
        v = gates.volatility
        artifact.vol_filter_metric = str(v.metric or "atr")
        artifact.vol_filter_window = v.window or 14
        artifact.vol_filter_min    = v.min
        artifact.vol_filter_max    = v.max

    if gates.event:
        artifact.event_skip_dates = list(gates.event.skip_dates or [])

    if gates.session:
        s = gates.session
        tz = s.timezone or "IST"
        if s.start:
            artifact.entry_window_start_utc = _local_hhmm_to_utc_minutes(s.start, tz)
        if s.end:
            artifact.entry_window_end_utc = _local_hhmm_to_utc_minutes(s.end, tz)

    if gates.relative_strength:
        rs = gates.relative_strength
        artifact.rs_filter_window   = rs.window
        artifact.rs_filter_min_ratio = rs.min_ratio or 1.0
        artifact.reference_symbol   = rs.reference_symbol

    if gates.volume_ratio:
        vr = gates.volume_ratio
        artifact.volume_ratio_threshold = vr.min_ratio


# ── HTF mapping ───────────────────────────────────────────────────────────────

def _map_htf_rules(htf_rules, kb) -> list[dict]:
    result: list[dict] = []
    # engine supports only the first HTF rule (engine_capability gap is surfaced by validator)
    for rule in htf_rules[:1]:
        cond = rule.condition or ""
        if not cond and rule.signal:
            cond = _resolve_condition(rule.signal, kb) or ""
        if rule.timeframe and cond:
            result.append({"timeframe": rule.timeframe, "condition": cond})
    return result


# ── Discovery config ──────────────────────────────────────────────────────────

def _map_discovery(universe) -> dict | None:
    from app.planner.sdl import DynamicUniverse
    if not isinstance(universe, DynamicUniverse):
        return None
    return {
        "type":       "dynamic",
        "asset_class": str(universe.asset_class),
        "screen":     list(universe.screen or []),
        "rank":       {"by": universe.rank.by, "order": universe.rank.order},
        "tie_break":  universe.tie_break,
    }


# ── Direction inference ───────────────────────────────────────────────────────

def _infer_direction(legs) -> str:
    directions = {leg.direction for leg in legs}
    if directions == {"long"}:
        return "long_only"
    if directions == {"short"}:
        return "short_only"
    return "both"


# ── Primary leg selection (for two-leg SDLs) ──────────────────────────────────

def _primary_leg(legs):
    """Pick the primary (first long, or first) leg for the entry/exit conditions."""
    for leg in legs:
        if leg.direction == "long":
            return leg
    return legs[0]


def _first_leg(legs, direction):
    """First leg matching `direction` ('long'/'short'), or None."""
    for leg in legs:
        if leg.direction == direction:
            return leg
    return None


# ── Objective normalisation ───────────────────────────────────────────────────

def _normalise_objective(objective: str, timeframe: str) -> str:
    o = str(objective or "").lower()
    tf = str(timeframe or "1d").lower()
    intraday_tfs = {"1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h"}
    if o in ("intraday", "scalping", "scalp", "day_trade") or tf in intraday_tfs:
        return "intraday"
    return "positional"


# ── Public entry point ────────────────────────────────────────────────────────

def compile_sdl(sdl, name: str = "") -> StrategyArtifact:
    """Compile a VALIDATED SDL into an immutable StrategyArtifact.

    Does NOT validate — call validate_sdl() first and only call this when ok=True.
    Raises CompileError on logic that should have been caught by validation.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    from app.kb import kb
    from app.planner.sdl import StaticUniverse

    legs = sdl.legs or []
    if not legs:
        raise CompileError("SDL has no legs — cannot compile.")

    _log.info(
        "🏗️  compiler | COMPILE | hash=%.8s | legs=%d | tf=%s | universe=%s",
        sdl.content_hash, len(legs),
        sdl.context.timeframe,
        getattr(sdl.universe, "type", "?"),
    )

    # Phase 12 — two-sided compilation. The long leg drives entry/exit; the
    # short leg (for a "both" strategy) is emitted into short_entry/exit so the
    # engine runs a separate short pass. A short_only strategy puts its single
    # leg in the primary entry/exit fields (direction tells the engine to short).
    direction = _infer_direction(legs)
    long_leg  = _first_leg(legs, "long")
    short_leg = _first_leg(legs, "short")

    short_entry_cond = ""
    short_exit_cond  = ""
    if direction == "short_only":
        primary = short_leg or legs[0]
        entry_cond = _build_entry_condition(primary, kb)
        exit_cond  = _build_exit_condition(primary, kb)
    else:
        primary = long_leg or legs[0]
        entry_cond = _build_entry_condition(primary, kb)
        exit_cond  = _build_exit_condition(primary, kb)
        if direction == "both" and short_leg is not None:
            short_entry_cond = _build_entry_condition(short_leg, kb)
            short_exit_cond  = _build_exit_condition(short_leg, kb)

    # Symbol: static → direct; dynamic → empty (scanner fills it at evaluation time)
    if isinstance(sdl.universe, StaticUniverse):
        symbol = sdl.universe.symbol
    else:
        symbol = ""

    stop_loss_pct, stop_loss_spec = _map_stop_loss(sdl.risk)
    take_profit_pct = _map_take_profit(sdl.risk, stop_loss_pct)
    trailing_spec   = _map_trailing(sdl.risk)
    objective       = _normalise_objective(sdl.context.objective, sdl.context.timeframe)
    htf_rules       = _map_htf_rules(sdl.htf_rules or [], kb)
    discovery_cfg   = _map_discovery(sdl.universe)

    strategy_name = name or (
        f"{symbol or 'dynamic'}_{sdl.context.timeframe}_{sdl.context.objective}"
    )

    artifact = StrategyArtifact(
        version         = sdl.version,
        content_hash    = sdl.content_hash,
        name            = strategy_name,
        symbol          = symbol,
        market          = sdl.context.market,
        timeframe       = sdl.context.timeframe,
        objective       = objective,
        direction       = direction,
        entry_condition = entry_cond,
        exit_condition  = exit_cond,
        short_entry_condition = short_entry_cond,
        short_exit_condition  = short_exit_cond,
        stop_loss_pct   = stop_loss_pct,
        take_profit_pct = take_profit_pct,
        stop_loss_spec  = stop_loss_spec,
        trailing_stop_spec = trailing_spec,
        htf_rules       = htf_rules,
        discovery_config = discovery_cfg,
    )

    _map_gates(sdl.gates, artifact)

    # Active gates summary for logging
    gates = sdl.gates
    active_gates = [
        g for g, v in [
            ("regime", gates.regime), ("volatility", gates.volatility),
            ("event", gates.event), ("session", gates.session),
            ("rs", gates.relative_strength), ("vol_ratio", gates.volume_ratio),
        ] if v
    ]
    sl_desc = (
        f"{sdl.risk.stop_loss.type}={sdl.risk.stop_loss.value or sdl.risk.stop_loss.multiple}"
        if sdl.risk.stop_loss else "none"
    )
    tp_desc = (
        f"{sdl.risk.take_profit.type}={sdl.risk.take_profit.ratio or sdl.risk.take_profit.value}"
        if sdl.risk.take_profit else "none"
    )
    _log.info(
        "📦 compiler | ARTIFACT | id=%.8s | sym=%s | tf=%s | dir=%s | "
        "entry=%.60s | sl=%s | tp=%s | gates=[%s] | htf=%d",
        artifact.artifact_id, artifact.symbol or "dynamic",
        artifact.timeframe, artifact.direction,
        artifact.entry_condition,
        sl_desc, tp_desc,
        ",".join(active_gates) if active_gates else "none",
        len(artifact.htf_rules),
    )
    return artifact


class CompileError(ValueError):
    """Raised when a validated SDL cannot be compiled (should not happen in normal flow)."""
