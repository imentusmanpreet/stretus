"""
app/kb/indicator_catalog.py — Canonical registry of every technical
indicator the system recognises.

This is the single source of truth used by:

  • SemanticExtractor              — to detect indicator names in prompts
  • prompt_summary.extract_threshold_conditions
                                   — to recognise threshold expressions
  • prompt_summary.build_prompt_summary
                                   — to surface every indicator in the summary
  • enforce_zero_loss_capture      — to write `_required_indicators` for
                                     indicators the user mentioned
  • SignalFilterConfig             — to compose filter logic from catalog tokens
  • quant_engine.engine.indicators — to compute precomputed columns
  • quant_engine.engine.conditions — to parse tokens in entry/exit_condition

Each entry carries:

  • name            — canonical, upper-case (e.g. "SUPERTREND").
  • category        — one of trend / momentum / volatility / volume /
                      oscillator / channel / support_resistance / composite.
  • aliases         — strings (and short multi-word phrases) the user might
                      type. Compared after lowercasing + whitespace squash.
  • params          — list of (name, default, kind) per positional argument
                      the parser token expects.
  • outputs         — tuple of column suffixes the engine will produce. For a
                      single-output indicator it's a 1-tuple. For multi-output
                      (e.g. Stochastic K + D) it's a 2-tuple.
  • engine_status   — "implemented" if the engine actually computes it,
                      "pending" if only the chat layer recognises it.
  • token_template  — the parser token form, with {} placeholders for the
                      param values. Examples: "SUPERTREND({period},{multiplier})",
                      "OBV()", "STOCH_K({k_period})".
  • description     — short trader-facing summary used in the prompt summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

CategoryName = Literal[
    "trend",
    "momentum",
    "volatility",
    "volume",
    "oscillator",
    "channel",
    "support_resistance",
    "composite",
]

EngineStatus = Literal["implemented", "pending"]


@dataclass(frozen=True)
class IndicatorParam:
    name:    str
    default: float | int | None
    kind:    Literal["int", "float"] = "int"


@dataclass(frozen=True)
class IndicatorSpec:
    name:           str
    category:       CategoryName
    aliases:        tuple[str, ...]
    params:         tuple[IndicatorParam, ...]
    outputs:        tuple[str, ...]
    engine_status:  EngineStatus
    token_template: str
    description:    str = ""


# ── Catalog ─────────────────────────────────────────────────────────────────────

def _spec(
    name: str,
    *,
    category: CategoryName,
    aliases: Iterable[str] = (),
    params: Iterable[IndicatorParam] = (),
    outputs: Iterable[str] | None = None,
    engine_status: EngineStatus = "pending",
    token_template: str | None = None,
    description: str = "",
) -> IndicatorSpec:
    outs = tuple(outputs) if outputs else (name,)
    if token_template is None:
        if params:
            token_template = name + "(" + ",".join("{" + p.name + "}" for p in params) + ")"
        else:
            token_template = name + "()"
    return IndicatorSpec(
        name=name,
        category=category,
        aliases=tuple(aliases),
        params=tuple(params),
        outputs=outs,
        engine_status=engine_status,
        token_template=token_template,
        description=description,
    )


_PERIOD_14 = IndicatorParam("period", 14, "int")
_PERIOD_20 = IndicatorParam("period", 20, "int")
_PERIOD_10 = IndicatorParam("period", 10, "int")


CATALOG: dict[str, IndicatorSpec] = {}


def _register(spec: IndicatorSpec) -> None:
    CATALOG[spec.name] = spec


# ── TREND ─────────────────────────────────────────────────────────────────────

_register(_spec("SMA",  category="trend", aliases=("sma", "simple moving average"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Simple moving average of close over N bars."))
_register(_spec("EMA",  category="trend", aliases=("ema", "exponential moving average", "exp moving average"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Exponential moving average of close over N bars."))
_register(_spec("MACD", category="trend", aliases=("macd", "moving average convergence divergence"),
                outputs=("MACD", "MACD_SIGNAL", "MACD_HIST"),
                engine_status="implemented", token_template="MACD",
                description="MACD line, signal, and histogram (12,26,9)."))
_register(_spec("ADX",  category="trend", aliases=("adx", "average directional index", "dms", "directional movement system"),
                params=(_PERIOD_14,), outputs=("ADX", "DI_PLUS", "DI_MINUS"),
                engine_status="implemented",
                description="Trend strength (0-100). Above 25 indicates trending market."))
_register(_spec("SUPERTREND", category="trend", aliases=("supertrend", "super trend", "st"),
                params=(IndicatorParam("period", 10, "int"), IndicatorParam("multiplier", 3.0, "float")),
                outputs=("SUPERTREND", "SUPERTREND_DIR"),
                engine_status="implemented",
                description="ATR-based trailing trend follower; direction flips signal reversal."))
_register(_spec("ICHIMOKU", category="trend", aliases=("ichimoku", "ichimoku cloud", "ichimoku clouds", "kumo"),
                params=(IndicatorParam("tenkan", 9, "int"), IndicatorParam("kijun", 26, "int"), IndicatorParam("senkou", 52, "int")),
                outputs=("ICHIMOKU_CONV", "ICHIMOKU_BASE", "ICHIMOKU_SPAN_A", "ICHIMOKU_SPAN_B"),
                engine_status="pending",
                description="Cloud / equilibrium system — Tenkan, Kijun, Senkou A/B."))
_register(_spec("PSAR", category="trend", aliases=("psar", "parabolic sar", "sar"),
                params=(IndicatorParam("acceleration", 0.02, "float"), IndicatorParam("max_acceleration", 0.2, "float")),
                outputs=("PSAR",),
                engine_status="implemented",
                description="Parabolic stop-and-reverse; flip indicates trend change."))
_register(_spec("DONCHIAN", category="channel", aliases=("donchian", "donchian channel"),
                params=(_PERIOD_20,), outputs=("DON_UPPER", "DON_LOWER", "DON_MID"),
                engine_status="implemented",
                description="Channel of highest-high and lowest-low over N bars."))
_register(_spec("KELTNER", category="channel", aliases=("keltner", "keltner channel"),
                params=(IndicatorParam("period", 20, "int"), IndicatorParam("atr_mult", 2.0, "float")),
                outputs=("KC_UPPER", "KC_LOWER", "KC_MID"),
                engine_status="implemented",
                description="EMA + ATR-based channel."))
_register(_spec("DARVAS", category="support_resistance", aliases=("darvas", "darvas box"),
                params=(IndicatorParam("period", 20, "int"),),
                outputs=("DARVAS_TOP", "DARVAS_BOTTOM"),
                engine_status="pending",
                description="Darvas box: consolidation range upper / lower."))
_register(_spec("LINREG", category="trend", aliases=("linear regression", "linear regression forecast"),
                params=(_PERIOD_14,), outputs=("LINREG",),
                engine_status="pending",
                description="Linear regression forecast value."))
_register(_spec("LINREG_SLOPE", category="trend", aliases=("linear regression slope", "linreg slope"),
                params=(_PERIOD_14,), outputs=("LINREG_SLOPE",),
                engine_status="pending",
                description="Slope of linear regression line."))
_register(_spec("RAINBOW_MA", category="trend", aliases=("rainbow moving average", "rainbow ma"),
                params=(_PERIOD_10,), outputs=("RAINBOW_MA",),
                engine_status="pending",
                description="Multi-period MA stack."))
_register(_spec("BB_UPPER", category="channel", aliases=("bollinger upper",),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Bollinger Band upper (SMA + 2σ)."))
_register(_spec("BB_LOWER", category="channel", aliases=("bollinger lower",),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Bollinger Band lower (SMA − 2σ)."))
_register(_spec("BB_MID", category="channel", aliases=("bollinger middle", "bollinger mid"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Bollinger Band middle line (SMA)."))
_register(_spec("BB_WIDTH", category="volatility", aliases=("bollinger bandwidth", "bollinger width", "bb bandwidth"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="(Upper − Lower) / Mid — volatility expansion indicator."))
_register(_spec("BB_PCT_B", category="oscillator", aliases=("bollinger %b", "bb %b", "percent b"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Position of close inside the Bollinger band, 0-1."))
_register(_spec("ROC", category="momentum", aliases=("roc", "rate of change", "price rate of change"),
                params=(IndicatorParam("period", 10, "int"),), engine_status="implemented",
                description="Percentage change vs N bars ago."))
_register(_spec("DISPARITY", category="trend", aliases=("disparity index", "disparity"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="(Close − SMA) / SMA × 100."))
_register(_spec("TRIX", category="momentum", aliases=("trix",),
                params=(IndicatorParam("period", 15, "int"),), engine_status="implemented",
                description="Rate of change of triple-smoothed EMA."))
_register(_spec("TII", category="trend", aliases=("trend intensity index", "tii"),
                params=(_PERIOD_20,), engine_status="pending",
                description="Trend intensity index."))
_register(_spec("DMI", category="trend", aliases=("dmi", "directional movement index"),
                params=(_PERIOD_14,), outputs=("DI_PLUS", "DI_MINUS"),
                engine_status="implemented",
                description="+DI and −DI components of the DMS."))

# ── MOMENTUM ────────────────────────────────────────────────────────────────────

_register(_spec("RSI", category="momentum", aliases=("rsi", "relative strength index"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Relative Strength Index (Wilder), 0-100."))
_register(_spec("STOCH_K", category="momentum",
                aliases=("stochastic", "stoch", "stochastic oscillator", "stoch k", "stochastic k", "%k"),
                params=(IndicatorParam("k_period", 14, "int"),),
                engine_status="implemented",
                description="Stochastic %K, 0-100."))
_register(_spec("STOCH_D", category="momentum",
                aliases=("stoch d", "stochastic signal", "stoch signal", "stochastic d", "%d"),
                params=(IndicatorParam("d_period", 3, "int"),),
                engine_status="implemented",
                description="Stochastic %D — SMA of %K."))
_register(_spec("STOCH_MOM", category="momentum", aliases=("stochastic momentum index", "smi"),
                params=(_PERIOD_14,), engine_status="pending",
                description="Stochastic momentum index, −100 to 100."))
_register(_spec("MOMENTUM", category="momentum", aliases=("momentum indicator", "mom"),
                params=(_PERIOD_10,), engine_status="implemented",
                description="Close − Close[N bars ago]."))
_register(_spec("CMO", category="momentum", aliases=("chande momentum oscillator", "cmo"),
                params=(IndicatorParam("period", 9, "int"),), engine_status="implemented",
                description="Chande Momentum Oscillator, −100 to 100."))
_register(_spec("PMO", category="momentum", aliases=("price momentum oscillator", "pmo"),
                params=(IndicatorParam("period", 35, "int"),), engine_status="pending",
                description="Smoothed momentum oscillator."))
_register(_spec("UO", category="momentum", aliases=("ultimate oscillator",),
                params=(IndicatorParam("short", 7, "int"), IndicatorParam("mid", 14, "int"), IndicatorParam("long", 28, "int")),
                engine_status="pending",
                description="Ultimate oscillator combining three timeframes."))
_register(_spec("WILLR", category="momentum", aliases=("williams %r", "williams percent r", "willr", "%r"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Williams %R, −100 to 0."))
_register(_spec("IMI", category="momentum", aliases=("intraday momentum index", "imi"),
                params=(_PERIOD_14,), engine_status="pending",
                description="Intraday momentum index."))
_register(_spec("RVI", category="momentum", aliases=("relative vigor index", "rvi"),
                params=(_PERIOD_10,), engine_status="pending",
                description="Relative vigor index."))
_register(_spec("AO", category="momentum", aliases=("awesome oscillator", "ao"),
                params=(), outputs=("AO",), token_template="AO()",
                engine_status="pending",
                description="Awesome oscillator: SMA(median,5) − SMA(median,34)."))
_register(_spec("QSTICK", category="momentum", aliases=("qstick",),
                params=(_PERIOD_14,), engine_status="pending",
                description="Smoothed open-vs-close momentum."))
_register(_spec("STC", category="momentum", aliases=("schaff trend cycle", "stc"),
                params=(IndicatorParam("period", 23, "int"),), engine_status="pending",
                description="Cyclic MACD."))
_register(_spec("CCI", category="momentum", aliases=("cci", "commodity channel index"),
                params=(IndicatorParam("period", 20, "int"),), engine_status="implemented",
                description="Commodity channel index, ±100 lines."))

# ── VOLATILITY ─────────────────────────────────────────────────────────────────

_register(_spec("ATR", category="volatility", aliases=("atr", "average true range"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Wilder's average true range — volatility in price units."))
_register(_spec("ATR_UPPER", category="channel", aliases=("atr upper band", "atr band upper"),
                params=(IndicatorParam("period", 14, "int"), IndicatorParam("atr_mult", 2.0, "float")),
                engine_status="implemented",
                description="Close + atr_mult × ATR."))
_register(_spec("ATR_LOWER", category="channel", aliases=("atr lower band", "atr band lower"),
                params=(IndicatorParam("period", 14, "int"), IndicatorParam("atr_mult", 2.0, "float")),
                engine_status="implemented",
                description="Close − atr_mult × ATR."))
_register(_spec("STDEV", category="volatility", aliases=("standard deviation", "stdev", "std dev"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Rolling standard deviation of close."))
_register(_spec("HV", category="volatility", aliases=("historical volatility", "annualized volatility"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Annualised stdev of log returns (×√252)."))
_register(_spec("CHAIKIN_VOL", category="volatility", aliases=("chaikin volatility",),
                params=(_PERIOD_10,), engine_status="pending",
                description="Rate of change of EMA(high − low)."))
_register(_spec("CHOPPINESS", category="volatility", aliases=("choppiness", "choppiness index"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="0-100: high values mean choppy/sideways."))
_register(_spec("STARC_UPPER", category="channel", aliases=("starc upper",),
                params=(IndicatorParam("period", 15, "int"), IndicatorParam("atr_mult", 2.0, "float")),
                engine_status="pending"))
_register(_spec("STARC_LOWER", category="channel", aliases=("starc lower",),
                params=(IndicatorParam("period", 15, "int"), IndicatorParam("atr_mult", 2.0, "float")),
                engine_status="pending"))

# ── VOLUME ─────────────────────────────────────────────────────────────────────

_register(_spec("VOLUME", category="volume", aliases=("volume", "vol"),
                params=(), engine_status="implemented", token_template="VOLUME",
                description="Bar volume."))
_register(_spec("VOLUME_SMA", category="volume", aliases=("volume sma", "average volume", "volume average"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="SMA of volume over N bars."))
_register(_spec("VOLUME_OSC", category="volume", aliases=("volume oscillator",),
                params=(IndicatorParam("short", 5, "int"), IndicatorParam("long", 20, "int")),
                engine_status="pending",
                description="Difference of two volume EMAs, as % of long EMA."))
_register(_spec("VROC", category="volume", aliases=("volume rate of change", "vroc"),
                params=(IndicatorParam("period", 12, "int"),), engine_status="implemented",
                description="Percentage change in volume vs N bars ago."))
_register(_spec("OBV", category="volume", aliases=("obv", "on balance volume", "on-balance volume"),
                params=(), token_template="OBV", engine_status="implemented",
                description="On-balance volume: running cumulative of signed volume."))
_register(_spec("ACCDIST", category="volume", aliases=("accumulation distribution", "a/d", "accumulation/distribution"),
                params=(), token_template="ACCDIST", engine_status="implemented",
                description="Accumulation/distribution line."))
_register(_spec("CMF", category="volume", aliases=("chaikin money flow", "cmf"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Volume-weighted accumulation/distribution, −1..+1."))
_register(_spec("MFI", category="volume", aliases=("mfi", "money flow index"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Volume-weighted RSI, 0-100."))
_register(_spec("VWAP", category="volume", aliases=("vwap", "volume weighted average price"),
                params=(), token_template="VWAP", engine_status="implemented",
                description="Session volume-weighted average price."))
_register(_spec("KVO", category="volume", aliases=("klinger volume oscillator", "klinger"),
                params=(IndicatorParam("short", 34, "int"), IndicatorParam("long", 55, "int")),
                engine_status="pending"))
_register(_spec("TVI", category="volume", aliases=("trade volume index", "tvi"),
                params=(), engine_status="pending"))
_register(_spec("PVT", category="volume", aliases=("price volume trend", "pvt"),
                params=(), engine_status="pending"))
_register(_spec("EOM", category="volume", aliases=("ease of movement", "eom"),
                params=(_PERIOD_14,), engine_status="pending"))
_register(_spec("MFAC", category="volume", aliases=("market facilitation index", "mfac"),
                params=(), engine_status="pending"))
_register(_spec("EFI", category="volume", aliases=("elder force index", "efi"),
                params=(IndicatorParam("period", 13, "int"),),
                engine_status="pending"))
_register(_spec("CVD", category="volume", aliases=("cumulative volume delta", "cvd"),
                params=(), engine_status="pending"))
_register(_spec("TMF", category="volume", aliases=("twiggs money flow", "tmf"),
                params=(IndicatorParam("period", 21, "int"),), engine_status="pending"))
_register(_spec("PVI", category="volume", aliases=("positive volume index", "pvi"),
                params=(), engine_status="pending"))
_register(_spec("NVI", category="volume", aliases=("negative volume index", "nvi"),
                params=(), engine_status="pending"))

# ── OSCILLATORS ────────────────────────────────────────────────────────────────

_register(_spec("APO", category="oscillator", aliases=("absolute price oscillator", "apo"),
                params=(IndicatorParam("short", 12, "int"), IndicatorParam("long", 26, "int")),
                engine_status="pending"))
_register(_spec("AROON_UP", category="oscillator", aliases=("aroon up",),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Aroon Up: distance since last N-bar high."))
_register(_spec("AROON_DOWN", category="oscillator", aliases=("aroon down",),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Aroon Down: distance since last N-bar low."))
_register(_spec("AROON_OSC", category="oscillator", aliases=("aroon oscillator", "aroon osc"),
                params=(_PERIOD_14,), engine_status="implemented",
                description="Aroon Up − Aroon Down."))
_register(_spec("BOP", category="oscillator", aliases=("balance of power", "bop"),
                params=(), engine_status="pending"))
_register(_spec("DPO", category="oscillator", aliases=("detrended price oscillator", "dpo"),
                params=(_PERIOD_20,), engine_status="pending"))

# ── SUPPORT / RESISTANCE ──────────────────────────────────────────────────────

_register(_spec("PIVOT", category="support_resistance", aliases=("pivot", "pivot points", "daily pivot"),
                params=(), outputs=("PIVOT", "R1", "R2", "S1", "S2"),
                engine_status="implemented", token_template="PIVOT",
                description="Daily floor pivot + R1/R2 and S1/S2."))
_register(_spec("HHV", category="support_resistance", aliases=("highest high value", "hhv"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Highest high over N bars."))
_register(_spec("LLV", category="support_resistance", aliases=("lowest low value", "llv"),
                params=(_PERIOD_20,), engine_status="implemented",
                description="Lowest low over N bars."))

# ── COMPOSITE ─────────────────────────────────────────────────────────────────

_register(_spec("MEDIAN_PRICE", category="composite", aliases=("median price",),
                params=(), token_template="MEDIAN_PRICE", engine_status="implemented",
                description="(High + Low) / 2."))
_register(_spec("TYPICAL_PRICE", category="composite", aliases=("typical price",),
                params=(), token_template="TYPICAL_PRICE", engine_status="implemented",
                description="(High + Low + Close) / 3."))
_register(_spec("WEIGHTED_CLOSE", category="composite", aliases=("weighted close",),
                params=(), token_template="WEIGHTED_CLOSE", engine_status="implemented",
                description="(High + Low + 2×Close) / 4."))
_register(_spec("HIGH_LOW", category="composite", aliases=("high minus low", "high-low", "range"),
                params=(), token_template="HIGH_LOW", engine_status="implemented",
                description="High − Low (true bar range)."))
_register(_spec("TRUE_RANGE", category="composite", aliases=("true range", "tr"),
                params=(), token_template="TRUE_RANGE", engine_status="implemented",
                description="True range (max of HL / |H−prevC| / |L−prevC|)."))
_register(_spec("CORR", category="composite", aliases=("correlation coefficient", "correlation"),
                params=(_PERIOD_20,), engine_status="pending"))
_register(_spec("ULCER", category="composite", aliases=("ulcer index", "ulcer"),
                params=(_PERIOD_14,), engine_status="pending"))
_register(_spec("VORTEX_PLUS", category="composite", aliases=("vortex positive", "vi+"),
                params=(_PERIOD_14,), engine_status="pending"))
_register(_spec("VORTEX_MINUS", category="composite", aliases=("vortex negative", "vi-"),
                params=(_PERIOD_14,), engine_status="pending"))


# ── Lookup helpers ──────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text or "").strip().lower()


def lookup_by_alias(text: str) -> Optional[IndicatorSpec]:
    """Return the IndicatorSpec whose canonical name or alias matches `text`.
    Comparison is case-insensitive and whitespace-normalised."""
    key = _normalise(text)
    if not key:
        return None
    for spec in CATALOG.values():
        if spec.name.lower() == key:
            return spec
        for alias in spec.aliases:
            if alias == key:
                return spec
    return None


def all_aliases() -> dict[str, IndicatorSpec]:
    """Flat alias-to-spec map (used by extractors for tight regex unions)."""
    out: dict[str, IndicatorSpec] = {}
    for spec in CATALOG.values():
        out[spec.name.lower()] = spec
        for alias in spec.aliases:
            out[alias] = spec
    return out


def implemented_indicators() -> list[IndicatorSpec]:
    return [s for s in CATALOG.values() if s.engine_status == "implemented"]


def pending_indicators() -> list[IndicatorSpec]:
    return [s for s in CATALOG.values() if s.engine_status == "pending"]


def by_category() -> dict[CategoryName, list[IndicatorSpec]]:
    out: dict[CategoryName, list[IndicatorSpec]] = {}
    for spec in CATALOG.values():
        out.setdefault(spec.category, []).append(spec)
    return out


def format_token(name: str, params: dict[str, float | int] | None = None) -> str:
    """Render a parser token for `name` using `params`, falling back to each
    spec's default param value when a value isn't supplied."""
    spec = CATALOG.get(name.upper())
    if spec is None:
        return name.upper()
    if not spec.params:
        return spec.token_template
    values: dict[str, str] = {}
    for p in spec.params:
        v = (params or {}).get(p.name, p.default)
        if v is None:
            v = 0
        if p.kind == "int":
            values[p.name] = str(int(v))
        else:
            values[p.name] = str(float(v))
    try:
        return spec.token_template.format(**values)
    except KeyError:
        return spec.token_template


__all__ = [
    "CATALOG",
    "IndicatorParam",
    "IndicatorSpec",
    "all_aliases",
    "by_category",
    "format_token",
    "implemented_indicators",
    "lookup_by_alias",
    "pending_indicators",
]
