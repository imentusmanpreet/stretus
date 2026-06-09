"""
app/planner/sdl_selector.py — SDL Selector: the one AI call per turn.

Two public entry points, both guarded by SDL_SELECTOR_ENABLED:
  compile_to_sdl(prompt, menu)              → SDL   (CREATE turn)
  modify_sdl(existing_sdl, change_msg, menu) → SDL   (MODIFY turn)

Guardrails (enforced in the system prompt)
──────────────────────────────────────────
1. Every prompt clause maps to an SDL field OR unmapped_details — never neither.
2. EXPLICIT INDICATOR WINS: "Bollinger" → Bollinger card, not Keltner/sibling.
3. Unsupported → unmapped_details (never guessed). Universe asks beyond the
   scanner → unsupported_universe / engine_capability_gap.
4. Missing required value → pick default, record provenance=default, add
   a clarification.

Caching
───────
Cache key = SHA-256(prompt_or_sdl_hash | catalog_version | selector_version).
The same input always yields the same SDL (deterministic for test golden tests).
Cache is populated by the LLM call; in tests it is pre-seeded or the LLM is
mocked via set_llm_override().

Modify turn
───────────
modify_sdl merges the LLM's diff into the existing SDL: only fields that
changed are updated; changed fields get source="user" in provenance.
version++ and parent_version set automatically.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable

from app.planner.catalog_schema import CatalogMenu, build_menu, format_menu_for_llm
from app.planner.sdl import (
    SDL,
    ClarificationNeeded,
    DynamicRank,
    DynamicUniverse,
    EntrySpec,
    ExitSpec,
    FieldSource,
    GatesSpec,
    HTFRule,
    Leg,
    Provenance,
    RegimeGate,
    RelativeStrengthGate,
    RiskSpec,
    ScaleOutSpec,
    SessionGate,
    SignalRef,
    SizingSpec,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    TrailingSpec,
    UnmappedDetail,
    VolatilityGate,
    VolumeRatioGate,
)

logger = logging.getLogger(__name__)

_SELECTOR_VERSION = "0.2.0"

# ── Cache ─────────────────────────────────────────────────────────────────────

_cache: dict[str, str] = {}   # cache_key → SDL JSON string


def _cache_key(prompt_or_hash: str, catalog_version: str) -> str:
    payload = f"{prompt_or_hash}|{catalog_version}|{_SELECTOR_VERSION}"
    return hashlib.sha256(payload.encode()).hexdigest()


def clear_cache() -> None:
    _cache.clear()


# ── LLM override (for testing) ────────────────────────────────────────────────

_llm_override: Callable | None = None


def set_llm_override(fn: Callable | None) -> None:
    """Inject a synchronous mock: fn(messages, tools) -> sdl_json_str."""
    global _llm_override
    _llm_override = fn


# ── Tool schema for structured output ─────────────────────────────────────────

_SDL_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_sdl",
        "description": (
            "Submit the complete SDL ticket as a JSON string. "
            "Call this once with the fully filled SDL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sdl_json": {
                    "type": "string",
                    "description": "JSON-serialised SDL ticket (matches SDL schema).",
                }
            },
            "required": ["sdl_json"],
        },
    },
}


# ── System prompt ─────────────────────────────────────────────────────────────

def _enum_reference() -> str:
    """The EXACT allowed value of every SDL enum, generated from the schema so the
    prompt can never drift from the model. This is WHY the LLM stops guessing
    words like 'crypto'/'structural': we tell it the precise literals + how common
    trader phrases map onto them."""
    from typing import get_args
    from app.planner import sdl as _s
    j = lambda lit: " | ".join(get_args(lit))
    return (
        f"  universe.asset_class    : {j(_s.AssetClass)}\n"
        f"      crypto / ETH-BTC pairs → crypto_spot ;  NSE / Indian stocks → equity_cash\n"
        f"  legs[].direction        : long | short    (two legs when the user gives both)\n"
        f"  risk.stop_loss.type     : {j(_s.StopLossType)}\n"
        f"      'recent swing high/low' → opposite_side ;  'X% stop' → percent ;\n"
        f"      'N ATR' / 'N ATR stop' / 'N x ATR' → atr WITH multiple=N (e.g. '1 ATR' → multiple=1.0, '1.5x ATR' → multiple=1.5) ;\n"
        f"      'below swing low' → swing_low ;  'above swing high' → swing_high\n"
        f"  risk.take_profit.type   : {j(_s.TakeProfitType)}\n"
        f"      '2:1 risk-reward' → rr (ratio=2.0) ;  'X% target' → percent ;  'N ATR target' → atr\n"
        f"  risk.trailing.type      : {j(_s.TrailingType)}\n"
        f"      'trail with 20 EMA' → ema_based ;  'ATR trail' → atr_based ;  '% trail' → percent_based\n"
        f"  risk.sizing.mode        : {j(_s.SizingMode)}\n"
        f"  gates.volatility.metric : atr | natr\n"
        f"  universe.rank.by        : {j(_s.RankMetric)}\n"
    )


_SYSTEM_PROMPT_TEMPLATE = """You are the SDL Selector — a strategy-definition assistant.
Your job: read a user's trading-strategy description and fill out an SDL ticket
by choosing items ONLY from the catalog below.  You never invent formulas,
indicator names, or conditions not in the catalog.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EVERY prompt clause → a specific SDL field OR provenance.unmapped_details.
   Nothing is silently dropped or invented.

2. EXPLICIT INDICATOR WINS.
   • User says "Bollinger" → MUST use a bb_* card.  Never substitute Keltner/Donchian.
   • User says "RSI 14 below 30" → name="rsi_oversold", params.window=14, params.threshold=30.
   • User says "EMA(9) cross EMA(21)" → name="ema_cross_up", params.window_fast=9, window_slow=21.
   • "price above/below <INDICATOR>" → use THAT indicator's own card, never a
     different indicator. "price above Supertrend" → supertrend_bullish ;
     "price below Supertrend" → supertrend_bearish ; "price above the 20 EMA"
     → price_above_ema.  NEVER map "above/below Supertrend/VWAP/Keltner/…" to an
     SMA/EMA card just because the catalog has a price_above_* family.
   • SAME CONDITION STATED TWICE → ONE card. "Supertrend flips bullish" AND
     "price closes above Supertrend" are the SAME thing (supertrend_bullish's
     formula is CLOSE > SUPERTREND) — emit it ONCE as the trigger; do NOT add a
     redundant filter for the restatement.
   • Use EXACTLY the catalog name — never invent new signal names. If unsure a
     name exists, it doesn't: pick a real menu name or record missing_card.

3. Unsupported → unmapped_details (never guessed):
     missing_card           — no catalog card matches the concept
     missing_operator       — comparison the engine can't express
     engine_capability_gap  — engine can't run this today
     unsupported_universe   — dynamic universe ask beyond the scanner

4. Missing money value (SL, TP, RR) → MUST:
   (a) pick a sensible default (2% SL, 2:1 RR)
   (b) set field_sources source="default" for that field
   (c) add a clarification_needed entry asking the user to confirm.
   NEVER silently default a money field without a clarification.

5. PROVENANCE — field_sources MUST be exhaustive.  Use these exact paths:

   ALWAYS include these paths when the user stated them:
     "context.timeframe"                        ← user said "15m", "1h", "daily"
     "context.objective"                        ← user said "breakout", "scalping", etc.
     "universe.symbol"                          ← user named a stock/crypto
     "universe.asset_class"                     ← implied by the asset type
     "legs.0.direction"                         ← user said "buy/long/short/sell"
     "legs.0.entry.trigger.name"                ← user named an indicator/signal
     "legs.0.entry.trigger.params.window"       ← user gave the period (e.g. RSI 14)
     "legs.0.entry.trigger.params.threshold"    ← user gave the level (e.g. RSI < 30)
     "legs.0.entry.filters.0"                   ← user added a filter
     "legs.0.exit.triggers.0"                   ← user specified exit signal
     "risk.stop_loss"                           ← user gave a stop loss
     "risk.stop_loss.type"                      ← see ENUM VALUES (swing high/low → opposite_side)
     "risk.stop_loss.value"                     ← the numeric value (if percent)
     "risk.stop_loss.multiple"                  ← the ATR multiple (if atr)
     "risk.take_profit"                         ← user gave TP
     "risk.take_profit.type"                    ← see ENUM VALUES (2:1 RR → rr)
     "risk.take_profit.ratio"                   ← the RR ratio (if rr)
     "risk.take_profit.value"                   ← the % value (if percent)
     "risk.trailing"                            ← user said "trailing stop"
     "gates.regime"                             ← user said "trending regime"
     "gates.event"                              ← user said "skip earnings dates"
     "gates.session"                            ← user gave entry window times
     "gates.volatility"                         ← user asked for volatility gate
     "gates.relative_strength"                  ← user asked for RS filter
     "gates.volume_ratio"                       ← user asked for volume filter
     "htf_rules.0"                              ← user gave HTF condition

   source values:
     "user"     — the user EXPLICITLY stated this value
     "default"  — we filled it in (user did not state it) → MUST add clarification
     "inferred" — we deduced it from context (e.g. "buy" implies direction=long)

6. Static universe = ONE named symbol only.  No multi-asset lists.
7. Dynamic universe: rank.by must be one of the catalog's rank_metrics.
   tie_break must be a valid tie_break id.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CATALOG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{catalog_text}

━━━━━━━━━ ENUM VALUES — use EXACTLY one of these literals, never your own word ━━━━━━━━━
{enum_reference}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ SDL SCHEMA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return a JSON object:
  context   : {{market, timeframe, objective}}
  universe  : {{type:"static", asset_class, symbol}} OR
              {{type:"dynamic", asset_class, screen:[...], rank:{{by,order}}, tie_break}}
  legs      : [ {{direction:"long"|"short",
                  entry:{{trigger:{{name,params,intent}}, filters:[{{name,params,intent}}]}},
                  exit:{{triggers:[{{name,params}}], filters:[]}}}} ]

  intent (per ENTRY signal — the verifier uses ONLY this to catch a mis-map,
  never your prose): {{"user_span":"<verbatim words this signal came from>",
  "intended":{{"lhs":"price|close|<indicator>","rhs":"price|close|<indicator>",
  "op":"gt|lt|cross_up|cross_down|eq"}}}}.
  Example: user "9 EMA above close" → trigger may be price_below_ema, but
  intent={{"user_span":"9 EMA above close","intended":{{"lhs":"ema","rhs":"close","op":"gt"}}}}.
  risk      : {{stop_loss:{{type,value?,multiple?,window?,anchor?}},
               take_profit:{{type,ratio?,value?,multiple?,window?}},
               trailing?:{{enabled,type?,distance_atr?,distance_pct?,ema_period?}},
               scale_outs?:[{{at_rr,size_pct}}],
               sizing?:{{mode,risk_pct?}}}}
  gates     : {{regime?:{{allowed:[...]}}, volatility?:{{metric,window,min?,max?}},
               event?:{{skip_dates:[...]}}, session?:{{start,end,timezone}},
               relative_strength?:{{window,min_ratio,reference_symbol?}},
               volume_ratio?:{{window,min_ratio}}}}
  htf_rules : [{{timeframe, condition?, signal?:{{name,params}}, role}}]
  provenance: {{
    field_sources: {{"dotted.path.here": "user"|"default"|"inferred"}},
    unmapped_details: [{{"text":"exact user words", "kind":"missing_card|...", "note":""}}],
    clarifications_needed: [{{"field":"risk.take_profit", "question":"No TP given — 2:1?", "assumed_value":"2:1"}}]
  }}

EXAMPLE field_sources for "Buy ETH 15m when RSI 14 below 30, stop 2%, TP 2:1":
  {{
    "context.timeframe": "user",
    "universe.symbol": "user",
    "universe.asset_class": "user",
    "legs.0.direction": "user",
    "legs.0.entry.trigger.name": "user",
    "legs.0.entry.trigger.params.window": "user",
    "legs.0.entry.trigger.params.threshold": "user",
    "legs.0.exit.triggers.0": "inferred",
    "risk.stop_loss": "user",
    "risk.stop_loss.type": "user",
    "risk.stop_loss.value": "user",
    "risk.take_profit": "user",
    "risk.take_profit.type": "user",
    "risk.take_profit.ratio": "user"
  }}

Call the submit_sdl tool with the filled SDL as a JSON string.
"""

_MODIFY_PROMPT_TEMPLATE = """The user wants to MODIFY an existing strategy.
Apply ONLY the changes described.  Leave all other fields unchanged.
Changed fields MUST have source="user" in field_sources.
Increment version and set parent_version in the returned SDL.

━━━━━━━━━━━━━━ CHANGE REQUEST ━━━━━━━━━━━━━━
{change_msg}

━━━━━━━━━━━━━━ CURRENT SDL ━━━━━━━━━━━━━━
{current_sdl}

Apply the change and call submit_sdl with the FULL updated SDL JSON.
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

# The SDL JSON for a rich multi-signal strategy (legs + params + full
# provenance.field_sources) easily exceeds the legacy 1024-token tool budget,
# which truncates the JSON → parse failure → silent legacy fallback. Give the
# selector room.
_SELECTOR_MAX_TOKENS = 8192

# Transient LLM blips (rate-limit, timeout, a truncated/empty response) used to
# kill the whole SDL attempt and silently drop to the legacy pipeline. Retry the
# selector a few times before giving up so chat stays on the SDL path.
_SELECTOR_MAX_RETRIES = 3
_SELECTOR_RETRY_BACKOFF_S = 0.6   # linear backoff between attempts


async def _generate_sdl_json_with_retry(messages: list[dict]) -> str:
    """Call the selector and return a raw SDL JSON string that PARSES.

    Retries on any transient failure — LLM error, empty response, or a truncated
    JSON that fails to parse — so a single blip doesn't fall back to legacy.
    Validates the JSON parses BEFORE returning so a broken response is never
    cached (cache poisoning would otherwise replay it forever).
    """
    import asyncio

    last_exc: Exception | None = None
    for attempt in range(_SELECTOR_MAX_RETRIES):
        try:
            raw = await _call_llm(messages)
            _parse_sdl_json(raw)   # parse-check (catches truncation) before caching
            return raw
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "⚠️ sdl_selector | attempt %d/%d failed: %s",
                attempt + 1, _SELECTOR_MAX_RETRIES, exc,
            )
            if attempt + 1 < _SELECTOR_MAX_RETRIES and _llm_override is None:
                await asyncio.sleep(_SELECTOR_RETRY_BACKOFF_S * (attempt + 1))
    assert last_exc is not None
    raise last_exc


async def _call_llm(messages: list[dict]) -> str:
    """Call LLM (or the override) and return the SDL JSON string.

    Raises on an empty/unusable response so the caller's retry can fire — never
    returns a silent "{}" that would later masquerade as a (broken) SDL.
    """
    if _llm_override is not None:
        # Sync override used in tests
        result = _llm_override(messages, [_SDL_TOOL])
        return result if isinstance(result, str) else json.dumps(result)

    from app.services.ai.llm import LLMService
    llm = LLMService()
    resp = await llm.chat_with_tools(
        messages,
        [_SDL_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_sdl"}},
        max_tokens=_SELECTOR_MAX_TOKENS,
    )
    tool_calls = resp.get("tool_calls") or []
    if tool_calls:
        raw = tool_calls[0]["arguments"].get("sdl_json", "")
    else:
        # Fallback: try to extract JSON from text content
        raw = _extract_json_from_text(resp.get("content") or "")
    if not raw or raw.strip() in ("", "{}"):
        raise ValueError("SDL selector LLM returned an empty/unusable response")
    return raw


def _extract_json_from_text(text: str) -> str:
    """Last-resort: extract the first JSON object from free text."""
    import re
    m = re.search(r"\{[\s\S]+\}", text)
    return m.group(0) if m else "{}"


# ── SDL parsing ───────────────────────────────────────────────────────────────

# ── Enum hardening ────────────────────────────────────────────────────────────
# The LLM reliably picks the right SIGNAL names, but it slips on the small typed
# enums (asset_class, stop_loss.type, take_profit.type, …) because it uses the
# trader's word ("structural", "crypto", "risk reward") instead of our exact
# literal. A single such slip used to raise a Pydantic literal_error and drop the
# whole request into the (broken) legacy pipeline. We instead coerce every enum:
# exact value passes; a known alias is mapped; an unknown value falls back to a
# safe default AND records a clarification (never silent). General, not prompt-
# specific. Each spec = (valid set, alias map, safe default).
_ENUM_SPECS: dict[str, tuple[set[str], dict[str, str], str]] = {
    "stop_loss_type": (
        {"percent", "atr", "points", "swing_low", "swing_high", "candle_low",
         "candle_high", "orb_low", "orb_high", "vwap_deviation", "opposite_side"},
        {"structural": "opposite_side", "swing": "opposite_side",
         "swing_high_low": "opposite_side", "swing_high/low": "opposite_side",
         "recent_swing": "opposite_side", "recent_swing_high_low": "opposite_side",
         "swinghigh": "swing_high", "swinglow": "swing_low",
         "structural_stop": "opposite_side", "atr_based": "atr", "atr_stop": "atr",
         "percentage": "percent", "fixed_percent": "percent"},
        "percent",
    ),
    "take_profit_type": (
        {"percent", "rr", "atr", "points"},
        {"risk_reward": "rr", "reward_risk": "rr", "reward_to_risk": "rr",
         "risk_reward_ratio": "rr", "rrr": "rr", "ratio": "rr", "r:r": "rr",
         "percentage": "percent", "atr_based": "atr", "atr_target": "atr"},
        "rr",
    ),
    "trailing_type": (
        {"atr_based", "percent_based", "ema_based", "chandelier"},
        {"atr": "atr_based", "percent": "percent_based", "pct": "percent_based",
         "ema": "ema_based", "ema_trail": "ema_based"},
        "atr_based",
    ),
    "sizing_mode": (
        {"risk_based", "fixed_fractional", "fixed_units"},
        {"risk": "risk_based", "fixed": "fixed_fractional",
         "fixed_percent": "fixed_fractional", "fixed_fraction": "fixed_fractional",
         "units": "fixed_units", "fixed_unit": "fixed_units"},
        "risk_based",
    ),
    "vol_metric": ({"atr", "natr"}, {}, "atr"),
    "rank_by": (
        {"rvol", "rsi", "atr_pct", "distance_52w"},
        {"relative_volume": "rvol", "rel_vol": "rvol", "volume": "rvol",
         "52w": "distance_52w", "52_week": "distance_52w",
         "distance_52week": "distance_52w", "atr": "atr_pct", "atr_percent": "atr_pct"},
        "rvol",
    ),
}

# htf_rules[].role legal values + common LLM slips → nearest legal role.
_HTF_ROLE_VALID = {"gating", "confirmation", "optional_filter"}
_HTF_ROLE_ALIASES = {
    "filter": "optional_filter", "optional": "optional_filter",
    "optional_filter": "optional_filter", "soft_filter": "optional_filter",
    "gate": "gating", "gating": "gating", "hard_filter": "gating", "block": "gating",
    "confirm": "confirmation", "confirmation": "confirmation", "confirming": "confirmation",
}


_ASSET_CLASS_ALIASES = {
    "crypto": "crypto_spot", "crypto_spot": "crypto_spot", "spot": "crypto_spot",
    "cryptocurrency": "crypto_spot", "coin": "crypto_spot", "coins": "crypto_spot",
    "perp": "crypto_spot", "perpetual": "crypto_spot",
    "equity_cash": "equity_cash", "equity": "equity_cash", "equities": "equity_cash",
    "stock": "equity_cash", "stocks": "equity_cash", "cash": "equity_cash",
    "nse": "equity_cash", "bse": "equity_cash",
    "indian_stocks": "equity_cash", "indian_equity": "equity_cash", "india": "equity_cash",
}


def _coerce_enum(obj: dict, key: str, spec_name: str, path: str, notes: list) -> None:
    """Coerce obj[key] against an _ENUM_SPECS entry: pass-through if valid, map a
    known alias, else fall back to the safe default and record a clarification."""
    if not isinstance(obj, dict) or obj.get(key) is None or not isinstance(obj[key], str):
        return
    valid, aliases, default = _ENUM_SPECS[spec_name]
    v = obj[key].strip().lower()
    if v in valid:
        obj[key] = v
        return
    if v in aliases:
        obj[key] = aliases[v]          # clear correction → no clarification
        return
    obj[key] = default                 # genuine fallback → surface it
    notes.append((path, v, default))


# Symmetric name forms the catalog names inconsistently (e.g. `is_above_sma`
# exists but its mirror is `price_below_sma`). LLMs assume the symmetric name
# (`price_above_sma`) and emit a card that doesn't exist. These prefix swaps let
# us map such a near-miss back to the real card — general across indicators, not
# a per-name hardcode.
_NAME_PREFIX_SWAPS: tuple[tuple[str, str], ...] = (
    ("price_above_", "is_above_"),
    ("is_above_",    "price_above_"),
    ("price_below_", "is_below_"),
    ("is_below_",    "price_below_"),
)


def _canonical_card_name(name: str, valid: set[str]) -> str | None:
    """If `name` isn't a real card, try the symmetric prefix variant; return the
    variant when it IS a real card, else None. Only swaps the price_/is_ prefix
    for the SAME direction (above↔above, below↔below) — never flips the
    direction, so it can't silently invert the user's intent."""
    for a, b in _NAME_PREFIX_SWAPS:
        if name.startswith(a):
            candidate = b + name[len(a):]
            if candidate in valid:
                return candidate
    return None


def _repair_signal_names(data: dict, notes: list[tuple[str, str, str]]) -> None:
    """Remap any SignalRef whose card name doesn't exist to its symmetric
    catalog equivalent (e.g. `price_above_sma` → `is_above_sma`). Unrepairable
    names are left as-is for the validator to flag — we never invent or guess a
    semantically different card here."""
    from app.kb import kb
    valid = set(kb.signals.keys())

    def _fix(ref: Any) -> None:
        if not isinstance(ref, dict):
            return
        name = ref.get("name")
        if not isinstance(name, str) or not name or name in valid:
            return
        canon = _canonical_card_name(name, valid)
        if canon:
            ref["name"] = canon
            notes.append((f"signal:{name}", name, canon))

    for leg in data.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        entry = leg.get("entry") or {}
        _fix(entry.get("trigger"))
        for flt in entry.get("filters") or []:
            _fix(flt)
        ex = leg.get("exit") or {}
        for trig in ex.get("triggers") or []:
            _fix(trig)
        for flt in ex.get("filters") or []:
            _fix(flt)
    for rule in data.get("htf_rules") or []:
        if isinstance(rule, dict):
            _fix(rule.get("signal"))


def _normalize_sdl_data(data: dict) -> dict:
    """Coerce every SDL enum so a reasonable-but-imperfect LLM value never crashes
    the path (which would drop to the legacy pipeline). Records a clarification
    whenever a value had to be defaulted (never silent)."""
    if not isinstance(data, dict):
        return data
    notes: list[tuple[str, str, str]] = []

    # Repair near-miss card names (symmetric naming the catalog is inconsistent
    # about) BEFORE validation, so a `price_above_sma`-style guess resolves to
    # the real card instead of crashing the SDL path.
    name_notes: list[tuple[str, str, str]] = []
    _repair_signal_names(data, name_notes)
    for path, original, chosen in name_notes:
        logger.info("🔧 sdl_selector | NAME REPAIR | %s → %s", original, chosen)

    # universe.asset_class — alias, else derive from market (special-cased).
    uni = data.get("universe")
    if isinstance(uni, dict):
        ac = uni.get("asset_class")
        if isinstance(ac, str):
            uni["asset_class"] = _ASSET_CLASS_ALIASES.get(ac.strip().lower(), ac)
        if uni.get("asset_class") not in ("equity_cash", "crypto_spot"):
            market = str((data.get("context") or {}).get("market", "")).strip().lower()
            uni["asset_class"] = "crypto_spot" if "crypto" in market else "equity_cash"

        # Reconcile a static symbol ↔ asset_class against the KB (ground truth).
        # The LLM sometimes mis-tags a known equity (e.g. INFY → crypto_spot) and
        # leaves the symbol un-normalised ("INFY" instead of "INFY.NS"), which then
        # fails symbol validation. When the symbol resolves in the KB we adopt its
        # canonical symbol and asset_class — a deterministic correction, so (like
        # the alias case) we apply it silently rather than asking the user.
        symbol = uni.get("symbol")
        if uni.get("type", "static") == "static" and isinstance(symbol, str) and symbol.strip():
            try:
                from app.kb import kb as _kb
                kb_stock = _kb.lookup_stock(symbol)
            except Exception:
                kb_stock = None
            if kb_stock is not None:
                if kb_stock.symbol != symbol:
                    logger.info(
                        "🔧 sdl_selector | SYMBOL NORMALISE | %s → %s", symbol, kb_stock.symbol
                    )
                    uni["symbol"] = kb_stock.symbol
                kb_ac = str(kb_stock.asset_class)
                if uni.get("asset_class") != kb_ac:
                    logger.info(
                        "🔧 sdl_selector | ASSET_CLASS FIX | %s: %s → %s (from KB)",
                        kb_stock.symbol, uni.get("asset_class"), kb_ac,
                    )
                    uni["asset_class"] = kb_ac

        rank = uni.get("rank")
        if isinstance(rank, dict):
            _coerce_enum(rank, "by", "rank_by", "universe.rank.by", notes)

    # risk.* enums
    risk = data.get("risk")
    if isinstance(risk, dict):
        _coerce_enum(risk.get("stop_loss") or {}, "type", "stop_loss_type", "risk.stop_loss.type", notes)
        _coerce_enum(risk.get("take_profit") or {}, "type", "take_profit_type", "risk.take_profit.type", notes)
        _coerce_enum(risk.get("trailing") or {}, "type", "trailing_type", "risk.trailing.type", notes)
        _coerce_enum(risk.get("sizing") or {}, "mode", "sizing_mode", "risk.sizing.mode", notes)

    # gates.volatility.metric
    gates = data.get("gates")
    if isinstance(gates, dict):
        _coerce_enum(gates.get("volatility") or {}, "metric", "vol_metric", "gates.volatility.metric", notes)

    # htf_rules[].role — only gating|confirmation|optional_filter are legal. The
    # LLM slips ('filter', 'long', 'trend', …); a bad value used to crash the
    # WHOLE SDL (pydantic literal_error) and drop the request into the legacy
    # pipeline. Coerce to the nearest legal role (silent cosmetic normalisation —
    # an HTF role is not a money value), unknown → safe 'confirmation' default.
    for rule in data.get("htf_rules") or []:
        if isinstance(rule, dict) and isinstance(rule.get("role"), str):
            r = rule["role"].strip().lower()
            rule["role"] = _HTF_ROLE_ALIASES.get(r, r if r in _HTF_ROLE_VALID else "confirmation")

    # surface every defaulted enum as a clarification (never silent)
    if notes:
        prov = data.setdefault("provenance", {})
        clar = prov.setdefault("clarifications_needed", [])
        for path, original, chosen in notes:
            clar.append({
                "field": path,
                "question": f"I didn't recognise '{original}' for {path}; I used '{chosen}'. Want a different value?",
                "assumed_value": chosen,
            })
    return data


def _parse_sdl_json(raw_json: str) -> SDL:
    """Parse the LLM's JSON response into a validated SDL object."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SDL selector returned invalid JSON: {exc}\nRaw: {raw_json[:500]}")

    data = _normalize_sdl_data(data)
    return SDL.model_validate(data)


# ── Merge helper for modify_sdl ───────────────────────────────────────────────

def _is_unset(value: Any) -> bool:
    """A value the modify turn did NOT meaningfully set → fall back to existing.

    None and empty containers/strings count as unset. 0 / False / 0.0 are
    meaningful values and are kept (a user may legitimately set min_ratio=0)."""
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) == 0
    return False


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively overlay `overlay` onto `base`, keeping `base` wherever the
    overlay didn't set a value.

    * dict ⊕ dict → per-key recursive merge (union of keys).
    * list ⊕ list → element-wise by index; the overlay defines the list length
      (so adding/removing a leg or filter takes effect) while nested values the
      overlay dropped fall back to the corresponding base element.
    * scalar     → overlay wins unless it is "unset" (None/empty), else base.

    This is the deep-merge that lets edits ACCUMULATE across turns: a modify
    turn that only restates the changed field no longer wipes everything else,
    and a full-SDL re-emission that accidentally drops a nested value keeps the
    prior value instead of nulling it.
    """
    if _is_unset(overlay):
        return base
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, ov in overlay.items():
            merged[key] = _deep_merge(base.get(key), ov)
        return merged
    if isinstance(base, list) and isinstance(overlay, list):
        merged_list = []
        for i, ov in enumerate(overlay):
            merged_list.append(_deep_merge(base[i] if i < len(base) else None, ov))
        return merged_list
    return overlay


# Keys merged/handled explicitly by _merge_sdl rather than by structural deep
# merge (provenance has upgrade-only semantics; the rest are bookkeeping).
_MERGE_META_KEYS = {"provenance", "version", "parent_version", "created_at", "content_hash"}


def _merge_sdl(existing: SDL, updated: SDL) -> SDL:
    """Deep-merge a modify turn's SDL onto the existing one (design §7).

    Starts from `existing` and overlays only the fields `updated` actually set,
    so edits accumulate instead of replacing the whole ticket. The LLM is still
    prompted to emit the full SDL, but the merge no longer DEPENDS on it
    re-emitting everything — a dropped nested value falls back to `existing`.

    Provenance is merged upgrade-only (never demotes a prior `user` field).
    Version is bumped and parent_version recorded.
    """
    from app.planner.provenance_reconciler import merge_field_sources

    base = existing.model_dump(mode="python")
    over = updated.model_dump(mode="python")

    merged: dict[str, Any] = {}
    for key in set(base) | set(over):
        if key in _MERGE_META_KEYS:
            continue
        merged[key] = _deep_merge(base.get(key), over.get(key))

    # Provenance: union with upgrade-only field_sources; dedup notes.
    base_prov = base.get("provenance") or {}
    over_prov = over.get("provenance") or {}
    merged["provenance"] = {
        "field_sources": merge_field_sources(
            base_prov.get("field_sources"), over_prov.get("field_sources")
        ),
        "unmapped_details": _dedup_dicts(
            (base_prov.get("unmapped_details") or []) + (over_prov.get("unmapped_details") or []),
            key="text",
        ),
        "clarifications_needed": _dedup_dicts(
            (over_prov.get("clarifications_needed") or []) + (base_prov.get("clarifications_needed") or []),
            key="field",
        ),
    }

    merged["version"] = existing.version + 1
    merged["parent_version"] = existing.version
    merged.pop("content_hash", None)
    return SDL(**merged)


def _dedup_dicts(items: list[dict], *, key: str) -> list[dict]:
    """First-seen-wins dedup of a list of dicts by one key."""
    seen: set = set()
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        k = item.get(key)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def _edit_fields(sdl: SDL) -> dict[str, Any]:
    """Flatten the user-meaningful fields whose change we surface as an
    `edit.applied` log line (risk params + per-leg entry/exit signal names)."""
    out: dict[str, Any] = {}
    risk = sdl.risk
    if risk is not None:
        if risk.stop_loss is not None:
            out["risk.stop_loss.type"] = risk.stop_loss.type
            out["risk.stop_loss.value"] = risk.stop_loss.value
            out["risk.stop_loss.multiple"] = risk.stop_loss.multiple
        if risk.take_profit is not None:
            out["risk.take_profit.type"] = risk.take_profit.type
            out["risk.take_profit.value"] = risk.take_profit.value
            out["risk.take_profit.ratio"] = risk.take_profit.ratio
    for i, leg in enumerate(sdl.legs or []):
        out[f"legs.{i}.direction"] = leg.direction
        if leg.entry and leg.entry.trigger:
            out[f"legs.{i}.entry.trigger"] = leg.entry.trigger.name
        out[f"legs.{i}.entry.filters"] = ",".join(
            f.name for f in (leg.entry.filters if leg.entry else [])
        )
        out[f"legs.{i}.exit.triggers"] = ",".join(
            t.name for t in (leg.exit.triggers if leg.exit else [])
        )
    return out


def _log_sdl_edits(existing: SDL, merged: SDL) -> None:
    """Emit one `edit.applied` step log per field the modify turn changed (§11)."""
    from app.core import step_logger as _sl

    before = _edit_fields(existing)
    after = _edit_fields(merged)
    sid = existing.content_hash
    turn = merged.version
    for field in sorted(set(before) | set(after)):
        old = before.get(field)
        new = after.get(field)
        if old == new or (old in (None, "") and new in (None, "")):
            continue
        _sl.log_step(
            _sl.REPAIR, "verify", sid, turn, "edit.applied",
            field=field, change=f"{old}->{new}",
        )


# ── Public entry points ───────────────────────────────────────────────────────

_FLAG_ENV = "SDL_SELECTOR_ENABLED"


def _check_flag() -> None:
    if os.environ.get(_FLAG_ENV, "").strip() not in ("1", "true", "yes"):
        raise RuntimeError(
            f"SDL selector is disabled. Set {_FLAG_ENV}=1 to enable."
        )


async def compile_to_sdl(
    prompt: str,
    menu: CatalogMenu | None = None,
    *,
    skip_flag: bool = False,
) -> SDL:
    """CREATE turn: prompt → SDL.

    skip_flag=True bypasses the SDL_SELECTOR_ENABLED check (use in tests).
    """
    if not skip_flag:
        _check_flag()

    if menu is None:
        menu = build_menu()

    key = _cache_key(prompt, menu.catalog_version)
    raw_json = _cache.get(key)
    if raw_json is not None:
        logger.info(
            "🔄 sdl_selector | CACHE HIT | CREATE | prompt=%.40s…", prompt
        )
    else:
        logger.info(
            "🧠 sdl_selector | CREATE | prompt=%.60s… | catalog=%s",
            prompt, menu.catalog_version,
        )
        catalog_text = format_menu_for_llm(menu)
        system = _SYSTEM_PROMPT_TEMPLATE.format(catalog_text=catalog_text, enum_reference=_enum_reference())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Retry transient failures; only a JSON that actually parses is cached.
        raw_json = await _generate_sdl_json_with_retry(messages)
        _cache[key] = raw_json

    sdl = _parse_sdl_json(raw_json)

    # Deterministic provenance reconciliation — fixes the LLM's incomplete
    # self-reporting of field_sources (root cause of low match%). Runs on cache
    # hits too so a cached strategy reports the same match% as a fresh one.
    from app.planner.provenance_reconciler import reconcile
    sdl = reconcile(sdl, prompt)

    built = sum(1 for s in (sdl.provenance.field_sources or {}).values() if s == "user")
    missed = len(sdl.provenance.unmapped_details or [])
    total = built + missed
    pct = int(built / total * 100) if total else 100
    logger.info(
        "✅ sdl_selector | SDL CREATED | version=%d | legs=%d | "
        "hash=%.8s | match=%d/%d (%d%%) | clarifications=%d",
        sdl.version, len(sdl.legs), sdl.content_hash,
        built, total, pct,
        len(sdl.provenance.clarifications_needed or []),
    )
    return sdl


async def modify_sdl(
    existing_sdl: SDL,
    change_msg: str,
    menu: CatalogMenu | None = None,
    *,
    skip_flag: bool = False,
) -> SDL:
    """MODIFY turn: existing SDL + change_msg → updated SDL.

    skip_flag=True bypasses the SDL_SELECTOR_ENABLED check (use in tests).
    """
    if not skip_flag:
        _check_flag()

    if menu is None:
        menu = build_menu()

    key = _cache_key(
        f"{existing_sdl.content_hash}|{change_msg}", menu.catalog_version
    )
    raw_json = _cache.get(key)
    if raw_json is not None:
        logger.info(
            "🔄 sdl_selector | CACHE HIT | MODIFY | sdl=%.8s | change=%.40s…",
            existing_sdl.content_hash, change_msg,
        )
    else:
        logger.info(
            "🧠 sdl_selector | MODIFY | v%d→? | sdl=%.8s | change=%.60s…",
            existing_sdl.version, existing_sdl.content_hash, change_msg,
        )
        current_json = existing_sdl.model_dump_json(indent=2)
        user_content = _MODIFY_PROMPT_TEMPLATE.format(
            change_msg=change_msg,
            current_sdl=current_json,
        )
        catalog_text = format_menu_for_llm(menu)
        system = _SYSTEM_PROMPT_TEMPLATE.format(catalog_text=catalog_text, enum_reference=_enum_reference())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        # Retry transient failures; only a JSON that actually parses is cached.
        raw_json = await _generate_sdl_json_with_retry(messages)
        _cache[key] = raw_json

    updated = _parse_sdl_json(raw_json)

    # Reconcile provenance for the change message, carrying the prior turn's
    # provenance forward as the floor (upgrade-only): a value the user set on an
    # earlier turn keeps its "user" source even if this change didn't restate it.
    from app.planner.provenance_reconciler import reconcile
    updated = reconcile(
        updated, change_msg,
        prior_field_sources=(existing_sdl.provenance.field_sources or {}),
    )

    merged = _merge_sdl(existing_sdl, updated)
    _log_sdl_edits(existing_sdl, merged)
    hash_changed = merged.content_hash != existing_sdl.content_hash
    logger.info(
        "✅ sdl_selector | SDL MODIFIED | v%d→v%d | hash=%s%s",
        existing_sdl.version, merged.version,
        "%.8s → %.8s" % (existing_sdl.content_hash, merged.content_hash)
        if hash_changed else "unchanged (%.8s)" % merged.content_hash,
        " | ⚠️  logic unchanged" if not hash_changed else "",
    )
    return merged
