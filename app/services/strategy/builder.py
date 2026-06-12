"""
StrategyBuilder holds the evolving state of a strategy chat session.

Conversation flow:
1. collect_user_input
2. plan_signals
3. assemble_strategy

Entry/exit formulas are no longer collected from the user directly. They are
derived from the KB-driven signal plan so the chat flow can stay focused on
high-level user intent first.
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Optional, Tuple

import requests

from app.kb.compat import (
    get_all_market_configs,
    get_daily_loss_cap_bounds,
    get_experience_risk,
    get_risk_defaults,
)
from app.kb.signal_validation import (
    SignalPlacementValidationError,
    assert_valid_signal_lists,
    validate_signal_placement,
)
from app.services.chat.response_composer import (
    build_unsupported_timeframe_message,
    compose_response,
)

try:
    from nsepython import nse_eq_symbols
except Exception:  # pragma: no cover - optional runtime dependency
    nse_eq_symbols = None  # type: ignore[assignment]

DEFAULT_MARKET = "indian_stocks"
CORE_USER_INPUT_FIELDS = (
    "symbol",
    "timeframe",
    "objective",
    "sentiment",
    "experience",
    "goal",
)
_SIGNAL_PLAN_PUBLIC_KEYS = {
    "entry",
    "exit",
    "entry_condition",
    "exit_condition",
    "signals_available",
    "signals_used",
}

# Populated from app/kb/compat (built-in market defaults); edit that module to tune.
MARKET_CONFIG: dict = get_all_market_configs()

# Static labels for the risk-summary view; edit app/kb/compat.py to change.
DEFAULT_RISK_AND_EXECUTION: dict = get_risk_defaults()

VALID_TIMEFRAMES = {
    "1m", "3m", "5m", "10m", "15m", "30m", "45m",
    "1h", "2h", "3h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w",
}

# Curated presets shown to the user (chips / examples). NOT a validation
# allowlist — validation is range-based (see resolve_supported_user_timeframe).
# Any timeframe from TIMEFRAME_MIN_MINUTES to TIMEFRAME_MAX_MINUTES is accepted,
# because backtests fetch 1-minute data and the engine resamples it up to any
# timeframe (quant_engine/engine/resample.py).
SUPPORTED_USER_TIMEFRAMES = (
    "1m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "1d",
)

# Inclusive bounds for an accepted timeframe, in minutes.
#   floor = 1m  — we only have 1-minute source data; nothing finer can be built.
#   cap   = 1d  — resampling a full day (1440 bars) from 1m is the practical
#                 ceiling; anything coarser (multi-day / weekly) is rejected.
TIMEFRAME_MIN_MINUTES = 1
TIMEFRAME_MAX_MINUTES = 1440

SUPPORTED_USER_TIMEFRAME_TEXT = (
    "any timeframe from 1m up to 1d (for example 1m, 5m, 15m, 30m, 1h, 4h, 1d)"
)
UNSUPPORTED_USER_TIMEFRAME_CODE = "validation.unsupported_timeframe"
UNSUPPORTED_USER_TIMEFRAME_MESSAGE = build_unsupported_timeframe_message(
    SUPPORTED_USER_TIMEFRAME_TEXT
)


def unsupported_user_timeframe_validation_facts() -> dict[str, Any]:
    return {"supported_timeframes": SUPPORTED_USER_TIMEFRAME_TEXT}

_TF_ALIASES: dict[str, str] = {
    "1min": "1m", "1 min": "1m", "1minute": "1m", "1 minute": "1m",
    "3min": "3m", "3 min": "3m",
    "5min": "5m", "5 min": "5m", "5minute": "5m", "5 minutes": "5m",
    "10min": "10m", "10 min": "10m", "10minute": "10m", "10 minutes": "10m",
    "15min": "15m", "15 min": "15m", "15minute": "15m", "15 minutes": "15m",
    "30min": "30m", "30 min": "30m", "30minute": "30m", "30 minutes": "30m",
    "45min": "45m", "45 min": "45m", "45minute": "45m", "45 minutes": "45m",
    "1hour": "1h", "1 hour": "1h", "hourly": "1h",
    "2hour": "2h", "2 hour": "2h", "2hours": "2h",
    "3hour": "3h", "3 hour": "3h", "3hours": "3h",
    "4hour": "4h", "4 hour": "4h", "4hours": "4h",
    "6hour": "6h", "8hour": "8h", "12hour": "12h",
    "daily": "1d", "day": "1d", "1day": "1d", "1 day": "1d",
    "3day": "3d", "3 day": "3d", "3days": "3d",
    "weekly": "1w", "week": "1w", "1week": "1w", "1 week": "1w",
}

_SYMBOL_ALIASES = {
    "ADANIENT": "ADANIENT",
    "ADANI ENTERPRISES": "ADANIENT",
    "GMR AIRPORT": "GMRAIRPORT",
    "GMR AIRPORTS": "GMRAIRPORT",
    "GMRAIRPORT": "GMRAIRPORT",
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "INFY": "INFY",
    "INFOSYS": "INFY",
    "HDFC": "HDFCBANK",
    "HDFCBANK": "HDFCBANK",
    "ICICIBANK": "ICICIBANK",
    "IDEA": "IDEA",
    "NHPC": "NHPC",
    "SUZLON": "SUZLON",
    "SUZLON ENERGY": "SUZLON",
    "SBIN": "SBIN",
    "ITC": "ITC",
    "LT": "LT",
    "AXISBANK": "AXISBANK",
    "KOTAKBANK": "KOTAKBANK",
    "BHARTIARTL": "BHARTIARTL",
    "HCLTECH": "HCLTECH",
    "MARUTI": "MARUTI",
    "SUNPHARMA": "SUNPHARMA",
    "VODAFONE IDEA": "IDEA",
}

_TIMEFRAME_SHORT_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_TIMEFRAME_VALUE_RE = re.compile(r"^\s*(\d+)\s*(?:-|to)?\s*([a-zA-Z]+)\s*$")
_STOP_WORDS = {
    "a", "an", "and", "at", "bearish", "beginner", "build", "bullish", "bse", "create",
    "day", "for", "from", "hello", "hey", "hi", "i", "in", "indian", "intermediate",
    "intraday", "long", "market", "minutes", "month", "my", "need", "nse", "on",
    "or", "position", "positional", "profit", "quick", "risk", "steady", "stock", "stocks",
    "strategy", "swing", "term", "that", "the", "this", "timeframe", "to", "trade", "want",
    "with", "hours", "hour", "minute", "day", "days", "week", "weeks", "month", "months",
    "growth", "protection", "expert", "advanced", "experienced", "novice", "newbie", "fresher",
    "yes", "yep", "yeah", "ok", "okay", "proceed", "confirm", "confirmed", "continue",
    "please", "thanks", "thank", "good", "looks", "fine", "right", "correct",
    "go", "ahead", "do", "it", "setup", "sentiment", "view", "trader", "traders", "trading",
    "change", "update", "set", "use", "keep", "same", "show", "give", "tell",
}
_MISSING_INPUT_TEXT = {
    "none",
    "null",
    "nil",
    "n/a",
    "na",
    "not applicable",
    "unknown",
    "unspecified",
    "not specified",
}


def _normalize_optional_input_text(value: Any, *, lowercase: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    if text.lower() in _MISSING_INPUT_TEXT:
        return None
    return text.lower() if lowercase else text


@lru_cache(maxsize=1)
def _get_nse_symbol_set() -> set[str]:
    if nse_eq_symbols is None:
        return set()
    try:
        return {str(symbol).upper().strip() for symbol in nse_eq_symbols()}
    except Exception:
        return set()


def _strip_symbol_suffix(symbol: str) -> str:
    clean = symbol.upper().strip()
    if ":" in clean:
        _, clean = clean.split(":", 1)
    return re.sub(r"(\.NS|\.BO)$", "", clean)


def _exchange_hint(text: str) -> Optional[str]:
    if re.search(r"\bbse\b|\.bo\b", text, re.IGNORECASE):
        return "BO"
    if re.search(r"\bnse\b|\.ns\b", text, re.IGNORECASE):
        return "NS"
    return None


def _format_symbol_with_exchange(symbol: str, exchange: Optional[str] = None) -> str:
    base = _strip_symbol_suffix(symbol)
    suffix = ".BO" if exchange == "BO" else ".NS"
    return f"{base}{suffix}"


def _canonical_timeframe_from_value(value: int, unit: str) -> Optional[str]:
    unit = unit.lower().strip()
    if value <= 0:
        return None

    if unit in {"m", "min", "mins", "minute", "minutes"}:
        if value % 1440 == 0:
            return f"{value // 1440}d"
        if value % 60 == 0:
            return f"{value // 60}h"
        return f"{value}m"

    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        if value % 24 == 0:
            return f"{value // 24}d"
        return f"{value}h"

    if unit in {"d", "day", "days"}:
        return f"{value}d"

    if unit in {"w", "wk", "wks", "week", "weeks"}:
        return f"{value}w"

    return None


@lru_cache(maxsize=256)
def _search_indian_equity(query: str, exchange: Optional[str] = None) -> Optional[str]:
    cleaned_query = re.sub(r"\s+", " ", query).strip()
    if not cleaned_query:
        return None

    try:
        response = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": cleaned_query, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=4,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    preferred_suffix = ".BO" if exchange == "BO" else ".NS"
    query_tokens = {token for token in re.findall(r"[A-Z0-9]+", cleaned_query.upper()) if len(token) > 1}
    best_symbol = None
    best_score = float("-inf")

    for quote in payload.get("quotes", []):
        if quote.get("quoteType") != "EQUITY":
            continue

        symbol = str(quote.get("symbol") or "").upper().strip()
        if not symbol.endswith((".NS", ".BO")):
            continue

        score = 0.0
        exchange_name = str(quote.get("exchange") or "").upper()
        if symbol.endswith(preferred_suffix):
            score += 8.0
        if exchange == "BO" and exchange_name == "BSE":
            score += 4.0
        if exchange != "BO" and exchange_name in {"NSI", "NSE"}:
            score += 4.0

        base_symbol = _strip_symbol_suffix(symbol)
        if cleaned_query.upper() == base_symbol:
            score += 12.0

        names_blob = " ".join(
            part for part in [
                str(quote.get("shortname") or ""),
                str(quote.get("longname") or ""),
            ] if part
        ).upper()
        for token in query_tokens:
            if token == base_symbol:
                score += 5.0
            if token in names_blob:
                score += 2.0

        if "-" in base_symbol:
            score -= 1.5

        if score > best_score:
            best_score = score
            best_symbol = symbol

    return best_symbol


def _extract_search_query(text: str) -> str:
    cleaned = _TF_RE.sub(" ", text)
    cleaned = re.sub(r"\b([A-Za-z][A-Za-z0-9&-]{1,19})\.(?:NS|BO)\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(bullish|bearish|optimistic|positive|negative|cautious|long|short|buy|sell)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(intraday|day trade|day trading|scalp|quick profit|quick profits|positional|swing|steady growth|risk protection|long term)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9&.\-]+", " ", cleaned)
    tokens = [token for token in cleaned.split() if token.lower() not in _STOP_WORDS]
    return " ".join(tokens[:6])


def normalise_timeframe(raw: str) -> Optional[str]:
    original = raw.strip()
    if not original:
        return None

    short_match = _TIMEFRAME_SHORT_RE.match(original)
    if short_match:
        value = int(short_match.group(1))
        short_unit = short_match.group(2).lower()
        return _canonical_timeframe_from_value(value, short_unit)

    clean = original.lower()
    if clean in {tf.lower() for tf in VALID_TIMEFRAMES}:
        return clean
    if clean in _TF_ALIASES:
        return _TF_ALIASES[clean]

    match = _TIMEFRAME_VALUE_RE.match(clean)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        return _canonical_timeframe_from_value(value, unit)

    return None


def extract_company_name_query(text: str) -> str:
    return _extract_search_query(text)


def extract_exchange_hint(text: str) -> Optional[str]:
    return _exchange_hint(text)


def extract_goal_text(text: str) -> Optional[str]:
    cleaned = " ".join(str(text or "").split()).strip(" \t\r\n.,;:-")
    if not cleaned:
        return None

    patterns = (
        r"\b(?:my\s+)?goal\s*(?:is|=|:|-)\s*(.+)",
        r"\b(?:my\s+)?intent(?:ion)?\s*(?:is|=|:|-)\s*(.+)",
        r"\b(?:i(?:'m| am)?\s+)?looking\s+for\s+(.+)",
        r"\b(?:i(?:'d| would)?\s+)?want\s+(?:a\s+)?(?:setup|strategy)\s+(?:that\s+)?(.+)",
        r"\b(?:i\s+)?need\s+(?:a\s+)?(?:setup|strategy)\s+(?:that\s+)?(.+)",
        r"\b(?:the\s+)?strategy\s+should\s+(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            continue
        candidate = " ".join(str(match.group(1) or "").split()).strip(" \t\r\n.,;:-")
        if len(re.findall(r"[A-Za-z0-9]+", candidate)) >= 3:
            return candidate
    return None


def _timeframe_to_minutes(timeframe: str) -> Optional[int]:
    match = _TIMEFRAME_SHORT_RE.match((timeframe or "").strip())
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2).lower()

    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    if unit == "w":
        return value * 10080
    return None


def resolve_supported_user_timeframe(timeframe: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve a user-provided timeframe to its canonical form.

    Returns (canonical, None) on success, otherwise (None, validation_message).

    Validation is RANGE-BASED, not allowlist-based: any well-formed timeframe
    whose duration is within [TIMEFRAME_MIN_MINUTES, TIMEFRAME_MAX_MINUTES] is
    accepted (e.g. 2m, 7m, 45m, 4h), because the backtest fetches 1-minute data
    and the engine resamples it up to the requested timeframe. The user's value
    is canonicalised (e.g. "60 minute" → "1h") but never snapped to a different
    duration — what they ask for is what they get.
    """
    raw = (timeframe or "").strip()
    if not raw:
        return None, UNSUPPORTED_USER_TIMEFRAME_MESSAGE

    canonical = normalise_timeframe(raw)
    if not canonical:
        return None, UNSUPPORTED_USER_TIMEFRAME_MESSAGE

    minutes = _timeframe_to_minutes(canonical)
    if minutes is None or minutes < TIMEFRAME_MIN_MINUTES or minutes > TIMEFRAME_MAX_MINUTES:
        return None, UNSUPPORTED_USER_TIMEFRAME_MESSAGE

    return canonical, None


_TF_RE = re.compile(
    r"\b(1m|3m|5m|10m|15m|30m|45m|1h|2h|3h|4h|6h|8h|12h|1d|3d|1w|"
    r"daily|weekly|hourly|"
    r"\d+\s*(?:-|to)?\s*(?:M|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks))\b",
    re.IGNORECASE,
)


# Default trading window per asset class — applied when the user didn't
# explicitly set entry_window_start / entry_window_end on the builder.
# Indian stocks: NSE cash session 09:15–15:30 IST (the 375-minute session).
# Crypto:        no window (24/7 market) — leave as None so the gate is a no-op.
# Other markets (US, indices, ...): no opinion; let the data layer handle it.
def _default_entry_window_for_market(market: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    m = (market or "").lower().strip()
    if m == "indian_stocks":
        return ("09:15", "15:30")
    return (None, None)


# Display label of the default trading window per market. Same mapping logic
# as _default_entry_window_for_market but rendered as a single string for UI
# fields like `risk_and_execution.trading_window`. Crypto resolves to "24/7"
# because Binance has no session boundary.
def _default_trading_window_label_for_market(market: Optional[str]) -> Optional[str]:
    m = (market or "").lower().strip()
    if m in ("indian_stocks", "indian_indices"):
        return "09:15 - 15:30"
    if m == "crypto":
        return "24/7"
    return None


# ── Asset-class ↔ market mappings ─────────────────────────────────────────────
# The KB universe.csv uses `asset_class` ("equity_cash" / "crypto_spot"); the
# builder works in "market" terms ("indian_stocks" / "crypto"). These two
# dicts let us round-trip between the two views.
_ASSET_CLASS_TO_MARKET: dict[str, str] = {
    "equity_cash": "indian_stocks",
    "crypto_spot": "crypto",
}
_MARKET_TO_ASSET_CLASS: dict[str, str] = {
    "indian_stocks":  "equity_cash",
    "indian_indices": "equity_cash",
    "us_stocks":      "equity_cash",
    "crypto":         "crypto_spot",
}

# Symbol-shape heuristics — used as fallback when the symbol isn't in the KB.
_CRYPTO_QUOTE_TAIL_RE = re.compile(r"(USDT|USDC|BUSD|FDUSD|DAI|USD)$", re.IGNORECASE)
_CANONICAL_CRYPTO_PAIR_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")


def _infer_market_from_symbol(symbol: Optional[str]) -> Optional[str]:
    """Resolve the canonical market string from a symbol.

    Source-of-truth order:
      1. KB universe.csv (`stock.asset_class`) — definitive when the symbol is
         in the catalogue.
      2. Shape heuristics for symbols not in the KB:
         - "*.NS" / "*.BO" → indian_stocks
         - "BASE_QUOTE"     → crypto    (canonical, e.g. "BTC_USDT")
         - "BASEQUOTE" ending in a known quote tail → crypto (broker-native).
      3. None when ambiguous — caller keeps whatever market it had.

    This is called from `sync_market_from_symbol()` so every symbol assignment
    forces the market to follow what the symbol actually is, never the other
    way around.
    """
    if not symbol:
        return None
    s = str(symbol).strip()
    if not s:
        return None

    # 1. KB universe lookup. The KB import is lazy because the builder is
    # imported during app startup before the KB module is necessarily loaded.
    try:
        from app.kb import kb as _kb  # type: ignore
        stock = _kb.stocks.get(s) or _kb.stocks.get(s.upper())
        if stock is not None:
            asset_class = getattr(stock, "asset_class", None)
            mapped = _ASSET_CLASS_TO_MARKET.get(str(asset_class or "").lower())
            if mapped:
                return mapped
    except Exception:
        # KB unavailable / not loaded yet — fall through to heuristics.
        pass

    # 2. Shape heuristics.
    u = s.upper()
    if u.endswith((".NS", ".BO", ".NSE", ".BSE")):
        return "indian_stocks"
    if _CANONICAL_CRYPTO_PAIR_RE.fullmatch(u):
        return "crypto"
    # Broker-native crypto pair (no underscore). Require ≥6 chars to avoid
    # collapsing 4-letter tickers that happen to end in "USD".
    if len(u) >= 6 and _CRYPTO_QUOTE_TAIL_RE.search(u):
        return "crypto"

    return None


class StrategyBuilder:
    def __init__(self):
        self.market: Optional[str] = DEFAULT_MARKET
        self.symbol: Optional[str] = None
        self.symbol_validation_code: Optional[str] = None
        self.symbol_validation_facts: dict[str, Any] = {}
        self._symbol_validation_message: Optional[str] = None
        self.input_validation_code: Optional[str] = None
        self.input_validation_facts: dict[str, Any] = {}
        self._input_validation_message: Optional[str] = None
        self.timeframe: Optional[str] = None
        self.timeframe_validation_code: Optional[str] = None
        self.timeframe_validation_facts: dict[str, Any] = {}
        self._timeframe_validation_message: Optional[str] = None
        self.sentiment: Optional[str] = None
        # Backing store for the `experience` property below — the property
        # setter refreshes tier-derived Phase 10 defaults whenever the value
        # changes, so callers can keep doing `builder.experience = "..."`.
        self._experience: Optional[str] = None
        self._objective: Optional[str] = None
        self._goal: Optional[str] = None
        # original_user_prompt holds the *raw* first substantive user message
        # describing the strategy (e.g. "Buy when price crosses above 20 SMA,
        # SL below previous candle low, target 3:7 RR"). It is NEVER overwritten
        # by the agent router's summarized `goal` field, so downstream semantic
        # extraction always has access to the user's literal phrasing.
        self.original_user_prompt: Optional[str] = None
        self.daily_loss_cap: Optional[float] = None
        self.max_trade: Optional[str] = None
        self.entry_condition: Optional[str] = None
        self.exit_condition: Optional[str] = None
        # Phase 12 — SHORT leg for two-sided ("both") strategies. The engine
        # runs a separate short pass on these; None for single-direction.
        self.short_entry_condition: Optional[str] = None
        self.short_exit_condition: Optional[str] = None
        # SDL ticket for the current strategy (set by the SDL flow). Kept so a
        # later "change RSI to 20 / use EMA instead" turn edits THIS ticket via
        # the SDL modify path instead of the legacy modifier. Persisted across
        # turns as `sdl_json` in the draft (see to_draft_json / merge_preview).
        self._sdl: Any = None
        # stop_loss / take_profit / risk_reward are NOT stored here — they are
        # read-through @property accessors over the single risk store
        # `risk_execution_config` (keys stop_loss_pct / take_profit_pct /
        # risk_reward). This is the WS1 single-source-of-truth: there is no
        # second mirror to drift out of sync. The backing dict is initialised a
        # few lines down (self.risk_execution_config = {}); no property is
        # accessed before then.
        self.user_input_confirmed: bool = False
        self.signal_plan: Optional[dict] = None
        # Optional pinned strategy preset (e.g. "orb", "ema_pullback").
        # When set, the planner pipeline skips its filter+rank loop and uses
        # the preset's exact signal stack instead. Set either by the chat layer
        # (keyword detection in extract_strategy_details) or by API callers.
        self.strategy_preset: Optional[str] = None
        # Phase 3 — structural SL anchor + trailing SL specs. When the planner
        # applies a preset that pins these, they ride here and are written to
        # the generated strategy YAML so the simulator picks them up.
        self.stop_loss_spec: Optional[dict[str, Any]] = None
        self.trailing_stop_spec: Optional[dict[str, Any]] = None
        # Phase 4 — reference benchmark symbol (e.g. ^NSEI). When set, the API
        # layer fetches its OHLCV alongside the trade symbol and the simulator
        # exposes REF_CLOSE / RS(n) / reference_above_sma to the conditions.
        self.reference_symbol: Optional[str] = None
        # Phase 5 — higher-timeframe entry gates. Each item:
        # {"timeframe": "1d", "condition": "CLOSE > EMA(50)"}. When non-empty,
        # the API layer fetches the corresponding HTF OHLCV and the simulator
        # blocks entries that don't pass every gate on the most recent
        # CLOSED HTF bar.
        self.htf_rules: list[dict[str, Any]] = []
        # Phase 8b — optional wall-clock intraday cutoff. Shape:
        # {"exit_time": "15:15", "timezone": "Asia/Kolkata"}.
        # When set, the strategy YAML carries this block and the simulator
        # force-exits any open position once a bar's UTC time crosses the
        # cutoff (and blocks new entries past that point).
        self.time_exit: Optional[dict[str, Any]] = None
        # Phase 9 — discovery state for "dynamic" presets that don't take a
        # user-supplied symbol. Lifecycle:
        #   1. Preset has a discovery: block AND user hasn't given a symbol →
        #      orchestrator runs the scan.
        #   2. Result single: discovered_symbol + symbol set; flow continues.
        #   3. Result none: discovery_no_match=True; chat asks user to
        #      relax criteria.
        #   4. Result multiple: discovery_pending=True, candidates +
        #      tie_break_options stored; chat asks user to pick a method;
        #      next user reply is interpreted as the tie-break choice.
        self.discovery_pending:           bool                   = False
        self.discovery_no_match:          bool                   = False
        self.discovery_candidates:        Optional[list[dict]]   = None
        self.discovery_tie_break_options: Optional[list[dict]]   = None
        self.discovery_chosen_method:     Optional[str]          = None
        self.discovered_symbol:           Optional[str]          = None
        # Phase 9h — user-typed tunables that override the preset's
        # discovery `parameters` defaults (e.g. "volume spike 1.2x" sets
        # {"volume_multiplier": 1.2}). Populated by the chat layer
        # before maybe_dispatch_discovery, consumed by the orchestrator.
        self.discovery_parameter_overrides: Optional[dict[str, float]] = None
        # Phase 9h — actual parameter values used by the most recent
        # scan. Surfaces in the no-match message so the user can tell
        # whether their override was honored.
        self.discovery_parameters_used:     Optional[dict[str, float]] = None
        # Phase 9m — diagnostics from the most recent scan: asof date,
        # universe size, per-condition failure counts. Drives the
        # data-actionable no-match reply.
        self.discovery_last_scan_diagnostics: Optional[dict[str, Any]] = None
        # Phase 9k — compositional primitives parsed from the user's
        # prose. When non-empty, the orchestrator IGNORES the preset's
        # hardcoded `conditions` list and runs the scanner with only
        # the primitives the user explicitly mentioned. Format:
        # [{"name": "volume_spike", "params": {"multiplier": 1.2}}, ...]
        self.discovery_conditions:          Optional[list[dict]] = None
        # Issue 2 — explicit user-supplied universe restriction.
        # When the user types e.g. "choose among these stocks: TCS,
        # Infosys, Reliance", the chat layer resolves each name to a
        # canonical KB symbol and stashes the list here. The scanner
        # then evaluates ONLY these symbols instead of the full
        # kb_default universe — so subsequent diagnostics name the
        # exact stocks the user asked about. Sticky across turns
        # until the user explicitly broadens scope.
        self.discovery_universe_override:   Optional[list[str]]  = None
        self.risk_execution_config: dict[str, Any] = {}
        self.input_modification_requested: bool = False
        self.pending_input_modification_fields: list[str] = []
        # Normalized semantic intent extracted from user's raw message.
        # Populated in collect_input (before planning) so that the planner
        # can use strategy_family, htf_rules, reference_symbols, and
        # structural SL/trailing specs extracted from free-form prose.
        # Persisted as a plain dict in the draft JSON so it survives across turns.
        self.semantic_intent: Optional[dict[str, Any]] = None
        # Phase 10 — direction constraint
        self.direction: Optional[str] = None             # "long_only" | "short_only" | "both"
        # Phase 10 — session entry window (user-supplied, e.g. "09:30–11:30")
        self.entry_window_start: Optional[str] = None    # HH:MM in IST
        self.entry_window_end:   Optional[str] = None    # HH:MM in IST
        # Phase 10 — risk circuit breakers
        self.max_consecutive_losses: Optional[int] = None
        self.cooldown_bars_after_loss: Optional[int] = None
        self.cooldown_bars_after_profit: Optional[int] = None
        # Phase 10 — entry gate controls
        self.max_spread_bps: Optional[float] = None
        self.gap_filter: Optional[str] = None            # "none" | "ignore_gap_up" | "ignore_gap_down" | "ignore_both"
        self.gap_threshold_pct: Optional[float] = None
        self.entry_confirmation_bars: Optional[int] = None
        self.rsi_entry_band_min: Optional[float] = None
        self.rsi_entry_band_max: Optional[float] = None
        self.volume_ratio_threshold: Optional[float] = None
        # Phase 10 — position sizing
        self.position_sizing_mode: Optional[str] = None  # "fixed_fractional" | "risk_based" | "fixed_units"
        self.max_capital_allocation_pct: Optional[float] = None
        # Tracks which Phase 10 tier-derived fields the user has explicitly set
        # (so we never clobber them with tier defaults). Populated by callers
        # via mark_phase10_user_override() whenever they apply a user-supplied
        # value to one of the tier-derived fields. Persisted in draft JSON so
        # the set survives across turns.
        self._phase10_user_overrides: set[str] = set()

    # Field names whose defaults come from risk_tiers.yaml. Listed once here so
    # apply_tier_execution_defaults() and the override-tracking logic stay in
    # sync.
    _TIER_DERIVED_PHASE10_FIELDS: tuple[str, ...] = (
        "max_consecutive_losses",
        "cooldown_bars_after_loss",
        "cooldown_bars_after_profit",
        "max_spread_bps",
        "entry_confirmation_bars",
        "max_capital_allocation_pct",
        "position_sizing_mode",
        "gap_filter",
    )

    def mark_phase10_user_override(self, field_name: str) -> None:
        """Record that the user explicitly set a tier-derived Phase 10 field.

        Callers (chat_service, agent tool catalog) should call this whenever
        they apply a user-supplied value to one of the tier-derived fields.
        Once marked, apply_tier_execution_defaults() will preserve the value
        even when experience changes.
        """
        if field_name in self._TIER_DERIVED_PHASE10_FIELDS:
            self._phase10_user_overrides.add(field_name)

    @property
    def experience(self) -> Optional[str]:
        return self._experience

    @experience.setter
    def experience(self, value: Optional[str]) -> None:
        new_val = _normalize_optional_input_text(value, lowercase=True)
        old_val = self._experience
        self._experience = new_val
        # When experience changes, re-pull tier-derived Phase 10 defaults so
        # the new tier's values flow through. Fields the user has explicitly
        # set are preserved via the _phase10_user_overrides set.
        if old_val != new_val:
            try:
                self.apply_tier_execution_defaults()
            except Exception:
                # apply_tier_execution_defaults imports app.kb lazily; if the
                # KB isn't loaded yet (very early in process startup) leave
                # the refresh for the next apply_defaults() call.
                pass

    @property
    def objective(self) -> Optional[str]:
        return self._objective

    @objective.setter
    def objective(self, value: Optional[str]) -> None:
        self._objective = _normalize_optional_input_text(value, lowercase=True)

    @property
    def goal(self) -> Optional[str]:
        return self._goal

    @goal.setter
    def goal(self, value: Optional[str]) -> None:
        self._goal = _normalize_optional_input_text(value)

    def _render_validation_message(self, code: Optional[str], facts: dict[str, Any], fallback: Optional[str]) -> Optional[str]:
        if code:
            return compose_response(code, **facts)
        text = " ".join(str(fallback or "").split()).strip()
        return text or None

    @property
    def symbol_validation_message(self) -> Optional[str]:
        return self._render_validation_message(
            self.symbol_validation_code,
            self.symbol_validation_facts,
            self._symbol_validation_message,
        )

    @symbol_validation_message.setter
    def symbol_validation_message(self, value: Optional[str]) -> None:
        self.set_symbol_validation(None, message=value)

    @property
    def input_validation_message(self) -> Optional[str]:
        return self._render_validation_message(
            self.input_validation_code,
            self.input_validation_facts,
            self._input_validation_message,
        )

    @input_validation_message.setter
    def input_validation_message(self, value: Optional[str]) -> None:
        self.set_input_validation(None, message=value)

    @property
    def timeframe_validation_message(self) -> Optional[str]:
        return self._render_validation_message(
            self.timeframe_validation_code,
            self.timeframe_validation_facts,
            self._timeframe_validation_message,
        )

    @timeframe_validation_message.setter
    def timeframe_validation_message(self, value: Optional[str]) -> None:
        self.set_timeframe_validation(None, message=value)

    # ── Risk params: one store, read-through properties (WS1 §6) ──────────────
    # The single home for SL/TP/RR is `risk_execution_config`. These properties
    # read and write that dict so legacy call sites doing `builder.stop_loss`
    # keep working while there is exactly ONE place the value lives — killing
    # the double-store drift where an edit applied to one mirror and not the
    # other (and the `if attribute is None` guards that papered over it).
    def _risk_store(self) -> dict[str, Any]:
        store = getattr(self, "risk_execution_config", None)
        if not isinstance(store, dict):
            store = {}
            self.risk_execution_config = store
        return store

    @property
    def stop_loss(self) -> Optional[float]:
        v = self._risk_store().get("stop_loss_pct")
        return float(v) if v is not None else None

    @stop_loss.setter
    def stop_loss(self, value: Optional[float]) -> None:
        store = self._risk_store()
        if value is None:
            store.pop("stop_loss_pct", None)
        else:
            store["stop_loss_pct"] = float(value)

    @property
    def take_profit(self) -> Optional[float]:
        v = self._risk_store().get("take_profit_pct")
        return float(v) if v is not None else None

    @take_profit.setter
    def take_profit(self, value: Optional[float]) -> None:
        store = self._risk_store()
        if value is None:
            store.pop("take_profit_pct", None)
        else:
            store["take_profit_pct"] = float(value)

    @property
    def risk_reward(self) -> Optional[Any]:
        return self._risk_store().get("risk_reward")

    @risk_reward.setter
    def risk_reward(self, value: Optional[Any]) -> None:
        store = self._risk_store()
        if value is None:
            store.pop("risk_reward", None)
        else:
            store["risk_reward"] = value

    def set_symbol_validation(
        self,
        code: Optional[str],
        facts: Optional[dict[str, Any]] = None,
        *,
        message: Optional[str] = None,
    ) -> None:
        self.symbol_validation_code = code
        self.symbol_validation_facts = dict(facts or {})
        self._symbol_validation_message = " ".join(str(message or "").split()).strip() or None

    def set_input_validation(
        self,
        code: Optional[str],
        facts: Optional[dict[str, Any]] = None,
        *,
        message: Optional[str] = None,
    ) -> None:
        self.input_validation_code = code
        self.input_validation_facts = dict(facts or {})
        self._input_validation_message = " ".join(str(message or "").split()).strip() or None

    def set_timeframe_validation(
        self,
        code: Optional[str],
        facts: Optional[dict[str, Any]] = None,
        *,
        message: Optional[str] = None,
    ) -> None:
        self.timeframe_validation_code = code
        self.timeframe_validation_facts = dict(facts or {})
        self._timeframe_validation_message = " ".join(str(message or "").split()).strip() or None

    def clear_validation_state(self) -> None:
        self.set_symbol_validation(None)
        self.set_input_validation(None)
        self.set_timeframe_validation(None)

    def set_risk_execution_config(self, config: Optional[dict[str, Any]]) -> None:
        self.risk_execution_config = dict(config or {})

    def has_any_data(self) -> bool:
        return any([
            self.market,
            self.symbol,
            self.timeframe,
            self.sentiment,
            self.experience,
            self.objective,
            self.goal,
            self.signal_plan,
        ])

    def is_user_input_complete(self) -> bool:
        # Phase 9 — for dynamic-discovery presets, the symbol is optional from
        # the user's side: the runtime scanner fills it in. Once
        # discovered_symbol is set OR the preset has discovery declared
        # (so the chat layer will trigger the scan), we treat the symbol
        # requirement as satisfied for the purpose of moving past input
        # collection.
        symbol_ok = bool(self.symbol) or bool(self.discovered_symbol) or self.requires_discovery()
        if not all([
            symbol_ok,
            _normalize_optional_input_text(self.timeframe),
            _normalize_optional_input_text(self.objective),
            _normalize_optional_input_text(self.sentiment),
            _normalize_optional_input_text(self.experience),
            _normalize_optional_input_text(self.goal),
        ]):
            return False
        return True

    def requires_discovery(self) -> bool:
        """True when the active preset declares a discovery block AND the
        user hasn't pinned a symbol manually. Used as the signal for the
        chat layer to dispatch the universe scanner."""
        if self.symbol:
            return False
        preset_name = (self.strategy_preset or "").strip().lower()
        if not preset_name:
            return False
        try:
            from app.kb import kb as _kb
            preset = _kb.presets.get(preset_name)
        except Exception:
            return False
        if preset is None:
            return False
        discovery = preset.discovery or {}
        return bool(discovery.get("enabled", False) and discovery.get("conditions"))

    def missing_user_input_fields(self) -> list[str]:
        fields: list[str] = []
        # Same exemption as is_user_input_complete — discovery-driven presets
        # don't require a user-supplied symbol.
        if not self.symbol and not self.discovered_symbol and not self.requires_discovery():
            fields.append("symbol")
        if not _normalize_optional_input_text(self.timeframe):
            fields.append("timeframe")
        if not _normalize_optional_input_text(self.objective):
            fields.append("objective")
        if not _normalize_optional_input_text(self.sentiment):
            fields.append("sentiment")
        if not _normalize_optional_input_text(self.experience):
            fields.append("experience")
        if not _normalize_optional_input_text(self.goal):
            fields.append("goal")
        return fields

    def reset_generated_strategy_state(self) -> None:
        self.user_input_confirmed = False
        self.signal_plan = None
        self.entry_condition = None
        self.exit_condition = None
        self.daily_loss_cap = None
        self.max_trade = None
        # SL/TP live in the single risk store. Clear only values WE defaulted —
        # a user-supplied stop/target accumulates across turns and must survive
        # a regenerate (WS1). With the old double-store this happened implicitly
        # (the attr was cleared but apply_defaults restored the dict value); we
        # make it explicit and provenance-aware now.
        self._clear_defaulted_risk_param("stop_loss_pct")
        self._clear_defaulted_risk_param("take_profit_pct")

    def _clear_defaulted_risk_param(self, key: str) -> None:
        """Drop a risk-store value unless it was user-supplied."""
        store = self._risk_store()
        sources = store.get("rms_sources")
        if isinstance(sources, dict) and sources.get(key) == "user":
            return
        store.pop(key, None)
        if isinstance(sources, dict):
            sources.pop(key, None)

    def _clear_core_user_input_field(self, field: str) -> None:
        if field == "symbol":
            self.symbol = None
            self.set_symbol_validation(None)
        elif field == "timeframe":
            self.timeframe = None
            self.set_timeframe_validation(None)
        elif field == "objective":
            self.objective = None
        elif field == "sentiment":
            self.sentiment = None
        elif field == "experience":
            self.experience = None
        elif field == "goal":
            self.goal = None
        self.set_input_validation(None)

    def request_input_modification(self, fields: Optional[list[str]] = None) -> None:
        selected_fields = [
            field
            for field in (fields or [])
            if field in CORE_USER_INPUT_FIELDS
        ]
        self.input_modification_requested = True
        self.pending_input_modification_fields = selected_fields
        self.reset_generated_strategy_state()
        for field in selected_fields:
            self._clear_core_user_input_field(field)

    def refresh_pending_input_modification_fields(self) -> None:
        if not self.input_modification_requested:
            self.pending_input_modification_fields = []
            return

        self.pending_input_modification_fields = [
            field
            for field in self.pending_input_modification_fields
            if not getattr(self, field, None)
        ]
        if not self.pending_input_modification_fields:
            self.input_modification_requested = False

    def clear_input_modification_request(self) -> None:
        self.input_modification_requested = False
        self.pending_input_modification_fields = []

    def get_mode(self) -> str:
        if not self.is_user_input_complete() or not self.user_input_confirmed:
            return "collect_user_input"
        if not self.signal_plan:
            return "plan_signals"
        return "assemble_strategy"

    def ready_for_preview(self) -> bool:
        return self.is_complete()

    def validate_parameter_combinations(self) -> list[dict]:
        """
        Smart cross-parameter validation. Returns a list of warning/suggestion
        dicts, each with keys: level ("warning" | "suggestion"), field, message.
        Call this before assembling to surface issues to the user or log.
        """
        issues: list[dict] = []

        # ── SL/TP ratio sanity ─────────────────────────────────────────────────
        sl = self.stop_loss
        tp = self.take_profit
        if sl is not None and tp is not None and sl > 0:
            rr = tp / sl
            if rr < 1.0:
                issues.append({
                    "level": "warning",
                    "field": "take_profit",
                    "message": f"Risk:Reward is {rr:.2f}:1 — TP is smaller than SL. "
                               f"You need at least a 50% win rate to break even. Consider widening TP.",
                })
            elif rr < 1.5 and self.experience == "beginner":
                issues.append({
                    "level": "suggestion",
                    "field": "take_profit",
                    "message": "RR of {:.1f}:1 is tight for a beginner — suggest at least 1.5:1.".format(rr),
                })

        # ── Intraday + wide SL ────────────────────────────────────────────────
        if self.objective == "intraday" and sl is not None and sl > 2.0:
            issues.append({
                "level": "warning",
                "field": "stop_loss",
                "message": f"A {sl:.1f}% stop on an intraday strategy is very wide. "
                           "Consider tightening to < 1.5% or using a structural SL.",
            })

        # ── Positional + very tight SL ─────────────────────────────────────────
        if self.objective == "positional" and sl is not None and sl < 0.5:
            issues.append({
                "level": "warning",
                "field": "stop_loss",
                "message": f"A {sl:.1f}% stop on a positional strategy will get stopped out by normal noise. "
                           "Typical positional SL is 1.5–3%.",
            })

        # ── max_spread_bps with high max_confirmation_bars ─────────────────────
        ecb = self.entry_confirmation_bars
        if ecb is not None and ecb > 3 and self.objective == "intraday":
            issues.append({
                "level": "suggestion",
                "field": "entry_confirmation_bars",
                "message": f"entry_confirmation_bars={ecb} on an intraday strategy may miss most moves — "
                           "use 1–2 for intraday.",
            })

        # ── RSI band too narrow ────────────────────────────────────────────────
        rmin = self.rsi_entry_band_min
        rmax = self.rsi_entry_band_max
        if rmin is not None and rmax is not None:
            if rmax - rmin < 10:
                issues.append({
                    "level": "warning",
                    "field": "rsi_entry_band",
                    "message": f"RSI band [{rmin}–{rmax}] is very narrow — trades will be extremely rare. "
                               "Suggest widening to at least 20 points.",
                })

        # ── Volume ratio threshold above 3× is very restrictive ───────────────
        vrt = self.volume_ratio_threshold
        if vrt is not None and vrt > 3.0:
            issues.append({
                "level": "suggestion",
                "field": "volume_ratio_threshold",
                "message": f"volume_ratio_threshold={vrt}× is very restrictive — "
                           "most intraday entries require only 1.5–2× average volume.",
            })

        # ── Gap filter + intraday (info) ───────────────────────────────────────
        if self.gap_filter not in (None, "none") and self.objective == "positional":
            issues.append({
                "level": "suggestion",
                "field": "gap_filter",
                "message": "gap_filter is most useful on intraday strategies. "
                           "For positional, gaps often represent genuine price discovery.",
            })

        # ── Bearish sentiment but direction=long_only ──────────────────────────
        if self.sentiment == "bearish" and self.direction == "long_only":
            issues.append({
                "level": "warning",
                "field": "direction",
                "message": "Sentiment is bearish but direction=long_only — you won't be taking "
                           "short trades even though your view is bearish. Intended?",
            })

        return issues

    def is_complete(self) -> bool:
        return all([
            self.is_user_input_complete(),
            self.signal_plan,
            self.entry_condition,
            self.stop_loss is not None,
            self.take_profit is not None,
        ])

    def missing_fields(self) -> list[str]:
        fields = self.missing_user_input_fields()
        if not self.signal_plan:
            fields.append("signal_plan")
        if not self.entry_condition:
            fields.append("entry_condition")
        if self.stop_loss is None:
            fields.append("stop_loss_pct")
        if self.take_profit is None:
            fields.append("take_profit_pct")
        return fields

    def apply_tier_execution_defaults(self) -> None:
        """Apply Phase-2 execution defaults from risk_tiers.yaml (keyed by experience).

        Each tier-derived field is refreshed from the **current** tier unless
        the user has explicitly set it (tracked via _phase10_user_overrides).
        Previously this method only filled None fields, which meant that once
        a strategy was built with one experience tier and the user later
        switched to another, the original tier's values stuck around and
        silently drove the simulator — a stale-defaults leak.
        """
        from app.kb import kb

        tier = kb.get_risk_tier(self.experience)

        def _set_if_not_overridden(name: str, value: Any) -> None:
            if name not in self._phase10_user_overrides:
                setattr(self, name, value)

        _set_if_not_overridden("max_consecutive_losses",     int(tier.max_consecutive_losses))
        _set_if_not_overridden("cooldown_bars_after_loss",   int(tier.cooldown_bars_after_loss))
        _set_if_not_overridden("cooldown_bars_after_profit", int(tier.cooldown_bars_after_profit))
        _set_if_not_overridden("max_spread_bps",             float(tier.max_spread_bps))
        _set_if_not_overridden("entry_confirmation_bars",    int(tier.entry_confirmation_bars))
        _set_if_not_overridden("max_capital_allocation_pct", float(tier.max_capital_allocation_pct))
        _set_if_not_overridden("position_sizing_mode",       str(tier.position_sizing_mode))
        _set_if_not_overridden("gap_filter",                 str(tier.gap_filter))

    def apply_defaults(self):
        cfg = MARKET_CONFIG.get(self.market or "", {})
        risk_cfg = self.risk_execution_config or {}
        from app.services.strategy.risk_defaults import (
            resolve_default_stop_loss,
            resolve_default_take_profit,
        )

        # SL/TP defaults come from the ONE resolver and are tagged `default` in
        # the risk store's provenance so they (a) never masquerade as user input
        # and (b) get cleared on a regenerate (see reset_generated_strategy_state).
        store = self._risk_store()
        rms_sources = store.get("rms_sources")
        if not isinstance(rms_sources, dict):
            rms_sources = {}
            store["rms_sources"] = rms_sources

        if self.stop_loss is None:
            d = resolve_default_stop_loss(self.objective, self.experience)
            self.stop_loss = d.value
            rms_sources.setdefault("stop_loss_pct", d.source)
        if self.take_profit is None:
            d = resolve_default_take_profit(
                self.objective, self.experience, stop_loss=self.stop_loss
            )
            self.take_profit = d.value
            rms_sources.setdefault("take_profit_pct", d.source)
        if self.timeframe is None:
            self.timeframe = cfg.get("default_timeframe", "1d")
        if self.daily_loss_cap is None:
            if risk_cfg.get("daily_loss_cap") is not None:
                self.daily_loss_cap = float(risk_cfg["daily_loss_cap"])
            else:
                tier_risk = get_experience_risk(self.experience or "")
                recommended = float(tier_risk.get("daily_loss_cap_pct", 2.0))
                min_cap, max_cap = get_daily_loss_cap_bounds()
                self.daily_loss_cap = max(min_cap, min(max_cap, recommended))

        self.apply_tier_execution_defaults()

        if self.max_trade is None:
            objective = (self.objective or "").lower().strip()
            if objective == "intraday":
                self.max_trade = "1 trading day"   # display label for UI only
            elif objective == "positional":
                self.max_trade = "5 trading days"  # display label for UI only
            else:
                self.max_trade = "3 trading days"  # display label for UI only

    def _reward_factor(self) -> Optional[float]:
        stop_loss = self.stop_loss
        take_profit = self.take_profit
        if stop_loss is None or take_profit is None:
            from app.services.strategy.risk_defaults import (
                resolve_default_stop_loss,
                resolve_default_take_profit,
            )
            stop_loss = resolve_default_stop_loss(self.objective, self.experience).value
            take_profit = resolve_default_take_profit(
                self.objective, self.experience, stop_loss=stop_loss
            ).value
        if not stop_loss or stop_loss <= 0:
            return None
        return round(float(take_profit) / float(stop_loss), 2)

    def _per_trade_risk_pct(self) -> float:
        tier = get_experience_risk(self.experience or "")
        return float(tier.get("per_trade_risk_pct", 2.0))

    def _max_trades_per_day(self) -> int:
        objective = (self.objective or "").lower().strip()
        return 2 if objective == "intraday" else 1

    @staticmethod
    def _format_max_trades_display(max_trades: int) -> str:
        return "Unlimited trades per day" if max_trades <= 0 else f"{max_trades} trades per day"

    @staticmethod
    def _format_risk_reward_display(risk_reward: Any) -> Optional[str]:
        if risk_reward is None:
            return None
        raw_value = str(risk_reward).strip()
        if not raw_value:
            return None
        if ":" in raw_value:
            return raw_value
        try:
            return f"{float(raw_value):g}:1"
        except ValueError:
            return raw_value

    def _max_trades_display(self) -> str:
        objective = (self.objective or "").lower().strip()
        return "2 trades per day" if objective == "intraday" else "1 trade per day"

    def risk_and_execution_summary(self, mode: Optional[str] = None) -> Optional[dict]:
        resolved_mode = mode or self.get_mode()
        if resolved_mode not in {"assemble_strategy", "backtest_confirmation", "backtest_complete"}:
            return None

        risk_cfg = self.risk_execution_config or {}
        risk_reward = self._format_risk_reward_display(risk_cfg.get("risk_reward"))
        if risk_reward is None:
            reward_factor = self._reward_factor()
            risk_reward = f"{reward_factor:g}:1" if reward_factor is not None else None

        daily_loss_cap_pct = (
            float(risk_cfg["daily_loss_cap"])
            if risk_cfg.get("daily_loss_cap") is not None
            else (
                self.daily_loss_cap
                if self.daily_loss_cap is not None
                else 3.0
            )
        )
        per_trade_risk_pct = (
            float(risk_cfg["per_trade_risk"])
            if risk_cfg.get("per_trade_risk") is not None
            else self._per_trade_risk_pct()
        )
        max_trades = (
            int(risk_cfg["max_trades"])
            if risk_cfg.get("max_trades") is not None
            else self._max_trades_per_day()
        )
        # Trading window display: explicit user bounds > asset-class default >
        # risk_cfg seed. The seed in app/kb/compat.py hardcodes "9:15 - 15:30"
        # regardless of asset class, so we override it for known markets.
        ew_start = self.entry_window_start
        ew_end   = self.entry_window_end
        market_label = _default_trading_window_label_for_market(self.market)
        if ew_start and ew_end:
            trading_window_display = f"{ew_start} - {ew_end}"
        elif market_label is not None:
            trading_window_display = market_label
        else:
            trading_window_display = str(
                risk_cfg.get("trading_window", DEFAULT_RISK_AND_EXECUTION["trading_window"])
            )

        return {
            "daily_loss_cap": f"{daily_loss_cap_pct:.1f}% of capital per day",
            "position_sizing": str(
                risk_cfg.get("position_sizing", DEFAULT_RISK_AND_EXECUTION["position_sizing"])
            ),
            "execution_mode": str(
                risk_cfg.get("execution_mode", DEFAULT_RISK_AND_EXECUTION["execution_mode"])
            ),
            "trading_window": trading_window_display,
            "risk_validation": str(
                risk_cfg.get("risk_validation", DEFAULT_RISK_AND_EXECUTION["risk_validation"])
            ),
            "per_trade_risk": f"{per_trade_risk_pct:.1f}% of capital per trade",
            "risk_reward": risk_reward,
            "max_trades": self._format_max_trades_display(max_trades),
        }

    def sync_market_from_symbol(self) -> None:
        """Re-derive ``self.market`` from the current symbol.

        Symbol is the source-of-truth for asset class inside a session. This
        method is called from every site that assigns ``self.symbol`` (chat
        resolution, merge_preview, etc.) so a stale "indian_stocks" default
        can't survive into a BTC_USDT strategy. If the symbol is ambiguous
        and we can't infer anything, the prior market value is preserved.
        """
        inferred = _infer_market_from_symbol(self.symbol)
        if inferred:
            self.market = inferred

    @property
    def asset_class(self) -> Optional[str]:
        """KB-style asset class ('equity_cash' | 'crypto_spot') derived from market.

        Surfaced in to_draft_json() so every chat response carries it.
        """
        return _MARKET_TO_ASSET_CLASS.get((self.market or "").lower().strip())

    def format_symbol(self) -> str:
        if not self.symbol:
            return ""
        raw = self.symbol.upper().strip()

        if raw.endswith((".NS", ".BO")):
            return raw

        # Crypto pairs use the canonical form BASE_QUOTE (e.g. BTC_USDT).
        # They are already canonical — do not apply the equity .NS/.BO logic.
        if re.match(r"^[A-Z0-9]+_[A-Z0-9]+$", raw):
            return raw

        if ":" in raw:
            exchange, clean = raw.split(":", 1)
            if exchange in {"BSE", "BO"}:
                return _format_symbol_with_exchange(clean, "BO")
            if exchange in {"NSE", "NS"}:
                return _format_symbol_with_exchange(clean, "NS")
            return clean

        suffix = MARKET_CONFIG.get(self.market or "", {}).get("suffix", "")
        clean = _strip_symbol_suffix(raw)
        if suffix == ".BO":
            return _format_symbol_with_exchange(clean, "BO")
        if suffix == ".NS":
            return _format_symbol_with_exchange(clean, "NS")
        return f"{clean}{suffix}"

    def extract_indicators(self) -> dict:
        # Direct path supplies declared indicators + their exact params (e.g.
        # MACD {fast,slow,signal}); use them verbatim so the read-back shows what
        # the user actually requested instead of regex-extracted bare names.
        direct = getattr(self, "_direct_indicators", None)
        if direct:
            return dict(direct)
        self._normalize_legacy_signal_conditions()
        combined = f"{self.entry_condition or ''} {self.exit_condition or ''}"
        out: dict[str, list[int]] = {}
        for indicator in ("SMA", "EMA", "RSI", "ADX", "BB_UPPER", "BB_LOWER"):
            matches = re.findall(rf"{indicator}\((\d+)\)", combined, re.IGNORECASE)
            if matches:
                out[indicator] = sorted(set(map(int, matches)))
        if "MACD" in combined.upper():
            out["MACD"] = []
        if "VWAP" in combined.upper():
            out["VWAP"] = []
        return out

    @staticmethod
    def _normalise_market(raw: str) -> str:
        return DEFAULT_MARKET

    def _normalize_legacy_signal_conditions(self) -> None:
        if not isinstance(self.signal_plan, dict) or not self.entry_condition:
            return
        if "CLOSE > HIGH" not in self.entry_condition:
            return

        updated_condition = self.entry_condition
        changed = False
        for signal in self.signal_plan.get("entry", []):
            if not isinstance(signal, dict):
                continue
            if signal.get("name") != "opening_range_breakout":
                continue

            params = signal.get("params") if isinstance(signal.get("params"), dict) else {}
            opening_bars = int(params.get("opening_bars", 4) or 4)
            updated_condition = updated_condition.replace(
                "CLOSE > HIGH",
                f"CLOSE > OPENING_RANGE_HIGH({opening_bars})",
            )
            changed = True

        if changed:
            self.entry_condition = updated_condition

    def modify_signal(
        self,
        slot: str,
        signal_name: str,
        *,
        params: dict | None = None,
        replace_name: str | None = None,
        action: str = "replace",
    ) -> dict:
        """Add, replace, or swap a single signal in the current ``signal_plan``.

        Args:
            slot: ``"entry"`` or ``"exit"`` — which list to update.
            signal_name: KB signal name to place (already resolved to an
                exact name; use ``app.kb.signal_resolver`` for partial input).
            params: optional params dict for the signal.
            replace_name: full name of an existing signal to swap out
                in-place. When set, ``action`` is ignored and the matching
                entry is replaced (or the new signal is appended if no
                match is found).
            action: how to apply when ``replace_name`` is not given.
                ``"replace"`` (default) wipes the slot and sets it to the
                new signal — the right behavior for "change entry signal
                to X" phrasing. ``"add"`` appends without removing existing
                entries — for "add X" phrasing.

        Returns:
            The updated ``signal_plan`` dict.

        Raises:
            SignalPlacementValidationError: if ``signal_name`` cannot legally
                sit in ``slot`` (e.g. entry-only signal in exit list).
            ValueError: if ``slot`` is not ``"entry"`` or ``"exit"`` or
                ``action`` is not ``"replace"`` / ``"add"``.
        """
        slot_clean = str(slot or "").strip().lower()
        if slot_clean not in ("entry", "exit"):
            raise ValueError(f"slot must be 'entry' or 'exit', got {slot!r}")

        action_clean = str(action or "replace").strip().lower()
        if action_clean not in ("replace", "add"):
            raise ValueError(f"action must be 'replace' or 'add', got {action!r}")

        clean_name = str(signal_name or "").strip()
        if not clean_name:
            raise ValueError("signal_name must be non-empty.")

        if not isinstance(self.signal_plan, dict):
            self.signal_plan = {"entry": [], "exit": []}
        self.signal_plan.setdefault("entry", [])
        self.signal_plan.setdefault("exit", [])

        # Strict trader-mode validation of just the *new* signal. We do
        # not re-validate existing entries because planner-generated plans
        # legitimately mix entry filters into the entry list — re-checking
        # would surface false positives for prior, already-accepted picks.
        err = validate_signal_placement(clean_name, slot_clean, mode="strict")
        if err is not None:
            raise SignalPlacementValidationError([err])

        # Validation passed — commit the change.
        entries: list[dict] = self.signal_plan[slot_clean]
        new_entry = {"name": clean_name, "params": dict(params or {})}

        if replace_name:
            # Explicit swap — replace the named signal in place.
            for i, item in enumerate(entries):
                if isinstance(item, dict) and item.get("name") == replace_name:
                    entries[i] = new_entry
                    break
            else:
                entries.append(new_entry)
        elif action_clean == "replace":
            # No specific target named, but the trader said "change/swap/use".
            # Wipe the slot so the new signal becomes the sole occupant —
            # this matches "change entry signal to X" expectations rather
            # than silently appending to the prior planner-generated set.
            entries.clear()
            entries.append(new_entry)
        else:
            # action == "add" — append without disturbing existing entries.
            entries.append(new_entry)
        return self.signal_plan

    def remove_signal(self, slot: str, signal_name: str) -> dict:
        """Remove the first occurrence of ``signal_name`` from the ``slot`` list.

        Returns the updated ``signal_plan`` (or an empty plan dict if there was
        nothing to remove). Silent no-op if the signal is not present.
        """
        slot_clean = str(slot or "").strip().lower()
        if slot_clean not in ("entry", "exit"):
            raise ValueError(f"slot must be 'entry' or 'exit', got {slot!r}")
        if not isinstance(self.signal_plan, dict):
            return {}
        entries = self.signal_plan.get(slot_clean) or []
        for i, item in enumerate(entries):
            if isinstance(item, dict) and item.get("name") == signal_name:
                entries.pop(i)
                break
        self.signal_plan[slot_clean] = entries
        return self.signal_plan

    def apply_signal_plan(self, plan: dict):
        self.signal_plan = {
            key: value
            for key, value in (plan or {}).items()
            if key in _SIGNAL_PLAN_PUBLIC_KEYS
        }
        self.apply_defaults()
        if plan.get("entry_condition"):
            self.entry_condition = plan["entry_condition"]
        if plan.get("exit_condition"):
            self.exit_condition = plan["exit_condition"]
        # Phase 3 — capture optional SL/trailing specs the planner attached.
        # Underscore keys keep them out of the public signal_plan view.
        sl_spec = plan.get("_stop_loss_spec")
        if isinstance(sl_spec, dict) and sl_spec:
            self.stop_loss_spec = dict(sl_spec)
        ts_spec = plan.get("_trailing_stop_spec")
        if isinstance(ts_spec, dict) and ts_spec:
            self.trailing_stop_spec = dict(ts_spec)
        ref_sym = plan.get("_reference_symbol")
        if isinstance(ref_sym, str) and ref_sym.strip():
            self.reference_symbol = ref_sym.strip().upper()
        htf = plan.get("_htf_rules")
        if isinstance(htf, list) and htf:
            # Defensive copy + shape normalisation so downstream consumers
            # don't see partial/typed-wrong entries.
            cleaned: list[dict[str, Any]] = []
            for item in htf:
                if not isinstance(item, dict):
                    continue
                tf = str(item.get("timeframe") or "").strip().lower()
                cond = str(item.get("condition") or "").strip()
                if tf and cond:
                    cleaned.append({"timeframe": tf, "condition": cond})
            if cleaned:
                self.htf_rules = cleaned
        time_exit = plan.get("_time_exit")
        if isinstance(time_exit, dict) and time_exit.get("exit_time"):
            # Carry through only the public fields; the loader recomputes
            # utc_minutes_of_day at strategy-load time.
            self.time_exit = {
                "exit_time": str(time_exit["exit_time"]).strip(),
                "timezone":  str(time_exit.get("timezone") or "Asia/Kolkata").strip(),
            }
        self._normalize_legacy_signal_conditions()

    def merge_preview(self, preview: dict):
        if not preview:
            return

        if preview.get("market"):
            self.market = self._normalise_market(preview["market"])
        # A symbol-bound validation error (e.g. "TATAMOTORS not found") is about
        # ONE symbol. If this merge changes the symbol to a different one, that
        # error is stale and must NOT be carried over or restored — otherwise a
        # freshly-chosen INFY inherits the old TATAMOTORS error (WS1 task 5).
        _incoming_symbol = (self.symbol or "").strip()
        _preview_symbol = str(preview.get("symbol") or "").strip()
        _symbol_changed = bool(_preview_symbol) and _preview_symbol != _incoming_symbol
        if preview.get("symbol"):
            self.symbol = preview["symbol"]
            # Symbol → market is the canonical direction; always override the
            # market (or default) we may have just set above. Crypto symbols
            # accidentally carrying market="indian_stocks" from an earlier
            # session must NOT inherit the NSE trading window.
            self.sync_market_from_symbol()
            if _symbol_changed and _incoming_symbol:
                # The builder already carried a different symbol this turn; any
                # validation error it holds was about that prior symbol → clear it.
                self.set_symbol_validation(None)
        # Only restore a symbol-bound validation error from the draft when it
        # still pertains to the symbol now in effect (i.e. the merge didn't swap
        # the symbol out from under it).
        _validation_is_stale = _symbol_changed and bool(_incoming_symbol)
        if preview.get("symbol_validation_code") and not _validation_is_stale:
            self.set_symbol_validation(
                str(preview["symbol_validation_code"]).strip(),
                preview.get("symbol_validation_facts") if isinstance(preview.get("symbol_validation_facts"), dict) else {},
            )
        elif preview.get("symbol_validation_message") and not _validation_is_stale:
            self.set_symbol_validation(None, message=str(preview["symbol_validation_message"]).strip())
        if preview.get("input_validation_code"):
            self.set_input_validation(
                str(preview["input_validation_code"]).strip(),
                preview.get("input_validation_facts") if isinstance(preview.get("input_validation_facts"), dict) else {},
            )
        elif preview.get("input_validation_message"):
            self.set_input_validation(None, message=str(preview["input_validation_message"]).strip())
        if preview.get("timeframe"):
            self.timeframe = preview["timeframe"]
        if preview.get("timeframe_validation_code"):
            self.set_timeframe_validation(
                str(preview["timeframe_validation_code"]).strip(),
                preview.get("timeframe_validation_facts") if isinstance(preview.get("timeframe_validation_facts"), dict) else {},
            )
        elif preview.get("timeframe_validation_message"):
            self.set_timeframe_validation(None, message=str(preview["timeframe_validation_message"]).strip())
        if preview.get("sentiment"):
            self.sentiment = str(preview["sentiment"]).lower().strip()
        if preview.get("experience"):
            self.experience = str(preview["experience"]).lower().strip()
        if preview.get("objective"):
            self.objective = str(preview["objective"]).lower().strip()
        if preview.get("goal"):
            self.goal = " ".join(str(preview["goal"]).split()).strip()
        # original_user_prompt: never overwrite once set — it's the immutable
        # record of the user's first substantive message describing the
        # strategy. The agent router will keep summarizing into `goal` but
        # this anchor must survive turn boundaries.
        if preview.get("original_user_prompt") and not self.original_user_prompt:
            self.original_user_prompt = str(preview["original_user_prompt"]).strip()
        if preview.get("daily_loss_cap_pct") is not None:
            self.daily_loss_cap = float(preview["daily_loss_cap_pct"])
        if preview.get("max_trade"):
            self.max_trade = str(preview["max_trade"]).strip()
        if preview.get("entry_condition"):
            self.entry_condition = preview["entry_condition"]
        if preview.get("exit_condition"):
            self.exit_condition = preview["exit_condition"]
        if preview.get("short_entry_condition"):
            self.short_entry_condition = preview["short_entry_condition"]
        if preview.get("short_exit_condition"):
            self.short_exit_condition = preview["short_exit_condition"]
        # Restore the SDL ticket so a modify turn edits THIS strategy. Tolerant:
        # a malformed/legacy draft just leaves _sdl as None (legacy modify path).
        sdl_json = preview.get("sdl_json")
        if isinstance(sdl_json, dict) and sdl_json:
            try:
                from app.planner.sdl import SDL
                self._sdl = SDL.model_validate(sdl_json)
            except Exception:
                self._sdl = None
        # Single risk store: restore the whole risk_execution_config FIRST, then
        # overlay any explicit stop_loss_pct / take_profit_pct from the draft.
        # (stop_loss/take_profit are read-through properties over this dict, so
        # the previous order — set SL/TP, then replace the dict — silently wiped
        # the just-set values whenever the persisted config omitted them.)
        risk_execution_config = preview.get("risk_execution_config")
        if isinstance(risk_execution_config, dict):
            self.set_risk_execution_config(risk_execution_config)
        if preview.get("stop_loss_pct") is not None:
            self.stop_loss = float(preview["stop_loss_pct"])
        if preview.get("take_profit_pct") is not None:
            self.take_profit = float(preview["take_profit_pct"])
        # Restore structural SL and trailing stop specs persisted in the draft
        stop_loss_spec = preview.get("stop_loss_spec")
        if isinstance(stop_loss_spec, dict) and stop_loss_spec:
            self.stop_loss_spec = stop_loss_spec
        trailing_stop_spec = preview.get("trailing_stop_spec")
        if isinstance(trailing_stop_spec, dict) and trailing_stop_spec:
            self.trailing_stop_spec = trailing_stop_spec
        semantic_intent = preview.get("semantic_intent")
        if isinstance(semantic_intent, dict) and semantic_intent:
            self.semantic_intent = semantic_intent

        # Phase 10 — restore new strategy parameters
        _p10_str_fields = (
            "direction", "entry_window_start", "entry_window_end",
            "gap_filter", "position_sizing_mode",
        )
        for _f in _p10_str_fields:
            v = preview.get(_f)
            if v is not None:
                setattr(self, _f, str(v).strip() or None)

        _p10_int_fields = (
            "max_consecutive_losses", "cooldown_bars_after_loss",
            "cooldown_bars_after_profit", "entry_confirmation_bars",
        )
        for _f in _p10_int_fields:
            v = preview.get(_f)
            if v is not None:
                try:
                    setattr(self, _f, int(v))
                except (TypeError, ValueError):
                    pass

        _p10_float_fields = (
            "max_spread_bps", "gap_threshold_pct",
            "rsi_entry_band_min", "rsi_entry_band_max",
            "volume_ratio_threshold", "max_capital_allocation_pct",
        )
        for _f in _p10_float_fields:
            v = preview.get(_f)
            if v is not None:
                try:
                    setattr(self, _f, float(v))
                except (TypeError, ValueError):
                    pass

        # Rehydrate the user-override set so tier refreshes triggered on
        # subsequent turns continue to honour fields the user already
        # explicitly set on a prior turn.
        _overrides = preview.get("_phase10_user_overrides")
        if isinstance(_overrides, (list, tuple, set)):
            self._phase10_user_overrides = {
                str(name) for name in _overrides
                if str(name) in self._TIER_DERIVED_PHASE10_FIELDS
            }

        if preview.get("user_input_confirmed"):
            self.user_input_confirmed = True
        if preview.get("signal_plan"):
            self.signal_plan = preview["signal_plan"]
        if preview.get("strategy_preset"):
            self.strategy_preset = str(preview["strategy_preset"]).strip().lower()
        self.input_modification_requested = bool(preview.get("input_modification_requested"))
        pending_fields = preview.get("pending_input_modification_fields")
        if isinstance(pending_fields, list):
            self.pending_input_modification_fields = [
                str(field)
                for field in pending_fields
                if str(field) in CORE_USER_INPUT_FIELDS
            ]
        if not self.input_modification_requested:
            self.pending_input_modification_fields = []

        # Phase 9j — restore discovery state. The chat layer relies on
        # these fields persisting across turns: discovery_pending lets
        # the next user reply route to handle_pending_tie_break;
        # discovery_parameter_overrides keeps a user-typed "1.2x"
        # threshold alive while the user supplies remaining inputs
        # (timeframe, sentiment, ...) over multiple turns; the
        # discovered_symbol is reused once a stock has been picked.
        if "discovery_pending" in preview:
            self.discovery_pending = bool(preview.get("discovery_pending"))
        if "discovery_no_match" in preview:
            self.discovery_no_match = bool(preview.get("discovery_no_match"))
        candidates = preview.get("discovery_candidates")
        if isinstance(candidates, list):
            self.discovery_candidates = [
                dict(c) for c in candidates if isinstance(c, dict)
            ] or None
        options = preview.get("discovery_tie_break_options")
        if isinstance(options, list):
            self.discovery_tie_break_options = [
                dict(o) for o in options if isinstance(o, dict)
            ] or None
        method = preview.get("discovery_chosen_method")
        if isinstance(method, str) and method.strip():
            self.discovery_chosen_method = method.strip()
        discovered = preview.get("discovered_symbol")
        if isinstance(discovered, str) and discovered.strip():
            self.discovered_symbol = discovered.strip()
        overrides = preview.get("discovery_parameter_overrides")
        if isinstance(overrides, dict) and overrides:
            self.discovery_parameter_overrides = {
                str(k): float(v) for k, v in overrides.items()
                if isinstance(v, (int, float))
            } or None
        used = preview.get("discovery_parameters_used")
        if isinstance(used, dict) and used:
            self.discovery_parameters_used = {
                str(k): float(v) for k, v in used.items()
                if isinstance(v, (int, float))
            } or None
        # Phase 9m — restore the per-scan diagnostic snapshot.
        diag = preview.get("discovery_last_scan_diagnostics")
        if isinstance(diag, dict) and diag:
            self.discovery_last_scan_diagnostics = dict(diag)
        # Phase 9k — restore the parsed primitive list. Each item must
        # be a {name, params} dict; malformed entries are dropped so a
        # corrupt draft can't crash the orchestrator.
        conditions = preview.get("discovery_conditions")
        if isinstance(conditions, list):
            cleaned: list[dict] = []
            for item in conditions:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                params_raw = item.get("params") if isinstance(item.get("params"), dict) else {}
                params = {
                    str(k): float(v) for k, v in params_raw.items()
                    if isinstance(v, (int, float))
                }
                cleaned.append({"name": name, "params": params})
            self.discovery_conditions = cleaned or None
        # Issue 2 — restore an explicit user-supplied universe override.
        # Each entry must be a non-empty string; malformed items are
        # dropped and the field collapses to None if the list ends up empty.
        override = preview.get("discovery_universe_override")
        if isinstance(override, list):
            cleaned_universe = [
                str(sym).strip() for sym in override
                if isinstance(sym, str) and str(sym).strip()
            ]
            self.discovery_universe_override = cleaned_universe or None

        self._normalize_legacy_signal_conditions()

    def to_draft_json(
        self,
        mode_override: Optional[str] = None,
        processing_status: Optional[str] = None,
    ) -> dict:
        resolved_mode = mode_override or self.get_mode()
        draft = {
            "mode": resolved_mode,
            "market": self.market,
            # Asset class is derived from market and always surfaced so the UI
            # (and any downstream consumer) doesn't have to re-infer it from
            # the symbol. "equity_cash" | "crypto_spot" | None when unknown.
            "asset_class": self.asset_class,
            "symbol": self.format_symbol(),
            "symbol_validation_code": self.symbol_validation_code,
            "symbol_validation_facts": self.symbol_validation_facts,
            "symbol_validation_message": self.symbol_validation_message,
            "input_validation_code": self.input_validation_code,
            "input_validation_facts": self.input_validation_facts,
            "input_validation_message": self.input_validation_message,
            "timeframe": self.timeframe,
            "timeframe_validation_code": self.timeframe_validation_code,
            "timeframe_validation_facts": self.timeframe_validation_facts,
            "timeframe_validation_message": self.timeframe_validation_message,
            "sentiment": self.sentiment,
            "experience": self.experience,
            "objective": self.objective,
            "goal": self.goal,
            # original_user_prompt MUST be persisted across turns: builder is
            # rebuilt on every chat turn and merge_preview() restores it from
            # this draft. Without this serialization, the field resets to
            # None after the first turn, defeating the fidelity validator.
            "original_user_prompt": self.original_user_prompt,
            "entry_condition": self.entry_condition,
            "exit_condition": self.exit_condition,
            "short_entry_condition": self.short_entry_condition,
            "short_exit_condition": self.short_exit_condition,
            # SDL ticket (JSON) so the next turn can MODIFY this strategy via the
            # SDL path. None for legacy-built strategies.
            "sdl_json": (
                self._sdl.model_dump(mode="json")
                if getattr(self, "_sdl", None) is not None else None
            ),
            "stop_loss_pct": self.stop_loss,
            "take_profit_pct": self.take_profit,
            "indicators": self.extract_indicators(),
            "user_input_confirmed": self.user_input_confirmed,
            "signal_plan": self.signal_plan,
            "strategy_preset": self.strategy_preset,
            "risk_execution_config": self.risk_execution_config or {},
            "stop_loss_spec": self.stop_loss_spec,
            "trailing_stop_spec": self.trailing_stop_spec,
            "input_modification_requested": self.input_modification_requested,
            "pending_input_modification_fields": self.pending_input_modification_fields,
            "processing_status": processing_status or ("complete" if self.signal_plan else "in_progress"),
            # Phase 9j — persist the discovery state machine. Without
            # these fields, anything the chat layer learns mid-flow
            # about discovery (preset overrides typed by the user,
            # pending tie-break candidates, the resolved symbol) is
            # lost the next turn because the builder gets rehydrated
            # from this draft. The tie-break flow + parameter overrides
            # (Phases 9b/9h/9i) all depend on these surviving across
            # turns.
            "discovery_pending":             self.discovery_pending,
            "discovery_no_match":            self.discovery_no_match,
            "discovery_candidates":          self.discovery_candidates,
            "discovery_tie_break_options":   self.discovery_tie_break_options,
            "discovery_chosen_method":       self.discovery_chosen_method,
            "discovered_symbol":             self.discovered_symbol,
            "discovery_parameter_overrides": self.discovery_parameter_overrides,
            "discovery_parameters_used":     self.discovery_parameters_used,
            "discovery_conditions":          self.discovery_conditions,
            "discovery_last_scan_diagnostics": self.discovery_last_scan_diagnostics,
            "discovery_universe_override":   self.discovery_universe_override,
            # Semantic intent: plain dict so it round-trips cleanly through JSON.
            "semantic_intent":               self.semantic_intent,
            # Phase 10 — new strategy parameters
            "direction":                     self.direction,
            "entry_window_start":            self.entry_window_start,
            "entry_window_end":              self.entry_window_end,
            "max_consecutive_losses":        self.max_consecutive_losses,
            "cooldown_bars_after_loss":      self.cooldown_bars_after_loss,
            "cooldown_bars_after_profit":    self.cooldown_bars_after_profit,
            "max_spread_bps":                self.max_spread_bps,
            "gap_filter":                    self.gap_filter,
            "gap_threshold_pct":             self.gap_threshold_pct,
            "entry_confirmation_bars":       self.entry_confirmation_bars,
            "rsi_entry_band_min":            self.rsi_entry_band_min,
            "rsi_entry_band_max":            self.rsi_entry_band_max,
            "volume_ratio_threshold":        self.volume_ratio_threshold,
            "position_sizing_mode":          self.position_sizing_mode,
            "max_capital_allocation_pct":    self.max_capital_allocation_pct,
            # Sorted list so the draft JSON is stable across saves (sets
            # serialise in unspecified order otherwise). The receiver
            # converts back to a set in from_preview_json.
            "_phase10_user_overrides":       sorted(self._phase10_user_overrides),
        }

        return draft

    def _numeric_max_trades_per_day(self) -> int:
        """
        Convert the display label stored in `max_trade` to a numeric per-day trade limit.

        "1 trading day", "5 trading days" → these are duration labels, NOT trade counts.
        For those we return 0 (unlimited). Only a plain numeric string like "10" is treated
        as an actual per-day trade count.
        """
        if self.max_trade is None:
            return 0
        raw = str(self.max_trade).strip()
        # Accept only purely numeric strings as trade counts
        if raw.isdigit():
            return int(raw)
        # Descriptive labels ("1 trading day", "5 trading days") → unlimited
        return 0

    def to_yaml_dict(self) -> dict:
        self.apply_defaults()
        self._normalize_legacy_signal_conditions()
        plan = self.signal_plan or {}
        entry_signals: list[dict] = []
        for sig in plan.get("entry") or []:
            if isinstance(sig, dict) and sig.get("name"):
                entry_signals.append(
                    {"name": sig["name"], "params": dict(sig.get("params") or {})}
                )
        exit_signals: list[dict] = []
        for sig in plan.get("exit") or []:
            if isinstance(sig, dict) and sig.get("name"):
                exit_signals.append(
                    {"name": sig["name"], "params": dict(sig.get("params") or {})}
                )

        # Lenient safety net: the entry list legitimately contains filter
        # signals from the planner output (planner appends entry_filter
        # matches to entry_signals so the engine evaluates them as AND
        # filters). Strict trader-mode validation is enforced earlier in
        # `modify_signal` and `apply_signal_request`. This gate only
        # catches hand-edited YAML or out-of-band mutations that put an
        # exit-only signal in entry / vice versa.
        assert_valid_signal_lists(
            [sig["name"] for sig in entry_signals],
            [sig["name"] for sig in exit_signals],
            mode="lenient",
        )

        strategy_block: dict = {
            "name": f"{self.format_symbol()} {self.timeframe} Strategy",
            "symbol": self.format_symbol(),
            "market": self.market,
            "timeframe": self.timeframe,
            "profile": {
                "sentiment": self.sentiment,
                "experience": self.experience,
                "objective": self.objective,
                "goal": self.goal,
                "daily_loss_cap_pct": self.daily_loss_cap,
                "max_trade":          self.max_trade,             # display label (UI only)
                "max_trades_per_day": self._numeric_max_trades_per_day(),  # numeric for engine
            },
            "variables": {
                "TAKE_PROFIT_TARGET": self.take_profit,
                "STOP_LOSS_TARGET":   self.stop_loss,
            },
            "entry": {"condition": self.entry_condition or "CLOSE > EMA(20)"},
            "exit": {
                "condition": self.exit_condition
                or "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
            },
            "indicators": self.extract_indicators(),
            "risk_management": {
                "stop_loss_percent":  self.stop_loss,
                "take_profit_percent": self.take_profit,
            },
        }
        # Phase 3 — pass through structural / trailing SL specs at the strategy
        # level so the loader picks them up. Both are optional; when omitted
        # the simulator falls back to the legacy stop_loss_percent.
        if isinstance(self.stop_loss_spec, dict) and self.stop_loss_spec:
            strategy_block["stop_loss"] = dict(self.stop_loss_spec)
        if isinstance(self.trailing_stop_spec, dict) and self.trailing_stop_spec:
            strategy_block["trailing_stop"] = dict(self.trailing_stop_spec)
        if isinstance(self.reference_symbol, str) and self.reference_symbol:
            strategy_block["reference_symbol"] = self.reference_symbol
        if isinstance(self.htf_rules, list) and self.htf_rules:
            strategy_block["htf"] = [dict(r) for r in self.htf_rules]
        if isinstance(self.time_exit, dict) and self.time_exit.get("exit_time"):
            strategy_block["time_exit"] = dict(self.time_exit)
        if entry_signals:
            strategy_block["entry_evaluation_mode"] = "registry"
            strategy_block["entry_signals"] = entry_signals
        if exit_signals:
            strategy_block["exit_evaluation_mode"] = "registry"
            strategy_block["exit_signals"] = exit_signals

        # Phase 10 — direction constraint
        if self.direction and self.direction != "both":
            strategy_block["direction"] = self.direction

        # Phase 12 — short leg. When the SDL produced a two-sided strategy the
        # short conditions are formula strings; pass them through so the engine
        # runs a separate short pass. Force direction=both so both passes run.
        if self.short_entry_condition:
            strategy_block["short_entry_condition"] = self.short_entry_condition
            strategy_block["short_exit_condition"] = (
                self.short_exit_condition
                or "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
            )
            strategy_block["direction"] = "both"

        # Phase 10 — entry window. If the user explicitly set either bound,
        # those win. Otherwise apply an asset-class default: Indian stocks get
        # the NSE 09:15–15:30 IST session; crypto gets nothing (24/7 market).
        ew_start = self.entry_window_start
        ew_end   = self.entry_window_end
        if not (ew_start or ew_end):
            ew_start, ew_end = _default_entry_window_for_market(self.market)
        if ew_start or ew_end:
            # Crypto trades 24/7 with no IST session — its window times are UTC.
            # Equity (and unknown markets) keep IST. Emitting the right timezone
            # here keeps the YAML self-consistent (the engine also coerces crypto
            # to UTC, but a correct YAML matters for the live path / readability).
            ew_tz = "UTC" if (self.market or "").strip().lower() == "crypto" else "Asia/Kolkata"
            strategy_block["entry_window"] = {
                k: v for k, v in {
                    "start": ew_start,
                    "end":   ew_end,
                    "timezone": ew_tz,
                }.items() if v
            }

        # Phase 10 — risk circuit breakers (only write when non-zero / non-default)
        if self.max_consecutive_losses:
            strategy_block["max_consecutive_losses"] = self.max_consecutive_losses
        if self.cooldown_bars_after_loss:
            strategy_block["cooldown_bars_after_loss"] = self.cooldown_bars_after_loss
        if self.cooldown_bars_after_profit:
            strategy_block["cooldown_bars_after_profit"] = self.cooldown_bars_after_profit

        # Phase 10 — entry gate controls
        if self.max_spread_bps:
            strategy_block["max_spread_bps"] = self.max_spread_bps
        if self.gap_filter and self.gap_filter != "none":
            strategy_block["gap_filter"] = self.gap_filter
        if self.gap_threshold_pct is not None:
            strategy_block["gap_threshold_pct"] = self.gap_threshold_pct
        if self.entry_confirmation_bars and self.entry_confirmation_bars > 1:
            strategy_block["entry_confirmation_bars"] = self.entry_confirmation_bars
        if self.rsi_entry_band_min is not None or self.rsi_entry_band_max is not None:
            strategy_block["rsi_entry_band"] = {
                k: v for k, v in {
                    "min": self.rsi_entry_band_min,
                    "max": self.rsi_entry_band_max,
                }.items() if v is not None
            }
        if self.volume_ratio_threshold is not None:
            strategy_block["volume_ratio_threshold"] = self.volume_ratio_threshold

        # Phase 10 — position sizing
        if self.position_sizing_mode and self.position_sizing_mode != "fixed_fractional":
            strategy_block["position_sizing_mode"] = self.position_sizing_mode
        if self.max_capital_allocation_pct is not None and self.max_capital_allocation_pct < 100.0:
            strategy_block["max_capital_allocation_pct"] = self.max_capital_allocation_pct

        return {"strategy": strategy_block}

def _match_alias_symbol(text: str, exchange: Optional[str] = None) -> Optional[str]:
    upper = text.upper()
    compact = re.sub(r"[^A-Z0-9]", "", upper)

    for raw, symbol in _SYMBOL_ALIASES.items():
        if len(raw) <= 3:
            if re.search(rf"\b{re.escape(raw)}\b", upper):
                return _format_symbol_with_exchange(symbol, exchange)
            continue
        if re.search(rf"\b{re.escape(raw)}\b", upper) or raw in compact:
            return _format_symbol_with_exchange(symbol, exchange)
    return None


def _extract_symbol(text: str) -> Optional[str]:
    upper = text.upper()
    exchange = _exchange_hint(text)
    nse_symbols = _get_nse_symbol_set()

    explicit_suffix_match = re.search(r"\b([A-Z][A-Z0-9&-]{0,19})\.(NS|BO)\b", upper)
    if explicit_suffix_match:
        return _format_symbol_with_exchange(
            explicit_suffix_match.group(1),
            explicit_suffix_match.group(2),
        )

    explicit_exchange_match = re.search(r"\b(NSE|BSE)\s*[:\-]\s*([A-Z][A-Z0-9&-]{0,19})\b", upper)
    if explicit_exchange_match:
        return _format_symbol_with_exchange(
            explicit_exchange_match.group(2),
            "BO" if explicit_exchange_match.group(1) == "BSE" else "NS",
        )

    natural_exchange_match = re.search(
        r"\b([A-Z][A-Z0-9&-]{0,19})\s+(?:ON|IN)\s+(NSE|BSE)\b",
        upper,
    )
    if natural_exchange_match:
        candidate = natural_exchange_match.group(1)
        if (
            candidate in nse_symbols
            or candidate in _SYMBOL_ALIASES
            or candidate in _SYMBOL_ALIASES.values()
        ):
            return _format_symbol_with_exchange(
                candidate,
                "BO" if natural_exchange_match.group(2) == "BSE" else "NS",
            )

    alias_symbol = _match_alias_symbol(text, exchange)
    if alias_symbol:
        return alias_symbol

    for token in re.findall(r"\b[A-Z][A-Z0-9&-]{1,19}\b", upper):
        if token in {"NSE", "BSE", "NS", "BO"}:
            continue
        if token in nse_symbols:
            return _format_symbol_with_exchange(token, exchange)

    search_query = _extract_search_query(text)
    if search_query:
        return _search_indian_equity(search_query, exchange)

    return None


def _extract_rms_from_text(text: str) -> dict[str, Any]:
    """Parse risk-management settings from free-form user text.

    Returns a dict of extracted fields with provenance source="user".
    Only fields that are explicitly mentioned are included.
    All values are stored under canonical keys that mirror risk_execution_config.
    """
    rms: dict[str, Any] = {}

    # ── Stop loss ─────────────────────────────────────────────────────────────
    # "SL 2%", "stop loss at 1.5%", "stop at 1%", "2% stop".
    # Keyword-first is tried before digit-first so that in "Stop Loss: 0.50%
    # Take Profit: 1.00%" the keyword-anchored "0.50%" wins and the digit-first
    # alternative can't grab the take-profit number that follows a later keyword.
    sl_match = re.search(
        r"\b(?:stop[\s\-]*loss|stoploss|sl|stop)\s*[=:@]?\s*(?:(?:at|to)\s+)?"
        r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:%|percent|pct)(?=\s|$|[,.\)])?",
        text, re.IGNORECASE,
    ) or re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s+(?:stop|sl|stop[\s\-]*loss)\b",
        text, re.IGNORECASE,
    )
    if sl_match:
        val = float(sl_match.group(1) or 0)
        if val > 0:
            rms["stop_loss_pct"] = {"value": val, "source": "user"}

    # ── Take profit ───────────────────────────────────────────────────────────
    # "TP 5%", "target 3%", "take profit 4%", "3% target", "3% take profit".
    # Same keyword-first precedence as stop loss above.
    tp_match = re.search(
        r"\b(?:take[\s\-]*profit|takeprofit|tp|target|tgt)\s*[=:@]?\s*(?:(?:at|to)\s+)?"
        r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:%|percent|pct)(?=\s|$|[,.\)])?",
        text, re.IGNORECASE,
    ) or re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s+"
        r"(?:take[\s\-]*profit|takeprofit|target|tp|profit)\b",
        text, re.IGNORECASE,
    )
    if tp_match:
        val = float(tp_match.group(1) or 0)
        if val > 0:
            rms["take_profit_pct"] = {"value": val, "source": "user"}

    # ── Risk:Reward ratio ─────────────────────────────────────────────────────
    # Captures N:M ratios and resolves the order from any nearby keyword.
    #
    # Ordering rule:
    #   * The keyword that is *immediately adjacent* to the digits (within a
    #     ~25-char window before or after) determines which side is reward and
    #     which is risk. If the closest keyword starts with "reward" the first
    #     digit is reward; otherwise it is risk.
    #   * "3:7 reward risk ratio" → reward=3, risk=7, ratio=3/7≈0.43
    #   * "5:2 risk to reward"    → risk=5, reward=2, ratio=2/5=0.40
    #   * "1:3 RR" / "R:R 1:2"    → "rr" is symmetric — assume the legacy
    #                               1:N convention (first is risk).
    rr_keyword_re = re.compile(
        r"\b("
        r"reward(?:[\s/\-]+to)?[\s/\-]+risk(?:[\s\-]*ratio)?"
        r"|risk(?:[\s/\-]+to)?[\s/\-]+reward(?:[\s\-]*ratio)?"
        r"|reward[\s\-]*ratio"
        r"|risk[\s\-]*ratio"
        r"|r\s*:?\s*r"
        r"|rr"
        r")\b",
        re.IGNORECASE,
    )

    def _classify(label: str) -> str:
        """Return 'reward_first', 'risk_first', or 'symmetric'."""
        l = re.sub(r"\s+", " ", label.strip().lower())
        if l in {"rr", "r:r", "r r"}:
            return "symmetric"
        # Whichever word appears first names the first digit.
        ri = l.find("risk")
        re_ = l.find("reward")
        if ri == -1:
            return "reward_first"
        if re_ == -1:
            return "risk_first"
        return "reward_first" if re_ < ri else "risk_first"

    rr_val: float | None = None
    rr_resolved = False
    # Walk every N:M occurrence and try to attach the nearest keyword.
    for digit_match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\b", text):
        a = float(digit_match.group(1))
        b = float(digit_match.group(2))
        if a <= 0 or b <= 0:
            continue
        d_start, d_end = digit_match.span()
        # Look ±25 chars for a keyword
        window = text[max(0, d_start - 25): min(len(text), d_end + 25)]
        kw_match = rr_keyword_re.search(window)
        if kw_match:
            order = _classify(kw_match.group(1))
            if order == "reward_first":
                rr_val = a / b
            elif order == "risk_first":
                rr_val = b / a
            else:  # symmetric ("rr") → legacy 1:N convention (risk first)
                rr_val = b / a if abs(a - 1.0) < 1e-9 else a / b
            rr_resolved = True
            break
        # No keyword nearby — record as ambiguous if RR keywords exist anywhere
        if rr_keyword_re.search(text):
            rms.setdefault("risk_reward_ambiguous", {
                "raw": f"{a:g}:{b:g}",
                "candidates": [round(a / b, 4), round(b / a, 4)],
                "source": "user",
            })

    # Final legacy fallback: bare "1:N" anywhere
    if not rr_resolved:
        m = re.search(r"\b1\s*[:/]\s*(\d+(?:\.\d+)?)\b", text)
        if m:
            rr_val = float(m.group(1))

    # Bare-ratio fallback: "2.5 risk-reward", "2 RR", "RR of 1.5", "1.5R reward".
    # Captures a single float adjacent to a reward/risk keyword (no colon/slash).
    if rr_val is None:
        bare = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:r\b|risk[\s\-]*reward|reward[\s\-]*risk|risk[\s:]*reward|r:?r)"
            r"|(?:rr|risk[\s\-]*reward|reward[\s\-]*risk|risk[\s:]*reward)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)",
            text, re.IGNORECASE,
        )
        if bare:
            for g in bare.groups():
                if g:
                    try:
                        candidate = float(g)
                    except (TypeError, ValueError):
                        continue
                    # Reasonable RR range to avoid false positives (e.g. timeframe "5m" or year "2025")
                    if 0.1 <= candidate <= 20.0:
                        rr_val = candidate
                        break

    if rr_val is not None and rr_val > 0:
        rms["risk_reward"] = {"value": round(rr_val, 4), "source": "user"}
        rms.pop("risk_reward_ambiguous", None)

    # ── Per-trade risk ────────────────────────────────────────────────────────
    # Allow intervening adverbs: "risk only 2%", "risk just 1%", "only risk 1%".
    # The "\s*[=:@]?\s*" after the keyword lets "Per Trade Risk: 1%" match — the
    # "Label: value" form was previously dropped because only whitespace was
    # allowed between the keyword and the number.
    ptr_match = re.search(
        r"\b(?:risk|per[\s\-]*trade[\s\-]*risk|per\s+trade)\s*[=:@]?\s*"
        r"(?:only\s+|just\s+|max(?:imum)?\s+|up\s+to\s+|at\s+most\s+)?"
        r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(%|percent|pct)\s*(?:of\s+capital|per\s+trade)?"
        r"|\brisk\s*[=:@]?\s*(?:only\s+|just\s+|max(?:imum)?\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\b",
        text, re.IGNORECASE,
    )
    if ptr_match:
        val = float(ptr_match.group(1) or ptr_match.group(3) or 0)
        if val > 0:
            rms["per_trade_risk"] = {"value": val, "source": "user"}

    # ── Daily loss cap ────────────────────────────────────────────────────────
    # Allow optional "of" / "at" / ":" between the keyword and the number.
    dlc_match = re.search(
        r"\b(?:daily[\s\-]*(?:loss[\s\-]*)?(?:cap|limit|sl|stop|loss)"
        r"|max[\s\-]*daily[\s\-]*loss|cap\s+loss(?:es)?|stop\s+trading\s+after)\s*[=:@]?\s*"
        r"(?:(?:a|an)\s+)?(?:of\s+|at\s+|to\s+)?(?:₹\s*)?"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|pct|k)?"
        r"|\bdaily\s+(?:sl|stop[\s\-]*loss)\s+(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\b",
        text, re.IGNORECASE,
    )
    if dlc_match:
        raw = (dlc_match.group(1) or dlc_match.group(3) or "0").replace(",", "")
        unit = (dlc_match.group(2) or "").lower()
        val = float(raw)
        if unit == "k":
            # ₹ absolute — skip % storage; user can clarify
            pass
        elif val > 0:
            rms["daily_loss_cap"] = {"value": val, "source": "user"}

    # ── Max open positions ────────────────────────────────────────────────────
    pos_match = re.search(
        r"\b(?:max(?:imum)?\s+(?:\d+|one|two|three)\s+(?:trade|position|open)"
        r"|one\s+trade\s+at\s+a\s+time|single\s+trade)\b",
        text, re.IGNORECASE,
    )
    if pos_match:
        snippet = pos_match.group(0).lower()
        if "one" in snippet or "single" in snippet:
            rms["max_positions"] = {"value": 1, "source": "user"}
        else:
            n_match = re.search(r"(\d+)", snippet)
            if n_match:
                rms["max_positions"] = {"value": int(n_match.group(1)), "source": "user"}

    # Explicit "max N trades" pattern.
    # NOTE: This is the user's per-session/per-day cap on OPEN positions — it
    # maps to the canonical `max_trades` field used by risk_execution_config
    # (NOT `max_consecutive_losses`, which is a different concept). The legacy
    # alias `max_positions` is also written for callers that still read it.
    max_match = re.search(r"\bmax(?:imum)?\s+(\d+)\s+(?:open\s+)?(?:trades?|positions?)\b", text, re.IGNORECASE)
    if max_match:
        n = int(max_match.group(1))
        rms.setdefault("max_trades",   {"value": n, "source": "user"})
        rms.setdefault("max_positions", {"value": n, "source": "user"})

    # ── Max consecutive losses ────────────────────────────────────────────────
    # Distinct from max_trades: this is a session-level circuit breaker.
    cl_match = re.search(
        r"\b(?:stop|pause|halt)\s+(?:trading\s+)?after\s+(\d+)\s+(?:consecutive\s+)?(?:losses?|losing\s+trades?)"
        r"|max(?:imum)?\s+(\d+)\s+consecutive\s+(?:losses?|losing\s+trades?)"
        r"|(\d+)\s+(?:straight|consecutive)\s+(?:losses?|losing\s+trades?)\s+(?:and\s+)?stop"
        r"|no\s+more\s+than\s+(\d+)\s+(?:consecutive\s+)?(?:losses?|losing\s+trades?)",
        text, re.IGNORECASE,
    )
    if cl_match:
        for g in cl_match.groups():
            if g:
                rms["max_consecutive_losses"] = {"value": int(g), "source": "user"}
                break

    # ── Cooldown after loss / profit ──────────────────────────────────────────
    cd_loss = re.search(
        r"\b(?:wait|cooldown|rest)\s+(?:for\s+)?(\d+)\s+(?:bars?|candles?)\s+after\s+(?:a\s+)?(?:loss|losing\s+trade|stop)"
        r"|(\d+)[\s\-]+bar\s+cooldown\s+(?:after|post)[\s\-]+(?:loss|stop|losing)"
        r"|cooldown\s+of\s+(\d+)\s+bars?\s+after\s+(?:a\s+)?(?:loss|losing)"
        r"|after\s+(?:a\s+)?(?:loss|stop)\s+wait\s+(\d+)\s+(?:bars?|candles?)",
        text, re.IGNORECASE,
    )
    if cd_loss:
        for g in cd_loss.groups():
            if g:
                rms["cooldown_bars_after_loss"] = {"value": int(g), "source": "user"}
                break
    cd_win = re.search(
        r"\b(?:wait|cooldown|rest)\s+(?:for\s+)?(\d+)\s+(?:bars?|candles?)\s+after\s+(?:a\s+)?(?:win|winning|profit|gain)"
        r"|(\d+)[\s\-]+bar\s+cooldown\s+(?:after|post)[\s\-]+(?:win|profit|winning|gain)"
        r"|cooldown\s+of\s+(\d+)\s+bars?\s+after\s+(?:a\s+)?(?:win|profit|winning|gain)"
        r"|after\s+(?:a\s+)?(?:win|profit)\s+wait\s+(\d+)\s+(?:bars?|candles?)",
        text, re.IGNORECASE,
    )
    if cd_win:
        for g in cd_win.groups():
            if g:
                rms["cooldown_bars_after_profit"] = {"value": int(g), "source": "user"}
                break

    # ── Max spread (bps) ──────────────────────────────────────────────────────
    sp_match = re.search(
        r"\b(?:max(?:imum)?\s+)?spread\s+(?:of\s+|below\s+|under\s+|<\s*|≤\s*)?(\d+(?:\.\d+)?)\s*(?:bps?|basis\s+points?)"
        r"|spread\s+(?:filter|gate|limit)\s+(\d+(?:\.\d+)?)\s*(?:bps?|basis\s+points?)"
        r"|(\d+(?:\.\d+)?)\s*(?:bps?|basis\s+points?)\s+(?:max(?:imum)?\s+)?spread",
        text, re.IGNORECASE,
    )
    if sp_match:
        for g in sp_match.groups():
            if g:
                rms["max_spread_bps"] = {"value": float(g), "source": "user"}
                break

    # ── Entry confirmation bars ───────────────────────────────────────────────
    cf_match = re.search(
        r"\b(\d+)[\s\-]+(?:bar|candle)\s+(?:close|confirmation|confirm)"
        r"|(?:wait\s+for\s+|need\s+)?(\d+)\s+(?:consecutive\s+)?(?:bars?|candles?)\s+(?:to\s+)?confirm"
        r"|confirmation\s+on\s+(\d+)\s+(?:bars?|candles?)"
        r"|(\d+)\s+bars?\s+(?:of\s+)?(?:confirmation|signal)",
        text, re.IGNORECASE,
    )
    if cf_match:
        for g in cf_match.groups():
            if g:
                rms["entry_confirmation_bars"] = {"value": int(g), "source": "user"}
                break

    # ── Max capital allocation per trade ──────────────────────────────────────
    # Must explicitly mention "allocation" / "max" / "deploy" / "exposure" so
    # we don't accidentally match generic "risk 2% capital per trade" (that's
    # `per_trade_risk`, not allocation).
    ca_match = re.search(
        # "20% maximum capital allocation per trade"
        r"\b(\d+(?:\.\d+)?)\s*%\s+(?:max(?:imum)?\s+)?(?:capital\s+)?allocation(?:\s+per\s+trade)?"
        # "maximum 20% capital allocation"
        r"|\bmax(?:imum)?\s+(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?capital(?:\s+allocation)?(?:\s+per\s+trade)?"
        # "allocate up to 20%" / "deploy 20%"
        r"|\b(?:allocate|deploy|expose)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*%"
        # "cap position size at 20%", "limit allocation to 20%"
        r"|\b(?:cap|limit)\s+(?:position\s+|trade\s+)?(?:size|allocation|exposure)\s+(?:to\s+|at\s+)?(\d+(?:\.\d+)?)\s*%"
        # "20% exposure per trade"
        r"|\b(\d+(?:\.\d+)?)\s*%\s+exposure(?:\s+per\s+trade)?",
        text, re.IGNORECASE,
    )
    if ca_match:
        for g in ca_match.groups():
            if g:
                rms["max_capital_allocation_pct"] = {"value": float(g), "source": "user"}
                break

    # ── Position sizing mode ──────────────────────────────────────────────────
    ps_match_table = (
        (r"risk[\s\-]+based(?:\s+(?:position\s+)?sizing)?",      "risk_based"),
        (r"size\s+(?:based\s+on|by)\s+risk",                     "risk_based"),
        (r"fixed[\s\-]+(?:fractional|fraction)(?:\s+sizing)?",   "fixed_fractional"),
        (r"(?:fixed|flat)\s+lot\s+size",                         "fixed_units"),
        (r"fixed\s+units?(?:\s+per\s+trade)?",                   "fixed_units"),
    )
    for pat, mode in ps_match_table:
        if re.search(pat, text, re.IGNORECASE):
            rms.setdefault("position_sizing_mode", {"value": mode, "source": "user"})
            break

    # ── Gap filter ────────────────────────────────────────────────────────────
    # Tolerate intervening words: "avoid trading on big gap-up or gap-down".
    # The [\s\S]{0,40} window catches phrases like "avoid trading on ... gap".
    has_avoid_kw = re.search(
        r"\b(?:ignore|avoid|skip|filter\s+out|no\s+trade(?:s)?\s+on|no\s+entries?\s+on)\b",
        text, re.IGNORECASE,
    )
    if has_avoid_kw:
        # Detect direction (up/down/both) within a 60-char window after the keyword
        kw_end = has_avoid_kw.end()
        window = text[kw_end: kw_end + 100]
        gap_up   = re.search(r"\bgap(?:s|[\s\-]+up)\b|\bgap[\s\-]+up\b", window, re.IGNORECASE)
        gap_down = re.search(r"\bgap[\s\-]+down\b", window, re.IGNORECASE)
        if gap_up and gap_down:
            rms.setdefault("gap_filter", {"value": "ignore_both", "source": "user"})
        elif gap_up:
            rms.setdefault("gap_filter", {"value": "ignore_gap_up", "source": "user"})
        elif gap_down:
            rms.setdefault("gap_filter", {"value": "ignore_gap_down", "source": "user"})
        elif re.search(r"\bgap(?:s|ping)?\b", window, re.IGNORECASE):
            rms.setdefault("gap_filter", {"value": "ignore_both", "source": "user"})

    # Gap threshold % — captures "gaps above 1%", "gap-up over 0.5%", "more
    # than 1% gap", "gaps larger than 0.8%".
    gap_thresh = re.search(
        r"\bgap(?:s|[\s\-]+up|[\s\-]+down|[\s\-]+open(?:ing)?)?\b"
        r"[\s\S]{0,30}?"
        r"(?:above|over|>|>=|≥|larger\s+than|more\s+than|greater\s+than|exceeding)"
        r"\s+(\d+(?:\.\d+)?)\s*(?:%|percent|pct)"
        r"|(?:gap(?:s)?\s+(?:above|over|>=|larger\s+than|more\s+than))\s+(\d+(?:\.\d+)?)\s*(?:%|percent|pct)"
        r"|(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s+gap(?:s|[\s\-]+up|[\s\-]+down)?",
        text, re.IGNORECASE,
    )
    if gap_thresh:
        for g in gap_thresh.groups():
            if g:
                try:
                    rms["gap_threshold_pct"] = {"value": float(g), "source": "user"}
                except (TypeError, ValueError):
                    pass
                break

    # ── ATR-based stop loss (drives stop_loss_spec.type = "atr") ──────────────
    atr_stop_match = re.search(
        r"\b(?:atr[\s\-]+based|atr)\s+(?:stop|sl|stop[\s\-]*loss)"
        r"|stop\s+loss\s+(?:based\s+on|of)\s+atr"
        r"|use\s+atr\s+(?:for\s+)?(?:stop|sl)"
        r"|(\d+(?:\.\d+)?)\s*x?\s*atr\s+(?:stop|sl|stop[\s\-]*loss)",
        text, re.IGNORECASE,
    )
    if atr_stop_match:
        multiplier = None
        for g in atr_stop_match.groups():
            if g:
                try:
                    multiplier = float(g)
                except (TypeError, ValueError):
                    pass
                break
        rms["atr_stop"] = {
            "value": True,
            "source": "user",
            "type": "atr",
            "multiplier": multiplier,
        }

    # ── Square-off / time exit ────────────────────────────────────────────────
    # Captures "square off before close" / "exit all by 3:15" / "close all
    # positions at 15:15" — sets a time_exit cutoff that the live execution
    # layer reads as a hard intraday close.
    so_match = re.search(
        # Verb phrase: "square off", "exit", "close" — optionally followed
        # by "all positions/trades" etc.
        r"\b(?:square[\s\-]+off|exit|close|flatten)\b"
        r"[\s\S]{0,40}?"
        # Cutoff: "market close", "EOD", "by 15:15", "before close"
        r"(?:before|by|at|prior\s+to)?\s*"
        r"(?:market[\s\-]+close|eod|end[\s\-]+of[\s\-]+day|(\d{1,2}:\d{2}))",
        text, re.IGNORECASE,
    )
    if so_match:
        explicit_time = so_match.group(1)
        # Indian market close default: 15:30 IST. "before close" typically
        # means a few minutes before (15:15 is the conventional intraday
        # square-off used by Indian brokers).
        if explicit_time:
            parts = explicit_time.split(":")
            cutoff = f"{int(parts[0]):02d}:{parts[1]}"
        elif "before" in so_match.group(0).lower():
            cutoff = "15:15"
        else:
            cutoff = "15:30"
        rms["time_exit"] = {
            "value": cutoff,
            "source": "user",
            "timezone": "Asia/Kolkata",
        }

    # ── HTF (higher-timeframe) confirmation flag ──────────────────────────────
    # "higher timeframe confirmation", "HTF filter", "1h confirmation",
    # "daily trend filter". We capture the FACT that the user wants HTF
    # gating; the actual rule (TF + condition) is built by the planner.
    htf_match = re.search(
        r"\bhigher[\s\-]+timeframe\s+(?:confirmation|filter|trend|alignment)"
        r"|\bhtf\s+(?:confirmation|filter|trend|alignment)"
        r"|(\d+[hmd])\s+(?:confirmation|filter|trend|alignment)"
        r"|daily\s+(?:trend\s+)?(?:filter|alignment)"
        r"|weekly\s+(?:trend\s+)?(?:filter|alignment)"
        r"|multi[\s\-]+timeframe(?:\s+confirmation)?",
        text, re.IGNORECASE,
    )
    if htf_match:
        explicit_tf = None
        for g in htf_match.groups():
            if g and re.match(r"\d+[hmd]", g, re.IGNORECASE):
                explicit_tf = g.lower()
                break
        # Default HTF heuristic: one step up from the strategy's chart TF
        rms["htf_confirmation"] = {
            "value": True,
            "source": "user",
            "explicit_timeframe": explicit_tf,
        }

    # ── Trailing stop ─────────────────────────────────────────────────────────
    # Captures:
    #   - "trail SL after 2%", "trailing stop 2%", "trail 2%" → distance
    #   - "trailing stop once trade reaches 2% profit" → ACTIVATION %
    #   - "activate trailing stop after 1% gain" → activation
    # We try the activation pattern first because "trailing stop … N% profit"
    # is unambiguous; only fall back to distance interpretation otherwise.
    activation_match = re.search(
        r"\btrail(?:ing)?\s+(?:stop|sl|stop[\s\-]*loss)\s*"
        r"(?:once|after|when)?[\s\S]{0,40}?"
        r"(?:reach(?:es)?|hit(?:s)?|cross(?:es)?|exceed(?:s)?|gain(?:s)?|profit\s+of|at)\s+"
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s*(?:profit|gain)?"
        r"|\bactivate\s+trail(?:ing)?\s+(?:stop|sl)?\s*"
        r"(?:after|when)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent|pct)"
        r"|\btrail(?:ing)?\s+(?:stop|sl)?\s*"
        r"(?:once|after)\s+(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s+(?:profit|gain)",
        text, re.IGNORECASE,
    )
    if activation_match:
        for g in activation_match.groups():
            if g:
                try:
                    rms["trailing_stop_activate_after_pct"] = {
                        "value": float(g), "source": "user",
                    }
                except (TypeError, ValueError):
                    pass
                break
    else:
        # Plain distance: "trail SL by 2%", "trailing stop 2%"
        trail_match = re.search(
            r"\btrail(?:ing)?[\s\-]*(?:stop|sl|stop[\s\-]*loss|stop\s*loss)?\s*"
            r"(?:after|of|by|at)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent|pct)(?=\s|$|[,.\)])?",
            text, re.IGNORECASE,
        )
        if trail_match:
            val = float(trail_match.group(1))
            if val > 0:
                rms["trailing_stop_pct"] = {"value": val, "source": "user"}

    # ── Trading window ────────────────────────────────────────────────────────
    # Also accept "only trades between 9:30 AM to 11:30 AM" / "between 9:30 and
    # 11:30" / "during 9:30-11:30". The "trade/entry" verb prefix is now
    # optional; instead we anchor on the time-of-day pair near a window keyword.
    tw_match = re.search(
        # Verb-prefixed: "trade only from 9:30 to 11:30"
        r"\b(?:trade|entry|entries|no\s+entries?|trades?\s+only)\s+(?:only\s+)?"
        r"(?:from\s+|between\s+|during\s+)?"
        r"(\d{1,2}:\d{2})(?:\s*am|\s*pm)?\s*(?:-|to|–|till|until|and)\s*"
        r"(\d{1,2}:\d{2})(?:\s*am|\s*pm)?"
        # Bare "between 9:30 AM to 11:30 AM"
        r"|\bbetween\s+(\d{1,2}:\d{2})(?:\s*am|\s*pm)?\s*"
        r"(?:-|to|–|till|until|and)\s*(\d{1,2}:\d{2})(?:\s*am|\s*pm)?",
        text, re.IGNORECASE,
    )
    if tw_match:
        groups = tw_match.groups()
        start = next((g for g in groups[:2] if g), None) or groups[2]
        end = (groups[1] or groups[3]) if len(groups) >= 4 else groups[1]
        if start and end:
            # Zero-pad single-digit hours: "9:30" → "09:30"
            def _pad(t: str) -> str:
                parts = t.split(":")
                if len(parts) == 2 and parts[0].isdigit():
                    return f"{int(parts[0]):02d}:{parts[1]}"
                return t
            start, end = _pad(start), _pad(end)
            rms["trading_window"] = {
                "value": f"{start}-{end}",
                "source": "user",
            }
            rms["entry_window_start"] = {"value": start, "source": "user"}
            rms["entry_window_end"]   = {"value": end,   "source": "user"}
    else:
        cutoff = re.search(
            r"\b(?:no\s+entries?\s+after|exit\s+(?:all\s+)?by|close\s+(?:all\s+)?by)\s+(\d{1,2}:\d{2})\b",
            text, re.IGNORECASE,
        )
        if cutoff:
            rms["trading_window"] = {
                "value": f"09:15-{cutoff.group(1)}",
                "source": "user",
            }
            rms["entry_window_start"] = {"value": "09:15", "source": "user"}
            rms["entry_window_end"]   = {"value": cutoff.group(1), "source": "user"}

    # ── Volume ratio threshold ────────────────────────────────────────────────
    # "volume at least 1.5x average", "1.5× the volume", "above-average volume of 2x"
    vol_match = re.search(
        r"\bvolume\s+(?:at\s+least\s+|above\s+|over\s+|>\s*|≥\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:x|×|times?)\s*(?:the\s+)?(?:avg|average|mean)?"
        r"|(\d+(?:\.\d+)?)\s*(?:x|×|times?)\s+(?:the\s+)?(?:avg|average|mean)\s+volume"
        r"|volume\s+(?:spike|surge)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:x|×)",
        text, re.IGNORECASE,
    )
    if vol_match:
        for g in vol_match.groups():
            if g:
                try:
                    rms["volume_ratio_threshold"] = {"value": float(g), "source": "user"}
                except (TypeError, ValueError):
                    pass
                break

    # ── Structural stop loss ──────────────────────────────────────────────────
    # "opening range low as structural stop loss", "SL at ORB candle low",
    # "structural SL", "swing low as stop", "9:15 candle low"
    struct_sl_match = re.search(
        r"\b(?:"
        r"opening\s+range\s+(?:low|high)(?:\s+as\s+(?:structural\s+)?(?:stop|sl|stop[\s\-]*loss))?"
        r"|orb\s+(?:candle\s+)?(?:low|high)(?:\s+as\s+(?:stop|sl))?"
        r"|(?:structural\s+(?:stop|sl|stop[\s\-]*loss))"
        r"|sl\s+(?:at|below)\s+(?:the\s+)?(?:9:15\s+candle|opening\s+range|orb)\s+(?:low|candle\s+low)?"
        r"|stop\s+(?:at|below)\s+(?:the\s+)?(?:9:15\s+candle|opening\s+range|orb)\s+(?:low)?"
        r"|swing\s+(?:low|high)\s+(?:as\s+)?(?:stop|sl|stop[\s\-]*loss)"
        r")\b",
        text, re.IGNORECASE,
    )
    if struct_sl_match:
        anchor_text = struct_sl_match.group(0).lower()
        if "swing low" in anchor_text:
            anchor = "swing_low"
        elif "high" in anchor_text:
            anchor = "opening_range_high"
        else:
            anchor = "opening_range_low"
        rms["structural_sl"] = {
            "value": anchor,
            "source": "user",
            "type": "structural",
        }

    # ── EMA trailing stop ─────────────────────────────────────────────────────
    # "trail profits using EMA trailing stop", "EMA trail", "trail by EMA"
    ema_trail_match = re.search(
        r"\b(?:ema\s+trail(?:ing)?(?:\s+stop)?|trail(?:ing)?\s+(?:profits?\s+using\s+)?ema"
        r"|trail\s+(?:using|by|with)\s+ema|ema[\s\-]*based\s+trail(?:ing)?)\b",
        text, re.IGNORECASE,
    )
    if ema_trail_match:
        period_match = re.search(r"\b(\d+)\s*(?:period\s+)?ema\b", text, re.IGNORECASE)
        rms["trailing_stop_type"] = {
            "value": "ema",
            "period": int(period_match.group(1)) if period_match else 20,
            "source": "user",
        }

    return rms


def extract_strategy_details(text: str, builder: StrategyBuilder):
    builder.market = DEFAULT_MARKET
    builder.clear_validation_state()

    timeframe_source = re.sub(r"\bdaily\s+loss\b", " ", text, flags=re.IGNORECASE)
    timeframe_match = _TF_RE.search(timeframe_source)
    if timeframe_match:
        canonical = normalise_timeframe(timeframe_match.group(1))
        if canonical:
            resolved_timeframe, validation_message = resolve_supported_user_timeframe(canonical)
            if resolved_timeframe:
                builder.timeframe = resolved_timeframe
            elif validation_message:
                builder.timeframe = None
                builder.set_timeframe_validation(
                    UNSUPPORTED_USER_TIMEFRAME_CODE,
                    unsupported_user_timeframe_validation_facts(),
                )

    # ── Sentiment — infer from directional language + strategy mechanics ──────
    if re.search(
        r"\b(bullish|optimistic|positive|long|buy|uptrend|upside"
        r"|breakout|breakouts|break\s*above|price\s*breaks?\s*above"
        r"|momentum\s*up|gap\s*up)\b",
        text, re.IGNORECASE,
    ):
        builder.sentiment = "bullish"
    elif re.search(
        r"\b(bearish|cautious|negative|short|sell|downtrend|downside"
        r"|breakdown|break\s*below|price\s*breaks?\s*below"
        r"|fade|reversal|momentum\s*down|gap\s*down)\b",
        text, re.IGNORECASE,
    ):
        builder.sentiment = "bearish"

    # ── Objective ─────────────────────────────────────────────────────────────
    if re.search(
        r"\b(intraday|day trade|day trading|scalp|scalping|quick profit|quick profits"
        r"|same day|today only|short term|btst)\b",
        text,
        re.IGNORECASE,
    ):
        builder.objective = "intraday"
    elif re.search(
        r"\b(positional|swing|steady growth|risk protection|long term|multi day|carry"
        r"|hold\s+(?:for\s+)?(?:a\s+)?(?:few|several|multiple)\s+days?)\b",
        text,
        re.IGNORECASE,
    ):
        builder.objective = "positional"

    # ── Experience — direct keyword matches take priority ────────────────────
    if not builder.experience:
        direct = re.search(
            r"\b(beginner|beginner[\s\-]+friendly|beginner[\s\-]+level|novice)\b",
            text, re.IGNORECASE,
        )
        if direct:
            builder.experience = "beginner"
        elif re.search(r"\b(intermediate|moderate[\s\-]+level)\b", text, re.IGNORECASE):
            builder.experience = "intermediate"
        elif re.search(r"\b(expert|advanced|pro|seasoned|experienced)\b", text, re.IGNORECASE):
            builder.experience = "expert"

    # ── Experience — infer from vocabulary sophistication (fallback) ─────────
    if not builder.experience:
        expert_cues = re.search(
            r"\b(orb|vwap\s+anchor|swing\s+low|swing\s+high|bos|fvg|fair\s*value\s*gap"
            r"|atr\s+multiple|atr\s+stop|atr\s+trail|structural\s+sl|structural\s+stop"
            r"|partial\s+exit|trail\s+sl|break\s*even\s*stop|hh\s*/\s*hl|lh\s*/\s*ll"
            r"|risk\s*:\s*reward|r\s*:\s*r|lot\s+size|position\s+sizing|per\s*trade\s*risk"
            r"|daily\s+loss\s+cap|max\s+drawdown|opening\s+range\s+low)\b",
            text, re.IGNORECASE,
        )
        beginner_cues = re.search(
            r"\b(?:what\s+is|i\s+don.?t\s+know|kuch\s+nahi\s+pata|i.?m\s+new"
            r"|explain|how\s+does|beginner|beginner[- ]friendly|beginner[- ]level"
            # "simple|basic|easy" followed by a strategy-type word
            r"|(?:simple|basic|easy)\s+(?:intraday|swing|strategy|trading|rsi|macd"
            r"|breakout|setup|trade|approach|momentum|trend|reversal|scalp)"
            r"|easy[\s\-]+to[\s\-]+follow"
            r"|start(?:ing)?\s+(?:with|from\s+scratch)"
            r"|never\s+traded|first\s+time)\b",
            text, re.IGNORECASE,
        )
        if expert_cues and not beginner_cues:
            builder.experience = "expert"
        elif beginner_cues:
            builder.experience = "beginner"

    goal_text = extract_goal_text(text)
    if goal_text:
        builder.goal = goal_text

    # ── RMS extraction ────────────────────────────────────────────────────────
    rms = _extract_rms_from_text(text)
    if rms:
        existing = dict(builder.risk_execution_config or {})
        rms_sources = dict(existing.get("rms_sources", {}))
        for field, entry in rms.items():
            # First-write-wins: never overwrite a user-supplied value
            if field not in existing or rms_sources.get(field) != "user":
                if isinstance(entry, dict) and "value" in entry:
                    existing[field] = entry["value"]
                    rms_sources[field] = entry.get("source", "user")
        existing["rms_sources"] = rms_sources
        builder.risk_execution_config = existing

        # stop_loss / take_profit need no mirror: they are read-through
        # properties over risk_execution_config["stop_loss_pct" / "take_profit_pct"]
        # which we just wrote above. daily_loss_cap is still a plain attribute.
        if "daily_loss_cap" in rms and builder.daily_loss_cap is None:
            builder.daily_loss_cap = rms["daily_loss_cap"]["value"]

        # Structural SL → builder.stop_loss_spec (overrides preset default only
        # when user explicitly mentions a structural anchor).
        if "structural_sl" in rms and builder.stop_loss_spec is None:
            sl_entry = rms["structural_sl"]
            builder.stop_loss_spec = {
                "type": "structural",
                "anchor": sl_entry["value"],
                "source": "user",
            }

        # EMA trailing stop → builder.trailing_stop_spec
        if "trailing_stop_type" in rms and builder.trailing_stop_spec is None:
            ts_entry = rms["trailing_stop_type"]
            if ts_entry["value"] == "ema":
                builder.trailing_stop_spec = {
                    "type": "ema",
                    "period": ts_entry.get("period", 20),
                    "source": "user",
                }
        # Percent trailing stop (from trailing_stop_pct)
        elif "trailing_stop_pct" in rms and builder.trailing_stop_spec is None:
            builder.trailing_stop_spec = {
                "type": "percent",
                "distance_pct": rms["trailing_stop_pct"]["value"],
                "source": "user",
            }

        # Trailing-stop activation threshold (no distance given) — the
        # engine treats `activate_after_pct` as the "kick in once profit
        # reaches N%" gate. We default distance to 1% if user didn't say.
        if "trailing_stop_activate_after_pct" in rms:
            activate = rms["trailing_stop_activate_after_pct"]["value"]
            if builder.trailing_stop_spec is None:
                builder.trailing_stop_spec = {
                    "type": "percent",
                    "activate_after_pct": activate,
                    "source": "user",
                }
            else:
                spec = dict(builder.trailing_stop_spec)
                spec.setdefault("activate_after_pct", activate)
                builder.trailing_stop_spec = spec

        # ATR-based stop loss → builder.stop_loss_spec
        if "atr_stop" in rms and builder.stop_loss_spec is None:
            atr_entry = rms["atr_stop"]
            spec: dict[str, Any] = {
                "type": "atr",
                "source": "user",
            }
            if atr_entry.get("multiplier") is not None:
                spec["multiplier"] = atr_entry["multiplier"]
            spec["window"] = 14
            builder.stop_loss_spec = spec

        # Square-off / time_exit → builder.time_exit
        if "time_exit" in rms and not builder.time_exit:
            te = rms["time_exit"]
            builder.time_exit = {
                "exit_time": te["value"],
                "timezone": te.get("timezone", "Asia/Kolkata"),
                "source": "user",
            }

        # HTF confirmation flag → seed builder.htf_rules with a sensible default
        # if empty. The orchestrator may refine this with a more specific
        # condition; we only ensure the user's "higher timeframe confirmation"
        # intent results in AT LEAST one rule.
        if "htf_confirmation" in rms and not builder.htf_rules:
            tf_map = {"5m": "1h", "15m": "1h", "1h": "4h", "1d": "1w", "30m": "1h", "10m": "1h"}
            explicit_tf = rms["htf_confirmation"].get("explicit_timeframe")
            chosen_tf = explicit_tf or tf_map.get(builder.timeframe or "", "1h")
            sentiment_dir = "bullish" if (builder.sentiment or "bullish") == "bullish" else "bearish"
            builder.htf_rules = [{
                "timeframe": chosen_tf,
                "condition": (
                    f"CLOSE > EMA(50)" if sentiment_dir == "bullish"
                    else f"CLOSE < EMA(50)"
                ),
                "source": "user",
                "intent": "trend_confirmation",
            }]

        # Gate fields → direct builder attributes (only fill when empty so the
        # user's earlier turn isn't overwritten).
        _gate_attr_fields = (
            "max_consecutive_losses",
            "cooldown_bars_after_loss",
            "cooldown_bars_after_profit",
            "max_spread_bps",
            "entry_confirmation_bars",
            "max_capital_allocation_pct",
            "gap_filter",
            "gap_threshold_pct",
            "position_sizing_mode",
            "max_trades",
            "entry_window_start",
            "entry_window_end",
            "volume_ratio_threshold",
        )
        for fld in _gate_attr_fields:
            if fld not in rms:
                continue
            entry = rms[fld]
            value = entry.get("value")
            if value is None:
                continue
            # max_trades lives on max_trade (display) + the rms dict; sync both
            if fld == "max_trades":
                if builder.max_trade is None:
                    builder.max_trade = f"{int(value)} trades per day"
            elif hasattr(builder, fld) and getattr(builder, fld) is None:
                # Numeric int / float coercion happens implicitly on assignment
                setattr(builder, fld, value)

    # ── Preset detection ──────────────────────────────────────────────────────
    # Always re-detect from the current message so a fresh message can correct
    # a stale/wrong preset stored from a prior turn. When no keyword matches,
    # the builder's existing preset is preserved (it won't be cleared).
    try:
        from app.kb import kb as _kb
        detected_preset = _kb.detect_preset_in_text(text)
    except Exception:
        detected_preset = None
    if detected_preset is not None:
        builder.strategy_preset = detected_preset.name
