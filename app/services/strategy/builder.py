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
from typing import Any, Optional

import requests

from app.kb.compat import (
    get_all_market_configs,
    get_daily_loss_cap_bounds,
    get_experience_risk,
    get_risk_defaults,
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

SUPPORTED_USER_TIMEFRAMES = (
    "1m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "1d",
)

SUPPORTED_USER_TIMEFRAME_TEXT = "1m, 5m, 10m, 15m, 30m, 1h, 1d"
UNSUPPORTED_USER_TIMEFRAME_CODE = "validation.unsupported_timeframe"
UNSUPPORTED_USER_TIMEFRAME_MESSAGE = build_unsupported_timeframe_message(
    SUPPORTED_USER_TIMEFRAME_TEXT
)
# Only accept exact equivalents (e.g. "60 minute" → 1h). Anything that does not
# canonicalize to a supported value is rejected and reported back to the user
# instead of being silently snapped to the nearest supported timeframe.
MAX_TIMEFRAME_SNAP_GAP_MINUTES = 0


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
    """Resolve a user-provided timeframe to a supported value.

    Returns (resolved, None) on success, otherwise (None, validation_message).
    Never silently snaps unsupported timeframes to a different value: the user
    must pick from the supported list explicitly.
    """
    raw = (timeframe or "").strip()
    if not raw:
        return None, UNSUPPORTED_USER_TIMEFRAME_MESSAGE

    canonical = normalise_timeframe(raw) or raw
    requested_minutes = _timeframe_to_minutes(canonical)
    if requested_minutes is None:
        return None, UNSUPPORTED_USER_TIMEFRAME_MESSAGE

    nearest_timeframe = min(
        SUPPORTED_USER_TIMEFRAMES,
        key=lambda candidate: (
            abs((_timeframe_to_minutes(candidate) or 0) - requested_minutes),
            (_timeframe_to_minutes(candidate) or 0),
        ),
    )
    nearest_minutes = _timeframe_to_minutes(nearest_timeframe) or 0

    if abs(nearest_minutes - requested_minutes) <= MAX_TIMEFRAME_SNAP_GAP_MINUTES:
        return nearest_timeframe, None

    return None, UNSUPPORTED_USER_TIMEFRAME_MESSAGE


_TF_RE = re.compile(
    r"\b(1m|3m|5m|10m|15m|30m|45m|1h|2h|3h|4h|6h|8h|12h|1d|3d|1w|"
    r"daily|weekly|hourly|"
    r"\d+\s*(?:-|to)?\s*(?:M|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks))\b",
    re.IGNORECASE,
)


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
        self.experience: Optional[str] = None
        self.objective: Optional[str] = None
        self.goal: Optional[str] = None
        self.daily_loss_cap: Optional[float] = None
        self.max_trade: Optional[str] = None
        self.entry_condition: Optional[str] = None
        self.exit_condition: Optional[str] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
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
        self.risk_execution_config: dict[str, Any] = {}
        self.input_modification_requested: bool = False
        self.pending_input_modification_fields: list[str] = []

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
        if not all([
            self.symbol,
            self.timeframe,
            self.sentiment,
            self.experience,
            self.objective,
            self.goal,
        ]):
            return False
        return True

    def missing_user_input_fields(self) -> list[str]:
        fields: list[str] = []
        if not self.symbol:
            fields.append("symbol")
        if not self.timeframe:
            fields.append("timeframe")
        if not self.sentiment:
            fields.append("sentiment")
        if not self.experience:
            fields.append("experience")
        if not self.objective:
            fields.append("objective")
        if not self.goal:
            fields.append("goal")
        return fields

    def reset_generated_strategy_state(self) -> None:
        self.user_input_confirmed = False
        self.signal_plan = None
        self.entry_condition = None
        self.exit_condition = None
        self.daily_loss_cap = None
        self.max_trade = None
        self.stop_loss = None
        self.take_profit = None

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

    def apply_defaults(self):
        cfg = MARKET_CONFIG.get(self.market or "", {})
        risk_cfg = self.risk_execution_config or {}
        if self.stop_loss is None:
            self.stop_loss = float(
                risk_cfg.get("stop_loss_pct", cfg.get("default_stop_loss", 2.0))
            )
        if self.take_profit is None:
            self.take_profit = float(
                risk_cfg.get("take_profit_pct", cfg.get("default_take_profit", 5.0))
            )
        if self.timeframe is None:
            self.timeframe = cfg.get("default_timeframe", "1d")
        if self.daily_loss_cap is None:
            if risk_cfg.get("daily_loss_cap") is not None:
                self.daily_loss_cap = float(risk_cfg["daily_loss_cap"])
            else:
                tier = get_experience_risk(self.experience or "")
                recommended = float(tier.get("daily_loss_cap_pct", 2.0))
                min_cap, max_cap = get_daily_loss_cap_bounds()
                self.daily_loss_cap = max(min_cap, min(max_cap, recommended))
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
            cfg = MARKET_CONFIG.get(self.market or "", {})
            stop_loss = cfg.get("default_stop_loss", 2.0)
            take_profit = cfg.get("default_take_profit", 5.0)
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
        return {
            "daily_loss_cap": f"{daily_loss_cap_pct:.1f}% of capital per day",
            "position_sizing": str(
                risk_cfg.get("position_sizing", DEFAULT_RISK_AND_EXECUTION["position_sizing"])
            ),
            "execution_mode": str(
                risk_cfg.get("execution_mode", DEFAULT_RISK_AND_EXECUTION["execution_mode"])
            ),
            "trading_window": str(
                risk_cfg.get("trading_window", DEFAULT_RISK_AND_EXECUTION["trading_window"])
            ),
            "risk_validation": str(
                risk_cfg.get("risk_validation", DEFAULT_RISK_AND_EXECUTION["risk_validation"])
            ),
            "per_trade_risk": f"{per_trade_risk_pct:.1f}% of capital per trade",
            "risk_reward": risk_reward,
            "max_trades": self._format_max_trades_display(max_trades),
        }

    def format_symbol(self) -> str:
        if not self.symbol:
            return ""
        raw = self.symbol.upper().strip()

        if raw.endswith((".NS", ".BO")):
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
        self._normalize_legacy_signal_conditions()

    def merge_preview(self, preview: dict):
        if not preview:
            return

        if preview.get("market"):
            self.market = self._normalise_market(preview["market"])
        if preview.get("symbol"):
            self.symbol = preview["symbol"]
        if preview.get("symbol_validation_code"):
            self.set_symbol_validation(
                str(preview["symbol_validation_code"]).strip(),
                preview.get("symbol_validation_facts") if isinstance(preview.get("symbol_validation_facts"), dict) else {},
            )
        elif preview.get("symbol_validation_message"):
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
        if preview.get("daily_loss_cap_pct") is not None:
            self.daily_loss_cap = float(preview["daily_loss_cap_pct"])
        if preview.get("max_trade"):
            self.max_trade = str(preview["max_trade"]).strip()
        if preview.get("entry_condition"):
            self.entry_condition = preview["entry_condition"]
        if preview.get("exit_condition"):
            self.exit_condition = preview["exit_condition"]
        if preview.get("stop_loss_pct") is not None:
            self.stop_loss = float(preview["stop_loss_pct"])
        if preview.get("take_profit_pct") is not None:
            self.take_profit = float(preview["take_profit_pct"])
        risk_execution_config = preview.get("risk_execution_config")
        if isinstance(risk_execution_config, dict):
            self.set_risk_execution_config(risk_execution_config)
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
            "entry_condition": self.entry_condition,
            "exit_condition": self.exit_condition,
            "stop_loss_pct": self.stop_loss,
            "take_profit_pct": self.take_profit,
            "indicators": self.extract_indicators(),
            "user_input_confirmed": self.user_input_confirmed,
            "signal_plan": self.signal_plan,
            "strategy_preset": self.strategy_preset,
            "input_modification_requested": self.input_modification_requested,
            "pending_input_modification_fields": self.pending_input_modification_fields,
            "processing_status": processing_status or ("complete" if self.signal_plan else "in_progress"),
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
        if entry_signals:
            strategy_block["entry_evaluation_mode"] = "registry"
            strategy_block["entry_signals"] = entry_signals
        if exit_signals:
            strategy_block["exit_evaluation_mode"] = "registry"
            strategy_block["exit_signals"] = exit_signals

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

    if re.search(r"\b(bullish|optimistic|positive|long|buy|uptrend|upside)\b", text, re.IGNORECASE):
        builder.sentiment = "bullish"
    elif re.search(r"\b(bearish|cautious|negative|short|sell|downtrend|downside)\b", text, re.IGNORECASE):
        builder.sentiment = "bearish"

    if re.search(
        r"\b(intraday|day trade|day trading|scalp|scalping|quick profit|quick profits|same day|today only|short term)\b",
        text,
        re.IGNORECASE,
    ):
        builder.objective = "intraday"
    elif re.search(
        r"\b(positional|swing|steady growth|risk protection|long term|multi day|carry)\b",
        text,
        re.IGNORECASE,
    ):
        builder.objective = "positional"

    goal_text = extract_goal_text(text)
    if goal_text:
        builder.goal = goal_text

    # Detect a strategy preset by keyword anywhere in the message. The KB owns
    # the keyword-to-preset map; we only consult it here so a user typing
    # "let's run an ORB strategy on RELIANCE 5m" gets the ORB preset pinned
    # without any additional UI step. Explicit caller-set presets win.
    if not builder.strategy_preset:
        try:
            from app.kb import kb as _kb
            preset = _kb.detect_preset_in_text(text)
        except Exception:
            preset = None
        if preset is not None:
            builder.strategy_preset = preset.name
