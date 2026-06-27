"""
app/core/config.py
──────────────────
Central settings — every module imports from here, never from os.environ directly.
"""
import os
from pathlib import Path
import re
from functools import lru_cache
from typing import Literal, List, Optional, Tuple, Union

from dotenv import dotenv_values
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_OPENROUTER_KEY_ENV_PREFIX = "OPENROUTER_API_KEY_"
_OPENROUTER_KEY_TOKEN_SPLIT_RE = re.compile(r"[\r\n,;]+")
_OPENROUTER_KEY_INDEXED_RE = re.compile(
    r"^\s*(?:\[(?P<bracket_index>\d+)\]|(?P<plain_index>\d+))\s*[:=]\s*(?P<value>.+?)\s*$"
)


def _clean_openrouter_key_value(raw: Optional[str]) -> str:
    return str(raw or "").strip().strip('"').strip("'")


def _split_openrouter_key_values(raw: Optional[str]) -> List[str]:
    parts = [
        _clean_openrouter_key_value(part)
        for part in _OPENROUTER_KEY_TOKEN_SPLIT_RE.split(str(raw or ""))
    ]
    parts = [part for part in parts if part]

    indexed_parts: List[Tuple[int, str]] = []
    plain_parts: List[str] = []

    for part in parts:
        match = _OPENROUTER_KEY_INDEXED_RE.match(part)
        if not match:
            plain_parts.append(part)
            continue

        key_value = _clean_openrouter_key_value(match.group("value"))
        if not key_value:
            continue

        index_value = match.group("bracket_index") or match.group("plain_index") or "0"
        indexed_parts.append((int(index_value), key_value))

    ordered_indexed = [value for _, value in sorted(indexed_parts, key=lambda item: item[0])]
    return ordered_indexed + plain_parts if indexed_parts else plain_parts


def _dedupe_preserve(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _openrouter_key_suffix_sort_key(suffix: str) -> Tuple[int, Union[int, str]]:
    cleaned = str(suffix or "").strip()
    if cleaned.isdigit():
        return (0, int(cleaned))
    return (1, cleaned)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env:        str  = "development"
    app_secret_key: str  = "change-me"
    app_debug:      bool = True
    # Single-tenant deploy: resolve tenant from ref_data.tenants.code when
    # x-tenant-id header is absent. Leave empty for multi-tenant (header required
    # unless legacy env fallback is acceptable). Example: stretus_internal
    tenant_code:    str  = ""

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url:      str = "postgresql+asyncpg://stretus:password@localhost:5432/stretus"
    database_url_sync: str = "postgresql+psycopg2://stretus:password@localhost:5432/stretus"

    # ── AI Provider selection ─────────────────────────────────────────────────
    # LLM_PROVIDER controls which backend is used:
    #   "openrouter" → OpenRouter cloud API (unified access to multiple LLMs)
    #   "ollama"     → Local Ollama (private, needs Ollama running locally)
    #   "auto"       → Try OpenRouter first, fall back to Ollama
    llm_provider: Literal["openrouter", "ollama", "auto"] = "openrouter"

    # ── OpenRouter Cloud Settings ─────────────────────────────────────────────
    # Keys are managed via JSON state file: app/services/ai/openrouter_key_state.json
    # The system automatically rotates to the next key when one is exhausted (429 error)
    openrouter_api_key: str = ""
    openrouter_api_keys: str = ""
    # Available OpenRouter models:
    #   meta-llama/llama-3.3-70b-instruct  ← recommended (best quality)
    #   anthropic/claude-3.5-sonnet        ← excellent reasoning
    #   google/gemini-pro-1.5              ← good for long contexts
    #   openai/gpt-4-turbo                 ← OpenAI's best
    # ── Two-tier models ────────────────────────────────────────────────────────
    # reasoning_model — the CAPABLE model used ONLY for the quality-critical step:
    #   strategy generation (composing valid engine-grammar conditions that faithfully
    #   capture complex intent). Worth the extra latency/cost.
    # fast_model — the FAST/CHEAP model for everything else: routing, intent extraction,
    #   input gathering, clarifications, and simple replies. Runs on every turn, so speed
    #   matters most here. Empty → fall back to reasoning_model.
    # Env aliases keep the legacy OPENROUTER_MODEL / OPENROUTER_FAST_MODEL vars working.
    reasoning_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct",
        validation_alias=AliasChoices("reasoning_model", "REASONING_MODEL", "OPENROUTER_MODEL"),
    )
    fast_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct",
        validation_alias=AliasChoices("fast_model", "FAST_MODEL", "OPENROUTER_FAST_MODEL"),
    )

    # ── Ollama Local Settings ─────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    # Local model to use — must be pulled first with: ollama pull <model>
    # Recommended models:
    #   qwen2.5:32b    ← best quality (needs ~20GB RAM)
    #   qwen2.5:7b     ← good quality (needs ~6GB RAM)
    #   llama3.2:3b    ← fast, light  (needs ~3GB RAM)
    #   mistral:7b     ← solid general model
    ollama_model: str = "qwen2.5:7b"

    # ── Backward compatibility (old USE_OPENROUTER flag) ──────────────────────
    # If USE_OPENROUTER=true is set in .env, it overrides llm_provider
    use_openrouter: bool = False

    # ── Strategy generation path ──────────────────────────────────────────────
    # When True, chat strategy-building bypasses the KB planner and uses the direct
    # LLM → strict StrategySpec → validator → engine path (app/strategy/). When
    # False (default), the existing knowledge-base planner runs unchanged. The two
    # paths coexist; this flag is the only switch.
    use_direct_strategy_path: bool = False
    # How many times the direct path feeds validation errors back to the LLM to repair.
    direct_strategy_max_repairs: int = 2
    # Output-token budget for the StrategySpec generation call. The spec JSON is large
    # and reasoning models (e.g. GLM) spend tokens thinking, so the default chat cap of
    # 2048 truncates to an empty/invalid response. 8192 leaves room for reasoning + JSON.
    direct_strategy_max_output_tokens: int = 8192
    # Reasoning effort for the generation call on reasoning models (e.g. GLM). These
    # models can spend 20k+ chars "thinking" before the JSON, making a turn take
    # ~90s. We don't need deep reasoning to emit a spec, so cap it. Values:
    # "low"/"medium"/"high" (OpenRouter effort) or "off"/"none" to disable thinking.
    direct_strategy_reasoning_effort: str = "low"

    # ── Dynamic universe (KB-free direct-StrategySpec path) ───────────────────
    # A dynamic-universe strategy names a RULE for selecting instruments (re-resolved
    # on a cadence) rather than one symbol. The whole feature is gated by this flag
    # and is additive: when False, the static single-symbol path is byte-for-byte
    # unchanged. See docs/dynamic-universe-implementation.md (§2 invariants, §7.1 cap).
    dynamic_universe_enabled: bool = False
    # Hard platform ceiling on how many assets a dynamic strategy actually resolves
    # and trades, INDEPENDENT of the user's requested `take`. Applied AFTER ranking
    # (top-N survive) by the resolver, so it is identical in backtest and live (§7.1).
    # Conservative during rollout; raise via config (no code change) as it proves out.
    dynamic_universe_max_assets: int = 2
    # Hard ceiling on concurrent open positions for a dynamic strategy (§7.1).
    dynamic_universe_max_positions: int = 2
    # ADV (average-daily-value) floor window, in bars, for fail-closed eligibility.
    dynamic_universe_adv_window: int = 20
    # Warm-up bars loaded ahead of a member's activation so indicators are primed.
    dynamic_universe_warmup_bars: int = 300
    # Candidate-pool guard: refuse to resolve a pool larger than this (scale guard, §4.5).
    dynamic_universe_max_pool: int = 5000
    # Scale guard: the projected execution-tier working set (active members) must not
    # exceed this; a dynamic backtest that would breach it is refused, never OOM'd (§4.5).
    dynamic_universe_max_working_set: int = 50

    # ── Strategy files ────────────────────────────────────────────────────────
    strategy_folder: str = "./strategies"

    # ── Quant Engine ──────────────────────────────────────────────────────────
    quant_engine_url: str = "http://localhost:8001"

    # ── Internal market data (Backtest + Execution) ───────────────────────────
    # HTTP fallback: user-gateway base URL (User-Gateway → BFF → market data).
    # Used by backtest OHLCV fetch and InternalMarketDataClient (live candles/LTP).
    historical_data_url: str = ""
    historical_data_timeout_seconds: float = 120.0
    # In-cluster gRPC to marketdata-ingestion (port 50057). Preferred on EKS.
    # auto = use gRPC when MARKET_DATA_GRPC_TARGET is set, else HTTP.
    market_data_fetch_transport: str = "auto"
    market_data_grpc_target: str = ""
    market_data_grpc_timeout_seconds: float = 120.0
    market_data_grpc_secure: bool = False
    quant_engine_timeout_seconds: float = 600.0  # 10 minutes for complex backtests
    # Inclusive range from 2024-01-01 through current date (dynamically calculated).
    # Note: backtest_default_lookback_days is approximate and recalculated at runtime.
    backtest_default_lookback_days: int = 868
    # Short lookback used ONLY for signal parameter estimation (planner).
    # Kept small to avoid rate-limit errors on 1-minute interval fetches.
    signal_eval_lookback_days: int = 30
    # Chunk size for backtest OHLCV fetching. The full backtest range is split
    # into windows of this many days and fetched sequentially to avoid HTTP 429.
    # Indian equity APIs cope with ~180-day windows. Crypto venues return very
    # dense candles (24/7 markets) and reject the same window — use a smaller
    # default (~30 days) which is overridable per env.
    backtest_fetch_chunk_days: int = 180
    backtest_fetch_chunk_days_crypto: int = 45
    # Earliest UTC start users may request for a custom backtest window.
    # Requests before this date receive a friendly rejection in chat/API.
    backtest_earliest_from_utc: str = "2024-01-01T00:00:00Z"
    # When the user specifies a custom from/to, only this many calendar days of
    # pre-window OHLCV are fetched for indicator warm-up (sim stays on user range).
    backtest_user_range_max_padding_days: int = 90
    backtest_user_range_min_padding_days: int = 14

    # ── Live market data (Execution / Order Evaluation service only) ──────────
    # Live reads route through HISTORICAL_DATA_URL (user-gateway) via
    # InternalMarketDataClient. Adding a new broker only requires changes in
    # stretus-backend; this service and all its callers are unaffected.
    market_data_timeout_seconds: float = 30.0
    # Equity (NSE/BSE) live market-data source for the execution evaluator. The live broker
    # feed is reached through the stretus-backend gateway (InternalMarketDataClient):
    #   * "upstox"        — gateway live feed only (real-time; broker selection owned by gateway).
    #   * "backtest_feed" — the historical market-data service (HISTORICAL_DATA_URL) only; full
    #                       backtest/live parity, no broker token (paper/parity). LTP = last close.
    #   * "resilient"     — gateway primary + backtest_feed fallback (default): live when the gateway
    #                       is up, transparently degrades to the proven feed on failure so a
    #                       strategy never goes blind. Every fallback is logged (never silent).
    # Crypto always routes through the gateway as well and is unaffected by this setting.
    equity_market_data_source: Literal["upstox", "backtest_feed", "resilient"] = "resilient"
    # In-process cache TTL for execution market data (seconds)
    market_data_cache_ttl_seconds: int = 1
    # Skip entry if LTP is within this fraction of the circuit limit (2% buffer)
    market_data_circuit_threshold_pct: float = 0.98
    # Legacy direct-broker env vars kept for reference / local dev fallback.
    # These are no longer used by InternalMarketDataClient but may still be
    # read by the broker execution adapters (order placement, etc.).
    market_data_url: str = "https://api.upstox.com/v2"
    upstox_api_key: str = ""
    upstox_access_token: str = ""
    crypto_market_data_url: str = "https://api.binance.com"

    def effective_provider(self) -> str:
        """Resolve the actual provider, handling the legacy USE_OPENROUTER flag."""
        if self.use_openrouter:
            return "openrouter"
        return self.llm_provider

    def _settings_env_values(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        env_file = self.model_config.get("env_file")
        env_files = env_file if isinstance(env_file, (list, tuple)) else [env_file]

        for raw_path in env_files:
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if not path.exists():
                continue
            try:
                file_values = dotenv_values(path)
            except Exception:
                continue
            for key, value in file_values.items():
                if key and value is not None:
                    merged[str(key).upper()] = str(value).strip()

        for key, value in os.environ.items():
            if value is not None:
                merged[str(key).upper()] = str(value).strip()

        return merged

    def openrouter_api_key_pool(self) -> List[str]:
        keys: List[str] = []
        keys.extend(_split_openrouter_key_values(self.openrouter_api_key))
        keys.extend(_split_openrouter_key_values(self.openrouter_api_keys))

        numbered_keys: List[Tuple[Tuple[int, Union[int, str]], str]] = []
        for name, value in self._settings_env_values().items():
            if not name.startswith(_OPENROUTER_KEY_ENV_PREFIX):
                continue
            suffix = name[len(_OPENROUTER_KEY_ENV_PREFIX):]
            if not suffix:
                continue
            numbered_keys.append((_openrouter_key_suffix_sort_key(suffix), value))

        for _, value in sorted(numbered_keys, key=lambda item: item[0]):
            keys.extend(_split_openrouter_key_values(value))

        return _dedupe_preserve(keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def refresh_settings() -> Settings:
    """
    Re-read configuration from environment and .env.

    Helpful in development when temporary service URLs like ngrok change while
    the API process is still running.
    """
    get_settings.cache_clear()
    return get_settings()
