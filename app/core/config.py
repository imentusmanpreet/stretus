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
from pydantic_settings import BaseSettings, SettingsConfigDict

_GROQ_KEY_ENV_PREFIX = "GROQ_API_KEY_"
_GROQ_KEY_TOKEN_SPLIT_RE = re.compile(r"[\r\n,;]+")
_GROQ_KEY_INDEXED_RE = re.compile(
    r"^\s*(?:\[(?P<bracket_index>\d+)\]|(?P<plain_index>\d+))\s*[:=]\s*(?P<value>.+?)\s*$"
)


def _clean_groq_key_value(raw: Optional[str]) -> str:
    return str(raw or "").strip().strip('"').strip("'")


def _split_groq_key_values(raw: Optional[str]) -> List[str]:
    parts = [
        _clean_groq_key_value(part)
        for part in _GROQ_KEY_TOKEN_SPLIT_RE.split(str(raw or ""))
    ]
    parts = [part for part in parts if part]

    indexed_parts: List[Tuple[int, str]] = []
    plain_parts: List[str] = []

    for part in parts:
        match = _GROQ_KEY_INDEXED_RE.match(part)
        if not match:
            plain_parts.append(part)
            continue

        key_value = _clean_groq_key_value(match.group("value"))
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


def _groq_key_suffix_sort_key(suffix: str) -> Tuple[int, Union[int, str]]:
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

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url:      str = "postgresql+asyncpg://stretus:password@localhost:5432/stretus"
    database_url_sync: str = "postgresql+psycopg2://stretus:password@localhost:5432/stretus"

    # ── AI Provider selection ─────────────────────────────────────────────────
    # LLM_PROVIDER controls which backend is used:
    #   "groq"   → Groq cloud API  (fast, free tier, needs GROQ_API_KEY)
    #   "ollama" → Local Ollama    (private, needs Ollama running locally)
    #   "auto"   → Try Groq first, fall back to Ollama if Groq fails
    llm_provider: Literal["groq", "ollama", "auto"] = "groq"

    # ── Groq Cloud Settings ───────────────────────────────────────────────────
    # Supports:
    #   GROQ_API_KEY=gsk_primary
    #   GROQ_API_KEY=gsk_primary,gsk_backup
    #   GROQ_API_KEY=1:gsk_primary,2:gsk_backup,3:gsk_third
    groq_api_key: str = ""
    groq_api_keys: str = ""
    # Available Groq models (free tier):
    #   llama-3.3-70b-versatile  ← recommended (best quality)
    #   llama-3.1-8b-instant     ← fastest, lower quality
    #   mixtral-8x7b-32768       ← good for long contexts
    #   gemma2-9b-it             ← Google's model
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Ollama Local Settings ─────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    # Local model to use — must be pulled first with: ollama pull <model>
    # Recommended models:
    #   qwen2.5:32b    ← best quality (needs ~20GB RAM)
    #   qwen2.5:7b     ← good quality (needs ~6GB RAM)
    #   llama3.2:3b    ← fast, light  (needs ~3GB RAM)
    #   mistral:7b     ← solid general model
    ollama_model: str = "qwen2.5:7b"

    # ── Backward compatibility (old USE_GROQ flag) ────────────────────────────
    # If USE_GROQ=true is set in .env, it overrides llm_provider
    use_groq: bool = False

    # ── Strategy files ────────────────────────────────────────────────────────
    strategy_folder: str = "./strategies"

    # ── Quant Engine ──────────────────────────────────────────────────────────
    quant_engine_url: str = "http://localhost:8001"

    # ── Historical data (Backtest service only) ───────────────────────────────
    # Ngrok tunnel to your local OHLCV server.
    # Used ONLY by the backtest service for historical candle fetch.
    historical_data_url: str = ""
    historical_data_timeout_seconds: float = 60.0
    quant_engine_timeout_seconds: float = 180.0
    # Inclusive range from 2024-01-01 through 2026-03-31.
    backtest_default_lookback_days: int = 821
    # Short lookback used ONLY for signal parameter estimation (planner).
    # Kept small to avoid rate-limit errors on 1-minute interval fetches.
    signal_eval_lookback_days: int = 30
    # Chunk size for backtest OHLCV fetching. The full backtest range is split
    # into windows of this many days and fetched sequentially to avoid HTTP 429.
    backtest_fetch_chunk_days: int = 90

    # ── Live market data (Execution / Order Evaluation service only) ──────────
    # Points to Upstox v2 base URL.  All execution market data comes from here:
    #   candles  → GET {MARKET_DATA_URL}/historical-candle/{key}/{interval}/...
    #   LTP      → GET {MARKET_DATA_URL}/market-quote/ltp
    #   circuit  → GET {MARKET_DATA_URL}/market-quote/quotes
    # Set MARKET_DATA_URL in .env — never hardcode here.
    market_data_url: str = "https://api.upstox.com/v2"
    market_data_timeout_seconds: float = 30.0
    upstox_api_key: str = ""
    upstox_access_token: str = ""
    # In-process cache TTL for execution market data (seconds)
    market_data_cache_ttl_seconds: int = 1
    # Skip entry if LTP is within this fraction of the circuit limit (2% buffer)
    market_data_circuit_threshold_pct: float = 0.98

    def effective_provider(self) -> str:
        """Resolve the actual provider, handling the legacy USE_GROQ flag."""
        if self.use_groq:
            return "groq"
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

    def groq_api_key_pool(self) -> List[str]:
        keys: List[str] = []
        keys.extend(_split_groq_key_values(self.groq_api_key))
        keys.extend(_split_groq_key_values(self.groq_api_keys))

        numbered_keys: List[Tuple[Tuple[int, Union[int, str]], str]] = []
        for name, value in self._settings_env_values().items():
            if not name.startswith(_GROQ_KEY_ENV_PREFIX):
                continue
            suffix = name[len(_GROQ_KEY_ENV_PREFIX):]
            if not suffix:
                continue
            numbered_keys.append((_groq_key_suffix_sort_key(suffix), value))

        for _, value in sorted(numbered_keys, key=lambda item: item[0]):
            keys.extend(_split_groq_key_values(value))

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
