"""
app/services/ai/llm.py
═══════════════════════
LLMService — supports three providers:

  Provider    Config              Description
  ─────────   ──────────────────  ────────────────────────────────────────
  groq        LLM_PROVIDER=groq   Groq Cloud API (fast, free tier available)
  ollama      LLM_PROVIDER=ollama Local model via Ollama (private, no API key)
  auto        LLM_PROVIDER=auto   Try Groq first, fall back to Ollama

Set in .env:
  LLM_PROVIDER=groq
  GROQ_API_KEY=1:gsk_primary,2:gsk_backup
  GROQ_MODEL=llama-3.3-70b-versatile

  OR

  LLM_PROVIDER=ollama
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=qwen2.5:7b

  OR

  LLM_PROVIDER=auto   (tries groq, falls back to ollama automatically)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from app.core.config import get_settings
from app.core.errors import AppError

logger   = logging.getLogger(__name__)


# ── Available Groq models ─────────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Best quality — recommended",
    "llama-3.1-8b-instant":    "Fastest, lighter quality",
    "mixtral-8x7b-32768":      "Good for long contexts",
    "gemma2-9b-it":            "Google Gemma model",
}

# ── Available Ollama models (you must pull them first) ────────────────────────
OLLAMA_MODELS = {
    "qwen2.5:32b":  "Best quality (needs ~20GB RAM)",
    "qwen2.5:14b":  "Great quality (needs ~10GB RAM)",
    "qwen2.5:7b":   "Good quality (needs ~6GB RAM)",
    "llama3.2:3b":  "Fast and light (needs ~3GB RAM)",
    "mistral:7b":   "Solid general model (needs ~5GB RAM)",
    "codellama:7b": "Good at code/formulas (needs ~5GB RAM)",
}


class LLMService:
    """
    Unified LLM service supporting Groq Cloud and local Ollama.

    Usage:
        llm = LLMService()
        response = await llm.chat(messages)

    The provider is chosen from settings — no code changes needed,
    just update .env and restart.
    """

    _groq_key_cursor = 0

    def __init__(self):
        current_settings = get_settings()
        self._provider = current_settings.effective_provider()
        self._groq_keys = current_settings.groq_api_key_pool()
        self._groq_key = self._groq_keys[0] if self._groq_keys else current_settings.groq_api_key
        self._groq_model = current_settings.groq_model
        self._ollama_url = current_settings.ollama_base_url
        self._ollama_model = current_settings.ollama_model

        # What model name to expose (used for logging / DB storage)
        if self._provider == "groq":
            self.model_name = self._groq_model
        elif self._provider == "ollama":
            self.model_name = self._ollama_model
        else:  # auto
            self.model_name = f"{self._groq_model} (auto)"

        logger.info(
            f"LLMService initialised — provider={self._provider}  "
            f"model={self.model_name}"
        )

    # ── Public method ─────────────────────────────────────────────────────────

    async def chat(self, messages: list[dict]) -> str:
        """
        Send a message list to the LLM and return the text response.

        messages format:
            [
                {"role": "system",    "content": "You are ..."},
                {"role": "user",      "content": "I want to trade NIFTY"},
                {"role": "assistant", "content": "Great! Which timeframe?"},
                {"role": "user",      "content": "15m"},
            ]
        """
        if self._provider == "groq":
            return await self._call_groq(messages)

        elif self._provider == "ollama":
            return await self._call_ollama(messages)

        else:  # auto — try groq first, fall back to ollama
            return await self._auto(messages)

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
    ) -> dict:
        """
        Send messages with tool definitions and return a normalized response:

            {
                "content": "optional assistant text",
                "tool_calls": [
                    {"id": "...", "name": "run_backtest", "arguments": {...}}
                ]
            }

        Groq receives native function/tool definitions. Ollama uses native
        tools when the installed SDK supports them, otherwise a strict JSON
        tool-selection fallback is used.
        """
        if self._provider == "groq":
            return await self._call_groq_with_tools(messages, tools, tool_choice=tool_choice)
        if self._provider == "ollama":
            return await self._call_ollama_with_tools(messages, tools)
        return await self._auto_with_tools(messages, tools, tool_choice=tool_choice)

    # ── Provider info ─────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return current provider config — useful for /health endpoint."""
        return {
            "provider":      self._provider,
            "groq_model":    self._groq_model,
            "ollama_model":  self._ollama_model,
            "ollama_url":    self._ollama_url,
            "active_model":  self.model_name,
            "groq_key_set":  bool(self._groq_keys),
            "groq_keys_configured": len(self._groq_keys),
            "groq_failover_enabled": len(self._groq_keys) > 1,
        }

    # ── Groq Cloud ────────────────────────────────────────────────────────────

    def _ordered_groq_keys(self) -> list[tuple[int, str]]:
        if not self._groq_keys:
            return []

        key_count = len(self._groq_keys)
        start = self.__class__._groq_key_cursor % key_count
        return [
            ((start + offset) % key_count, self._groq_keys[(start + offset) % key_count])
            for offset in range(key_count)
        ]

    async def _call_groq_once(
        self,
        messages: list[dict],
        api_key: str,
        model: Optional[str] = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Call the Groq Cloud API.

        Free tier limits (as of 2024):
          llama-3.3-70b-versatile  → 6,000 requests/day,  30 req/min
          llama-3.1-8b-instant     → 14,400 requests/day, 30 req/min

        Get your free API key at: https://console.groq.com
        """
        use_model = model or self._groq_model

        try:
            from groq import (
                APIConnectionError,
                AsyncGroq,
                AuthenticationError,
                BadRequestError,
                GroqError,
                NotFoundError,
                RateLimitError,
            )
        except ImportError as exc:
            raise AppError(
                503,
                "Groq client is not available on the server. Please retry after some time.",
            ) from exc

        try:
            client = AsyncGroq(api_key=api_key)

            resp = await client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,      # low = more consistent, less creative
                max_tokens=max_tokens,
            )

            content = resp.choices[0].message.content
            logger.debug(f"Groq response — model={use_model}  tokens={resp.usage.total_tokens}")
            return content

        except RateLimitError as exc:
            logger.warning("Groq rate limit reached for model=%s: %s", use_model, exc)
            raise AppError(
                429,
                (
                    "Rate limit exceeded. You have reached your API usage limit. "
                    "Please retry after some time."
                ),
            ) from exc
        except AuthenticationError as exc:
            logger.error("Groq authentication failed for model=%s: %s", use_model, exc)
            raise AppError(
                401,
                "Authentication failed. Please verify the configured Groq API key and try again.",
            ) from exc
        except NotFoundError as exc:
            logger.error("Groq model not found: %s", use_model)
            raise AppError(
                404,
                f"The configured Groq model '{use_model}' was not found. Please verify the model name and try again.",
            ) from exc
        except APIConnectionError as exc:
            logger.error("Groq connection error for model=%s: %s", use_model, exc)
            raise AppError(
                503,
                "Unable to reach Groq right now. Please retry after some time.",
            ) from exc
        except BadRequestError as exc:
            logger.error("Groq request rejected for model=%s: %s", use_model, exc)
            raise AppError(
                400,
                str(exc).strip() or "The Groq request was invalid. Please review the request and try again.",
            ) from exc
        except GroqError as exc:
            logger.error("Groq error for model=%s: %s", use_model, exc)
            raise AppError(
                502,
                "Groq could not process the request right now. Please retry after some time.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected Groq error for model=%s", use_model)
            raise AppError(
                500,
                "An unexpected LLM error occurred. Please retry after some time.",
            ) from exc

    async def _call_groq(self, messages: list[dict], model: Optional[str] = None) -> str:
        if not self._groq_keys:
            raise AppError(
                503,
                "Groq is not configured on the server. Please retry after some time.",
            )

        use_model = model or self._groq_model
        retryable_errors: list[AppError] = []

        for index, api_key in self._ordered_groq_keys():
            try:
                result = await self._call_groq_once(messages, api_key, use_model)
                self._groq_key = api_key
                self.__class__._groq_key_cursor = index
                return result
            except AppError as exc:
                if exc.status_code in {401, 429} and len(self._groq_keys) > 1:
                    retryable_errors.append(exc)
                    logger.warning(
                        "Groq key failover triggered for configured key %s/%s with status=%s.",
                        index + 1,
                        len(self._groq_keys),
                        exc.status_code,
                    )
                    continue
                raise

        statuses = {error.status_code for error in retryable_errors}

        if statuses == {429}:
            raise AppError(
                429,
                "All configured Groq API keys are currently rate limited. Please retry after some time.",
            )
        if statuses == {401}:
            raise AppError(
                401,
                "All configured Groq API keys failed authentication. Please verify the configured keys and try again.",
            )
        if retryable_errors:
            raise AppError(
                503,
                "No configured Groq API key is currently usable. Please verify the configured keys or retry after some time.",
            )
        raise AppError(
            503,
            "Groq is not configured on the server. Please retry after some time.",
        )

    async def _call_groq_tools_once(
        self,
        messages: list[dict],
        tools: list[dict],
        api_key: str,
        model: Optional[str] = None,
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """Call Groq with native tool definitions and normalize the result."""
        use_model = model or self._groq_model

        try:
            from groq import (
                APIConnectionError,
                AsyncGroq,
                AuthenticationError,
                BadRequestError,
                GroqError,
                NotFoundError,
                RateLimitError,
            )
        except ImportError as exc:
            raise AppError(
                503,
                "Groq client is not available on the server. Please retry after some time.",
            ) from exc

        try:
            client = AsyncGroq(api_key=api_key)
            resp = await client.chat.completions.create(
                model=use_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _normalise_tool_response(resp.choices[0].message)
        except RateLimitError as exc:
            logger.warning("Groq rate limit reached for tool call model=%s: %s", use_model, exc)
            raise AppError(429, "Rate limit exceeded. Please retry after some time.") from exc
        except AuthenticationError as exc:
            logger.error("Groq authentication failed for tool call model=%s: %s", use_model, exc)
            raise AppError(
                401,
                "Authentication failed. Please verify the configured Groq API key and try again.",
            ) from exc
        except NotFoundError as exc:
            logger.error("Groq tool-call model not found: %s", use_model)
            raise AppError(
                404,
                f"The configured Groq model '{use_model}' was not found.",
            ) from exc
        except APIConnectionError as exc:
            logger.error("Groq connection error for tool call model=%s: %s", use_model, exc)
            raise AppError(503, "Unable to reach Groq right now. Please retry after some time.") from exc
        except BadRequestError as exc:
            logger.error("Groq tool-call request rejected for model=%s: %s", use_model, exc)
            raise AppError(400, str(exc).strip() or "The Groq tool request was invalid.") from exc
        except GroqError as exc:
            logger.error("Groq tool-call error for model=%s: %s", use_model, exc)
            raise AppError(502, "Groq could not process the tool request right now.") from exc
        except Exception as exc:
            logger.exception("Unexpected Groq tool-call error for model=%s", use_model)
            raise AppError(500, "An unexpected LLM tool-call error occurred.") from exc

    async def _call_groq_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: Optional[str] = None,
        *,
        tool_choice: str | dict = "auto",
    ) -> dict:
        if not self._groq_keys:
            raise AppError(
                503,
                "Groq is not configured on the server. Please retry after some time.",
            )

        use_model = model or self._groq_model
        retryable_errors: list[AppError] = []

        for index, api_key in self._ordered_groq_keys():
            try:
                result = await self._call_groq_tools_once(
                    messages,
                    tools,
                    api_key,
                    use_model,
                    tool_choice=tool_choice,
                )
                self._groq_key = api_key
                self.__class__._groq_key_cursor = index
                return result
            except AppError as exc:
                if exc.status_code in {401, 429} and len(self._groq_keys) > 1:
                    retryable_errors.append(exc)
                    logger.warning(
                        "Groq key failover triggered for tool call key %s/%s with status=%s.",
                        index + 1,
                        len(self._groq_keys),
                        exc.status_code,
                    )
                    continue
                raise

        statuses = {error.status_code for error in retryable_errors}
        if statuses == {429}:
            raise AppError(
                429,
                "All configured Groq API keys are currently rate limited. Please retry after some time.",
            )
        if statuses == {401}:
            raise AppError(
                401,
                "All configured Groq API keys failed authentication. Please verify the configured keys.",
            )
        raise AppError(503, "No configured Groq key is currently usable for tool calling.")

    # ── Ollama Local ──────────────────────────────────────────────────────────

    async def _call_ollama(self, messages: list[dict], model: Optional[str] = None) -> str:
        """
        Call a local model running in Ollama.

        Setup:
            1. Install Ollama: https://ollama.com/download
            2. Start it:       ollama serve
            3. Pull a model:   ollama pull qwen2.5:7b
            4. Set in .env:    LLM_PROVIDER=ollama
                               OLLAMA_MODEL=qwen2.5:7b

        Ollama runs on http://localhost:11434 by default.
        The model runs 100% on your machine — no API key, no internet needed.
        """
        use_model = model or self._ollama_model

        try:
            import ollama as ollama_lib
        except ImportError as exc:
            raise AppError(
                503,
                "Ollama client is not available on the server. Please retry after some time.",
            ) from exc

        try:

            # Ollama SDK is sync — run in executor so we don't block event loop.
            # SDK signatures differ by version; prefer Client(host=...) when available.
            loop = asyncio.get_event_loop()
            options = {
                "temperature": 0.2,
                "num_predict": 2048,    # max tokens to generate
            }

            def _sync_call():
                client_cls = getattr(ollama_lib, "Client", None)
                if client_cls is not None:
                    client = client_cls(host=self._ollama_url)
                    return client.chat(
                        model=use_model,
                        messages=messages,
                        options=options,
                    )
                # Fallback for older/newer SDK variants where module-level chat is used.
                return ollama_lib.chat(
                    model=use_model,
                    messages=messages,
                    options=options,
                )

            resp = await loop.run_in_executor(None, _sync_call)

            content = resp["message"]["content"]
            logger.debug(f"Ollama response — model={use_model}")
            return content

        except Exception as exc:
            err = str(exc)
            logger.error(f"Ollama error: {err}")

            if "connection refused" in err.lower() or "connect" in err.lower():
                raise AppError(
                    503,
                    (
                        f"Cannot reach Ollama at {self._ollama_url}. "
                        "Please make sure the local model service is running."
                    ),
                ) from exc
            if "model" in err.lower() and "not found" in err.lower():
                raise AppError(
                    404,
                    f"Ollama model '{use_model}' is not available on the server.",
                ) from exc
            raise AppError(
                502,
                "Ollama could not process the request right now. Please retry after some time.",
            ) from exc

    async def _call_ollama_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: Optional[str] = None,
    ) -> dict:
        """Call Ollama with tools when available, otherwise strict JSON fallback."""
        use_model = model or self._ollama_model

        try:
            import ollama as ollama_lib
        except ImportError as exc:
            raise AppError(
                503,
                "Ollama client is not available on the server. Please retry after some time.",
            ) from exc

        loop = asyncio.get_event_loop()
        options = {"temperature": 0.1, "num_predict": 1024}
        fallback_messages = _with_json_tool_fallback_instruction(messages, tools)

        def _sync_call():
            client_cls = getattr(ollama_lib, "Client", None)
            if client_cls is not None:
                client = client_cls(host=self._ollama_url)
                try:
                    return client.chat(
                        model=use_model,
                        messages=messages,
                        tools=tools,
                        options=options,
                    )
                except TypeError:
                    return client.chat(
                        model=use_model,
                        messages=fallback_messages,
                        options=options,
                    )
            try:
                return ollama_lib.chat(
                    model=use_model,
                    messages=messages,
                    tools=tools,
                    options=options,
                )
            except TypeError:
                return ollama_lib.chat(
                    model=use_model,
                    messages=fallback_messages,
                    options=options,
                )

        try:
            resp = await loop.run_in_executor(None, _sync_call)
            message = resp.get("message", resp) if isinstance(resp, dict) else resp
            return _normalise_tool_response(message)
        except Exception as exc:
            err = str(exc)
            logger.error("Ollama tool-call error: %s", err)
            if "connection refused" in err.lower() or "connect" in err.lower():
                raise AppError(
                    503,
                    (
                        f"Cannot reach Ollama at {self._ollama_url}. "
                        "Please make sure the local model service is running."
                    ),
                ) from exc
            raise AppError(
                502,
                "Ollama could not process the tool request right now. Please retry after some time.",
            ) from exc

    # ── Auto (try Groq, fall back to Ollama) ─────────────────────────────────

    async def _auto(self, messages: list[dict]) -> str:
        """
        Try Groq first. If it fails (no key, rate limit, network issue),
        automatically fall back to the local Ollama model.

        Good for development — use Groq when available, Ollama offline.
        """
        groq_error: AppError | None = None

        if self._groq_keys:
            try:
                result = await self._call_groq(messages)
                self.model_name = self._groq_model
                return result
            except AppError as exc:
                groq_error = exc
                logger.warning(
                    "Groq failed in auto mode, falling back to Ollama. Reason: %s",
                    exc.message,
                )

        # Fall back to Ollama
        logger.info(f"Auto mode: using Ollama ({self._ollama_model})")
        self.model_name = self._ollama_model
        try:
            return await self._call_ollama(messages)
        except AppError as exc:
            if groq_error and groq_error.status_code == 429:
                raise groq_error from exc
            raise

    async def _auto_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
    ) -> dict:
        groq_error: AppError | None = None

        if self._groq_keys:
            try:
                result = await self._call_groq_with_tools(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                )
                self.model_name = self._groq_model
                return result
            except AppError as exc:
                groq_error = exc
                logger.warning(
                    "Groq tool call failed in auto mode, falling back to Ollama. Reason: %s",
                    exc.message,
                )

        logger.info("Auto mode: using Ollama tool fallback (%s)", self._ollama_model)
        self.model_name = self._ollama_model
        try:
            return await self._call_ollama_with_tools(messages, tools)
        except AppError as exc:
            if groq_error and groq_error.status_code == 429:
                raise groq_error from exc
            raise

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """
        Check if the configured LLM provider is reachable.
        Called by GET /health/llm endpoint.
        """
        result = {
            "provider":     self._provider,
            "groq_status":  "not_configured",
            "ollama_status": "not_configured",
        }

        # Check Groq
        if self._groq_keys:
            last_error: Exception | None = None
            for index, api_key in self._ordered_groq_keys():
                try:
                    await self._call_groq_once(
                        [{"role": "user", "content": "hi"}],
                        api_key,
                        self._groq_model,
                        max_tokens=5,
                    )
                    self._groq_key = api_key
                    self.__class__._groq_key_cursor = index
                    result["groq_status"] = "ok"
                    result["groq_model"] = self._groq_model
                    result["groq_keys_configured"] = len(self._groq_keys)
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            else:
                result["groq_status"] = f"error: {str(last_error)[:80]}" if last_error else "error"
        else:
            result["groq_status"] = "no_api_key"

        # Check Ollama
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                if resp.status_code == 200:
                    tags   = resp.json().get("models", [])
                    models = [m["name"] for m in tags]
                    result["ollama_status"] = "ok"
                    result["ollama_models"] = models
                    result["ollama_model_active"] = self._ollama_model
                    result["ollama_model_pulled"] = self._ollama_model in models
                else:
                    result["ollama_status"] = f"http_{resp.status_code}"
        except Exception as e:
            result["ollama_status"] = f"unreachable: {str(e)[:60]}"

        return result


def _normalise_tool_response(message: Any) -> dict:
    """Normalize provider-specific assistant messages into content/tool_calls."""
    content = ""
    raw_tool_calls = None

    if isinstance(message, dict):
        content = str(message.get("content") or "")
        raw_tool_calls = message.get("tool_calls")
    else:
        content = str(getattr(message, "content", "") or "")
        raw_tool_calls = getattr(message, "tool_calls", None)

    tool_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls or []:
        call_id = getattr(item, "id", None)
        name = None
        arguments: Any = {}

        if isinstance(item, dict):
            call_id = item.get("id") or item.get("tool_call_id")
            function = item.get("function") or item
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments", {})
        else:
            function = getattr(item, "function", None)
            name = getattr(function, "name", None) if function is not None else getattr(item, "name", None)
            arguments = (
                getattr(function, "arguments", {})
                if function is not None
                else getattr(item, "arguments", {})
            )

        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
                arguments = parsed_args if isinstance(parsed_args, dict) else {}
            except Exception:
                arguments = {}
        elif not isinstance(arguments, dict):
            arguments = {}

        if name:
            tool_calls.append(
                {
                    "id": str(call_id or ""),
                    "name": str(name),
                    "arguments": arguments,
                }
            )

    return {"content": content, "tool_calls": tool_calls}


def _with_json_tool_fallback_instruction(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Append tool JSON instructions for providers without native tools."""
    tool_brief = [
        {
            "name": (tool.get("function") or {}).get("name"),
            "description": (tool.get("function") or {}).get("description"),
            "parameters": (tool.get("function") or {}).get("parameters"),
        }
        for tool in tools
    ]
    instruction = {
        "role": "system",
        "content": (
            "Native tool calling may be unavailable. Select exactly one tool "
            "from this catalog and return only JSON in this shape: "
            '{"tool_name":"tool_name","parameters":{...}}.\n'
            f"Tool catalog: {json.dumps(tool_brief, ensure_ascii=True)}"
        ),
    }
    return [*messages, instruction]
