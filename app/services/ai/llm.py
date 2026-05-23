"""
app/services/ai/llm.py
═══════════════════════
LLMService — supports three providers:

  Provider     Config                   Description
  ──────────   ─────────────────────── ────────────────────────────────────────
  openrouter   LLM_PROVIDER=openrouter OpenRouter Cloud API (unified LLM access)
  ollama       LLM_PROVIDER=ollama     Local model via Ollama (private, no API key)
  auto         LLM_PROVIDER=auto       Try OpenRouter → Ollama

Set in .env:
  LLM_PROVIDER=openrouter
  # OpenRouter keys are managed via JSON state file (openrouter_key_state.json)
  OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

  OR

  LLM_PROVIDER=ollama
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=qwen2.5:7b

  OR

  LLM_PROVIDER=auto   (tries openrouter → ollama automatically)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from app.core.config import get_settings
from app.core.errors import AppError
from app.services.ai.openrouter_key_manager import get_openrouter_key_manager
from app.core.token_tracker import track_tokens

logger   = logging.getLogger(__name__)


# ── Available OpenRouter models ───────────────────────────────────────────────
OPENROUTER_MODELS = {
    "meta-llama/llama-3.3-70b-instruct": "Best quality — recommended",
    "anthropic/claude-3.5-sonnet": "Excellent reasoning and coding",
    "google/gemini-pro-1.5": "Good for long contexts",
    "openai/gpt-4-turbo": "OpenAI's best model",
    "mistralai/mixtral-8x7b-instruct": "Good balance of speed and quality",
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
    Unified LLM service supporting OpenRouter Cloud and local Ollama.

    Usage:
        llm = LLMService()
        response = await llm.chat(messages)

    The provider is chosen from settings — no code changes needed,
    just update .env and restart.
    """

    def __init__(self):
        current_settings = get_settings()
        self._provider = current_settings.effective_provider()
        
        # OpenRouter configuration - load keys from .env and pass to manager
        openrouter_keys_from_env = current_settings.openrouter_api_key_pool()
        if openrouter_keys_from_env:
            self._openrouter_key_manager = get_openrouter_key_manager(openrouter_keys_from_env)
            self._openrouter_key = self._openrouter_key_manager.get_active_key()
        else:
            self._openrouter_key_manager = None
            self._openrouter_key = None
        
        self._openrouter_model = current_settings.openrouter_model
        
        # Ollama configuration
        self._ollama_url = current_settings.ollama_base_url
        self._ollama_model = current_settings.ollama_model

        # What model name to expose (used for logging / DB storage)
        if self._provider == "openrouter":
            self.model_name = self._openrouter_model
        elif self._provider == "ollama":
            self.model_name = self._ollama_model
        else:  # auto
            self.model_name = f"{self._openrouter_model} (auto)"

        keys_info = "not configured"
        if self._openrouter_key_manager:
            stats = self._openrouter_key_manager.get_stats()
            keys_info = f"{stats['keys_remaining']} keys available (loaded from .env)"
        
        logger.info(
            f"LLMService initialised — provider={self._provider}  "
            f"model={self.model_name}  "
            f"openrouter_keys={keys_info}"
        )

    # ── Public method ─────────────────────────────────────────────────────────

    async def chat(self, messages: list[dict], session_id: str | None = None) -> str:
        """
        Send a message list to the LLM and return the text response.

        messages format:
            [
                {"role": "system",    "content": "You are ..."},
                {"role": "user",      "content": "I want to trade NIFTY"},
                {"role": "assistant", "content": "Great! Which timeframe?"},
                {"role": "user",      "content": "15m"},
            ]
        
        Args:
            messages: List of message dictionaries
            session_id: Optional session ID for token tracking
        """
        if self._provider == "openrouter":
            return await self._call_openrouter(messages, session_id=session_id)

        elif self._provider == "ollama":
            return await self._call_ollama(messages)

        else:  # auto — try openrouter first, fall back to ollama
            return await self._auto(messages, session_id=session_id)

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
        session_id: str | None = None,
    ) -> dict:
        """
        Send messages with tool definitions and return a normalized response:

            {
                "content": "optional assistant text",
                "tool_calls": [
                    {"id": "...", "name": "run_backtest", "arguments": {...}}
                ]
            }

        OpenRouter receives native function/tool definitions. Ollama uses native
        tools when the installed SDK supports them, otherwise a strict JSON
        tool-selection fallback is used.
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            tool_choice: Tool selection strategy
            session_id: Optional session ID for token tracking
        """
        if self._provider == "openrouter":
            return await self._call_openrouter_with_tools(messages, tools, tool_choice=tool_choice, session_id=session_id)
        if self._provider == "ollama":
            return await self._call_ollama_with_tools(messages, tools)
        return await self._auto_with_tools(messages, tools, tool_choice=tool_choice, session_id=session_id)

    # ── Provider info ─────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return current provider config — useful for /health endpoint."""
        result = {
            "provider":      self._provider,
            "openrouter_model":    self._openrouter_model,
            "ollama_model":  self._ollama_model,
            "ollama_url":    self._ollama_url,
            "active_model":  self.model_name,
            "openrouter_key_set":  bool(self._openrouter_key),
        }
        
        if self._openrouter_key_manager:
            stats = self._openrouter_key_manager.get_stats()
            result.update({
                "openrouter_keys_remaining": stats['keys_remaining'],
                "openrouter_active_key_index": stats['active_key_index'],
                "openrouter_keys_total": stats['total_keys'],
            })
        
        return result

    # ── OpenRouter Cloud ──────────────────────────────────────────────────────

    async def _call_openrouter_once(
        self,
        messages: list[dict],
        api_key: str,
        model: Optional[str] = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        session_id: str | None = None,
    ) -> str:
        """
        Call the OpenRouter Cloud API.

        OpenRouter provides unified access to multiple LLM providers.
        Get your API key at: https://openrouter.ai/keys
        """
        use_model = model or self._openrouter_model

        try:
            from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, BadRequestError, NotFoundError, RateLimitError, APIError
        except ImportError as exc:
            raise AppError(
                503,
                "OpenAI client is not available on the server. Please retry after some time.",
            ) from exc

        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            resp = await client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = resp.choices[0].message.content
            logger.debug(f"OpenRouter response — model={use_model}  tokens={resp.usage.total_tokens if resp.usage else 'N/A'}")
            
            # Track token usage
            if session_id and resp.usage:
                track_tokens(
                    session_id=session_id,
                    provider="openrouter",
                    model=use_model,
                    total_tokens=resp.usage.total_tokens,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
            
            return content

        except RateLimitError as exc:
            logger.warning("OpenRouter rate limit reached for model=%s: %s", use_model, exc)
            
            # Rotate to next key
            new_key = self._openrouter_key_manager.rotate_key_on_exhaustion(api_key)
            if new_key:
                self._openrouter_key = new_key
            
            raise AppError(
                429,
                (
                    "Rate limit exceeded. You have reached your API usage limit. "
                    "Please retry after some time."
                ),
            ) from exc
        except AuthenticationError as exc:
            logger.error("OpenRouter authentication failed for model=%s: %s", use_model, exc)
            raise AppError(
                401,
                "Authentication failed. Please verify the configured OpenRouter API key and try again.",
            ) from exc
        except NotFoundError as exc:
            logger.error("OpenRouter model not found: %s", use_model)
            raise AppError(
                404,
                f"The configured OpenRouter model '{use_model}' was not found. Please verify the model name and try again.",
            ) from exc
        except APIConnectionError as exc:
            logger.error("OpenRouter connection error for model=%s: %s", use_model, exc)
            raise AppError(
                503,
                "Unable to reach OpenRouter right now. Please retry after some time.",
            ) from exc
        except BadRequestError as exc:
            logger.error("OpenRouter request rejected for model=%s: %s", use_model, exc)
            raise AppError(
                400,
                str(exc).strip() or "The OpenRouter request was invalid. Please review the request and try again.",
            ) from exc
        except APIError as exc:
            logger.error("OpenRouter error for model=%s: %s", use_model, exc)
            raise AppError(
                502,
                "OpenRouter could not process the request right now. Please retry after some time.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected OpenRouter error for model=%s", use_model)
            raise AppError(
                500,
                "An unexpected LLM error occurred. Please retry after some time.",
            ) from exc

    async def _call_openrouter(self, messages: list[dict], model: Optional[str] = None, session_id: str | None = None) -> str:
        if not self._openrouter_key:
            raise AppError(
                503,
                "OpenRouter is not configured on the server. Please retry after some time.",
            )

        use_model = model or self._openrouter_model
        
        # Get the active key from the manager
        api_key = self._openrouter_key_manager.get_active_key()
        
        try:
            result = await self._call_openrouter_once(messages, api_key, use_model, session_id=session_id)
            return result
        except AppError as exc:
            if exc.status_code == 429:
                # Key was already rotated in _call_openrouter_once
                # Try once more with the new key
                stats = self._openrouter_key_manager.get_stats()
                if stats['keys_remaining'] > 0:
                    logger.info(f"Retrying with rotated key ({stats['keys_remaining']} keys remaining)")
                    new_key = self._openrouter_key_manager.get_active_key()
                    try:
                        result = await self._call_openrouter_once(messages, new_key, use_model, session_id=session_id)
                        return result
                    except AppError:
                        pass  # Fall through to raise the original error
                
                raise AppError(
                    429,
                    "All configured OpenRouter API keys are currently rate limited. Please retry after some time.",
                )
            raise

    async def _call_openrouter_tools_once(
        self,
        messages: list[dict],
        tools: list[dict],
        api_key: str,
        model: Optional[str] = None,
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        session_id: str | None = None,
    ) -> dict:
        """Call OpenRouter with native tool definitions and normalize the result."""
        use_model = model or self._openrouter_model

        try:
            from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, BadRequestError, NotFoundError, RateLimitError, APIError
        except ImportError as exc:
            raise AppError(
                503,
                "OpenAI client is not available on the server. Please retry after some time.",
            ) from exc

        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            resp = await client.chat.completions.create(
                model=use_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Track token usage
            if session_id and resp.usage:
                track_tokens(
                    session_id=session_id,
                    provider="openrouter",
                    model=use_model,
                    total_tokens=resp.usage.total_tokens,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
            
            return _normalise_tool_response(resp.choices[0].message)
        except RateLimitError as exc:
            logger.warning("OpenRouter rate limit reached for tool call model=%s: %s", use_model, exc)
            
            # Rotate to next key
            new_key = self._openrouter_key_manager.rotate_key_on_exhaustion(api_key)
            if new_key:
                self._openrouter_key = new_key
            
            raise AppError(429, "Rate limit exceeded. Please retry after some time.") from exc
        except AuthenticationError as exc:
            logger.error("OpenRouter authentication failed for tool call model=%s: %s", use_model, exc)
            raise AppError(
                401,
                "Authentication failed. Please verify the configured OpenRouter API key and try again.",
            ) from exc
        except NotFoundError as exc:
            logger.error("OpenRouter tool-call model not found: %s", use_model)
            raise AppError(
                404,
                f"The configured OpenRouter model '{use_model}' was not found.",
            ) from exc
        except APIConnectionError as exc:
            logger.error("OpenRouter connection error for tool call model=%s: %s", use_model, exc)
            raise AppError(503, "Unable to reach OpenRouter right now. Please retry after some time.") from exc
        except BadRequestError as exc:
            logger.error("OpenRouter tool-call request rejected for model=%s: %s", use_model, exc)
            raise AppError(400, str(exc).strip() or "The OpenRouter tool request was invalid.") from exc
        except APIError as exc:
            logger.error("OpenRouter tool-call error for model=%s: %s", use_model, exc)
            raise AppError(502, "OpenRouter could not process the tool request right now.") from exc
        except Exception as exc:
            logger.exception("Unexpected OpenRouter tool-call error for model=%s", use_model)
            raise AppError(500, "An unexpected LLM tool-call error occurred.") from exc

    async def _call_openrouter_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: Optional[str] = None,
        *,
        tool_choice: str | dict = "auto",
        session_id: str | None = None,
    ) -> dict:
        if not self._openrouter_key:
            raise AppError(
                503,
                "OpenRouter is not configured on the server. Please retry after some time.",
            )

        use_model = model or self._openrouter_model
        
        # Get the active key from the manager
        api_key = self._openrouter_key_manager.get_active_key()
        
        try:
            result = await self._call_openrouter_tools_once(
                messages,
                tools,
                api_key,
                use_model,
                tool_choice=tool_choice,
                session_id=session_id,
            )
            return result
        except AppError as exc:
            if exc.status_code == 429:
                # Key was already rotated in _call_openrouter_tools_once
                # Try once more with the new key
                stats = self._openrouter_key_manager.get_stats()
                if stats['keys_remaining'] > 0:
                    logger.info(f"Retrying tool call with rotated key ({stats['keys_remaining']} keys remaining)")
                    new_key = self._openrouter_key_manager.get_active_key()
                    try:
                        result = await self._call_openrouter_tools_once(
                            messages,
                            tools,
                            new_key,
                            use_model,
                            tool_choice=tool_choice,
                            session_id=session_id,
                        )
                        return result
                    except AppError:
                        pass  # Fall through to raise the original error
                
                raise AppError(
                    429,
                    "All configured OpenRouter API keys are currently rate limited. Please retry after some time.",
                )
            raise

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

    # ── Auto (try OpenRouter, fall back to Ollama) ───────────────────────────

    async def _auto(self, messages: list[dict], session_id: str | None = None) -> str:
        """
        Try OpenRouter first. If it fails (no key, rate limit, network issue),
        automatically fall back to the local Ollama model.

        Good for development — use OpenRouter when available, Ollama offline.
        """
        openrouter_error: AppError | None = None

        if self._openrouter_key:
            try:
                result = await self._call_openrouter(messages, session_id=session_id)
                self.model_name = self._openrouter_model
                return result
            except AppError as exc:
                openrouter_error = exc
                logger.warning(
                    "OpenRouter failed in auto mode, falling back to Ollama. Reason: %s",
                    exc.message,
                )

        # Fall back to Ollama
        logger.info(f"Auto mode: using Ollama ({self._ollama_model})")
        self.model_name = self._ollama_model
        try:
            return await self._call_ollama(messages)
        except AppError as exc:
            if openrouter_error and openrouter_error.status_code == 429:
                raise openrouter_error from exc
            raise

    async def _auto_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
        session_id: str | None = None,
    ) -> dict:
        openrouter_error: AppError | None = None

        if self._openrouter_key:
            try:
                result = await self._call_openrouter_with_tools(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                    session_id=session_id,
                )
                self.model_name = self._openrouter_model
                return result
            except AppError as exc:
                openrouter_error = exc
                logger.warning(
                    "OpenRouter tool call failed in auto mode, falling back to Ollama. Reason: %s",
                    exc.message,
                )

        logger.info("Auto mode: using Ollama tool fallback (%s)", self._ollama_model)
        self.model_name = self._ollama_model
        try:
            return await self._call_ollama_with_tools(messages, tools)
        except AppError as exc:
            if openrouter_error and openrouter_error.status_code == 429:
                raise openrouter_error from exc
            raise

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """
        Check if the configured LLM provider is reachable.
        Called by GET /health/llm endpoint.
        """
        result = {
            "provider":     self._provider,
            "openrouter_status":  "not_configured",
            "ollama_status": "not_configured",
        }

        # Check OpenRouter
        if self._openrouter_key:
            try:
                api_key = self._openrouter_key_manager.get_active_key()
                await self._call_openrouter_once(
                    [{"role": "user", "content": "hi"}],
                    api_key,
                    self._openrouter_model,
                    max_tokens=5,
                )
                stats = self._openrouter_key_manager.get_stats()
                result["openrouter_status"] = "ok"
                result["openrouter_model"] = self._openrouter_model
                result["openrouter_keys_remaining"] = stats['keys_remaining']
                result["openrouter_active_key_index"] = stats['active_key_index']
            except Exception as exc:
                result["openrouter_status"] = f"error: {str(exc)[:80]}"
        else:
            result["openrouter_status"] = "no_api_key"

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
