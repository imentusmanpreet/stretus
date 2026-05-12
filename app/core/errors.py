"""
Shared API error helpers for Stretus.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_DEFAULT_ERROR_MESSAGES = {
    400: "Bad request. Please review the submitted data and try again.",
    401: "Authentication failed. Please verify your credentials and try again.",
    403: "You do not have permission to perform this action.",
    404: "The requested resource was not found.",
    422: "Validation failed. Please review the submitted data and try again.",
    429: (
        "Rate limit exceeded. You have reached your API usage limit. "
        "Please retry after some time."
    ),
    500: "Something went wrong on our side. Please retry after some time.",
    502: "The upstream service returned an unexpected response. Please retry after some time.",
    503: "A required service is currently unavailable. Please retry after some time.",
    504: "The request timed out. Please retry after some time.",
}


class AppError(Exception):
    def __init__(self, status_code: int, message: str, detail: Any = None):
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message
        self.detail = detail


def _default_message(status_code: int) -> str:
    return _DEFAULT_ERROR_MESSAGES.get(status_code, _DEFAULT_ERROR_MESSAGES[500])


def _clean_message(message: Any) -> str | None:
    text = " ".join(str(message or "").split()).strip()
    return text or None


def _message_from_detail(detail: Any) -> str | None:
    if detail is None:
        return None

    if isinstance(detail, str):
        return _clean_message(detail)

    if isinstance(detail, dict):
        nested_error = detail.get("error")
        if isinstance(nested_error, dict):
            nested_message = _message_from_detail(nested_error.get("message"))
            if nested_message:
                return nested_message

        for key in ("message", "detail", "error_message", "reason"):
            nested_message = _message_from_detail(detail.get(key))
            if nested_message:
                return nested_message
        return None

    if isinstance(detail, list):
        parts = [_message_from_detail(item) for item in detail[:3]]
        joined = "; ".join(part for part in parts if part)
        return joined or None

    return _clean_message(detail)


def _resolved_message(status_code: int, message: Any = None) -> str:
    return _clean_message(message) or _default_message(status_code)


def build_error_content(status_code: int, message: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": int(status_code),
            "message": _resolved_message(status_code, message),
        }
    }


def build_error_object(status_code: int, message: str | None = None) -> dict[str, Any]:
    return build_error_content(status_code, message)["error"]


def _validation_message(exc: RequestValidationError) -> str:
    issues: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        message = _clean_message(error.get("msg")) or "Invalid value."
        issues.append(f"{location}: {message}" if location else message)

    if not issues:
        return _default_message(422)

    return f"Validation failed. {'; '.join(issues)}"


def _httpx_response_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    payload_message = _message_from_detail(payload)
    if payload_message:
        return _resolved_message(response.status_code, payload_message)

    return _resolved_message(response.status_code, response.text)


def _looks_like_rate_limit_error(exc: Exception, message: str | None) -> bool:
    text = (message or "").lower()
    class_name = exc.__class__.__name__.lower()
    return (
        "ratelimit" in class_name
        or "rate limit" in text
        or "too many requests" in text
        or "usage limit" in text
        or "429" in text
    )


def _looks_like_auth_error(exc: Exception, message: str | None) -> bool:
    text = (message or "").lower()
    class_name = exc.__class__.__name__.lower()
    return (
        "authentication" in class_name
        or "unauthorized" in text
        or "authentication" in text
        or "api key" in text
        or "401" in text
    )


def normalize_exception(exc: Exception, default_status_code: int = 500) -> AppError:
    if isinstance(exc, AppError):
        return exc

    if isinstance(exc, HTTPException):
        return AppError(
            exc.status_code,
            _resolved_message(exc.status_code, _message_from_detail(exc.detail)),
            detail=exc.detail,
        )

    if isinstance(exc, RequestValidationError):
        return AppError(422, _validation_message(exc), detail=exc.errors())

    if isinstance(exc, httpx.TimeoutException):
        return AppError(504, _default_message(504))

    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response is not None:
            return AppError(
                response.status_code,
                _httpx_response_message(response),
                detail=response.text,
            )
        return AppError(502, _default_message(502))

    if isinstance(exc, httpx.RequestError):
        return AppError(503, _default_message(503))

    message = _message_from_detail(getattr(exc, "detail", None)) or _clean_message(str(exc))

    if _looks_like_rate_limit_error(exc, message):
        return AppError(429, _default_message(429), detail=message)

    if _looks_like_auth_error(exc, message):
        return AppError(401, _default_message(401), detail=message)

    if isinstance(exc, PermissionError):
        return AppError(
            500,
            "The server cannot write strategy files right now. "
            "Please fix the strategies folder permissions and retry.",
            detail=message,
        )

    if isinstance(exc, ValueError) and message:
        return AppError(400, message, detail=message)

    return AppError(default_status_code, _default_message(default_status_code), detail=message)


def install_exception_handlers(app: FastAPI) -> None:
    async def _app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        logger.log(level, "API error %s: %s", exc.status_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_content(exc.status_code, exc.message),
        )

    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return await _app_error_handler(request, normalize_exception(exc))

    async def _validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return await _app_error_handler(request, normalize_exception(exc))

    async def _unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        app_error = normalize_exception(exc)
        if app_error.status_code >= 500:
            logger.exception(
                "Unhandled exception while serving %s %s",
                request.method,
                request.url.path,
            )
        else:
            logger.warning(
                "Request failed for %s %s with %s: %s",
                request.method,
                request.url.path,
                app_error.status_code,
                app_error.message,
            )

        return JSONResponse(
            status_code=app_error.status_code,
            content=build_error_content(app_error.status_code, app_error.message),
        )

    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unexpected_exception_handler)
