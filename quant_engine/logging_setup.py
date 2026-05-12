"""
Readable console logging for the quant engine service.
"""
from __future__ import annotations

from copy import copy
import logging
from logging.config import dictConfig

_LOGGING_CONFIGURED = False


class PrettyFormatter(logging.Formatter):
    LEVEL_ICONS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }
    COMPONENT_ALIASES = {
        "main": "quant",
        "uvicorn": "server",
        "uvicorn.error": "server",
        "uvicorn.access": "http",
    }

    def format(self, record: logging.LogRecord) -> str:
        cloned = copy(record)
        cloned.level_icon = self.LEVEL_ICONS.get(cloned.levelno, "•")
        cloned.component = self._component_name(cloned.name)

        if cloned.name == "uvicorn.access" and isinstance(cloned.args, tuple) and len(cloned.args) >= 5:
            client_addr, method, full_path, _http_version, status_code = cloned.args[:5]
            cloned.msg = "%s %s -> %s [%s]"
            cloned.args = (method, full_path, status_code, client_addr)
            cloned.level_icon = "🌐"
            cloned.component = "http"

        return super().format(cloned)

    def _component_name(self, logger_name: str) -> str:
        alias = self.COMPONENT_ALIASES.get(logger_name)
        if alias:
            return alias

        parts = [part for part in str(logger_name or "").split(".") if part]
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return logger_name or "quant"


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "pretty": {
                    "()": "logging_setup.PrettyFormatter",
                    "format": "%(asctime)s | %(levelname)-7s | %(level_icon)s %(component)s | %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "pretty",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"],
            },
            "loggers": {
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "sqlalchemy": {"level": "WARNING"},
                "sqlalchemy.engine": {"level": "WARNING"},
                "sqlalchemy.pool": {"level": "WARNING"},
            },
        }
    )
    _LOGGING_CONFIGURED = True
