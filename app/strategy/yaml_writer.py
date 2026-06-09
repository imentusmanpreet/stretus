"""
app/strategy/yaml_writer.py — render a validated StrategySpec to an engine YAML file (🏗️ render).

The quant engine's ``/run-sync`` reads a YAML FILE (its loader expects a top-level
``strategy:`` mapping). This writes ``{"strategy": spec.to_engine_yaml_dict()}`` to the
configured strategy folder and returns the path, reusing the same folder-resolution +
permission-fallback behaviour as the legacy ``app/services/strategy/yaml_generator.py``
so both paths drop files in the same place.
"""
from __future__ import annotations

import logging
import os
import re

import yaml

from app.core import step_logger
from app.core.config import get_settings
from app.core.errors import AppError
from app.services.strategy.yaml_generator import _candidate_strategy_folders
from app.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)
settings = get_settings()


def _safe_filename(spec: StrategySpec) -> str:
    base = f"{spec.symbol}_{spec.timeframe}_{spec.objective}".strip() or spec.name
    return re.sub(r"[^a-zA-Z0-9]", "_", base).lower() or "strategy"


def write_spec_yaml(spec: StrategySpec, *, session_id: str | None = None, turn: int = 0) -> str:
    """Serialise ``spec`` to a YAML file the engine can load. Returns the file path.

    Raises AppError(500) only if every candidate folder is unwritable — mirrors the
    legacy generator so the failure mode is identical and observable.
    """
    document = {"strategy": spec.to_engine_yaml_dict()}
    filename = f"{_safe_filename(spec)}.yaml"

    last_permission_error: PermissionError | None = None
    for index, folder in enumerate(_candidate_strategy_folders()):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "w") as handle:
                yaml.dump(document, handle, sort_keys=False, default_flow_style=False)
        except PermissionError as exc:
            last_permission_error = exc
            logger.warning("⚠️ render | folder not writable, trying next | folder=%s", folder)
            continue
        if index > 0:
            logger.warning(
                "⚠️ render | strategy YAML written to fallback folder | configured=%s used=%s",
                settings.strategy_folder, folder,
            )
        step_logger.log_step(
            step_logger.RENDER, step_logger.RENDER, session_id, turn,
            "yaml_written", file=filepath, symbol=spec.symbol, tf=spec.timeframe,
        )
        logger.info("🏗️ render | strategy YAML written | file=%s", filepath)
        return filepath

    raise AppError(
        500,
        "The server cannot write strategy files right now. "
        "Please fix the strategies folder permissions and retry.",
    ) from last_permission_error
