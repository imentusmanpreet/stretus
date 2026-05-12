"""
app/services/strategy/yaml_generator.py
"""
import logging
import os
import re
import yaml
from app.core.config import get_settings
from app.core.errors import AppError
from app.services.strategy.builder import StrategyBuilder

settings = get_settings()
logger = logging.getLogger(__name__)


def _candidate_strategy_folders() -> list[str]:
    folders = [settings.strategy_folder, "./data/strategies"]
    ordered: list[str] = []
    for folder in folders:
        cleaned = str(folder or "").strip()
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return ordered


def generate_yaml(builder: StrategyBuilder) -> str:
    """Write strategy YAML to disk. Returns the file path."""
    data      = builder.to_yaml_dict()
    name      = data["strategy"]["name"]
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()

    last_permission_error: PermissionError | None = None
    for index, folder in enumerate(_candidate_strategy_folders()):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{safe_name}.yaml")
        try:
            with open(filepath, "w") as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False)
            if index > 0:
                logger.warning(
                    "Strategy YAML fallback activated | configured_folder=%s fallback_folder=%s file=%s",
                    settings.strategy_folder,
                    folder,
                    filepath,
                )
            return filepath
        except PermissionError as exc:
            last_permission_error = exc
            continue

    raise AppError(
        500,
        "The server cannot write strategy files right now. "
        "Please fix the strategies folder permissions and retry.",
    ) from last_permission_error
