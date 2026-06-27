"""
app/services/strategy/yaml_generator.py
"""
from __future__ import annotations

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


def generate_yaml(builder: StrategyBuilder, *, symbol_override: str | None = None) -> str:
    """Write strategy YAML to disk. Returns the file path.

    ``symbol_override`` swaps only the engine-facing ``strategy.symbol`` so the
    backtest's trades are tagged with the asset actually being run — used by a
    single-asset backtest whose symbol was overridden via the run_backtest
    ``symbols`` arg. The ``builder``/session is NOT mutated; None → builder's own
    symbol (legacy behaviour, unchanged).
    """
    data      = builder.to_yaml_dict()
    if symbol_override:
        strategy_section = data.get("strategy")
        if isinstance(strategy_section, dict):
            strategy_section["symbol"] = str(symbol_override)
    name      = data["strategy"]["name"]
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()

    # Dynamic universe: emit the `universe:` block alongside the strategy template so the
    # backtest route branches to the portfolio path. The strategy body is symbol-agnostic
    # (members are stamped per-resolution at backtest time) — stamp a placeholder symbol.
    universe_block = getattr(builder, "universe", None)
    if isinstance(universe_block, dict) and universe_block:
        data["universe"] = universe_block
        if isinstance(data.get("strategy"), dict):
            data["strategy"]["symbol"] = data["strategy"].get("symbol") or "UNIVERSE_MEMBER"

    strategy_block = data.get("strategy", {}) if isinstance(data, dict) else {}
    signals = strategy_block.get("signals") or strategy_block.get("entry") or []
    logger.info(
        "🏗️  Assembling strategy YAML | name=%s symbol=%s timeframe=%s signals=%s",
        name,
        symbol_override or getattr(builder, "symbol", None),
        getattr(builder, "timeframe", None),
        len(signals) if isinstance(signals, (list, dict)) else signals,
    )

    last_permission_error: PermissionError | None = None
    for index, folder in enumerate(_candidate_strategy_folders()):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{safe_name}.yaml")
        try:
            with open(filepath, "w") as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False)
            if index > 0:
                logger.warning(
                    "⚠️ Strategy YAML fallback activated | configured_folder=%s fallback_folder=%s file=%s",
                    settings.strategy_folder,
                    folder,
                    filepath,
                )
            logger.info("✅ Strategy YAML written | name=%s file=%s", name, filepath)
            return filepath
        except PermissionError as exc:
            last_permission_error = exc
            continue

    raise AppError(
        500,
        "The server cannot write strategy files right now. "
        "Please fix the strategies folder permissions and retry.",
    ) from last_permission_error
