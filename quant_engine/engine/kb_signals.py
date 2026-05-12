"""
Knowledge-base signal evaluation for the quant engine.

Maps OHLCV frames produced by engine/data.py (lowercase columns) to the
column names expected by stretus_kb signal functions (Title Case), then
delegates to RuleRegistry.evaluate().

This path unlocks every registered KB signal without translating them into
formula strings for conditions.py.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

KB_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR: Exception | None = None
RuleRegistry: Any = None

try:
    import stretus_knowledge_base.stretus_kb  # noqa: F401 — registers all signals
    from stretus_knowledge_base.stretus_kb.registry import RuleRegistry as _RuleRegistry

    RuleRegistry = _RuleRegistry
    KB_REGISTRY_AVAILABLE = True
except Exception as exc:  # pragma: no cover — deployment without KB package
    KB_REGISTRY_AVAILABLE = False
    _REGISTRY_IMPORT_ERROR = exc
    logger.warning(
        "KB signal registry could not be loaded — registry evaluation disabled (%s).",
        exc,
    )


def kb_registry_import_error() -> Exception | None:
    return _REGISTRY_IMPORT_ERROR


def ohlcv_df_for_kb(df: pd.DataFrame) -> pd.DataFrame:
    """Rename quant-engine columns to the Title Case schema used by stretus_kb."""
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns required for KB signals: {missing}")
    out = df[list(rename_map.keys())].rename(columns=rename_map)
    return out


def evaluate_kb_signal_rules(
    rules: list[dict[str, Any]],
    df: pd.DataFrame,
    bar_index: int,
) -> bool:
    """
    True when every rule fires on the slice ending at bar_index (inclusive).

    Rules use the same shape as RuleRegistry.evaluate: {"name": str, "params": {...}}.
    """
    if not rules:
        return False
    if not KB_REGISTRY_AVAILABLE or RuleRegistry is None:
        raise RuntimeError(
            "Knowledge-base signals are not available (import failed). "
            "Install stretus_knowledge_base and dependencies (e.g. ta) or use formula mode."
        ) from _REGISTRY_IMPORT_ERROR

    slice_df = df.iloc[: bar_index + 1]
    kb_df = ohlcv_df_for_kb(slice_df)

    for rule in rules:
        name = str(rule.get("name") or "").strip()
        if not name:
            return False
        payload = {"name": name, "params": dict(rule.get("params") or {})}
        if not RuleRegistry.evaluate(payload, kb_df):
            return False
    return True


_WARMUP_PARAM_KEYS = frozenset({
    "window",
    "window_fast",
    "window_slow",
    "window_sign",
    "opening_bars",
    "smooth_window",
    "lbp",
})


def estimate_kb_warmup(rules: list[dict[str, Any]]) -> int:
    """
    Conservative warm-up bar count so the last row of each slice has valid indicators.

    Used when entry/exit use registry evaluation instead of YAML indicator specs.
    """
    max_n = 20
    for rule in rules:
        params = dict(rule.get("params") or {})
        for key, val in params.items():
            if key in _WARMUP_PARAM_KEYS or str(key).endswith("_period"):
                try:
                    max_n = max(max_n, int(float(val)))
                except (TypeError, ValueError):
                    continue
        name = str(rule.get("name") or "").lower()
        if "macd" in name:
            max_n = max(max_n, 35)
        if "adx" in name:
            max_n = max(max_n, 28)
        if "supertrend" in name or "ichimoku" in name:
            max_n = max(max_n, 52)
    return min(max_n + 15, 600)
