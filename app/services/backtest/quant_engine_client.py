"""
HTTP helpers for talking to the quant engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import httpx

from app.core.config import refresh_settings
from app.schemas.backtest import BacktestTriggerRequest

logger = logging.getLogger(__name__)


async def queue_quant_backtest(
    *,
    backtest_id: str,
    strategy_id: str,
    yaml_path: str,
    ohlcv_data: list[dict[str, Any]],
    run_config: BacktestTriggerRequest,
    market_data_request: dict[str, Any],
    # Phase 7 — auxiliary OHLCV the engine needs when the strategy declares
    # reference_symbol (Phase 4) and/or htf rules (Phase 5). Both optional.
    reference_ohlcv: list[dict[str, Any]] | None = None,
    htf_ohlcv: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    settings = refresh_settings()
    yaml_content = Path(yaml_path).read_text(encoding="utf-8")
    payload: dict[str, Any] = {
        "backtest_id": backtest_id,
        "strategy_id": strategy_id,
        "yaml_content": yaml_content,
        "ohlcv_data": ohlcv_data,
        "run_config": run_config.model_dump(),
        "market_data_request": market_data_request,
    }
    if reference_ohlcv is not None:
        payload["reference_ohlcv"] = reference_ohlcv
    if htf_ohlcv:
        payload["htf_ohlcv"] = htf_ohlcv

    try:
        async with httpx.AsyncClient(timeout=settings.quant_engine_timeout_seconds) as client:
            response = await client.post(
                f"{settings.quant_engine_url}/run",
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.exception("Quant engine rejected queued backtest with HTTP %s", exc.response.status_code)
        raise RuntimeError(
            f"Quant engine rejected the backtest request with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("Failed to queue backtest with quant engine")
        raise RuntimeError("Failed to reach the quant engine service.") from exc


async def run_quant_backtest_sync(
    *,
    yaml_path: str,
    ohlcv_data: list[dict[str, Any]],
    run_config: BacktestTriggerRequest,
    market_data_request: dict[str, Any],
    backtest_ref_id: str | None = None,
    reference_ohlcv: list[dict[str, Any]] | None = None,
    htf_ohlcv: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    settings = refresh_settings()
    yaml_content = Path(yaml_path).read_text(encoding="utf-8")
    payload: dict[str, Any] = {
        "yaml_content": yaml_content,
        "ohlcv_data": ohlcv_data,
        "run_config": run_config.model_dump(),
        "market_data_request": market_data_request,
    }
    if backtest_ref_id:
        payload["backtest_ref_id"] = backtest_ref_id
    if reference_ohlcv is not None:
        payload["reference_ohlcv"] = reference_ohlcv
    if htf_ohlcv:
        payload["htf_ohlcv"] = htf_ohlcv

    try:
        async with httpx.AsyncClient(timeout=settings.quant_engine_timeout_seconds) as client:
            response = await client.post(
                f"{settings.quant_engine_url}/run-sync",
                json=payload,
            )
            if response.status_code == 404:
                raise RuntimeError(
                    "The running quant engine does not expose /run-sync yet. "
                    "It is likely an older container/build. Rebuild and restart the quant_engine service, "
                    "then retry the backtest."
                )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        logger.exception("Quant engine synchronous run failed with HTTP %s", exc.response.status_code)
        raise RuntimeError(
            f"Quant engine synchronous backtest failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("Failed to reach quant engine for synchronous backtest")
        raise RuntimeError("Failed to reach the quant engine service.") from exc

    if not isinstance(result, dict) or "metrics" not in result:
        raise ValueError("Quant engine returned an unexpected backtest response payload.")

    return result

