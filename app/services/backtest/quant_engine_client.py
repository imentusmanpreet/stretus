"""
HTTP helpers for talking to the quant engine.

The forward path posts a full OHLCV payload (up to ~1.3M rows for long
intraday backtests) to quant_engine's /run-sync endpoint. The naive
``httpx.post(..., json=payload)`` route allocates the full JSON twice
(once as a Python str, once as bytes) and ~doubles the API pod's
resident memory during the call. On a 2 GiB-limit pod that's enough to
trip an OOM-kill mid-backtest.

This module instead:
  - Converts the per-row dicts to 6-element lists at the wire boundary so
    the on-wire JSON is ~50% smaller (no repeated key names). Quant's
    loader at engine/data._normalize_record already accepts both shapes,
    so this is a wire-only optimization with no engine-side change.
  - Streams the request body via httpx ``content=AsyncIterator[bytes]``
    so the encoded body never has to exist in full in memory. Peak per
    chunk is ~1 MiB instead of ~150 MiB.
  - Encodes with orjson, which is materially faster than stdlib json for
    this shape of payload (~3–10×) and avoids the intermediate str.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import orjson

from app.core.config import refresh_settings
from app.schemas.backtest import BacktestTriggerRequest

logger = logging.getLogger(__name__)


# How many OHLCV rows to encode per chunk while streaming. 5_000 rows per
# chunk = ~300 KiB of JSON, which keeps the per-chunk allocation small but
# avoids the per-chunk overhead of going too granular.
_OHLCV_STREAM_BATCH = 5_000

# Six fixed columns in the order quant_engine/_normalize_record expects when
# the input is a list (timestamp, open, high, low, close, volume).
_OHLCV_KEYS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


def _row_to_list(row: dict[str, Any]) -> list[Any]:
    """Project an OHLCV dict to the 6-element list shape quant accepts.

    Hot path: called once per OHLCV row. Kept as a free function so the
    interpreter avoids the per-call dict.get attribute lookup.
    """
    return [row[k] for k in _OHLCV_KEYS]


def _encode_meta_open(payload_meta: dict[str, Any]) -> bytes:
    """Encode the JSON envelope, leaving an open ``"ohlcv_data":[`` for streaming.

    `payload_meta` must NOT contain the ``ohlcv_data`` key — it's added by
    the streaming body. Returns bytes that look like::

        {"yaml_content":"...","run_config":{...},...,"ohlcv_data":[
    """
    encoded = orjson.dumps(payload_meta)  # → b'{...}'
    if not encoded.startswith(b"{") or not encoded.endswith(b"}"):
        # Defensive: orjson should always produce a JSON object here.
        raise RuntimeError("Expected orjson to produce a JSON object envelope.")
    # Drop the trailing '}' and append the ohlcv_data array opener.
    head = encoded[:-1]
    if encoded == b"{}":
        # Empty envelope — no leading comma needed.
        return head + b'"ohlcv_data":['
    return head + b',"ohlcv_data":['


async def _stream_run_payload(
    payload_meta: dict[str, Any],
    ohlcv_data: list[dict[str, Any]],
) -> AsyncIterator[bytes]:
    """Yield the JSON request body for /run or /run-sync in chunks.

    Memory model:
      - The full ``payload_meta`` (yaml + run_config + market_data_request +
        optional reference/htf OHLCV) is encoded once up front — these are
        small.
      - The main ``ohlcv_data`` array is streamed in ``_OHLCV_STREAM_BATCH``-row
        chunks. Each chunk is converted to list-of-lists, encoded, and yielded;
        the chunk bytes are released between yields.
    """
    yield _encode_meta_open(payload_meta)

    first = True
    for start in range(0, len(ohlcv_data), _OHLCV_STREAM_BATCH):
        batch = ohlcv_data[start : start + _OHLCV_STREAM_BATCH]
        rows = [_row_to_list(r) for r in batch]
        body = orjson.dumps(rows)  # b'[[...], [...], ...]'
        # Strip outer brackets so we can splice into the open array.
        inner = body[1:-1]
        if first:
            yield inner
            first = False
        else:
            yield b"," + inner

    yield b"]}"


def _build_payload_meta(
    *,
    yaml_content: str,
    run_config: BacktestTriggerRequest,
    market_data_request: dict[str, Any],
    backtest_ref_id: str | None,
    backtest_id: str | None,
    strategy_id: str | None,
    reference_ohlcv: list[dict[str, Any]] | None,
    htf_ohlcv: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Build the non-OHLCV part of the request body.

    Auxiliary OHLCV (reference + HTF) is small enough relative to the main
    series that we leave it inline in the envelope — encoding it streamingly
    is not worth the complexity. The main 1M+ row series is the only one
    that fits the streaming code path.
    """
    meta: dict[str, Any] = {
        "yaml_content": yaml_content,
        "run_config": run_config.model_dump(),
        "market_data_request": market_data_request,
    }
    if backtest_id is not None:
        meta["backtest_id"] = backtest_id
    if strategy_id is not None:
        meta["strategy_id"] = strategy_id
    if backtest_ref_id is not None:
        meta["backtest_ref_id"] = backtest_ref_id
    if reference_ohlcv is not None:
        meta["reference_ohlcv"] = [_row_to_list(r) for r in reference_ohlcv]
    if htf_ohlcv:
        meta["htf_ohlcv"] = {
            tf: [_row_to_list(r) for r in rows]
            for tf, rows in htf_ohlcv.items()
        }
    return meta


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
    payload_meta = _build_payload_meta(
        yaml_content=yaml_content,
        run_config=run_config,
        market_data_request=market_data_request,
        backtest_ref_id=None,
        backtest_id=backtest_id,
        strategy_id=strategy_id,
        reference_ohlcv=reference_ohlcv,
        htf_ohlcv=htf_ohlcv,
    )

    headers = {"Content-Type": "application/json"}
    logger.info(
        "📡 Quant engine POST /run | backtest_id=%s strategy_id=%s candles=%d aux_ref=%d htf=%s",
        backtest_id,
        strategy_id,
        len(ohlcv_data),
        len(reference_ohlcv) if reference_ohlcv else 0,
        list(htf_ohlcv.keys()) if htf_ohlcv else [],
    )
    try:
        async with httpx.AsyncClient(timeout=settings.quant_engine_timeout_seconds) as client:
            response = await client.post(
                f"{settings.quant_engine_url}/run",
                content=_stream_run_payload(payload_meta, ohlcv_data),
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.exception("❌ Quant engine rejected queued backtest | http_status=%s backtest_id=%s", exc.response.status_code, backtest_id)
        raise RuntimeError(
            f"Quant engine rejected the backtest request with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("📡❌ Failed to queue backtest with quant engine | backtest_id=%s", backtest_id)
        raise RuntimeError("Failed to reach the quant engine service.") from exc

    logger.info("✅ Quant engine accepted /run | backtest_id=%s http_status=%s", backtest_id, response.status_code)


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
    payload_meta = _build_payload_meta(
        yaml_content=yaml_content,
        run_config=run_config,
        market_data_request=market_data_request,
        backtest_ref_id=backtest_ref_id,
        backtest_id=None,
        strategy_id=None,
        reference_ohlcv=reference_ohlcv,
        htf_ohlcv=htf_ohlcv,
    )

    headers = {"Content-Type": "application/json"}
    logger.info(
        "📡 Quant engine POST /run-sync | candles=%d aux_ref=%d htf=%s engine_ref=%s",
        len(ohlcv_data),
        len(reference_ohlcv) if reference_ohlcv else 0,
        list(htf_ohlcv.keys()) if htf_ohlcv else [],
        backtest_ref_id,
    )
    try:
        async with httpx.AsyncClient(timeout=settings.quant_engine_timeout_seconds) as client:
            response = await client.post(
                f"{settings.quant_engine_url}/run-sync",
                content=_stream_run_payload(payload_meta, ohlcv_data),
                headers=headers,
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
        logger.exception("❌ Quant engine synchronous run failed | http_status=%s", exc.response.status_code)
        raise RuntimeError(
            f"Quant engine synchronous backtest failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("📡❌ Failed to reach quant engine for synchronous backtest")
        raise RuntimeError("Failed to reach the quant engine service.") from exc

    if not isinstance(result, dict) or "metrics" not in result:
        raise ValueError("Quant engine returned an unexpected backtest response payload.")

    logger.info(
        "✅ Quant engine /run-sync complete | total_trades=%s engine_ref=%s",
        (result.get("metrics", {}) or {}).get("total_trades"),
        result.get("backtest_ref_id"),
    )
    return result
