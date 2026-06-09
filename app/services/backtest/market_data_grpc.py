"""
Direct gRPC client for MarketDataService (marketdata-ingestion, port 50057).

Bypasses User-Gateway → BFF → HTTP when streutus-ai runs in the same K8s cluster.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import grpc

from app.core.config import Settings, get_settings
from app.proto.gen import marketdata_pb2
from app.proto.gen.marketdata_pb2_grpc import MarketDataServiceStub

logger = logging.getLogger(__name__)

# Intervals documented on GetOHLCVRequest in marketdata.proto.
GRPC_OHLCV_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d"})


def use_grpc_transport(settings: Settings | None = None) -> bool:
    """Return True when backtest OHLCV should be fetched via gRPC."""
    cfg = settings or get_settings()
    transport = (cfg.market_data_fetch_transport or "auto").strip().lower()
    target = (cfg.market_data_grpc_target or "").strip()
    if transport == "http":
        return False
    if transport == "grpc":
        if not target:
            raise ValueError(
                "MARKET_DATA_FETCH_TRANSPORT=grpc requires MARKET_DATA_GRPC_TARGET "
                "(e.g. marketdata-ingestion.stretus-ai-dev2.svc.cluster.local:50057)."
            )
        return True
    # auto: prefer in-cluster gRPC when target is configured
    return bool(target)


def validate_grpc_interval(interval: str) -> str:
    normalized = str(interval or "").strip()
    if normalized not in GRPC_OHLCV_INTERVALS:
        supported = ", ".join(sorted(GRPC_OHLCV_INTERVALS))
        raise ValueError(
            f"Interval {normalized!r} is not supported by MarketDataService gRPC "
            f"(supported: {supported}). Use HTTP transport or a supported timeframe."
        )
    return normalized


def bars_to_record_dicts(bars: Any) -> list[dict[str, Any]]:
    """Convert protobuf OHLCVBar messages to dicts for normalize_ohlcv_payload."""
    records: list[dict[str, Any]] = []
    for bar in bars:
        records.append(
            {
                "ts": bar.ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
    return records


async def fetch_ohlcv_chunk_grpc(
    stub: MarketDataServiceStub,
    *,
    symbol: str,
    interval: str,
    from_utc: str,
    to_utc: str,
    chunk_idx: int,
    total_chunks: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Single GetOHLCV RPC for one date chunk. Returns raw dict rows (ts/open/...)."""
    request = marketdata_pb2.GetOHLCVRequest(
        symbol=symbol,
        interval=validate_grpc_interval(interval),
    )
    # Set 'from' field using setattr since it's a Python keyword
    setattr(request, 'from', from_utc)
    request.to = to_utc

    chunk_start = time.perf_counter()
    logger.info(
        "📡 gRPC GetOHLCV chunk %d/%d | symbol=%s interval=%s from=%s to=%s",
        chunk_idx,
        total_chunks,
        symbol,
        interval,
        from_utc,
        to_utc,
    )

    try:
        response = await stub.GetOHLCV(request, timeout=timeout_seconds)
        chunk_duration = time.perf_counter() - chunk_start
        logger.info(
            "⏱️  TIMING|step=grpc_fetch_chunk_%d|status=COMPLETE|duration=%.4fs|"
            "symbol=%s|interval=%s|bars=%d",
            chunk_idx,
            chunk_duration,
            symbol,
            interval,
            len(response.bars),
        )
        # Per-bar dict construction is sync CPU work; offload so we don't
        # block the event loop while gRPC keeps streaming the next chunk.
        return await asyncio.to_thread(bars_to_record_dicts, response.bars)
    except grpc.aio.AioRpcError as exc:
        chunk_duration = time.perf_counter() - chunk_start
        logger.exception(
            "⏱️  TIMING|step=grpc_fetch_chunk_%d|status=FAILED|duration=%.4fs|"
            "gRPC %s (chunk %d/%d): %s",
            chunk_idx,
            chunk_duration,
            exc.code(),
            chunk_idx,
            total_chunks,
            exc.details(),
        )
        raise RuntimeError(
            f"Market data gRPC rejected the request ({exc.code().name}): {exc.details()}"
        ) from exc


def open_grpc_channel(settings: Settings | None = None) -> grpc.aio.Channel:
    cfg = settings or get_settings()
    target = (cfg.market_data_grpc_target or "").strip()
    if not target:
        raise ValueError("MARKET_DATA_GRPC_TARGET is not configured.")

    if cfg.market_data_grpc_secure:
        return grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
    return grpc.aio.insecure_channel(target)
