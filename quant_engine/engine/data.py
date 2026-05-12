"""
Normalizes OHLCV payloads supplied by the API into a pandas DataFrame.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_timestamp(value: Any) -> datetime:
    if value is None:
        raise ValueError("OHLCV record is missing timestamp.")

    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, (int, float)):
        if value > 10_000_000_000:
            timestamp = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        else:
            timestamp = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("OHLCV record has an empty timestamp.")
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OHLCV record has invalid {field_name} value: {value!r}") from exc


def _normalize_record(record: Any) -> dict[str, Any]:
    if isinstance(record, (list, tuple)) and len(record) >= 6:
        source = {
            "timestamp": record[0],
            "open": record[1],
            "high": record[2],
            "low": record[3],
            "close": record[4],
            "volume": record[5],
        }
    elif isinstance(record, dict):
        lowered = {str(key).lower(): value for key, value in record.items()}
        source = {
            "timestamp": lowered.get("timestamp", lowered.get("datetime", lowered.get("time", lowered.get("date")))),
            "open": lowered.get("open", lowered.get("o")),
            "high": lowered.get("high", lowered.get("h")),
            "low": lowered.get("low", lowered.get("l")),
            "close": lowered.get("close", lowered.get("c")),
            "volume": lowered.get("volume", lowered.get("vol", lowered.get("v"))),
        }
    else:
        raise ValueError("OHLCV record must be a dict or a 6-item list.")

    normalized = {
        "timestamp": _normalize_timestamp(source["timestamp"]),
        "open": _coerce_float(source["open"], "open"),
        "high": _coerce_float(source["high"], "high"),
        "low": _coerce_float(source["low"], "low"),
        "close": _coerce_float(source["close"], "close"),
        "volume": _coerce_float(source["volume"], "volume"),
    }
    _validate_ohlcv_row(normalized)
    return normalized


def _validate_ohlcv_row(record: dict[str, Any]) -> None:
    open_ = float(record["open"])
    high = float(record["high"])
    low = float(record["low"])
    close = float(record["close"])
    volume = float(record["volume"])

    if min(open_, high, low, close) < 0:
        raise ValueError("OHLCV record contains negative price values.")
    if high < max(open_, close, low):
        raise ValueError("OHLCV record has an invalid high value.")
    if low > min(open_, close, high):
        raise ValueError("OHLCV record has an invalid low value.")
    if volume < 0:
        raise ValueError("OHLCV record contains negative volume.")


def load_ohlcv_data(ohlcv_data: list[dict[str, Any]] | list[list[Any]] | None) -> pd.DataFrame:
    if not ohlcv_data:
        raise ValueError("Quant engine requires OHLCV data to run a backtest.")

    records = [_normalize_record(record) for record in ohlcv_data]
    if not records:
        raise ValueError("No OHLCV records were supplied to the quant engine.")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be strictly increasing after normalization.")
    if df.index.has_duplicates:
        raise ValueError("OHLCV data still contains duplicate timestamps after normalization.")
    if df.empty:
        raise ValueError("No OHLCV rows were available after normalization.")

    logger.info(
        "Normalized OHLCV data | rows=%s start=%s end=%s",
        len(df),
        df.index[0].isoformat(),
        df.index[-1].isoformat(),
    )
    return df


def download_data(symbol: str, market: str, timeframe: str) -> pd.DataFrame:
    raise ValueError(
        "Quant engine no longer downloads market data directly. "
        "Supply OHLCV data from the market data server instead."
    )


# ─── Phase 4: reference symbol merge ──────────────────────────────────────────
#
# Strategies that compare a stock to a benchmark (Relative Strength vs Nifty,
# real-Nifty trend filter, etc.) need a SECOND OHLCV series joined onto the
# main df. The convention is: each OHLCV column gets a "REF_" prefix
# (REF_open, REF_high, …). The AST identifier REF_CLOSE then resolves to
# REF_close at the current bar.
#
# We re-index the reference to the main df's timestamps and forward-fill any
# small gaps (e.g. a holiday where the benchmark didn't print but the stock
# did — rare on NSE but possible on BSE-only or low-volume references).

_REF_PREFIX = "REF_"
REF_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def merge_reference_data(
    main_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    prefix: str = _REF_PREFIX,
) -> pd.DataFrame:
    """Join a reference OHLCV DataFrame onto `main_df` with `prefix`-ed columns.

    Both frames must already be normalised by `load_ohlcv_data`. The reference
    is re-indexed to the main df's timestamps with forward-fill; any leading
    gap (reference data starts later than main) leaves NaN in the prefixed
    columns and condition evaluation will treat that as False (no entry).

    Returns a new DataFrame; the inputs are not mutated.
    """
    if reference_df is None or reference_df.empty:
        raise ValueError("merge_reference_data: reference_df is empty.")
    if not isinstance(reference_df.index, pd.DatetimeIndex):
        raise ValueError("merge_reference_data: reference_df must have a DatetimeIndex.")
    if not isinstance(main_df.index, pd.DatetimeIndex):
        raise ValueError("merge_reference_data: main_df must have a DatetimeIndex.")

    available = [c for c in REF_OHLCV_COLUMNS if c in reference_df.columns]
    if not available:
        raise ValueError(
            f"merge_reference_data: reference_df is missing all OHLCV columns "
            f"(expected any of {REF_OHLCV_COLUMNS})."
        )

    # Align reference to main's timestamps. ffill propagates the last-known
    # reference bar across any tiny gap; bars before the reference's first
    # observation stay NaN.
    aligned = reference_df[available].reindex(main_df.index, method="ffill")
    aligned.columns = [f"{prefix}{c}" for c in aligned.columns]

    out = pd.concat([main_df, aligned], axis=1)
    return out
