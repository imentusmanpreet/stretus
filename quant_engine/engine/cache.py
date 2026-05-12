"""
quant_engine/engine/cache.py
════════════════════════════
Simple file-based cache for market data using parquet files.
Stores OHLCV data locally to avoid repeated yfinance API calls.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Cache directory - stores parquet files
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache expiration time (in hours)
CACHE_EXPIRY_HOURS = 24


def _generate_cache_key(symbol: str, market: str, timeframe: str, start_date: str, end_date: str) -> str:
    """Generate a unique cache key from the request parameters."""
    key_string = f"{symbol}_{market}_{timeframe}_{start_date}_{end_date}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get the file path for a cache key."""
    return CACHE_DIR / f"{cache_key}.parquet"


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    
    # Check if cache is expired
    file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age_hours = (datetime.now() - file_mtime).total_seconds() / 3600
    
    if age_hours > CACHE_EXPIRY_HOURS:
        logger.info(f"Cache expired for {cache_path.name} (age: {age_hours:.1f} hours)")
        return False
    
    return True


def get_cached_data(symbol: str, market: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Try to get cached data if it exists and is valid.
    
    Returns:
        DataFrame if cache hit, None if cache miss or invalid.
    """
    cache_key = _generate_cache_key(symbol, market, timeframe, start_date, end_date)
    cache_path = _get_cache_path(cache_key)
    
    if _is_cache_valid(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"✅ Cache HIT for {symbol} | {timeframe} | {start_date} to {end_date}")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {cache_key}: {e}")
    
    logger.info(f"❌ Cache MISS for {symbol} | {timeframe} | {start_date} to {end_date}")
    return None


def save_to_cache(symbol: str, market: str, timeframe: str, start_date: str, end_date: str, df: pd.DataFrame) -> None:
    """
    Save downloaded data to cache.
    
    Args:
        symbol: Stock/crypto symbol
        market: Market type (us_stocks, crypto, etc.)
        timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
        start_date: Start date string
        end_date: End date string
        df: DataFrame with OHLCV data
    """
    cache_key = _generate_cache_key(symbol, market, timeframe, start_date, end_date)
    cache_path = _get_cache_path(cache_key)
    
    try:
        df.to_parquet(cache_path, index=True)
        logger.info(f"💾 Cached {len(df)} rows for {symbol} | {timeframe}")
    except Exception as e:
        logger.warning(f"Failed to save cache for {cache_key}: {e}")


def clear_cache() -> int:
    """Clear all cached files. Returns count of files deleted."""
    count = 0
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob("*.parquet"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}")
    logger.info(f"Cleared {count} cache files")
    return count


def get_cache_size() -> int:
    """Get total size of cache directory in bytes."""
    total = 0
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob("*.parquet"):
            total += file.stat().st_size
    return total
