"""
Timing and profiling utilities for backtest pipeline.

Provides context managers and decorators to measure execution time
of each step in the backtest flow.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


@contextmanager
def time_step(
    step_name: str,
    backtest_id: str | None = None,
    extra_context: dict[str, Any] | None = None,
    log_level: int = logging.INFO,
):
    """
    Context manager to time a specific step in the backtest pipeline.
    
    Usage:
        with time_step("fetch_ohlcv", backtest_id="abc-123", extra_context={"symbol": "RELIANCE"}):
            # ... your code ...
    
    Logs:
        - Start time with ISO timestamp
        - End time with ISO timestamp
        - Duration in human-readable format
    """
    start_time = time.perf_counter()
    start_dt = _utcnow()
    
    context_str = ""
    if backtest_id:
        context_str += f"|backtest_id={backtest_id}"
    if extra_context:
        context_str += "|" + "|".join(f"{k}={v}" for k, v in extra_context.items())
    
    logger.log(
        log_level,
        "⏱️  TIMING|step=%s|status=START|start_time=%s%s",
        step_name,
        start_dt.isoformat(),
        context_str,
    )
    
    try:
        yield
        
        end_time = time.perf_counter()
        end_dt = _utcnow()
        duration = end_time - start_time
        
        logger.log(
            log_level,
            "⏱️  TIMING|step=%s|status=COMPLETE|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f%s",
            step_name,
            start_dt.isoformat(),
            end_dt.isoformat(),
            _format_duration(duration),
            duration,
            context_str,
        )
        
    except Exception as exc:
        end_time = time.perf_counter()
        end_dt = _utcnow()
        duration = end_time - start_time
        
        logger.log(
            logging.ERROR,
            "⏱️  TIMING|step=%s|status=FAILED|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f|error=%s%s",
            step_name,
            start_dt.isoformat(),
            end_dt.isoformat(),
            _format_duration(duration),
            duration,
            type(exc).__name__,
            context_str,
        )
        raise


def timed_function(step_name: str | None = None):
    """
    Decorator to time a function execution.
    
    Usage:
        @timed_function("load_strategy_yaml")
        async def load_strategy(yaml_path: str):
            # ... your code ...
    """
    def decorator(func: Callable) -> Callable:
        name = step_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            start_dt = _utcnow()
            
            logger.info(
                "⏱️  TIMING|function=%s|status=START|start_time=%s",
                name,
                start_dt.isoformat(),
            )
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = time.perf_counter()
                end_dt = _utcnow()
                duration = end_time - start_time
                
                logger.info(
                    "⏱️  TIMING|function=%s|status=COMPLETE|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f",
                    name,
                    start_dt.isoformat(),
                    end_dt.isoformat(),
                    _format_duration(duration),
                    duration,
                )
                
                return result
                
            except Exception as exc:
                end_time = time.perf_counter()
                end_dt = _utcnow()
                duration = end_time - start_time
                
                logger.error(
                    "⏱️  TIMING|function=%s|status=FAILED|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f|error=%s",
                    name,
                    start_dt.isoformat(),
                    end_dt.isoformat(),
                    _format_duration(duration),
                    duration,
                    type(exc).__name__,
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            start_dt = _utcnow()
            
            logger.info(
                "⏱️  TIMING|function=%s|status=START|start_time=%s",
                name,
                start_dt.isoformat(),
            )
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.perf_counter()
                end_dt = _utcnow()
                duration = end_time - start_time
                
                logger.info(
                    "⏱️  TIMING|function=%s|status=COMPLETE|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f",
                    name,
                    start_dt.isoformat(),
                    end_dt.isoformat(),
                    _format_duration(duration),
                    duration,
                )
                
                return result
                
            except Exception as exc:
                end_time = time.perf_counter()
                end_dt = _utcnow()
                duration = end_time - start_time
                
                logger.error(
                    "⏱️  TIMING|function=%s|status=FAILED|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f|error=%s",
                    name,
                    start_dt.isoformat(),
                    end_dt.isoformat(),
                    _format_duration(duration),
                    duration,
                    type(exc).__name__,
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class BacktestTimer:
    """
    Accumulator for tracking multiple timing steps in a backtest.
    Useful for generating a summary report at the end.
    """
    
    def __init__(self, backtest_id: str):
        self.backtest_id = backtest_id
        self.steps: list[dict[str, Any]] = []
        self.overall_start = time.perf_counter()
        self.overall_start_dt = _utcnow()
    
    @contextmanager
    def step(self, step_name: str, extra_context: dict[str, Any] | None = None):
        """Time a step and add it to the accumulator."""
        start_time = time.perf_counter()
        start_dt = _utcnow()
        
        context_str = f"|backtest_id={self.backtest_id}"
        if extra_context:
            context_str += "|" + "|".join(f"{k}={v}" for k, v in extra_context.items())
        
        logger.info(
            "⏱️  TIMING|step=%s|status=START|start_time=%s%s",
            step_name,
            start_dt.isoformat(),
            context_str,
        )
        
        try:
            yield
            
            end_time = time.perf_counter()
            end_dt = _utcnow()
            duration = end_time - start_time
            
            step_info = {
                "step": step_name,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "duration_seconds": duration,
                "duration_formatted": _format_duration(duration),
                "status": "COMPLETE",
                "context": extra_context or {},
            }
            self.steps.append(step_info)
            
            logger.info(
                "⏱️  TIMING|step=%s|status=COMPLETE|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f%s",
                step_name,
                start_dt.isoformat(),
                end_dt.isoformat(),
                _format_duration(duration),
                duration,
                context_str,
            )
            
        except Exception as exc:
            end_time = time.perf_counter()
            end_dt = _utcnow()
            duration = end_time - start_time
            
            step_info = {
                "step": step_name,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "duration_seconds": duration,
                "duration_formatted": _format_duration(duration),
                "status": "FAILED",
                "error": type(exc).__name__,
                "context": extra_context or {},
            }
            self.steps.append(step_info)
            
            logger.error(
                "⏱️  TIMING|step=%s|status=FAILED|start_time=%s|end_time=%s|duration=%s|duration_seconds=%.4f|error=%s%s",
                step_name,
                start_dt.isoformat(),
                end_dt.isoformat(),
                _format_duration(duration),
                duration,
                type(exc).__name__,
                context_str,
            )
            raise
    
    def summary(self) -> dict[str, Any]:
        """Generate a summary report of all timed steps."""
        overall_end = time.perf_counter()
        overall_end_dt = _utcnow()
        overall_duration = overall_end - self.overall_start
        
        total_step_duration = sum(s["duration_seconds"] for s in self.steps)
        overhead = overall_duration - total_step_duration
        
        summary = {
            "backtest_id": self.backtest_id,
            "overall_start_time": self.overall_start_dt.isoformat(),
            "overall_end_time": overall_end_dt.isoformat(),
            "overall_duration_seconds": overall_duration,
            "overall_duration_formatted": _format_duration(overall_duration),
            "steps": self.steps,
            "step_count": len(self.steps),
            "total_step_duration": total_step_duration,
            "overhead": overhead,
        }
        
        # Log comprehensive summary
        logger.info(
            "⏱️  TIMING|backtest_id=%s|status=SUMMARY|overall_duration=%s|step_count=%d|"
            "total_step_duration=%.4f|overhead=%.4f",
            self.backtest_id,
            _format_duration(overall_duration),
            len(self.steps),
            total_step_duration,
            overhead,
        )
        
        # Log detailed breakdown table
        logger.info("=" * 100)
        logger.info("⏱️  BACKTEST TIMING SUMMARY - %s", self.backtest_id)
        logger.info("=" * 100)
        logger.info("%-40s | %12s | %12s | %10s", "Step", "Duration", "% of Total", "Status")
        logger.info("-" * 100)
        
        for step in self.steps:
            percentage = (step["duration_seconds"] / overall_duration * 100) if overall_duration > 0 else 0
            logger.info(
                "%-40s | %12s | %11.2f%% | %10s",
                step["step"][:40],
                step["duration_formatted"],
                percentage,
                step["status"],
            )
        
        logger.info("-" * 100)
        logger.info("%-40s | %12s | %11.2f%% | %10s", 
                   "TOTAL (all steps)", 
                   _format_duration(total_step_duration),
                   (total_step_duration / overall_duration * 100) if overall_duration > 0 else 0,
                   "")
        logger.info("%-40s | %12s | %11.2f%% | %10s", 
                   "Overhead (context switching, etc.)", 
                   _format_duration(overhead),
                   (overhead / overall_duration * 100) if overhead > 0 else 0,
                   "")
        logger.info("=" * 100)
        logger.info("%-40s | %12s | %11s | %10s", 
                   "OVERALL DURATION", 
                   _format_duration(overall_duration),
                   "100.00%",
                   "")
        logger.info("=" * 100)
        
        return summary
