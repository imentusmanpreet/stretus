"""Phase 9n — surface infrastructure errors to the user.

The user's screenshot revealed the real problem: the no-match reply
said "Scanned 19 stocks (19 excluded due to data fetch errors)" but
the user had no idea WHICH error caused every fetch to fail. The
fix has three parts:

  1. _fetch_and_evaluate captures the exception message and returns
     it in a structured dict (instead of None which loses info).

  2. scan_universe aggregates one example per unique error message
     into ScanResult.fetch_errors so the chat layer can quote it.

  3. The no-match builder branches on the diagnostic snapshot:
       • EVERY fetch failed → emit the fetch-failure reply
         (different message, different suggestions — "retry,
         contact ops, pin a stock manually" instead of "loosen
         your thresholds").
       • SOME fetches failed → surface an example error inline in
         the diagnostic line so the user can spot a partial outage.

These tests pin all three layers.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

from app.services.chat.strategy_flow import build_discovery_no_match_reply
from app.services.discovery.scanner import scan_universe
from app.services.discovery.types import DiscoveryConfig, TieBreakOption


# ── Total fetch failure → different reply, no threshold hints ──────────────


def test_total_fetch_failure_emits_infrastructure_reply():
    """When every fetch failed, threshold-relaxation hints are useless.
    Reply must be focused on the infrastructure problem instead."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T00:00:00Z",
            "scanned_count": 19,
            "failed_fetches": 19,
            "fail_counts": {},
            "fetch_errors": {
                "HDFCBANK.NS": "HTTPError: 503 Service Unavailable",
            },
        },
    )
    assert "infrastructure issue" in msg.lower()
    # Should NOT mention threshold tuning when no fetch succeeded.
    assert "Lower the volume threshold" not in msg
    assert "Loosen the conditions" not in msg
    # MUST quote the example error so the user can act on it.
    assert "503 Service Unavailable" in msg
    assert "HDFCBANK.NS" in msg


def test_total_fetch_failure_includes_asof_date():
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike"}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T15:30:00Z",
            "scanned_count": 19,
            "failed_fetches": 19,
            "fetch_errors": {"TCS.NS": "TimeoutError: timed out after 10s"},
        },
    )
    assert "2026-05-13" in msg


def test_total_fetch_failure_offers_retry_and_manual_pin_as_actions():
    """The hints should reflect what the user can actually DO about
    an infrastructure outage."""
    msg = build_discovery_no_match_reply(
        scan_diagnostics={
            "asof_iso": "2026-05-13T00:00:00Z",
            "scanned_count": 19,
            "failed_fetches": 19,
            "fetch_errors": {"TCS.NS": "ConnectionError: refused"},
        },
    )
    assert "retry" in msg.lower() or "Wait a minute" in msg
    assert "Pin a specific stock" in msg


# ── Partial fetch failure → inline example error in diagnostic line ────────


def test_partial_fetch_failure_surfaces_example_inline():
    """SOME fetches failed but some succeeded → still emit the normal
    no-match reply, but the diagnostic line should quote an example
    error so the user knows there's an issue."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T00:00:00Z",
            "scanned_count": 19,
            "failed_fetches": 5,        # 5 of 19 failed
            "fail_counts": {"VOL > AVG(VOL, 20) * 1.2": 14},
            "fetch_errors": {"TCS.NS": "HTTPError: 429 Too Many Requests"},
        },
    )
    # Normal no-match path — threshold suggestions still useful here.
    assert "Lower the volume threshold" in msg
    # And the diagnostic line surfaces the example. The transparent
    # renderer phrases this as "Couldn't fetch data" when per-symbol
    # lists are present, else "N stocks excluded ..." as fallback.
    assert ("Couldn't fetch data" in msg) or ("5 stocks excluded" in msg)
    assert "429 Too Many Requests" in msg


def test_partial_fetch_failure_without_examples_uses_terse_diagnostic():
    """If somehow fetch_errors is empty (legacy data, race), fall
    back to the pre-9n diagnostic line."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T00:00:00Z",
            "scanned_count": 19,
            "failed_fetches": 5,
            "fail_counts": {"VOL > AVG(VOL, 20) * 1.2": 14},
            "fetch_errors": {},
        },
    )
    assert "5 stocks excluded due to data fetch errors" in msg


# ── Scanner aggregates fetch errors across the universe ────────────────────


@pytest.mark.asyncio
async def test_scan_universe_aggregates_fetch_errors_by_unique_message():
    """One example per unique error string. If three symbols all
    fail with the same exception, the result has one entry — not
    three duplicates."""
    discovery = DiscoveryConfig(
        enabled=True,
        universe="kb_default",
        conditions=["VOL > AVG(VOL, 20) * 2"],
        tie_break_options=[TieBreakOption(method="highest_relative_volume",
                                          label="Highest relative volume")],
        lookback_days=280,
        scan_timeframe="1d",
    )

    call_count = {"n": 0}
    async def failing_fetcher(symbol, interval, from_utc, to_utc):
        call_count["n"] += 1
        if "HDFC" in symbol:
            raise ConnectionError("network unreachable")
        # Different error for the rest — surface BOTH in fetch_errors.
        raise RuntimeError("auth token expired")

    result = await scan_universe(discovery, interval="5m", fetch_ohlcv=failing_fetcher)
    assert result.status == "none"
    assert result.failed_fetches == result.scanned_count
    # Two unique errors → two entries in fetch_errors.
    error_values = set(result.fetch_errors.values())
    assert any("network unreachable" in v for v in error_values)
    assert any("auth token expired" in v for v in error_values)
    # And each unique message has at most ONE entry (deduplicated).
    assert len(error_values) <= 2


@pytest.mark.asyncio
async def test_scan_universe_preserves_fetch_errors_when_mixed_success():
    """Mixed outcomes: some symbols fetch successfully, others fail.
    The fetch_errors dict must still capture the failures even when
    some candidates passed."""
    import pandas as pd

    discovery = DiscoveryConfig(
        enabled=True, universe="kb_default",
        conditions=["VOL > AVG(VOL, 20) * 0.0"],  # always-true → all pass
        tie_break_options=[TieBreakOption(method="highest_relative_volume",
                                          label="Highest relative volume")],
        lookback_days=280, scan_timeframe="1d",
    )

    def _flat_records(n=300):
        base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
        return [
            {
                "timestamp": (base + pd.Timedelta(minutes=i))
                    .isoformat().replace("+00:00", "Z"),
                "open": 100.0, "high": 100.5, "low": 99.5,
                "close": 100.0, "volume": 100_000.0,
            }
            for i in range(n)
        ]

    failures_seen = []
    async def partly_failing_fetcher(symbol, interval, from_utc, to_utc):
        if "HDFC" in symbol or "TCS" in symbol:
            failures_seen.append(symbol)
            raise TimeoutError("read timeout")
        return _flat_records()

    result = await scan_universe(
        discovery, interval="5m", fetch_ohlcv=partly_failing_fetcher,
    )
    # Some passed (the always-true condition + flat records → all
    # qualifying symbols pass).
    assert result.status in {"single", "multiple"}
    # And the fetch errors are recorded.
    assert any("read timeout" in v for v in result.fetch_errors.values())
    assert result.failed_fetches == len(failures_seen)


# ── ScanResult schema invariants ───────────────────────────────────────────


def test_scan_result_includes_fetch_errors_field():
    from app.services.discovery.types import ScanResult
    sr = ScanResult(status="none", candidates=[], tie_break_options=[])
    assert hasattr(sr, "fetch_errors")
    assert sr.fetch_errors == {}


def test_scan_result_dict_serialises_fetch_errors():
    from app.services.discovery.types import ScanResult
    sr = ScanResult(
        status="none", candidates=[], tie_break_options=[],
        fetch_errors={"TCS.NS": "HTTPError: 503"},
    )
    out = sr.to_dict()
    assert out["fetch_errors"] == {"TCS.NS": "HTTPError: 503"}


# ── Backward compat: pre-9n callers still work ────────────────────────────


def test_diagnostics_without_fetch_errors_key_does_not_raise():
    """Older callers that don't pass fetch_errors must still get a
    valid reply."""
    msg = build_discovery_no_match_reply(
        conditions_used=[{"name": "volume_spike", "params": {"multiplier": 1.2}}],
        scan_diagnostics={
            "asof_iso": "2026-05-13T00:00:00Z",
            "scanned_count": 19,
            "failed_fetches": 0,
            "fail_counts": {},
            # no fetch_errors key
        },
    )
    assert "No stocks matched" in msg


def test_total_fetch_failure_detector_handles_partial_diagnostics():
    """The total-failure detector must not crash on a half-populated
    diagnostic dict (e.g. only scanned_count present)."""
    msg = build_discovery_no_match_reply(
        scan_diagnostics={"scanned_count": 0},
    )
    # Falls through to the normal no-match reply since scanned_count
    # is 0 (we can't tell if it's a "no universe" or "no fetches"
    # situation).
    assert "No stocks matched" in msg
