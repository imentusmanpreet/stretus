"""Arbitrary-timeframe support: any interval from 1m up to 1d is accepted
(range-based validation), because the backtest fetches 1-minute data and the
engine resamples it up. These cover the DB-free pieces: validation, the dynamic
lookback/holding helpers, the KB normaliser, and that resampling actually works
for non-preset timeframes.
"""
import pandas as pd
import pytest

from app.services.strategy.builder import resolve_supported_user_timeframe

# market_data imports the gRPC client (optional dep). When grpc isn't installed
# locally, skip the helper/guard tests that need it — they run in CI.
try:
    import grpc  # noqa: F401
    _HAS_GRPC = True
except ImportError:
    _HAS_GRPC = False

_needs_market_data = pytest.mark.skipif(
    not _HAS_GRPC, reason="grpc not installed (market_data import) — runs in CI"
)


class TestRangeValidation:
    @pytest.mark.parametrize("tf", ["1m", "2m", "5m", "7m", "15m", "45m", "1h", "4h", "90m", "1d"])
    def test_in_range_accepted(self, tf):
        resolved, err = resolve_supported_user_timeframe(tf)
        assert err is None
        assert resolved is not None

    @pytest.mark.parametrize("tf", ["30s", "0m", "1w", "3d", "2d", "abc", ""])
    def test_out_of_range_or_malformed_rejected(self, tf):
        resolved, err = resolve_supported_user_timeframe(tf)
        assert resolved is None
        assert err is not None

    @pytest.mark.parametrize("raw,expected", [
        ("60 minute", "1h"),
        ("2 hours", "2h"),
        ("120m", "2h"),
        ("1440m", "1d"),
        ("7m", "7m"),
    ])
    def test_canonicalisation(self, raw, expected):
        resolved, err = resolve_supported_user_timeframe(raw)
        assert err is None
        assert resolved == expected


class TestKbNormalise:
    def test_range_based_normalise(self):
        from app.kb.schemas import TimeframeConfig
        tc = TimeframeConfig(supported=["1m", "5m", "15m", "1h", "1d"], synonyms={"60min": "1h"})
        assert tc.normalize("7m") == "7m"        # not a preset, but in range
        assert tc.normalize("45m") == "45m"
        assert tc.normalize("4h") == "4h"
        assert tc.normalize("60min") == "1h"     # synonym still wins
        assert tc.normalize("1w") is None        # out of range
        assert tc.normalize("30s") is None


@_needs_market_data
class TestDynamicLookbackHelpers:
    def test_bars_per_calendar_day(self):
        from app.services.backtest.market_data import _bars_per_calendar_day
        assert _bars_per_calendar_day("5m") == pytest.approx(75.0)     # 375/5
        assert _bars_per_calendar_day("1h") == pytest.approx(6.25)     # 375/60
        assert _bars_per_calendar_day("45m") == pytest.approx(375 / 45)  # arbitrary
        assert _bars_per_calendar_day("4h") == pytest.approx(375 / 240)
        assert _bars_per_calendar_day("1d") == 1.0

    def test_interval_buffer_days_monotonic(self):
        from app.services.backtest.market_data import _interval_buffer_days
        assert _interval_buffer_days("2m") == 7
        assert _interval_buffer_days("15m") == 14
        assert _interval_buffer_days("45m") == 14
        assert _interval_buffer_days("4h") == 21
        assert _interval_buffer_days("1d") == 90


@_needs_market_data
class TestIntrabarGuard:
    def test_non_native_interval_forces_intrabar(self):
        from app.services.backtest.market_data import resolve_intrabar_execution
        # Explicitly disabling intrabar on a non-native interval must be overridden
        # (the provider can't serve "7m"; it has to be resampled from 1m).
        assert resolve_intrabar_execution({"intrabar_execution": False}, signal_interval="7m") is True

    def test_native_interval_honours_explicit_flag(self):
        from app.services.backtest.market_data import resolve_intrabar_execution
        # 5m IS provider-native, so an explicit opt-out is honoured.
        assert resolve_intrabar_execution({"intrabar_execution": False}, signal_interval="5m") is False

    def test_auto_mode_on_for_coarse(self):
        from app.services.backtest.market_data import resolve_intrabar_execution
        assert resolve_intrabar_execution(None, signal_interval="45m") is True
        assert resolve_intrabar_execution(None, signal_interval="1m") is False


class TestResampleArbitrary:
    def _minute_frame(self, n=600):
        idx = pd.date_range("2024-01-01 09:15", periods=n, freq="1min")
        return pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0},
            index=idx,
        )

    @pytest.mark.parametrize("tf,expected_minutes", [("2m", 2), ("7m", 7), ("45m", 45), ("4h", 240)])
    def test_resample_produces_correct_bar_width(self, tf, expected_minutes):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "quant_engine"))
        from engine.resample import resample_ohlcv

        df1m = self._minute_frame()
        out = resample_ohlcv(df1m, tf)
        assert len(out) >= 1
        if len(out) >= 2:
            delta = (out.index[1] - out.index[0]).total_seconds() / 60
            assert delta == expected_minutes
        # OHLC aggregation sanity
        assert (out["high"] >= out["low"]).all()
