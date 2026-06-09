from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.backtest.backtest_window import (
    BacktestWindowError,
    extend_fetch_start,
    resolve_backtest_window,
)


def test_resolve_defaults_when_no_dates():
    from_utc, to_utc = resolve_backtest_window(None, None)
    assert from_utc.startswith("2024-01-01")
    assert to_utc.endswith("Z")


def test_resolve_user_window():
    from_utc, to_utc = resolve_backtest_window(
        "2024-01-01T00:00:00Z",
        "2024-02-01T23:59:59Z",
    )
    assert from_utc == "2024-01-01T00:00:00Z"
    assert to_utc == "2024-02-01T23:59:59Z"


def test_only_from_uses_default_to(monkeypatch):
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.default_backtest_window",
        lambda: ("2024-01-01T00:00:00Z", "2099-12-31T23:59:59Z"),
    )
    from_utc, to_utc = resolve_backtest_window("2024-06-01T00:00:00Z", None)
    assert from_utc == "2024-06-01T00:00:00Z"
    end_dt = datetime.fromisoformat(to_utc.replace("Z", "+00:00"))
    assert end_dt <= datetime.now(timezone.utc)


def test_only_to_uses_default_from(monkeypatch):
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.default_backtest_window",
        lambda: ("2024-01-01T00:00:00Z", "2026-06-15T12:00:00Z"),
    )
    from_utc, to_utc = resolve_backtest_window(None, "2025-09-15T23:59:59Z")
    assert from_utc == "2024-01-01T00:00:00Z"
    assert to_utc == "2025-09-15T23:59:59Z"


def test_rejects_inverted_range():
    with pytest.raises(BacktestWindowError, match="earlier than the end"):
        resolve_backtest_window(
            "2024-06-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        )


def test_rejects_before_earliest_floor(monkeypatch):
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.settings.backtest_earliest_from_utc",
        "2024-01-01T00:00:00Z",
    )
    with pytest.raises(BacktestWindowError, match="2024"):
        resolve_backtest_window(
            "2020-01-01T00:00:00Z",
            "2020-06-01T00:00:00Z",
        )


def test_clamps_future_end_to_now(monkeypatch):
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.settings.backtest_earliest_from_utc",
        "2024-01-01T00:00:00Z",
    )
    future_end = "2099-12-31T23:59:59Z"
    from_utc, to_utc = resolve_backtest_window(
        "2024-06-01T00:00:00Z",
        future_end,
    )
    end_dt = datetime.fromisoformat(to_utc.replace("Z", "+00:00"))
    assert end_dt <= datetime.now(timezone.utc)


def test_extend_fetch_start_skips_padding_for_default_window():
    extended = extend_fetch_start(
        "2024-06-01T00:00:00Z",
        interval="1d",
        indicators={"SMA": [20]},
        user_specified_window=False,
    )
    assert extended == "2024-06-01T00:00:00Z"


def test_extend_fetch_start_moves_earlier_for_user_range():
    extended = extend_fetch_start(
        "2025-08-01T00:00:00Z",
        interval="1m",
        indicators={"EMA": [20]},
        user_specified_window=True,
    )
    assert extended < "2025-08-01T00:00:00Z"
    assert extended >= "2024-01-01T00:00:00Z"


def test_user_range_padding_capped_not_full_history(monkeypatch):
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.settings.backtest_earliest_from_utc",
        "2024-01-01T00:00:00Z",
    )
    monkeypatch.setattr(
        "app.services.backtest.backtest_window.settings.backtest_user_range_max_padding_days",
        30,
    )
    extended = extend_fetch_start(
        "2025-08-01T00:00:00Z",
        interval="1m",
        indicators={},
        user_specified_window=True,
    )
    assert extended >= "2025-07-02T00:00:00Z"
    assert extended != "2024-01-01T00:00:00Z"
