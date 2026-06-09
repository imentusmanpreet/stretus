from __future__ import annotations

from app.services.chat.inputs_snapshot import build_inputs_snapshot_from_draft
from app.services.strategy.builder import StrategyBuilder


def test_none_placeholders_do_not_count_as_core_inputs():
    builder = StrategyBuilder()
    builder.symbol = "HINDALCO.NS"
    builder.timeframe = "1m"
    builder.sentiment = "bullish"
    builder.objective = "none"
    builder.experience = "none"
    builder.goal = "None"

    assert builder.objective is None
    assert builder.experience is None
    assert builder.goal is None
    assert builder.is_user_input_complete() is False
    assert builder.missing_user_input_fields() == ["objective", "experience", "goal"]


def test_build_inputs_snapshot_from_draft_skips_none_placeholders():
    draft = {
        "mode": "collect_user_input",
        "symbol": "HINDALCO.NS",
        "timeframe": "1m",
        "objective": "none",
        "sentiment": "bullish",
        "experience": "None",
        "goal": "n/a",
    }

    snapshot = build_inputs_snapshot_from_draft(
        draft,
        missing=["objective", "experience", "goal"],
    )
    labels = [row["label"] for row in snapshot["summary_rows"]]

    assert labels == ["Stock", "Timeframe", "Market view"]
    assert snapshot["details"]["missing_fields"] == ["objective", "experience", "goal"]
