"""Unit tests for RMS capture: the validator (LLM-primary path guardrail) and
the regex fallback (_extract_rms_from_text), focused on the formats that were
previously dropped — notably the "Label: value" colon form.

These cover the pure, DB-free pieces. The end-to-end chat wiring (LLM-primary
capture + commit-on-confirm) is exercised in the chat integration suite, which
requires the database driver and runs in CI.
"""
from app.services.strategy.builder import _extract_rms_from_text
from app.services.strategy.rms_validator import validate_rms


# The exact risk block from the reported failing prompt.
_USER_PROMPT = (
    "Stop Loss: 0.50% Take Profit: 1.00% Risk Reward: 1:2 "
    "Per Trade Risk: 1% Maximum 10 trades per day Daily Loss Cap: 4%"
)


def _value(rms: dict, key: str):
    entry = rms.get(key)
    return entry["value"] if isinstance(entry, dict) else entry


class TestRegexFallback:
    def test_full_prompt_captures_every_field(self):
        rms = _extract_rms_from_text(_USER_PROMPT)
        assert _value(rms, "stop_loss_pct") == 0.5
        # Regression: TP previously stole the stop-loss "0.50%" via leftmost match.
        assert _value(rms, "take_profit_pct") == 1.0
        assert _value(rms, "risk_reward") == 2.0
        # Regression: colon-form per-trade-risk / daily-cap were dropped.
        assert _value(rms, "per_trade_risk") == 1.0
        assert _value(rms, "daily_loss_cap") == 4.0
        assert _value(rms, "max_trades") == 10

    def test_per_trade_risk_colon_form(self):
        assert _value(_extract_rms_from_text("Per Trade Risk: 1%"), "per_trade_risk") == 1.0

    def test_daily_loss_cap_colon_form(self):
        assert _value(_extract_rms_from_text("Daily Loss Cap: 4%"), "daily_loss_cap") == 4.0

    def test_keyword_first_take_profit_wins(self):
        # "target 5% stop loss 1%" → TP=5, SL=1 (digit-first must not cross-assign).
        rms = _extract_rms_from_text("target 5% stop loss 1%")
        assert _value(rms, "take_profit_pct") == 5.0
        assert _value(rms, "stop_loss_pct") == 1.0

    def test_natural_language_still_works(self):
        rms = _extract_rms_from_text("risk 2% per trade and stop loss 1.5%")
        assert _value(rms, "per_trade_risk") == 2.0
        assert _value(rms, "stop_loss_pct") == 1.5


class TestValidator:
    def test_in_range_values_ok(self):
        assert validate_rms("daily_loss_cap", 4).status == "ok"
        assert validate_rms("per_trade_risk", 1).status == "ok"
        assert validate_rms("take_profit_pct", 1.0).value == 1.0
        assert validate_rms("max_trades", 10).value == 10

    def test_out_of_range_needs_confirmation(self):
        v = validate_rms("daily_loss_cap", 50)
        assert v.needs_confirmation
        assert v.value == 50.0  # value preserved so the caller can echo it back

    def test_absurd_max_trades_needs_confirmation(self):
        assert validate_rms("max_trades", 5000).needs_confirmation

    def test_non_positive_is_invalid(self):
        assert validate_rms("per_trade_risk", 0).status == "invalid"
        assert validate_rms("stop_loss_pct", -1).status == "invalid"

    def test_non_numeric_is_invalid(self):
        assert validate_rms("daily_loss_cap", "abc").status == "invalid"

    def test_coerces_string_and_percent_and_dict(self):
        assert validate_rms("per_trade_risk", "1%").value == 1.0
        assert validate_rms("stop_loss_pct", {"value": 0.5}).value == 0.5

    def test_unknown_field_is_invalid(self):
        assert validate_rms("not_a_field", 1).status == "invalid"
