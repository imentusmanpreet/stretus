"""
tests/test_planner/test_readback.py — Phase 5 read-back + match% + versioning tests.

Covers:
  - build_readback() returns ReadBack with correct Built/Assumed/Couldn't-do
  - ReadBack.format() produces non-empty human-readable text
  - compute_match_pct() counts correctly at atomic-clause granularity
  - match% = built / (built + missed); defaults/inferred excluded
  - is_stale() and mark_stale_reason()
  - edit_sdl_field() bumps version, sets source=user, recomputes hash
  - Acceptance #4: same SDL → same match% (deterministic)
  - Acceptance #14: edit loop updates version, marks backtest stale
"""
import pytest

from app.planner.sdl import (
    SDL,
    ClarificationNeeded,
    DynamicRank,
    DynamicUniverse,
    EntrySpec,
    ExitSpec,
    GatesSpec,
    Leg,
    Provenance,
    RiskSpec,
    SignalRef,
    StaticUniverse,
    StopLossSpec,
    StrategyContext,
    TakeProfitSpec,
    UnmappedDetail,
)
from app.planner.compiler import compile_sdl
from app.planner.readback import (
    ReadBack,
    build_readback,
    compute_match_pct,
    edit_sdl_field,
    is_stale,
    mark_stale_reason,
)


# ── SDL fixtures ──────────────────────────────────────────────────────────────

def _eth_sdl(
    field_sources=None,
    unmapped=None,
    clarifications=None,
) -> SDL:
    return SDL(
        context=StrategyContext(market="crypto", timeframe="15m", objective="mean_reversion"),
        universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
        legs=[
            Leg(
                direction="long",
                entry=EntrySpec(
                    trigger=SignalRef(name="rsi_oversold", params={"window": 14, "threshold": 30})
                ),
                exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
            )
        ],
        risk=RiskSpec(
            stop_loss=StopLossSpec(type="percent", value=2.0),
            take_profit=TakeProfitSpec(type="rr", ratio=2.0),
        ),
        provenance=Provenance(
            field_sources=field_sources or {
                "universe.symbol": "user",
                "legs.0.entry.trigger": "user",
                "risk.stop_loss": "user",
                "risk.take_profit": "user",
                "legs.0.exit": "inferred",
            },
            unmapped_details=unmapped or [],
            clarifications_needed=clarifications or [],
        ),
    )


# ── ReadBack construction ─────────────────────────────────────────────────────

class TestReadBackConstruction:
    def test_returns_readback(self):
        rb = build_readback(_eth_sdl())
        assert isinstance(rb, ReadBack)

    def test_built_count_equals_user_sources(self):
        sdl = _eth_sdl(field_sources={
            "universe.symbol":           "user",
            "legs.0.entry.trigger":      "user",
            "risk.stop_loss":            "user",
            "risk.take_profit":          "user",
            "legs.0.exit":               "inferred",   # excluded from built
            "context.timeframe":         "default",    # excluded from built
        })
        rb = build_readback(sdl)
        assert rb.built_count == 4

    def test_built_lines_nonempty_for_user_fields(self):
        rb = build_readback(_eth_sdl())
        assert len(rb.built_lines) > 0

    def test_assumed_lines_for_inferred_fields(self):
        sdl = _eth_sdl(field_sources={
            "universe.symbol": "user",
            "legs.0.exit":     "inferred",
        })
        rb = build_readback(sdl)
        assert len(rb.assumed_lines) > 0

    def test_clarifications_appear_in_assumed(self):
        sdl = _eth_sdl(
            field_sources={"universe.symbol": "user"},
            clarifications=[
                ClarificationNeeded(
                    field="risk.take_profit",
                    question="No TP given — use 2:1?",
                    assumed_value="2:1",
                )
            ],
        )
        rb = build_readback(sdl)
        assert any("2:1" in ln or "take_profit" in ln.lower() or "TP" in ln or "No TP" in ln for ln in rb.assumed_lines)

    def test_unmapped_details_in_couldnt_do(self):
        sdl = _eth_sdl(
            unmapped=[
                UnmappedDetail(text="top-20 stocks", kind="engine_capability_gap", note="scanner picks one"),
            ]
        )
        rb = build_readback(sdl)
        assert len(rb.couldnt_do_lines) == 1
        assert "top-20 stocks" in rb.couldnt_do_lines[0]

    def test_missed_count_equals_unmapped(self):
        sdl = _eth_sdl(
            unmapped=[
                UnmappedDetail(text="requirement A", kind="missing_card"),
                UnmappedDetail(text="requirement B", kind="missing_operator"),
            ]
        )
        rb = build_readback(sdl)
        assert rb.missed_count == 2

    def test_validation_errors_appear(self):
        from app.planner.sdl_validator import ValidationResult, ValidationError
        sdl = _eth_sdl()
        vr = ValidationResult(ok=False, errors=[
            ValidationError(field="universe.symbol", code="unknown_symbol", message="Symbol not found.")
        ])
        rb = build_readback(sdl, vr)
        assert len(rb.validation_errors) == 1
        assert "unknown_symbol" in rb.validation_errors[0] or "Symbol" in rb.validation_errors[0]

    def test_engine_gaps_appear(self):
        from app.planner.sdl_validator import ValidationResult
        sdl = _eth_sdl()
        vr = ValidationResult(ok=True, engine_gaps=["scale_outs + trailing not supported"])
        rb = build_readback(sdl, vr)
        assert len(rb.engine_gaps) == 1


# ── ReadBack format ───────────────────────────────────────────────────────────

class TestReadBackFormat:
    def test_format_nonempty(self):
        rb = build_readback(_eth_sdl())
        text = rb.format()
        assert text.strip()

    def test_format_contains_built_section(self):
        rb = build_readback(_eth_sdl())
        text = rb.format()
        assert "Built" in text

    def test_format_contains_match_pct(self):
        rb = build_readback(_eth_sdl())
        text = rb.format()
        assert "%" in text or "Captured" in text

    def test_format_100_pct_no_missed(self):
        rb = build_readback(_eth_sdl())
        assert rb.missed_count == 0
        text = rb.format()
        assert "100%" in text or "0 requirement" in text.lower() or "100" in text

    def test_format_shows_missed_count(self):
        sdl = _eth_sdl(
            unmapped=[
                UnmappedDetail(text="top-20", kind="engine_capability_gap"),
                UnmappedDetail(text="pairs trading", kind="unsupported_universe"),
            ]
        )
        rb = build_readback(sdl)
        text = rb.format()
        assert "2" in text or "Missed" in text or "miss" in text.lower()

    def test_couldnt_do_section(self):
        sdl = _eth_sdl(
            unmapped=[UnmappedDetail(text="pairs trading", kind="unsupported_universe")]
        )
        rb = build_readback(sdl)
        text = rb.format()
        assert "Couldn't do" in text or "couldn't" in text.lower() or "pairs" in text


# ── match% tests ──────────────────────────────────────────────────────────────

class TestMatchPct:
    def test_perfect_match_no_unmapped(self):
        sdl = _eth_sdl(field_sources={
            "universe.symbol": "user",
            "legs.0.entry.trigger": "user",
            "risk.stop_loss": "user",
        })
        built, missed, pct = compute_match_pct(sdl)
        assert built == 3
        assert missed == 0
        assert pct == pytest.approx(100.0)

    def test_partial_match(self):
        sdl = _eth_sdl(
            field_sources={"universe.symbol": "user", "risk.stop_loss": "user"},
            unmapped=[
                UnmappedDetail(text="trailing stop", kind="engine_capability_gap"),
            ],
        )
        built, missed, pct = compute_match_pct(sdl)
        assert built == 2
        assert missed == 1
        assert pct == pytest.approx(2 / 3 * 100)

    def test_all_missed(self):
        # Construct directly (fixture's `or {}` treats empty dict as falsy)
        sdl = SDL(
            context=StrategyContext(market="crypto", timeframe="15m", objective="scalping"),
            universe=StaticUniverse(asset_class="crypto_spot", symbol="ETH_USDC"),
            legs=[
                Leg(
                    direction="long",
                    entry=EntrySpec(trigger=SignalRef(name="rsi_oversold", params={})),
                    exit=ExitSpec(triggers=[SignalRef(name="rsi_overbought", params={})]),
                )
            ],
            risk=RiskSpec(stop_loss=StopLossSpec(type="percent", value=2.0)),
            provenance=Provenance(
                field_sources={},   # no user fields
                unmapped_details=[
                    UnmappedDetail(text="A", kind="missing_card"),
                    UnmappedDetail(text="B", kind="missing_card"),
                ],
            ),
        )
        built, missed, pct = compute_match_pct(sdl)
        assert built == 0
        assert missed == 2
        assert pct == pytest.approx(0.0)

    def test_defaults_do_not_count(self):
        sdl = _eth_sdl(field_sources={
            "universe.symbol": "user",
            "risk.take_profit": "default",    # should NOT count
            "legs.0.exit": "inferred",        # should NOT count
        })
        built, missed, pct = compute_match_pct(sdl)
        assert built == 1
        assert pct == pytest.approx(100.0)

    def test_deterministic_same_sdl_same_pct(self):
        sdl = _eth_sdl()
        result1 = compute_match_pct(sdl)
        result2 = compute_match_pct(sdl)
        assert result1 == result2


# ── Versioning tests ──────────────────────────────────────────────────────────

class TestVersioning:
    def _compiled_artifact(self, sdl):
        return compile_sdl(sdl)

    def test_is_stale_when_versions_differ(self):
        sdl = _eth_sdl()
        art = self._compiled_artifact(sdl)
        modified = sdl.bump_version()   # version 2
        assert is_stale(modified, art)

    def test_not_stale_when_same_version(self):
        sdl = _eth_sdl()
        art = self._compiled_artifact(sdl)
        assert not is_stale(sdl, art)

    def test_mark_stale_reason_returns_none_when_current(self):
        sdl = _eth_sdl()
        art = self._compiled_artifact(sdl)
        assert mark_stale_reason(sdl, art) is None

    def test_mark_stale_reason_returns_message(self):
        sdl = _eth_sdl()
        art = self._compiled_artifact(sdl)
        modified = sdl.bump_version()
        msg = mark_stale_reason(modified, art)
        assert msg is not None
        assert "out of date" in msg.lower() or "modified" in msg.lower()


# ── edit_sdl_field tests ──────────────────────────────────────────────────────

class TestEditSDLField:
    def test_version_incremented(self):
        sdl = _eth_sdl()
        assert sdl.version == 1
        edited = edit_sdl_field(sdl, "context.timeframe", "1h")
        assert edited.version == 2

    def test_parent_version_set(self):
        sdl = _eth_sdl()
        edited = edit_sdl_field(sdl, "context.timeframe", "1h")
        assert edited.parent_version == 1

    def test_field_updated(self):
        sdl = _eth_sdl()
        edited = edit_sdl_field(sdl, "context.timeframe", "1h")
        assert edited.context.timeframe == "1h"

    def test_edited_field_source_is_user(self):
        sdl = _eth_sdl()
        edited = edit_sdl_field(sdl, "context.timeframe", "1h")
        assert edited.provenance.field_sources.get("context.timeframe") == "user"

    def test_content_hash_changes_after_edit(self):
        sdl = _eth_sdl()
        h_before = sdl.content_hash
        edited = edit_sdl_field(sdl, "context.timeframe", "1h")
        assert edited.content_hash != h_before

    def test_content_hash_same_if_no_logic_change(self):
        # Editing a provenance note doesn't change hash (provenance excluded from hash)
        # We can test this by editing a field to the SAME value
        sdl = _eth_sdl()
        h_before = sdl.content_hash
        # Edit to the same value (timeframe stays 15m)
        edited = edit_sdl_field(sdl, "context.timeframe", "15m")
        assert edited.content_hash == h_before

    def test_edit_marks_prior_backtest_stale(self):
        sdl = _eth_sdl()
        art = compile_sdl(sdl)
        edited = edit_sdl_field(sdl, "context.timeframe", "1h")
        assert is_stale(edited, art)

    def test_edit_nested_risk_field(self):
        sdl = _eth_sdl()
        edited = edit_sdl_field(sdl, "risk.stop_loss.value", 3.0)
        assert edited.risk.stop_loss is not None
        assert edited.risk.stop_loss.value == pytest.approx(3.0)
