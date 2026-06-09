"""
tests/test_planner/test_catalog_schema.py — Phase 1 catalog menu tests.

Snapshot-tests the CatalogMenu output from build_menu():
  - all 119 registered signal cards appear
  - required fields on every entry
  - scanner rank metrics and tie-break ids present
  - catalog_version is stable (same hash every call)
  - format_menu_for_llm() produces a non-empty string with the right structure
"""
import json

import pytest

from app.planner.catalog_schema import (
    CatalogMenu,
    SignalEntry,
    build_menu,
    format_menu_for_llm,
    invalidate_menu_cache,
)


@pytest.fixture(scope="module")
def menu() -> CatalogMenu:
    invalidate_menu_cache()
    return build_menu()


# ── Structure tests ───────────────────────────────────────────────────────────

class TestMenuStructure:
    def test_signal_count_matches_registry(self, menu):
        from app.kb import kb
        assert len(menu.signals) == len(kb.signals)

    def test_all_signals_have_name(self, menu):
        for s in menu.signals:
            assert s.name, f"Empty name in entry: {s}"

    def test_all_signals_have_description(self, menu):
        empty = [s.name for s in menu.signals if not s.description]
        assert not empty, f"Missing descriptions: {empty}"

    def test_all_signals_have_roles(self, menu):
        empty = [s.name for s in menu.signals if not s.roles]
        assert not empty, f"Missing roles: {empty}"

    def test_all_roles_are_valid(self, menu):
        valid = {"entry_trigger", "entry_filter", "exit_trigger", "exit_filter"}
        for s in menu.signals:
            bad = [r for r in s.roles if r not in valid]
            assert not bad, f"{s.name} has invalid roles: {bad}"

    def test_all_signals_have_direction(self, menu):
        valid = {"bullish", "bearish", "neutral"}
        for s in menu.signals:
            assert s.direction in valid, f"{s.name} direction={s.direction!r}"

    def test_all_signals_have_family(self, menu):
        for s in menu.signals:
            assert s.family, f"{s.name} has no family"

    def test_all_signals_have_timeframes(self, menu):
        empty = [s.name for s in menu.signals if not s.timeframes]
        # Some cards may have no works_best_on — warn but don't fail hard
        # (they still compile and run; they just have no preferred timeframe)
        # We do require that at least 90% of cards have timeframes.
        pct = (len(menu.signals) - len(empty)) / len(menu.signals)
        assert pct >= 0.9, f"Too many cards without timeframes: {empty}"

    def test_signal_names_are_unique(self, menu):
        names = [s.name for s in menu.signals]
        assert len(names) == len(set(names))

    def test_signal_names_match_kb(self, menu):
        from app.kb import kb
        menu_names = menu.signal_names
        kb_names = frozenset(kb.signals.keys())
        assert menu_names == kb_names

    def test_get_by_name(self, menu):
        entry = menu.get("rsi_oversold")
        assert entry is not None
        assert entry.name == "rsi_oversold"
        assert "entry_trigger" in entry.roles

    def test_get_unknown_returns_none(self, menu):
        assert menu.get("nonexistent_signal_xyz") is None


# ── Params tests ──────────────────────────────────────────────────────────────

class TestMenuParams:
    def test_rsi_oversold_has_window_and_threshold(self, menu):
        entry = menu.get("rsi_oversold")
        assert entry is not None
        assert "window" in entry.params
        assert "threshold" in entry.params
        assert entry.params["window"].default == 14
        assert entry.params["threshold"].default == 30.0

    def test_ema_above_has_fast_and_slow(self, menu):
        entry = menu.get("ema_above")
        assert entry is not None
        assert "window_fast" in entry.params
        assert "window_slow" in entry.params

    def test_param_types_are_valid_strings(self, menu):
        valid_types = {"int", "float", "str"}
        for s in menu.signals:
            for pname, pdesc in s.params.items():
                assert pdesc.type in valid_types, (
                    f"{s.name}.{pname}: type={pdesc.type!r}"
                )

    def test_param_defaults_match_type(self, menu):
        for s in menu.signals:
            for pname, pdesc in s.params.items():
                if pdesc.type == "int":
                    assert isinstance(pdesc.default, int), (
                        f"{s.name}.{pname}: expected int, got {type(pdesc.default)}"
                    )
                elif pdesc.type == "float":
                    assert isinstance(pdesc.default, (int, float)), (
                        f"{s.name}.{pname}: expected float, got {type(pdesc.default)}"
                    )


# ── Scanner meta tests ────────────────────────────────────────────────────────

class TestScannerMeta:
    def test_rank_metrics_present(self, menu):
        expected = {"rvol", "rsi", "atr_pct", "distance_52w"}
        assert expected.issubset(set(menu.scanner.rank_metrics))

    def test_tie_break_ids_present(self, menu):
        assert len(menu.scanner.tie_break_ids) >= 5
        assert "highest_relative_volume" in menu.scanner.tie_break_ids
        assert "closest_to_52w_high" in menu.scanner.tie_break_ids

    def test_screen_condition_notes_nonempty(self, menu):
        assert menu.scanner.screen_condition_notes


# ── Catalog version (stability / snapshot) ────────────────────────────────────

class TestCatalogVersion:
    def test_catalog_version_is_16_chars(self, menu):
        assert len(menu.catalog_version) == 16

    def test_catalog_version_stable_across_builds(self):
        invalidate_menu_cache()
        m1 = build_menu()
        invalidate_menu_cache()
        m2 = build_menu()
        assert m1.catalog_version == m2.catalog_version

    def test_catalog_version_is_hex(self, menu):
        assert all(c in "0123456789abcdef" for c in menu.catalog_version)


# ── LLM format tests ──────────────────────────────────────────────────────────

class TestLLMFormat:
    def test_format_is_nonempty(self, menu):
        text = format_menu_for_llm(menu)
        assert text

    def test_format_has_signal_catalog_header(self, menu):
        text = format_menu_for_llm(menu)
        assert "SIGNAL CATALOG" in text

    def test_format_has_scanner_section(self, menu):
        text = format_menu_for_llm(menu)
        assert "DYNAMIC UNIVERSE" in text or "DISCOVERY SCANNER" in text

    def test_format_contains_rsi_oversold(self, menu):
        text = format_menu_for_llm(menu)
        assert "rsi_oversold" in text

    def test_format_json_lines_parseable(self, menu):
        text = format_menu_for_llm(menu)
        parsed_count = 0
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                obj = json.loads(line)
                assert "name" in obj or "rank_by" in obj
                parsed_count += 1
        assert parsed_count >= len(menu.signals)

    def test_format_has_catalog_version(self, menu):
        text = format_menu_for_llm(menu)
        assert menu.catalog_version in text

    def test_format_token_budget(self, menu):
        text = format_menu_for_llm(menu)
        # Rough 4-char/token estimate; require < 10 000 tokens
        estimated_tokens = len(text) / 4
        assert estimated_tokens < 10_000, (
            f"Menu too large: ~{estimated_tokens:.0f} tokens"
        )
