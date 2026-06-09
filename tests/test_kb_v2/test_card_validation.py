"""
tests/test_kb_v2/test_card_validation.py — WS3 catalog integrity gate.

Guards the invariants that let the gate/compiler trust the cards as data:
  * every enabled card loads & validates (kb load would raise otherwise);
  * every cross-card reference (contradicts / pairs_well_with / mirrors_to)
    resolves to a real enabled card — catches the broken-ref class of bugs;
  * every card carries a ParamSpec for each param it uses (auto-derived);
  * every formula compiles in the engine DSL (skipped where talib is absent).
"""
import pytest

from app.kb import kb


def _names():
    return set(kb.signals)


def test_all_cards_loaded():
    assert len(kb.signals) >= 124


def test_all_cross_references_resolve():
    names = _names()
    dangling = []
    for n, c in kb.signals.items():
        for ref in (c.contradicts or []):
            if ref not in names:
                dangling.append(f"{n}.contradicts -> {ref}")
        for ref in (c.pairs_well_with or []):
            if ref not in names:
                dangling.append(f"{n}.pairs_well_with -> {ref}")
        mw = getattr(c, "match_when", None)
        if mw and mw.mirrors_to and mw.mirrors_to not in names:
            dangling.append(f"{n}.mirrors_to -> {mw.mirrors_to}")
    assert not dangling, "dangling card references:\n" + "\n".join(dangling)


def test_every_param_has_a_spec():
    missing = []
    for n, c in kb.signals.items():
        by = c.params_by_timeframe or {}
        defaults = by.get("15m") or (next(iter(by.values())) if by else {})
        for p in defaults:
            if p not in c.param_specs:
                missing.append(f"{n}.{p}")
    assert not missing, "params without a ParamSpec:\n" + "\n".join(missing)


def test_new_coverage_cards_present():
    names = _names()
    assert {"volume_above_average", "volume_dry_up"} <= names


def test_macd_cross_cards_use_signal_line_not_zero_line():
    for name in ("macd_bullish_cross", "macd_bearish_cross"):
        card = kb.signals[name]
        assert "MACD_SIGNAL" in (card.formula or ""), f"{name} must cross the signal line"
        assert card.comparison is not None and card.comparison.op in ("cross_up", "cross_down")


def test_formulas_compile_in_engine_dsl():
    try:
        from engine.conditions import compile_condition  # noqa: F401
    except Exception:  # talib/engine not installed in this env → CI covers it
        pytest.skip("engine DSL (talib) unavailable in this environment")
    from app.planner.compiler import _resolve_condition

    bad = []
    for n, c in kb.signals.items():
        if not c.formula:
            continue
        from app.planner.sdl import SignalRef
        rendered = _resolve_condition(SignalRef(name=n), kb)
        if not rendered:
            continue
        try:
            compile_condition(rendered)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{n}: {rendered!r} -> {exc}")
    assert not bad, "formulas that don't compile:\n" + "\n".join(bad)
