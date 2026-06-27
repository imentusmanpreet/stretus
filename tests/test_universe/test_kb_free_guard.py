"""
KB-free guard (Invariant 11 / §16 / §18): no dynamic-path module may import the knowledge
base or planner packages. The dynamic universe's pool comes from the universe-source registry,
not ``kb.stocks``; screen conditions use the engine's (already KB-free) formula compiler.

This is a static-import lint over the new packages so a regression is caught in CI, not prod.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DYNAMIC_PATH_DIRS = [
    _ROOT / "app" / "services" / "universe",
    _ROOT / "app" / "services" / "data",
]
_FORBIDDEN_PREFIXES = ("app.kb", "stretus_knowledge_base", "app.planner")


def _module_files() -> list[Path]:
    files: list[Path] = []
    for d in _DYNAMIC_PATH_DIRS:
        files.extend(sorted(d.glob("*.py")))
    return files


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_no_kb_or_planner_imports(path: Path):
    offending = {
        mod for mod in _imported_modules(path)
        if any(mod == p or mod.startswith(p + ".") for p in _FORBIDDEN_PREFIXES)
    }
    assert not offending, f"{path.name} imports forbidden KB/planner modules: {sorted(offending)}"


def test_guard_actually_scanned_files():
    # Guard against the test silently passing because it found no files.
    assert _module_files(), "no dynamic-path modules found to scan"
