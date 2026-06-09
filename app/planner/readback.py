"""
app/planner/readback.py — Read-back, match%, and versioning utilities.

Read-back
─────────
Plain-English summary of what the SDL captured, assumed, and couldn't do.
Sections:
  Built    — user fields that were captured (field_sources=user + validated)
  Assumed  — defaulted/inferred fields with their clarification question
  Couldn't do — unmapped_details + validation errors + engine_gaps
  N of M requirements captured (X%)

match%
──────
Deterministic, no LLM.  Counting unit = one atomic user clause that maps
to one SDL field or one catalog signal:
  built   = user-sourced fields in field_sources (field_sources=user)
  missed  = entries in unmapped_details
  match   = built / (built + missed)    (defaults/inferred never count)

Versioning helpers
──────────────────
mark_backtest_stale() / is_stale() — compare artifact version to SDL version.

edit_sdl_field()    — structured field edit (no LLM), bumps version.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Read-back model ───────────────────────────────────────────────────────────

@dataclass
class ReadBack:
    """Plain-English summary of one SDL ticket."""
    built_lines: list[str] = field(default_factory=list)
    assumed_lines: list[str] = field(default_factory=list)
    couldnt_do_lines: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    engine_gaps: list[str] = field(default_factory=list)

    # match% components
    built_count: int = 0
    missed_count: int = 0

    @property
    def match_pct(self) -> float:
        total = self.built_count + self.missed_count
        return (self.built_count / total * 100) if total else 100.0

    def format(self) -> str:
        """Return a multi-line plain-English summary."""
        lines: list[str] = []

        if self.built_lines:
            lines.append("**Built** (captured from your prompt):")
            for ln in self.built_lines:
                lines.append(f"  • {ln}")
        else:
            lines.append("**Built**: (nothing captured from the prompt)")

        if self.assumed_lines:
            lines.append("")
            lines.append("**Assumed** (defaulted or inferred — please confirm):")
            for ln in self.assumed_lines:
                lines.append(f"  • {ln}")

        if self.couldnt_do_lines:
            lines.append("")
            lines.append("**Couldn't do** (not in catalog or not supported):")
            for ln in self.couldnt_do_lines:
                lines.append(f"  • {ln}")

        if self.engine_gaps:
            lines.append("")
            lines.append("**Engine limitations** (feature noted; backtest may differ):")
            for g in self.engine_gaps:
                lines.append(f"  • {g}")

        if self.validation_errors:
            lines.append("")
            lines.append("**Validation errors** (must fix before backtest):")
            for e in self.validation_errors:
                lines.append(f"  ✗ {e}")

        lines.append("")
        total = self.built_count + self.missed_count
        if total == 0:
            lines.append("**Captured: 100% (no scoreable requirements detected)**")
        else:
            lines.append(
                f"**Captured: {self.built_count} of {total} requirements "
                f"({self.match_pct:.0f}%)**"
            )
            if self.missed_count:
                lines.append(f"  Missed: {self.missed_count} requirement(s) in 'Couldn't do' list above.")

        return "\n".join(lines)


# ── Human-readable field name lookup ─────────────────────────────────────────

_FIELD_LABELS: dict[str, str] = {
    "context.timeframe":         "Timeframe",
    "context.market":            "Market",
    "context.objective":         "Objective",
    "universe.symbol":           "Symbol",
    "universe":                  "Universe rule",
    "universe.screen":           "Screen conditions",
    "universe.rank.by":          "Rank metric",
    "universe.tie_break":        "Tie-break method",
    "risk.stop_loss":            "Stop loss",
    "risk.take_profit":          "Take profit",
    "risk.trailing":             "Trailing stop",
    "risk.scale_outs":           "Scale-out targets",
    "risk.sizing":               "Position sizing",
    "gates.regime":              "Regime gate",
    "gates.volatility":          "Volatility gate",
    "gates.event":               "Event filter",
    "gates.session":             "Session window",
    "gates.relative_strength":   "Relative-strength gate",
    "gates.volume_ratio":        "Volume-ratio gate",
}


def _label(field_path: str) -> str:
    """Human-readable label for a dotted SDL path."""
    # Exact match first
    if field_path in _FIELD_LABELS:
        return _FIELD_LABELS[field_path]
    # Strip list indices: legs.0.entry.trigger.name → legs.entry.trigger.name
    import re
    stripped = re.sub(r"\.\d+", "", field_path)
    if stripped in _FIELD_LABELS:
        return _FIELD_LABELS[stripped]
    # Fallback: prettify the last segment
    last = field_path.rsplit(".", 1)[-1]
    return last.replace("_", " ").title()


# ── Build read-back from SDL + validation result ──────────────────────────────

def build_readback(sdl, validation_result=None) -> ReadBack:
    """Build a ReadBack from an SDL ticket and optional ValidationResult.

    Args:
        sdl: SDL instance.
        validation_result: result from validate_sdl() (optional).
    """
    logger.info(
        "📝 readback | BUILD | hash=%.8s | v%d",
        sdl.content_hash, sdl.version,
    )
    rb = ReadBack()
    prov = sdl.provenance

    # ── Built lines (user-sourced fields) ─────────────────────────────────────
    user_paths: list[str] = sorted(
        p for p, src in (prov.field_sources or {}).items() if src == "user"
    )
    for path in user_paths:
        value = _describe_field_value(sdl, path)
        label = _label(path)
        rb.built_lines.append(f"{label}: {value}" if value else label)
        rb.built_count += 1

    # ── Assumed lines (default/inferred + clarifications) ─────────────────────
    inferred_paths: list[str] = sorted(
        p for p, src in (prov.field_sources or {}).items() if src in ("default", "inferred")
    )
    for path in inferred_paths:
        value = _describe_field_value(sdl, path)
        label = _label(path)
        rb.assumed_lines.append(f"{label}: {value}" if value else label)

    for c in prov.clarifications_needed or []:
        question = c.question
        assumed = c.assumed_value
        line = question
        if assumed is not None:
            line = f"{question} (I assumed: {assumed})"
        # Only add if not already in assumed_lines to avoid duplication
        if line not in rb.assumed_lines:
            rb.assumed_lines.append(line)

    # ── Couldn't-do lines (unmapped_details) ──────────────────────────────────
    for detail in prov.unmapped_details or []:
        kind_label = {
            "missing_card":          "no catalog card",
            "missing_operator":      "unsupported operator",
            "missing_param":         "unsupported parameter",
            "engine_capability_gap": "engine limitation",
            "unsupported_universe":  "unsupported universe",
        }.get(detail.kind, detail.kind)
        line = f'"{detail.text}" [{kind_label}]'
        if detail.note:
            line += f" — {detail.note}"
        rb.couldnt_do_lines.append(line)
        rb.missed_count += 1

    # ── Validation errors ──────────────────────────────────────────────────────
    if validation_result:
        for err in validation_result.errors or []:
            rb.validation_errors.append(f"[{err.field}] {err.message}")
        for gap in validation_result.engine_gaps or []:
            rb.engine_gaps.append(gap)

    total = rb.built_count + rb.missed_count
    pct = int(rb.built_count / total * 100) if total else 100
    status = "✅" if not rb.validation_errors else "❌"
    logger.info(
        "📊 readback | MATCH %d/%d (%d%%) | built=%d | missed=%d | "
        "assumed=%d | gaps=%d | errors=%d %s",
        rb.built_count, total, pct,
        rb.built_count, rb.missed_count,
        len(rb.assumed_lines),
        len(rb.engine_gaps),
        len(rb.validation_errors),
        status,
    )
    return rb


def _describe_field_value(sdl, path: str) -> str:
    """Best-effort human-readable description of a field's value in the SDL."""
    try:
        # Walk the dotted path (with numeric index support)
        parts = path.split(".")
        obj: Any = sdl
        for part in parts:
            if isinstance(obj, list):
                try:
                    obj = obj[int(part)]
                except (ValueError, IndexError):
                    return ""
            elif isinstance(obj, dict):
                obj = obj.get(part, "")
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                return ""

        if obj is None or obj == "":
            return ""

        # Format known types
        if hasattr(obj, "model_dump"):
            d = obj.model_dump(mode="json", exclude_none=True)
            # Compact: pick the most important fields
            if "name" in d:
                params = d.get("params") or {}
                if params:
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    return f"{d['name']}({param_str})"
                return d["name"]
            if "symbol" in d:
                return d["symbol"]
            if "value" in d and "type" in d:
                return f"{d['value']}% ({d['type']})"
            if "ratio" in d:
                return f"{d['ratio']}:1 ({d.get('type', '')})"
            if "multiple" in d:
                return f"{d['multiple']}× ATR({d.get('window', 14)})"
            return str(d)

        if isinstance(obj, (list, tuple)):
            return ", ".join(str(x) for x in obj)

        return str(obj)
    except Exception:
        return ""


# ── match% calculation ────────────────────────────────────────────────────────

def compute_match_pct(sdl) -> tuple[int, int, float]:
    """Compute (built_count, missed_count, match_pct) from the SDL provenance.

    Counting unit: one atomic user clause = one user-sourced field OR one
    unmapped_detail entry.  Defaults/inferred fields never count for or against.

    Returns: (built, missed, pct)
    """
    prov = sdl.provenance
    built = sum(1 for src in (prov.field_sources or {}).values() if src == "user")
    missed = len(prov.unmapped_details or [])
    total = built + missed
    pct = (built / total * 100) if total else 100.0
    return built, missed, pct


# ── Versioning ────────────────────────────────────────────────────────────────

def is_stale(sdl, artifact) -> bool:
    """True if the artifact was compiled from an older SDL version."""
    return artifact.version != sdl.version


def mark_stale_reason(sdl, artifact) -> str | None:
    """Human-readable stale reason, or None if up to date."""
    if not is_stale(sdl, artifact):
        return None
    return (
        f"Backtest result is out of date — strategy was modified "
        f"(current version {sdl.version}, last backtest was version {artifact.version}). "
        "Re-run the backtest to get updated results."
    )


# ── Structured field edit (no LLM) ───────────────────────────────────────────

def edit_sdl_field(sdl, path: str, value: Any) -> "SDL":  # type: ignore[name-defined]
    """Apply a single field edit to an SDL, returning a new SDL with version++.

    path: dotted SDL field path (e.g. "risk.stop_loss.value")
    value: new value to set

    Changed field gets source="user" in provenance.
    """
    # We work through model_dump / reconstruction to stay immutable
    data = sdl.model_dump(mode="python")
    data.pop("content_hash", None)

    # Navigate and set the nested value
    parts = path.split(".")
    obj = data
    for i, part in enumerate(parts[:-1]):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid path segment '{part}' in '{path}'")
        elif isinstance(obj, dict):
            if part not in obj:
                raise ValueError(f"Path '{path}' not found in SDL (missing '{part}')")
            obj = obj[part]
        else:
            raise ValueError(f"Cannot traverse '{part}' in path '{path}'")

    last = parts[-1]
    if isinstance(obj, list):
        try:
            obj[int(last)] = value
        except (ValueError, IndexError):
            raise ValueError(f"Cannot set list index '{last}' in path '{path}'")
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        raise ValueError(f"Cannot set field '{last}' in path '{path}'")

    # Update provenance: mark changed field as user-sourced
    prov = data.get("provenance") or {}
    if "field_sources" not in prov:
        prov["field_sources"] = {}
    prov["field_sources"][path] = "user"
    data["provenance"] = prov

    # Bump version
    data["version"] = sdl.version + 1
    data["parent_version"] = sdl.version

    from app.planner.sdl import SDL
    return SDL(**data)
