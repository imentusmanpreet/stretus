"""app/planner/strategy_gate.py — the deterministic verification gate (design §5). Runs AFTER propose and BEFORE render, on create+modify turns; repairs silently, asks rarely, never errors. STRUCTURE-ONLY (never reads prose); never silently flips intent or swaps a concept. Pure/unit-testable; optional repropose callback enables bounded self-heal (§5b.6)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core import step_logger as _sl
from app.planner import engine_contract as _ec
from app.planner.sdl import SDL

# ── Output contracts (design §5c) ─────────────────────────────────────────────


@dataclass
class CoverageReport:
    """Honest, gate-computed coverage (design §5c). Buckets: captured/inferred/defaults={path,...}; repaired={path,from,to,reason}; unmapped={concept,note}."""

    captured: list[dict] = field(default_factory=list)
    inferred: list[dict] = field(default_factory=list)
    defaults: list[dict] = field(default_factory=list)
    repaired: list[dict] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)

    def fully_captured(self) -> bool:
        """True iff nothing was left unmapped (design §5c)."""
        return not self.unmapped

    def as_dict(self) -> dict:
        return {
            "captured": self.captured,
            "inferred": self.inferred,
            "defaults": self.defaults,
            "repaired": self.repaired,
            "unmapped": self.unmapped,
        }


@dataclass
class GateResult:
    sdl: SDL | None                       # the repaired, engine-safe SDL (None on hard block)
    coverage: CoverageReport
    clarification: dict | None = None     # one pre-filled question, or None
    blocked: str | None = None            # rare, human-phrased "closest I can do"
    notes: list[str] = field(default_factory=list)   # non-blocking "ℹ️ adjusted" chips

    @property
    def ok(self) -> bool:
        return self.blocked is None and self.clarification is None


@dataclass
class SymbolContext:
    """The resolved-symbol truth the gate reconciles the SDL against (built by the caller from the resolved builder symbol)."""

    symbol: str | None = None
    asset_class: str | None = None        # equity_cash | crypto_spot
    market: str | None = None             # indian_stocks | crypto | …
    alias_of: str | None = None           # the raw token this symbol resolved from

    @classmethod
    def from_builder(cls, builder: Any) -> "SymbolContext":
        # asset_class/market are the symbol-DERIVED truth, so they are only
        # authoritative when a symbol is actually resolved. With no symbol the
        # builder still returns equity defaults (equity_cash / indian_stocks) —
        # treating those as "resolved" would wrongly overwrite the proposer's
        # asset guess (e.g. flip a crypto strategy to equity). So leave them None.
        symbol = getattr(builder, "symbol", None)
        if not symbol:
            return cls(symbol=None, asset_class=None, market=None)
        return cls(
            symbol=symbol,
            asset_class=getattr(builder, "asset_class", None),
            market=getattr(builder, "market", None),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entry_signals(leg) -> list:
    """All entry signal refs of a leg (trigger + filters)."""
    out = []
    if leg.entry and leg.entry.trigger:
        out.append(leg.entry.trigger)
    for f in (leg.entry.filters if leg.entry else []) or []:
        out.append(f)
    return out


def _card(name: str):
    from app.kb import kb
    return kb.signals.get(name)


def _opposite_direction(d: str) -> str:
    return "short" if d == "long" else "long"


# ── The gate ───────────────────────────────────────────────────────────────────

def verify(
    sdl: SDL,
    *,
    symbol_ctx: SymbolContext,
    catalog: Any,
    prior_provenance: dict | None = None,
    sid: str | None = None,
    turn: int = 1,
    repropose: Callable[[str], SDL] | None = None,
) -> GateResult:
    """Verify + repair an SDL (see module docstring). Works on a deep copy; returns the repaired SDL + coverage + optional clarification/block."""
    work = copy.deepcopy(sdl)
    cov = CoverageReport()
    notes: list[str] = []
    clarification: dict | None = None

    sid = sid or (getattr(work, "content_hash", "") or "")

    # 1) Market / symbol reconciliation — symbol is the source of truth (§5).
    _reconcile_market(work, symbol_ctx, cov, notes, sid, turn)

    # 2) Alias resolution — wire the resolved symbol onto the SDL universe.
    _reconcile_symbol(work, symbol_ctx, cov, notes, sid, turn)

    # 2b) Prune params the card doesn't define — cosmetic, silent (§5d). The LLM
    # sometimes stuffs sibling-signal params (e.g. window_fast/window_slow onto a
    # price_above_ema that only takes window); the validator rejects those as
    # unknown_param and the whole strategy fails to build. Dropping unused params
    # changes no meaning (the formula never reads them).
    _prune_unknown_params(work, cov, sid, turn)

    # 3) Engine-contract normalisation — SL/TP must be engine-legal or clarify.
    clarification = _normalize_engine_contract(work, cov, notes, sid, turn) or clarification

    # 4) Contradiction + direction coherence (intent-proof repair, else heal/ask).
    healed = False
    contra = _check_contradictions(work, cov, notes, sid, turn)
    if contra and repropose is not None:
        # §5b.6 — one bounded re-proposal pass with the contradiction described.
        _sl.log_step(_sl.SELF_HEAL, "verify", sid, turn, "self_heal.reproposed", attempt=1)
        try:
            reproposed = repropose(contra["reason"])
            if isinstance(reproposed, SDL):
                work = copy.deepcopy(reproposed)
                healed = True
                # Re-run the structural reconciliations on the healed proposal.
                _reconcile_market(work, symbol_ctx, cov, notes, sid, turn)
                _reconcile_symbol(work, symbol_ctx, cov, notes, sid, turn)
                clarification = _normalize_engine_contract(work, cov, notes, sid, turn) or clarification
                contra = _check_contradictions(work, cov, notes, sid, turn)
        except Exception:
            contra = contra  # heal failed → fall through to clarify
    if contra and clarification is None:
        clarification = contra.get("clarification") or {
            "field": contra.get("field", "legs"),
            "question": contra["question"],
            "assumed_value": contra.get("assumed_value"),
        }
        _sl.log_step(_sl.CLARIFY, "verify", sid, turn, "contradiction.clarify", reason=contra.get("reason", ""))

    # 5) Satisfiability — does the entry rule ever fire? (§5b.6 self-heal hook.)
    if clarification is None:
        sat_clar = _check_satisfiable(work, cov, notes, sid, turn)
        clarification = sat_clar or clarification

    # 6) Input sanity — risk%/drawdown/SL/TP plausibility (reframe-with-note/ask).
    sanity_clar = _check_input_sanity(work, cov, notes, sid, turn)
    if clarification is None:
        clarification = sanity_clar

    # 7) Unmapped honesty — never substitute; surface every unmapped concept.
    _record_unmapped(work, cov, notes, sid, turn)

    # 8) Provenance honesty — recompute captured/inferred/defaults from the SDL.
    _recompute_provenance_coverage(work, cov)

    # Fold any gate clarification into the SDL so the existing clarifications UI
    # surfaces it alongside the (still-shown) in-progress strategy.
    if clarification is not None:
        _append_clarification(work, clarification)

    _sl.log_step(
        _sl.PASS if clarification is None else _sl.CLARIFY,
        "verify", sid, turn, "gate_pass" if clarification is None else "gate_clarify",
        repaired=len(cov.repaired), inferred=len(cov.inferred),
        defaults=len(cov.defaults), unmapped=len(cov.unmapped),
        healed=healed,
    )

    return GateResult(sdl=work, coverage=cov, clarification=clarification, blocked=None, notes=notes)


# ── Check 1 — market / asset reconciliation ────────────────────────────────────

def _reconcile_market(work: SDL, ctx: SymbolContext, cov, notes, sid, turn) -> None:
    if not ctx.asset_class:
        return
    uni = work.universe
    cur = getattr(uni, "asset_class", None)
    if cur and cur != ctx.asset_class:
        cov.repaired.append({
            "path": "universe.asset_class", "from": cur, "to": ctx.asset_class,
            "reason": f"derived from {ctx.symbol or 'symbol'}",
        })
        notes.append(f"Set asset class to {ctx.asset_class} (from {ctx.symbol}).")
        _sl.log_step(_sl.REPAIR, "verify", sid, turn, "repair.market",
                     **{"from": cur, "to": ctx.asset_class, "reason": "from_symbol"})
        # Universe is a pydantic model; reassign the field.
        try:
            uni.asset_class = ctx.asset_class
        except Exception:
            pass
    # context.market follows the symbol too.
    if ctx.market and work.context is not None and getattr(work.context, "market", None) != ctx.market:
        old = getattr(work.context, "market", None)
        try:
            work.context.market = ctx.market
            cov.repaired.append({"path": "context.market", "from": old, "to": ctx.market,
                                 "reason": f"derived from {ctx.symbol or 'symbol'}"})
        except Exception:
            pass


# ── Check 2 — symbol / alias ───────────────────────────────────────────────────

def _reconcile_symbol(work: SDL, ctx: SymbolContext, cov, notes, sid, turn) -> None:
    if not ctx.symbol:
        return
    uni = work.universe
    if getattr(uni, "type", None) != "static":
        return
    cur = getattr(uni, "symbol", None)
    if cur != ctx.symbol:
        try:
            uni.symbol = ctx.symbol
        except Exception:
            return
        cov.repaired.append({"path": "universe.symbol", "from": cur, "to": ctx.symbol,
                             "reason": "resolved_symbol"})
        _sl.log_step(_sl.REPAIR, "verify", sid, turn, "repair.symbol",
                     **{"from": cur or "?", "to": ctx.symbol})


# ── Check 3 — engine contract (SL/TP) ──────────────────────────────────────────

def _normalize_engine_contract(work: SDL, cov, notes, sid, turn) -> dict | None:
    # Force SL/TP into the engine's accepted vocabulary, or clarify. Guarantees
    # the planner can never emit an anchor the engine rejects: an unmappable stop
    # (vwap_deviation/points/unknown) is rewritten to a percent stop WITH a
    # visible clarification — never silently turned into ORB.
    risk = work.risk
    if risk is None:
        return None
    clarification: dict | None = None

    sl = risk.stop_loss
    if sl is not None:
        direction = work.legs[0].direction if work.legs else None
        mapped = _ec.map_stop_loss_type(sl.type, direction)
        if mapped == _ec.NEEDS_CLARIFY:
            # Don't let it reach the engine; substitute a percent stop, BUT make
            # it a visible clarification (never a silent concept swap).
            old_type = sl.type
            sl.type = "percent"
            sl.value = sl.value or 2.0
            sl.anchor = None
            cov.unmapped.append({"concept": f"{old_type} stop",
                                 "note": "engine has no such stop; using 2% — confirm or change"})
            clarification = {
                "field": "risk.stop_loss",
                "question": (f"I can't place a '{old_type}' stop on this engine — "
                             f"I've set a 2% stop for now. Keep 2%, or use ATR/structural?"),
                "assumed_value": "2%",
            }
            _sl.log_step(_sl.CLARIFY, "verify", sid, turn, "engine_contract.sl_unmappable",
                         **{"from": old_type, "to": "percent"})
        elif isinstance(mapped, dict) and mapped.get("type") == "structural":
            anchor = mapped["anchor"]
            if sl.anchor != anchor:
                cov.repaired.append({"path": "risk.stop_loss.anchor", "from": sl.anchor or sl.type,
                                     "to": anchor, "reason": "engine_anchor"})
                _sl.log_step(_sl.REPAIR, "verify", sid, turn, "repair.sl_anchor",
                             **{"from": sl.type, "to": anchor})
            sl.anchor = anchor
            # keep sl.type as the SDL structural type; compiler reads .anchor.

    tp = risk.take_profit
    if tp is not None:
        mapped_tp = _ec.map_take_profit_type(tp.type)
        if mapped_tp == _ec.NEEDS_CLARIFY:
            old_type = tp.type
            tp.type = "rr"
            tp.ratio = tp.ratio or 2.0
            cov.unmapped.append({"concept": f"{old_type} target",
                                 "note": "engine has no such target; using 2:1 RR — confirm"})
            if clarification is None:
                clarification = {
                    "field": "risk.take_profit",
                    "question": (f"I can't use a '{old_type}' target here — I've set a 2:1 "
                                 f"reward:risk for now. Keep it, or set a % target?"),
                    "assumed_value": "2:1",
                }
            _sl.log_step(_sl.CLARIFY, "verify", sid, turn, "engine_contract.tp_unmappable",
                         **{"from": old_type, "to": "rr"})
    return clarification


# ── Check 4 — contradictions + direction coherence ─────────────────────────────

def _check_contradictions(work: SDL, cov, notes, sid, turn) -> dict | None:
    # Enforce card `contradicts` + direction coherence on entry signals. Returns
    # a contradiction descriptor (for self-heal / clarify) or None. A mis-mapped
    # signal is silently re-mapped ONLY with intent-stub proof (§5b.5).
    for li, leg in enumerate(work.legs or []):
        signals = _entry_signals(leg)
        names = [s.name for s in signals]

        # (a) explicit `contradicts` pairs among the entry signals.
        for i, s in enumerate(signals):
            card = _card(s.name)
            if card is None:
                continue
            for other in card.contradicts or []:
                if other in names and other != s.name:
                    repaired = _try_intent_repair(leg, s, cov, notes, sid, turn)
                    if repaired:
                        return None  # repaired; no further action this pass
                    return {
                        "field": f"legs.{li}.entry",
                        "reason": f"{s.name} ∧ {other} contradict",
                        "question": (f"'{s.name}' and '{other}' can't both be entry conditions — "
                                     f"which one should trigger the entry?"),
                        "assumed_value": s.name,
                    }

        # (b) direction coherence: an entry TRIGGER whose card direction opposes
        # the leg direction is a likely mapping error.
        trig = leg.entry.trigger if leg.entry else None
        if trig is not None:
            card = _card(trig.name)
            if card is not None:
                cdir = str(getattr(card, "direction", "neutral") or "neutral")
                if cdir in ("bullish", "bearish"):
                    want = "bullish" if leg.direction == "long" else "bearish"
                    if cdir != want:
                        repaired = _try_intent_repair(leg, trig, cov, notes, sid, turn)
                        if repaired:
                            return None
                        return {
                            "field": f"legs.{li}.entry.trigger",
                            "reason": f"{trig.name} is {cdir} but leg is {leg.direction}",
                            "question": (f"Your entry signal '{trig.name}' is {cdir}, but the trade "
                                         f"is {leg.direction}. Did you mean the {want} version?"),
                            "assumed_value": trig.name,
                        }
    return None


def _try_intent_repair(leg, ref, cov, notes, sid, turn) -> bool:
    # Silently re-map a mis-mapped signal ONLY when the proposer's intent stub
    # proves the intended comparison resolves to a DIFFERENT existing card
    # (§5b.5). Returns True if repaired; without proof → no repair (caller asks).
    intent = getattr(ref, "intent", None)
    if intent is None or intent.intended is None:
        return False
    target = _card_for_intent(intent.intended, want_direction=("bullish" if leg.direction == "long" else "bearish"))
    if target and target != ref.name and _card(target) is not None:
        old = ref.name
        ref.name = target
        cov.repaired.append({"path": "signal", "from": old, "to": target,
                             "reason": f"intent rhs={intent.intended.rhs} op={intent.intended.op}"})
        notes.append(f"Mapped '{old}' → '{target}' from your stated comparison.")
        _sl.log_step(_sl.REPAIR, "verify", sid, turn, "repair.signal_map",
                     **{"from": old, "to": target, "reason": f"intent_rhs={intent.intended.rhs}"})
        return True
    return False


def _card_for_intent(intended, *, want_direction: str) -> str | None:
    # Best catalog card matching a structured intent comparison, in the wanted
    # direction. Prefers cards' explicit comparison{lhs,rhs,op} metadata (WS3);
    # falls back to a light name heuristic for cards not yet backfilled.
    from app.kb import kb
    lhs = (intended.lhs or "").lower()
    rhs = (intended.rhs or "").lower()
    op = (intended.op or "").lower()

    # (a) Data-driven: a card whose declared comparison matches the intent (in
    # either orientation) and the wanted direction.
    if lhs and rhs and op:
        for name, card in kb.signals.items():
            comp = getattr(card, "comparison", None)
            if comp is None or str(card.direction) != want_direction:
                continue
            if _comparison_matches(comp, lhs, rhs, op):
                return name

    # (b) Heuristic fallback: "<price/close> below/above <indicator>".
    price_tokens = _PRICE_TOKENS()
    ind = rhs if lhs in price_tokens else (lhs if rhs in price_tokens else None)
    px_is_lhs = lhs in price_tokens
    if ind:
        below = (op == "lt" and px_is_lhs) or (op == "gt" and not px_is_lhs)
        prefix = "price_below_" if below else "price_above_"
        candidate = f"{prefix}{ind}"
        if candidate in kb.signals and str(kb.signals[candidate].direction) == want_direction:
            return candidate
        alt = f"{ind}_{'bearish' if below else 'bullish'}"
        if alt in kb.signals and str(kb.signals[alt].direction) == want_direction:
            return alt
    return None


# ── Check 5 — satisfiability ─────────────────────────────────────────────────

def _PRICE_TOKENS() -> set[str]:
    return {"price", "close", "candle"}


def _comparison_matches(comp, lhs: str, rhs: str, op: str) -> bool:
    # Does a card's declared comparison match an intent comparison? Matches in
    # either orientation, flipping the op when sides are swapped; treats
    # price/close/candle as one 'price' token.
    flip = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge",
            "cross_up": "cross_down", "cross_down": "cross_up", "eq": "eq"}
    px = _PRICE_TOKENS()

    def norm(x: str) -> str:
        return "price" if x in px else x

    clhs, crhs, cop = norm(str(comp.lhs)), norm(str(comp.rhs)), str(comp.op)
    ilhs, irhs = norm(lhs), norm(rhs)
    if clhs == ilhs and crhs == irhs and cop == op:
        return True
    if clhs == irhs and crhs == ilhs and cop == flip.get(op, op):
        return True
    return False


def _check_satisfiable(work: SDL, cov, notes, sid, turn) -> dict | None:
    # Compile the primary leg's entry rule and check it can ever fire. A rule
    # that can NEVER fire (e.g. RSI>90 AND RSI<10) is reframed as a clarification
    # — never a raw "0 trades" error reaching the user.
    if not work.legs:
        return None
    try:
        from app.kb import kb
        from app.planner.compiler import _build_entry_condition
        from app.planner.condition_satisfiability import check_entry_satisfiable
    except Exception:
        return None
    cond = _build_entry_condition(work.legs[0], kb)
    if not cond:
        return None
    finding = check_entry_satisfiable(cond)
    if finding is None:
        _sl.log_step(_sl.VERIFY, "verify", sid, turn, "satisfiable", ok=True)
        return None
    _sl.log_step(_sl.CLARIFY, "verify", sid, turn, "satisfiable", ok=False, code=finding.get("code"))
    return {
        "field": "entry_condition",
        "question": (
            "These entry conditions can never be true at the same time, so the "
            "strategy would take zero trades. Which condition should I keep?"
        ),
        "assumed_value": None,
    }


# ── Check 6 — input sanity ─────────────────────────────────────────────────────

def _check_input_sanity(work: SDL, cov, notes, sid, turn) -> dict | None:
    # Reframe/cap implausible numeric inputs with a note (or one clarify):
    # per-trade risk%, SL%/TP% positivity, impossible asks (100% risk / 0% dd).
    risk = work.risk
    if risk is None:
        return None
    RISK_CAP = 10.0  # % of capital per trade — anything above is a typo/impossible
    sizing = getattr(risk, "sizing", None)
    if sizing is not None and getattr(sizing, "risk_pct", None) is not None:
        rp = sizing.risk_pct
        if rp is not None and rp > RISK_CAP:
            old = rp
            sizing.risk_pct = RISK_CAP
            notes.append(f"Capped per-trade risk from {old:g}% to {RISK_CAP:g}% (safety limit).")
            cov.repaired.append({"path": "risk.sizing.risk_pct", "from": old, "to": RISK_CAP,
                                 "reason": "risk_cap"})
            _sl.log_step(_sl.REPAIR, "verify", sid, turn, "sanity.risk_capped",
                         **{"from": old, "to": RISK_CAP})
    # SL/TP must be positive percentages where used.
    sl = risk.stop_loss
    if sl is not None and sl.type == "percent" and (sl.value is None or sl.value <= 0):
        return {
            "field": "risk.stop_loss",
            "question": "A 0% stop loss can't work — what stop should I use (e.g. 2%)?",
            "assumed_value": "2%",
        }
    return None


# ── Check 7 — param hygiene + unmapped honesty ─────────────────────────────────

def _all_signals(work: SDL) -> list:
    """Every signal ref in the SDL (entry trigger+filters, exit triggers+filters)."""
    out = []
    for leg in (work.legs or []):
        if leg.entry and leg.entry.trigger:
            out.append(leg.entry.trigger)
        for f in (leg.entry.filters if leg.entry else []) or []:
            out.append(f)
        for t in (leg.exit.triggers if leg.exit else []) or []:
            out.append(t)
        for f in (leg.exit.filters if leg.exit else []) or []:
            out.append(f)
    return out


def _known_params(card) -> set[str]:
    """Param names the card defines — the SAME set the validator checks (union of params_by_timeframe keys)."""
    known: set[str] = set()
    for tf_params in (card.params_by_timeframe or {}).values():
        known.update(tf_params.keys())
    return known


def _prune_unknown_params(work: SDL, cov, sid, turn) -> None:
    # Drop params a card doesn't define (cosmetic, silent — §5d). The proposer
    # sometimes attaches sibling-signal params (e.g. window_fast on a
    # price_above_ema that only takes window); the formula never reads them but
    # the validator rejects them as unknown_param and the strategy fails to build.
    for ref in _all_signals(work):
        card = _card(ref.name)
        if card is None or not ref.params:
            continue
        known = _known_params(card)
        if not known:
            continue  # card declares no params → validator skips it too
        extra = [p for p in list(ref.params) if p not in known]
        for p in extra:
            ref.params.pop(p, None)
        if extra:
            cov.repaired.append({"path": f"signal:{ref.name}.params", "from": ",".join(extra),
                                 "to": "removed", "reason": "param_not_in_card"})
            _sl.log_step(_sl.REPAIR, "verify", sid, turn, "repair.prune_params",
                         signal=ref.name, dropped=",".join(extra))


def _record_unmapped(work: SDL, cov, notes, sid, turn) -> None:
    # Every unmapped concept the proposer flagged → an honest `unmapped` entry.
    # NEVER substitute (we only record here).
    for um in (work.provenance.unmapped_details or []):
        concept = getattr(um, "text", None) or ""
        already = any(e.get("concept") == concept for e in cov.unmapped)
        if concept and not already:
            cov.unmapped.append({"concept": concept, "note": getattr(um, "note", "") or "no catalog card"})
            _sl.log_step(_sl.CLARIFY, "verify", sid, turn, "unmapped", concept=concept[:40])


# ── Check 8 — provenance honesty ───────────────────────────────────────────────

def _recompute_provenance_coverage(work: SDL, cov: CoverageReport) -> None:
    # Fill captured/inferred/defaults from the reconciled field_sources — the
    # gate's own count, not the LLM's self-report (design §5c).
    fs = work.provenance.field_sources or {}
    for path, src in fs.items():
        entry = {"path": path}
        if src == "user":
            cov.captured.append(entry)
        elif src == "inferred":
            cov.inferred.append(entry)
        elif src == "default":
            cov.defaults.append(entry)


# ── Misc ────────────────────────────────────────────────────────────────────────

def _append_clarification(work: SDL, clarification: dict) -> None:
    # Add the gate's clarification to the SDL's clarifications_needed (dedup by
    # field) so the existing chat UI surfaces it.
    from app.planner.sdl import ClarificationNeeded
    field_id = clarification.get("field", "")
    existing = work.provenance.clarifications_needed or []
    if any(getattr(c, "field", None) == field_id for c in existing):
        return
    existing.append(ClarificationNeeded(
        field=field_id,
        question=clarification.get("question", ""),
        assumed_value=clarification.get("assumed_value"),
    ))
    work.provenance.clarifications_needed = existing
