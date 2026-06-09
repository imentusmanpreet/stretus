"""
app/planner/pipeline.py — Orchestrator for the strategy planning flow.

Reads a populated StrategyBuilder, runs the deterministic pipeline:
  1. Resolve inputs against the KB
  2. Extract structured intent (single LLM call)
  3. Hard-filter candidate signals
  4. Soft-rank surviving candidates
  5. Resolve params (perf cache → OHLCV → defaults)
  6. Resolve SL/TP via realized ATR%
  7. Validate invariants (raise on violation)
  8. Emit decision trace

Returns a StrategyPlan. Convert to the existing StrategyConfig shape via
adapters.plan_to_strategy_config() before sending to /strategy/evaluate/execute.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.kb import kb as default_kb
from app.kb.loader import KB
from app.kb.schemas import (
    Intent,
    PickedSignal,
    SignalCard,
    SignalDirection,
    StrategyPlan,
)
from app.planner import filters as hard_filters
from app.planner import ranker as soft_ranker
from app.planner.intent_extractor import IntentExtractor
from app.planner.param_resolver import (
    build_risk_dict,
    resolve_params,
    resolve_sl_tp,
)
from app.planner.trace import DecisionTrace
from app.planner.validator import PlanInvariantViolation, validate_plan
from app.planner.semantic_signal_composer import SemanticSignalComposer

# Import regime classifier — it lives in the quant engine, which may not be
# installed in all environments.  Guard the import so the app layer still
# works if the quant engine package is not available.
try:
    from quant_engine.engine.regime import classify_regime  # type: ignore[import]
    _REGIME_AVAILABLE = True
except ImportError:
    _REGIME_AVAILABLE = False
    def classify_regime(ohlcv, lookback=50):  # type: ignore[misc]
        return {"type": "ranging"}

logger = logging.getLogger(__name__)


# ── Typed planner errors (chat layer translates to user-facing messages) ──────

class PlannerError(RuntimeError):
    """Base class for planner failures."""


class UnsupportedStock(PlannerError):
    pass


class UnsupportedTimeframe(PlannerError):
    pass


class NoValidCandidate(PlannerError):
    def __init__(self, role: str, reason: str = ""):
        super().__init__(f"no valid candidate for role={role}: {reason}")
        self.role = role


# ── Pipeline ──────────────────────────────────────────────────────────────────


# Maximum entry filters the planner will stack on top of the entry trigger.
# Real-world setups (ORB + volume spike + VWAP filter, EMA pullback + ADX
# strength, etc.) typically need 1-3 confirmations. Past 3 the marginal value
# drops sharply and the rules become brittle, so we cap here rather than let
# the ranker pile on more.
MAX_ENTRY_FILTERS = 3

# Maximum exit filters (AND-ed confirmations before the exit trigger fires).
# Typically 1 is enough; 2 is the practical ceiling for real strategies.
MAX_EXIT_FILTERS = 2


class Pipeline:
    """One instance per request (light); safe to construct inline."""

    def __init__(
        self,
        kb: KB | None = None,
        intent_extractor: IntentExtractor | None = None,
    ) -> None:
        self.kb = kb or default_kb
        self.intent_extractor = intent_extractor or IntentExtractor()

    async def plan(
        self,
        builder: Any,
        ohlcv: pd.DataFrame | None = None,
    ) -> StrategyPlan:
        trace = DecisionTrace()
        ohlcv_bars = len(ohlcv) if isinstance(ohlcv, pd.DataFrame) else 0

        # ── Step 1: Resolve inputs ──────────────────────────────────────────
        symbol_query = (builder.symbol or "").strip()
        stock = self.kb.lookup_stock(symbol_query)
        if stock is None:
            logger.warning("planner|step1_inputs|stock_not_found|query=%r", symbol_query)
            raise UnsupportedStock(f"unknown stock: {symbol_query!r}")

        raw_tf = (builder.timeframe or "").strip()
        timeframe = self.kb.timeframes.normalize(raw_tf)
        if timeframe is None:
            logger.warning(
                "planner|step1_inputs|timeframe_unsupported|raw=%r|supported=%s",
                raw_tf, self.kb.timeframes.supported,
            )
            raise UnsupportedTimeframe(f"unsupported timeframe: {raw_tf!r}")

        sentiment: SignalDirection = (builder.sentiment or "bullish").lower()
        if sentiment not in {"bullish", "bearish"}:
            sentiment = "bullish"

        experience = (builder.experience or "default").lower()
        objective = (builder.objective or "intraday").lower()
        risk_tier = self.kb.get_risk_tier(experience)

        logger.info(
            "planner|step1_inputs|symbol=%s|tf=%s|sentiment=%s|experience=%s"
            "|objective=%s|ohlcv_bars=%d",
            stock.symbol, timeframe, sentiment, experience, objective, ohlcv_bars,
        )

        # ── Step 2: Intent extraction (only LLM call) ───────────────────────
        intent = await self.intent_extractor.extract(
            getattr(builder, "goal", None),
            self.kb.intent_taxonomy,
        )
        trace.set_intent(intent)
        logger.info(
            "planner|step2_intent|style=%s|frequency=%s|hold=%s|profit=%s|risk=%s",
            intent.style, intent.frequency, intent.hold_horizon,
            intent.profit_size, intent.risk_appetite,
        )

        # ── Step 3: Pick signals (preset short-circuit, else filter+rank) ────
        # If the user pinned a strategy preset (explicitly or via a keyword in
        # their goal), skip the ranker entirely and use the preset's pinned
        # signal stack. This is what lets a chat message like "use ORB strategy"
        # produce the canonical ORB combination instead of whatever the ranker
        # would have chosen for the inferred intent.
        all_cards = list(self.kb.signals.values())
        preset = self._resolve_preset(builder)
        if preset is not None:
            entry_trigger_card, entry_filter_cards, exit_trigger_card, exit_filter_cards = (
                self._apply_preset(preset, sentiment=sentiment, timeframe=timeframe, trace=trace)
            )
        else:
            entry_trigger_card, _ = self._filter_and_pick(
                all_cards,
                role="entry_trigger",
                sentiment=sentiment,
                timeframe=timeframe,
                intent=intent,
                experience=experience,
                avoid_family=None,
                pair_with=None,
                trace=trace,
            )

            # Step 3b: pick up to N entry filters (multi-AND). Each pick excludes
            # its own family so we don't end up with redundant signals (two EMAs,
            # two volume checks, etc.). The pair_with hint cycles to the most
            # recently picked card so each new filter is rewarded for
            # complementing what's already there.
            entry_filter_cards = []
            avoid_families: set[str] = {entry_trigger_card.family}
            most_recent_pick: SignalCard = entry_trigger_card
            for _ in range(MAX_ENTRY_FILTERS):
                filter_card, _ = self._filter_and_pick(
                    all_cards,
                    role="entry_filter",
                    sentiment=sentiment,
                    timeframe=timeframe,
                    intent=intent,
                    experience=experience,
                    avoid_family=None,                  # we use a set check below
                    pair_with=most_recent_pick,
                    trace=trace,
                    optional=True,
                    exclude_families=avoid_families,
                    exclude_names={entry_trigger_card.name, *(c.name for c in entry_filter_cards)},
                )
                if filter_card is None:
                    break
                entry_filter_cards.append(filter_card)
                avoid_families.add(filter_card.family)
                most_recent_pick = filter_card

            exit_trigger_card, _ = self._filter_and_pick(
                all_cards,
                role="exit_trigger",
                sentiment=sentiment,
                timeframe=timeframe,
                intent=intent,
                experience=experience,
                avoid_family=entry_trigger_card.family,
                pair_with=None,
                trace=trace,
            )

            # Step 3b-exit: pick up to MAX_EXIT_FILTERS exit filters (AND-ed
            # confirmations that must hold alongside the exit trigger). Same
            # deduplication logic as entry filters.
            exit_filter_cards = []
            exit_avoid_families: set[str] = {
                exit_trigger_card.family,
                entry_trigger_card.family,
            }
            exit_most_recent: SignalCard = exit_trigger_card
            for _ in range(MAX_EXIT_FILTERS):
                ef_card, _ = self._filter_and_pick(
                    all_cards,
                    role="exit_filter",
                    sentiment=sentiment,
                    timeframe=timeframe,
                    intent=intent,
                    experience=experience,
                    avoid_family=None,
                    pair_with=exit_most_recent,
                    trace=trace,
                    optional=True,
                    exclude_families=exit_avoid_families,
                    exclude_names={
                        exit_trigger_card.name,
                        entry_trigger_card.name,
                        *(c.name for c in exit_filter_cards),
                        *(c.name for c in entry_filter_cards),
                    },
                )
                if ef_card is None:
                    break
                exit_filter_cards.append(ef_card)
                exit_avoid_families.add(ef_card.family)
                exit_most_recent = ef_card

        # ── Step 3c: Semantic signal injection ──────────────────────────────
        # After preset-or-ranker signal selection, inject entry filter signals
        # derived from the canonical semantic intent (HTF confluence rules,
        # relative-strength conditions, ADX/volume confirmations).  This
        # ensures the planner CONSUMES the semantic object rather than ignoring
        # it.  The composer respects MAX_ENTRY_FILTERS and deduplication.
        semantic_intent_dict = getattr(builder, "semantic_intent", None) or {}
        if semantic_intent_dict:
            composer = SemanticSignalComposer(self.kb)
            entry_filter_cards = composer.inject_entry_filters(
                semantic_intent_dict,
                sentiment=sentiment,
                entry_trigger_card=entry_trigger_card,
                existing_filter_cards=entry_filter_cards,
                timeframe=timeframe,
                max_total_filters=MAX_ENTRY_FILTERS,
            )

        # ── Step 5: Resolve params ──────────────────────────────────────────
        et_params, et_src = resolve_params(
            entry_trigger_card, timeframe=timeframe, symbol=stock.symbol,
            objective=objective, ohlcv=ohlcv,
        )
        trace.record_param_source(entry_trigger_card.name, et_src)
        entry_trigger_picked = PickedSignal.from_card(
            entry_trigger_card, "entry_trigger", et_params, timeframe,
        )

        entry_filter_picks: list[PickedSignal] = []
        for filter_card in entry_filter_cards:
            ef_params, ef_src = resolve_params(
                filter_card, timeframe=timeframe, symbol=stock.symbol,
                objective=objective, ohlcv=ohlcv,
            )
            trace.record_param_source(filter_card.name, ef_src)
            entry_filter_picks.append(PickedSignal.from_card(
                filter_card, "entry_filter", ef_params, timeframe,
            ))

        ex_params, ex_src = resolve_params(
            exit_trigger_card, timeframe=timeframe, symbol=stock.symbol,
            objective=objective, ohlcv=ohlcv,
        )
        trace.record_param_source(exit_trigger_card.name, ex_src)
        exit_trigger_picked = PickedSignal.from_card(
            exit_trigger_card, "exit_trigger", ex_params, timeframe,
        )

        exit_filter_picks: list[PickedSignal] = []
        for ef_card in exit_filter_cards:
            ef_params, ef_src = resolve_params(
                ef_card, timeframe=timeframe, symbol=stock.symbol,
                objective=objective, ohlcv=ohlcv,
            )
            trace.record_param_source(ef_card.name, ef_src)
            exit_filter_picks.append(PickedSignal.from_card(
                ef_card, "exit_filter", ef_params, timeframe,
            ))

        # ── Step 5b: Classify market regime ─────────────────────────────────
        # Done after param resolution (signals are already picked) so the
        # regime can inform SL/TP scaling but not signal selection (which
        # would require re-running the ranker in a feedback loop).
        # Regime is best-effort: if OHLCV is unavailable it defaults to "ranging".
        regime: dict = {}
        if isinstance(ohlcv, pd.DataFrame) and len(ohlcv) >= 20:
            try:
                regime = classify_regime(ohlcv)
                trace.record_custom("regime", regime)
                logger.info(
                    "planner|step5b_regime|type=%s|adx=%s|trend_strength=%.2f|slope=%s",
                    regime.get("type"),
                    regime.get("adx"),
                    regime.get("trend_strength", 0.0),
                    regime.get("ema_slope_pct"),
                )
            except Exception as exc:
                logger.warning("planner|regime_classification_failed|err=%s", exc)

        # ── Step 6: SL/TP ───────────────────────────────────────────────────
        # The user can pin a risk:reward via builder.risk_execution_config.
        # When supplied (e.g. "1:2"), SL stays ATR-scaled but TP = SL × RR so
        # the plan honors the user's stated reward expectation.
        risk_cfg = getattr(builder, "risk_execution_config", None) or {}
        user_rr = risk_cfg.get("risk_reward")
        sl_pct, tp_pct, vol_mult = resolve_sl_tp(
            risk_tier, ohlcv=ohlcv, user_risk_reward=user_rr, regime=regime,
        )
        trace.record_sl_tp(sl_pct, tp_pct, vol_mult)
        logger.info(
            "planner|step6_sl_tp|sl=%s%%|tp=%s%%|vol_mult=%s|user_rr=%s"
            "|baseline_sl=%s%%|baseline_tp=%s%%",
            sl_pct, tp_pct, vol_mult, user_rr,
            risk_tier.base_sl_pct, risk_tier.base_tp_pct,
        )

        # ── Step 7: Build + validate ────────────────────────────────────────
        plan = StrategyPlan(
            stock=stock,
            timeframe=timeframe,
            sentiment=sentiment,
            intent=intent,
            entry_trigger=entry_trigger_picked,
            entry_filters=entry_filter_picks,
            exit_trigger=exit_trigger_picked,
            exit_filters=exit_filter_picks,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            risk=build_risk_dict(risk_tier),
            # User-specified structural SL/trailing overrides the preset default.
            stop_loss_spec=(
                getattr(builder, "stop_loss_spec", None)
                or (preset.stop_loss if preset is not None else None)
            ),
            trailing_stop_spec=(
                getattr(builder, "trailing_stop_spec", None)
                or (preset.trailing_stop if preset is not None else None)
            ),
            reference_symbol=(
                (preset.reference_symbol if preset is not None else None)
                or getattr(builder, "reference_symbol", None)
            ),
            # HTF rules are leg-scoped (direction-dependent), so we resolve
            # them against the chosen sentiment leg of the preset.
            # Also include any HTF rules already on the builder (put there by
            # a prior turn's semantic extraction or by the user explicitly).
            htf_rules=(
                list(preset.leg(sentiment).htf)
                if (preset is not None
                    and preset.leg(sentiment) is not None
                    and preset.leg(sentiment).htf)
                else list(getattr(builder, "htf_rules", None) or [])
            ),
            time_exit=preset.time_exit if preset is not None else None,
            trace=trace.to_dict(),
        )

        try:
            validate_plan(plan)
            trace.mark_validated()
        except PlanInvariantViolation as exc:
            logger.error(
                "planner|step7_validate|FAIL|symbol=%s|sentiment=%s|err=%s",
                stock.symbol, sentiment, exc,
            )
            trace.mark_failed(str(exc))
            trace.log()
            raise

        # ── Step 8: Final trace ─────────────────────────────────────────────
        plan.trace = trace.to_dict()
        trace.log(level=logging.DEBUG)        # full trace at DEBUG; summary above
        filters_summary = (
            ",".join(p.name for p in entry_filter_picks)
            if entry_filter_picks else "—"
        )
        exit_filters_summary = (
            ",".join(p.name for p in exit_filter_picks)
            if exit_filter_picks else "—"
        )
        logger.info(
            "planner|done|symbol=%s|tf=%s|entry=%s|entry_filters=[%s]"
            "|exit=%s|exit_filters=[%s]|sl=%s%%|tp=%s%%",
            stock.symbol, timeframe,
            entry_trigger_picked.name,
            filters_summary,
            exit_trigger_picked.name,
            exit_filters_summary,
            sl_pct, tp_pct,
        )
        return plan

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_preset(self, builder: Any):
        """Return the preset pinned by the user, or None.

        Resolution order:
          1. Explicit `builder.strategy_preset` (set by chat layer or API),
             but only when it is compatible with the requested timeframe.
             If the stored preset doesn't support the timeframe (e.g. because
             a prior turn matched the wrong keyword or the user later changed
             the timeframe), fall through to goal-text detection rather than
             raising UnsupportedTimeframe immediately.
          2. Semantic family mapping from `builder.semantic_intent.strategy_family`.
             The SemanticExtractor identifies the *primary framework* of the
             strategy (ORB, EMA_PULLBACK, VWAP_RECLAIM…) from prose structure,
             not just keyword co-occurrence.  When the KB keyword scorer would
             otherwise latch onto a secondary filter/modifier preset (e.g.
             relative_strength when the user said "ORB with RS filter"), this
             layer corrects the selection by using the structurally-identified
             framework as the primary preset.
          3. Keyword scan over `builder.goal` so a user can just type
             "ORB strategy on RELIANCE 5m" and get the ORB preset.
        """
        timeframe = (getattr(builder, "timeframe", None) or "").strip()
        explicit = (getattr(builder, "strategy_preset", None) or "").strip()
        if explicit:
            preset = self.kb.lookup_preset(explicit)
            if preset is not None:
                if (
                    not timeframe
                    or not preset.supported_timeframes
                    or timeframe in preset.supported_timeframes
                ):
                    return preset
                logger.warning(
                    "planner|preset|explicit_timeframe_mismatch|name=%r"
                    "|tf=%r|supported=%s — falling back to semantic/goal-text detection",
                    explicit, timeframe, preset.supported_timeframes,
                )
            else:
                logger.info("planner|preset|explicit_unknown|name=%r", explicit)

        # ── Layer 2: semantic base_framework → preset via KB ────────────────
        # The canonical intent stores base_framework — a preset name that the
        # SemanticIntentNormalizer resolved directly via
        # KB.detect_primary_framework_in_text(source_prompt).  We ask the KB
        # directly; no hardcoded family-name tables exist anywhere.
        semantic_intent: dict = getattr(builder, "semantic_intent", None) or {}
        canonical_framework: str = (semantic_intent.get("base_framework") or "").strip()

        if canonical_framework:
            family_preset = self.kb.lookup_preset(canonical_framework)
            # Exact lookup failed — let the KB keyword scorer resolve the name.
            # This avoids hardcoded suffix patterns like "_structural", "_trail";
            # instead we treat the framework string as free text and let the KB
            # detect the best matching primary-framework preset from its own
            # keyword lists.
            if family_preset is None:
                family_preset = self.kb.detect_primary_framework_in_text(canonical_framework)
            if family_preset is not None:
                if (
                    not timeframe
                    or not family_preset.supported_timeframes
                    or timeframe in family_preset.supported_timeframes
                ):
                    logger.info(
                        "planner|preset|semantic_framework_override"
                        "|framework=%r|preset=%r|tf=%r",
                        canonical_framework, family_preset.name, timeframe,
                    )
                    return family_preset
                logger.info(
                    "planner|preset|semantic_framework_tf_mismatch|framework=%r"
                    "|preset=%r|tf=%r|supported=%s — falling through to goal-text",
                    canonical_framework, family_preset.name, timeframe,
                    family_preset.supported_timeframes,
                )

        # ── Layer 3: goal-text keyword detection ───────────────────────────
        goal_text = getattr(builder, "goal", None) or ""
        detected = self.kb.detect_preset_in_text(goal_text)
        if detected is not None and timeframe and detected.supported_timeframes:
            if timeframe not in detected.supported_timeframes:
                logger.info(
                    "planner|preset|goal_detected_timeframe_mismatch|name=%r"
                    "|tf=%r|supported=%s — no preset applied",
                    detected.name, timeframe, detected.supported_timeframes,
                )
                return None
        return detected

    def _apply_preset(
        self,
        preset,
        *,
        sentiment: SignalDirection,
        timeframe: str,
        trace: DecisionTrace,
    ) -> tuple[SignalCard, list[SignalCard], SignalCard, list[SignalCard]]:
        """Pin the preset's signal stack for the requested sentiment.

        Raises NoValidCandidate when the preset has no leg matching sentiment,
        or UnsupportedTimeframe when the requested timeframe is not declared.
        """
        if preset.supported_timeframes and timeframe not in preset.supported_timeframes:
            raise UnsupportedTimeframe(
                f"preset '{preset.name}' supports {preset.supported_timeframes}, "
                f"got {timeframe!r}"
            )
        leg = preset.leg(sentiment)
        if leg is None:
            raise NoValidCandidate(
                role="entry_trigger",
                reason=f"preset '{preset.name}' has no {sentiment} leg",
            )

        try:
            entry_trigger = self.kb.signals[leg.entry_trigger]
            entry_filters = [self.kb.signals[s] for s in leg.entry_filters]
            exit_trigger = self.kb.signals[leg.exit_trigger]
            exit_filters = [self.kb.signals[s] for s in leg.exit_filters]
        except KeyError as exc:
            # _load_presets validates references at startup; if we land here
            # the KB was reloaded with mismatched files. Surface as a planner
            # failure rather than a silent broken plan.
            raise NoValidCandidate(
                role="entry_trigger",
                reason=f"preset '{preset.name}' references missing signal {exc}",
            ) from exc

        # Record what the preset selected so the trace shows it as the source
        # of truth (instead of leaving the trace's role-pick fields blank).
        trace.record_initial("entry_trigger", [entry_trigger.name])
        trace.record_pick("entry_trigger", entry_trigger.name)
        for card in entry_filters:
            trace.record_initial("entry_filter", [card.name])
            trace.record_pick("entry_filter", card.name)
        trace.record_initial("exit_trigger", [exit_trigger.name])
        trace.record_pick("exit_trigger", exit_trigger.name)
        for card in exit_filters:
            trace.record_initial("exit_filter", [card.name])
            trace.record_pick("exit_filter", card.name)
        logger.info(
            "planner|preset_pinned|preset=%s|sentiment=%s|tf=%s"
            "|trigger=%s|entry_filters=%s|exit=%s|exit_filters=%s",
            preset.name, sentiment, timeframe,
            entry_trigger.name,
            [c.name for c in entry_filters],
            exit_trigger.name,
            [c.name for c in exit_filters],
        )
        return entry_trigger, entry_filters, exit_trigger, exit_filters

    def _filter_and_pick(
        self,
        all_cards: list[SignalCard],
        *,
        role: str,
        sentiment: SignalDirection,
        timeframe: str,
        intent: Intent,
        experience: str,
        avoid_family: str | None,
        pair_with: SignalCard | None,
        trace: DecisionTrace,
        optional: bool = False,
        exclude_families: set[str] | None = None,
        exclude_names: set[str] | None = None,
    ) -> tuple[SignalCard | None, PickedSignal | None]:
        trace.record_initial(role, [c.name for c in all_cards if role in c.roles])

        survivors, eliminations = hard_filters.filter_candidates(
            all_cards,
            role=role,
            sentiment=sentiment,
            timeframe=timeframe,
            intent=intent,
        )
        # Drop any signal whose family or name is already represented in the
        # plan — used by the multi-filter loop to avoid stacking duplicates.
        if exclude_families:
            survivors = [c for c in survivors if c.family not in exclude_families]
        if exclude_names:
            survivors = [c for c in survivors if c.name not in exclude_names]
        trace.record_eliminations(role, eliminations)

        if not survivors:
            if optional:
                logger.info(
                    "planner|filter|role=%s|survivors=0|skipped (optional role)",
                    role,
                )
                trace.record_pick(role, "")
                return None, None
            logger.error(
                "planner|filter|role=%s|survivors=0|sentiment=%s|tf=%s — all eliminated",
                role, sentiment, timeframe,
            )
            raise NoValidCandidate(role, f"all eliminated by hard filters")

        winner_card, ranking = soft_ranker.pick(
            survivors,
            intent=intent,
            experience=experience,
            timeframe=timeframe,
            avoid_family=avoid_family,
            pair_with=pair_with,
        )
        trace.record_ranking(role, ranking)
        trace.record_pick(role, winner_card.name)
        top3 = [(r.signal, r.score) for r in ranking[:3]]
        logger.info(
            "planner|pick|role=%s|winner=%s|score=%.3f|survivors=%d|top3=%s",
            role, winner_card.name, ranking[0].score, len(survivors), top3,
        )
        return winner_card, None      # PickedSignal is built later, after params resolve


# ── Convenience entrypoint ────────────────────────────────────────────────────

_default_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline


async def plan_strategy(builder: Any, ohlcv: pd.DataFrame | None = None) -> StrategyPlan:
    return await get_pipeline().plan(builder, ohlcv=ohlcv)
