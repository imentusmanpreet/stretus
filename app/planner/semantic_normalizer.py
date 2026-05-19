"""
app/planner/semantic_normalizer.py — Build CanonicalSemanticIntent from
SemanticInstructions and StrategyBuilder state.

All signal and preset lookups go through the KB.  There are no hardcoded
strategy-family→preset mappings, indicator→signal-name tables, or filter-type
dictionaries.  Adding a new preset or signal to the KB automatically makes it
reachable without touching this file.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.kb import kb as default_kb
from app.kb.canonical_intent import (
    CanonicalFilter,
    CanonicalHTFRule,
    CanonicalRiskModel,
    CanonicalRiskReward,
    CanonicalSemanticIntent,
    CanonicalStopLoss,
    CanonicalTrailingStop,
)
from app.kb.execution_schemas import SemanticInstructions
from app.kb.loader import KB
from app.kb.market_aliases import normalise_benchmark
from app.kb.schemas import SignalCard

logger = logging.getLogger(__name__)


# ── KB signal search helpers ───────────────────────────────────────────────────

def _score_signal_for_keywords(
    signal: SignalCard,
    keywords: list[str],
    bias: str,
) -> float:
    """Score a KB signal card against a list of search keywords.

    Higher score = better match.  Direction compatibility is checked:
    a bearish bias prefers bearish/neutral signals; bullish bias prefers
    bullish/neutral.
    """
    name = signal.name.lower()
    desc = (signal.description or "").lower()
    sig_dir = (signal.direction or "neutral").lower()

    # Direction gate
    if bias == "bearish" and sig_dir == "bullish":
        return 0.0
    if bias == "bullish" and sig_dir == "bearish":
        return 0.0

    score = 0.0
    for kw in keywords:
        kw_lower = kw.lower()
        if name.startswith(kw_lower):
            score += 10.0
        elif kw_lower in name:
            score += 6.0
        elif kw_lower in desc:
            score += 2.0

    # Small bonus for direction-specific signals over neutral
    if sig_dir != "neutral":
        score += 0.5

    return score


def _search_kb_entry_filter(
    kb: KB,
    keywords: list[str],
    bias: str,
) -> str | None:
    """Return the name of the best KB entry_filter signal for the given keywords
    and bias, or None if nothing matches."""
    best_name: str | None = None
    best_score: float = 0.0

    for signal in kb.signals.values():
        if "entry_filter" not in (signal.roles or []):
            continue
        score = _score_signal_for_keywords(signal, keywords, bias)
        if score > best_score:
            best_score = score
            best_name = signal.name

    if best_name:
        logger.debug(
            "semantic_normalizer|kb_signal_search|kws=%s|bias=%r"
            "|best=%r|score=%.1f",
            keywords, bias, best_name, best_score,
        )
    return best_name


def _opposite_direction_signal(signal_name: str, kb: KB) -> str | None:
    """Return the opposite-direction variant of a signal using KB contradicts.

    E.g. "ema_above" → "ema_below", "rs_outperforms_reference" → "rs_underperforms_reference".
    Returns None if no opposite is found.
    """
    card = kb.signals.get(signal_name)
    if not card:
        return None
    for contradicted in (card.contradicts or []):
        other = kb.signals.get(contradicted)
        if other and other.direction == "bearish":
            return contradicted
    return None


# ── SL / trailing conversion helpers ─────────────────────────────────────────

def _sl_spec_to_canonical(
    sl_spec: dict[str, Any] | None,
    source: str = "user",
) -> CanonicalStopLoss | None:
    if not sl_spec:
        return None
    sl_type = sl_spec.get("type") or "structural"
    if sl_type in ("percent", "stop_loss_pct"):
        return CanonicalStopLoss(
            type="percent",
            percent=sl_spec.get("value") or sl_spec.get("percent"),
            source=source,
        )
    return CanonicalStopLoss(
        type="structural",
        anchor=sl_spec.get("anchor") or sl_type,
        atr_multiple=sl_spec.get("atr_multiple"),
        source=source,
    )


def _trailing_spec_to_canonical(
    ts_spec: dict[str, Any] | None,
    source: str = "user",
) -> CanonicalTrailingStop | None:
    if not ts_spec:
        return None
    return CanonicalTrailingStop(
        type=ts_spec.get("type") or "percent_based",
        ema_period=ts_spec.get("ema_period"),
        activate_after_pct=ts_spec.get("activate_after_pct"),
        source=source,
    )


# ── Main normaliser ────────────────────────────────────────────────────────────

class SemanticIntentNormalizer:
    """Merge SemanticInstructions + builder state → CanonicalSemanticIntent.

    All preset and signal lookups go through the KB instance passed at
    construction time (defaults to the global KB singleton).
    """

    def __init__(self, kb: KB | None = None):
        self._kb = kb or default_kb

    def normalize(
        self,
        sem: SemanticInstructions | None,
        *,
        strategy_preset: str | None = None,
        timeframe: str | None = None,
        sentiment: str | None = None,
        stop_loss_spec: dict[str, Any] | None = None,
        trailing_stop_spec: dict[str, Any] | None = None,
        risk_execution_config: dict[str, Any] | None = None,
        source_prompt: str | None = None,
    ) -> CanonicalSemanticIntent:
        """Build a CanonicalSemanticIntent.

        Priority for each field:
          user-provided builder field > semantic extraction > None
        """
        rms = risk_execution_config or {}
        rms_sources = rms.get("rms_sources", {})
        bias = (sentiment or "bullish").lower()

        intent = CanonicalSemanticIntent(
            base_framework=self._resolve_framework(sem, strategy_preset, source_prompt),
            execution_timeframe=timeframe,
            market_bias=bias or None,
            source_prompt=(source_prompt or "")[:200] or None,
        )

        intent.htf_confluence = self._build_htf_confluence(sem, bias)
        intent.filters = self._build_filters(sem, bias)
        intent.risk_model = self._build_risk_model(
            sem,
            stop_loss_spec=stop_loss_spec,
            trailing_stop_spec=trailing_stop_spec,
            rms=rms,
            rms_sources=rms_sources,
        )
        intent.indicators = dict(sem.indicators) if sem else {}
        intent.quality_score = self._score(intent)

        logger.debug(
            "semantic_normalizer|normalized|framework=%r|bias=%r"
            "|htf=%d|filters=%d|sl=%r|rr=%r|quality=%.2f",
            intent.base_framework, intent.market_bias,
            len(intent.htf_confluence), len(intent.filters),
            intent.risk_model.stop_loss.type if intent.risk_model.stop_loss else None,
            intent.risk_model.risk_reward.ratio if intent.risk_model.risk_reward else None,
            intent.quality_score,
        )
        return intent

    # ── Framework resolution ───────────────────────────────────────────────────

    def _resolve_framework(
        self,
        sem: SemanticInstructions | None,
        strategy_preset: str | None,
        source_prompt: str | None,
    ) -> str | None:
        """Resolve the base_framework preset name.

        Priority:
          1. builder.strategy_preset (explicit user/API choice)
          2. KB primary-framework detection on the source prompt — uses
             KB.detect_primary_framework_in_text() which only considers
             is_primary_framework=True presets, preventing filter presets
             (e.g. relative_strength) from being promoted to framework level.

        No hardcoded family-name table here.  Every preset in the KB is
        auto-discoverable as long as its YAML has appropriate keywords and the
        correct is_primary_framework flag.
        """
        if strategy_preset and strategy_preset.strip():
            return strategy_preset.strip().lower()
        if source_prompt:
            preset = self._kb.detect_primary_framework_in_text(source_prompt)
            if preset is not None:
                return preset.name
        return None

    # ── HTF confluence ─────────────────────────────────────────────────────────

    def _build_htf_confluence(
        self,
        sem: SemanticInstructions | None,
        bias: str,
    ) -> list[CanonicalHTFRule]:
        if not sem or not sem.htf_rules:
            return []
        rules: list[CanonicalHTFRule] = []
        for htf in sem.htf_rules:
            # Build keyword list from the condition text and description
            condition_text = (htf.condition or "") + " " + (htf.description or "")
            keywords = self._extract_keywords_from_text(condition_text)
            # Ask the KB for the best entry_filter signal for these keywords
            signal = _search_kb_entry_filter(self._kb, keywords, bias)
            # Extract EMA period hint if present
            params: dict[str, Any] = {}
            if signal and "ema" in (signal or "").lower() and sem.indicators.get("EMA"):
                params["period"] = max(sem.indicators["EMA"])
            rules.append(CanonicalHTFRule(
                timeframe=htf.timeframe or "1h",
                condition=htf.condition,
                indicator=self._dominant_keyword(keywords),
                signal=signal,
                params=params,
                role=htf.role or "gating",
                source="semantic",
            ))
        return rules

    # ── Secondary filters ──────────────────────────────────────────────────────

    def _build_filters(
        self,
        sem: SemanticInstructions | None,
        bias: str,
    ) -> list[CanonicalFilter]:
        if not sem:
            return []
        filters: list[CanonicalFilter] = []

        # RS / benchmark-relative conditions
        for rs_cond in sem.reference_symbols:
            benchmark = normalise_benchmark(rs_cond.reference_symbol or "NIFTY50")
            # Search for the rs/outperformance signal in the KB
            signal = _search_kb_entry_filter(
                self._kb,
                ["rs_", "outperform", "relative", "reference"],
                bias,
            )
            filters.append(CanonicalFilter(
                type="relative_strength",
                benchmark=benchmark,
                signal=signal,
                params={"lookback": 20},
                source="semantic",
            ))

        # Volume condition
        if sem.volume_momentum and sem.volume_momentum.volume:
            vol_type = (sem.volume_momentum.volume.filter_type or "spike").lower()
            signal = _search_kb_entry_filter(self._kb, ["volume", vol_type], bias)
            filters.append(CanonicalFilter(
                type="volume",
                signal=signal,
                source="semantic",
            ))

        # Momentum / ADX / MACD conditions
        if sem.volume_momentum and sem.volume_momentum.momentum:
            mom = sem.volume_momentum.momentum
            mom_type = (mom.filter_type or "").lower()
            params: dict[str, Any] = {}
            if mom.adx_threshold:
                params["threshold"] = mom.adx_threshold
            # Use the momentum description to search the KB
            keywords = [w for w in re.split(r"[_\s]+", mom_type) if w]
            if not keywords:
                keywords = ["momentum"]
            signal = _search_kb_entry_filter(self._kb, keywords, bias)
            if signal:
                filters.append(CanonicalFilter(
                    type=mom_type or "momentum",
                    signal=signal,
                    params=params,
                    source="semantic",
                ))

        return filters

    # ── Risk model ─────────────────────────────────────────────────────────────

    def _build_risk_model(
        self,
        sem: SemanticInstructions | None,
        *,
        stop_loss_spec: dict[str, Any] | None,
        trailing_stop_spec: dict[str, Any] | None,
        rms: dict[str, Any],
        rms_sources: dict[str, Any],
    ) -> CanonicalRiskModel:
        # ── Stop loss ────────────────────────────────────────────────────────
        # Priority: builder.stop_loss_spec (user) wins, BUT the structural
        # anchor is cross-checked against the semantic extractor's finding.
        # When an LLM extraction sets the anchor to a breakout LEVEL (e.g.
        # "opening_range_high") rather than a genuine SL level, and the regex
        # extractor found the correct anchor (e.g. "opening_range_low"), the
        # semantic extractor's anchor is preferred — breakout levels are never
        # valid structural stop-loss anchors.
        sl: CanonicalStopLoss | None = None
        if stop_loss_spec:
            sl = _sl_spec_to_canonical(stop_loss_spec, source="user")
            # Correction: if anchor smells like a breakout level but semantic
            # extractor found a real SL anchor, override.
            if sl and sem and sem.stop_loss and sem.stop_loss.anchor:
                sem_anchor = sem.stop_loss.anchor
                current_anchor = (sl.anchor or "").lower()
                # "high" in a bullish structural SL is always suspect
                if "high" in current_anchor and "high" not in sem_anchor.lower():
                    logger.info(
                        "semantic_normalizer|sl_anchor_correction"
                        "|llm_anchor=%r → sem_anchor=%r",
                        sl.anchor, sem_anchor,
                    )
                    sl = sl.model_copy(update={"anchor": sem_anchor, "source": "semantic"})
        elif sem and sem.stop_loss:
            s = sem.stop_loss
            anchor = s.anchor or s.type
            sl = CanonicalStopLoss(
                type="structural" if s.type not in ("percent", "atr_multiple") else s.type,
                anchor=anchor,
                atr_multiple=(s.padding.atr_multiple
                              if s.padding and s.padding.method == "atr" else None),
                source="semantic",
            )

        # ── Risk:Reward ──────────────────────────────────────────────────────
        # Guard against LLM-hallucinated RR values: only trust a builder RMS
        # RR when the source is "user" or "semantic" AND the semantic extractor
        # did NOT find a contradicting RR in the prompt.  When neither the
        # extractor nor the user explicitly stated an RR, skip it entirely so
        # the planner uses the preset default or leaves it open.
        rr: CanonicalRiskReward | None = None
        rr_val = rms.get("risk_reward")
        if rr_val is not None:
            ratio = rr_val["value"] if isinstance(rr_val, dict) else float(rr_val)
            src = rms_sources.get("risk_reward", "user")
            # Only carry forward if the semantic extractor also found an RR
            # (corroborates the builder value) OR if the source is explicitly
            # "user" AND the extractor found nothing to contradict it.
            if sem and sem.risk_reward and sem.risk_reward.ratio:
                # Both agree — use builder's value (may be more precise)
                rr = CanonicalRiskReward(ratio=ratio, source=src)
            elif src == "user":
                # Builder claims user-set, but the semantic extractor found
                # no RR in the original text.  This almost always means the
                # LLM hallucinated a "reasonable default" (2.5 is the most
                # common phantom value for intraday strategies).  Only carry
                # the builder value forward when it does NOT match a known
                # LLM-default constant.  Without extractor corroboration we
                # cannot trust any round default.
                _LLM_DEFAULT_RR = {2.5, 3.0}
                if ratio not in _LLM_DEFAULT_RR:
                    rr = CanonicalRiskReward(ratio=ratio, source="user")
            # else: builder has a value but source is not "user" and extractor
            # found nothing → discard to prevent phantom RR.
        elif sem and sem.risk_reward and sem.risk_reward.ratio:
            rr = CanonicalRiskReward(
                ratio=sem.risk_reward.ratio,
                type=sem.risk_reward.type or "minimum",
                source="semantic",
            )

        # ── Trailing stop ────────────────────────────────────────────────────
        # Builder wins for type; semantic extractor adds activate_after_pct
        # when the builder didn't capture it (LLM often misses thresholds).
        ts: CanonicalTrailingStop | None = None
        if trailing_stop_spec:
            ts = _trailing_spec_to_canonical(trailing_stop_spec, source="user")
            # Fill in missing activate_after_pct from semantic extractor
            if ts and ts.activate_after_pct is None and sem and sem.trailing_stop:
                if sem.trailing_stop.activate_after_pct is not None:
                    ts = ts.model_copy(update={
                        "activate_after_pct": sem.trailing_stop.activate_after_pct
                    })
        elif sem and sem.trailing_stop and sem.trailing_stop.enabled:
            ts = CanonicalTrailingStop(
                type=sem.trailing_stop.type or "percent_based",
                ema_period=sem.trailing_stop.ema_period,
                activate_after_pct=sem.trailing_stop.activate_after_pct,
                source="semantic",
            )

        return CanonicalRiskModel(stop_loss=sl, risk_reward=rr, trailing_stop=ts)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _extract_keywords_from_text(self, text: str) -> list[str]:
        """Split text into meaningful keywords for KB signal search."""
        text_lower = text.lower()
        # Keep only alpha-numeric tokens, ≥ 2 chars
        tokens = [t for t in re.split(r"[^a-z0-9]+", text_lower) if len(t) >= 2]
        # Exclude stop words
        stop = {"the", "a", "an", "is", "be", "in", "on", "at", "to", "of",
                "and", "or", "should", "must", "will", "that", "this", "with",
                "above", "below", "up", "down", "it", "its", "for", "if",
                "we", "are", "was"}
        return [t for t in tokens if t not in stop] or ["trend"]

    def _dominant_keyword(self, keywords: list[str]) -> str | None:
        """Return the single most informative keyword from the list."""
        return max(keywords, key=len) if keywords else None

    def _score(self, intent: CanonicalSemanticIntent) -> float:
        score = 0.0
        if intent.base_framework:
            score += 0.2
        if intent.execution_timeframe:
            score += 0.1
        if intent.market_bias:
            score += 0.1
        if intent.htf_confluence:
            score += 0.15
        if intent.filters:
            score += 0.15
        if intent.risk_model.stop_loss:
            score += 0.1
        if intent.risk_model.risk_reward:
            score += 0.1
        if intent.risk_model.trailing_stop:
            score += 0.05
        if intent.indicators:
            score += 0.05
        return min(score, 1.0)
