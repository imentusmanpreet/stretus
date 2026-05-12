"""
app/kb/loader.py — typed in-memory knowledge base.

The whole KB is read once on startup, validated by Pydantic, and held in memory
as a singleton. Reads after that are O(1) dict lookups — no I/O, no JSON parsing,
no embedding queries. Hot-reloadable via `kb.reload()` for admin endpoints.
"""
from __future__ import annotations

import csv
import logging
from functools import cached_property
from pathlib import Path

import yaml

from app.kb.schemas import (
    IntentTaxonomy,
    MarketConfig,
    RiskTier,
    SignalCard,
    Stock,
    StrategyPreset,
    TimeframeConfig,
)

logger = logging.getLogger(__name__)

_KB_ROOT = Path(__file__).parent


class KBLoadError(RuntimeError):
    """Raised when KB files are malformed or required ones are missing."""


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise KBLoadError(f"KB file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise KBLoadError(f"KB file must contain a YAML mapping at the top level: {path}")
    return data


class KB:
    """Process-wide knowledge base singleton.

    Usage:
        from app.kb import kb
        card = kb.signals["ema_cross_up"]
        stock = kb.lookup_stock("hdfc")
    """

    def __init__(self, root: Path = _KB_ROOT) -> None:
        self._root = root
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        self.timeframes: TimeframeConfig = self._load_timeframes()
        self.markets:    dict[str, MarketConfig] = self._load_markets()
        self.risk_tiers: dict[str, RiskTier] = self._load_risk_tiers()
        self.intent_taxonomy: IntentTaxonomy = self._load_intent_taxonomy()
        self.signals:    dict[str, SignalCard] = self._load_signals()
        self.stocks:     dict[str, Stock] = self._load_stocks()
        self.aliases:    dict[str, str] = self._load_aliases()
        self.presets:    dict[str, StrategyPreset] = self._load_presets()
        self._invalidate_indexes()
        logger.info(
            "kb|loaded|signals=%d stocks=%d aliases=%d timeframes=%d presets=%d",
            len(self.signals), len(self.stocks), len(self.aliases),
            len(self.timeframes.supported), len(self.presets),
        )

    def reload(self) -> None:
        """Re-read every file. For admin /admin/kb/reload endpoint."""
        self._load()

    # ── Per-file loaders ──────────────────────────────────────────────────────

    def _load_timeframes(self) -> TimeframeConfig:
        return TimeframeConfig(**_read_yaml(self._root / "timeframes.yaml"))

    def _load_markets(self) -> dict[str, MarketConfig]:
        raw = _read_yaml(self._root / "markets.yaml")
        return {name: MarketConfig(**cfg) for name, cfg in raw.items()}

    def _load_risk_tiers(self) -> dict[str, RiskTier]:
        raw = _read_yaml(self._root / "risk_tiers.yaml")
        return {name: RiskTier(**cfg) for name, cfg in raw.items()}

    def _load_intent_taxonomy(self) -> IntentTaxonomy:
        return IntentTaxonomy(**_read_yaml(self._root / "intent_taxonomy.yaml"))

    def _load_signals(self) -> dict[str, SignalCard]:
        registry_path = self._root / "signals" / "_registry.yaml"
        registry = _read_yaml(registry_path)
        enabled: list[str] = list(registry.get("enabled") or [])
        if not enabled:
            raise KBLoadError(f"No enabled signals in {registry_path}")

        cards: dict[str, SignalCard] = {}
        for name in enabled:
            card_path = self._root / "signals" / f"{name}.yaml"
            data = _read_yaml(card_path)
            card = SignalCard(**data)
            if card.name != name:
                raise KBLoadError(
                    f"Signal card name mismatch in {card_path}: file says '{card.name}', "
                    f"registry says '{name}'"
                )
            cards[card.name] = card
        return cards

    def _load_stocks(self) -> dict[str, Stock]:
        path = self._root / "stocks" / "universe.csv"
        if not path.exists():
            raise KBLoadError(f"Stock universe not found: {path}")
        stocks: dict[str, Stock] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                row["enabled"] = str(row.get("enabled", "true")).lower() == "true"
                stock = Stock(**row)
                if not stock.enabled:
                    continue
                stocks[stock.symbol] = stock
        return stocks

    def _load_presets(self) -> dict[str, StrategyPreset]:
        presets_dir = self._root / "presets"
        if not presets_dir.exists():
            return {}
        presets: dict[str, StrategyPreset] = {}
        for path in sorted(presets_dir.glob("*.yaml")):
            data = _read_yaml(path)
            preset = StrategyPreset(**data)
            if preset.name in presets:
                raise KBLoadError(f"duplicate preset name '{preset.name}' in {path}")
            # Validate every signal referenced by the preset is loaded.
            for leg_name in ("bullish", "bearish"):
                leg = getattr(preset, leg_name)
                if leg is None:
                    continue
                for sig in leg.all_signal_names():
                    if sig not in self.signals:
                        raise KBLoadError(
                            f"preset '{preset.name}' references unknown signal "
                            f"'{sig}' in {leg_name} leg ({path})"
                        )
            presets[preset.name] = preset
        return presets

    def _load_aliases(self) -> dict[str, str]:
        path = self._root / "stocks" / "aliases.yaml"
        if not path.exists():
            return {}
        raw = _read_yaml(path)
        # Lowercase keys; values are canonical symbols.
        return {str(k).strip().lower(): str(v).strip() for k, v in raw.items()}

    # ── Derived indexes ───────────────────────────────────────────────────────

    def _invalidate_indexes(self) -> None:
        # Clear cached_property values so the next access recomputes.
        for key in ("signals_by_role", "_auto_aliases"):
            self.__dict__.pop(key, None)
        # presets is a plain dict (re-populated on _load), so nothing to clear here.

    @cached_property
    def signals_by_role(self) -> dict[str, list[SignalCard]]:
        """Map role → list of cards eligible for that role."""
        index: dict[str, list[SignalCard]] = {}
        for card in self.signals.values():
            for role in card.roles:
                index.setdefault(role, []).append(card)
        return index

    @cached_property
    def _auto_aliases(self) -> dict[str, str]:
        """Auto-derived aliases from stock display names (lowercased token sets)."""
        out: dict[str, str] = {}
        for symbol, stock in self.stocks.items():
            base = symbol.split(".", 1)[0].lower()
            out[base] = symbol                      # "hdfcbank" → HDFCBANK.NS
            full = stock.display_name.strip().lower()
            out[full] = symbol                      # full name
        return out

    # ── Public lookups ────────────────────────────────────────────────────────

    def lookup_stock(self, query: str) -> Stock | None:
        """Resolve a free-text query to a Stock. Returns None if unknown."""
        if not query:
            return None
        q = query.strip().lower()

        # Direct symbol match (e.g. "HDFCBANK.NS")
        for symbol, stock in self.stocks.items():
            if symbol.lower() == q:
                return stock

        # Manual aliases
        if q in self.aliases:
            sym = self.aliases[q]
            return self.stocks.get(sym)

        # Auto-derived aliases
        if q in self._auto_aliases:
            return self.stocks.get(self._auto_aliases[q])

        return None

    def signals_with_role(self, role: str) -> list[SignalCard]:
        """All cards that can serve in a given role."""
        return list(self.signals_by_role.get(role, []))

    def get_risk_tier(self, experience: str | None) -> RiskTier:
        key = (experience or "").strip().lower()
        return self.risk_tiers.get(key) or self.risk_tiers["default"]

    def lookup_preset(self, query: str) -> StrategyPreset | None:
        """Resolve a preset by exact name or any of its declared keywords.
        Returns None when no preset matches; the planner then falls back to
        its filter+rank pipeline."""
        if not query:
            return None
        q = query.strip().lower()
        if q in self.presets:
            return self.presets[q]
        for preset in self.presets.values():
            if any(k.strip().lower() == q for k in preset.keywords):
                return preset
        return None

    def detect_preset_in_text(self, text: str) -> StrategyPreset | None:
        """Scan free text for a preset keyword. Used by the chat layer to
        pin a preset from the user's goal sentence — e.g. 'I want an ORB
        strategy on RELIANCE 5m'. Picks the longest matching keyword to avoid
        'orb' grabbing a sentence that says 'opening range breakout'."""
        if not text:
            return None
        haystack = f" {text.strip().lower()} "
        best: tuple[int, StrategyPreset] | None = None
        for preset in self.presets.values():
            for keyword in preset.keywords:
                kw = keyword.strip().lower()
                if not kw:
                    continue
                if f" {kw} " in haystack or haystack.startswith(f"{kw} ") or haystack.endswith(f" {kw}"):
                    if best is None or len(kw) > best[0]:
                        best = (len(kw), preset)
        return best[1] if best else None


# ── Singleton ─────────────────────────────────────────────────────────────────

kb = KB()
