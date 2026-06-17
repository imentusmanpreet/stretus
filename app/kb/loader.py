"""
app/kb/loader.py — typed in-memory knowledge base.

The whole KB is read once on startup, validated by Pydantic, and held in memory
as a singleton. Reads after that are O(1) dict lookups — no I/O, no JSON parsing,
no embedding queries. Hot-reloadable via `kb.reload()` for admin endpoints.
"""
from __future__ import annotations

import csv
import difflib
import logging
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StockLookupResolution:
    """Result of resolving a user-entered stock query."""

    stock: Stock | None = None
    ambiguous_matches: tuple[Stock, ...] = ()
    # A single fuzzy "did you mean" candidate the caller should confirm with the
    # user rather than adopt silently. Left None for exact/prefix/alias/token
    # resolutions; only set by the last-resort fuzzy tier.
    suggestion: Stock | None = None

    @property
    def is_ambiguous(self) -> bool:
        return len(self.ambiguous_matches) > 1


def _strip_exchange_hint(value: str) -> str:
    text = str(value or "").strip()
    if ":" in text:
        _, text = text.split(":", 1)
    return re.sub(r"\.(?:NS|BO)$", "", text, flags=re.IGNORECASE).strip()


def _stock_key(value: str) -> str:
    """Compact stock names/symbols for exact and prefix comparisons."""
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


# Fuzzy stock-name matching is a last-resort "did you mean" tier: it runs only
# after every exact/prefix/alias/token lookup has missed, and its best hit is
# surfaced as a suggestion the user confirms — never auto-adopted. The cutoff is
# intentionally strict so a real, distinct symbol is not silently rewritten into
# a different one, and very short queries are skipped (too little signal).
_FUZZY_STOCK_CUTOFF = 0.82
_FUZZY_MIN_QUERY_LEN = 4


def _stock_tokens(value: str) -> tuple[str, ...]:
    """Split a name/symbol into normalized alphanumeric tokens (order kept)."""
    cleaned = _strip_exchange_hint(value)
    return tuple(tok.lower() for tok in re.split(r"[^0-9A-Za-z]+", cleaned) if tok)


def _stock_token_key(value: str) -> str:
    """Order-independent key: alphanumeric tokens sorted, then concatenated.

    "Vodafone Idea", "idea vodafone" and "ideavodafone" all collapse to the same
    key, so a reversed or space-variant word order still resolves. A single
    already-joined token sorts to itself, so this mainly rescues multi-word names
    whose tokens the user reordered.
    """
    return "".join(sorted(_stock_tokens(value)))


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
        stock = kb.lookup_stock("hdfc bank")
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
        for key in (
            "signals_by_role",
            "_auto_aliases",
            "_stock_exact_index",
            "_stock_prefix_keys",
            "_stock_token_index",
            "_stock_fuzzy_keys",
        ):
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

    @cached_property
    def _stock_exact_index(self) -> dict[str, str]:
        """Normalized exact stock names/symbols to canonical symbols."""
        out: dict[str, str] = {}
        for symbol, stock in self.stocks.items():
            bare = symbol.split(".", 1)[0]
            keys = {
                _stock_key(symbol),
                _stock_key(bare),
                _stock_key(stock.display_name),
            }
            for key in keys:
                if key:
                    out.setdefault(key, symbol)
        return out

    @cached_property
    def _stock_prefix_keys(self) -> dict[str, tuple[str, ...]]:
        """Canonical symbol to normalized strings that users naturally prefix."""
        out: dict[str, tuple[str, ...]] = {}
        for symbol, stock in self.stocks.items():
            bare = symbol.split(".", 1)[0]
            keys = {
                _stock_key(bare),
                _stock_key(stock.display_name),
            }
            out[symbol] = tuple(sorted(key for key in keys if key))
        return out

    @cached_property
    def _stock_token_index(self) -> dict[str, tuple[str, ...]]:
        """Order-independent token-set key → canonical symbols sharing it.

        Built from display names so a reordered multi-word query still maps to
        its stock. Consulted only as a fallback (see resolve_stock_query), and a
        key shared by more than one symbol yields ambiguity instead of a guess.
        """
        out: dict[str, list[str]] = {}
        for symbol, stock in self.stocks.items():
            key = _stock_token_key(stock.display_name)
            if not key:
                continue
            bucket = out.setdefault(key, [])
            if symbol not in bucket:
                bucket.append(symbol)
        return {key: tuple(symbols) for key, symbols in out.items()}

    @cached_property
    def _stock_fuzzy_keys(self) -> tuple[str, ...]:
        """All normalized exact keys, reused for difflib close-match suggestions."""
        return tuple(self._stock_exact_index.keys())

    def _fuzzy_stock_matches(self, normalized_query: str) -> list[Stock]:
        """Close-but-not-exact symbol/name matches for a normalized query.

        Returns at most a few candidates, best-first. Empty for short queries —
        too little signal to fuzzy-match safely.
        """
        if not normalized_query or len(normalized_query) < _FUZZY_MIN_QUERY_LEN:
            return []
        close_keys = difflib.get_close_matches(
            normalized_query, self._stock_fuzzy_keys, n=3, cutoff=_FUZZY_STOCK_CUTOFF
        )
        out: list[Stock] = []
        seen: set[str] = set()
        for key in close_keys:
            symbol = self._stock_exact_index.get(key)
            if not symbol or symbol in seen:
                continue
            stock = self.stocks.get(symbol)
            if stock is not None:
                seen.add(symbol)
                out.append(stock)
        return out

    # ── Public lookups ────────────────────────────────────────────────────────

    def prefix_stock_matches(self, query: str) -> list[Stock]:
        """Stocks whose symbol or display name starts with the query."""
        q = _stock_key(_strip_exchange_hint(query))
        if not q:
            return []

        matches: dict[str, Stock] = {}
        for symbol, keys in self._stock_prefix_keys.items():
            if any(key.startswith(q) for key in keys):
                stock = self.stocks.get(symbol)
                if stock is not None:
                    matches[symbol] = stock

        return sorted(matches.values(), key=lambda stock: stock.symbol)

    def resolve_stock_query(self, query: str) -> StockLookupResolution:
        """Resolve a user stock query without guessing on ambiguous prefixes.

        Exact symbol/name matches win first. If the query is only a shared
        prefix (for example "hdfc" or "icici"), the caller gets the matching
        choices and can ask the user to pick one.
        """
        if not query:
            return StockLookupResolution()

        raw = str(query).strip()
        if not raw:
            return StockLookupResolution()
        q_lower = raw.lower()
        q_key = _stock_key(_strip_exchange_hint(raw))

        # Direct symbol match (e.g. "HDFCBANK.NS")
        for symbol, stock in self.stocks.items():
            if symbol.lower() == q_lower:
                return StockLookupResolution(stock=stock)

        exact_symbol = self._stock_exact_index.get(q_key)
        if exact_symbol:
            stock = self.stocks.get(exact_symbol)
            if stock is not None:
                return StockLookupResolution(stock=stock)

        prefix_matches = self.prefix_stock_matches(raw)
        if len(prefix_matches) > 1:
            return StockLookupResolution(ambiguous_matches=tuple(prefix_matches))
        if len(prefix_matches) == 1:
            return StockLookupResolution(stock=prefix_matches[0])

        # Manual aliases are a fallback only after prefix ambiguity has been
        # ruled out, so broad aliases like "hdfc" cannot silently pick a stock.
        if q_lower in self.aliases:
            sym = self.aliases[q_lower]
            return StockLookupResolution(stock=self.stocks.get(sym))

        # Auto-derived aliases
        if q_lower in self._auto_aliases:
            return StockLookupResolution(stock=self.stocks.get(self._auto_aliases[q_lower]))

        # Order-independent token-set fallback. Rescues reordered / space-variant
        # multi-word names (e.g. "idea vodafone" → "Vodafone Idea") once the
        # precise tiers above have missed. More than one symbol on the same token
        # set is ambiguous rather than a silent guess.
        token_symbols = self._stock_token_index.get(_stock_token_key(raw))
        if token_symbols:
            token_stocks = [self.stocks[s] for s in token_symbols if s in self.stocks]
            if len(token_stocks) > 1:
                return StockLookupResolution(ambiguous_matches=tuple(token_stocks))
            if len(token_stocks) == 1:
                return StockLookupResolution(stock=token_stocks[0])

        # Fuzzy "did you mean" — last resort for typos. A single hit becomes a
        # suggestion the caller asks the user to confirm; multiple hits reuse the
        # ambiguous picker. Never auto-adopted by lookup_stock().
        fuzzy_matches = self._fuzzy_stock_matches(q_key)
        if len(fuzzy_matches) > 1:
            return StockLookupResolution(ambiguous_matches=tuple(fuzzy_matches))
        if len(fuzzy_matches) == 1:
            return StockLookupResolution(suggestion=fuzzy_matches[0])

        return StockLookupResolution()

    def lookup_stock(self, query: str) -> Stock | None:
        """Resolve a free-text query to a Stock. Returns None if unknown."""
        resolution = self.resolve_stock_query(query)
        if resolution.is_ambiguous:
            return None
        return resolution.stock

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
        """Scan free text for a preset keyword and return the best match.

        Scoring: for each preset, sum the lengths of *all* matching keywords in
        the text.  The preset with the highest total score wins.  Ties are
        broken by the count of matching keywords (more matches = better fit),
        then by the length of the single longest matching keyword.

        Why sum-of-lengths instead of longest-single-match:
        - A user saying "ORB strategy with structural SL at opening range low"
          produces a higher combined score for `orb_structural` (which has a
          long keyword covering that exact phrase) than for `orb` (which only
          has shorter keywords matching the first half of the sentence).
        - This makes every strategy variant self-detecting from its own keyword
          list without any hard-coded upgrade logic.
        """
        if not text:
            return None
        haystack = f" {text.strip().lower()} "

        # score_map: preset → (total_length, match_count, longest_match)
        score_map: dict[str, tuple[int, int, int]] = {}
        preset_map: dict[str, StrategyPreset] = {}

        for preset in self.presets.values():
            total_len = 0
            count = 0
            longest = 0
            for keyword in preset.keywords:
                kw = keyword.strip().lower()
                if not kw:
                    continue
                if (
                    f" {kw} " in haystack
                    or haystack.startswith(f"{kw} ")
                    or haystack.endswith(f" {kw}")
                ):
                    klen = len(kw)
                    total_len += klen
                    count += 1
                    if klen > longest:
                        longest = klen
            if count > 0:
                score_map[preset.name] = (total_len, count, longest)
                preset_map[preset.name] = preset

        if not score_map:
            return None

        best_name = max(
            score_map,
            key=lambda n: score_map[n],   # tuple comparison: total_len, count, longest
        )
        logger.debug(
            "kb|detect_preset|scores=%s|winner=%s",
            {n: score_map[n] for n in score_map},
            best_name,
        )
        return preset_map[best_name]

    def detect_primary_framework_in_text(self, text: str) -> StrategyPreset | None:
        """Like ``detect_preset_in_text`` but restricted to primary-framework
        presets (``is_primary_framework=True``).

        Use this when you need the *structural* strategy framework from the
        user's text and want to prevent filter/modifier presets (e.g.
        ``relative_strength``) from being promoted to primary-framework status
        just because their keywords appear in the sentence.

        Example: "ORB strategy with relative strength filter vs NIFTY"
          - ``detect_preset_in_text``       → might return ``relative_strength``
          - ``detect_primary_framework_in_text`` → returns ``orb_structural``
        """
        if not text:
            return None
        haystack = f" {text.strip().lower()} "

        score_map: dict[str, tuple[int, int, int]] = {}
        preset_map: dict[str, StrategyPreset] = {}

        for preset in self.presets.values():
            if not preset.is_primary_framework:
                continue
            total_len = 0
            count = 0
            longest = 0
            for keyword in preset.keywords:
                kw = keyword.strip().lower()
                if not kw:
                    continue
                if (
                    f" {kw} " in haystack
                    or haystack.startswith(f"{kw} ")
                    or haystack.endswith(f" {kw}")
                ):
                    klen = len(kw)
                    total_len += klen
                    count += 1
                    if klen > longest:
                        longest = klen
            if count > 0:
                score_map[preset.name] = (total_len, count, longest)
                preset_map[preset.name] = preset

        if not score_map:
            return None

        best_name = max(score_map, key=lambda n: score_map[n])
        logger.debug(
            "kb|detect_primary_framework|scores=%s|winner=%s",
            {n: score_map[n] for n in score_map},
            best_name,
        )
        return preset_map[best_name]


# ── Singleton ─────────────────────────────────────────────────────────────────

kb = KB()
