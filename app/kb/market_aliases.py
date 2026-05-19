"""
app/kb/market_aliases.py — Single source of truth for market/benchmark symbol
aliases.

Import this wherever you need to normalise a user-typed benchmark name into
a canonical symbol string.  Having a single module prevents the silent drift
that occurs when the same dict is duplicated in the extractor, normalizer, and
signal composer.

To add a new index or benchmark: add one entry here — nothing else changes.
"""
from __future__ import annotations

# Keys are lower-cased user phrasings; values are canonical uppercase symbols.
BENCHMARK_ALIASES: dict[str, str] = {
    # Nifty family
    "nifty":                "NIFTY50",
    "nifty50":              "NIFTY50",
    "nifty 50":             "NIFTY50",
    "nifty500":             "NIFTY500",
    "nifty 500":            "NIFTY500",
    "nifty100":             "NIFTY100",
    "nifty 100":            "NIFTY100",
    "nifty next 50":        "NIFTY_NEXT50",
    "nifty midcap":         "NIFTY_MIDCAP_100",
    "nifty midcap 100":     "NIFTY_MIDCAP_100",
    "midcap":               "NIFTY_MIDCAP_100",
    "nifty smallcap":       "NIFTY_SMALLCAP_100",
    "smallcap":             "NIFTY_SMALLCAP_100",
    # BankNifty
    "banknifty":            "BANKNIFTY",
    "bank nifty":           "BANKNIFTY",
    "niftybank":            "BANKNIFTY",
    "nifty bank":           "BANKNIFTY",
    # Fin / sector
    "finnifty":             "FINNIFTY",
    "fin nifty":            "FINNIFTY",
    "nifty it":             "NIFTY_IT",
    "it index":             "NIFTY_IT",
    "nifty pharma":         "NIFTY_PHARMA",
    "pharma index":         "NIFTY_PHARMA",
    "nifty fmcg":           "NIFTY_FMCG",
    "nifty metal":          "NIFTY_METAL",
    "nifty auto":           "NIFTY_AUTO",
    "nifty energy":         "NIFTY_ENERGY",
    "nifty psu bank":       "NIFTY_PSU_BANK",
    "nifty realty":         "NIFTY_REALTY",
    "nifty media":          "NIFTY_MEDIA",
    # Sensex
    "sensex":               "SENSEX",
    "bse sensex":           "SENSEX",
    "bse 30":               "SENSEX",
    # Generic fallbacks — user said "the index" / "the market"
    "index":                "NIFTY50",
    "the index":            "NIFTY50",
    "market":               "NIFTY50",
    "the market":           "NIFTY50",
    "benchmark":            "NIFTY50",
    "broad market":         "NIFTY50",
}


def normalise_benchmark(raw: str) -> str:
    """Return the canonical benchmark symbol for any raw user-typed string.

    Falls back to ``raw.strip().upper().replace(" ", "")`` when the input is
    not in the alias table (e.g. the user typed an exact ticker already).

    Examples
    --------
    >>> normalise_benchmark("bank nifty")
    'BANKNIFTY'
    >>> normalise_benchmark("RELIANCE")
    'RELIANCE'
    """
    key = raw.strip().lower()
    return BENCHMARK_ALIASES.get(key, raw.strip().upper().replace(" ", ""))
