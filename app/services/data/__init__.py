"""
app/services/data — lazy, streaming, cached OHLCV access for the dynamic-universe path.

This package holds the :class:`DataProvider` abstraction (a ``typing.Protocol`` so the
underlying store is swappable: an injected fetcher in Phase A, a Parquet-on-disk store
in Phase B, ClickHouse later — without touching callers). It is deliberately kept free
of heavy back-imports so the resolver and portfolio loop can depend on it without an
import cycle (§12.2). KB-free (Invariant 11).
"""
