"""
app/services/universe — KB-free dynamic-universe resolution (the direct StrategySpec path).

Turns a :class:`app.strategy.spec.UniverseSpec` rule into a ranked, eligible, content-hashed
set of symbols at a point in time (a *snapshot*). It ports the proven async patterns of
``app/services/discovery/scanner.py`` — concurrent fetch, point-in-time condition evaluation,
per-symbol failure tolerance, metric pre-computation — WITHOUT the ``kb.stocks`` dependency
the scanner has (Invariant 11). The candidate pool comes from the universe-source registry
(:mod:`sources`), not the knowledge base.

Public surface:
  * :func:`resolver.resolve_universe` — the pipeline (source→screen→eligibility→rank→take-N→
    breadth→hash).
  * :class:`types.ResolvedUniverse` — the persisted, replayable snapshot shape.
"""
