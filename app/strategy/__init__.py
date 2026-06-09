"""
app/strategy/ — the direct chat → LLM → strict StrategySpec → backtest path.

This package replaces the knowledge-base-driven planner. The LLM emits a strict
``StrategySpec`` (formula mode, in the quant engine's own grammar); a single
fail-closed validator checks it against the engine's parser; the spec is then
rendered to YAML and handed to the same backtest engine the old path used.

The quant engine (``quant_engine/engine``) is the SINGLE source of truth for the
condition grammar and the strategy contract — never a parallel rulebook. All
access to the engine goes through :mod:`app.strategy.engine_bridge`.
"""
