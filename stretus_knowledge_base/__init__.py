"""
Stretus knowledge-base package — runtime signal evaluators only.

The structured KB (signal cards, stocks, timeframes, risk tiers) lives in
`app/kb/`. This package now contains just the Python signal-evaluator
registry consumed by the quant engine and the planner formula renderer.

Subpackages:
- `stretus_knowledge_base.stretus_kb` — RuleRegistry + signal evaluators.
"""

__all__ = ["stretus_kb"]
