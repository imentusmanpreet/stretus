"""Signal rule registry for the Stretus knowledge base."""
from __future__ import annotations


class RuleRegistry:
    _registry: dict = {}

    # Stores formula templates for signals that declare them.
    # Template syntax: Python str.format() with param names as keys.
    # Example: "RSI({window}) < {threshold}"
    # The quant engine renders these into condition strings at strategy build time.
    _formulas: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, formula: str | None = None):
        """
        Decorator that registers a signal evaluation function.

        Args:
            name:    Unique signal identifier (e.g. "rsi_oversold").
            formula: Optional condition formula template using Python str.format()
                     syntax, e.g. "RSI({window}) < {threshold}".
                     Param names in the template must match the function's
                     keyword argument names.
                     Signals without a formula are excluded from automatic
                     formula generation and cannot be backtested via the
                     formula-string path (they can still be used via the
                     direct RuleRegistry.evaluate() path planned in Option B).
        """
        def wrapper(fn):
            cls._registry[name] = fn
            if formula is not None:
                cls._formulas[name] = formula
            return fn
        return wrapper

    @classmethod
    def get_formula(cls, name: str) -> str | None:
        """
        Return the formula template for a signal, or None if not declared.

        The template is rendered by the strategy builder by calling
        template.format(**params) where params are the resolved signal params.
        """
        return cls._formulas.get(name)

    @classmethod
    def evaluate(cls, rule, df):
        rule_name = rule["name"]
        if rule_name not in cls._registry:
            raise ValueError(f"Unknown signal: {rule_name}")
        params = dict(rule.get("params", rule.get("parameters", {})))
        # param aliases
        if "short_period" in params and "window_fast" not in params:
            params["window_fast"] = params.pop("short_period")
        if "long_period" in params and "window_slow" not in params:
            params["window_slow"] = params.pop("long_period")
        if "period" in params:
            params.setdefault("window", params["period"])
            del params["period"]
        # All KB signal implementations expect PascalCase column names ("Close",
        # "High", etc.).  The live-eval pipeline produces lowercase columns
        # ("close", "high") for consistency with engine/conditions.py.  Rename
        # only the known OHLCV columns; any extra columns (e.g. indicator
        # columns added by a caller) are left as-is.
        _OHLCV_TO_PASCAL = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }
        if not df.empty and any(c in df.columns for c in _OHLCV_TO_PASCAL):
            df = df.rename(columns=_OHLCV_TO_PASCAL)
        return cls._registry[rule_name](df, **params)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._registry.keys())
