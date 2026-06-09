"""
Async helpers for the vertical ref_data.risk_execution_configs table.

Storage format:
  keys       -> field name for global values like "max_trades", or
                composite storage key like "strategy:<uuid>.stop_loss_pct"
  values     -> text payload (numeric settings stored as float-like strings;
                descriptive settings remain plain text)
  created_at -> first insert timestamp
  updated_at -> last upsert timestamp
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "global"
SESSION_SCOPE = "session"
STRATEGY_SCOPE = "strategy"
GLOBAL_SCOPE_ID = "global-default"
VALID_SCOPES = {GLOBAL_SCOPE, SESSION_SCOPE, STRATEGY_SCOPE}

# Default display labels for the `trading_window` summary field, per market.
# Used by build_risk_and_execution_from_builder() when the user didn't set an
# explicit entry_window. Indian stocks reflect the NSE cash session; crypto is
# labelled "24/7" because Binance has no session boundary. Unknown markets fall
# through to whatever the seed / risk_cfg provided.
_TRADING_WINDOW_BY_MARKET: dict[str, str] = {
    "indian_stocks":  "09:15 - 15:30",
    "indian_indices": "09:15 - 15:30",
    "crypto":         "24/7",
}


def _default_trading_window_label(market: Optional[str]) -> Optional[str]:
    return _TRADING_WINDOW_BY_MARKET.get((market or "").lower().strip())

def _seed_from_default_tier() -> dict[str, Any]:
    # Pull risk numbers from risk_tiers.yaml's `default` tier so the seed
    # never drifts from the tier baseline. Non-risk fields (execution mode,
    # trading window, etc.) stay hardcoded — they have no tier counterpart.
    from app.kb import kb

    tier = kb.get_risk_tier("default")
    return {
        "max_trades": 2,
        "risk_reward": 2.5,
        "daily_loss_cap": float(tier.daily_loss_cap_pct),
        "execution_mode": "Backtest",
        "per_trade_risk": float(tier.per_trade_risk_pct),
        "trading_window": "9:15 - 15:30",
        "position_sizing": "Risk based",
        "risk_validation": "system risk guardials",
        "stop_loss_pct": float(tier.base_sl_pct),
        "take_profit_pct": float(tier.base_tp_pct),
        "minimum_trade_value": float(tier.min_trade_value),
    }


DEFAULT_SEED_VALUES: dict[str, Any] = _seed_from_default_tier()

_CONFIG_FIELDS = tuple(DEFAULT_SEED_VALUES.keys())
_STRING_FIELDS = {
    "execution_mode",
    "trading_window",
    "position_sizing",
    "risk_validation",
}
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass
class RiskExecutionConfigSnapshot:
    config_scope: str
    scope_id: str
    session_id: Optional[str]
    strategy_id: Optional[str]
    max_trades: int
    risk_reward: Optional[float]
    daily_loss_cap: float
    execution_mode: str
    per_trade_risk: float
    trading_window: str
    position_sizing: str
    risk_validation: str
    stop_loss_pct: float
    take_profit_pct: float
    minimum_trade_value: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def daily_loss_cap_pct(self) -> float:
        return self.daily_loss_cap

    @property
    def per_trade_risk_pct(self) -> float:
        return self.per_trade_risk

    @property
    def min_trade_value(self) -> float:
        return self.minimum_trade_value

    @property
    def max_trades_per_day(self) -> int:
        return self.max_trades

    def to_builder_context(self) -> dict[str, Any]:
        return {
            "max_trades": self.max_trades,
            "risk_reward": self.risk_reward,
            "daily_loss_cap": self.daily_loss_cap,
            "execution_mode": self.execution_mode,
            "per_trade_risk": self.per_trade_risk,
            "trading_window": self.trading_window,
            "position_sizing": self.position_sizing,
            "risk_validation": self.risk_validation,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "minimum_trade_value": self.minimum_trade_value,
        }


def _normalize_optional_uuid(value: str | uuid.UUID | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolved_scope_id(config_scope: str, scope_id: str | uuid.UUID | None) -> str:
    if config_scope == GLOBAL_SCOPE:
        return GLOBAL_SCOPE_ID
    resolved = _normalize_optional_uuid(scope_id)
    if not resolved:
        raise ValueError(f"scope_id is required for config_scope='{config_scope}'")
    return resolved


def _scope_prefix(config_scope: str, scope_id: str) -> str:
    if config_scope == GLOBAL_SCOPE:
        return ""
    return f"{config_scope}:{scope_id}"


def _storage_key(config_scope: str, scope_id: str, field_name: str) -> str:
    prefix = _scope_prefix(config_scope, scope_id)
    return field_name if not prefix else f"{prefix}.{field_name}"


def _parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    match = _FLOAT_RE.search(str(value).strip())
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _serialize_db_value(field_name: str, value: Any) -> str:
    if field_name in _STRING_FIELDS:
        return str(value)

    numeric_value = _parse_numeric_value(value)
    if numeric_value is None:
        raise ValueError(f"Expected numeric value for '{field_name}', got {value!r}")

    return str(float(numeric_value))


def _parse_db_value(field_name: str, raw_value: Any) -> Any:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if field_name in _STRING_FIELDS:
        return value

    numeric_value = _parse_numeric_value(value)
    if numeric_value is None:
        return None

    if field_name == "max_trades":
        return int(numeric_value)

    return float(numeric_value)


def _build_snapshot(
    *,
    config_scope: str,
    scope_id: str,
    values: dict[str, Any],
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> RiskExecutionConfigSnapshot:
    merged = dict(DEFAULT_SEED_VALUES)
    merged.update(values)
    return RiskExecutionConfigSnapshot(
        config_scope=config_scope,
        scope_id=scope_id,
        session_id=scope_id if config_scope == SESSION_SCOPE else None,
        strategy_id=scope_id if config_scope == STRATEGY_SCOPE else None,
        max_trades=int(merged["max_trades"]),
        risk_reward=_parse_numeric_value(merged.get("risk_reward")),
        daily_loss_cap=float(merged["daily_loss_cap"]),
        execution_mode=str(merged["execution_mode"]),
        per_trade_risk=float(merged["per_trade_risk"]),
        trading_window=str(merged["trading_window"]),
        position_sizing=str(merged["position_sizing"]),
        risk_validation=str(merged["risk_validation"]),
        stop_loss_pct=float(merged["stop_loss_pct"]),
        take_profit_pct=float(merged["take_profit_pct"]),
        minimum_trade_value=float(merged["minimum_trade_value"]),
        created_at=created_at,
        updated_at=updated_at,
    )


async def _lookup_strategy_session_id(
    db: AsyncSession,
    strategy_id: str,
) -> Optional[str]:
    result = await db.execute(
        text(
            """
            SELECT session_id
            FROM ai_strategy.strategies
            WHERE id = :strategy_id
            LIMIT 1
            """
        ),
        {"strategy_id": strategy_id},
    )
    row = result.fetchone()
    return str(row[0]) if row and row[0] else None


async def _fetch_scope_values(
    db: AsyncSession,
    *,
    config_scope: str,
    scope_id: str,
) -> tuple[dict[str, Any], Optional[datetime], Optional[datetime]]:
    stmt = text(
        """
        SELECT "keys", "values", created_at, updated_at
        FROM ref_data.risk_execution_configs
        WHERE "keys" IN :storage_keys
        """
    ).bindparams(bindparam("storage_keys", expanding=True))
    storage_keys = [
        _storage_key(config_scope, scope_id, field_name)
        for field_name in _CONFIG_FIELDS
    ]
    result = await db.execute(stmt, {"storage_keys": storage_keys})
    rows = result.mappings().all()
    if not rows:
        return {}, None, None

    values: dict[str, Any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    for row in rows:
        storage_key = str(row["keys"])
        field_name = storage_key.rsplit(".", 1)[-1]
        if field_name not in _CONFIG_FIELDS:
            continue
        values[field_name] = _parse_db_value(field_name, row["values"])
        row_created = row.get("created_at")
        row_updated = row.get("updated_at")
        if created_at is None or (row_created is not None and row_created < created_at):
            created_at = row_created
        if updated_at is None or (row_updated is not None and row_updated > updated_at):
            updated_at = row_updated

    return values, created_at, updated_at


async def _fetch_snapshot_by_scope(
    db: AsyncSession,
    *,
    config_scope: str,
    scope_id: str,
) -> Optional[RiskExecutionConfigSnapshot]:
    values, created_at, updated_at = await _fetch_scope_values(
        db,
        config_scope=config_scope,
        scope_id=scope_id,
    )
    if not values:
        return None
    return _build_snapshot(
        config_scope=config_scope,
        scope_id=scope_id,
        values=values,
        created_at=created_at,
        updated_at=updated_at,
    )


async def ensure_default_risk_execution_config(
    db: AsyncSession,
) -> RiskExecutionConfigSnapshot:
    scope_id = GLOBAL_SCOPE_ID
    for field_name, value in DEFAULT_SEED_VALUES.items():
        await db.execute(
            text(
                """
                INSERT INTO ref_data.risk_execution_configs ("keys", "values")
                VALUES (:key, :value)
                ON CONFLICT ("keys") DO NOTHING
                """
            ),
            {
                "key": _storage_key(GLOBAL_SCOPE, scope_id, field_name),
                "value": _serialize_db_value(field_name, value),
            },
        )

    snapshot = await _fetch_snapshot_by_scope(
        db,
        config_scope=GLOBAL_SCOPE,
        scope_id=scope_id,
    )
    if snapshot is None:
        raise RuntimeError("Global risk/execution configuration could not be initialized.")
    return snapshot


async def resolve_active_risk_execution_config(
    db: AsyncSession,
    *,
    session_id: str | uuid.UUID | None = None,
    strategy_id: str | uuid.UUID | None = None,
) -> RiskExecutionConfigSnapshot:
    del session_id, strategy_id
    return await ensure_default_risk_execution_config(db)


async def upsert_risk_execution_config(
    db: AsyncSession,
    *,
    config_scope: str,
    scope_id: str | uuid.UUID | None,
    session_id: str | uuid.UUID | None = None,
    strategy_id: str | uuid.UUID | None = None,
    max_trades: int,
    risk_reward: float | str | None,
    daily_loss_cap: float,
    execution_mode: str,
    per_trade_risk: float,
    trading_window: str,
    position_sizing: str,
    risk_validation: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    minimum_trade_value: float,
) -> RiskExecutionConfigSnapshot:
    del session_id, strategy_id

    if config_scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported config_scope '{config_scope}'")

    if config_scope != GLOBAL_SCOPE:
        logger.info(
            "risk_execution_config upsert skipped for scope=%s | using global postgres defaults",
            config_scope,
        )
        return await ensure_default_risk_execution_config(db)

    resolved_scope_id = _resolved_scope_id(config_scope, scope_id)
    values = {
        "max_trades": int(max_trades),
        "risk_reward": _parse_numeric_value(risk_reward),
        "daily_loss_cap": float(daily_loss_cap),
        "execution_mode": str(execution_mode),
        "per_trade_risk": float(per_trade_risk),
        "trading_window": str(trading_window),
        "position_sizing": str(position_sizing),
        "risk_validation": str(risk_validation),
        "stop_loss_pct": float(stop_loss_pct),
        "take_profit_pct": float(take_profit_pct),
        "minimum_trade_value": float(minimum_trade_value),
    }

    for field_name, value in values.items():
        if value is None:
            continue
        await db.execute(
            text(
                """
                INSERT INTO ref_data.risk_execution_configs ("keys", "values")
                VALUES (:key, :value)
                ON CONFLICT ("keys") DO UPDATE
                SET
                    "values" = EXCLUDED."values",
                    updated_at = now()
                """
            ),
            {
                "key": _storage_key(config_scope, resolved_scope_id, field_name),
                "value": _serialize_db_value(field_name, value),
            },
        )

    snapshot = await _fetch_snapshot_by_scope(
        db,
        config_scope=config_scope,
        scope_id=resolved_scope_id,
    )
    if snapshot is None:
        raise RuntimeError("Risk/execution configuration upsert did not persist any rows.")

    logger.info(
        "risk_execution_config upserted | scope=%s scope_id=%s updated_at=%s",
        snapshot.config_scope,
        snapshot.scope_id,
        snapshot.updated_at,
    )
    return snapshot


def _derive_max_trades_from_builder(builder: Any) -> int:
    """Parse max trades per day from builder objective / max_trade label."""
    objective = str(getattr(builder, "objective", None) or "positional").lower()
    if objective != "intraday":
        return 0
    max_trade_str = str(getattr(builder, "max_trade", None) or "")
    match = re.search(r"(\d+)\s*trade", max_trade_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    plain_match = re.search(r"^\s*(\d+)\s*$", max_trade_str)
    if plain_match:
        return int(plain_match.group(1))
    return 0


def build_risk_execution_values_from_builder(
    builder: Any,
    base_snapshot: RiskExecutionConfigSnapshot | None = None,
) -> dict[str, Any]:
    """Merge builder RMS + risk_execution_config over DB defaults (builder wins)."""
    base = base_snapshot or _build_snapshot(
        config_scope=GLOBAL_SCOPE,
        scope_id=GLOBAL_SCOPE_ID,
        values=dict(DEFAULT_SEED_VALUES),
    )
    risk_cfg = dict(getattr(builder, "risk_execution_config", None) or {})

    stop_loss_pct = (
        float(builder.stop_loss)
        if getattr(builder, "stop_loss", None) is not None
        else float(risk_cfg.get("stop_loss_pct", base.stop_loss_pct))
    )
    take_profit_pct = (
        float(builder.take_profit)
        if getattr(builder, "take_profit", None) is not None
        else float(risk_cfg.get("take_profit_pct", base.take_profit_pct))
    )
    daily_loss_cap = (
        float(builder.daily_loss_cap)
        if getattr(builder, "daily_loss_cap", None) is not None
        else float(risk_cfg.get("daily_loss_cap", base.daily_loss_cap))
    )
    max_trades = int(risk_cfg.get("max_trades", base.max_trades))
    derived = _derive_max_trades_from_builder(builder)
    if derived > 0:
        max_trades = derived
    elif getattr(builder, "max_consecutive_losses", None) is not None:
        # Intraday circuit breaker often implies low trade count
        max_trades = max(max_trades, int(builder.max_consecutive_losses))

    # Trading window display string. Priority:
    #   1. Explicit builder.entry_window_start/_end — these are populated only
    #      when the user (or the agent on the user's behalf) typed a custom
    #      window; see builder.py where every user-supplied trading_window
    #      writes BOTH the structured bounds AND the display string.
    #   2. Asset-class default derived from builder.market. This overrides
    #      whatever risk_cfg["trading_window"] holds, because the seed and the
    #      DB-persisted snapshot both default to the NSE label "9:15 - 15:30"
    #      regardless of asset class — without this override, a crypto strategy
    #      would inherit that stale label.
    #   3. risk_cfg / base seed only when the market is unknown.
    ew_start = getattr(builder, "entry_window_start", None)
    ew_end = getattr(builder, "entry_window_end", None)
    market_label = _default_trading_window_label(getattr(builder, "market", None))
    if ew_start and ew_end:
        trading_window = f"{ew_start} - {ew_end}"
    elif market_label is not None:
        trading_window = market_label
    else:
        trading_window = str(risk_cfg.get("trading_window", base.trading_window))

    position_sizing = str(risk_cfg.get("position_sizing", base.position_sizing))
    ps_mode = getattr(builder, "position_sizing_mode", None)
    if ps_mode:
        position_sizing = str(ps_mode).replace("_", " ").title()

    # Per-trade risk: user risk_cfg → DB global → experience tier (risk_tiers.yaml)
    per_trade_risk = float(risk_cfg.get("per_trade_risk", base.per_trade_risk))
    if "per_trade_risk" not in risk_cfg:
        try:
            from app.kb.compat import get_experience_risk

            tier_risk = get_experience_risk(getattr(builder, "experience", None))
            per_trade_risk = float(tier_risk.get("per_trade_risk_pct", per_trade_risk))
        except Exception:
            pass

    return compose_risk_execution_values(
        max_trades=max_trades,
        daily_loss_cap=daily_loss_cap,
        execution_mode=str(risk_cfg.get("execution_mode", base.execution_mode)),
        per_trade_risk=per_trade_risk,
        trading_window=trading_window,
        position_sizing=position_sizing,
        risk_validation=str(risk_cfg.get("risk_validation", base.risk_validation)),
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        minimum_trade_value=float(
            risk_cfg.get("minimum_trade_value", base.minimum_trade_value)
        ),
    )


def sync_builder_risk_from_state(builder: Any) -> None:
    """Keep builder attrs and risk_execution_config in sync (draft root + RMS dict)."""
    cfg = dict(getattr(builder, "risk_execution_config", None) or {})

    if getattr(builder, "stop_loss", None) is not None:
        cfg["stop_loss_pct"] = float(builder.stop_loss)
    elif cfg.get("stop_loss_pct") is not None:
        builder.stop_loss = float(cfg["stop_loss_pct"])

    if getattr(builder, "take_profit", None) is not None:
        cfg["take_profit_pct"] = float(builder.take_profit)
    elif cfg.get("take_profit_pct") is not None:
        builder.take_profit = float(cfg["take_profit_pct"])

    # Preserve any already-recorded source label. NEVER promote a builder
    # attribute to source="user" just because it has a non-None value — that
    # used to mislabel seed defaults (DEFAULT_SEED_VALUES) as user-given and
    # silently violated the "RMS must be user-given" rule. Unlabeled values
    # are tagged "system_default" so downstream layers (planner, validator,
    # chat-layer gate) can detect them and either prompt the user or refuse
    # to assemble until the user confirms.
    sources = dict(cfg.get("rms_sources") or {})
    # Every RMS field we know about — if a value is present but no source was
    # ever set, tag it as system_default. This catches risk_reward, daily_loss_cap,
    # max_trades, per_trade_risk, etc. that get seeded from DEFAULT_SEED_VALUES.
    _RMS_FIELDS_TO_TAG = (
        "stop_loss_pct", "take_profit_pct", "risk_reward",
        "daily_loss_cap", "max_trades", "per_trade_risk",
        "trading_window", "position_sizing", "minimum_trade_value",
        "execution_mode", "risk_validation",
    )
    for fld in _RMS_FIELDS_TO_TAG:
        if cfg.get(fld) is not None:
            sources.setdefault(fld, "system_default")
    if sources:
        cfg["rms_sources"] = sources

    builder.set_risk_execution_config(cfg)


def build_risk_and_execution_from_builder(
    builder: Any,
    base_snapshot: RiskExecutionConfigSnapshot | None = None,
) -> dict[str, Any]:
    """API/draft-aligned risk block — always derived from current builder state."""
    values = build_risk_execution_values_from_builder(builder, base_snapshot)
    return build_risk_execution_response_from_values(values)


def compose_risk_execution_values(
    *,
    max_trades: int,
    daily_loss_cap: float,
    execution_mode: str,
    per_trade_risk: float,
    trading_window: str,
    position_sizing: str,
    risk_validation: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    minimum_trade_value: float,
) -> dict[str, Any]:
    stop_loss_value = float(stop_loss_pct)
    take_profit_value = float(take_profit_pct)
    risk_reward = None
    if stop_loss_value > 0:
        risk_reward = round(take_profit_value / stop_loss_value, 2)

    return {
        "max_trades": int(max_trades),
        "risk_reward": risk_reward,
        "daily_loss_cap": float(daily_loss_cap),
        "execution_mode": str(execution_mode),
        "per_trade_risk": float(per_trade_risk),
        "trading_window": str(trading_window),
        "position_sizing": str(position_sizing),
        "risk_validation": str(risk_validation),
        "stop_loss_pct": stop_loss_value,
        "take_profit_pct": take_profit_value,
        "minimum_trade_value": float(minimum_trade_value),
    }


def build_risk_execution_response(
    snapshot: RiskExecutionConfigSnapshot,
) -> dict[str, Any]:
    return {
        "max_trades": float(snapshot.max_trades),
        "risk_reward": (
            float(snapshot.risk_reward)
            if snapshot.risk_reward is not None
            else None
        ),
        "stop_loss_pct": float(snapshot.stop_loss_pct),
        "daily_loss_cap": float(snapshot.daily_loss_cap),
        "execution_mode": str(snapshot.execution_mode),
        "per_trade_risk": float(snapshot.per_trade_risk),
        "trading_window": str(snapshot.trading_window),
        "position_sizing": str(snapshot.position_sizing),
        "risk_validation": str(snapshot.risk_validation),
        "take_profit_pct": float(snapshot.take_profit_pct),
        "minimum_trade_value": float(snapshot.minimum_trade_value),
    }


def build_risk_execution_response_from_values(
    values: dict[str, Any],
) -> dict[str, Any]:
    snapshot = _build_snapshot(
        config_scope=GLOBAL_SCOPE,
        scope_id=GLOBAL_SCOPE_ID,
        values=values,
    )
    return build_risk_execution_response(snapshot)
