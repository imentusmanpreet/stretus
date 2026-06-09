"""Multi-asset execution support: instrument_metadata + crypto system_configs

Revision ID: a7b8c9d0e1f2
Revises: e5f6a7b8c9d0
Create Date: 2026-05-28 12:00:00.000000

Two changes, both additive and reversible:

1. ``ai_strategy.instrument_metadata`` is extended with three nullable columns
   so a single row can represent both equity and crypto instruments:

     - ``asset_class``       TEXT   (e.g. 'equity_cash', 'crypto_spot')
     - ``adapter_id``        TEXT   (e.g. 'upstox_rest', 'binance_rest')
     - ``adapter_symbol``    TEXT   (venue-native symbol)
     - ``qty_step_size``     NUMERIC(28,10)  (fractional crypto qty step)
     - ``min_notional``      NUMERIC(28,10)  (Binance min trade value, USDT)

   The legacy ``upstox_instrument_key`` column is preserved untouched so
   existing equity callers continue to work without a backfill. A backfill
   script can copy values into ``adapter_symbol`` later.

2. ``ref_data.system_configs`` is seeded with crypto-spot defaults so the
   ``ref_data_service`` lookup_instrument_defaults() function returns sensible
   values for Binance pairs without a per-instrument override:

     - crypto_spot.tick_size_default     = 0.01
     - crypto_spot.qty_step_default      = 0.00001
     - crypto_spot.min_notional_default  = 10        (USDT)
     - crypto_spot.upper_bound_pct       = 10        (10% soft band)
     - crypto_spot.lower_bound_pct       = 10

   Rows use ``ON CONFLICT (key) DO NOTHING`` so re-running the migration is
   safe and idempotent.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── system_configs seeds (idempotent) ─────────────────────────────────────────

_CRYPTO_SYSTEM_CONFIG_SEEDS: list[tuple[str, str, str]] = [
    (
        "crypto_spot.tick_size_default",
        "0.01",
        "Crypto spot price tick size fallback when no per-instrument value exists "
        "in ref_data.instruments (BTCUSDT-class default).",
    ),
    (
        "crypto_spot.qty_step_default",
        "0.00001",
        "Crypto spot quantity step fallback (5 decimal places). Overridden per "
        "instrument from ref_data.instruments.qty_step_size when present.",
    ),
    (
        "crypto_spot.min_notional_default",
        "10",
        "Crypto spot minimum trade value fallback in quote asset (USDT). Binance "
        "rejects orders below this notional.",
    ),
    (
        "crypto_spot.upper_bound_pct",
        "10",
        "Soft upper-price band as a percentage of LTP for the risk-manager "
        "circuit guard. Binance has no exchange-side circuit breaker.",
    ),
    (
        "crypto_spot.lower_bound_pct",
        "10",
        "Soft lower-price band as a percentage of LTP for the risk-manager "
        "circuit guard. Binance has no exchange-side circuit breaker.",
    ),
]


def upgrade() -> None:
    # ── 1. Extend instrument_metadata ─────────────────────────────────────────
    op.add_column(
        "instrument_metadata",
        sa.Column("asset_class", sa.Text(), nullable=True,
                  comment="Asset class (e.g. equity_cash, crypto_spot). "
                          "Mirrors ref_data.instruments.asset_class_id."),
        schema="ai_strategy",
    )
    op.add_column(
        "instrument_metadata",
        sa.Column("adapter_id", sa.Text(), nullable=True,
                  comment="Venue adapter id (e.g. upstox_rest, binance_rest). "
                          "Mirrors ref_data.instrument_adapter_mappings.adapter_id."),
        schema="ai_strategy",
    )
    op.add_column(
        "instrument_metadata",
        sa.Column("adapter_symbol", sa.Text(), nullable=True,
                  comment="Venue-native symbol (e.g. NSE_EQ|INE002A01018 for Upstox, "
                          "BTCUSDT for Binance)."),
        schema="ai_strategy",
    )
    op.add_column(
        "instrument_metadata",
        sa.Column("qty_step_size", sa.Numeric(28, 10), nullable=True,
                  comment="Fractional quantity step (crypto). Equity rows leave NULL."),
        schema="ai_strategy",
    )
    op.add_column(
        "instrument_metadata",
        sa.Column("min_notional", sa.Numeric(28, 10), nullable=True,
                  comment="Min trade value enforced by the venue (USDT for crypto)."),
        schema="ai_strategy",
    )

    # Backfill: existing equity rows have asset_class='equity_cash',
    # adapter_id='upstox_rest', adapter_symbol=upstox_instrument_key.
    op.execute(
        """
        UPDATE ai_strategy.instrument_metadata
        SET    asset_class    = COALESCE(asset_class, 'equity_cash'),
               adapter_id     = COALESCE(adapter_id,  'upstox_rest'),
               adapter_symbol = COALESCE(adapter_symbol, upstox_instrument_key)
        """
    )

    # ── 2. Seed crypto system_configs (idempotent) ────────────────────────────
    for key, value, description in _CRYPTO_SYSTEM_CONFIG_SEEDS:
        op.execute(
            sa.text(
                """
                INSERT INTO ref_data.system_configs (key, value, description)
                VALUES (:key, :value, :description)
                ON CONFLICT (key) DO NOTHING
                """
            ).bindparams(key=key, value=value, description=description)
        )


def downgrade() -> None:
    # 1. Remove crypto system_configs seeds.
    for key, _, _ in _CRYPTO_SYSTEM_CONFIG_SEEDS:
        op.execute(
            sa.text(
                "DELETE FROM ref_data.system_configs WHERE key = :key"
            ).bindparams(key=key)
        )

    # 2. Drop the new columns from instrument_metadata.
    op.drop_column("instrument_metadata", "min_notional",   schema="ai_strategy")
    op.drop_column("instrument_metadata", "qty_step_size",  schema="ai_strategy")
    op.drop_column("instrument_metadata", "adapter_symbol", schema="ai_strategy")
    op.drop_column("instrument_metadata", "adapter_id",     schema="ai_strategy")
    op.drop_column("instrument_metadata", "asset_class",    schema="ai_strategy")
