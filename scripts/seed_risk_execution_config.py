#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from app.db.session import AsyncSessionLocal
from app.services.execution.risk_execution_config_service import (
    ensure_default_risk_execution_config,
)


async def _main() -> None:
    async with AsyncSessionLocal() as db:
        try:
            snapshot = await ensure_default_risk_execution_config(db)
            await db.commit()
            print(
                "risk_execution_config ready | "
                f"scope={snapshot.config_scope} scope_id={snapshot.scope_id} updated_at={snapshot.updated_at}"
            )
        except Exception:
            await db.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(_main())
