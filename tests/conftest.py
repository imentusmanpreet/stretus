from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"

for candidate in (ROOT, QUANT_ENGINE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
