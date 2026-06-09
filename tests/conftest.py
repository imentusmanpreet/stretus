from __future__ import annotations

import sys
from pathlib import Path

# Generated protobuf stubs may target a newer runtime than the local venv.
try:
    import google.protobuf.runtime_version as _pb_runtime

    def _noop_protobuf_runtime_check(*_args: object, **_kwargs: object) -> None:
        return None

    _pb_runtime.ValidateProtobufRuntimeVersion = _noop_protobuf_runtime_check  # type: ignore[method-assign]
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"

for candidate in (ROOT, QUANT_ENGINE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
