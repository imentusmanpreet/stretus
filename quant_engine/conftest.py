"""
Make the quant_engine package importable from tests regardless of pytest's
chosen rootdir. The application runs with quant_engine/ as the working
directory (``from engine.runner import ...``); this mirrors that for tests.
"""
import os
import sys

_QUANT_ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _QUANT_ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _QUANT_ENGINE_ROOT)
