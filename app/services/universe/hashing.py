"""
app/services/universe/hashing.py — deterministic content hashing for reproducibility (§8).

Ports the SDL hashing STYLE (SHA-256 over canonical, sorted-key JSON) without importing the
SDL/planner types (Invariant 11). Two hashes:

  * :func:`rule_hash` — a stable fingerprint of the universe RULE (source/screen/rank/take/
    eligibility/breadth/refresh). Same rule ⇒ same ``rule_hash``, independent of when it runs.
  * :func:`snapshot_hash` — a fingerprint of one RESOLUTION: ``rule_hash`` + ``asof`` +
    the sorted resolved members. Same spec + same data window ⇒ identical snapshots ⇒ an
    identical, fully replayable result (Invariant 8).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _sha256_canonical(payload: Any) -> str:
    """SHA-256 of ``payload`` serialized as canonical (sorted-key, compact) JSON."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rule_hash(universe_dict: dict[str, Any]) -> str:
    """Stable hash of the universe RULE (pass ``UniverseSpec.model_dump(mode='json')``).

    Excludes nothing structural — the whole declared rule is the identity. Two specs that
    differ only in a screen threshold or rank order produce different hashes.
    """
    return _sha256_canonical(universe_dict)


def snapshot_hash(*, rule_hash: str, asof: str, members: Iterable[str]) -> str:
    """Deterministic hash of one resolution: rule + asof + sorted members (§7).

    Member ORDER does not affect the hash (we sort), so the snapshot identity is the SET
    that was active at ``asof`` under that rule — exactly what reproducibility needs.
    """
    payload = {
        "rule_hash": rule_hash,
        "asof": asof,
        "members": sorted(members),
    }
    return _sha256_canonical(payload)
