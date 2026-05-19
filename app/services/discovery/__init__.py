"""
app/services/discovery
──────────────────────
Phase 9 — runtime universe scanner + tie-break framework for "dynamic"
strategies that don't pin a symbol up front.

Public API:

    from app.services.discovery import (
        DiscoveryConfig,            # KB schema (declared on a preset)
        ScanResult,                 # outcome of one scan
        TieBreakOption,             # one tie-break method offered to the user
        scan_universe,              # async — runs the scan
        apply_tie_break,            # picks one candidate per a method
    )
"""
from app.services.discovery.chat_integration import (
    DiscoveryStepResult,
    handle_pending_tie_break,
    maybe_dispatch_discovery,
)
from app.services.discovery.orchestrator import resolve_tie_break, run_discovery
from app.services.discovery.scanner import scan_universe
from app.services.discovery.tie_break import (
    TIE_BREAK_METHODS,
    apply_tie_break,
    available_tie_break_options,
    parse_user_tie_break_reply,
)
from app.services.discovery.types import (
    Candidate,
    DiscoveryConfig,
    DiscoveryStatus,
    ScanResult,
    TieBreakOption,
)

__all__ = [
    "Candidate",
    "DiscoveryConfig",
    "DiscoveryStatus",
    "DiscoveryStepResult",
    "ScanResult",
    "TIE_BREAK_METHODS",
    "TieBreakOption",
    "apply_tie_break",
    "available_tie_break_options",
    "handle_pending_tie_break",
    "maybe_dispatch_discovery",
    "parse_user_tie_break_reply",
    "resolve_tie_break",
    "run_discovery",
    "scan_universe",
]
