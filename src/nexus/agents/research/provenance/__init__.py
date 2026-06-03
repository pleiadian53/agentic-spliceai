"""Provenance accounting: report manifest, output validation, .provenance.md sidecars.

Public surface:
- `manifest` — generation metadata (legacy; per-run report bookkeeping)
- `output_validator` — heuristic validation of the final draft
- `ProvenanceSidecar` — accounting ledger for sources + verification status
- `VerificationStatus` — PASS / PASS WITH NOTES / BLOCKED enum
"""

from .sidecar import ProvenanceSidecar
from .status import VerificationStatus

__all__ = ["ProvenanceSidecar", "VerificationStatus"]
