"""VerificationStatus enum — the three outcomes of a verification pass.

The status appears as `Verification: PASS / PASS WITH NOTES / BLOCKED`
in `.provenance.md` sidecars.

`BLOCKED` is the most distinctive of the three: it is nexus's "honest
partial failure" contract. When verification cannot complete (dead
links, paywalled sources, network errors), the pipeline does NOT abort
to a chat message — it writes a partial artifact with `BLOCKED` status
and records the specific failure in the sidecar's notes. The user gets
something actionable, not a runtime error and an empty outputs dir.
"""

from __future__ import annotations

from enum import Enum


class VerificationStatus(str, Enum):
    """Outcome of the verifier + reviewer pipeline pass.

    - `PASS`: all checks completed successfully.
    - `PASS_WITH_NOTES`: most checks passed; some skipped or partially
      completed (e.g., paywalled source where only abstract verification
      was possible). Recorded for transparency.
    - `BLOCKED`: at least one verification check could not be completed
      and the affected claim is preserved with a `BLOCKED` marker in
      the body. The output is partial-but-honest, not silently truncated.
    """

    PASS = "PASS"
    PASS_WITH_NOTES = "PASS WITH NOTES"
    BLOCKED = "BLOCKED"

    @classmethod
    def from_findings(
        cls,
        *,
        unresolved_claims: int,
        notes_only_issues: int,
    ) -> "VerificationStatus":
        """Derive status from finding counts.

        At least one unresolved claim → BLOCKED. Otherwise, any
        notes-only issues → PASS_WITH_NOTES. Else → PASS.
        """
        if unresolved_claims > 0:
            return cls.BLOCKED
        if notes_only_issues > 0:
            return cls.PASS_WITH_NOTES
        return cls.PASS


__all__ = ["VerificationStatus"]
