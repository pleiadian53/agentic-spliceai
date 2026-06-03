"""`.provenance.md` sidecar — the accounting ledger that ships next to each output.

Modeled after Feynman's `<slug>.provenance.md` sidecar
(`~/work/feynman/prompts/deepresearch.md` lines 173-184, as of 2026-06-02).

The sidecar is NOT a Sources section. It is an accounting ledger that
commits to four things:

1. Source accounting — consulted vs. accepted vs. rejected. Negative
   space (rejected sources) is first-class; a naive agent reports only
   what it kept and silently drops the rest. The sidecar forces both
   halves on the record.
2. Verification status — PASS / PASS WITH NOTES / BLOCKED.
3. Plan reference — path to the plan artifact (which contains the
   task ledger + decision log).
4. Research files used — the slug-prefixed files the synthesis was
   built from. A skeptical reader can open these and trace any claim
   back to source.

A skeptical reader can open the brief, find a claim, follow the
citation number, open the research file, find the evidence-table row,
click the URL, and confirm the paper says what the brief says it
says. The chain is preserved on disk. Compare to a naive agent's
output where the chain ends at "the model said so".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .status import VerificationStatus


@dataclass
class ProvenanceSidecar:
    """The accounting ledger for one research-pipeline output.

    Write via `.write(path)` to emit the canonical `.provenance.md`
    Markdown format.
    """

    topic: str
    date: str  # ISO-8601, passed in by the caller (no Date.now() inside the dataclass)
    rounds: int = 1
    sources_consulted: list[str] = field(default_factory=list)
    sources_accepted: list[str] = field(default_factory=list)
    sources_rejected: list[str] = field(default_factory=list)
    verification: VerificationStatus = VerificationStatus.PASS
    plan_ref: str = ""
    research_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render to Markdown — the exact `.provenance.md` body."""
        lines = [
            f"# Provenance: {self.topic}",
            "",
            f"- **Date:** {self.date}",
            f"- **Rounds:** {self.rounds}",
            f"- **Sources consulted:** {len(self.sources_consulted)}",
        ]
        if self.sources_consulted:
            lines.extend(f"  - {u}" for u in self.sources_consulted)
        lines.append(f"- **Sources accepted:** {len(self.sources_accepted)}")
        if self.sources_accepted:
            lines.extend(f"  - {u}" for u in self.sources_accepted)
        lines.append(
            f"- **Sources rejected:** {len(self.sources_rejected)}"
        )
        if self.sources_rejected:
            lines.extend(f"  - {u}" for u in self.sources_rejected)
        else:
            lines.append("  - (none)")
        lines.append(f"- **Verification:** {self.verification.value}")
        if self.plan_ref:
            lines.append(f"- **Plan:** {self.plan_ref}")
        if self.research_files:
            lines.append(f"- **Research files:**")
            lines.extend(f"  - {f}" for f in self.research_files)
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write the sidecar to disk and return the resolved path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(), encoding="utf-8")
        return out

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty list = OK).

        Used by `examples/nexus/03_provenance_sidecar.py` to assert
        schema integrity.
        """
        problems: list[str] = []
        if not self.topic.strip():
            problems.append("topic is empty")
        if not self.date.strip():
            problems.append("date is empty")
        if self.rounds < 1:
            problems.append(f"rounds must be >= 1, got {self.rounds}")
        # Accounting must add up: accepted + rejected ⊆ consulted (set semantics)
        consulted = set(self.sources_consulted)
        accepted = set(self.sources_accepted)
        rejected = set(self.sources_rejected)
        unaccounted_accepted = accepted - consulted
        unaccounted_rejected = rejected - consulted
        if unaccounted_accepted:
            problems.append(
                f"accepted sources not in consulted: {sorted(unaccounted_accepted)}"
            )
        if unaccounted_rejected:
            problems.append(
                f"rejected sources not in consulted: {sorted(unaccounted_rejected)}"
            )
        # BLOCKED status should be accompanied by at least one note explaining why
        if self.verification == VerificationStatus.BLOCKED and not self.notes:
            problems.append("BLOCKED verification but no notes explaining the blocker")
        return problems


__all__ = ["ProvenanceSidecar"]
