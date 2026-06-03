"""Verify the .provenance.md sidecar's accounting contract.

The sidecar is NOT a Sources section — it is an accounting ledger
that records sources consulted + accepted + REJECTED (negative space
is first-class) plus the verification status. A naive agent silently
drops rejected sources; the sidecar forces both halves on the record.

This script asserts:

1. ProvenanceSidecar.validate() catches accounting inconsistencies
   (accepted ⊄ consulted; rejected ⊄ consulted; BLOCKED without notes).
2. ProvenanceSidecar.write() emits Markdown with every required ledger
   field: topic, date, rounds, sources consulted/accepted/rejected
   with counts, verification status, plan ref, research files.
3. The rendered Markdown round-trips: every accepted/rejected source
   appears in the output verbatim.
4. VerificationStatus.from_findings() derives correctly from finding counts.

Run:
    mamba run -n agentic-spliceai python examples/nexus/03_provenance_sidecar.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from nexus.agents.research.provenance import ProvenanceSidecar, VerificationStatus


def test_status_from_findings() -> None:
    assert VerificationStatus.from_findings(
        unresolved_claims=0, notes_only_issues=0
    ) == VerificationStatus.PASS
    assert VerificationStatus.from_findings(
        unresolved_claims=0, notes_only_issues=2
    ) == VerificationStatus.PASS_WITH_NOTES
    assert VerificationStatus.from_findings(
        unresolved_claims=1, notes_only_issues=0
    ) == VerificationStatus.BLOCKED
    assert VerificationStatus.from_findings(
        unresolved_claims=3, notes_only_issues=5
    ) == VerificationStatus.BLOCKED, "BLOCKED dominates PASS_WITH_NOTES"
    print("[OK] VerificationStatus.from_findings dispatches correctly")


def test_validate_catches_accounting_errors() -> None:
    # Case 1: accepted source not in consulted
    bad1 = ProvenanceSidecar(
        topic="test",
        date="2026-06-02",
        sources_consulted=["https://example.com/a"],
        sources_accepted=["https://example.com/a", "https://example.com/b"],
    )
    problems = bad1.validate()
    assert any("accepted" in p for p in problems), problems

    # Case 2: rejected source not in consulted
    bad2 = ProvenanceSidecar(
        topic="test",
        date="2026-06-02",
        sources_consulted=["https://example.com/a"],
        sources_rejected=["https://example.com/c"],
    )
    problems = bad2.validate()
    assert any("rejected" in p for p in problems), problems

    # Case 3: BLOCKED without notes
    bad3 = ProvenanceSidecar(
        topic="test",
        date="2026-06-02",
        verification=VerificationStatus.BLOCKED,
        notes=[],
    )
    problems = bad3.validate()
    assert any("BLOCKED" in p for p in problems), problems

    # Case 4: empty topic
    bad4 = ProvenanceSidecar(topic="", date="2026-06-02")
    problems = bad4.validate()
    assert any("topic" in p for p in problems), problems

    print("[OK] validate() catches all 4 accounting + integrity errors")


def test_round_trip_writes_complete_ledger() -> None:
    sidecar = ProvenanceSidecar(
        topic="What is JEPA?",
        date="2026-06-02",
        rounds=2,
        sources_consulted=[
            "https://arxiv.org/abs/2301.08243",
            "https://example.com/blog",
            "https://example.com/dead-link",
        ],
        sources_accepted=[
            "https://arxiv.org/abs/2301.08243",
            "https://example.com/blog",
        ],
        sources_rejected=["https://example.com/dead-link"],
        verification=VerificationStatus.PASS_WITH_NOTES,
        plan_ref="outputs/.plans/jepa.md",
        research_files=["jepa-research-web.md", "jepa-research-papers.md"],
        notes=["Found one dead link to example.com/dead-link; removed claims dependent on it."],
    )
    problems = sidecar.validate()
    assert problems == [], f"unexpected validation problems: {problems}"

    rendered = sidecar.render()

    # Required schema fields all present
    for required in [
        "# Provenance: What is JEPA?",
        "**Date:** 2026-06-02",
        "**Rounds:** 2",
        "**Sources consulted:** 3",
        "**Sources accepted:** 2",
        "**Sources rejected:** 1",
        "**Verification:** PASS WITH NOTES",
        "**Plan:** outputs/.plans/jepa.md",
        "**Research files:**",
        "jepa-research-web.md",
        "jepa-research-papers.md",
    ]:
        assert required in rendered, f"missing: {required!r}\n--- rendered ---\n{rendered}"

    # Negative-space accounting: rejected sources appear verbatim
    assert "https://example.com/dead-link" in rendered, "rejected source must appear in ledger"
    print("[OK] sidecar.render() includes all required ledger fields incl. negative space")


def test_disk_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.provenance.md"
        sidecar = ProvenanceSidecar(
            topic="smoke",
            date="2026-06-02",
            sources_consulted=["https://a"],
            sources_accepted=["https://a"],
            verification=VerificationStatus.PASS,
        )
        out = sidecar.write(path)
        assert out == path, out
        content = path.read_text()
        assert "PASS" in content
        assert "https://a" in content
    print("[OK] .write() round-trips through disk")


def test_empty_rejected_says_none() -> None:
    """When no sources rejected, the ledger says '(none)' rather than just leaving it blank."""
    sidecar = ProvenanceSidecar(
        topic="t",
        date="2026-06-02",
        sources_consulted=["https://a"],
        sources_accepted=["https://a"],
    )
    rendered = sidecar.render()
    assert "**Sources rejected:** 0" in rendered
    assert "(none)" in rendered, rendered
    print("[OK] empty rejected list explicitly says '(none)' rather than silently omitting")


def main() -> int:
    print("Provenance sidecar verification")
    print("=" * 50)
    tests = [
        test_status_from_findings,
        test_validate_catches_accounting_errors,
        test_round_trip_writes_complete_ledger,
        test_disk_round_trip,
        test_empty_rejected_says_none,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print("=" * 50)
    if failed:
        print(f"{failed} test(s) FAILED.")
        return 1
    print("All checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
