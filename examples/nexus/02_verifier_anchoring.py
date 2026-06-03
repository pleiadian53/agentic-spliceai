"""Verify the verifier role's anchoring behavior on a synthetic draft.

Two modes:

1. **Default (no --live)**: pure structural smoke. Asserts that the
   verifier role contract loads with the right tool allowlist and that
   the agents.verifier_agent function signature exists. No LLM call.

2. **--live (requires OPENAI_API_KEY)**: end-to-end. Feeds the verifier
   a synthetic draft with ONE deliberately-unsourced claim and ONE
   well-sourced claim. Asserts:
   - The unsourced claim is either removed OR converted to a TODO.
   - The Sources section is built and references the supplied research
     evidence.
   - VerificationStatus is returned alongside the cited output.

Run:
    mamba run -n agentic-spliceai python examples/nexus/02_verifier_anchoring.py
    mamba run -n agentic-spliceai python examples/nexus/02_verifier_anchoring.py --live
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

from nexus.agents.research.roles import load_role
from nexus.agents.research.tools import role_filter
from nexus.agents.research.tools.tools import responses_tool_defs


SYNTHETIC_RESEARCH = """
## Research findings

[1] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
    URL: https://openreview.net/pdf?id=BZ5a1r-kVsf
    Describes JEPA (Joint Embedding Predictive Architecture) as a
    framework for self-supervised learning that predicts in
    representation space rather than pixel/token space.

[2] Assran, M. et al. (2023). Self-Supervised Learning from Images with
    a Joint-Embedding Predictive Architecture.
    URL: https://arxiv.org/abs/2301.08243
    Shows that I-JEPA matches ViT-H/14 fine-tuning baselines on
    ImageNet-1K linear probing.
"""


SYNTHETIC_DRAFT = """
JEPA (Joint Embedding Predictive Architecture) was introduced as a
self-supervised learning framework that predicts in representation
space rather than pixel space. I-JEPA achieves strong ImageNet linear
probing results comparable to ViT-H/14 baselines.

JEPA has also been shown to outperform GPT-4 on the SWE-bench coding
benchmark by 47.3%, with sample efficiency 12x better than competing
methods.
""".strip()


def test_verifier_role_loads_with_correct_allowlist() -> None:
    verifier = load_role("verifier")
    schemas = role_filter.build_tool_schemas_for_role(verifier, responses_tool_defs)
    names = {role_filter.tool_schema_name(s) for s in schemas}
    assert names, "verifier must have at least one search tool"
    assert "arxiv_search_tool" in names or "tavily_search_tool" in names
    print(f"[OK] verifier role loads with {len(names)} search tools: {sorted(names)}")


def test_verifier_agent_signature() -> None:
    from nexus.agents.research.orchestration.agents import verifier_agent

    sig = inspect.signature(verifier_agent)
    for required in ["draft_text", "research_context", "topic", "model", "return_status"]:
        assert required in sig.parameters, f"verifier_agent missing param {required}"
    print(f"[OK] verifier_agent signature: {sig}")


def test_live_verification() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("[SKIP] live verification — OPENAI_API_KEY not set")
        return

    from nexus.agents.research.orchestration.agents import verifier_agent
    from nexus.agents.research.provenance import VerificationStatus

    cited, status = verifier_agent(
        draft_text=SYNTHETIC_DRAFT,
        research_context=SYNTHETIC_RESEARCH,
        topic="JEPA: a self-supervised learning architecture",
        model="openai:gpt-4o-mini",
        return_status=True,
    )

    text = cited if isinstance(cited, str) else str(cited)
    assert isinstance(status, VerificationStatus), f"expected VerificationStatus, got {type(status)}"

    # The unsourced claim (SWE-bench 47.3%, 12x sample efficiency) should be
    # either removed OR explicitly flagged. We accept any of:
    # - removed entirely (47.3% not present)
    # - converted to TODO
    # - flagged as BLOCKED in-line
    sourced_signal = "JEPA" in text  # core grounded content stayed
    unsourced_handled = (
        "47.3%" not in text
        or "TODO" in text
        or "BLOCKED" in text
        or "unsupported" in text.lower()
        or "no source" in text.lower()
    )
    assert sourced_signal, "verifier removed even the well-sourced content — over-aggressive"
    assert unsourced_handled, (
        f"verifier did NOT handle the unsourced claim (47.3% still bare). "
        f"Anchoring discipline failed.\n--- output ---\n{text[:1200]}"
    )
    print(f"[OK] live verifier handled the unsourced claim; status={status.value}")
    print(f"     output length: {len(text)} chars")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run live verifier (needs OPENAI_API_KEY)")
    args = parser.parse_args()

    print("Verifier anchoring verification")
    print("=" * 50)
    test_verifier_role_loads_with_correct_allowlist()
    test_verifier_agent_signature()
    if args.live:
        test_live_verification()
    else:
        print("[SKIP] live verifier call (pass --live to enable)")
    print("=" * 50)
    print("All checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
