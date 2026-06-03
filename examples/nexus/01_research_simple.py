"""End-to-end research pipeline smoke + verification.

Modes:

1. **Default (no --live)**: structural smoke. Asserts:
   - `generate_research_report` is callable with the expected signature
   - The new `enable_verification` kwarg exists
   - All major agent functions (planner/research/writer/editor/
     verifier/reviewer) are exposed via `orchestration.agents`

2. **--live (requires OPENAI_API_KEY, ~$0.10-0.30 in API cost)**:
   runs the full pipeline on a narrow topic with `enable_verification=True`.
   Asserts:
   - A final report is produced.
   - The verifier produced a cited output.
   - The reviewer produced a review document.
   - The verification status is recorded.

Run:
    mamba run -n agentic-spliceai python examples/nexus/01_research_simple.py
    mamba run -n agentic-spliceai python examples/nexus/01_research_simple.py --live
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

from nexus.agents.research.orchestration import agents, pipeline


def test_pipeline_signature() -> None:
    sig = inspect.signature(pipeline.generate_research_report)
    assert "enable_verification" in sig.parameters, (
        "pipeline.generate_research_report missing the Feynman Tier 1 opt-in flag"
    )
    assert sig.parameters["enable_verification"].default is False, (
        "enable_verification must default to False (preserve baseline behavior)"
    )
    print(f"[OK] pipeline signature includes enable_verification kwarg")


def test_agent_surface() -> None:
    for name in [
        "planner_agent",
        "research_agent",
        "writer_agent",
        "editor_agent",
        "executor_agent",
        "verifier_agent",
        "reviewer_agent",
    ]:
        assert hasattr(agents, name), f"agents.{name} missing"
    print(f"[OK] orchestration.agents exposes all 7 agent functions")


def test_live_research() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("[SKIP] live research run — OPENAI_API_KEY not set")
        return

    result = pipeline.generate_research_report(
        topic="What is JEPA (Joint Embedding Predictive Architecture)?",
        model="openai:gpt-4o-mini",
        report_length="brief",
        verbose=False,
        enable_verification=True,
    )

    assert "final_report" in result, "no final_report in pipeline output"
    assert result["final_report"], "final_report is empty"
    print(f"[OK] final_report produced ({len(result['final_report'])} chars)")

    assert "cited_output" in result, "verifier did not produce cited_output"
    assert result["cited_output"], "cited_output is empty"
    print(f"[OK] cited_output produced ({len(result['cited_output'])} chars)")

    assert "verification_status" in result, "no verification_status recorded"
    print(f"[OK] verification_status = {result['verification_status'].value}")

    assert "review" in result, "reviewer did not produce a review"
    assert result["review"], "review is empty"
    print(f"[OK] reviewer produced review ({len(result['review'])} chars)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run full pipeline with real LLM calls (~$0.10-0.30; needs OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    print("Research-pipeline smoke + verification")
    print("=" * 50)
    test_pipeline_signature()
    test_agent_surface()
    if args.live:
        test_live_research()
    else:
        print("[SKIP] live pipeline run (pass --live to enable)")
    print("=" * 50)
    print("All checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
