"""Linchpin verification: per-role tool restrictions are mechanically enforced.

This is the load-bearing test for nexus's role-separation discipline.
If this script runs green, the principle that a role cannot solve a
problem with the same level of thinking that created it is not
aspirational — it is enforced at the LLM tool-schema layer.

The mechanism:
- Each role contract (`roles/<name>.md`) declares a `tools:` allowlist in
  YAML frontmatter.
- `tools.role_filter.build_tool_schemas_for_role(role, all_schemas)`
  intersects the allowlist with the global tool registry.
- The LLM call sites pass the FILTERED schema list as the `tools=` arg,
  so the model literally cannot emit a tool call for a tool whose
  schema isn't in its payload.

This script makes NO LLM calls. The assertions are purely structural —
they verify the schema filter does what its name says.

Run:
    mamba run -n agentic-spliceai python examples/nexus/04_role_tool_restrictions.py
"""

from __future__ import annotations

import sys

from nexus.agents.research.roles import load_role, available_roles
from nexus.agents.research.tools import role_filter
from nexus.agents.research.tools.tools import (
    aisuite_tools,
    responses_tool_defs,
    tool_mapping,
)


SEARCH_TOOLS = {
    "tavily_search_tool",
    "arxiv_search_tool",
    "wikipedia_search_tool",
    "europe_pmc_search_tool",
    "semantic_scholar_search_tool",
    "reddit_search_tool",
}


def test_all_roles_load() -> None:
    names = available_roles()
    assert set(names) >= {"researcher", "writer", "verifier", "reviewer"}, names
    print(f"[OK] all four roles loadable: {names}")


def test_writer_excludes_all_search() -> None:
    """The writer must have ZERO search tools — mechanical, not prompted."""
    writer = load_role("writer")
    schemas = role_filter.build_tool_schemas_for_role(writer, responses_tool_defs)
    schema_names = {role_filter.tool_schema_name(s) for s in schemas}
    forbidden = SEARCH_TOOLS & schema_names
    assert not forbidden, (
        f"writer MUST NOT have any search tool. Found: {forbidden}. "
        f"Discipline broken — writer can search the web."
    )
    assert schemas == [], (
        f"writer should have ZERO tool schemas (tools: []), got {len(schemas)}"
    )
    print("[OK] writer has zero tool schemas — cannot emit any tool call")


def test_writer_mapping_is_empty() -> None:
    """The runtime tool_mapping passed to the LLM must also be empty for writer."""
    writer = load_role("writer")
    filtered_map = role_filter.filter_tool_mapping_for_role(writer, tool_mapping)
    assert filtered_map == {}, f"writer's tool_mapping should be empty, got {filtered_map.keys()}"

    filtered_aisuite = role_filter.filter_aisuite_tools_for_role(writer, aisuite_tools)
    assert filtered_aisuite == [], f"writer's aisuite_tools should be empty, got {filtered_aisuite}"
    print("[OK] writer's runtime executor map + aisuite list are both empty")


def test_verifier_has_search_for_anchoring() -> None:
    """The verifier MUST have search tools — verification without network is self-grading."""
    verifier = load_role("verifier")
    schemas = role_filter.build_tool_schemas_for_role(verifier, responses_tool_defs)
    schema_names = {role_filter.tool_schema_name(s) for s in schemas}
    assert SEARCH_TOOLS & schema_names, (
        f"verifier MUST have at least one search tool to anchor claims. "
        f"Got: {schema_names}. Verification without tools is self-grading."
    )
    print(f"[OK] verifier has {len(schema_names)} search tool(s) for claim anchoring")


def test_reviewer_is_read_only() -> None:
    """The reviewer is read-only — no authoring, no searching."""
    reviewer = load_role("reviewer")
    schemas = role_filter.build_tool_schemas_for_role(reviewer, responses_tool_defs)
    schema_names = {role_filter.tool_schema_name(s) for s in schemas}
    assert not (SEARCH_TOOLS & schema_names), (
        f"reviewer should not search. Found: {SEARCH_TOOLS & schema_names}"
    )
    assert schemas == [], (
        f"reviewer should have ZERO tool schemas (tools: []), got {len(schemas)}"
    )
    print("[OK] reviewer has zero tool schemas — pure read-only critique")


def test_researcher_has_full_search() -> None:
    """The researcher is the only role with ALL search tools."""
    researcher = load_role("researcher")
    schemas = role_filter.build_tool_schemas_for_role(researcher, responses_tool_defs)
    schema_names = {role_filter.tool_schema_name(s) for s in schemas}
    missing = SEARCH_TOOLS - schema_names
    assert not missing, (
        f"researcher must have all 6 search tools. Missing: {missing}"
    )
    print(f"[OK] researcher has all {len(SEARCH_TOOLS)} search tools")


def test_provocation_writer_schema_excludes_search() -> None:
    """The critical assertion: even if a user prompt explicitly tells the
    writer to search the web, the writer's tool schema sent to the LLM
    PHYSICALLY does not contain web_search/fetch_content. The model has
    no way to know those tools exist when serving the writer role.

    This is what makes role separation mechanical, not aspirational.
    """
    writer = load_role("writer")
    schemas_for_writer = role_filter.build_tool_schemas_for_role(
        writer, responses_tool_defs
    )

    # No matter what the prompt says, the LLM cannot call what isn't in its schema.
    assert schemas_for_writer == [], (
        "writer's tool schema list is non-empty — provocation possible"
    )
    print(
        "[OK] writer's tool schema is empty; LLM cannot emit any tool call "
        "regardless of prompt content"
    )


def main() -> int:
    print("Role tool restriction — mechanical enforcement verification")
    print("=" * 60)
    tests = [
        test_all_roles_load,
        test_writer_excludes_all_search,
        test_writer_mapping_is_empty,
        test_verifier_has_search_for_anchoring,
        test_reviewer_is_read_only,
        test_researcher_has_full_search,
        test_provocation_writer_schema_excludes_search,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print("=" * 60)
    if failed:
        print(f"{failed} test(s) FAILED. Discipline is not yet mechanical.")
        return 1
    print(
        "All role restrictions verified.\n"
        "Role separation is mechanically enforced, not aspirationally prompted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
