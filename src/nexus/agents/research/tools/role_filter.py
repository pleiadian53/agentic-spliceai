"""Per-role tool allowlist enforcement (Feynman Tier 1 discipline).

The Einstein-quote principle ported to Python: a role cannot solve a
problem with the same level of thinking that created it. Mechanical
enforcement of "different level of thinking" is per-role tool
restriction — the LLM serving the writer role gets a tool schema that
literally excludes `tavily_search_tool`, so the model physically
cannot emit a tool call for web search regardless of how the prompt is
phrased.

This module is the schema-filtering layer. Call sites build the LLM
request with `build_tool_schemas_for_role(role, ...)` instead of
passing the global `responses_tool_defs` directly. The filter takes
the role's `tools:` allowlist (from `roles/<name>.md`'s YAML
frontmatter) and intersects it with the global registry.

Usage
-----
    >>> from nexus.agents.research.roles import load_role
    >>> from nexus.agents.research.tools import role_filter
    >>> from nexus.agents.research.tools.tools import responses_tool_defs, tool_mapping, aisuite_tools

    >>> writer = load_role("writer")
    >>> filtered = role_filter.build_tool_schemas_for_role(writer, responses_tool_defs)
    >>> filtered
    []   # writer has empty tools list → no tool schemas → LLM cannot emit tool calls

    >>> verifier = load_role("verifier")
    >>> filtered = role_filter.build_tool_schemas_for_role(verifier, responses_tool_defs)
    >>> [s["function"]["name"] for s in filtered]
    ['tavily_search_tool', 'arxiv_search_tool', ...]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..roles import Role


def tool_schema_name(schema: dict[str, Any]) -> str:
    """Extract the tool name from a Responses-API tool schema dict.

    Handles both the wrapped form (`{"type": "function", "function": {"name": ...}}`)
    and the flat form (`{"name": ...}`). aisuite normalizes to the
    wrapped form by default.
    """
    if "function" in schema and isinstance(schema["function"], dict):
        name = schema["function"].get("name")
        if name:
            return str(name)
    if "name" in schema:
        return str(schema["name"])
    raise KeyError(f"Tool schema has no extractable name field: {schema!r}")


def build_tool_schemas_for_role(
    role: Role,
    all_tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the schemas whose tool name is in `role.tools`.

    This is the load-bearing filter. The LLM call should be built with
    the result of THIS function, not the global `all_tool_schemas`.

    A role with `tools: []` (e.g. writer, reviewer) gets back `[]` —
    the LLM serving them has no tool schemas in its request payload and
    physically cannot emit a tool call for any tool.
    """
    allowlist = set(role.tools)
    return [s for s in all_tool_schemas if tool_schema_name(s) in allowlist]


def filter_tool_mapping_for_role(
    role: Role,
    tool_mapping: dict[str, Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Return only the tool callables whose name is in `role.tools`.

    Used together with `build_tool_schemas_for_role` when invoking
    `nexus.llm.tool_loop.call_llm_with_tools` — the tool_mapping passed
    to the loop should be role-filtered so even if the model somehow
    emitted a forbidden tool name (it cannot, by construction), the
    executor would refuse to run it.
    """
    allowlist = set(role.tools)
    return {name: fn for name, fn in tool_mapping.items() if name in allowlist}


def filter_aisuite_tools_for_role(
    role: Role,
    aisuite_tools: list[Callable[..., Any]],
) -> list[Callable[..., Any]]:
    """Return only the aisuite callables whose `__name__` is in `role.tools`."""
    allowlist = set(role.tools)
    return [fn for fn in aisuite_tools if getattr(fn, "__name__", "") in allowlist]


__all__ = [
    "tool_schema_name",
    "build_tool_schemas_for_role",
    "filter_tool_mapping_for_role",
    "filter_aisuite_tools_for_role",
]
