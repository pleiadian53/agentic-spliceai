"""Markdown-based role contracts with YAML frontmatter.

Modeled after Feynman's `.feynman/agents/*.md` pattern. The contract IS
the prompt artifact — Python wraps it in a dataclass for plumbing but
the source of truth is the markdown file, which is human-readable and
auditable in plain text.

Usage
-----
    >>> from nexus.agents.research.roles import load_role
    >>> writer = load_role("writer")
    >>> "web_search" in writer.tools
    False
    >>> verifier = load_role("verifier")
    >>> "fetch_content" in verifier.tools
    True

The role's tool list is a mechanical guardrail enforced by
`nexus.agents.research.tools.role_filter.build_tool_schemas_for_role`:
when an LLM call is made with a role-filtered tool schema, the model
literally cannot emit a tool call for a tool whose schema isn't in
the list. This is what makes "different level of thinking"
mechanically enforced (not aspirational).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


ROLES_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = ROLES_DIR / "SYSTEM.md"


@dataclass(frozen=True)
class Role:
    """One role contract: frontmatter metadata + system-prompt body.

    Parsed from `roles/<name>.md`. `tools` is the role's allowlist —
    a tool not in this list is not callable by the role, enforced at
    the tool-schema layer when the LLM call is made.
    """

    name: str
    description: str
    thinking: str  # "low" | "medium" | "high"
    tools: list[str] = field(default_factory=list)
    output: str = ""
    system_prompt: str = ""

    def with_global_system(self, global_system_text: str) -> "Role":
        """Return a copy with the global SYSTEM.md prepended to the role body."""
        combined = f"{global_system_text.strip()}\n\n---\n\n{self.system_prompt.strip()}\n"
        return Role(
            name=self.name,
            description=self.description,
            thinking=self.thinking,
            tools=list(self.tools),
            output=self.output,
            system_prompt=combined,
        )


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_role_file(path: Path) -> Role:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        raise ValueError(
            f"Role file {path} is missing YAML frontmatter (--- ... ---)."
        )
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()

    name = meta.get("name")
    if not name:
        raise ValueError(f"Role file {path} frontmatter missing required `name`.")

    return Role(
        name=name,
        description=meta.get("description", ""),
        thinking=meta.get("thinking", "medium"),
        tools=list(meta.get("tools", [])),
        output=meta.get("output", ""),
        system_prompt=body,
    )


def load_role(name: str, *, with_global_system: bool = True) -> Role:
    """Load `roles/<name>.md` and return its Role.

    By default, the global `SYSTEM.md` operating rules are prepended to
    the role's body — that's how the verb-binding, anti-fabrication,
    and source-quality rules apply to every role.
    """
    path = ROLES_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"No role contract at {path}. Available roles: "
            f"{sorted(p.stem for p in ROLES_DIR.glob('*.md') if p.name != 'SYSTEM.md')}"
        )
    role = _parse_role_file(path)
    if with_global_system and SYSTEM_PROMPT_PATH.exists():
        role = role.with_global_system(SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"))
    return role


def available_roles() -> list[str]:
    """List the role names with contract files on disk."""
    return sorted(
        p.stem for p in ROLES_DIR.glob("*.md") if p.name != "SYSTEM.md"
    )


__all__ = ["Role", "load_role", "available_roles", "ROLES_DIR", "SYSTEM_PROMPT_PATH"]
