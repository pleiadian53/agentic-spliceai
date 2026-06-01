"""Generate ``output/REGISTRY.md`` from MANIFEST.yaml files.

The registry is a **generated artifact** — never hand-edited. Edit
individual MANIFESTs instead, then run::

    python -m agentic_spliceai.registry build

The output is a markdown file organized by topic, with one table per
topic showing artifact, status, produced-by, superseded-by, notes, and
tags.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

from .manifest import Manifest

logger = logging.getLogger(__name__)

# Stable topic ordering for the generated registry — keeps the most active
# / important topics at the top, then alphabetical for the rest.
_TOPIC_ORDER = (
    "meta_layer",
    "m4_benchmarks",
    "exon_classifier",
    "fm_scalars",
    "gpu_runs",
    "splice_classifier",
    "bio_cache",
    "biomol_design",
    "(root)",
)

_HEADER = """# Output Registry — generated

**Generated** from per-artifact `MANIFEST.yaml` files by
`python -m agentic_spliceai.registry build`.
Do not edit by hand — edit the underlying MANIFESTs instead.

See [`docs/system_design/output_management.md`](../docs/system_design/output_management.md)
for the convention. Status tags: `active`, `baseline`, `experimental`,
`archived`, `placeholder`.

"""


def _topic_sort_key(topic: str) -> tuple[int, str]:
    """Sort key that puts known topics first in fixed order, then alphabetical."""
    try:
        return (_TOPIC_ORDER.index(topic), topic)
    except ValueError:
        return (len(_TOPIC_ORDER), topic)


def _format_produced_by(items: List[str]) -> str:
    if not items:
        return "—"
    if len(items) == 1:
        return f"`{items[0]}`"
    return "<br>".join(f"`{item}`" for item in items)


def _format_tags(tags: List[str]) -> str:
    if not tags:
        return "—"
    return " ".join(f"`{t}`" for t in tags)


def _format_notes(notes: str) -> str:
    """Collapse a multi-line note to a single-line table cell."""
    if not notes:
        return ""
    # Collapse newlines and runs of whitespace; markdown table cells dislike
    # raw newlines.
    return " ".join(notes.split())


def _format_referenced_by(refs: List[str]) -> str:
    if not refs:
        return ""
    bullets = "\n".join(f"  - `{r}`" for r in refs)
    return f"\n  - **Referenced by**:\n{bullets}"


def render_registry(manifests: Iterable[Manifest]) -> str:
    """Render the registry markdown from a collection of manifests.

    Tables are grouped by topic. Within a topic, rows are sorted by status
    (active first, then baseline, experimental, archived, placeholder),
    then by artifact name.
    """
    by_topic: "OrderedDict[str, List[Manifest]]" = OrderedDict()
    for m in manifests:
        by_topic.setdefault(m.topic, []).append(m)

    # Stable topic order.
    ordered_topics = sorted(by_topic.keys(), key=_topic_sort_key)

    out = [_HEADER.lstrip()]
    out.append(f"_{sum(len(v) for v in by_topic.values())} artifacts across "
               f"{len(ordered_topics)} topics._\n\n")

    status_rank = {
        "active": 0,
        "baseline": 1,
        "experimental": 2,
        "archived": 3,
        "placeholder": 4,
    }

    for topic in ordered_topics:
        rows = by_topic[topic]
        rows.sort(key=lambda m: (status_rank.get(m.status, 99), m.name))

        out.append(f"## `output/{topic}/`\n\n" if topic != "(root)"
                   else "## `output/` (top-level)\n\n")

        out.append("| Artifact | Status | Produced by | Superseded by | Tags | Notes |\n")
        out.append("|---|---|---|---|---|---|\n")
        for m in rows:
            out.append(
                f"| `{m.name}/` | **{m.status}** | "
                f"{_format_produced_by(m.produced_by)} | "
                f"{m.superseded_by or '—'} | "
                f"{_format_tags(m.tags)} | "
                f"{_format_notes(m.notes)} |\n"
            )
        out.append("\n")

        # Per-artifact `referenced_by` details (only if any artifact in this
        # topic has them) as a detail block under the table.
        any_refs = any(m.referenced_by for m in rows)
        if any_refs:
            out.append("<details><summary>Cross-references for this topic</summary>\n\n")
            for m in rows:
                if m.referenced_by:
                    out.append(f"- `{m.name}/`:\n")
                    for ref in m.referenced_by:
                        out.append(f"  - `{ref}`\n")
            out.append("\n</details>\n\n")

    return "".join(out)


def build_registry(
    output_root: Path,
    registry_path: Path | None = None,
) -> Path:
    """Generate ``REGISTRY.md`` at ``registry_path`` (default ``output_root / REGISTRY.md``).

    Returns the path written.
    """
    from .discovery import load_all_manifests

    registry_path = registry_path or (output_root / "REGISTRY.md")
    manifests = load_all_manifests(output_root)
    text = render_registry(manifests)
    registry_path.write_text(text)
    return registry_path
