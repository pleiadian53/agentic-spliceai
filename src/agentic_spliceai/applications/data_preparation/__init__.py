"""Data Preparation Application — genomic ingestion layer.

Packaged version of the use cases demonstrated in
``examples/data_preparation/`` and ``examples/diagnostics/``. Takes the
two inputs users actually have to provide — a reference FASTA and a gene
annotation file (GTF or GFF3) — and derives the artifacts every
downstream application needs: gene metadata DB, per-gene sequences,
splice-site ground truth, chromosome splits, and a manifest.

Quick start
-----------

    from agentic_spliceai.applications.data_preparation import (
        prepare_build, get_status,
    )

    # Prepare a fresh build into a throwaway output dir
    result = prepare_build(
        build="GRCh38",
        annotation_source="mane",
        output_dir="output/ingest/my_run",
    )

    status = get_status(output_dir="output/ingest/my_run")
    print(status.summary())

CLI
---

    agentic-spliceai-ingest prepare --build GRCh38 --annotation-source mane \\
        --output-dir output/ingest/GRCh38_mane
    agentic-spliceai-ingest status --output-dir output/ingest/GRCh38_mane
    agentic-spliceai-ingest validate --output-dir output/ingest/GRCh38_mane

Production safety
-----------------

``prepare`` requires ``--output-dir`` explicitly. It will never write to
the resource-manager-resolved production path (e.g.
``data/mane/GRCh38/``) unless the user passes that path themselves AND
adds ``--force`` to overwrite existing artifacts. ``status`` and
``validate`` are read-only.
"""

from .manifest import IngestManifest, ArtifactRecord, InputRecord
from .pipeline import prepare_build, resolve_canonical_output_dir, PreparationResult
from .status import get_status, DataPrepStatus

__all__ = [
    "prepare_build",
    "resolve_canonical_output_dir",
    "get_status",
    "IngestManifest",
    "ArtifactRecord",
    "InputRecord",
    "PreparationResult",
    "DataPrepStatus",
]
