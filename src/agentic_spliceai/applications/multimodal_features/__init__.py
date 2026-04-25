"""Multimodal Features Application — meta-layer ingestion layer.

Packaged version of the use cases demonstrated in ``examples/features/``.
Handles the inputs the meta layer needs beyond the base-layer predictions:
external tracks (conservation, epigenetic, chromatin accessibility),
RNA-seq junction evidence, and RBP eCLIP binding — and produces the
per-position multimodal feature parquets consumed by M1-M4 training.

Complements :mod:`agentic_spliceai.applications.data_preparation`: that
application handles the FASTA/GTF side (the "what is a gene" inputs);
this one handles the multimodal evidence (the "what does the rest of the
cell look like at this position" inputs).

Quick start
-----------

    from agentic_spliceai.applications.multimodal_features import (
        prepare_features, get_status, list_profiles,
    )

    profiles = list_profiles()
    status = get_status(output_dir="output/features/my_run")

    result = prepare_features(
        profile="full_stack",
        chromosomes=["22"],
        input_dir="data/mane/GRCh38/openspliceai_eval/precomputed",
        output_dir="output/features/chr22_full_stack",
    )

CLI
---

    agentic-spliceai-features list-profiles
    agentic-spliceai-features list-tracks --build GRCh38
    agentic-spliceai-features status --profile full_stack \\
        --output-dir data/mane/GRCh38/openspliceai_eval/analysis_sequences
    agentic-spliceai-features prepare --profile full_stack \\
        --chromosomes 22 \\
        --input-dir  data/mane/GRCh38/openspliceai_eval/precomputed \\
        --output-dir output/features/chr22_full_stack

Production safety
-----------------

``prepare`` requires ``--output-dir`` explicitly. It will never write to
production feature directories unless the user passes that path and
``--force``. ``status``, ``list-profiles``, ``list-tracks``, and
``validate`` are read-only.
"""

from .manifest import FeatureManifest, FeatureArtifactRecord
from .pipeline import prepare_features, PreparationResult
from .profiles import list_profiles, load_profile, FeatureProfile
from .status import get_status, FeaturePrepStatus
from .tracks import list_tracks, TrackRecord

__all__ = [
    "prepare_features",
    "get_status",
    "list_profiles",
    "list_tracks",
    "load_profile",
    "FeatureProfile",
    "FeatureManifest",
    "FeatureArtifactRecord",
    "FeaturePrepStatus",
    "PreparationResult",
    "TrackRecord",
]
