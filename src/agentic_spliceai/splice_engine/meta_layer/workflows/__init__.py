"""
High-level workflows for the meta-layer.

Training Workflows:
- canonical_training.py: Training on canonical splice sites for recalibration
  - Uses base layer artifacts as training data
  - Evaluates on held-out canonical sites + SpliceVarDB variants
  - Original approach: paired prediction (ref vs alt)

- validated_delta_training.py: Delta prediction with validated targets (TODO)
  - Uses SpliceVarDB-validated deltas as training targets
  - Single-pass inference (more efficient)
  - BEST SO FAR: r=0.41 correlation

- hyenadna_training.py: GPU training with HyenaDNA encoder (TODO)
  - Requires RunPods or similar GPU environment
  - Higher capacity but more computationally expensive

Key Design Principle:
  SpliceVarDB is NOT used for training in canonical_training workflow,
  only for evaluation. This validates whether the meta-layer improves
  variant effect detection without directly training on variant labels.

For validated_delta_training, SpliceVarDB IS used to filter/validate
training targets, addressing the limitation of learning from potentially
inaccurate base model deltas.

Ported from: meta_spliceai/splice_engine/meta_layer/workflows/
"""

from .canonical_training import (
    CanonicalTrainingWorkflow,
    CanonicalTrainingConfig,
    CanonicalTrainingResult,
    run_canonical_training
)

__all__ = [
    "CanonicalTrainingWorkflow",
    "CanonicalTrainingConfig",
    "CanonicalTrainingResult",
    "run_canonical_training",
]












