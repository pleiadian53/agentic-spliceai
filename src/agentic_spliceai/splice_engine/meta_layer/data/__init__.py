"""
Data loading and processing for the meta-layer.

- dataset.py: PyTorch Dataset and DataLoader utilities
- splicevardb_loader.py: SpliceVarDB data loading for evaluation
- variant_dataset.py: Variant delta prediction dataset

Ported from: meta_spliceai/splice_engine/meta_layer/data/
"""

from .dataset import (
    MetaLayerDataset,
    create_dataloaders,
    prepare_training_data
)
from .splicevardb_loader import (
    SpliceVarDBLoader,
    VariantRecord,
    load_splicevardb
)
from .variant_dataset import (
    VariantSample,
    VariantDeltaDataset,
    prepare_variant_data,
    create_variant_dataloader
)

__all__ = [
    # Dataset
    "MetaLayerDataset",
    "create_dataloaders",
    "prepare_training_data",
    # SpliceVarDB
    "SpliceVarDBLoader",
    "VariantRecord",
    "load_splicevardb",
    # Variant Dataset
    "VariantSample",
    "VariantDeltaDataset",
    "prepare_variant_data",
    "create_variant_dataloader",
]












