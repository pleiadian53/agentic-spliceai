# Feature Extractor

Sequence feature extraction utilities for genomic analysis and splice site prediction.

## Features

- K-mer frequency analysis
- Sequence motif identification
- Conservation score integration
- Secondary structure prediction features
- Position-specific scoring matrix (PSSM) utilities
- Deep learning feature embeddings

## Usage

```python
from openspliceai.feature_extractor import extract_kmer_features, extract_motif_features

# Extract k-mer features from a sequence
sequence = "ACGTACGTACGT"
kmer_features = extract_kmer_features(sequence, k=3)

# Extract motif features
motif_features = extract_motif_features(sequence, motifs=["GT", "AG"])
```

## Components

- `kmer.py`: K-mer based feature extraction
- `motif.py`: Sequence motif identification and scoring
- `conservation.py`: Conservation score integration
- `structure.py`: Secondary structure prediction features
- `pssm.py`: Position-specific scoring matrix utilities
- `embedding.py`: Deep learning feature embeddings
