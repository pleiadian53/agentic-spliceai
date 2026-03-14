# SpliceBERT Integration Guide

## Model Overview

**SpliceBERT** (Chen & Zheng 2024, *Briefings in Bioinformatics*) is a bidirectional
BERT model pre-trained on 2 million pre-mRNA sequences from 72 vertebrate species
(~65 billion nucleotides). It uses single-nucleotide tokenization with a 26-token
sentencepiece vocabulary and supports sequences up to 1024 nt.

| Property | Value |
|---|---|
| Architecture | BERT (6 layers, 8 heads) |
| Hidden dim | 512 |
| Max context | 1024 nt (default), 510 nt (`.510nt` variant) |
| Tokenization | Single-nucleotide (character-level via sentencepiece) |
| Vocab | 26 tokens: `<pad>`, `<cls>`, `<eos>`, `<unk>`, N, A, C, G, U, + special |
| Training data | 2M pre-mRNA from 72 vertebrates (RNA convention: U, not T) |
| HuggingFace ID | `multimolecule/splicebert`, `multimolecule/splicebert.510nt` |
| Parameters | ~19.4M |

### Why bidirectional matters

Unlike causal models (Evo2, HyenaDNA) where each position only "sees" upstream context,
SpliceBERT's bidirectional attention means every position's embedding already encodes
**full left+right context**. This eliminates the need for dual-strand extraction or
reverse-complement augmentation when capturing downstream splice signals.

---

## Tokenization: DNA vs RNA Convention

This was the most subtle integration issue. SpliceBERT was trained on **pre-mRNA**,
which uses RNA convention:

| Convention | Nucleotides | Used by |
|---|---|---|
| DNA | A, C, G, **T** | Evo2, HyenaDNA, DNABERT-2, genome FASTA files |
| RNA (pre-mRNA) | A, C, G, **U** | **SpliceBERT**, RNA-seq data |

Since genomic sequences from FASTA files use T (thymine), input sequences must be
converted T→U before tokenization.

### Tokenization comparison across models

| Model | Tokenization | Vocab size | Input format | Special tokens |
|---|---|---|---|---|
| **SpliceBERT** | Single-nucleotide (sentencepiece) | 26 | Space-separated, T→U | `<cls>`, `<eos>` |
| **Evo2** | Byte-level (custom `evo2` package) | 512 | Raw DNA string | None |
| **HyenaDNA** | Character-level (custom, `trust_remote_code`) | ~16 | Raw DNA string | None |
| **DNABERT-2** | BPE (Byte Pair Encoding) | ~4096 | Raw DNA string | `[CLS]`, `[SEP]` |

### SpliceBERT tokenization pipeline

```
Input:   "ACGTACGT"                      # DNA from FASTA
Step 1:  "ACGUACGU"                      # T→U conversion (RNA convention)
Step 2:  "A C G U A C G U"              # Space-separate (WordPiece requirement)
Step 3:  [<cls>, A, C, G, U, A, C, G, U, <eos>]  # BertTokenizer adds special tokens
Step 4:  [1, 6, 7, 8, 9, 6, 7, 8, 9, 2]          # Token IDs
Output:  embeddings[1:-1]                # Strip <cls>/<eos>, shape [8, 512]
```

Why space-separation is necessary: `BertTokenizer` uses WordPiece, which splits on
whitespace first. Without spaces, `"ACGUACGU"` is treated as one "word" and maps to
`<unk>` since it's not in the 26-token vocabulary.

---

## Architecture Decisions

### No `multimolecule` dependency

The `multimolecule` Python package provides custom model registration for SpliceBERT
with HuggingFace's `AutoModel` system. We bypass it entirely because:

1. **Version fragility**: `multimolecule 0.0.9` requires transformers ~5.0-5.2, but
   NVIDIA Docker images ship 5.3.0 and other models may need different versions
2. **Unnecessary complexity**: SpliceBERT is architecturally a standard BERT — no
   custom model code is needed
3. **Single dependency**: Only `sentencepiece` is required (for the tokenizer)

### Config-first model loading

transformers 5.x removed the ability to pass `state_dict=` alongside a model ID in
`from_pretrained()`. We use a three-step pattern:

```python
config = BertConfig.from_pretrained(model_id)   # 1. Load config from HuggingFace
model = BertModel(config)                        # 2. Create empty model
model.load_state_dict(remapped, strict=False)    # 3. Load remapped weights
```

### State dict key remapping

SpliceBERT's checkpoint uses non-standard naming that requires three transformations:

```python
# Strip "model." prefix:    model.encoder.layer.0... -> encoder.layer.0...
# Rename layer_norm:         ...layer_norm.weight    -> ...LayerNorm.weight
# Skip lm_head:             lm_head.dense.weight     -> (discarded)
```

See `SpliceBERTModel._remap_state_dict()` for the implementation.

---

## Implementation

### Key files

| File | Purpose |
|---|---|
| `foundation_models/foundation_models/splicebert/config.py` | `SpliceBERTConfig` dataclass, model variants |
| `foundation_models/foundation_models/splicebert/model.py` | `SpliceBERTModel` — load, encode, metadata |
| `foundation_models/foundation_models/base.py` | `BaseEmbeddingModel` ABC, `load_embedding_model()` factory |

### Usage

```python
from foundation_models import load_embedding_model

# Via factory (recommended)
model = load_embedding_model("splicebert", device="cuda")

# Via direct instantiation
from foundation_models.splicebert import SpliceBERTModel, SpliceBERTConfig
config = SpliceBERTConfig(model_variant="splicebert", device="cuda")
model = SpliceBERTModel(config=config)

# Encode — T→U conversion is handled automatically
embeddings = model.encode("ACGTACGTACGT")
print(embeddings.shape)  # torch.Size([12, 512])

# Batch encode
embeddings = model.encode(["ACGTACGT", "GCTAGCTA"])
print(embeddings.shape)  # torch.Size([2, 8, 512])
```

### Model variants

| Variant | Max context | Hidden dim | HuggingFace ID |
|---|---|---|---|
| `splicebert` | 1024 nt | 512 | `multimolecule/splicebert` |
| `splicebert.510nt` | 510 nt | 512 | `multimolecule/splicebert.510nt` |

---

## Diagnostics

The model wrapper includes built-in diagnostics that run at load time:

### Tokenizer diagnostic

```
Tokenizer diagnostic: 'ACGUACGU' -> ids=[1, 6, 7, 8, 9, 6, 7, 8, 9, 2],
  tokens=['<cls>', 'A', 'C', 'G', 'U', 'A', 'C', 'G', 'U', '<eos>']
```

**What to check**: Every nucleotide should map to a named token. If you see `<unk>`,
either T→U conversion is missing or space-separation is broken.

### Embedding diagnostic

```
Embedding diagnostic: mean=-0.0762, std=0.3328, cross-position variance=0.053093
```

**What to check**:
- `std` should be >> 0 (near-zero means weights weren't loaded correctly)
- `cross-position variance` should be > 0 (different positions should have different
  embeddings; zero means the model is producing identical outputs for every position)

### State dict diagnostic

```
State dict: 101 remapped, 103 model, 101 matched, 2 missing, 0 unexpected
Missing keys (OK if only pooler): ['pooler.dense.weight', 'pooler.dense.bias']
```

**What to check**:
- `matched` should equal `remapped` count (101/101)
- `missing` should only be `pooler.*` keys (not used for embeddings)
- `unexpected` should be 0

---

## Performance on Sparse Exon Classifier

Tested with `05_sparse_exon_classifier.py --model splicebert` on RunPod RTX A5000.

### Results progression (debugging timeline)

| Stage | AUROC | AUPRC | Issue |
|---|---|---|---|
| Initial (multimolecule) | Error | Error | Package incompatible with transformers 5.3.0 |
| BertModel, no key remap | 0.50 | ~0.15 | 0/101 weights loaded (random embeddings) |
| + key remapping | 0.50 | ~0.15 | Weights correct, but tokenizer broken |
| + space separation | 0.50 | ~0.15 | A/C/G correct, T→`<unk>` |
| + partial signal (T=unk) | 0.6993 | 0.3697 | 75% of nucleotides correct |
| **+ T→U conversion** | **0.8594** | **0.5458** | **All nucleotides correct** |

The T→U fix alone improved AUROC by +0.16 and AUPRC by +0.18. The final AUROC of 0.8594
falls within the paper's reference range (0.82-0.99 across species).

### Full test metrics (final run)

| Metric | Value |
|---|---|
| AUROC | 0.8594 |
| AUPRC | 0.5458 |
| Accuracy | 0.8354 |
| F1 | 0.5652 |
| Precision | 0.4727 |
| Recall | 0.7027 |
| Test set | 243 positions (37 exonic) |
| Training | Early stop at epoch 34, best at epoch 14 (val AUPRC=0.6419) |
| Time | 1.2 min total (embedding extraction: ~5s for 766 positions) |

### Comparison with Evo2

| Metric | Evo2 7B | SpliceBERT |
|---|---|---|
| Hidden dim | 4096 | 512 |
| Parameters | 7B | 19.4M |
| AUROC | ~0.85-0.90 | 0.8594 |
| Embedding speed | Slow (GPU required) | Fast (~5s for 766 positions) |
| Max context | 131K+ | 1024 |
| Embedding size | 1.5 MB (766×512) | 1.4 MB (766×512) |

SpliceBERT's much smaller size (360x fewer parameters) makes it practical for rapid
iteration and CPU inference, while Evo2's longer context and deeper representations
give it an edge on tasks requiring long-range genomic context. For the sparse exon
classification task, SpliceBERT achieves comparable AUROC to Evo2 despite being
orders of magnitude smaller.

---

## Troubleshooting

### "sentencepiece is required" but it's installed

The error message can be misleading. If `sentencepiece` is installed but you're still
getting tokenizer errors, the issue is likely `AutoTokenizer` routing to the fast
tokenizer path. Switch to `BertTokenizer` directly.

### AUROC = 0.5 (random chance)

Three possible causes, in order of likelihood:
1. **Tokenizer**: Check diagnostic — are nucleotides mapping to `<unk>`?
2. **Weights**: Check state dict diagnostic — are keys matched?
3. **Space separation**: Is `" ".join(sequence)` being applied before tokenization?

### embeddings.npz is 0 bytes

All sequences mapped to `<unk>`, producing zero-variance embeddings that got
compressed to nothing. Fix tokenization first.

### Embedding shape mismatch

SpliceBERT adds `[CLS]` and `[SEP]` tokens. The `encode()` method strips them
automatically, but if you're calling the model directly, remember:

```
model output: [batch, seq_len + 2, 512]   # includes [CLS] and [SEP]
after strip:  [batch, seq_len, 512]        # 1:1 with input nucleotides
```

---

## Dependencies

```bash
# Required
pip install transformers sentencepiece

# Also needed (usually already present)
pip install safetensors huggingface_hub torch
```

No `multimolecule` package. No `tokenizers` library (fast tokenizer not used).

## References

- Chen & Zheng (2024). "Self-supervised learning on millions of pre-mRNA sequences
  improves sequence-based RNA splicing prediction." *Briefings in Bioinformatics*, 25(3).
- HuggingFace: https://huggingface.co/multimolecule/splicebert
- Error details: [splicebert-tokenizer-transformers5.md](splicebert-tokenizer-transformers5.md)

## Environment

- Docker: `nvcr.io/nvidia/pytorch:25.02-py3`
- transformers: 5.3.0
- sentencepiece: 0.2.1
- Python: 3.12
- GPU: NVIDIA RTX A5000 (RunPod, CA-MTL-1)
