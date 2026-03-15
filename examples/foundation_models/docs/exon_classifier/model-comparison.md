# Foundation Model Comparison: Sparse Exon Classification

## Overview

Phase A of the foundation models sub-project evaluates four pre-trained DNA/RNA
language models on a **sparse exon classification** task: given a genomic position
and its surrounding context window, predict whether the position falls within an
exon (coding) or intron (non-coding) region.

This reproduces and extends the evaluation from the [Evo2 paper](https://arcinstitute.org/manuscripts/Evo2),
which demonstrated that frozen embeddings from large-scale DNA language models
encode enough biological signal to distinguish exons from introns with high
accuracy. We apply the same framework to three additional models (SpliceBERT,
HyenaDNA, DNABERT-2) to assess their embedding quality and suitability for
splice site prediction in Phase B.

---

## Results

All models were evaluated on the same 766 sampled positions (115 exonic, 651
intronic) from 19,226 MANE genes (GRCh38), using identical train/val/test
chromosome splits and classifier hyperparameters.

### Test Set Performance

| Model | Params | Hidden | Feature dim | AUROC | AUPRC | F1 | Accuracy | Best epoch |
|---|---|---|---|---|---|---|---|---|
| **Evo2 7B** | 7B | 4096 | 8192 | **0.9937** | **0.9759** | **0.9189** | **0.9753** | 1 |
| **SpliceBERT** | 19.4M | 512 | 512 | 0.8594 | 0.5458 | 0.5652 | 0.8354 | 14 |
| **HyenaDNA** | 6.6M | 256 | 512 | 0.8242 | 0.3917 | 0.4600 | 0.7778 | 2 |
| **DNABERT-2** | 117M | 768 | 768 | 0.6556 | 0.2642 | 0.3063 | 0.6831 | 5 |

### Training Dynamics

| Model | Convergence | Time | Embedding extraction |
|---|---|---|---|
| Evo2 7B | Best at epoch 1, early stop 21 | 45.9 min | ~42 min (766 pos, GPU) |
| SpliceBERT | Best at epoch 14, early stop 34 | 1.2 min | ~5s (766 pos, GPU) |
| HyenaDNA | Best at epoch 2, early stop 22 | 2.6 min | ~1.5 min (766 pos, GPU) |
| DNABERT-2 | Best at epoch 5, early stop 25 | 2.2 min | ~1 min (766 pos, GPU) |

### Data Split

All models used the SpliceAI chromosome split preset:

- **Train** (19 chroms): 2, 4, 6, 8, 10-22, X — 471 positions
- **Val** (19 chroms): 2, 4, 6, 8, 10-22, X, Y — 52 positions
- **Test** (5 chroms): 1, 3, 5, 7, 9 — 243 positions (37 exonic)

---

## Architecture

### Embedding Extraction Pipeline

```
                                    ┌─────────────┐
                                    │  Foundation  │
 Sampled position ─── Context ────► │    Model     │ ──► Per-position embedding
   (chr, pos)        window         │   (frozen)   │     [feature_dim]
                                    └─────────────┘
```

1. **Sample positions**: 1500 positions across 19,226 genes, weighted by gene
   length, with 15% exon enrichment (oversampling exonic regions from the
   natural ~2% to address class imbalance)
2. **Extract context**: a DNA window around each position (size depends on model)
3. **Embed**: pass the context through the frozen foundation model, extract a
   single embedding vector per position
4. **Classify**: train a lightweight MLP to predict exon vs intron from the
   embedding

### Embedding Strategy by Model Type

The strategy differs based on whether the model is **causal** (next-token
prediction) or **bidirectional** (masked language modeling):

**Causal models** (Evo2, HyenaDNA) — dual-strand, last-position:

```
Forward:   ───[context before position]──►  take emb[-1]  → [hidden_dim]
Reverse:   ───[context after position]───►  take emb[-1]  → [hidden_dim]
                                                             ─────────────
Concatenate:                                              → [2 × hidden_dim]
```

Causal models only attend to past tokens. To give the model access to context
on both sides of the position, we extract embeddings from both strands
independently. The position of interest is always the last token in the
sequence, where the model has seen the most context.

**Bidirectional models** (SpliceBERT, DNABERT-2) — centered window:

```
───[context before]─── POSITION ───[context after]───►  take emb[center]  → [hidden_dim]
```

Bidirectional models attend in both directions. A single centered window
captures context from both sides. The embedding at the center position
represents the model's understanding of that position given full surrounding
context.

### Classifier Head

All models use the same `ExonClassifier` architecture — a 2-layer MLP:

```
Input [feature_dim] → Linear(feature_dim, 1024) → ReLU → Dropout(0.1)
                     → Linear(1024, 1)           → Sigmoid → P(exon)
```

Shared hyperparameters:
- Learning rate: 5e-5, weight decay: 2e-4
- Batch size: 16, max epochs: 100
- Early stopping: patience 20, monitoring AUPRC
- Loss: weighted binary cross-entropy (class weights based on exon ratio)

---

## Tokenization

Each model processes DNA sequences differently, which has significant
implications for embedding quality and positional resolution.

### Character-level Tokenizers

**Evo2** — byte-level, 1:1 mapping:
```
Input:    A  T  C  G  A  T  C  G
Tokens:  [A][T][C][G][A][T][C][G]     (1 token = 1 nucleotide)
```
Uses its own `evo2` package tokenizer. Each nucleotide maps to exactly one
token. Output embeddings have a direct 1:1 correspondence with input
nucleotides.

**HyenaDNA** — character-level, 1:1 mapping:
```
Input:    A  T  C  G  A  T  C  G
Tokens:  [A][T][C][G][A][T][C][G]     (1 token = 1 nucleotide)
```
Uses a custom `CharacterTokenizer` via `trust_remote_code=True`. Similar 1:1
mapping. No special preprocessing needed.

**SpliceBERT** — single-nucleotide, requires preprocessing:
```
Input:    A  T  C  G  A  T  C  G       (DNA, has T)
Convert:  A  U  C  G  A  U  C  G       (RNA convention: T→U)
Space:   "A U C G A U C G"             (space-separate for BertTokenizer)
Tokens:  [CLS][A][U][C][G][A][U][C][G][SEP]
Output:       [A][U][C][G][A][U][C][G]  (strip special tokens)
```

Two critical preprocessing steps:
1. **T→U conversion**: SpliceBERT was trained on pre-mRNA (RNA convention).
   Its vocabulary has `U` (id=9) but not `T`. Without conversion, all
   thymine positions map to `<unk>`, losing ~25% of nucleotide information.
2. **Space separation**: `BertTokenizer` uses WordPiece which splits on
   whitespace first. Without spaces, the entire DNA string becomes one
   "word" that doesn't match any vocabulary entry, mapping to a single
   `<unk>` token.

### BPE Tokenizer (Variable-length)

**DNABERT-2** — byte-pair encoding, N:1 mapping:
```
Input:    A  T  C  G  A  T  C  G       (8 nucleotides)
Tokens:  [CLS][ATC][GAT][CG][SEP]      (4 tokens, variable length)
Output:       [ATC][GAT][CG]           (strip special tokens)
```

DNABERT-2 uses BPE tokenization that groups nucleotides into variable-length
k-mers (typically 3-6 nt per token). This creates a fundamental mismatch for
per-nucleotide tasks:

- **No 1:1 mapping**: a single embedding represents 3-6 nucleotides
- **Center-token approximation**: for the sparse classifier, we take
  `emb[center_token_idx]`, which may be offset by 3-6 nt from the actual
  position of interest
- **Positional blur**: the model cannot distinguish between nucleotides
  within the same BPE token

This likely explains DNABERT-2's lower performance on this per-nucleotide
classification task — the BPE tokenization was designed for sequence-level
tasks (classification, similarity) rather than nucleotide-level annotation.

### Tokenization Summary

| Model | Strategy | Vocab size | Resolution | Preprocessing |
|---|---|---|---|---|
| Evo2 | Byte-level | 512 | 1:1 | None |
| HyenaDNA | Character | 16 | 1:1 | None |
| SpliceBERT | Nucleotide | 26 | 1:1 | T→U, space-separate |
| DNABERT-2 | BPE | 4096 | N:1 (~3-6 nt) | None (but positional blur) |

---

## Integration Challenges

### Evo2
- FP8 monkey-patch required for GPUs < compute 8.9 (A40, A100)
- BFloat16 → float32 conversion before `.numpy()`
- Hidden dim probed at runtime (7B=4096, 40B=8192)
- GPU memory: ~14 GB for 7B model, chunk_size=4096 to avoid OOM
- See: [evo2/evo2-gpu-memory.md](../evo2/evo2-gpu-memory.md)

### SpliceBERT
- `multimolecule` package incompatible with transformers 5.x — bypassed entirely
- Load as standard `BertModel` with manual state dict key remapping
  (`model.` prefix → stripped, `layer_norm` → `LayerNorm`, skip `lm_head.*`)
- `BertTokenizer` directly (not `AutoTokenizer` which forces fast tokenizer path)
- See: [splicebert/splicebert-integration.md](../splicebert/splicebert-integration.md),
  [splicebert/splicebert-tokenizer-transformers5.md](../splicebert/splicebert-tokenizer-transformers5.md)

### HyenaDNA
- `__init__` must accept `**kwargs` and forward to config (registry pattern)
- `trust_remote_code=True` required for custom model/tokenizer classes
- `lm_head.weight` reported as UNEXPECTED — safe to ignore for embeddings
- See: [hyenadna/hyenadna-kwargs-init.md](../hyenadna/hyenadna-kwargs-init.md)

### DNABERT-2
- `pad_token_id` missing from config in transformers 5.x — must set manually
- Meta-device initialization (`low_cpu_mem_usage`) incompatible with custom
  ALiBi code — load config + empty model + manual state dict instead
- State dict keys have `bert.` prefix + `cls.predictions.*` (LM head) — need
  remapping similar to SpliceBERT
- Custom `flash_attn_triton.py` uses deprecated `tl.dot(q, k, trans_b=True)`
  — auto-patched to `tl.dot(q, tl.trans(k))` at load time
- ALiBi size warnings when input exceeds 512 tokens (trained size) — functional
  but may degrade attention quality for long sequences
- See: [dnabert/pad-token-id-transformers5.md](../dnabert/pad-token-id-transformers5.md)

---

## Analysis

### Why Evo2 Dominates

Evo2 7B achieves near-perfect performance (AUROC 0.9937) on epoch 1,
suggesting that its embeddings already encode a strong exon/intron boundary
signal. Contributing factors:

1. **Scale**: 7B parameters (vs 6.6M-117M for others) trained on 9.3 trillion
   tokens across 128K+ genomes
2. **Dual-strand feature dim**: 2 x 4096 = 8192 features per position (vs
   256-768 for others), giving the classifier more signal to work with
3. **Long context**: 8,192 bp window captures broader genomic context around
   each position
4. **Layer selection**: embeddings from intermediate layer `blocks.26`
   (optimized for genomic annotation) rather than the final layer

### Why DNABERT-2 Underperforms

Despite having the second-largest model (117M params), DNABERT-2 achieves the
lowest AUROC (0.6556). Key factors:

1. **BPE positional blur**: the center-token approximation introduces 3-6 nt
   of spatial noise — equivalent to the width of a splice site motif
2. **Low cross-position variance** (0.003 vs 0.053 for SpliceBERT): embeddings
   are less differentiated across positions, providing less spatial signal
3. **Training objective mismatch**: DNABERT-2 was optimized for sequence-level
   tasks with BPE, not nucleotide-level annotation
4. **ALiBi extrapolation**: model was trained with ~128-512 token sequences
   but receives ~800+ tokens, degrading attention quality

### SpliceBERT vs HyenaDNA

Both achieve solid AUROC (0.86 vs 0.82) despite being much smaller than Evo2:

- **SpliceBERT** (19.4M): bidirectional attention + trained specifically on
  pre-mRNA sequences from 72 vertebrate species. Its training data is
  directly relevant to exon/intron boundaries. Only 512 hidden dim but
  high cross-position variance (0.053) suggests informative embeddings.
- **HyenaDNA** (6.6M): the smallest model, using Hyena convolution operators
  instead of attention. Despite only 256 hidden dim, the dual-strand
  embedding (2 x 256 = 512) captures directional context. Converges very
  fast (best at epoch 2) but plateaus early.

---

## Implications for Phase B (Splice Site Prediction)

Phase B targets per-nucleotide splice site prediction (3-class: none/acceptor/
donor). Key considerations from Phase A results:

1. **Character-level tokenizers are essential**: per-nucleotide resolution is
   critical for splice site prediction where the signal is at a single
   position. DNABERT-2's BPE tokenization makes it poorly suited without
   a different embedding strategy (e.g., interpolation or nucleotide-level
   upsampling).

2. **SpliceBERT is the efficiency sweet spot**: 360x fewer parameters than
   Evo2 yet achieves 86% of its AUROC. Its pre-mRNA training data is
   directly relevant to splice sites.

3. **Evo2 embeddings are powerful but expensive**: near-perfect exon
   classification suggests these embeddings may generalize well to the
   harder splice site task, but 45 min per 766 positions limits iteration
   speed.

4. **Dual-strand helps causal models**: the dual-strand strategy effectively
   doubles the feature dimension and provides bidirectional context for
   models that are otherwise unidirectional.

---

## Reproducing

```bash
# Run on GPU pod (each model independently)
python examples/foundation_models/05_sparse_exon_classifier.py \
    --foundation-model evo2 --n-positions 1500 \
    -o /workspace/output/sparse-evo2/

python examples/foundation_models/05_sparse_exon_classifier.py \
    --foundation-model splicebert --n-positions 1500 \
    -o /workspace/output/sparse-splicebert/

python examples/foundation_models/05_sparse_exon_classifier.py \
    --foundation-model hyenadna --n-positions 1500 \
    -o /workspace/output/sparse-hyenadna/

python examples/foundation_models/05_sparse_exon_classifier.py \
    --foundation-model dnabert --n-positions 1500 \
    -o /workspace/output/sparse-dnabert/

# Transfer results back
rsync -Pavz <cluster>:/workspace/output/ ./output/exon_classifier/
```

---

## Output Artifacts

Each run produces:
```
sparse-<model>/
  sparse_embeddings.npz     # [N, feature_dim] embeddings + labels
  model/
    best_model.pt           # Classifier checkpoint (best val AUPRC)
    eval_metrics.json       # Test set metrics
    split_info.json         # Chromosome split details
  plots/
    loss_curves.png         # Train/val loss per epoch
    metric_curves.png       # AUROC/AUPRC per epoch
    training_overview.png   # Combined training summary
```
