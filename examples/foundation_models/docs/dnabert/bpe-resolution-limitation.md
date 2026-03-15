# DNABERT-2: BPE Token Resolution Limitation

DNABERT-2 uses Byte-Pair Encoding (BPE) tokenization, making it the only foundation
model in the project where tokens do NOT map 1:1 to nucleotides. This document explains
the implications for position-level genomic tasks.

## Background: Tokenization Strategies

| Model | Tokenization | Token : Nucleotide | Center token = center nucleotide? |
|-------|-------------|-------------------|----------------------------------|
| Evo2 | Byte-level (custom) | 1:1 | Yes (exact) |
| SpliceBERT | Single-nucleotide (sentencepiece) | 1:1 | Yes (exact) |
| HyenaDNA | Character-level | 1:1 | Yes (exact) |
| **DNABERT-2** | **BPE** | **1:~5-6** | **No (approximate)** |

## How the Sparse Exon Classifier Handles This

For each sampled genomic position, the classifier:

1. Extracts a centered context window (e.g., 4000 bp around the target)
2. Runs the foundation model to get embeddings
3. Takes a **single embedding vector** to represent the target position

For character-level models, step 3 takes the exact nucleotide embedding:

```python
# Character-level: emb.shape[0] == number of nucleotides
center_idx = emb.shape[0] // 2  # exact center nucleotide
```

For DNABERT-2, the same code runs but `emb.shape[0]` is the number of **BPE tokens**,
not nucleotides. The center token is spatially *near* but not *at* the target nucleotide:

```python
# BPE: emb.shape[0] == number of BPE tokens (~seq_len / 5.5)
center_idx = emb.shape[0] // 2  # approximate center, off by ~3-6 nt
```

### Why This Works for Exon Classification

Exon classification is a **coarse task** — exons average ~150 bp and introns average
~3,400 bp in human genes. Being off by a few nucleotides doesn't change the label.
The center BPE token still falls within the same exon or intron as the target position.

### Why This Breaks for Splice Site Prediction (Phase B)

Splice sites are **point features** — the exact nucleotide at an exon-intron boundary.
A 3-6 nucleotide offset means the embedding might come from a different token than
the one covering the splice junction. For Phase B, this approximation is insufficient.

## Solutions for Nucleotide-Resolution Tasks

### Option 1: Token-to-Nucleotide Alignment

Use the tokenizer's offset mapping to identify which BPE token(s) cover each nucleotide,
then extract the corresponding embedding:

```python
tokens = tokenizer(sequence, return_offsets_mapping=True, return_tensors="pt")
offsets = tokens["offset_mapping"][0]  # [(start_char, end_char), ...]

# Find which token covers target nucleotide position
target_nt = len(sequence) // 2
for tok_idx, (start, end) in enumerate(offsets):
    if start <= target_nt < end:
        embedding = hidden_states[0, tok_idx, :]
        break
```

**Pros**: Exact alignment, no interpolation needed.
**Cons**: Slightly more complex; one nucleotide's embedding comes from a multi-nt token.

### Option 2: Interpolation / Upsampling

Upsample BPE embeddings back to nucleotide resolution using token spans:

```python
# Each BPE token's embedding is repeated for each nucleotide it covers
nt_embeddings = []
for tok_idx, (start, end) in enumerate(offsets):
    span = end - start
    nt_embeddings.extend([hidden_states[0, tok_idx, :]] * span)
```

**Pros**: Full nucleotide-resolution output, compatible with position-level metrics.
**Cons**: Adjacent nucleotides within the same BPE token get identical embeddings.

### Option 3: Multi-Token Pooling

Average embeddings from the 2-3 BPE tokens surrounding the target position:

```python
# Pool a small window of tokens around the target
target_tok = len(offsets) // 2
window = 2  # tokens on each side
pooled = hidden_states[0, target_tok-window:target_tok+window+1, :].mean(dim=0)
```

**Pros**: Smooths over token boundary effects.
**Cons**: Blurs spatial precision, which is the opposite of what splice prediction needs.

### Recommendation for Phase B

**Option 1 (token-to-nucleotide alignment)** is the best fit for splice site prediction.
It gives exact position mapping without information loss. The fact that multiple nucleotides
share the same token embedding is acceptable — the model's representation at that token
already encodes the local sequence context.

Alternatively, DNABERT-2 may be more suited as a **context encoder** (providing regional
features) while finer-grained models (SpliceBERT, Evo2) handle position-level prediction.

## Impact on Current Results

The BPE approximation contributes to DNABERT-2's slightly lower performance compared to
character-level models of similar scale. However, the dominant factor is likely model
architecture and training data rather than tokenization resolution, since exon boundaries
are large enough that the center-token approximation is rarely wrong.

## Related

- [DNABERT-2 pad_token_id fix](./pad-token-id-transformers5.md) — transformers 5.x compatibility
- [SpliceBERT tokenizer issues](../splicebert/splicebert-tokenizer-transformers5.md) — different tokenizer challenges
- [Foundation Model Catalog](../foundation-model-catalog.md) — all models and their properties
