# Foundation Model Embeddings Modality

## Why Foundation Model Embeddings Matter for Splicing

Purpose-built splice site predictors like SpliceAI use dilated CNNs with ~10 kb
context to detect donor/acceptor sites. Foundation models (Evo2, SpliceBERT,
HyenaDNA) are pre-trained on billions of nucleotides from thousands of species,
learning general-purpose representations of DNA/RNA that encode:

- **Long-range dependencies** (Evo2's attention captures >100 kb context)
- **Cross-species conservation patterns** (SpliceBERT trained on 72 vertebrates)
- **Regulatory grammar** (motif combinations, secondary structure propensity)

These representations are complementary to the engineered features in the other
modalities. A position's embedding encodes biological context that cannot be
distilled into a single conservation score or histone mark value.

### Experimental Evidence

Using frozen SpliceBERT embeddings (512-dim, no fine-tuning) with a lightweight
dilated CNN classifier head:

| Model | AUPRC | Notes |
|-------|-------|-------|
| SpliceAI (purpose-built) | ~0.95+ | End-to-end CNN, 10 kb context |
| Meta-layer XGBoost M1 | 0.998 | 83 features, canonical sites |
| **SpliceBERT frozen head** | **0.832** | Generic embeddings, no fine-tuning |

The 12% gap vs purpose-built models is expected for a frozen-head approach. But
the key insight is that these embeddings carry **orthogonal signal** to what the
other modalities provide. Adding PCA-projected scalar features from Evo2 (which
significantly outperforms SpliceBERT) may improve M2-M4 meta-models for
alternative and novel splice site detection.

## Two-Tier Architecture

### Tier 1: Scalar Summary Features (Implemented)

The `fm_embeddings` modality extracts **8 label-agnostic scalar features** per
position from pre-computed embeddings. All features are computed without using
splice site annotations, so there is no risk of label leakage. These plug
directly into XGBoost and other tabular meta-layer models:

| Feature | Description | Biological Rationale |
|---------|-------------|---------------------|
| `fm_pca_1` ... `fm_pca_6` | Top-k PCA components (unsupervised) | Dominant variation modes in embedding space correlate with functional categories (exon/intron, splice type, regulatory context) |
| `fm_embedding_norm` | L2 magnitude of embedding vector | Correlates with model confidence and sequence complexity; unusual sequences have atypical norms |
| `fm_local_gradient` | L2 norm of embedding change vs neighbors | Detects boundary signals — splice sites are transitions between distinct regulatory contexts |

All features use **strand-aware dual-strand embeddings** for causal models
(Evo2, HyenaDNA): the gene is encoded on both the forward strand and the
reverse complement, the RC embeddings are flipped and averaged with the forward
embeddings, so every position sees both upstream and downstream context.
Bidirectional models (SpliceBERT) already see full context from a single pass.

**Optional (disabled by default):** `fm_donor_cosine_sim` and
`fm_acceptor_cosine_sim` — cosine similarity to splice site embedding centroids.
These use ground truth labels to compute centroids (supervised feature
engineering, not leakage with proper train/test split), but are redundant with
base model scores that already capture splice-likeness more precisely. Enable
via `include_cosine_centroids: true` in config if needed for experimentation.

### Tier 2: Full Embedding Vectors (Future)

Raw dense vectors (4096-dim for Evo2) stored in sidecar HDF5 files for deep
neural meta-models. See `dev/planning/wishlist.md`.

## Workflow

The recommended workflow uses **streaming extraction**
(`07_streaming_fm_scalars.py`) which computes all 8 scalar features on the
GPU pod without saving intermediate embeddings. Total pod storage: ~65 MB.

### Step 1: Run Streaming Extraction on GPU Pod

Prerequisites: existing feature parquets from `06_multimodal_genome_workflow.py`
(for sampled position lookup). The script runs two phases:

- **Phase 1**: IncrementalPCA fit on training chromosomes (streaming, one gene at a time)
- **Phase 2**: Compute 8 scalar features per sampled position, write per-chromosome parquets

```bash
# On GPU pod (e.g., A40 48GB)
# Full genome — both phases (wrap in nohup for long runs)
nohup python -u examples/features/07_streaming_fm_scalars.py \
    --foundation-model evo2 --model-size 7b \
    --feature-dir /runpod-volume/data/mane/GRCh38/openspliceai_eval/analysis_sequences/ \
    --chromosomes all \
    -o /runpod-volume/output/fm_scalars/evo2_7b/ \
    > /runpod-volume/output/fm_scalars/evo2_7b/extraction.log 2>&1 &

# Single chromosome test (quick validation, ~2 hrs with Evo2 dual-strand)
python -u examples/features/07_streaming_fm_scalars.py \
    --foundation-model evo2 --model-size 7b \
    --feature-dir /runpod-volume/data/mane/GRCh38/openspliceai_eval/analysis_sequences/ \
    --chromosomes 22 \
    -o /runpod-volume/output/fm_scalars/evo2_7b/

# Resume after interruption (skip Phase 1 + completed chromosomes)
nohup python -u examples/features/07_streaming_fm_scalars.py \
    --foundation-model evo2 --model-size 7b \
    --feature-dir /runpod-volume/data/mane/GRCh38/openspliceai_eval/analysis_sequences/ \
    --chromosomes all \
    --pca-artifacts /runpod-volume/output/fm_scalars/evo2_7b/pca_artifacts.npz \
    -o /runpod-volume/output/fm_scalars/evo2_7b/ \
    > /runpod-volume/output/fm_scalars/evo2_7b/extraction.log 2>&1 &
```

**Resume support**: Completed chromosomes (those with `fm_scalars_{chrom}.parquet`)
are automatically skipped. Pass `--pca-artifacts` to also skip Phase 1 PCA fitting.
The only loss on interruption is the in-progress chromosome.

Output on pod (~65 MB total):
```
/runpod-volume/output/fm_scalars/evo2_7b/
    pca_artifacts.npz              # PCA components + mean (~200 KB)
    fm_scalars_chr1.parquet        # 8 scalar columns per sampled position
    fm_scalars_chr2.parquet
    ...
    fm_scalars_chrY.parquet
    extraction_summary.json        # Timing, counts, metadata
```

### Step 2: Transfer Scalars to Local Machine

```bash
# rsync back (~65 MB — NOT the 4-64 GB of raw embeddings)
rsync -Pavz pod:/runpod-volume/output/fm_scalars/evo2_7b/ \
    data/mane/GRCh38/fm_embeddings/evo2_7b/
```

### Step 3: Enable in YAML Config

Uncomment the `fm_embeddings` modality with `scalar_dir` pointing to the
transferred parquets:

```yaml
pipeline:
  modalities:
    - base_scores
    - annotation
    - sequence
    - genomic
    - conservation
    - epigenetic
    - junction
    - rbp_eclip
    - chrom_access
    - fm_embeddings                # Enable once scalars are extracted

modality_configs:
  fm_embeddings:
    foundation_model: evo2_7b
    scalar_dir: data/mane/GRCh38/fm_embeddings/evo2_7b/  # pre-computed scalars
    pca_components: 6
    include_cosine_centroids: false
    include_gradient: true
```

### Step 4: Augment Feature Parquets

```bash
python 06_multimodal_genome_workflow.py \
    --config configs/full_stack.yaml \
    --chromosomes chr22 \
    --augment
```

## Key Design Decisions

### 1. Streaming Extraction (No Intermediate Storage)

Unlike conservation/epigenetic modalities (which query bigWig files on the fly),
embeddings are too expensive to compute per-`transform()` call. The recommended
approach is `07_streaming_fm_scalars.py` which runs on a GPU pod:
extract embedding → compute scalars → discard embedding, one gene at a time.
Raw embeddings never touch disk. The modality then reads the ~65 MB of
pre-computed scalar parquets locally.

### 2. Strand-Aware Dual-Strand Extraction

Causal models (Evo2, HyenaDNA) are autoregressive — position *i*'s embedding
only sees context from positions 0..*i* (left-to-right). For minus-strand genes,
this means the embedding at each position only has upstream (genomic left) context,
which is actually *downstream* in the transcript.

The fix: each gene is encoded twice — once on the forward strand and once on the
reverse complement. The RC embeddings are flipped to realign with forward positions
and averaged:

```
Forward:     emb_fwd[i] sees context 0..i     (upstream in genome)
Rev-comp:    emb_rc[i]  sees context 0..i     (downstream in genome, after flip)
Final:       (emb_fwd[i] + emb_rc_flipped[i]) / 2
```

This gives every position bidirectional context at the cost of 2x inference time.
Bidirectional models (SpliceBERT) already see full context and use single-strand
extraction.

### 3. PCA Fit on Training Chromosomes Only

This prevents data leakage. The PCA transformer captures variation patterns
learned from training data only. Test chromosome embeddings are projected through
the same transformation but don't influence the PCA components.

### 4. Foundation-Model-Agnostic Column Names

All columns use the `fm_` prefix (not `evo2_` or `splicebert_`), so swapping
the underlying foundation model doesn't break the downstream feature schema.
The config specifies which model was used.

### 5. Graceful Degradation

If embeddings or PCA artifacts are missing, all columns are filled with NaN.
This matches the pattern used by conservation (GRCh37) and epigenetic (GRCh37)
modalities. The pipeline runs without error; the meta-model handles NaN features
via its own imputation strategy.

## Storage Estimates

| Foundation Model | Hidden Dim | Embedding Parquets (if saved) | Scalar Output | Streaming |
|-----------------|------------|-------------------------------|---------------|-----------|
| SpliceBERT | 512 | ~4 GB | ~64 MB (8 cols) | ~1 MB (PCA artifact only) |
| Evo2 7B | 4096 | ~32 GB | ~64 MB (8 cols) | ~1 MB |
| Evo2 40B | 8192 | ~64 GB | ~64 MB (8 cols) | ~1 MB |

Tier 1 scalar output is tiny (~64 MB for 2M positions x 8 float64 columns).
With **streaming extraction** (recommended), the embedding parquets are never
saved — the GPU pod only stores the PCA artifact (~1 MB) and the scalar output
parquets (~64 MB). Total pod storage: **~65 MB instead of 4-64 GB**.

## Time Estimates (A40 48GB)

Dual-strand extraction (2x inference per gene for causal models):

| Foundation Model | Phase 1 (PCA fit) | Phase 2 (scalars) | Total | Cost ($0.44/hr) |
|-----------------|-------------------|-------------------|-------|-----------------|
| SpliceBERT | ~4-8 hrs | ~6-12 hrs | ~10-20 hrs | ~$5-9 |
| Evo2 7B | ~45-90 hrs | ~60-120 hrs | ~105-210 hrs | ~$46-92 |
| Evo2 40B | ~100-200 hrs | ~130-260 hrs | ~230-460 hrs | ~$100-200 |

SpliceBERT is bidirectional (single-strand, no 2x penalty). Evo2 times are
~2x the single-strand estimates due to dual-strand extraction.

Observed rate on A40: **~80-160 genes/hr** for Evo2 7B dual-strand (varies with
gene length — chr1 has the largest genes, smaller chromosomes are faster).

Phase 1 processes training chromosomes only (~14K genes). Phase 2 processes
all chromosomes (~19K genes). Bottleneck is GPU inference throughput, not I/O.

Use `--pca-artifacts` to skip Phase 1 on subsequent runs (e.g., resuming
after interruption or adding a missing chromosome).

## Supported Foundation Models

| Model Key | Hidden Dim | Source | Notes |
|-----------|-----------|--------|-------|
| `evo2_7b` | 4096 | Arc Institute | Attention-based, up to 1M bp context |
| `evo2_40b` | 8192 | Arc Institute | Larger model, better embeddings |
| `splicebert` | 512 | Chen et al. | BERT, 512 bp context, 72 vertebrates |
| `hyenadna` | 256 | Nguyen et al. | Long-range Hyena operator |
| `nucleotide_transformer` | 1024 | InstaDeep | Multi-species nucleotide model |

## Where This Modality Adds Value

| Meta-Model | Expected Impact | Rationale |
|------------|----------------|-----------|
| M1 (canonical) | Marginal | Already 99.72% AUPRC with 83 features; diminishing returns |
| M2 (alternative) | Moderate | Evo2's long-range context captures alternative splice patterns CNN models miss |
| M3 (novel) | Highest | Novel sites lack junction/annotation support; PCA components from a different model architecture provide independent evidence |
| M4 (perturbation) | Moderate | Embedding gradient detects context shifts from mutations; PCA drift measures how much a variant changes the local embedding landscape |

The primary value proposition is **M2-M4**: tasks where existing base models
underperform and additional biological context from a fundamentally different
model architecture (attention vs dilated CNN) matters most.
