# Meta-Layer Model Variants: M1 through M4

## Two Model Lines: Position-Level (-P) vs Sequence-Level (-S)

Each model variant (M1-M4) exists in two forms that differ in how they
consume input and produce output:

| Suffix | Name | Input | Output | Purpose |
|--------|------|-------|--------|---------|
| **-P** | Position-level | Pre-sampled positions + tabular features | `[1, 3]` per position | Proof-of-concept, ablation studies |
| **-S** | Sequence-level | Arbitrary DNA sequence | `[L, 3]` per nucleotide | Practical splice site prediction |

**Position-level models (M1-P, M2-P, M3-P)** operate on individual
pre-sampled positions (~1% of the genome). They require base model scores
to decide which positions to evaluate, and they output a single 3-class
prediction per position. These are useful for proving that multimodality
helps (e.g., M1-P showed 62% FN reduction, 68% FP reduction) but are not
practical splice predictors.

**Sequence-level models (M1-S, M2-S, M3-S, M4-S)** follow the same
input/output protocol as SpliceAI and OpenSpliceAI: given an arbitrary
DNA sequence, they produce per-nucleotide splice site scores `[L, 3]`.
This is the practical form used for genome-scale prediction, variant
analysis, and novel isoform discovery.

### Why the distinction matters

For M2-M4, you cannot know the "interesting positions" in advance — the
whole point is to discover positions the base model missed (M2/M3) or
that appear under perturbation (M4). A position-level model requires
someone to pre-select query points, which defeats the purpose.

The sequence-level architecture (see *Sequence-Level Architecture* below)
incorporates multimodal features as dense input channels at every
position, rather than as pre-computed tabular features at sampled points.

### Implementation

| Model | Position-level (-P) | Sequence-level (-S) |
|-------|--------------------|--------------------|
| M1 | `01_xgboost_baseline.py` (XGBoost, 103 features) | `meta_splice_model_v3.py` (3-stream CNN) |
| M2 | `04_m2a_ensembl_vs_mane.py` (planned) | Same architecture, evaluated on alternative sites |
| M3 | `05_m3_novel_prediction.py` (planned) | Same architecture, junction as target |
| M4 | — | `predict_with_delta()` on ref/alt sequences |

---

## The Problem: Why One Model Isn't Enough

A splice site prediction system faces fundamentally different questions
depending on what it's trying to predict. Consider these four scenarios:

1. A clinician asks: "Is this position a donor or acceptor site?" — She
   needs high accuracy on *known* canonical splice sites.

2. A genomics researcher asks: "Which splice sites are active in liver but
   not in brain?" — He needs to detect *alternative* splice sites that are
   tissue-specific.

3. A computational biologist asks: "Can I predict splice sites in a newly
   sequenced genome with no RNA-seq data?" — She needs to discover *novel*
   splice sites from sequence and epigenomic context alone.

4. A precision medicine team asks: "Does this patient's mutation create a
   cryptic splice site?" — They need to predict splice sites *induced by
   perturbations* that don't exist in the reference genome.

These are not four versions of the same task. They differ in what features
are available, what constitutes ground truth, and how rare the positive
signal is. Trying to address all four with one model leads to compromises
that serve none of them well.

The meta-layer addresses this with four model variants — **M1 through M4** —
each optimized for one of these scenarios. They share the same multimodal
feature engineering pipeline but differ in three dimensions:

- **Which modalities are enabled** (subset vs. full)
- **What role junction reads play** (feature, target, or validation)
- **What defines the ground truth label**

---

## The Shared Foundation: Multimodal Feature Engineering

All four variants build on the same base: a configurable pipeline that
combines up to 9 modalities of biological evidence into a unified feature
matrix. Each position in the genome gets annotated with:

```
base_scores (43 cols)    — SpliceAI/OpenSpliceAI predictions + derivatives
annotation (3 cols)      — GTF splice type, transcript info
sequence (3 cols)        — DNA context window
genomic (4 cols)         — positional features, GC content
conservation (9 cols)    — PhyloP/PhastCons evolutionary constraint
epigenetic (12 cols)     — H3K36me3/H3K4me3 histone marks across tissues
junction (12 cols)       — GTEx RNA-seq splice junction evidence
rbp_eclip (8 cols)       — RNA-binding protein binding sites (ENCODE)
chrom_access (6 cols)    — ATAC-seq chromatin accessibility (ENCODE)
```

The key design insight is that **junction reads serve dual roles**. In some
models they're features (input evidence); in others they're the prediction
target. This single toggle — junction as feature vs. junction as label —
defines the boundary between detecting known alternative sites (M2) and
discovering truly novel ones (M3).

---

## M1: Enhanced Canonical Classification

### What it does

M1 recalibrates base model predictions for canonical splice sites — the
GT-AG donor-acceptor pairs annotated in reference transcriptomes. This is
the "easy" task: the base model (SpliceAI or OpenSpliceAI) already achieves
~99.5% accuracy. M1 adds multimodal context to push error rates lower,
primarily by reducing false positives and rescuing false negatives.

### The setup

```
Features:  base_scores + annotation + genomic + conservation + epigenetic
Junction:  NOT USED (canonical sites don't need RNA-seq evidence)
Target:    splice_type from GTF annotations (donor / acceptor / neither)
Config:    default.yaml or full_stack.yaml with junction disabled
```

### Why it matters

M1 is the foundation — it validates that the multimodal pipeline works at
all. If adding conservation, epigenetic marks, and genomic context doesn't
improve over base scores alone for canonical sites, there's no reason to
believe it will help for harder tasks.

### What we learned

**M1-P (position-level, XGBoost)** on chr19-22 (466K positions):

| Model | Accuracy | PR-AUC (donor) | PR-AUC (acceptor) |
|-------|----------|----------------|-------------------|
| Base scores only | 99.62% | 0.996 | 0.995 |
| Full-stack (M1-P) | **99.78%** | **0.999** | **0.998** |

SHAP analysis revealed that conservation and genomic context contribute
6-7% of the model's decision-making — despite appearing to contribute
<1% by XGBoost's gain metric. This hidden importance becomes critical
in M2-M4 where base scores are no longer sufficient.

**M1-S (sequence-level, CNN)** on full genome (12,334 train / 1,370 val
genes, SpliceAI chromosome split):

| Model | Val Accuracy | Val PR-AUC (macro) | Architecture |
|-------|-------------|-------------------|--------------|
| OpenSpliceAI (base) | — | — | 5M params, 10kb context |
| M1-S (epoch 46/50) | **99.99%** | **0.9899** | 367K params, 400bp + 9ch multimodal |

M1-S achieves near-perfect canonical accuracy with 14x fewer parameters
than the base model. The 2-stream dilated CNN + residual blending
architecture validates that multimodal context (conservation, epigenetic,
chromatin, junction, RBP) carries genuine complementary signal to base
model scores.

### Limitations

M1 optimizes for the wrong question. 99.99% accuracy on canonical sites
doesn't help when the goal is discovering sites the annotation doesn't
contain. The next three variants address progressively harder versions of
the splice site prediction problem.

---

## M2: Alternative Splice Site Detection

### What it does

M2 detects alternative splice sites — positions that function as splice
sites in some tissues or conditions but not others. These are the sites
responsible for alternative splicing: exon skipping, alternative 5'/3'
site usage, and intron retention events that generate transcript diversity.

### The setup

```
Features:  ALL modalities (base_scores + annotation + genomic +
           conservation + epigenetic + junction + rbp_eclip + chrom_access)
Junction:  FEATURE — input to the model
Target:    Multi-level: GTF annotations + junction support + PSI
Config:    full_stack.yaml
```

### The key idea: junction reads as features

The defining characteristic of M2 is that junction reads from GTEx
(353K junctions, 54 tissues) are used as *input features*, not as
prediction targets. This gives the model direct evidence of which
splice sites are actively used:

- `junction_has_support = 1` + `junction_tissue_breadth = 54` →
  constitutive splice site, used in all tissues
- `junction_has_support = 1` + `junction_tissue_breadth = 3` →
  tissue-specific alternative site
- `junction_has_support = 0` → no RNA-seq evidence at this position

Combined with RBP binding (which SR proteins and hnRNPs are bound?),
chromatin accessibility (is the DNA open?), and conservation (is this
site under selection?), M2 can learn *why* certain sites are alternative
and *which tissues* they're active in.

### The M2 variant series (M2a through M2f)

M2 is not a single model — it is a family of training and evaluation
protocols that vary along two orthogonal axes:

- **Training axis**: which labels and how they're weighted
- **Evaluation axis**: where performance is measured

The architecture (MetaSpliceModel, 3-stream CNN, ~370K params) is constant
across all variants.

#### M2a: Baseline evaluation (Ensembl \ MANE)

**Training**: M1-S model trained on MANE labels (no retraining).
**Evaluation**: Filter predictions at positions in (Ensembl \ MANE).

This is the simplest M2 experiment — take the existing M1-S model and
ask whether it generalizes to alternative splice sites it has never
seen. Ensembl sites are well-curated, so this evaluates against
high-quality positives.

Two sub-variants exist depending on base score availability:
- **Option A**: Base scores from running OpenSpliceAI on Ensembl genes
  (full coverage at alternative sites)
- **Option B**: MANE precomputed base scores only (Ensembl-only genes
  get a uniform 1/3 prior — tests multimodal-only rescue)

#### M2b: Stress test (GENCODE comprehensive \ MANE)

**Training**: M1-S model, no retraining.
**Evaluation**: Filter at (GENCODE comprehensive \ MANE) sites.

GENCODE comprehensive includes rare isoforms, computational predictions,
and lncRNA transcripts that Ensembl filters out. This is a harder test:
the positive set is noisier but includes the biologically interesting
edge cases. Report tiered recall: Tier 1 (GENCODE ∩ Ensembl \ MANE)
vs Tier 2 (GENCODE-only).

#### M2c: Annotation-tier weighted training

**Training**: Train on GENCODE comprehensive labels with confidence
weights based on annotation tier:
- Tier 0 (MANE): weight = 1.0
- Tier 1 (GENCODE ∩ Ensembl \ MANE): weight = 0.8-0.9
- Tier 2 (GENCODE-only): weight = 0.5-0.8

**Evaluation**: M2a and M2b protocols.

This is the first variant that changes training data. The confidence
weights prevent noisy GENCODE-only labels from dominating the loss
while still allowing the model to learn from them.

#### M2d: Junction-informed soft labels

**Training**: Combine annotation tier with GTEx junction evidence:

```
splice_usage_score = w0 × tier_prior + w1 × psi_normalized + w2 × breadth_normalized
```

Sites with junction support in many tissues get high labels; sites with
annotation but no junction evidence get discounted. This is a soft
labeling scheme — the target is a continuous confidence rather than
hard 0/1.

**Evaluation**: M2a and M2b protocols, plus per-tissue recall.

#### M2e: Tissue-conditioned input

**Training**: Add tissue context as an additional input channel to the
model. The input becomes `[base_scores, mm_features, tissue_embedding]`
where tissue context can be a one-hot tissue ID, an RBP expression
profile, or a learned embedding.

This is the only variant that modifies the architecture — the model
receives an extra input stream that conditions predictions on biological
context. A site labeled "active in liver" gets a different prediction
than the same site labeled "active in brain."

**Evaluation**: Per-tissue recall, tissue-specific PR-AUC.

#### M2f: PU learning (annotation as incomplete positive set)

**Training**: Treat annotations as an incomplete positive set (all
annotated sites are positive, but unannotated sites may be positive or
negative). Use Positive-Unlabeled learning frameworks rather than
standard cross-entropy.

**Status**: Deferred to M3, which naturally operates in this regime
(predicting sites with no annotation).

### Architecture considerations for M2-S

M2 operates on Ensembl/GENCODE genes with more isoform complexity than
MANE — alternative exons, cassette exons, competing splice sites within
close proximity. The M1-S architecture (H=32, 400bp receptive field)
was designed for the simpler MANE case. M2-S may benefit from:

- **Wider receptive field (500-600bp)**: Better disambiguation of
  competing splice sites. Achievable via dilation schedule tweak.
- **Larger hidden dim (H=48 or H=64)**: More capacity for isoform
  diversity patterns. H=48 roughly doubles parameters to ~800K.
- **Light transformer layer**: A single self-attention layer after the
  CNN encoders for long-range dependencies between splice sites.

These are configurable via `MetaSpliceConfig` — the recommended approach
is to first run M2-S with the same architecture as M1-S (baseline), then
increase capacity if the alternative site recall plateaus.

### What M2 learns that M1 cannot

M1 sees canonical sites where the base model already excels. M2 operates
in the regime where the base model is weakest — alternative sites with
lower base scores, tissue-specific usage, and regulatory complexity. This
is where multimodality earns its keep.

### Further reading

- [02_annotation_driven_splice_prediction.md](02_annotation_driven_splice_prediction.md) —
  annotation choice as a modeling decision, tier-based confidence, GENCODE \ MANE
- [05_m2_variant_formulations.md](05_m2_variant_formulations.md) —
  detailed training and evaluation protocols for M2a through M2f

---

## M3: Novel Splice Site Prediction

### What it does

M3 predicts completely novel splice sites — positions with no annotation
in any reference transcriptome. The goal is to discover functional splice
sites from sequence, conservation, and epigenomic context alone, without
requiring RNA-seq data at inference time.

### The setup

```
Features:  base_scores + annotation + genomic + conservation +
           epigenetic + rbp_eclip + chrom_access
           (ALL modalities EXCEPT junction)
Junction:  TARGET — held-out as the prediction label
Target:    junction_has_support (binary) or junction_psi (continuous)
Config:    meta_m3_novel.yaml
```

### The key idea: junction reads as labels

M3 inverts the relationship between junction data and the model. Instead of
using junction reads as input (M2), it uses them as the ground truth label:
"does this position have RNA-seq junction support?" The model must predict
this from everything *except* junction evidence.

This is the critical test of multimodality's value. If conservation,
histone marks, chromatin accessibility, and RBP binding can predict
junction support, then these signals carry genuine information about
splice site function — they're not just correlates of what the base model
already knows.

### Why exclude junction from features?

If junction reads were both feature and target, the model would trivially
learn `junction_has_support → junction_has_support`. Excluding junction
from the feature set forces the model to learn the underlying biology:

- High conservation (PhyloP > 2) at an unannotated position → purifying
  selection preserves this site → likely functional
- H3K36me3 enrichment → position is in a transcribed exon body →
  consistent with exon inclusion
- ATAC-seq open chromatin → DNA is accessible to the spliceosome
- RBP binding (SRSF1/RBFOX2) → splice-regulatory proteins are present

If the model can predict junction support from these signals, it can
generalize to genomes without RNA-seq — enabling novel splice site
discovery in any sequenced organism with conservation and epigenomic data.

### Connection to novel isoform discovery

M3 is the workhorse for the project's ultimate goal: building a catalog of
novel isoforms. The pipeline would be:

1. Run OpenSpliceAI to get base scores (positions with any splice signal)
2. Run M3 to predict which unannotated positions are likely functional
3. Apply junction assembly (pairing donors with acceptors) to construct
   candidate transcript structures
4. Validate candidates against independent RNA-seq or proteomics data

Step 3 — junction assembly — is the "virtual transcript" problem discussed
in `02_virtual_transcripts.md`. It requires knowing not just *where* splice
sites are, but *which donor pairs with which acceptor*. M3 handles step 2;
the pairing problem remains an open challenge (see Strategy 3 in the
virtual transcripts document).

---

## M4: Perturbation-Induced Splice Sites

### What it does

M4 predicts splice sites that are *created or destroyed* by genetic
variants, drug treatments, or disease states. These are splice sites that
don't exist in the reference genome — they're induced by perturbations.

### The setup

```
Features:  M1 features + variant context (ref/alt alleles, position
           relative to nearest splice site, predicted delta scores)
Junction:  VALIDATION — used to evaluate whether predicted effects
           match real junction changes in carriers vs. non-carriers
Target:    Delta splice potential (continuous: how much does splice
           probability change?) or binary (splice-altering vs. not)
Config:    Planned (not yet implemented)
```

### The key idea: perturbation as input

M4 extends the feature space with perturbation-specific information. For
genetic variants, this includes:

- Reference and alternate alleles
- Distance to nearest canonical splice site
- Base model predictions on both reference and alternate sequences
- Predicted delta scores (alt - ref) from the base model

The meta-layer then learns whether the multimodal context (conservation,
chromatin state, RBP binding) modulates the variant's effect on splicing.
A variant in highly conserved, open chromatin near an RBP binding site is
more likely to be splice-altering than the same variant in closed,
unconserved sequence.

### Data sources

M4's training data comes from databases that link variants to
experimentally validated splicing effects:

- **SpliceVarDB**: ~50K variants with binary splice-altering labels
- **GTEx sQTLs**: ~50K variant-junction associations across 49 tissues
- **ClinVar**: Pathogenic variants with known splicing mechanism
- **TCGA**: Somatic variants with matched tumor RNA-seq

### Connection to earlier experiments

The meta-spliceai project (predecessor to agentic-spliceai) ran four
experiments on this problem:

| Experiment | Approach | Result | Lesson |
|-----------|---------|--------|--------|
| 001 | Canonical classification | 99.1% acc, 17% variant detection | Classification != variant detection |
| 002 | Paired delta prediction | r=0.38 | Target constrained by base model |
| 003 | Binary splice-altering | AUC=0.61, F1=0.53 | Signal exists but insufficient context |
| 004 | Validated delta prediction | **r=0.41 (best)** | Target quality is the bottleneck |

The consistent lesson: the bottleneck is not model capacity but **label
quality**. SpliceVarDB provides binary labels (splice-altering vs. not)
when what we need is position-level delta scores. The label hierarchy
from the analysis document captures this:

```
Level 0: Position-level      "Which positions are splice sites?"        (GTF — abundant)
Level 1: Variant-level       "Does this variant affect splicing?"       (SpliceVarDB — binary)
Level 2: Effect type          "Gain or loss? Donor or acceptor?"        (inferred — noisy)
Level 3: Position-level       "At which position does the effect occur?" (absent in SpliceVarDB)
Level 3.5: Junction-level    "Which donor pairs with which acceptor?"   (RNA-seq — direct evidence)
Level 4: Magnitude (PSI)     "How strong is the effect?"                (RNA-seq — quantitative)
```

M4 needs Level 3+ labels to train effectively. GTEx sQTLs and junction
reads can fill this gap, but significant data engineering is required.

### Cancer cell line bias becomes signal

An interesting inversion happens with M4: the cancer cell line bias in
ENCODE data (K562, HepG2 — both cancer lines) that's problematic for M1
canonical prediction becomes *useful signal* for M4. Cancer cells have
globally altered chromatin accessibility and RBP expression patterns that
drive aberrant splicing. RBP binding patterns specific to cancer cell lines
may indicate positions susceptible to splicing dysregulation — exactly what
M4 needs to learn.

---

## How the Four Variants Relate

### The junction toggle

The most elegant aspect of the M1-M4 design is how junction data changes
role across variants:

```
M1:  Junction = (not used)       → canonical sites don't need RNA-seq
M2:  Junction = FEATURE          → "this site has RNA-seq support in 3 tissues"
M3:  Junction = TARGET           → "can we predict junction support without RNA-seq?"
M4:  Junction = VALIDATION       → "does the predicted delta match real junction changes?"
```

This is possible because the YAML-driven feature pipeline makes
modalities composable. Switching junction from feature to target is a
one-line config change — swap `full_stack.yaml` for `meta_m3_novel.yaml`.

### Progressive difficulty

The variants form a progression from easiest to hardest:

```
M1 (canonical)     →  "Is this a known splice site?"
                       Labels: abundant (millions from GTF)
                       Base model: excellent (99.5%+)
                       Room for improvement: minimal

M2 (alternative)   →  "Is this an alternative splice site?"
                       Labels: moderate (Ensembl annotations, junction reads)
                       Base model: weak at these sites
                       Room for improvement: substantial

M3 (novel)         →  "Is this a completely unannotated splice site?"
                       Labels: indirect (junction reads as proxy)
                       Base model: no training signal for these
                       Room for improvement: maximal

M4 (induced)       →  "Does this perturbation create/destroy a splice site?"
                       Labels: scarce (SpliceVarDB, sQTLs)
                       Base model: not designed for perturbations
                       Room for improvement: unknown (label bottleneck)
```

Each step increases the value of multimodal evidence. At M1, base scores
alone achieve 99.5%. At M3, base scores are necessary but far from
sufficient — conservation, chromatin state, and RBP binding carry the
marginal information that separates true novel sites from noise.

### Shared infrastructure

All four variants use the same codebase:

| Component | Path | Shared across |
|-----------|------|---------------|
| Feature pipeline | `src/.../features/pipeline.py` | All |
| Modality registry | `src/.../features/modalities/` | All |
| Feature schema | `src/.../meta_layer/core/feature_schema.py` | All |
| Workflow script | `examples/features/06_multimodal_genome_workflow.py` | All |
| XGBoost baseline | `examples/meta_layer/01_xgboost_baseline.py` | M1, M2 |
| Ablation analysis | `examples/meta_layer/03_modality_ablation.py` | M1, M2 |

The only things that change between variants are:
1. The YAML config (which modalities are enabled)
2. The training script (what column is the label)
3. The evaluation metrics (accuracy vs. recall at novel sites vs. delta correlation)

---

## Sequence-Level Architecture (M*-S models)

The sequence-level meta-layer is a **two-pass, three-stream refinement
model** implemented in `meta_splice_model_v3.py` (class: `MetaSpliceModel`,
config: `MetaSpliceConfig`).

### Two-pass inference

```
Pass 1 (frozen): DNA sequence → OpenSpliceAI → base_scores [L, 3]
                  Genomic coords → DenseFeatureExtractor → mm_features [L, C]
Pass 2 (trainable): sequence + base_scores + mm_features → MetaSpliceModel → refined [L, 3]
```

The base model runs first to produce raw per-nucleotide scores. The
meta-layer then refines these using multimodal evidence. A learnable
residual blend (`α × refined + (1-α) × base`) ensures the meta-layer
never degrades below the base model.

### Three-stream architecture

```
Stream A: DNA sequence [B, 4, L]       → dilated 1D CNN (8 blocks) → [B, H, L]
Stream B: Base model scores [B, L, 3]  → per-position MLP          → [B, H, L]
Stream C: Multimodal features [B, C, L] → 1D CNN (4 blocks)        → [B, H, L]
                        |
                cat → [B, 3H, L] → fusion CNN (4 blocks) → [B, H, L]
                        |
                output head → logits [B, L, 3] → residual blend → [B, L, 3]
```

- **H=32** (hidden dim), **~370K parameters**
- Dilated convolutions with rates [1,1,1,1,4,4,4,4] → **400 bp receptive field**
- No transformers — runs on M1 Mac (MPS backend)
- Window-based training (W=5001), sliding window inference for full genes

### Dense multimodal input channels (C=9)

These features are extracted at **every position** via BigWig/Parquet
lookups, not at pre-sampled positions:

| Channel | Feature | Source | Signal |
|---------|---------|--------|--------|
| 0 | phylop_score | PhyloP BigWig | Dense |
| 1 | phastcons_score | PhastCons BigWig | Dense |
| 2 | h3k36me3_max | ENCODE ChIP-seq BigWig | Dense |
| 3 | h3k4me3_max | ENCODE ChIP-seq BigWig | Dense |
| 4 | atac_max | ATAC-seq BigWig | Dense |
| 5 | dnase_max | DNase-seq BigWig | Dense |
| 6 | junction_log1p | GTEx RNA-seq (0-filled) | Sparse |
| 7 | junction_has_support | Binary junction indicator | Sparse |
| 8 | rbp_n_bound | ENCODE eCLIP (0-filled) | Sparse |

For M3-S, channels 6-7 are excluded (junction becomes the prediction
target), giving C=7.

### Variant support (M4-S)

```python
ref_probs, alt_probs, delta = model.predict_with_delta(
    ref_seq, alt_seq, ref_base, alt_base, ref_mm, alt_mm,
)
# delta: [B, L, 3] — per-nucleotide splice score changes
```

---

## Current Status and Next Steps

### Position-level models (-P)

| Variant | Status | Key results |
|---------|--------|-------------|
| M1-P | **Done** (full-genome, SpliceAI split) | 99.74% acc, 103 features, 6.27M positions |
| M1-P ablation | **Done** | Multimodal reduces FN -62%, FP -68% vs base-only |
| M2-P | Planned | Ensembl evaluation on tabular features |
| M3-P | Planned | Junction-as-target with tabular features |

### Sequence-level models (-S)

| Variant | Status | Next milestone |
|---------|--------|----------------|
| M1-S | **Done** (epoch 46, PR-AUC 0.9899, 99.99% acc) | M2a evaluation |
| M2a | Next | Evaluate M1-S on Ensembl \ MANE sites |
| M2b | Planned | Evaluate M1-S on GENCODE \ MANE sites |
| M2c-S | Planned | Train on GENCODE with tier-weighted labels |
| M2d-S | Planned | Train with junction-informed soft labels |
| M2e-S | Research | Tissue-conditioned input (architecture change) |
| M3-S | Planned | Train with junction as target (C=7) |
| M4-S | `predict_with_delta()` implemented | Data engineering: SpliceVarDB + GTEx sQTLs |

### Data pipeline

| Component | Status |
|-----------|--------|
| Full-genome feature parquets (9 modalities, 24 chroms) | **Done** (2.88 GB, 116 cols) |
| FM embeddings (10th modality, Evo2 7B) | Extraction in progress on GPU pod |
| DenseFeatureExtractor (BigWig → [L, C] arrays) | **Done** |
| SequenceLevelDataset (disk-backed .npz cache) | **Done** |
| ShardedSequenceLevelDataset (HDF5 shards) | **Done** (not yet tested) |
| Dense label arrays (`build_splice_labels()`) | **Done** |
| Ensembl base score precomputation | Needed for M2a Option A |
| GENCODE splice site annotations | Needed for M2b/M2c |

---

## References

### Methods documentation (read in order)

1. [00_model_variants_m1_m4.md](00_model_variants_m1_m4.md) — this document
2. [01_label_hierarchy_and_weak_supervision.md](01_label_hierarchy_and_weak_supervision.md) — label hierarchy (L0-L4), weak-supervision framing
3. [02_annotation_driven_splice_prediction.md](02_annotation_driven_splice_prediction.md) — annotation as latent variable, tier-based confidence
4. [03_virtual_transcripts_and_junction_pairing.md](03_virtual_transcripts_and_junction_pairing.md) — junction-level prediction, donor-acceptor pairing
5. [04_data_sources_and_landscape.md](04_data_sources_and_landscape.md) — GTEx, SpliceVault, ENCODE data sources + ML landscape
6. [05_m2_variant_formulations.md](05_m2_variant_formulations.md) — detailed M2a-f training and evaluation protocols

### Within the codebase

- **Sequence-level model**: `src/.../meta_layer/models/meta_splice_model_v3.py`
- **Shared training utilities**: `src/.../meta_layer/training/data_utils.py`
- **Dense feature extractor**: `src/.../features/dense_feature_extractor.py`
- **Disk-backed dataset**: `src/.../meta_layer/data/sequence_level_dataset.py`
- **HDF5 shard packing**: `src/.../meta_layer/data/shard_packing.py`
- **Training script**: `examples/meta_layer/07_train_sequence_model.py`
- **M1-P full-genome results**: `examples/meta_layer/docs/m1_fullgenome_results.md`
- **Dense label builder**: `foundation_models/foundation_models/utils/chunking.py`
- **XGBoost baseline (M1-P)**: `examples/meta_layer/01_xgboost_baseline.py`
- **Modality ablation**: `examples/meta_layer/03_modality_ablation.py`
- **M3 config**: `examples/features/configs/meta_m3_novel.yaml`
- **Full-stack config (M2)**: `examples/features/configs/full_stack.yaml`

### Meta-SpliceAI archive

The predecessor project (Meta-SpliceAI) ran four experiments on variant-level
splice prediction. Results and methods are archived in
`docs/meta_layer/meta-spliceai-archive/`. Key finding: the bottleneck is
label quality (binary variant labels), not model capacity (best: r=0.41
on validated delta prediction). The M2 series addresses this through
annotation-tier and junction-weighted labeling.

### External

- Jaganathan et al. (2019). Predicting Splicing from Primary Sequence with
  Deep Learning. *Cell* 176, 535-548. (SpliceAI)
- Chen et al. (2024). OpenSpliceAI: An Efficient, Modular Implementation
  of SpliceAI. *bioRxiv*.
- GTEx Consortium (2020). The GTEx Consortium atlas of genetic regulatory
  effects across human tissues. *Science* 369, 1318-1330.
