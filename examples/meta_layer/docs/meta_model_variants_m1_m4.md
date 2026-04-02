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

The XGBoost baseline on chr19-22 (466K positions) showed:

| Model | Accuracy | PR-AUC (donor) | PR-AUC (acceptor) |
|-------|----------|----------------|-------------------|
| Base scores only | 99.62% | 0.996 | 0.995 |
| Full-stack (M1) | **99.78%** | **0.999** | **0.998** |

The improvement is real but modest — because the canonical task is nearly
saturated. The more telling result came from SHAP analysis: conservation
and genomic context contribute 6-7% of the model's decision-making (by
SHAP), despite appearing to contribute <1% by XGBoost's gain metric.
This hidden importance becomes critical in M2-M4 where base scores are
no longer sufficient.

### Limitations

M1 optimizes for the wrong question. 99.78% accuracy on canonical sites
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

### M2a: Ensembl vs. MANE evaluation

A particularly clean evaluation for M2 exploits the annotation gap between
MANE and Ensembl. OpenSpliceAI was trained on MANE/GRCh38, which documents
mostly canonical protein-coding transcripts (~19K genes). Ensembl/GRCh38
contains a superset with more isoform diversity.

The splice sites in (Ensembl \ MANE) — present in Ensembl but absent from
MANE — are alternative sites that the base model was never trained to
recognize. OpenSpliceAI's base scores will be systematically weak at these
positions. The question: **can the meta-layer rescue them using multimodal
evidence?**

```
Evaluation set:     Ensembl/GRCh38 splice sites NOT in MANE/GRCh38
Baseline:           OpenSpliceAI base scores (expect low recall)
Meta-layer (M2):    Base scores + all modalities → measure recall improvement
Ablation:           Which modalities contribute most to the rescue?
```

This is conceptually related to the "virtual transcripts" framework from
`docs/meta_layer/predicting_induced_splice_sites/02_virtual_transcripts.md`
— except instead of variant-induced splice sites, these are
isoform-diversity-induced sites that already exist in reference annotations.
No perturbation required; the alternative sites are readily available as
ground truth.

### What M2 learns that M1 cannot

M1 sees canonical sites where the base model already excels. M2 operates
in the regime where the base model is weakest — alternative sites with
lower base scores, tissue-specific usage, and regulatory complexity. This
is where multimodality earns its keep.

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
| M2-P | Planned | `04_m2a_ensembl_vs_mane.py` |
| M3-P | Planned | `05_m3_novel_prediction.py` |

### Sequence-level models (-S)

| Variant | Status | Next milestone |
|---------|--------|----------------|
| M1-S | Model architecture done | Dense data pipeline → training script |
| M2-S | Same architecture as M1-S | Evaluate on Ensembl-only alternative sites |
| M3-S | Config ready (7 channels, binary output) | Train with junction as target |
| M4-S | `predict_with_delta()` implemented | Data engineering: SpliceVarDB + GTEx sQTLs |

### Data pipeline

| Component | Status |
|-----------|--------|
| Full-genome feature parquets (9 modalities, 24 chroms) | **Done** (2.88 GB, 116 cols) |
| FM embeddings (10th modality, Evo2 7B) | Extraction in progress on GPU pod |
| DenseFeatureExtractor (BigWig → [L, C] arrays) | Planned |
| SequenceLevelDataset (window-level PyTorch Dataset) | Planned |
| Dense label arrays (from splice_sites_enhanced.tsv) | Existing infrastructure (`build_splice_labels()`) |

---

## References

### Within the codebase

- **Sequence-level model**:
  `src/.../meta_layer/models/meta_splice_model_v3.py` (`MetaSpliceModel`, `MetaSpliceConfig`)
- **Shared training utilities**:
  `src/.../meta_layer/training/data_utils.py`
- **M1-P full-genome results**:
  `examples/meta_layer/docs/m1_fullgenome_results.md`
- **Ground truth generation**:
  `examples/data_preparation/04_generate_ground_truth.py`
- **Dense label builder**:
  `foundation_models/foundation_models/utils/chunking.py` (`build_splice_labels()`)
- Alternative splice prediction analysis:
  `docs/meta_layer/predicting_induced_splice_sites/01_alternative_splice_prediction_analysis.md`
- Virtual transcripts framework:
  `docs/meta_layer/predicting_induced_splice_sites/02_virtual_transcripts.md`
- Data sources and landscape:
  `docs/meta_layer/predicting_induced_splice_sites/03_data_sources_and_landscape.md`
- XGBoost baseline (M1-P):
  `examples/meta_layer/01_xgboost_baseline.py`
- Modality ablation:
  `examples/meta_layer/03_modality_ablation.py`
- M3 config:
  `examples/features/configs/meta_m3_novel.yaml`
- Full-stack config (M2):
  `examples/features/configs/full_stack.yaml`

### Key sessions

- M1-M4 formalization: `dev/sessions/2026-03-17`
- Junction wiring + full-stack baseline: `dev/sessions/2026-03-18`
- SHAP analysis + ablation: `dev/sessions/2026-03-19`

### External

- Jaganathan et al. (2019). Predicting Splicing from Primary Sequence with
  Deep Learning. *Cell* 176, 535-548. (SpliceAI)
- Chen et al. (2024). OpenSpliceAI: An Efficient, Modular Implementation
  of SpliceAI. *bioRxiv*.
- GTEx Consortium (2020). The GTEx Consortium atlas of genetic regulatory
  effects across human tissues. *Science* 369, 1318-1330.
