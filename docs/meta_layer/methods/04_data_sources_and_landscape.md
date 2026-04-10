# Data Sources, Derivation Strategies, and Landscape Analysis

**Created**: March 2026
**Prerequisite**: [01_alternative_splice_prediction_analysis.md](01_alternative_splice_prediction_analysis.md), [02_virtual_transcripts.md](02_virtual_transcripts.md)

---

## Motivation

Docs 01 and 02 define *what* we want to predict (alternative splice sites induced by external
factors, at junction level) and *how* to frame the labels (Strategies 1-3, Levels 0-4). This
document addresses two practical questions:

1. **Where do we get junction-level labeled data** tied to specific external factors?
2. **What already exists** in the ML landscape, so we don't reinvent the wheel?

---

## Part I: Data Sources for Junction-Level Labels

The fundamental bottleneck for predicting induced splice sites is not algorithm design — it's
assembling junction-level labeled datasets at sufficient scale. Below are concrete sources,
ordered by effort required.

### Tier 1: Available Now (No New Data Generation)

#### GTEx Splice QTLs (sQTLs)

The single most actionable data source. GTEx v8 publishes pre-computed associations between
genetic variants and differential junction usage across 49 tissues.

| Attribute | Detail |
|-----------|--------|
| What it provides | Variant → junction usage change (intron excision ratio) per tissue |
| Scale | ~50K significant sQTLs across tissues; many are tissue-specific |
| Label format | (variant, gene, junction_cluster, effect_size, tissue, p-value) |
| How to derive splice-site labels | Each sQTL's intron cluster defines which junctions are gained/lost/shifted; the variant is the external factor |
| Key advantage | Pre-computed — no need to re-process BAMs. Population-level (common variants) |
| Limitation | Biased toward common variants (MAF > 1%); rare pathogenic variants underrepresented |
| Access | [GTEx Portal](https://gtexportal.org/home/downloads) → sQTL summary statistics |
| Processing tools | `leafcutter` (intron clustering), `tensorQTL` (association testing) |

**Integration path**: Download sQTL summary statistics → filter to significant associations →
map intron clusters to donor-acceptor coordinates → format as
`(variant, gene, junction_gained/lost, tissue, effect_size)`.

#### SpliceVault

A lookup database aggregating 335K+ junction-level observations from clinical RNA-seq,
cross-referenced with known pathogenic variants (Dawes et al., Nature Genetics 2023).

| Attribute | Detail |
|-----------|--------|
| What it provides | Novel/cryptic junction coordinates (donor-acceptor pairs) with read counts |
| Scale | 335K junction observations from ~300K RNA-seq samples (via recount3) |
| Key insight | Cryptic splice sites activated by pathogenic variants also appear as rare stochastic events in healthy population data |
| Performance | Top-4 unannotated events predict mis-splicing outcome with 92% sensitivity |
| Label format | Junction coordinates + read counts + whether annotated or novel |
| Key advantage | Bridges "variant X causes aberrant splicing" → "variant X creates junction (d', a') with N reads" |
| Limitation | Lookup-based (not a trained model); requires the junction to have been observed at least once |
| Access | [github.com/kidsneuro-lab/SpliceVault](https://github.com/kidsneuro-lab/SpliceVault) |
| Paper | [Nature Genetics 2023](https://www.nature.com/articles/s41588-022-01293-8) |

**Integration path**: Query SpliceVault for junctions near known splice-altering variants →
provides ground-truth junction coordinates and read support → format matches our
`splice_sites_enhanced.tsv` schema with additional junction-pair columns.

#### ClinVar + SpliceVarDB (Variant-Level, Needs Junction Mapping)

These provide variant-level labels ("this variant affects splicing") but not junction-level
coordinates. Useful as a starting point when cross-referenced with SpliceVault or GTEx.

| Source | Scale | Label Level | Limitation |
|--------|-------|-------------|------------|
| SpliceVarDB | ~50K variants | Binary (splice-altering or not) | No junction positions |
| ClinVar (splice subset) | ~15K splice variants | Review status + disease | Inconsistent annotation depth |

### Tier 2: Moderate Effort

#### TCGA Junction Quantification

For cancer/disease-specific alternative splicing. Matched somatic variant calls + tumor RNA-seq
across 33 cancer types.

| Attribute | Detail |
|-----------|--------|
| What it provides | Somatic variant → aberrant junction usage in same tumor |
| Scale | ~11K patients across 33 cancer types |
| Key advantage | Somatic mutations near splice sites directly linked to tumor-specific junction usage |
| Processing | SplAdder reprocessing provides alternative splicing event catalogs (exon skip, intron retention, alt 3'/5') |
| Related resource | PCAWG (Pan-Cancer Analysis of Whole Genomes) has systematically cataloged splicing-associated somatic mutations |
| Effort | Need to download and process BAMs or use pre-computed junction tables |

**Integration path**: Use TCGA junction tables (or SplAdder output) → filter to junctions
near somatic mutations → label as cancer-specific alternative splicing events.

#### ENCODE Perturbation RNA-seq

Knockdown of splicing regulators (SRSF1, hnRNPA1, U2AF1, SF3B1, etc.) with matched RNA-seq.
Junction changes after knockdown directly reveal which junctions depend on that factor.

| Attribute | Detail |
|-----------|--------|
| What it provides | Splicing factor knockdown → differential junction usage |
| Experiment types | shRNA knockdown, CRISPRi, in K562 and HepG2 cell lines |
| Key advantage | Clean causal signal — if junction X disappears when factor Y is knocked down, X depends on Y |
| Scale | ~200+ splicing-related knockdown experiments with RNA-seq |
| Access | [ENCODE Portal](https://www.encodeproject.org/) → search "RNA-seq" + "shRNA" + splicing factor |

**Integration path**: Download perturbation vs. control RNA-seq → run STAR junction
quantification → compute differential junction usage → label as factor-dependent junctions.

### Tier 3: Heavy Lift, Highest Value

#### Reprocess GTEx/TCGA BAMs with Uniform Junction Calling

The most comprehensive approach — run all ~17K GTEx + ~11K TCGA samples through a uniform
STAR 2-pass pipeline to get per-sample junction counts.

| Attribute | Detail |
|-----------|--------|
| What it enables | Discovery of rare-variant sQTLs not in the published catalog |
| Scale | ~28K samples, ~billions of junction observations |
| Compute cost | ~$5-10K in cloud compute (or months on institutional cluster) |
| Key advantage | Enables junction-level labels for rare pathogenic variants |
| Alternative | Use recount3 (pre-processed junction tables for ~750K public RNA-seq samples) |

**Note**: recount3 provides pre-computed junction tables that may be sufficient without
reprocessing BAMs. See [recount3.org](https://recount.bio/).

#### Drug Perturbation and ASO Studies

For the hardest case — condition-induced alternative splicing:

| Source | What It Provides | Example |
|--------|-----------------|---------|
| Splicing modulator studies | Drug → junction changes | Risdiplam (SMN2 exon 7 inclusion) |
| ASO (antisense oligonucleotide) studies | Targeted splice site blocking → clean junction changes | Nusinersen (SMN2), Eteplirsen (DMD exon 51 skip) |
| Stress response RNA-seq | Cellular stress → differential splicing | Heat shock, hypoxia, DNA damage |

These produce the cleanest training signal for perturbation-aware models but require
study-by-study curation.

---

## Part II: Derivation Strategy — Building the Dataset Incrementally

```
Phase 1 (weeks): Curate existing resources
  ├── GTEx sQTLs → ~50K variant-junction associations (common variants)
  ├── SpliceVault → ~335K junction observations (pathogenic variants)
  └── Format: (variant, gene, junction_gained/lost, tissue, read_count)
      ≈ 100K unique variant-junction associations

Phase 2 (months): Expand with processed data
  ├── TCGA junctions → somatic variant-junction (cancer-specific)
  ├── ENCODE knockdowns → factor-junction dependencies
  └── Format: same + external_factor column (variant / factor / condition)
      ≈ 500K associations

Phase 3 (ongoing): Deep integration
  ├── recount3 junction tables → rare variant sQTLs
  ├── Drug/ASO perturbation studies → condition-specific junctions
  └── ≈ 1M+ associations across variant, factor, and condition types
```

The key insight: **all external factors produce the same output format** — a set of
junctions that are gained or lost relative to a reference. The meta-layer doesn't need to
know whether the cause is a genetic variant, a splicing factor knockdown, or a drug — it
learns the mapping from (sequence + context features + perturbation signal) to
(junction usage change).

---

## Part III: Landscape Analysis — What Already Exists (as of March 2026)

### Per-Position Splice Site Prediction (Mature)

| Tool | Architecture | Context | Tissue-Aware | Training Data | Key Limitation |
|------|-------------|---------|--------------|---------------|----------------|
| **SpliceAI** (2019) | ResNet, 3-channel output | 10kb | No | GENCODE V24lift37 canonical (hg19) | Per-position only; argmax discards information |
| **CI-SpliceAI** (2022) | Same as SpliceAI | 10kb | No | Curated alt sites | Marginal improvement (~1%) |
| **OpenSpliceAI** (2025) | Same architecture, retrained | 10kb | No | RefSeq MANE v1.3 (GRCh38) | Per-position only |
| **Pangolin** (2022) | SpliceAI architecture | 5kb | 4 tissues | RNA-seq junction counts (4 tissues x 4 species) | Bug affected ~28% of hg38 scores in 2024 |
| **SpTransformer** (2024) | Transformer | Variable | 56 tissues | GTEx + GENCODE | Per-position; claims 83% cross-tissue inference |

**Our base layer**: SpliceAI + OpenSpliceAI. Feature pipeline extracts 43+ derived features.

### Variant Effect on Splicing (Binary/Categorical)

| Tool | Prediction Level | Tissue-Aware | Data Sources | Key Advance |
|------|-----------------|--------------|-------------|-------------|
| **AbSplice** (2023) | Binary (aberrant or not) | 50 tissues | GTEx rare variants + FRASER | First tissue-specific aberrant splicing classifier; 60% precision |
| **AbSplice2** (2025) | Binary + continuous usage | Developmental stages | GTEx + FRASER2 + Pangolin | Adds developmental stage predictions |
| **MMSplice/MTSplice** (2019/2021) | Delta-logit-PSI (exon-level) | 56 tissues (MTSplice) | VEX-seq + GTEx | Limited to cassette exon events |
| **SpliceVault** (2023) | Outcome type lookup | No | 335K RNA-seq samples | 92% sensitivity for outcome prediction; lookup, not learned |

**Gap**: AbSplice predicts *whether* aberrant splicing occurs but not *which junction*.
SpliceVault predicts outcomes but only via lookup of previously observed events.

### Junction-Level Prediction (Emerging)

| Tool | What It Predicts | Architecture | Open Source | Key Limitation |
|------|-----------------|-------------|-------------|----------------|
| **Splam** (2024) | P(valid junction \| donor, acceptor) | Deep ResNet, 800nt input | Yes | Requires enumeration of candidate pairs; classifies, doesn't discover |
| **AlphaGenome** (2025) | 2D junction strength (donor x acceptor) | 1Mb U-Net + Transformer | **API only** | Non-commercial; cannot train/extend; doesn't model perturbations |

**AlphaGenome** is the state-of-the-art for junction-level prediction from sequence — its 2D
pairwise branch explicitly scores donor-acceptor pairs. However, it is API-only, non-commercial,
and does not model external perturbation factors.

**Splam** is the closest open-source junction-level model, but it classifies given candidate
pairs rather than discovering novel junctions from genomic regions.

### Perturbation-Aware Splicing (Very Early)

| Tool | External Factor | Architecture | Status |
|------|----------------|-------------|--------|
| **ChemSplice** (2024) | Drug perturbations | BERT (AllSplice) + chemical embeddings | Preprint only; not peer-reviewed |
| **KATMAP** (2025) | Splicing factor knockdowns | Linear regression (interpretable) | Published; predicts factor targets, not junction outcomes |
| **TrASPr+BOS** (2025) | Experimental conditions | Generative VAE + Bayesian optimization | Predicts PSI/dPSI; can design sequences |

**This space is wide open.** ChemSplice is the only model attempting drug-perturbed junction
prediction, and it's a preprint. No tool combines variant effects + perturbation context +
junction-level prediction in a single framework.

### Foundation Models for Splicing

| Model | Splice Capability | Junction-Level | Zero-Shot |
|-------|------------------|---------------|-----------|
| **Evo 2** (2025) | Variant effect via log-likelihood ratio | No | Yes (best zero-shot performance) |
| **Borzoi** (2025) | RNA-seq coverage prediction → indirect splice scoring | Indirect (coverage-based) | No (trained on specific tracks) |
| **SpliceBERT** (2024) | Splice site classification after fine-tuning | No | No |
| **AlphaGenome** (2025) | Full junction-level scoring | **Yes (2D branch)** | No (trained on functional genomics tracks) |

**Our approach**: Use Evo2 embeddings as features in the meta-layer, not as a standalone
predictor. The sparse exon classifier demonstrates that Evo2 embeddings capture splicing-relevant
structure (see `examples/foundation_models/05_sparse_exon_classifier.py`).

---

## Part IV: What's Novel About Our Approach

### The Gap in the Landscape

No existing open-source tool combines all three of:

1. **Junction-level prediction** (donor-acceptor pairing, not just per-position)
2. **External factor conditioning** (variants, disease, perturbations)
3. **Tissue specificity** (different junctions active in different contexts)

AlphaGenome achieves (1) but not (2) or (3), and is API-only.
AbSplice achieves (3) partially but not (1).
ChemSplice attempts (2) but is a preprint with narrow scope.

### Our Position

```
                        Per-Position    Junction-Level
                        ────────────    ──────────────
Canonical only          SpliceAI ✓      Splam ✓
                        OpenSpliceAI ✓  AlphaGenome ✓ (API)

Variant-induced         AbSplice ✓      ← GAP: no open tool
                        SpliceVault ✓     predicts which junctions
                        (lookup)          a variant creates/destroys

Factor/condition-       ChemSplice       ← GAP: no tool predicts
induced                 (preprint)        junction outcome under
                        KATMAP (linear)   perturbation
```

The meta-layer we're building aims to fill the bottom-right quadrant:
**junction-level predictions conditioned on external factors**, using a multimodal
feature fusion approach (base scores + conservation + epigenetic + junction evidence +
perturbation signal).

### Why Multimodal Fusion (Not End-to-End)

AlphaGenome shows that end-to-end 2D junction prediction is possible, but requires
massive compute (1Mb input, thousands of output tracks, DeepMind-scale training).

Our approach is complementary and more practical:
- **Base models** (SpliceAI/OpenSpliceAI) handle the "splicing grammar" from sequence
- **Conservation** tells us which sites are under selective constraint (and which are evolvable)
- **Epigenetic context** tells us which sites are active in a given cell type
- **Junction reads** provide ground truth for donor-acceptor pairing
- **Meta-layer** learns to combine these signals, conditioned on external factors

This is more data-efficient than training a single massive model, and allows each modality
to be updated independently as better data becomes available.

---

## Part V: Recommended Reading Order

For someone new to this problem:

1. **01_alternative_splice_prediction_analysis.md** — Problem formulation, label hierarchy,
   initial experiments showing the difficulty (best: r=0.41)
2. **02_virtual_transcripts.md** — Why junction-level prediction is needed, three label
   strategies, the pairing problem
3. **This document** — Concrete data sources, what exists in the landscape, what's novel
4. `docs/foundation_models/evo2/junction_support_labels.md` — How to build soft labels
   from junction read support (implementation-level detail)
5. `docs/isoform_discovery/README.md` — The end goal: context-aware isoform discovery

---

## References

### Data Sources
- GTEx sQTLs: [gtexportal.org](https://gtexportal.org/home/downloads)
- SpliceVault: [Nature Genetics 2023](https://www.nature.com/articles/s41588-022-01293-8), [github.com/kidsneuro-lab/SpliceVault](https://github.com/kidsneuro-lab/SpliceVault)
- recount3: [recount.bio](https://recount.bio/)
- ENCODE: [encodeproject.org](https://www.encodeproject.org/)
- TCGA SplAdder: [bioinformatics.mdanderson.org/TCGA_Splicing](https://bioinformatics.mdanderson.org/TCGA_Splicing)

### Tools and Models
- SpliceAI: [Cell 2019](https://doi.org/10.1016/j.cell.2018.12.015)
- OpenSpliceAI: [eLife 2025](https://elifesciences.org/articles/107454)
- Pangolin: [Genome Biology 2022](https://doi.org/10.1186/s13059-022-02664-4)
- AbSplice: [Nature Genetics 2023](https://doi.org/10.1038/s41588-023-01373-3), [github.com/gagneurlab/absplice](https://github.com/gagneurlab/absplice)
- AbSplice2: [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.07.16.665183v2)
- SpliceMap: [github.com/gagneurlab/splicemap](https://github.com/gagneurlab/splicemap)
- MMSplice/MTSplice: [Nature Methods 2019](https://doi.org/10.1038/s41592-019-0339-6), [Genome Biology 2021](https://doi.org/10.1186/s13059-021-02273-7)
- Splam: [Genome Biology 2024](https://doi.org/10.1186/s13059-024-03379-4)
- AlphaGenome: [Nature 2025](https://doi.org/10.1038/s41586-025-10014-0)
- Evo 2: [bioRxiv 2025](https://doi.org/10.1101/2025.02.18.638918)
- Borzoi: [Nature Genetics 2025](https://doi.org/10.1038/s41588-024-02053-6)
- ChemSplice/AllSplice: [bioRxiv 2024](https://doi.org/10.1101/2024.03.20.585793)
- KATMAP: [Nature Biotechnology 2025](https://doi.org/10.1038/s41587-025-02881-9)
- SpTransformer: [Nature Communications 2024](https://doi.org/10.1038/s41467-024-53088-6)
- SpliceBERT: [Briefings in Bioinformatics 2024](https://doi.org/10.1093/bib/bbae163)
- TrASPr+BOS: [eLife 2025](https://elifesciences.org/reviewed-preprints/106043)

### Within Codebase
- Junction support labels: `docs/foundation_models/evo2/junction_support_labels.md`
- Isoform discovery vision: `docs/isoform_discovery/README.md`
- Feature pipeline: `src/agentic_spliceai/splice_engine/features/`
- Multimodal exploration: `examples/features/05_multimodal_exploration.py`
- Sparse exon classifier: `examples/foundation_models/05_sparse_exon_classifier.py`
