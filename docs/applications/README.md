# Applications — From Discovery to Therapeutics

This directory (`docs/applications/`) contains domain-specific workflows and use cases for Agentic-SpliceAI. Each application demonstrates how the multi-layer pipeline translates computational predictions into biological and clinical insights. As applications mature, they graduate to `docs/products/` for production deployment guides.

---

## The Translational Pathway

### Current State: Limited by Canonical Annotations

Drug discovery today is constrained by incomplete gene annotations:

```
Known Genes (20,000)
    |
Canonical Isoforms (~20,000 in MANE)
    |
Druggable Targets (~3,000)
    |
Approved Drugs (~1,500 targets)
```

**Bottleneck**: 90% of splice sites are non-canonical. Current annotations capture only the most common isoforms, leaving the vast majority of biologically active splice variants undiscovered.

### Expanding the Druggable Genome

Agentic-SpliceAI discovers context-specific isoforms to expand the therapeutic space. Consider **BRCA1** as a concrete example:

| Scope | Splice Sites | Isoforms | Targeting Strategy |
|-------|-------------|----------|-------------------|
| Canonical (MANE) | 44 | 1-2 main | Target "BRCA1" broadly |
| Full potential (Ensembl + Discovery) | 1,218+ (27x more) | 100+ (tissue/disease-specific) | Target tumor-specific isoforms selectively |

By moving beyond canonical annotations, we unlock 10-100x more therapeutic targets with the potential for reduced toxicity through isoform-selective drug design.

---

## Application Domains

> **Note**: CLI commands shown in the workflows below reflect planned Phase 7-9 functionality. Current capabilities cover base layer prediction, feature engineering, meta layer training (Phases 1-6), and variant effect prediction (M4 Phase 1A+1B).

### 0. Variant Analysis: Splice Effect Prediction and Validation

**Status**: Active development (M4 Phase 1A+1B complete, Phase 2-3 planned)

Predict how genetic variants disrupt splicing, identify cryptic splice site locations, and validate against experimental evidence.

- [Saturation Mutagenesis & SpliceVarDB Validation](variant_analysis/saturation_mutagenesis_and_validation.md) — Phase 3 plan for gene-wide splice vulnerability mapping with SpliceVarDB cross-validation and GTEx junction verification
- [Current validation results](../../examples/variant_analysis/results/variant_effect_validation.md) — 13 disease-gene variants, 4 RNA-seq validated cases from SpliceAI paper

### 1. Oncology: Cancer-Specific Isoform Targets

**Challenge**: Tumors express aberrant splice isoforms that drive proliferation, evade apoptosis, and resist therapy. These cancer-specific isoforms are invisible to standard genomic pipelines.

**Workflow**:

```bash
# Predict splice sites for BRCA1 with cancer context
agentic-spliceai-predict --genes BRCA1 --base-model openspliceai

# Discover cancer-specific isoforms (Phase 9)
agentic-spliceai discover --genes BRCA1 \
  --context cancer:breast \
  --evidence-sources gtex,tcga,clinvar

# Validate with agentic layer (Phase 7)
agentic-spliceai validate --isoforms output/brca1_novel.parquet \
  --agents literature,expression,clinical
```

**Impact**: Identify tumor-specific BRCA1 isoforms for isoform-selective inhibitors that spare normal tissue function.

### 2. Clinical Genetics: VUS Interpretation

**Challenge**: Thousands of variants of uncertain significance (VUS) in clinical sequencing may affect splicing but lack functional evidence for classification.

**Workflow**:

```bash
# Analyze splice impact of TP53 variants (Phase 8)
agentic-spliceai variant --vcf patient_variants.vcf \
  --genes TP53 \
  --clinvar-cross-reference

# Generate clinical report (Phase 7)
agentic-spliceai report --variants output/tp53_splice_impact.parquet \
  --format clinical \
  --evidence-level acmg
```

**Impact**: Reclassify VUS variants by quantifying their splice-disrupting potential with multi-source evidence, enabling actionable clinical decisions.

### 3. Neurology: Brain-Specific Isoforms

**Challenge**: The brain has the highest rate of alternative splicing of any tissue, with many neurological disorders linked to isoform dysregulation. Autism risk genes alone have hundreds of brain-specific splice variants.

**Workflow**:

```bash
# Discover brain-specific isoforms for autism risk genes (Phase 9)
agentic-spliceai discover \
  --genes SHANK3 NRXN1 SYNGAP1 CHD8 \
  --context tissue:brain \
  --evidence-sources gtex,brainspan

# Cross-reference with known autism associations (Phase 7)
agentic-spliceai validate --isoforms output/asd_novel.parquet \
  --agents literature,expression,conservation
```

**Impact**: Map the brain-specific isoform landscape of autism risk genes to identify therapeutic targets for splice-modulating interventions.

### 4. Drug Development: Isoform-Selective Therapeutics

**Challenge**: Most drugs target proteins without distinguishing between isoforms, leading to off-target effects when disease-relevant and normal isoforms are both inhibited.

**Workflow**:

```bash
# Identify druggable isoform-specific regions (Phase 9)
agentic-spliceai druggability --genes BCL2 MCL1 \
  --isoform-catalog output/novel_isoforms.parquet \
  --assess-selectivity

# Generate drug target report (Phase 7)
agentic-spliceai report --targets output/druggable_isoforms.parquet \
  --format pharma \
  --include-structure-prediction
```

**Impact**: Enable isoform-selective drug design that targets disease-specific variants while sparing normal protein function, reducing toxicity and improving therapeutic windows.

### 5. Biomarker Discovery: Liquid Biopsy

**Challenge**: Current liquid biopsy approaches rely on known mutations and methylation patterns. Disease-specific splice isoforms detectable in circulating RNA represent an untapped biomarker source.

**Workflow**:

```bash
# Discover tissue-specific isoform biomarkers (Phase 9)
agentic-spliceai discover --gene-list data/cancer_panel.txt \
  --context cancer:lung \
  --biomarker-mode \
  --min-tissue-specificity 0.8

# Validate biomarker candidates (Phase 7)
agentic-spliceai validate --isoforms output/biomarker_candidates.parquet \
  --agents expression,clinical \
  --cross-tissue-comparison
```

**Impact**: Identify cancer-specific splice isoforms as novel liquid biopsy biomarkers for early detection, treatment monitoring, and companion diagnostics.

---

Last Updated: April 2026
