# Agentic-SpliceAI — Development Roadmap

**North Star**: Enable novel isoform discovery for drug target identification by building a multi-layer pipeline that goes beyond canonical splice annotations to uncover disease-specific, tissue-specific, and variant-induced RNA isoforms with therapeutic potential.

---

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Base Layer | Done |
| 2 | Data Preparation | Done |
| 2.5 | Bioinformatics Lab UI | Done |
| 3 | Workflow Orchestration | Done |
| 4 | Feature Engineering & Multimodal Evidence | Done |
| 5 | Foundation Models | Experimental |
| 6 | Meta Layer Training | Active Research |
| 7 | Agentic Validation Layer | Planned |
| 8 | Variant Analysis | Planned |
| 9 | Isoform Discovery | Ultimate Goal |
| 10+ | Drug Target Validation & Deployment | Future |

---

## Phase Details

### Phase 1: Base Layer — COMPLETE

- Port SpliceAI and OpenSpliceAI prediction engines
- Set up genomic resources (GTF, FASTA, annotations)
- Build BaseModelRunner with data preparation
- **Deliverable**: Canonical splice site predictions (MANE baseline)

### Phase 2: Data Preparation — COMPLETE

- Data preparation module with CLI (`agentic-spliceai-prepare`)
- MANE annotation support for OpenSpliceAI consistency
- **Deliverable**: Production-ready data pipeline

### Phase 2.5: Bioinformatics Lab UI — COMPLETE

- Gene Browser (browse, search, filter ~19K genes)
- Metrics Dashboard (evaluation results, model comparison)
- Genome View (on-demand prediction, 3-track Plotly visualization)
- LRU prediction cache + peak-preserving downsampling
- **Deliverable**: FastAPI + Jinja2 + Plotly.js web service at `server/bio/` (port 8005)

### Phase 3: Workflow Orchestration — COMPLETE

- Chunking and checkpointing for genome-scale processing (PredictionWorkflow)
- Artifact management (ArtifactManager with atomic writes)
- Mode-aware output paths, evaluation metrics
- **Deliverable**: Production base layer with full workflows
- **Verified**: chr22 -- 423 genes, 17.6M positions, 5 chunks, 12.4 min

### Phase 4: Feature Engineering & Multimodal Evidence — COMPLETE

- Modality protocol with auto-registration (FeaturePipeline)
- 9 modalities with 100 feature columns:
  - base_scores (43), annotation (3), sequence (3), genomic (4)
  - conservation (9), epigenetic (12), junction (12), rbp_eclip (8), chrom_access (6)
- Genome-scale FeatureWorkflow with `--augment` for incremental modality addition
- YAML-driven config system with 4 profiles
- Position alignment verification (`features/verification.py`)
- **Deliverable**: 9-modality feature pipeline -- 100 feature columns
- **Verified**: Full-genome feature generation across 17 chromosomes
- **See**: `examples/features/docs/` for per-modality tutorials

### Phase 5: Foundation Models — EXPERIMENTAL

- Evo2-based exon classifier, HDF5 embedding cache
- Device-aware quantization routing, 4 classifier architectures
- SkyPilot + RunPod cloud workflows
- **Deliverable**: Independent sub-project at `foundation_models/`

### Phase 6: Meta Layer Training — ACTIVE RESEARCH

Hierarchical multi-task prediction framework — shared 9-modality feature
infrastructure with specialized model heads for progressively harder tasks:

| Variant | Purpose | Status |
|---------|---------|--------|
| M1 | Canonical Classification | XGBoost baseline 99.78% acc, PR-AUC 0.999/0.998 |
| **M2a** | **Ensembl vs MANE evaluation** — predict alternative splice sites in (Ensembl \ MANE) that the base model never saw | **Next priority** |
| M2 | Alternative Splice Sites (tissue-specific, isoform-specific) | Planned |
| M3 | Novel Site Discovery (junction as held-out target) | Planned |
| M4 | Perturbation-Induced (variant/disease/treatment effects) | Planned |

**M2a is the strongest justification for multimodality**: M1 already achieves
99.7% on canonical sites (little room to prove multimodality's value).
Ensembl-only sites are where the base model is weakest — the delta
(meta - base) at these positions directly measures the value of multimodal
evidence fusion. Requires: Ensembl/GRCh38 annotations, set difference
computation, base score evaluation, then full meta-layer rescue + ablation.

**Key Insight**: Junction support is the #2 feature by SHAP (31.3%), reducing false negatives by 60-70%.

**See**: `examples/meta_layer/docs/meta_model_variants_m1_m4.md` for the full M1-M4 framework and M2a evaluation design

### Phase 7: Agentic Validation Layer — PLANNED

- Literature Agent (PubMed, arXiv, splice databases)
- Expression Agent (GTEx, TCGA, ENCODE)
- Clinical Agent (ClinVar, COSMIC, disease associations)
- Conservation Agent (cross-species PhyloP)
- Nexus Research Agent orchestration
- Self-improvement feedback loop (validation results refine meta layer)
- **Deliverable**: AI-validated predictions with biological context

### Phase 8: Variant Analysis — PLANNED

- VCF processing and variant-induced splicing analysis
- Pathogenicity scoring for splice-affecting variants
- Clinical interpretation and reporting
- ClinVar integration and VUS reclassification

### Phase 9: Isoform Discovery — ULTIMATE GOAL

- Novel splice site detector (high-delta-score sites beyond MANE)
- Isoform reconstruction (virtual transcripts from predicted splice sites)
- RNA-seq junction validation across GTEx tissues
- Confidence scoring with multi-source evidence
- Drug target pipeline: isoform -> druggability assessment -> lead candidates

### Phase 10+: Drug Target Validation & Deployment — FUTURE

- Druggability assessment for novel isoform targets
- Biomarker development (liquid biopsy, companion diagnostics)
- Production platform deployment
- Cloud-native scaling and API services

---

## Success Metrics

### Discovery Metrics (Phase 9)

- 100+ novel isoforms discovered with high confidence
- >70% RNA-seq junction validation rate across GTEx tissues
- >50% literature confirmation for top candidates

### Clinical Metrics (Phases 8-9)

- >30% of VUS variants reclassified through splice impact analysis
- >90% diagnostic accuracy for splice-affecting variants

### Foundation Model Metrics (Phase 5)

- >0.9 AUROC for exon boundary classification
- 10K+ base pairs per second inference on A40 GPU

---

Last Updated: March 2026
