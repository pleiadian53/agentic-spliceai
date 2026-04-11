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
| 8 | Variant Analysis | **Phase 1A+1B Done** |
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
| M1-S v2 | Canonical Classification | **Done** — logit-space blend, PR-AUC 0.9954, FPs -15.5% |
| Eval-Ensembl-Alt | Ensembl alternative sites evaluation | **Done** — M2-S PR-AUC 0.965 |
| Eval-GENCODE-Alt | GENCODE alternative sites evaluation | **Done** — M2-S PR-AUC 0.907 |
| M2-S | Ensembl-trained model | **Done** — 59% recall on alternative sites |
| M3 | Novel Site Discovery (junction as held-out target) | Planned |
| M4 | Perturbation-Induced (variant/disease/treatment effects) | **Phase 1A+1B Done** |

**Logit-space blend (v2)**: Replaced the probability-space residual blend with
a product-of-experts formulation: `softmax((alpha * meta_logits + (1-alpha) * log(base_probs)) / T)`.
Per-class learned temperature subsumes post-hoc calibration.  blend_alpha now
receives gradients during training (was stuck at 0.5 in v1).

**OOD generalization fixed**: v1 meta model hurt on alternative sites (PR-AUC
0.704 < base 0.749). v2 logit-space blend enables graceful degradation — when
the meta-CNN is uncertain, the base model signal dominates. v2 now exceeds the
base model on alternative sites (0.775 > 0.749).

**Key Insight**: Junction support is the #2 feature by SHAP (31.3%), reducing false negatives by 60-70%.

**See**:
- `docs/meta_layer/methods/00_model_variants_m1_m4.md` for the full M1-M4 framework
- `examples/meta_layer/results/` for M1-S v2, M2, and ablation results
- `examples/meta_layer/docs/ood_generalization.md` for OOD analysis

### Phase 7: Agentic Validation Layer — PLANNED

- Literature Agent (PubMed, arXiv, splice databases)
- Expression Agent (GTEx, TCGA, ENCODE)
- Clinical Agent (ClinVar, COSMIC, disease associations)
- Conservation Agent (cross-species PhyloP)
- Nexus Research Agent orchestration
- Self-improvement feedback loop (validation results refine meta layer)
- **Deliverable**: AI-validated predictions with biological context

### Phase 8: Variant Analysis — PHASE 1A+1B DONE

Use-case-driven R&D for M4 variant effect prediction.

| Sub-phase | Description | Status |
|-----------|-------------|--------|
| **1A** | VariantRunner — ref/alt delta computation | **Done** |
| **1B** | SpliceEventDetector — consequence classification | **Done** |
| 2 | ClinVar integration & benchmarking | Planned |
| 3 | Saturation mutagenesis & SpliceVarDB validation | Planned |
| 5 | Agentic variant interpretation | Planned |

**Validated**: 13 disease-gene variants (10 genes, both strands), 4 SpliceAI
paper cases with RNA-seq confirmed cryptic site positions (MYBPC3 and FAM229B
match within 2bp of RNA-seq ground truth).

**Variant delta recovery**: v2 logit-space blend preserves 45-95% of base model
signal (v1: 20-71%). Cryptic donor gains amplified beyond base model.

**See**:
- `examples/variant_analysis/` for scripts and results
- `docs/applications/variant_analysis/` for Phase 3 application plan

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

Last Updated: April 2026
