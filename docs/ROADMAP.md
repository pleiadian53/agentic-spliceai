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
| 8 | Variant Analysis | **Phase 2 Done, Phase 3 Next** |
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

### Phase 8: Variant Analysis — PHASE 2 DONE

Use-case-driven R&D for M4 variant effect prediction.

| Sub-phase | Description | Status |
|-----------|-------------|--------|
| **1A** | VariantRunner — ref/alt delta computation | **Done** |
| **1B** | SpliceEventDetector — consequence classification | **Done** |
| **2** | ClinVar + MutSpliceDB benchmarking, radius sweep | **Done** |
| 3 | Clinical pathogenicity head (stack variant-level features) | **Next** |
| 4 | Saturation mutagenesis & SpliceVarDB validation | Planned |
| 5 | Agentic variant interpretation | Planned |

**Validated**: 13 disease-gene variants (10 genes, both strands), 4 SpliceAI
paper cases with RNA-seq confirmed cryptic site positions (MYBPC3 and FAM229B
match within 2bp of RNA-seq ground truth).

**Variant delta recovery**: v2 logit-space blend preserves 45-95% of base model
signal (v1: 20-71%). Cryptic donor gains amplified beyond base model.

**Phase 2 benchmark findings** (2026-04-15):

- On splice-filtered ClinVar (N=2,059; 77% pathogenic prevalence), base model,
  M1-S v2, and M2-S v2 all reach PR-AUC ≈ 0.92 / ROC-AUC ≈ 0.75 — a
  statistical tie. Unfiltered ClinVar (N=11,310) floors at PR-AUC ≈ 0.72
  because ~89% of pathogenic variants there are non-splicing mechanisms.
- On MutSpliceDB (N=434), M2-S v2 wins consequence concordance by **+23 pts**
  (68% vs 45%) — the clinically meaningful metric for "what type of splice
  defect".
- **Key architectural insight**: M2-S v2 is a strict feature-set superset of
  base but doesn't outperform base on ClinVar Δ-based ranking. Reason:
  multimodal features (conservation, junction, RBP, chromatin, epigenetic)
  are *locus-level* and identical between ref and alt at SNV positions — so
  they cancel in the Δ computation and carry no variant-specific signal.
  Meta-layer value is in **locus classification** (alt-site recall,
  consequence type), not pathogenicity ranking.

### Phase 8 — Sub-phase 3: Clinical Pathogenicity Head

To move PR-AUC meaningfully above the base-model ceiling on ClinVar, the
next phase stacks **variant-level features** with the splice-Δ score in a
downstream classifier — mirroring CADD, REVEL, ClinPred, and SpliceAI's
own pipeline convention.

Architecture: small lightweight classifier (logistic regression or
gradient-boosted trees) on top of features that genuinely differentiate
pathogenic from benign *at the variant level* (not just locus level):

| Feature | Source | Expected contribution |
|---|---|---:|
| `log(gnomAD_AF + ε)` | gnomAD v4 | Single strongest non-splice feature; typically +0.05 to +0.15 PR-AUC on ClinVar |
| Gene constraint (LOEUF, pLI) | gnomAD v4 constraint | Refines prior by gene essentiality |
| Variant-differential motif disruption | ESEfinder / RESCUE-ESE + variant-aware scoring | Captures ESE/ISE/branchpoint disruption that splice-Δ alone misses |
| Protein-level deleteriousness | AlphaMissense / ESM-variant | For the ~40% of splice-proximal variants that also affect coding sequence |
| Splice-Δ score (from M1-S v2 or M2-S v2) | This pipeline | The current splice-specific signal |
| Splice consequence type (from M2-S v2) | This pipeline | Classification-level evidence |

This clinical head is **not a replacement** for the meta-layer — it sits
downstream and composes the splice-delta signal with orthogonal variant-
level information. M1-S v2 (or base model directly) can feed the Δ input;
M2-S v2 feeds the consequence-type feature. The head is trained on
ClinVar Pathogenic/Benign with a held-out test split, and evaluated on
SpliceVarDB and HGMD (if licensed) for out-of-distribution validation.

**See**:
- `examples/variant_analysis/` for scripts and results
- `examples/variant_analysis/results/m4_benchmark_sweep.md` for the Phase 2 benchmark report
- `docs/applications/variant_analysis/` for Phase 3+ application plan

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
