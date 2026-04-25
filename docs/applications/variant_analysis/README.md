# Variant Effect Analysis (M4)

**Goals served**: adaptive splice prediction + drug target identification

**Tier**: Active (most mature of the active applications)

**Last updated**: 2026-04

---

## Problem

Genetic variants can disrupt splicing by abolishing canonical sites or
creating cryptic ones. Quantifying variant effect on splicing is essential
for (a) interpreting clinical variants of uncertain significance (VUS),
(b) mapping per-gene splice vulnerability, and (c) identifying
variant-induced isoforms as potential drug targets. A production-grade
variant effect pipeline must handle both strands, multiple consequence
types (donor loss, acceptor gain, etc.), and benchmark honestly against
clinical ground truth.

## User-facing functionality

- Compute per-variant splice delta scores (ref vs alt) using M1-S v2 or M2-S v2
- Classify splice consequence (donor/acceptor gain/loss, cryptic sites)
- Benchmark against ClinVar (splice-filtered and unfiltered) with
  pathogenic/benign discrimination
- Benchmark against MutSpliceDB (RNA-seq-validated variants) for
  consequence-type concordance
- Batch process variants from YAML, VCF, or custom TSV
- Generate delta-score plots for single variants with flanking context

## Driving examples

- [`examples/variant_analysis/01_single_variant_delta.py`](../../../examples/variant_analysis/01_single_variant_delta.py) — single-variant delta scoring
- [`examples/variant_analysis/01b_splice_consequences.py`](../../../examples/variant_analysis/01b_splice_consequences.py) — consequence classification with JSON output
- [`examples/variant_analysis/02_clinvar_download.py`](../../../examples/variant_analysis/02_clinvar_download.py) — ClinVar VCF download + splice-SNV filter
- [`examples/variant_analysis/03_clinvar_benchmark.py`](../../../examples/variant_analysis/03_clinvar_benchmark.py) — ClinVar benchmark (pathogenic vs benign)
- [`examples/variant_analysis/04_mutsplicedb_benchmark.py`](../../../examples/variant_analysis/04_mutsplicedb_benchmark.py) — MutSpliceDB benchmark (consequence concordance)
- [`examples/variant_analysis/test_variants.yaml`](../../../examples/variant_analysis/test_variants.yaml) — 13 validated disease-gene variants across 10 genes
- Pod ops: [`ops_m4_benchmarks_pod.sh`](../../../examples/variant_analysis/ops_m4_benchmarks_pod.sh), [`_post_orchestrator_regen_plots.sh`](../../../examples/variant_analysis/_post_orchestrator_regen_plots.sh)

Additional plan docs:

- [Saturation mutagenesis + SpliceVarDB validation (Phase 3)](saturation_mutagenesis_and_validation.md)

## `src/` surface

- `agentic_spliceai.splice_engine.meta_layer.inference.variant_runner.VariantRunner` — ref/alt delta computation
- `agentic_spliceai.splice_engine.meta_layer.inference.splice_event_detector.SpliceEventDetector` — consequence classification
- `agentic_spliceai.splice_engine.meta_layer.models.sequence_model` — M1-S / M2-S loaded for inference
- `agentic_spliceai.splice_engine.features.dense_feature_extractor` — locus-level features at ref/alt (note: multimodal features cancel for SNVs)

## Evaluation

- **Datasets**:
  - ClinVar splice-filtered (N=2,059; 77% pathogenic prevalence)
  - ClinVar unfiltered (N=11,310; ~89% non-splicing mechanisms)
  - MutSpliceDB (N=434; RNA-seq-validated variants)
  - 13 disease-gene variants (curated, dual-strand)
  - 4 SpliceAI paper cases with RNA-seq-confirmed cryptic positions
- **Baselines**: OpenSpliceAI base model, M1-S v2 logit blend, M2-S v2
- **Key metrics**: PR-AUC, ROC-AUC, consequence concordance (MutSpliceDB)
- **Results**:
  - [Variant effect validation](../../../examples/variant_analysis/results/variant_effect_validation.md) — 13 variants + 4 RNA-seq cases
  - [M4 benchmark sweep](../../../examples/variant_analysis/results/m4_benchmark_sweep.md) — full Phase 2 benchmark report
  - [MutSpliceDB analysis](../../../examples/variant_analysis/results/mutsplicedb_analysis/) — detailed breakdown

**Key findings (Phase 2)**:

- Base, M1-S v2, M2-S v2 all reach PR-AUC ≈ 0.92 on splice-filtered ClinVar (tie)
- M2-S v2 wins consequence concordance by +23 pts on MutSpliceDB (68% vs 45%)
- Locus-level multimodal features cancel for SNVs in Δ computation —
  meta-layer value is in locus classification, not pathogenicity ranking
- Cryptic site positions match RNA-seq within 2bp (MYBPC3, FAM229B)

## Maturity tier and signals

**Current tier**: Active

**Signals supporting the tier**:

- Most mature active application by benchmark rigor
- Phase 1A (delta), 1B (consequences), and 2 (benchmarks) all complete
- Both strands validated; dual-strand fix applied
- Reproducible benchmark sweep via pod orchestration
- Stable args across the 4 driver scripts

## Graduation signals

**To advance to Mature, the application needs**:

- Phase 8.3 clinical pathogenicity head (stacks splice Δ with
  gnomAD AF, LOEUF, AlphaMissense — see [ROADMAP.md](../../ROADMAP.md))
- CLI entry point (`agentic-spliceai-variant` — proposed, not implemented)
- Inference-path tests
- SpliceVarDB cross-validation for OOD generalization

## Known limitations

- Locus-level multimodal features do not differentiate SNVs at the same
  position; meta layer gains are in classification, not Δ ranking
- Base-model PR-AUC ceiling on ClinVar Δ-ranking (~0.92) — further
  gains require variant-level features (Phase 8.3)
- Unfiltered ClinVar floors at PR-AUC ~0.72 because ~89% of pathogenic
  variants are non-splicing
- Cross-strand coordinate handling requires careful attention (see
  [negative strand tutorial](../../variant_analysis/negative_strand_and_variant_effects.md))
- Bootstrap confidence intervals needed for smaller benchmarks (MutSpliceDB)

## Related

- [Saturation mutagenesis + SpliceVarDB validation](saturation_mutagenesis_and_validation.md) — Phase 3 plan
- [Adaptive Splice Prediction](../adaptive_splice_prediction/README.md) — source of M1-S / M2-S models
- [Canonical Splice Prediction](../canonical_splice_prediction/README.md) — base layer fallback
- [Use cases: Clinical Genetics VUS interpretation](../use_cases.md#2-clinical-genetics-vus-interpretation)
- [Roadmap: Phase 8](../../ROADMAP.md)
