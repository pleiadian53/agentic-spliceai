# Evaluation Hierarchy for Meta-Layer Predictions

## Context

The meta-layer evaluation script (`08_evaluate_sequence_model.py`) supports four
operating modes with increasing generality:

| Case | Mode | Ground truth source | Evaluation? |
|------|------|---------------------|-------------|
| 1 | SpliceAI test split (chr1,3,5,7,9) | MANE splice annotations | Full metrics |
| 2 | Custom chromosomes (`--test-chroms`) | MANE/Ensembl annotations | Full metrics |
| 3 | Specific genes (`--genes`) | MANE/Ensembl annotations | Full metrics |
| 4 | Arbitrary FASTA (`--fasta`) | None available | Inference only |

Cases 1-3 have known splice site annotations, so we can compute PR-AUC, FN
reduction, top-k accuracy, and all other metrics.  Case 4 produces `[L, 3]`
probability scores but has no ground truth to evaluate against.

This document describes how to evaluate Case 4 predictions across different
scenarios.

---

## The Problem: Evaluating Predictions on Arbitrary Sequences

When the input is an arbitrary FASTA (e.g., from T2T-CHM13, a non-model
organism, or a synthetic construct), we face a fundamental challenge:

- No splice site annotations exist for the sequence
- No precomputed base model scores are available
- No multimodal features (conservation, epigenetic, junction) are available

The model runs with uniform 1/3 base-score prior and zero multimodal features.
Predictions are valid hypotheses, but validating them requires external evidence.

---

## Evidence Tiers

### Tier 1: Known Annotations (Cases 1-3)

**When available**: The sequence comes from a reference genome (GRCh38, GRCh37)
with curated splice site annotations (MANE, Ensembl, RefSeq).

**Evaluation**: Standard classification metrics (PR-AUC, precision, recall, FN/FP
counts, top-k accuracy).

**Scripts**: `08_evaluate_sequence_model.py` with `--build-cache` or `--cache-dir`.

### Tier 2: Alignment to Annotated Reference

**When available**: The sequence is from a different genome build (T2T-CHM13,
pangenome) or a closely related species, and can be aligned to a reference with
known splice annotations.

**Approach**:
1. Align FASTA sequences to GRCh38 using minimap2 or BLAST
2. Map predicted splice positions to reference coordinates via alignment
3. Look up known splice sites at mapped coordinates
4. Compute concordance metrics

**Status**: Not yet implemented.  The alignment step is a natural agentic-layer
tool (see `examples/agentic_layer/docs/alignment_validation_tool.md`).

**Use cases**:
- Cross-build validation (T2T-CHM13 genes vs GRCh38 annotations)
- Ortholog analysis (comparing splice predictions across species)
- Validating predictions on genome patches or alternate haplotypes

### Tier 3: Junction Read Support (M3 Territory)

**When available**: RNA-seq data exists for the tissue/condition where the
sequence is expressed.  Junction reads provide independent evidence of splice
site usage.

**Approach**:
1. Obtain junction counts from RNA-seq (e.g., GTEx, ENCODE, tissue-specific)
2. For each predicted splice site, check if supporting junction reads exist
3. Junction support serves as the validation target (not input feature)

**Key design**: This is the M3 model variant's core principle.  M3 *excludes*
junction features from input and uses junction presence as the evaluation
signal.  This prevents information leakage while providing an independent
validation metric.

**Connection to M3 training**:
- M3 feature config: `meta_m3_novel` (8 modalities, junction excluded)
- JunctionModality is label-agnostic: same columns whether used as feature (M2)
  or evaluation target (M3)
- GTEx v8 junction data: 353K junctions, 54 tissues
- See `docs/meta_layer/methods/00_model_variants_m1_m4.md` for the M1-M4 framework

**Status**: Planned for M3 development phase.

### Tier 4: Multi-Source Evidence Fusion (Agentic Layer)

**When available**: Multiple independent signals can be assembled:
- Cross-species conservation (PhyloP, PhastCons)
- RNA secondary structure (RNAfold, predicted splice site context)
- Protein domain boundaries (if coding region)
- Literature evidence (known functional splice sites in related genes)

**Approach**: An agentic workflow orchestrates multiple evidence sources,
weights them, and produces a confidence-weighted validation report.

**Status**: Planned for Phase 7 (Agentic Workflows).

---

## Practical Guidance

### "I have a FASTA and want to know if the predictions are correct"

1. **If the sequence is from a known genome build**: Don't use `--fasta`.
   Use `--genes` or `--test-chroms` with `--build-cache` instead.  This gives
   you ground truth evaluation.

2. **If the sequence is from a different build** (e.g., T2T-CHM13):
   - Run `--fasta` for initial predictions
   - Align to GRCh38 to map splice positions to annotated coordinates
   - Compare predicted sites against known annotations at mapped positions
   - This is Tier 2 evaluation (alignment tool not yet available)

3. **If the sequence is truly novel** (no reference match):
   - Run `--fasta` for predictions
   - Look for supporting RNA-seq junction evidence (Tier 3)
   - If no RNA-seq data: predictions remain hypotheses until validated

### "How much should I trust FASTA-mode predictions?"

The model's accuracy degrades gracefully without multimodal features:
- **Sequence-only signal** (DNA motifs, dinucleotide patterns) is the primary
  contributor in M1-S, accounting for the majority of the model's performance
- **Base model scores** (uniform prior in FASTA mode) normally provide a strong
  signal but are absent here
- **Multimodal features** (conservation, junction, epigenetic) add context-specific
  refinement

Expect FASTA-mode predictions to identify **strong canonical splice sites**
(GT-AG dinucleotide motifs) reliably, but to miss **weak/alternative sites**
that require multimodal evidence.

---

## Related Documents

- [Model variants M1-M4](../../../docs/meta_layer/methods/00_model_variants_m1_m4.md) — M3 excludes junction from features
- [M2 variant formulations](../../../docs/meta_layer/methods/05_m2_variant_formulations.md) — alternative splice site evaluation
- [Alignment validation tool](../../agentic_layer/docs/alignment_validation_tool.md) — Tier 2 implementation (agentic layer)
- [Cross-build FASTA validation](cross_build_fasta_validation.md) — end-to-end pipeline for custom genome builds
- [Ground truth for custom genomes](../../data_preparation/docs/ground_truth_custom_genomes.md) — `04_generate_ground_truth.py` with `--gtf`
- [OOM gene caching](oom_gene_caching.md) — disk-backed cache for large-scale evaluation
