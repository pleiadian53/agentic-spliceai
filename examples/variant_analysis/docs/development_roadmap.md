# Variant Analysis Development Roadmap

Active R&D plan for M4 variant effect prediction, from single-variant
delta scoring through systematic gene-wide vulnerability mapping.

> **Current blocker (2026-07-29): label granularity, not model capacity.**
> A per-nucleotide Δ model needs per-nucleotide supervision, and essentially none exists —
> ClinVar labels are clinical only, SpliceVarDB is variant-level binary, SpliceVault's top-4 events
> describe the *site* rather than the variant, and MutSpliceDB (the only corpus with induced-site
> coordinates) holds **441 variants, 433 of them intron retention**. That is why the archived
> delta-prediction line plateaued at r ≈ 0.38–0.41 / AUC ≈ 0.58–0.61. The unlock is a corpus of
> (variant → per-site quantitative splicing change) — sQTL catalogs or MPRA/saturation assays —
> not a bigger model. Full analysis:
> [`../results/m4_variant_arm_status.md`](../results/m4_variant_arm_status.md).

---

## Completed

### Phase 1A: Single-Variant Delta Pipeline

- `VariantRunner` — end-to-end ref/alt delta computation
- Base model integration via resource manager (pluggable)
- Minus-strand handling (RC + position reversal, no channel swap)
- CLI: `01_single_variant_delta.py` with YAML batch config

### Phase 1B: Splice Consequence Prediction

- `SpliceEventDetector` — maps delta events to exon boundaries
- Consequence classification: exon_skipping, intron_retention,
  donor_shift, donor_destruction, acceptor_shift, cryptic_exon
- CDS extraction for reading frame analysis
- CLI: `01b_splice_consequences.py` with JSON output

### Logit-Space Blend (v2 Model)

- Fixed double-softmax bug and dead blend_alpha
- Per-class learned temperature [T_donor, T_acceptor, T_neither]
- Variant delta recovery: 45-95% of base signal (v1: 20-71%)
- PR-AUC: 0.9954 (v1: 0.9899), FPs reduced 15.5%

### Validation

- 13 disease-gene variants across 10 genes (77% HIGH confidence)
- 4 SpliceAI paper cases: 2/4 match cryptic site positions within
  2bp of RNA-seq validated junctions (MYBPC3, FAM229B)
- Results: `results/variant_effect_validation.md`

---

### Phase 2: ClinVar & MutSpliceDB Benchmarking — DONE (2026-04-15, MutSpliceDB re-run on v4)

- `02_clinvar_download.py` (VCF → filtered parquet), `03_clinvar_benchmark.py` (ROC/PR),
  `04_mutsplicedb_benchmark.py` (detection + consequence concordance)
- **Answer to "can delta scores distinguish pathogenic from benign?" — barely, and no better than
  the base model:** ClinVar splice-filtered ROC-AUC **0.754 (base)** vs **0.753 (M2-S)**.
  Where the meta layer wins is consequence concordance: **M2-S 72.1%** vs **M1-S 50.7%**.
- Full analysis + why: [`../results/m4_variant_arm_status.md`](../results/m4_variant_arm_status.md)
- ⚠️ ClinVar was **not** re-run on the v4 checkpoints — all ClinVar numbers are v2-era.

### M2-S — DONE (promoted)

`m2s_v4_cleanannot` is the promoted alternative-site model. Held-out alt-site PR-AUC
**0.911 → 0.990**, recall ~17% → ~90%. (Superseded the v2 numbers previously listed here as
"in progress": Eval-Ensembl-Alt 0.965 / Eval-GENCODE-Alt 0.907.)

---

## Planned

### Phase 3: Clinical Pathogenicity Head — design-only

Stack variant-level features (`log(gnomAD AF)`, LOEUF/pLI, AlphaMissense, splice-Δ, consequence
type) into a small classifier. **Never implemented; no measured numbers.** Note the ceiling: ~89%
of ClinVar pathogenic variants act by non-splicing mechanisms, so this improves the *clinical*
answer, not the splicing one.

### Phase 4: Saturation Mutagenesis & Systematic Validation

Gene-wide splice vulnerability mapping with experimental cross-validation.

- Scan every position × 3 SNVs → vulnerability map
- Cross-validate against SpliceVarDB (precision/recall/AUROC)
- Validate cryptic site positions against GTEx junction reads
- Compute optimization: batched inference, sparse storage

**Full specification**: [docs/applications/variant_analysis/saturation_mutagenesis_and_validation.md](../../../docs/applications/variant_analysis/saturation_mutagenesis_and_validation.md)

### Phase 5: Agentic Variant Interpretation

LLM-powered interpretation combining delta scores with literature
evidence. Uses Nexus research agent for PubMed search, gene-disease
associations, structured clinical reports.

---

## Architecture Decisions

### Base Model Override (Deferred)

The base model (OpenSpliceAI) produces zero signal for some genes
(PYGB, CDC25B in SpliceAI paper validation). The meta-layer currently
cannot override a confident base model "no splice" prediction.

Options considered:
1. **M2-S training** (preferred) — broader training labels teach the
   model to recognize splice sites the base model misses
2. **Confidence-gated alpha** — position-dependent blend weight based
   on base model confidence (~100 extra parameters)
3. **Threshold-based fallback** — if base model is uncertain, increase
   meta-CNN weight (zero extra parameters)

Decision: pursue M2-S first. If OOD failures persist after Ensembl
training, implement confidence-gated alpha.

### Reading Frame Analysis (Partially Implemented)

CDS extraction is complete (`extract_cds_annotations()` in
`genomic_extraction.py`). Full reading frame analysis (frameshift
detection, NMD prediction) is stubbed in `SpliceEventDetector` —
CDS data is loaded but the frame preservation logic needs the
cross-type donor-acceptor pairing to compute junction size changes.

---

## Related Documentation

- [**Mutation-induced arm status**](../results/m4_variant_arm_status.md) — consolidated results + blockers
- [M4 benchmark sweep](../results/m4_benchmark_sweep.md)
- [Variant effect validation results](../results/variant_effect_validation.md)
- [OOD generalization](../../meta_layer/docs/ood_generalization.md)
- [Negative strand tutorial](../../../docs/variant_analysis/negative_strand_and_variant_effects.md)
- [M1-M4 model variants](../../../docs/meta_layer/methods/00_model_variants_m1_m4.md)
- [M2 variant formulations](../../../docs/meta_layer/methods/05_m2_variant_formulations.md)
- [Saturation mutagenesis application plan](../../../docs/applications/variant_analysis/saturation_mutagenesis_and_validation.md)
