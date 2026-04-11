# Variant Analysis Development Roadmap

Active R&D plan for M4 variant effect prediction, from single-variant
delta scoring through systematic gene-wide vulnerability mapping.

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

## In Progress

### Eval-Ensembl-Alt / Eval-GENCODE-Alt

- Eval-Ensembl-Alt: M2-S PR-AUC 0.965 (base 0.749)
- Eval-GENCODE-Alt: M2-S PR-AUC 0.907 (base 0.631)

### M2-S Preparation

- Ops script ready: `examples/meta_layer/ops_train_m2c_pod.sh`
- Ensembl train/val gene cache needs building (~28K genes)
- Retrains meta-layer on Ensembl labels (base model unchanged)

---

## Planned

### Phase 2: ClinVar Integration & Benchmarking

Batch-score ClinVar pathogenic splice variants. Compare meta-layer
delta vs base model delta for pathogenic/benign classification.

- ClinVar VCF loader (pattern from `splicevardb_loader.py`)
- Batch delta scoring script
- ROC/PR analysis: can delta scores distinguish pathogenic from benign?

### Phase 3: Saturation Mutagenesis & Systematic Validation

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

- [Variant effect validation results](../results/variant_effect_validation.md)
- [OOD generalization](../../meta_layer/docs/ood_generalization.md)
- [Negative strand tutorial](../../../docs/variant_analysis/negative_strand_and_variant_effects.md)
- [M1-M4 model variants](../../../docs/meta_layer/methods/00_model_variants_m1_m4.md)
- [M2 variant formulations](../../../docs/meta_layer/methods/05_m2_variant_formulations.md)
- [Saturation mutagenesis application plan](../../../docs/applications/variant_analysis/saturation_mutagenesis_and_validation.md)
