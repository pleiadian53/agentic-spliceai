# In Silico Saturation Mutagenesis and Splice Variant Validation

A planned application for systematically predicting splice-altering
positions across entire genes and validating predictions against
experimental evidence from SpliceVarDB and GTEx RNA-seq junction data.

**Status**: Planned (Phase 3 deliverable)
**Prerequisites**: M4 Phase 1A+1B (complete), Phase 2 ClinVar integration

---

## 1. Motivation

Clinical sequencing frequently identifies variants of uncertain
significance (VUS) in disease genes.  A key question for each VUS:
*does this variant disrupt splicing?*

Currently, this is answered case-by-case: run the variant through
SpliceAI or our meta-layer, check the delta score, consult ClinVar.
But for a gene like MYBPC3 (35 exons, ~21K coding positions), there
are ~63K possible SNVs — each potentially splice-altering.

**In silico saturation mutagenesis** systematically scans every position,
predicting which mutations would disrupt splicing, where the cryptic
splice sites would appear, and what the consequence would be.  The
result is a **splice vulnerability map** for the entire gene.

---

## 2. Workflow

### Step 1: Saturation Scan

For every position in the gene (or a focused region), introduce all
possible single-nucleotide substitutions and run the variant effect
pipeline:

```
For each position p in [gene_start, gene_end]:
    ref = FASTA[chrom][p]
    For each alt in {A, C, G, T} - {ref}:
        delta = VariantRunner.run(chrom, p, ref, alt, gene, strand)
        If max(|delta|) > threshold:
            consequence = SpliceEventDetector.analyze(delta)
            Store: (p, ref, alt, delta_scores, consequence, cryptic_positions)
```

### Step 2: Build Splice Vulnerability Map

Aggregate results into a per-position vulnerability score:

```
vulnerability[p] = max over all alt alleles of max(|delta|)
```

Positions with high vulnerability are "splice-sensitive" — any mutation
there is predicted to disrupt splicing.  The map reveals:

- **Invariant splice sites** (GT/AG): vulnerability ~1.0 (expected)
- **Extended splice motifs** (-3 to +6 for donors, -20 to +3 for
  acceptors): vulnerability 0.3-0.8
- **Exonic splice enhancers/silencers (ESE/ESS)**: vulnerability
  0.1-0.5 (sequence motifs that regulate splice site selection)
- **Deep intronic positions**: rare but occasionally high vulnerability
  (positions that create cryptic exons when mutated)

### Step 3: Cross-Validate with SpliceVarDB

SpliceVarDB is a curated database of experimentally validated splice
variants classified as "splice-altering" or "non-splice-altering" based
on RNA-seq, minigene assays, or clinical evidence.

```
For each variant in SpliceVarDB for this gene:
    Look up our saturation scan prediction at (chrom, pos, ref, alt)
    Compare:
        Our prediction: splice-altering (delta > 0.2)?
        SpliceVarDB:    splice-altering?
    
    Agreement types:
        True positive:  both say splice-altering
        True negative:  both say non-splice-altering
        False positive: we predict splice-altering, SpliceVarDB says no
        False negative: we miss it, SpliceVarDB says splice-altering
```

This gives us precision, recall, and AUROC against experimental ground
truth — not just at the gene level, but at the per-position level.

### Step 4: Validate Cryptic Site Positions with GTEx Junctions

For each position where we predict a cryptic splice site gain:

```
For each predicted cryptic site (gain event at position X):
    Query GTEx junction parquet for junctions involving position X
    If junction reads found:
        → Biological validation: the cryptic site is used in vivo
        Record: tissue(s), read count, PSI
    If no junction reads:
        → Either our prediction is wrong, or the variant doesn't
          exist in the GTEx cohort (most likely for rare variants)
```

This leverages the 353K GTEx junctions across 54 tissues already wired
into our pipeline via the junction modality.

---

## 3. Expected Outputs

### Per-Gene Splice Vulnerability Map

```
Gene: MYBPC3 (chr11:47,331,406-47,352,702, minus strand)
Total positions scanned: 21,296
Splice-sensitive positions (Δ > 0.2): 847 (4.0%)
Splice-destroying positions (Δ > 0.5): 312 (1.5%)

Hotspots:
  Exon 25 donor boundary:  15 sensitive positions in 20bp window
  Exon 13 acceptor region: 22 sensitive positions in 30bp window
  Intron 11 (c.1227-13):   Deep intronic cryptic acceptor
```

### SpliceVarDB Concordance

```
Gene: MYBPC3
SpliceVarDB variants evaluated: 47
  True positives:  31 (splice-altering, correctly predicted)
  True negatives:  12 (non-splice-altering, correctly classified)
  False positives:  2 (we over-predict)
  False negatives:  2 (we miss)
  
Precision: 0.94
Recall:    0.94
AUROC:     0.97
```

### GTEx Junction Validation

```
Predicted cryptic sites with GTEx junction support:
  Position 47343155 (cryptic acceptor): 27 junction reads in heart tissue
  Position 47333227 (cryptic donor):    4 junction reads in muscle
  Position 47332576 (shifted donor):    12 junction reads in 3 tissues
  
Validation rate: 3/5 predicted cryptic sites have GTEx support (60%)
```

---

## 4. Compute Requirements

| Gene size | Positions | SNVs (×3) | Time (CPU) | Time (GPU) |
|-----------|-----------|-----------|------------|------------|
| Small (5K bp) | 5,000 | 15,000 | ~4 hours | ~12 min |
| Medium (20K bp) | 20,000 | 60,000 | ~17 hours | ~50 min |
| Large (100K bp) | 100,000 | 300,000 | ~3.5 days | ~4 hours |

**Optimization opportunities**:
- **Batched base model inference**: Run multiple variants through the
  base model simultaneously (current pipeline processes one at a time)
- **Sparse storage**: Only store positions with Δ > threshold (typically
  <5% of all positions), reducing storage 20x
- **Focused scanning**: Scan only exon-proximal regions (±500bp from
  exon boundaries) for the first pass, then deep intronic for high-value
  genes

---

## 5. Relationship to Other Phases

```
Phase 1A (VariantRunner)          ← foundation, complete
Phase 1B (SpliceEventDetector)    ← consequence classification, complete
Phase 2  (ClinVar integration)    ← pathogenicity labels for benchmarking
Phase 3  (Saturation mutagenesis) ← THIS APPLICATION
Phase 5  (Agentic interpretation) ← automated PubMed validation
```

Phase 3 builds directly on Phase 1A+1B (which provide the per-variant
delta and consequence prediction) and Phase 2 (which provides the
ClinVar pathogenicity labels for benchmarking alongside SpliceVarDB).

---

## 6. Existing Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| VariantRunner (single variant) | Complete | `meta_layer/inference/variant_runner.py` |
| SpliceEventDetector | Complete | `meta_layer/inference/splice_event_detector.py` |
| SpliceVarDB loader | Complete | `meta_layer/data/splicevardb_loader.py` |
| GTEx junction data | Complete | `features/modalities/junction.py` |
| ClinVar loader | Planned (Phase 2) | `meta_layer/data/clinvar_loader.py` |
| Batched variant inference | Not started | Needed for compute efficiency |

---

## 7. Clinical Impact

### VUS Reclassification

For a patient with a MYBPC3 VUS at position X:
1. Look up position X in the pre-computed vulnerability map
2. If vulnerability > 0.5: strong evidence for splice disruption
3. Cross-reference with SpliceVarDB: has this position been
   experimentally validated?
4. Check GTEx: does RNA-seq show junction evidence at the predicted
   cryptic site?
5. Combine evidence for ACMG classification (PS3/BS3 criteria for
   functional studies)

### Drug Target Discovery (ASO Design)

Antisense oligonucleotides (ASOs) can modulate splicing by blocking
splice sites or regulatory elements.  The vulnerability map identifies:
- Which splice sites to target (high-vulnerability donors/acceptors)
- Which positions to avoid (mutations at these positions would disrupt
  the ASO's intended effect)
- Cryptic sites that could be blocked to restore normal splicing

---

## References

- Jaganathan et al. (2019). "Predicting Splicing from Primary Sequence
  with Deep Learning." *Cell*, 176(3), 535-548. (Saturation mutagenesis
  in Figure 1D)
- Anna & Bhatt (2018). "Splicing mutations in human genetic disorders:
  examples, detection, and confirmation." *J Appl Genetics*, 59, 253-268.
- SpliceVarDB — curated database of experimentally validated splice
  variants with pathogenicity classification
- Findlay et al. (2018). "Accurate classification of BRCA1 variants
  with saturation genome editing." *Nature*, 562, 217-222. (Wet-lab
  saturation mutagenesis of BRCA1)

## Related

- [Variant effect validation](../../examples/variant_analysis/results/variant_effect_validation.md) — current validation results
- [OOD generalization](../../examples/meta_layer/docs/ood_generalization.md) — model limitations on unseen genes
- [M2 evaluation](../../examples/meta_layer/results/m2_evaluation_results.md) — alternative splice site generalization
- [Isoform discovery](../isoform_discovery/README.md) — downstream transcript assembly
