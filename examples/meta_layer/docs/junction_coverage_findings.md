# Junction Coverage Findings: Strand Asymmetry & Biotype Composition

Detailed findings from the junction coverage audit
(`examples/meta_layer/11_junction_coverage_audit.py`). Focused on two
non-obvious results that surfaced during the investigation:

1. An apparent **+ vs − strand asymmetry** in alt-only coverage — which
   turned out not to be a pipeline bug.
2. A **biotype composition effect** that explains the asymmetry and
   shapes how the alt-only headline number should be interpreted.

For the research framing, the "is M3 worth developing?" decision, and
the connection to alternative data sources, see
[m3_prerequisites.md](m3_prerequisites.md).

---

## 1. The Observation

The audit reports overall junction coverage of Ensembl splice sites at
**68.3%** genome-wide. Broken down:

| Split | Unique sites | Coverage |
|---|---:|---:|
| MANE canonical | 379,649 | **94.6%** |
| Ensembl alt-only (not in MANE) | 382,292 | **42.3%** |

A 52-point gap between canonical and alt-only is striking on its own,
but the *strand* breakdown is more surprising:

| Strand | MANE coverage | Alt-only coverage |
|---|---:|---:|
| + | 99.2% | **73.2%** |
| − | 89.9% | **25.2%** |

MANE is roughly balanced (the 9pt gap is minor). But alt-only coverage
is **three times higher on + strand than on −**. That's a 48pt gap in
a quantity that should, biologically, be strand-symmetric — gene
content on + and − strand is close to 50/50 in mammalian genomes.

---

## 2. What It Wasn't — Ruling Out Pipeline Artifacts

Three hypotheses to rule out before concluding the asymmetry is real:

### 2.1 Gene-strand assignment artifact

The GTEx junction parquet has no strand column; strand is inferred by
joining each junction's `gene_id` to Ensembl's gene→strand table. If a
junction lies in a region where + and − strand genes overlap and its
`gene_id` was assigned to the "wrong" one, its splice sites would be
labeled on the wrong strand and miss the Ensembl match.

**Test:** For each uncovered − strand alt-only site, check whether
there's *any* junction at the same (chrom, position) on either strand.

**Result:** Of 51,257 uncovered − strand alt-only sites on test
chromosomes, only **48 (0.1%)** have any junction at the same
coordinate, and among those 48, both strands are represented.

**Conclusion:** Gene-strand assignment is not causing the asymmetry —
these positions genuinely have no GTEx junction activity nearby.

### 2.2 Overlapping gene double-count

Could the same coordinate be annotated as a − strand splice site in
gene A and also a + strand splice site in gene B? If so, the − strand
version might look "uncovered" while the + strand version is covered.

**Test:** Of 51,257 uncovered − strand alt-only sites, how many *also*
have an Ensembl splice site at the same coordinate on + strand?

**Result:** 13 sites (0.0%).

**Conclusion:** Overlapping gene double-counting is not a material
contributor.

### 2.3 Off-by-one in coordinate derivation

Maybe the − strand splice-site positions are systematically one base
off from where GTEx junctions would land (e.g., the TSV generation
mixed up `start` vs `start+1` on − strand only).

**Test:** Shift uncovered − strand alt-only positions by ±1, ±2, ±3
and check how many hit a junction's intron boundary after the shift.

**Result:**
| Offset | Hit rate |
|---:|---:|
| −3 | 0.3% |
| −2 | 0.4% |
| −1 | 0.6% |
| +1 | 0.5% |
| +2 | 0.2% |
| +3 | 0.2% |

**Conclusion:** No systematic offset — shifts don't recover coverage.
These positions have no nearby junction activity at any small offset.

---

## 3. What It Actually Is — Biotype Composition

The uncovered − strand alt-only sites come from Ensembl's GTF
annotation. The question is why Ensembl enumerates so many more such
sites on − strand specifically.

### 3.1 Transcript biotype imbalance

Breaking down alt-only rows by `transcript_biotype` (from
`splice_sites_enhanced.tsv`):

| Biotype | + strand rows | − strand rows | − / + ratio |
|---|---:|---:|---:|
| lncRNA | 47,718 | 47,465 | 1.0x |
| nonsense_mediated_decay | 7,281 | 10,654 | 1.5x |
| protein_coding (alt) | 13,122 | 23,872 | **1.8x** |
| protein_coding_CDS_not_defined | 7,435 | 11,046 | 1.5x |
| **retained_intron** | 2,961 | **10,262** | **3.5x** |
| pseudogenes (all) | ~3,800 | ~3,600 | ~1.0x |

Note: row counts > unique-site counts because each site can be
annotated for multiple transcripts with different biotypes.

**lncRNAs and pseudogenes are balanced.** The asymmetry comes almost
entirely from **protein_coding alternative transcripts, NMD
transcripts, and especially retained_intron transcripts** — all of
which are 1.5x to 3.5x more populous on − strand in these chromosomes.

### 3.2 These biotypes are intrinsically low-coverage

Coverage rates by biotype for − strand alt-only sites:

| Biotype | Coverage rate |
|---|---:|
| lncRNA | 60% |
| nonsense_mediated_decay | 39% |
| protein_coding (alt) | 29% |
| protein_coding_CDS_not_defined | 28% |
| retained_intron | **16%** |
| processed_pseudogene | **5%** |

None of these are surprising when you look at what the biotypes mean:

- **retained_intron**: by definition keeps an intron *unspliced* in this
  isoform. The boundaries of that intron are splice sites in other
  transcripts, but this transcript isn't using them. 16% coverage is
  exactly what you'd expect.
- **pseudogene**: rarely transcribed. 5% coverage is biology.
- **nonsense_mediated_decay**: transcripts that contain premature
  stop codons and get degraded rapidly. Often below bulk RNA-seq
  detection.
- **protein_coding_CDS_not_defined**: incomplete annotations where
  the CDS couldn't be reliably defined — often predicted/inferred
  placeholders rather than confirmed expression.

### 3.3 The cascade

Combining 3.1 and 3.2:

1. Ensembl has ~2x more `protein_coding` alt-transcript entries and
   ~3.5x more `retained_intron` entries on − strand in these
   chromosomes.
2. Those biotypes are intrinsically poorly expressed in bulk RNA-seq,
   so junction evidence is thin.
3. Result: − strand alt-only coverage rate is dragged down far more
   than + strand.

**This is not a pipeline bug.** It's the interaction of an
annotation-density imbalance (why Ensembl has more of these on −
strand in certain chromosomes is a separate historical annotation
question) with the real biology of which transcript biotypes get
captured by bulk GTEx RNA-seq.

### 3.4 Why more speculative alts on − strand in these chromosomes?

Not investigated in detail, but plausible sources:

- **Annotation history:** HAVANA curators and automated gene-builder
  pipelines have produced denser alt-transcript catalogs for certain
  gene families, some of which cluster on − strand (e.g., olfactory
  receptor clusters on chr1, KIR/LILR on chr19). These families
  contribute many speculative alt isoforms.
- **GTF conventions:** a single gene with many predicted alternative
  transcripts contributes many splice-site rows regardless of strand,
  but if predictions are more aggressive for certain − strand gene
  families, it inflates the − strand row count.

The important thing for M2-S/M3 interpretation is that **the alt-only
coverage rate is a composite** of (a) real biology and (b) annotation
density variation across strands.

---

## 4. Biotype-Filtered Coverage (Q4)

The audit's Q4 analysis filters out sites whose transcript rows are
*all* in the excluded biotype set (retained_intron, pseudogenes, NMD,
protein_coding_CDS_not_defined). Sites that appear in at least one
"real" biotype (lncRNA, protein_coding) are kept.

| Split | Unfiltered | Biotype-filtered | Δ |
|---|---:|---:|---:|
| MANE canonical | 94.6% | 94.6% | 0.0 |
| Alt-only | 42.3% | **45.0%** | **+2.7 pts** |

The filter removes ~130,000 "speculative-only" alt sites and lifts the
alt-only coverage rate by 2.7 points. **Modest.** That tells us
biotype noise is not the main driver of the canonical/alt gap — even
legitimate protein_coding alternatives (not MANE) sit at roughly 29%
coverage on their own.

**The bulk of the 50-point canonical-vs-alt gap is real thin junction
support for annotated alt isoforms, not annotation artifacts.**

---

## 5. Implications

### For interpreting M2-S v2 results

M2-S v2's alt-site recall gains (+6.7 points over v1) are driven by
junction features that actually fire. The ~45% of alt-only sites with
legitimate junction support is where those gains come from. The other
~55% must be recovered (if at all) via sequence + conservation, which
is harder.

### For M2-S v3 design

- Filtering the Ensembl splice-site TSV to exclude retained_intron,
  NMD, and pseudogene-only sites before training would reduce label
  noise — the model was being asked to predict a lot of positions
  that are biological dead-ends.
- The strand asymmetry in alt-only coverage suggests any per-strand
  evaluation should be reported separately; aggregated numbers hide
  the imbalance.

### For M3 (novel-site discovery)

- GTEx alone is insufficient as the sole "novel site" signal source:
  it can't even confirm over half of *annotated* alt isoforms.
  Expecting it to surface *unannotated* splice sites reliably is a
  stretch.
- The 67,490 novel junction sides (coords not in any Ensembl site)
  are real raw material with median depth 4–7K reads and 51/54
  tissues, but their biological validity needs independent
  confirmation.
- See [m3_prerequisites.md](m3_prerequisites.md) for the M3 go/no-go
  analysis.

### For the splice-sites TSV generation pipeline

Worth a separate audit of how `splice_sites_enhanced.tsv` is generated
from Ensembl GTF: why does − strand produce so many more
retained_intron / protein_coding_CDS_not_defined rows? This could be:

- A real reflection of Ensembl's GTF.
- A filter applied during generation that behaves differently by
  strand.
- An artifact of how transcript biotypes propagate to splice-site
  rows.

Run `11_junction_coverage_audit.py` after any change and watch
`q4_coverage_by_biotype.csv` to detect regressions.

---

## Reproduction

```bash
# Test chromosomes (default)
python examples/meta_layer/11_junction_coverage_audit.py

# Full genome
python examples/meta_layer/11_junction_coverage_audit.py --chroms all \
    --output-dir output/meta_layer/junction_coverage_audit_fullgenome

# Disable biotype filtering
python examples/meta_layer/11_junction_coverage_audit.py \
    --exclude-biotypes none
```

Outputs of interest for this document:
- `q1_coverage_by_type.csv` — strand × splice_type breakdown
- `q2_canonical_vs_alt_by_strand.csv` — strand × MANE breakdown
- `q4_coverage_by_biotype.csv` — biotype × in_mane coverage rates
- `q4_biotype_filtered_summary.csv` — MANE vs alt after filter
