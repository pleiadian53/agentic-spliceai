# A1 — Cross-Annotation Label Audit (Experimental Results)

The first experimental step of M3 v1: how many of the GTEx-novel junction
sides are *genuinely* absent from every major annotation, not just from
Ensembl? This determines the size and quality of the M3 positive pool.

Full machine-readable outputs and the detailed methods writeup live under
`output/meta_layer/m3_label_audit/`. This document is the public summary.

> **Correction (2026-05-24).** The original A1 figure of "65,163 novel GTEx
> sites" was a bug artifact: the annotation `splice_sites_enhanced.tsv` files
> placed minus-strand splice sites at wrong positions, so the novel-filter
> anti-joins failed to remove minus-strand *annotated* sites. After fixing the
> extractor (`genomic_extraction.extract_splice_sites_from_exons`) and
> regenerating the annotation, **only 748 GTEx junction sides are genuinely
> novel** (52,710 of the original survivors were minus-strand annotated sites
> that leaked). The real M3 novel pool is carried by **SpliceVault** (~154K
> cryptic sites, which are non-annotated by construction and unaffected by the
> bug). **Corrected pooled positives: 154,113.** The §3 tables below reflect
> the pre-fix (buggy) audit and are retained for the record; trust the
> corrected numbers here and in `m3_design.md` §3.

---

## 1. Question

The junction-coverage audit (`examples/meta_layer/11_junction_coverage_audit.py`,
Q3) surfaced **67,490** junction "sides" (donor/acceptor positions) that are
supported by GTEx RNA-seq but **absent from Ensembl**. "Absent from Ensembl"
is a weak definition of novel — the position could still be annotated in
GENCODE, RefSeq, or UCSC. A1 re-checks the 67,490 against those.

## 2. Inputs

| Annotation | Unique splice sites (by `chrom, position, strand, splice_type`) |
|---|---:|
| GTEx-novel candidates (target) | 67,490 |
| Ensembl (defined "novel") | 737,639 |
| GENCODE v47 | 928,092 |
| RefSeq curated (NM_/NR_) | 450,025 |
| RefSeq full (incl. predicted XM_/XR_) | 562,676 |
| MANE v1.3 (sanity) | 366,757 |
| UCSC Known Genes | **skipped** — in modern hg38, UCSC's "Known Genes" track *is* GENCODE; it would duplicate the GENCODE result |

RefSeq was pulled fresh from the UCSC `hg38.ncbiRefSeq.gtf` dump and reduced
to splice-site positions. Chromosome naming was normalised to bare format
(`1`, not `chr1`) across all sources before joining.

## 3. Headline result

Removing every annotated position cumulatively:

| Cumulative filter | Sides surviving | % of 67,490 |
|---|---:|---:|
| Original (not in Ensembl, by construction) | 67,490 | 100.00% |
| also not in **GENCODE v47** | 65,771 | 97.45% |
| also not in **RefSeq curated** (NM_/NR_) | **65,163** | **96.55%** |
| also not in RefSeq full (incl. predicted XM_/XR_) | 63,715 | 94.41% |

**The full cross-annotation check removes only ~3.5% of the pool.** 65,163
junction-supported sites are absent from Ensembl, GENCODE, *and* RefSeq.

### 3.1 Each annotation's marginal contribution

| Annotation | Marginal sites caught | % of upstream pool |
|---|---:|---:|
| GENCODE (over Ensembl) | 1,719 | 2.55% |
| RefSeq curated (over Ensembl ∪ GENCODE) | 608 | 0.92% |
| RefSeq predicted (over the above) | 1,448 | 2.21% |

The annotations are **nearly orthogonal** in their incremental coverage of
GTEx junction evidence. By the time GENCODE is applied, RefSeq adds almost
nothing curated.

### 3.2 Biotype of the rescued sites

- **GENCODE-rescued (1,719):** 1,701 (99%) are in **lncRNA** genes, only 20
  protein-coding. GENCODE's incremental annotation over Ensembl is almost
  entirely lncRNA biology.
- **RefSeq-curated-rescued (608):** 367 in protein-coding contexts, 241
  noncoding-only. RefSeq contributes a small but more protein-coding-relevant
  slice — which is why RefSeq curated is worth including for an M3 focused on
  protein-coding cryptic splicing.

## 4. Survivor quality

The 65,163 survivors are well-supported, not marginal hits:

| Quantile | `sum_reads` | `n_tissues` (of 54) | `tissue_breadth` |
|---|---:|---:|---:|
| Q10 | 57 | 11 | 0.20 |
| Q25 | 529 | 35 | 0.65 |
| Q50 | 5,462 | 52 | 0.96 |
| Q75 | 64,378 | 54 | 1.00 |
| Q90 | 565,817 | 54 | 1.00 |

Split: 37,412 donors, 27,751 acceptors.

Filter pass rates (the Phase B1 candidates):

| Filter | Survivors | % of 65,163 |
|---|---:|---:|
| `sum_reads ≥ 100` AND `n_tissues ≥ 5` (B1 minimum) | 56,214 | 86.3% |
| `sum_reads ≥ 1,000` AND `n_tissues ≥ 25` (stricter) | 43,490 | 66.7% |

Even the strict filter yields ~43.5K candidates — two orders of magnitude
more than the combined Tier-1 disease catalogs.

## 5. What A1 does and does not establish

**Establishes:** the unconditional novel pool is large and robust to
annotation choice. Candidate generation is *not* the M3 bottleneck.

**Does not establish:** whether a given survivor is real novel biology vs
annotation lag in a release we did not check, vs a STAR alignment artifact
shared across samples, vs non-functional noisy splicing. Those are Phase B
(sequence/conservation filters) and Phase D (long-read truth set) questions.

## 6. Implication for M3 v1

The pool size means M3 v1 can afford to **discard aggressively** on quality
rather than fight to keep marginal candidates. Combined with the
architecture finding (the lever is data/labels — see [`m3_design.md`](m3_design.md) §4),
the project's effort should go to the Phase B label and negative-sampling
stack, not to model capacity.

The biotype split (§3.2) also suggests M3 v1 should consider **stratifying
labels by protein-coding vs lncRNA** — the lncRNA fraction may carry
different splicing-grammar statistics.

## 7. Outputs

```
output/meta_layer/m3_label_audit/
├── A1_cross_annotation.md            detailed methods + decision log
├── A1_survivors_final.parquet        65,163 sites (Ensembl ∪ GENCODE ∪ RefSeq-curated removed)
└── A1_survivors_final_strict.parquet 63,715 sites (also excludes RefSeq-predicted) — sensitivity
```

Schema: `chrom, strand, position, splice_type, gene_id, max_reads, sum_reads, n_tissues, tissue_breadth`.

## Related
- [`m3_design.md`](m3_design.md) — what these labels feed into.
- [`m3_prerequisites.md`](m3_prerequisites.md) — the go/no-go that motivated the audit.
- [`../junction_coverage_findings.md`](../junction_coverage_findings.md) — the upstream junction audit.
