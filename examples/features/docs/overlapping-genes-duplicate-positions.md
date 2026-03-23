# Overlapping Genes and Duplicate Positions in Analysis Sequences

**Date**: 2026-03-23
**Observed in**: `analysis_sequences_{chrom}.parquet` artifacts

---

## What Are Duplicate Positions?

The analysis_sequences parquets contain a small fraction (~0.12%) of duplicate
`(chrom, position)` rows. These are **not data corruption** — they arise from
overlapping gene annotations in the MANE/Ensembl GTF. When two or more genes
share a genomic position (same chromosome, same coordinate), the base-layer
prediction workflow produces one row per gene at that position.

```
Gene A: ═══════════════════════════╗
                                    \
Position P:  shared exon boundary ──── appears once per gene
                                    /
Gene B: ═══════════════════════════╝

In the parquet:
  chrom   position    gene_id        splice_type   donor_score
  chr5    141509112   gene-PCDHGA1   donor         0.87
  chr5    141509112   gene-PCDHGA3   donor         0.87
  chr5    141509112   gene-PCDHGA5   donor         0.87
  ...     ...         ...            ...           ...
```

Each row is the same genomic position, but in the **context of a different gene**.
The `gene_id`, `transcript_id`, and potentially `splice_type` columns differ across
duplicates. Base model scores (`donor_score`, etc.) are identical because the model
predicts from DNA sequence regardless of gene context.

---

## Prevalence

Measured across 10 chromosomes (full-stack, 104-column parquets):

| Chromosome | Positions | Unique | Duplicates | Rate |
|-----------|----------:|-------:|-----------:|-----:|
| chr1 | 562,668 | 562,489 | 179 | 0.03% |
| chr2 | 517,337 | 517,044 | 293 | 0.06% |
| chr3 | 454,349 | 454,300 | 49 | 0.01% |
| chr4 | 338,233 | 338,170 | 63 | 0.02% |
| chr5 | 354,573 | 351,923 | **2,650** | **0.75%** |
| chr19 | 167,151 | 167,054 | 97 | 0.06% |
| chr20 | 144,432 | 144,358 | 74 | 0.05% |
| chr21 | 58,192 | 58,166 | 26 | 0.04% |
| chr22 | 96,467 | 96,438 | 29 | 0.03% |
| chrX | 245,763 | 245,736 | 27 | 0.01% |
| **Total** | **2,939,165** | **2,935,678** | **3,487** | **0.12%** |

chr5 is the outlier (0.75%) due to the protocadherin gamma gene cluster.

### Multiplicity Distribution

Most duplicate positions involve 2 overlapping genes. The extreme cases
(up to 22x) are all from the PCDH gamma cluster:

| Multiplicity | Positions | Example |
|:---:|---:|---|
| 2x | 1,146 | Typical overlapping gene pairs |
| 3-5x | 162 | Small gene clusters |
| 6-10x | 66 | Medium clusters |
| 11-15x | 57 | Large clusters |
| 16-22x | 37 | Protocadherin gamma (PCDHGA1-PCDHGC5, chr5) |

---

## Which Gene Families Cause This?

All duplicate positions come from **overlapping gene annotations** (100%
multi-gene, 0% same-gene duplicates). The major contributors:

### Protocadherin Gamma Cluster (chr5q31.3)

The largest contributor. 22 PCDH-gamma genes (PCDHGA1 through PCDHGC5) share
variable first exons but converge on common constant exons. A single position
in the constant region appears in up to 22 gene contexts.

```
PCDHGA1:  [var exon 1]───────╲
PCDHGA2:  [var exon 1]────────╲
PCDHGA3:  [var exon 1]─────────╲
...                             ╲
PCDHGC5:  [var exon 1]──────────── [constant exons] ←── shared positions
                                    (up to 22 rows per position)
```

### Other Known Overlapping Gene Families

- **Protocadherin alpha/beta** (chr5): Similar cluster architecture
- **HLA region** (chr6): Highly polymorphic, many overlapping annotations
- **Olfactory receptors**: Tandem duplications with overlapping coordinates
- **Histone clusters** (chr6): Compact gene arrangement
- **Small antisense gene pairs**: Bidirectional promoters, overlapping 3' ends

---

## Implications for the Meta-Layer

### Training: Keep Duplicates As-Is

The duplicates should **not** be deduplicated before training. Each row represents
a valid training example: the same genomic position in the context of a specific
gene. The meta-layer benefits from seeing the same position with different:

- `gene_id` / `transcript_id` (different gene context)
- `splice_type` annotations (a position may be a donor for gene A but intronic
  for gene B)
- Gene-scoped context features (`.over('gene_id')` features differ per gene)

Deduplication would lose this gene-context information.

### Training: Minor Class Balance Effect

At 0.12% of positions, duplicates have negligible effect on class balance or
gradient statistics. Even the worst case (chr5 at 0.75%) is dominated by a
single gene cluster — it introduces slight overweighting of protocadherin
splice sites, which is unlikely to affect model generalization.

### Evaluation: Deduplicate for Position-Level Metrics

When computing position-level evaluation metrics (precision, recall, PR-AUC),
duplicates should be deduplicated to avoid double-counting:

```python
# For position-level evaluation
eval_df = df.unique(subset=["chrom", "position"], keep="first")
```

Or equivalently, group by position and take the max prediction:

```python
# Conservative: take the strongest splice signal per position
eval_df = df.group_by(["chrom", "position"]).agg(
    pl.col("donor_score").max(),
    pl.col("acceptor_score").max(),
    pl.col("splice_type").first(),  # arbitrary — same position, different context
)
```

### Cross-Validation: No Special Handling Needed

The chromosome-split CV strategy already handles this correctly because:
1. All duplicates of a position are on the same chromosome (same split)
2. No data leakage — duplicates don't appear in both train and test
3. The balanced chromosome split assigns by gene count, not position count

---

## Relationship to Feature Engineering

The RBP eCLIP modality (and all other modalities) joins on `(chrom, position)`.
When a position has N gene-context rows, the join produces N rows with identical
RBP features — the binding evidence is position-level, not gene-level.

This is correct behavior: RBP binding is a property of the genomic position,
not of the gene annotation. All N rows at the same position should have the
same `rbp_n_bound`, `rbp_max_signal`, etc.

---

## References

- MANE Select annotation: ~19K protein-coding genes, most non-overlapping
- GENCODE v44 (GRCh38): documents overlapping gene clusters
- Protocadherin cluster: Wu & Bhatt (2007) "Evolution of protocadherin genes"
  Genome Research 17:1476-1483
