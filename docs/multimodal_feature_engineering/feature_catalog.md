# Multimodal Feature Catalog

Complete reference for all 10 feature modalities in the agentic-spliceai meta-layer
feature engineering pipeline. Each modality is a registered `Modality` subclass in
`src/agentic_spliceai/splice_engine/features/modalities/`.

## Summary

| Modality | Columns | Source Type | Data Source | Description |
|----------|---------|-------------|-------------|-------------|
| `base_scores` | 43 | Model output | Base model predictions (SpliceAI/OpenSpliceAI) | Derived features from raw splice site probabilities |
| `annotation` | 3 | Genomic annotation | GTF/GFF gene annotations | Ground truth splice site labels and transcript info |
| `sequence` | 3 | Genomic reference | Reference FASTA (hg38/hg19) | Contextual DNA sequence windows around each position |
| `genomic` | 4 | Genomic annotation | Gene boundary coordinates + sequence | Positional and compositional features within genes |
| `conservation` | 9 | External data | UCSC PhyloP/PhastCons bigWig tracks | Evolutionary constraint from multi-species alignment |
| `epigenetic` | 12 | External data | ENCODE ChIP-seq bigWig tracks | Histone modification signals across cell types |
| `junction` | 12 | External data | STAR SJ.out.tab / GTEx / recount3 | RNA-seq splice junction read evidence |
| `rbp_eclip` | 8 | External data | ENCODE eCLIP narrowPeak (K562, HepG2) | RNA-binding protein occupancy at splice sites |
| `chrom_access` | 12 | External data | ENCODE ATAC-seq (5 cell lines) + DNase-seq (5 primary tissues) | Chromatin accessibility (open vs closed chromatin) |
| `fm_embeddings` | 8 | Foundation model | Pre-extracted embeddings (Evo2, SpliceBERT, etc.) | Label-agnostic scalar features from foundation model representations |

**Total**: 114 feature columns (full-stack with fm_embeddings enabled).
**Default full-stack**: 106 columns (fm_embeddings commented out by default — requires GPU-extracted embeddings).

---

## Modalities by Source Type

### Group 1: Model Output

#### base_scores (43 columns)

Derives ~43 engineered features from the three raw per-nucleotide probabilities
(`donor_prob`, `acceptor_prob`, `neither_prob`) output by the base model. All features
are **derived** via vectorized Polars expressions. Context-aware features use
`.over('gene_id')` to prevent cross-gene leakage.

Source file: `modalities/base_scores.py`

**Score aliases (3 columns)**

| Column | Description |
|--------|-------------|
| `donor_score` | Alias of `donor_prob` for FeatureSchema compatibility |
| `acceptor_score` | Alias of `acceptor_prob` for FeatureSchema compatibility |
| `neither_score` | Alias of `neither_prob` for FeatureSchema compatibility |

**Context scores (4 columns, default window=2)**

Raw predicted splice probabilities at neighboring positions, extracted via `shift()` within gene groups.

| Column | Description |
|--------|-------------|
| `context_score_m2` | Max(donor, acceptor) probability 2 positions upstream |
| `context_score_m1` | Max(donor, acceptor) probability 1 position upstream |
| `context_score_p1` | Max(donor, acceptor) probability 1 position downstream |
| `context_score_p2` | Max(donor, acceptor) probability 2 positions downstream |

**Derived probability features (7 columns)**

| Column | Type | Description |
|--------|------|-------------|
| `relative_donor_probability` | Derived | donor / (donor + acceptor); donor fraction of splice signal |
| `splice_probability` | Derived | (donor + acceptor) / total; overall splice confidence |
| `donor_acceptor_diff` | Derived | (donor - acceptor) / max(donor, acceptor); normalized type difference |
| `splice_neither_diff` | Derived | (max_splice - neither) / max_all; splice vs background contrast |
| `donor_acceptor_logodds` | Derived | log(donor) - log(acceptor); log-odds of donor vs acceptor |
| `splice_neither_logodds` | Derived | log(donor + acceptor) - log(neither); log-odds of splice vs background |
| `probability_entropy` | Derived | Shannon entropy of the 3-class probability distribution |

**Context pattern features (3 columns)**

| Column | Type | Description |
|--------|------|-------------|
| `context_neighbor_mean` | Derived | Mean of all context scores in the window |
| `context_asymmetry` | Derived | Sum of upstream context - sum of downstream context |
| `context_max` | Derived | Maximum context score across all neighbor positions |

**Donor gradient features (11 columns)**

| Column | Type | Description |
|--------|------|-------------|
| `donor_diff_m1` | Derived | donor_prob minus context score at -1 position |
| `donor_diff_m2` | Derived | donor_prob minus context score at -2 position |
| `donor_diff_p1` | Derived | donor_prob minus context score at +1 position |
| `donor_diff_p2` | Derived | donor_prob minus context score at +2 position |
| `donor_surge_ratio` | Derived | donor_prob / (neighbor_m1 + neighbor_p1); sharpness of peak |
| `donor_is_local_peak` | Derived | Binary: 1 if donor_prob > both immediate neighbors and > 0.001 |
| `donor_weighted_context` | Derived | Gaussian-weighted sum of donor and context scores |
| `donor_peak_height_ratio` | Derived | donor_prob / mean(context); how much position stands out |
| `donor_second_derivative` | Derived | 2*donor_prob - m1 - p1; curvature of score profile |
| `donor_signal_strength` | Derived | donor_prob - mean(context); absolute signal above background |
| `donor_context_diff_ratio` | Derived | donor_prob / max(context); ratio to strongest neighbor |

**Acceptor gradient features (11 columns)**

Same structure as donor gradient features, computed from `acceptor_prob`:

| Column | Type | Description |
|--------|------|-------------|
| `acceptor_diff_m1` | Derived | acceptor_prob minus context score at -1 position |
| `acceptor_diff_m2` | Derived | acceptor_prob minus context score at -2 position |
| `acceptor_diff_p1` | Derived | acceptor_prob minus context score at +1 position |
| `acceptor_diff_p2` | Derived | acceptor_prob minus context score at +2 position |
| `acceptor_surge_ratio` | Derived | acceptor_prob / (neighbor_m1 + neighbor_p1) |
| `acceptor_is_local_peak` | Derived | Binary: 1 if acceptor_prob > both immediate neighbors and > 0.001 |
| `acceptor_weighted_context` | Derived | Gaussian-weighted sum of acceptor and context scores |
| `acceptor_peak_height_ratio` | Derived | acceptor_prob / mean(context) |
| `acceptor_second_derivative` | Derived | 2*acceptor_prob - m1 - p1 |
| `acceptor_signal_strength` | Derived | acceptor_prob - mean(context) |
| `acceptor_context_diff_ratio` | Derived | acceptor_prob / max(context) |

**Cross-type comparative features (4 columns)**

| Column | Type | Description |
|--------|------|-------------|
| `donor_acceptor_peak_ratio` | Derived | donor_peak_height_ratio / acceptor_peak_height_ratio |
| `type_signal_difference` | Derived | donor_signal_strength - acceptor_signal_strength |
| `score_difference_ratio` | Derived | (donor - acceptor) / (donor + acceptor); normalized difference |
| `signal_strength_ratio` | Derived | donor_signal_strength / acceptor_signal_strength |

---

### Group 2: Genomic Annotation

#### annotation (3 columns)

Joins known donor/acceptor positions from pre-extracted GTF splice site annotations
onto the prediction DataFrame. Matches by `(chrom, position, strand)`.

Source file: `modalities/annotation.py`

| Column | Type | Description |
|--------|------|-------------|
| `splice_type` | Raw (from GTF) | Ground truth label: `'donor'`, `'acceptor'`, or `''` (neither) |
| `transcript_id` | Raw (from GTF) | Ensembl transcript ID of the first matching transcript |
| `transcript_count` | Derived | Number of distinct transcripts containing this splice site |

> **Data leakage warning**: `splice_type` is the training label itself and must NEVER
> be used as a feature. It is listed in `FeatureSchema.LEAKAGE_COLS` along with
> `pred_type`, `true_position`, `predicted_position`, `is_correct`, and `error_type`.
> The `transcript_id` and `transcript_count` columns are metadata
> (`FeatureSchema.METADATA_COLS`) and should also be excluded from training features
> due to high cardinality and poor generalization.

#### genomic (4 columns)

Lightweight positional and compositional features derived from gene boundary
coordinates and optionally from the DNA sequence column (requires the `sequence`
modality to run first).

Source file: `modalities/genomic.py`

| Column | Type | Description |
|--------|------|-------------|
| `relative_gene_position` | Derived | Transcriptomic-aware position within gene (0.0 = 5'/TSS, 1.0 = 3'/TES); strand-corrected |
| `distance_to_gene_start` | Derived | Absolute distance (bp) from position to genomic gene start |
| `distance_to_gene_end` | Derived | Absolute distance (bp) from position to genomic gene end |
| `gc_content` | Derived | GC fraction in a central window of the DNA sequence (default 100bp window) |

Optional (when `include_dinucleotides=True`): `cpg_density` (CpG dinucleotide frequency).

---

### Group 3: External Data

#### sequence (3 columns)

Extracts fixed-length DNA windows from the reference FASTA using pyfaidx for efficient
random access. The `sequence` column is consumed by downstream modalities (genomic
context for GC content) and by the meta-layer model as a raw input.

Source file: `modalities/sequence.py`

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | Raw (from FASTA) | DNA sequence string of length `2 * window_size + 1` (default: 1001nt) centered on position |
| `window_start` | Derived | Genomic start coordinate of the extracted window |
| `window_end` | Derived | Genomic end coordinate of the extracted window |

Note: `window_start` and `window_end` are metadata columns, not training features.
The `sequence` column itself is a raw string consumed by the meta-layer's sequence
encoder, not a numeric feature.

#### conservation (9 columns)

Evolutionary constraint scores from UCSC multi-species alignment bigWig tracks.
Build-matched: GRCh38 uses 100-way vertebrate alignment, GRCh37 uses 46-way.
Requires `pyBigWig`. Supports local files and remote UCSC HTTP streaming.

Source file: `modalities/conservation.py`

| Column | Type | Description |
|--------|------|-------------|
| `phylop_score` | Raw (from bigWig) | PhyloP score at the exact position; positive = conserved, negative = fast-evolving |
| `phylop_context_mean` | Derived | Mean PhyloP score in a window around position (default +/-10bp) |
| `phylop_context_max` | Derived | Maximum PhyloP score in the context window |
| `phylop_context_std` | Derived | Standard deviation of PhyloP scores in the context window |
| `phastcons_score` | Raw (from bigWig) | PhastCons probability of being in a conserved element (0-1) |
| `phastcons_context_mean` | Derived | Mean PhastCons score in the context window |
| `phastcons_context_max` | Derived | Maximum PhastCons score in the context window |
| `phastcons_context_std` | Derived | Standard deviation of PhastCons scores in the context window |
| `conservation_contrast` | Derived | `phylop_score - phylop_context_mean`; how much this position stands out from local context |

#### epigenetic (12 columns)

Histone modification signals from ENCODE ChIP-seq fold-change-over-control bigWig
tracks. Default mode is **summarized** (Strategy B): cross-tissue summary statistics
from 8 ENCODE cell lines (K562, GM12878, H1, HepG2, A549, keratinocyte, MCF-7, SK-N-SH).

GRCh38 only. For GRCh37 builds, all columns are filled with NaN (graceful degradation).
Requires `pyBigWig`.

Source file: `modalities/epigenetic.py`

**H3K36me3 (exon body mark) -- 6 columns**

| Column | Type | Description |
|--------|------|-------------|
| `h3k36me3_max_across_tissues` | Derived | Maximum H3K36me3 signal across all cell lines at this position |
| `h3k36me3_mean_across_tissues` | Derived | Mean H3K36me3 signal across cell lines |
| `h3k36me3_tissue_breadth` | Derived | Number of cell lines with signal above threshold (default > 1.5 fold-change) |
| `h3k36me3_variance` | Derived | Variance of H3K36me3 signal across cell lines; high = tissue-specific |
| `h3k36me3_context_mean` | Derived | Mean signal in a 200bp window around position (averaged across cell lines) |
| `h3k36me3_exon_intron_ratio` | Derived | Log2-ratio of upstream (exonic) to downstream (intronic) signal; positive = exon enrichment |

**H3K4me3 (promoter mark) -- 6 columns**

| Column | Type | Description |
|--------|------|-------------|
| `h3k4me3_max_across_tissues` | Derived | Maximum H3K4me3 signal across all cell lines at this position |
| `h3k4me3_mean_across_tissues` | Derived | Mean H3K4me3 signal across cell lines |
| `h3k4me3_tissue_breadth` | Derived | Number of cell lines with signal above threshold |
| `h3k4me3_variance` | Derived | Variance of H3K4me3 signal across cell lines |
| `h3k4me3_context_mean` | Derived | Mean signal in a 200bp window around position |
| `h3k4me3_exon_intron_ratio` | Derived | Log2-ratio of upstream to downstream signal |

#### junction (12 columns)

RNA-seq splice junction read evidence. Features are **sparse** -- they are attributed
to splice site boundary positions (donor and acceptor) via a left-join. Most genomic
positions receive zero values. Supports STAR SJ.out.tab (single sample) and
pre-aggregated multi-tissue tables (GTEx/recount3).

This modality is **label-agnostic**: it produces the same columns regardless of
downstream usage. The meta-layer config determines whether junction columns are
used as features (M2: alternative site detector) or as held-out targets
(M3: novel junction predictor).

Source file: `modalities/junction.py`

| Column | Type | Description |
|--------|------|-------------|
| `junction_log1p` | Derived | `log1p(total_reads)` across all junctions at this boundary position |
| `junction_has_support` | Derived | Binary: 1.0 if any junction evidence exists at this position, 0.0 otherwise |
| `junction_n_partners` | Derived | Number of distinct partner positions (competing donors or acceptors sharing this boundary) |
| `junction_max_reads` | Raw/Derived | Maximum read count from any single junction anchored at this position |
| `junction_entropy` | Derived | Shannon entropy (log2) of read distribution across partner junctions; high = many competing junctions |
| `junction_is_annotated` | Raw (from STAR/GTF) | Binary: 1.0 if any junction at this position is annotated in the reference GTF |
| `junction_tissue_breadth` | Derived | Number of tissues with reads >= breadth_threshold (multi-tissue data only; 0 for single-sample) |
| `junction_tissue_max` | Derived | Maximum read count across tissues |
| `junction_tissue_mean` | Derived | Mean read count across tissues |
| `junction_tissue_variance` | Derived | Variance of read counts across tissues; high = tissue-specific junction usage |
| `junction_psi` | Derived | Percent Spliced In: max_reads / total_reads at position; measures dominance of strongest junction |
| `junction_psi_variance` | Derived | Variance of PSI across tissues (multi-tissue only; NaN for single-sample) |

#### rbp_eclip (8 columns)

RNA-binding protein (RBP) occupancy from ENCODE eCLIP experiments. Features are
**sparse** — most positions have zero values (no overlapping peaks). Uses pre-aggregated
parquet from `scripts/aggregate_eclip_peaks.py` which queries the ENCODE REST API
for IDR-filtered replicate-merged narrowPeak files.

Default cell lines: K562, HepG2. GRCh38 only; GRCh37 returns zero-filled columns.

Source file: `modalities/rbp_eclip.py`

**See**: [`examples/features/docs/rbp-eclip-tutorial.md`](../../examples/features/docs/rbp-eclip-tutorial.md) for biology background and interpretation guide.

| Column | Type | Description |
|--------|------|-------------|
| `rbp_n_bound` | Derived | Count of unique RBPs with binding peaks overlapping this position |
| `rbp_max_signal` | Derived | Maximum fold-enrichment across all overlapping RBP peaks |
| `rbp_max_neg_log10_pvalue` | Derived | Maximum significance (-log10 p-value) among overlapping peaks |
| `rbp_has_splice_regulator` | Derived | Binary: 1.0 if any known splice regulator (SR protein, hnRNP, or core factor) is bound |
| `rbp_n_sr_proteins` | Derived | Count of SR proteins (SRSF1, SRSF3, etc.) with peaks at this position |
| `rbp_n_hnrnps` | Derived | Count of hnRNP proteins (HNRNPA1, HNRNPC, etc.) with peaks |
| `rbp_cell_line_breadth` | Derived | Number of cell lines (0-2) with binding evidence at this position |
| `rbp_mean_signal` | Derived | Mean fold-enrichment across all overlapping peaks |

#### chrom_access (12 columns)

Chromatin accessibility from two complementary ENCODE data sources:
- **ATAC-seq** (fold-change-over-control): 5 cancer cell lines (K562, GM12878, HepG2, A549, IMR-90)
- **DNase-seq** (read-depth normalized): 5 primary tissues (brain cortex, heart, lung, muscle, liver)

Both assays measure nucleosome-free DNA but use different signal normalization
(different scales), so they are kept as separate column groups (`atac_*` and `dnase_*`).
The meta-layer model learns their individual contributions.

GRCh38 only. For GRCh37 builds, all columns are filled with NaN (graceful degradation).
Requires `pyBigWig`.

Source file: `modalities/chrom_access.py`

**See**: [`examples/features/docs/chromatin-accessibility-tutorial.md`](../../examples/features/docs/chromatin-accessibility-tutorial.md) for biology background, ENCODE data sources, and why ATAC/DNase use separate registries.

**ATAC-seq (fold-change, cancer cell lines) — 6 columns**

| Column | Type | Description |
|--------|------|-------------|
| `atac_max_across_tissues` | Derived | Maximum ATAC-seq fold-change signal across all cell lines at this position |
| `atac_mean_across_tissues` | Derived | Mean ATAC-seq signal across cell lines |
| `atac_tissue_breadth` | Derived | Number of cell lines with signal above threshold (default > 2.0 fold-change) |
| `atac_variance` | Derived | Variance of ATAC-seq signal across cell lines; high = tissue-specific accessibility |
| `atac_context_mean` | Derived | Mean signal in a 150bp window around position (averaged across cell lines) |
| `atac_has_peak` | Derived | Binary: 1.0 if maximum signal > peak threshold (default > 3.0 fold-change), 0.0 otherwise |

**DNase-seq (read-depth normalized, primary tissues) — 6 columns**

| Column | Type | Description |
|--------|------|-------------|
| `dnase_max_across_tissues` | Derived | Maximum DNase-seq read-depth signal across primary tissues |
| `dnase_mean_across_tissues` | Derived | Mean DNase-seq signal across tissues |
| `dnase_tissue_breadth` | Derived | Number of tissues with signal above threshold (default > 5.0 read-depth) |
| `dnase_variance` | Derived | Variance of DNase-seq signal across tissues; high = tissue-specific accessibility |
| `dnase_context_mean` | Derived | Mean signal in a 150bp window around position (averaged across tissues) |
| `dnase_has_peak` | Derived | Binary: 1.0 if maximum signal > peak threshold (default > 10.0 read-depth), 0.0 otherwise |

#### fm_embeddings (10 columns)

Label-agnostic scalar features derived from pre-computed foundation model
per-position embeddings. All features are computed without using splice site
annotations, avoiding any risk of label leakage. This modality is a **reader** —
embeddings must be pre-extracted on a GPU pod using the `foundation_models`
sub-project, then PCA-projected into scalar features. Foundation-model-agnostic:
all columns use the `fm_` prefix regardless of the underlying model (Evo2,
SpliceBERT, etc.).

Requires pre-extracted per-chromosome embedding parquets and PCA artifacts
(`.npz` file fit on training chromosomes only). If unavailable, all columns
are filled with NaN (graceful degradation).

Source file: `modalities/fm_embeddings.py`

**See**: [`examples/features/docs/fm-embeddings-tutorial.md`](../../examples/features/docs/fm-embeddings-tutorial.md) for the extraction workflow, PCA fitting, and feature interpretation.

**PCA components (6 columns, default)**

| Column | Type | Description |
|--------|------|-------------|
| `fm_pca_1` | Derived | 1st principal component of the embedding vector (captures dominant variation) |
| `fm_pca_2` | Derived | 2nd principal component |
| `fm_pca_3` | Derived | 3rd principal component |
| `fm_pca_4` | Derived | 4th principal component |
| `fm_pca_5` | Derived | 5th principal component |
| `fm_pca_6` | Derived | 6th principal component |

**Summary statistics (2 columns)**

| Column | Type | Description |
|--------|------|-------------|
| `fm_embedding_norm` | Derived | L2 magnitude of the embedding vector; correlates with model confidence and sequence complexity |
| `fm_local_gradient` | Derived | L2 norm of difference between this position's embedding and the mean of its neighbors within the same gene; detects splice boundary transitions |

**Optional centroid features (disabled by default, `include_cosine_centroids=True`)**

| Column | Type | Description |
|--------|------|-------------|
| `fm_donor_cosine_sim` | Derived | Cosine similarity to the mean donor site embedding centroid (fit on training chromosomes). Disabled by default: uses ground truth labels for centroid computation, redundant with base model scores |
| `fm_acceptor_cosine_sim` | Derived | Cosine similarity to the mean acceptor site embedding centroid. Same caveat as above |

---

## Data Leakage Reference

The `FeatureSchema` (in `meta_layer/core/feature_schema.py`) explicitly tracks columns
that must never be used as training features:

**Leakage columns** (`LEAKAGE_COLS`) -- directly encode or correlate with the label:

- `splice_type` -- the target label itself (from annotation modality)
- `pred_type` -- base model prediction type
- `true_position` -- exact coordinate of real splice site
- `predicted_position` -- tightly correlated with label
- `is_correct` -- whether base model was correct (TP/TN)
- `error_type` -- FP/FN/TP/TN classification

**Metadata columns** (`METADATA_COLS`) -- high cardinality, do not generalize:

- `gene_id`, `transcript_id`, `gene_name`, `gene_type`
- `chrom`, `strand`, `position`, `absolute_position`
- `window_start`, `window_end`, `transcript_count`

Use `FeatureSchema.is_leaky_column(col)` and `FeatureSchema.get_excluded_cols()`
to programmatically enforce these exclusions.

---

## Feature Type Summary

| Category | Count | Examples |
|----------|-------|---------|
| Raw (direct from data source) | ~10 | `donor_score`, `phylop_score`, `phastcons_score`, `sequence`, `junction_is_annotated` |
| Derived (engineered from raw) | ~100 | `probability_entropy`, `donor_surge_ratio`, `conservation_contrast`, `h3k36me3_tissue_breadth`, `rbp_n_bound`, `atac_has_peak`, `fm_pca_1`, `fm_embedding_norm` |
| Labels (never use as features) | 1 | `splice_type` |
| Metadata (not for training) | ~11 | `gene_id`, `chrom`, `position`, `window_start`, `transcript_id` |

---

## Per-Modality Tutorials

For detailed biology background, data source descriptions, and interpretation guidance:

- [Epigenetic Marks Tutorial](../../examples/features/docs/epigenetic-marks-tutorial.md) — H3K36me3/H3K4me3 ChIP-seq
- [RBP eCLIP Tutorial](../../examples/features/docs/rbp-eclip-tutorial.md) — ENCODE RBP binding
- [Chromatin Accessibility Tutorial](../../examples/features/docs/chromatin-accessibility-tutorial.md) — ENCODE ATAC-seq
- [Foundation Model Embeddings Tutorial](../../examples/features/docs/fm-embeddings-tutorial.md) — Evo2/SpliceBERT scalar features

---

*Last Updated: March 27, 2026*
