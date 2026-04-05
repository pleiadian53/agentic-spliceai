# Junction Reads Modality — Tutorial

**Date**: 2026-03-17
**Modality**: `junction` (7th registered modality)
**Output**: 12 columns per position

---

## What Are Junction Reads?

A **splice junction** is a (donor, acceptor) pair representing one intron-removal event.
RNA-seq aligners like STAR detect junctions from **split reads** — reads that span the
exon-intron boundary. Each junction receives a **read count**: the number of RNA-seq reads
supporting it.

```
Exon 1                    Exon 2
=========>                <=========
          GT............AG
          ^                ^
          donor             acceptor
          (intron_start-1)  (intron_end+1)

Split read: =====>---...---<=====
                  ↑ junction support count
```

Junction reads are the **most direct evidence** of splicing. Unlike conservation (a proxy)
or histone marks (a correlate), a junction read can only exist if splicing happened at
that exact pair of coordinates.

---

## Data Preparation

### Option A: From STAR SJ.out.tab (Single Sample)

STAR produces a headerless 9-column TSV (`SJ.out.tab`) during alignment:

```
chr1    14830    14970    2    2    0    3    0    38
chr1    17056    17233    2    2    0    6    0    15
```

Columns: `chrom, intron_start, intron_end, strand, intron_motif, annotated, unique_reads, multi_reads, max_overhang`

The junction modality auto-detects this format and converts intron coordinates to boundary positions:
- `donor_pos = intron_start - 1` (last exonic base)
- `acceptor_pos = intron_end + 1` (first exonic base of next exon)

### Option B: Pre-Aggregated Multi-Tissue Table (Recommended)

For meta-layer training, use a TSV with header containing per-tissue junction counts:

```tsv
chrom    donor_pos    acceptor_pos    strand    unique_reads    annotated    gene_id    tissue
chr1     14829        14971           -         45              1            ENSG00000227232    brain
chr1     14829        14971           -         120             1            ENSG00000227232    liver
chr1     14829        15039           -         8               0            ENSG00000227232    brain
```

This enables tissue-breadth and variance features. Sources:
- **GTEx v8**: 17K samples, 54 tissues (download sQTL/junction files from GTEx Portal)
- **recount3**: 750K+ uniformly-processed samples (use `recount3` R package)
- **ENCODE**: Factor knockdown experiments for induced splicing

### Aggregation Script

The `scripts/data/aggregate_gtex_junctions.py` script downloads GTEx v8
junction files from the GTEx Portal, aggregates across tissues, and writes
the parquet files consumed by the junction modality.

```bash
python scripts/data/aggregate_gtex_junctions.py \
    --output data/mane/GRCh38/junction_data/
```

Output: `junctions_gtex_v8.parquet` (353K junctions, 54 tissues) plus
per-tissue parquets in `data/mane/GRCh38/junction_data/by_tissue/`.

### Placing the Data

The junction modality resolves data via the resource registry:

```
data/{annotation_source}/{build}/junctions.tsv
```

For OpenSpliceAI (GRCh38/MANE): `data/mane/GRCh38/junctions.tsv`
For SpliceAI (GRCh37/Ensembl): `data/ensembl/GRCh37/junctions.tsv`

Or specify explicitly in the YAML config:

```yaml
modality_configs:
  junction:
    junction_data_path: /path/to/my/junctions.tsv
```

---

## Feature Schema (12 Columns)

Junction features are **sparse**: they are attributed to splice site boundary positions
(donor and acceptor). Non-boundary positions receive 0.0 for count-based features.

### Count-Based Features (0.0 for non-boundaries)

| Column | Description | Range |
|--------|-------------|-------|
| `junction_log1p` | log1p(total reads across junctions at this position) | [0, ~12] |
| `junction_has_support` | Binary: any junction with reads >= min_support | {0.0, 1.0} |
| `junction_n_partners` | Count of alternative partner positions | [0, ~20] |
| `junction_max_reads` | Max reads across junctions anchored here | [0, ~10K] |
| `junction_entropy` | Shannon entropy of read distribution across partners | [0, ~4] |
| `junction_is_annotated` | Any junction at this position annotated in GTF | {0.0, 1.0} |
| `junction_tissue_breadth` | Tissues with reads >= breadth_threshold | [0, ~54] |
| `junction_tissue_max` | Max reads across tissues | [0, ~10K] |
| `junction_tissue_mean` | Mean reads across tissues | [0, ~5K] |
| `junction_tissue_variance` | Variance across tissues (tissue specificity) | [0, ~1M] |

### Ratio-Based Features (NaN for non-boundaries)

| Column | Description | Range |
|--------|-------------|-------|
| `junction_psi` | Simplified PSI: max_reads / total_reads at position | [0, 1] or NaN |
| `junction_psi_variance` | PSI variance across tissues | [0, 0.25] or NaN |

---

## Feature Interpretation

### Competing Junctions and Entropy

When multiple junctions share a boundary position, this indicates **alternative splicing**:

```
                    Acceptor A (100 reads)
                   /
Donor ────────────
                   \
                    Acceptor B (50 reads)
                    \
                     Acceptor C (10 reads)
```

At the donor position:
- `junction_n_partners = 3`
- `junction_entropy = -( 0.625*log2(0.625) + 0.3125*log2(0.3125) + 0.0625*log2(0.0625) ) ≈ 1.30`
- `junction_psi = 100/160 = 0.625` (dominance of strongest junction)

High entropy = balanced alternative splicing. Low entropy = one dominant junction.

### Tissue Breadth and Variance

- **High breadth, low variance**: Constitutive splice site (used in most tissues similarly)
- **High breadth, high variance**: Tissue-regulated alternative splicing
- **Low breadth**: Tissue-specific event (only used in a few cell types)

### Absence of Evidence

A position with `junction_has_support = 0` is informative:
- High base model score + zero junction support + high conservation = possibly tissue-specific (not in available samples)
- Moderate base score + zero junction support + low conservation = likely false positive
- This is why non-boundary positions get **0.0** (not NaN) — the meta-layer learns from absence.

---

## YAML Configuration

### In full_stack.yaml (M2: junction as feature)

```yaml
pipeline:
  modalities:
    - base_scores
    - annotation
    - sequence
    - genomic
    - conservation
    - epigenetic
    - junction          # junction evidence as input feature

modality_configs:
  junction:
    min_support: 3      # minimum unique reads to count
    aggregation: summarized
    breadth_threshold: 3
    include_psi: true
```

### In meta_m3_novel.yaml (M3: junction as target)

```yaml
pipeline:
  modalities:
    - base_scores
    - annotation
    - sequence
    - genomic
    - conservation
    - epigenetic
    # junction: EXCLUDED — used as prediction target
```

In M3, junction features (e.g., `junction_has_support`, `junction_psi`) are loaded
separately at training time as the label columns.

---

## Connection to Meta-Layer Models

| Meta Model | Junction Role | Purpose |
|---|---|---|
| **M1**: Enhanced canonical | Not needed | Confirm known splice sites (GTF labels) |
| **M2**: Alternative site detector | **Feature** | Tissue breadth + PSI signal alt splicing |
| **M3**: Novel site predictor | **Target** | Predict which sites have junction support without RNA-seq |
| **M4**: Perturbation/induced | **Validation** | Validate predicted delta scores against observed junctions |

The modality itself is **label-agnostic** — it always produces the same 12 columns.
The feature-vs-target distinction is handled by the meta-layer training configuration.

---

## Graceful Degradation

If no junction data file is available:
- All 12 columns are filled with 0.0 (count-based) or NaN (PSI)
- A warning is logged
- The pipeline continues with other modalities
- This allows the same YAML config to work across environments with and without RNA-seq data

---

## References

- STAR Manual: [github.com/alexdobin/STAR](https://github.com/alexdobin/STAR)
- GTEx Portal: [gtexportal.org](https://gtexportal.org)
- recount3: [rna.recount.bio](https://rna.recount.bio)
- SpliceVault: [github.com/kidsneuro-lab/SpliceVault](https://github.com/kidsneuro-lab/SpliceVault)
- MFASS (validation): Cheung et al., Molecular Cell 2019
