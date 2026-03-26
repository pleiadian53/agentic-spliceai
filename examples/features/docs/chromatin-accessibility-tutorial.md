# Chromatin Accessibility Modality (ATAC-seq)

## Why Chromatin Accessibility Matters for Splicing

DNA in the nucleus is wrapped around nucleosomes — octamers of histone proteins
that compact ~147 bp of DNA into dense, inaccessible structures. **Chromatin
accessibility** refers to whether a genomic region is nucleosome-free (open) or
nucleosome-occupied (closed):

```
CLOSED chromatin (inaccessible):
  ──[NUC]──[NUC]──[NUC]──[NUC]──[NUC]──
  DNA fully wrapped; TFs, splicing regulators cannot bind

OPEN chromatin (accessible):
  ──[NUC]──     GAP     ──[NUC]──[NUC]──
  Nucleosome-depleted region (NDR);
  DNA exposed; regulatory factors can bind
```

For splice site prediction, accessibility provides orthogonal information to
histone marks:

- **Histone marks** (H3K36me3, H3K4me3) indicate chromatin *state* — what
  modifications are present on nucleosome tails
- **ATAC-seq** measures whether the DNA is *physically accessible* — whether
  regulatory factors can actually bind

A position can have high H3K36me3 (actively transcribed exon body) while being
nucleosome-occupied (not accessible to regulators). This apparent contradiction
is biologically meaningful: exon-body nucleosomes decorated with H3K36me3
recruit splicing repressors like PTBP1 via MRG15, *because* the nucleosome is
positioned there. The meta-layer can learn these interaction patterns when both
modalities are present.

## What ATAC-seq Measures

**ATAC-seq** (Assay for Transposase-Accessible Chromatin using sequencing)
uses Tn5 transposase to insert sequencing adapters preferentially into
accessible (nucleosome-free) DNA. The resulting reads are piled up into
a genome-wide signal track:

- **High signal**: nucleosome-depleted region — DNA is exposed
- **Low/zero signal**: nucleosome-occupied — DNA is wrapped and inaccessible

The ENCODE consortium provides ATAC-seq data as **fold-change-over-control
bigWig** tracks — the same format used for ChIP-seq histone marks. This
normalization accounts for tagmentation bias and library size.

Resolution is ~100–200 bp at peak centers, matching the nucleosome scale.

## Feature Descriptions

| Column | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| `atac_max_across_tissues` | Maximum fold-change across all cell lines | [0, ~50+] | Strongest accessibility in any tissue |
| `atac_mean_across_tissues` | Mean fold-change across cell lines | [0, ~20+] | Average accessibility level |
| `atac_tissue_breadth` | Count of cell lines with signal > threshold | [0, 5] | 5 = constitutively open; 1 = tissue-specific |
| `atac_variance` | Variance of signal across cell lines | [0, ∞) | High = variable accessibility (tissue-regulated) |
| `atac_context_mean` | Mean signal in ±150bp window | [0, ~20+] | Regional accessibility (broader than point signal) |
| `atac_has_peak` | Binary: max signal > peak_threshold | {0, 1} | Strong accessibility at this exact position |

### Key signals for alternative splicing

- **`atac_tissue_breadth = 5`**: Constitutively open — likely a structural
  regulatory element (promoter, insulator). Splice sites here are used across
  tissues.
- **`atac_tissue_breadth = 1–2`**: Tissue-specific accessibility — the
  position is open in some cell types but closed in others. This is where
  tissue-specific alternative splicing events are most likely.
- **`atac_variance` high + `atac_has_peak = 1`**: Variable accessibility
  with strong signal in at least one tissue — candidate for regulated
  alternative splicing.

## ENCODE Data Sources

### Cell line panel (5 cell types)

| Cell Line | Tissue | Experiment | File | Size |
|-----------|--------|------------|------|------|
| K562 | Blood/CML | ENCSR483RKN | ENCFF754EAC | 1.15 GB |
| GM12878 | Lymphoblastoid | ENCSR095QNB | ENCFF487LOB | 870 MB |
| HepG2 | Liver | ENCSR042AWH | ENCFF664EJT | 1.11 GB |
| A549 | Lung adenocarcinoma | ENCSR032RGS | ENCFF872SDF | 5.0 GB |
| IMR-90 | Normal fibroblast | ENCSR200OML | ENCFF282RNO | 1.12 GB |

All files are replicate-pooled, fold-change-over-control bigWig tracks
aligned to GRCh38. Accessed via ENCODE S3 direct links (remote streaming
with pyBigWig).

### Why these cell lines?

- **K562 + GM12878**: ENCODE Tier 1 — best-characterized, most replicates
- **HepG2**: Liver — tissue-specific splicing in metabolic genes
- **A549**: Lung — cancer-related alternative splicing programs
- **IMR-90**: Normal diploid fibroblast — non-cancer baseline for comparison

Note: H1-hESC (embryonic stem cells) does not have released ATAC-seq data on
ENCODE. IMR-90 replaces it as the 5th cell line in this panel.

### Cancer cell line bias

Three of five cell lines are cancer-derived (K562, HepG2, A549). Cancer cells
often have globally altered chromatin accessibility. However:

1. The cross-tissue summary (Strategy B) mitigates single-cell-line artifacts
2. IMR-90 provides a non-cancer reference
3. `atac_tissue_breadth` distinguishes constitutive (all 5) from cancer-specific (1–2) accessibility
4. The meta-layer sees both accessibility and other modalities — it can learn
   to weight cancer-specific signals appropriately

## Relationship to Other Modalities

```
Modality              What it measures                    Complementary signal
────────────────────  ──────────────────────────────────  ──────────────────────
Epigenetic (ChIP)     Histone modification state           ← chromatin marks
Chrom Access (ATAC)   Physical DNA accessibility           ← nucleosome occupancy
RBP eCLIP             Protein binding at splice sites      ← regulatory binding
Conservation          Evolutionary constraint              ← purifying selection
Junction (GTEx)       RNA-seq splice evidence              ← transcript validation
```

ATAC-seq is most complementary to:
- **Epigenetic marks**: H3K36me3 + low ATAC → nucleosome-positioned exon body;
  H3K4me3 + high ATAC → active promoter (NDR)
- **RBP eCLIP**: Open chromatin is a prerequisite for RBP binding — positions
  with high ATAC + RBP binding are actively regulated splice sites

## Configuration Reference

### YAML config (full_stack.yaml)

```yaml
pipeline:
  modalities:
    - chrom_access              # +6 columns

modality_configs:
  chrom_access:
    cell_lines:
      - K562
      - GM12878
      - HepG2
      - A549
      - IMR-90
    window: 150                 # half-window (bp) for context_mean
    aggregation: summarized     # Strategy B (cross-tissue summary)
    signal_threshold: 2.0       # fold-change cutoff for tissue_breadth
    peak_threshold: 3.0         # fold-change cutoff for has_peak
    remote_fallback: true       # stream from ENCODE S3 if local cache miss
```

### Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_lines` | K562, GM12878, HepG2, A549, IMR-90 | ENCODE cell lines to query |
| `window` | 150 | Half-window (bp) for context mean. ATAC peaks ~200bp. |
| `aggregation` | summarized | Only 'summarized' (Strategy B) is supported |
| `signal_threshold` | 2.0 | Fold-change cutoff for tissue breadth count |
| `peak_threshold` | 3.0 | Higher cutoff for binary `atac_has_peak` |
| `cache_dir` | None | Local bigWig cache (avoids remote streaming) |
| `remote_fallback` | True | Fall back to ENCODE S3 when local files not found |

### Build compatibility

- **GRCh38 (openspliceai)**: Full support — all 5 cell lines available
- **GRCh37 (spliceai)**: All columns filled with NaN (ENCODE ATAC-seq is GRCh38 only)

## References

- Buenrostro et al. (2013). Transposition of native chromatin for fast and
  sensitive epigenomic profiling of open chromatin, DNA-binding proteins and
  nucleosome position. *Nature Methods* 10, 1213–1218.
- Tilgner et al. (2009). Nucleosome positioning as a determinant of exon
  recognition. *Nature Structural & Molecular Biology* 16, 996–1001.
- Schwartz et al. (2009). Chromatin organization marks exon-intron structure.
  *Nature Structural & Molecular Biology* 16, 990–995.
- ENCODE Project Consortium (2020). Expanded encyclopaedias of DNA elements
  in the human and mouse genomes. *Nature* 583, 699–710.
