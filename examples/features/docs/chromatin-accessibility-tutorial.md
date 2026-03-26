# Chromatin Accessibility Modality (ATAC-seq + DNase-seq)

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
- **ATAC-seq / DNase-seq** measure whether the DNA is *physically accessible* —
  whether regulatory factors can actually bind

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

## What DNase-seq Measures

**DNase-seq** (DNase I hypersensitivity sequencing) uses the DNase I enzyme
to cleave accessible DNA. Like ATAC-seq, it maps nucleosome-free regions,
but with slightly different bias profiles and signal normalization:

- **ATAC-seq**: fold-change over control (typical values 0–20)
- **DNase-seq**: read-depth normalized signal (typical values 0–100+)

The two assays correlate highly (r > 0.9 in matched samples) but use
**fundamentally different scales**. This is why they are stored in separate
registries (`ATAC_TRACK_REGISTRY` and `DNASE_TRACK_REGISTRY`) and produce
separate column groups (`atac_*` and `dnase_*`) within the same modality.

### Why not merge them into one registry?

Mixing fold-change and read-depth values in the same `max_across_tissues`
statistic would be meaningless — a fold-change of 3.0 and a read-depth of
3.0 represent completely different signal strengths. The thresholds also
differ: `signal_threshold=2.0` for ATAC fold-change vs `5.0` for DNase
read-depth. Keeping them as separate column groups lets the meta-layer
model learn their individual contributions without scale confusion.

### Complementary tissue coverage

The key advantage of including DNase-seq is **tissue diversity**:

- **ATAC-seq panel**: 5 cancer cell lines (K562, GM12878, HepG2, A549, IMR-90)
- **DNase-seq panel**: 5 primary tissues (brain cortex, heart, lung, muscle, liver)

Primary tissues provide non-cancer, non-immortalized accessibility data
that is critical for predicting tissue-specific alternative splicing events
in normal biology — something the cancer cell line panel cannot capture.

## Feature Descriptions

### ATAC-seq features (6 columns, `atac_*` prefix)

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

### DNase-seq features (6 columns, `dnase_*` prefix)

| Column | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| `dnase_max_across_tissues` | Maximum read-depth signal across primary tissues | [0, ~200+] | Strongest accessibility in any tissue |
| `dnase_mean_across_tissues` | Mean read-depth signal across tissues | [0, ~100+] | Average accessibility level |
| `dnase_tissue_breadth` | Count of tissues with signal > threshold | [0, 5] | 5 = constitutively open; 1 = tissue-specific |
| `dnase_variance` | Variance of signal across tissues | [0, ∞) | High = variable accessibility (tissue-regulated) |
| `dnase_context_mean` | Mean signal in ±150bp window | [0, ~100+] | Regional accessibility |
| `dnase_has_peak` | Binary: max signal > peak_threshold | {0, 1} | Strong accessibility at this position |

Note: DNase values are on a **different scale** from ATAC values (read-depth
normalized vs fold-change). Do not directly compare `atac_max_across_tissues`
with `dnase_max_across_tissues` — let the model learn their separate contributions.

## ENCODE Data Sources

### ATAC-seq panel (5 cancer cell lines)

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

### Cancer vs normal: noise or signal?

Three of five ATAC-seq cell lines are cancer-derived (K562, HepG2, A549),
while all five DNase-seq samples are normal primary tissues. Cancer cells
have globally altered chromatin accessibility — aberrant enhancer activation,
widespread remodeling, and accessibility patterns that don't exist in normal
tissues. This is not a confound to eliminate; it's a biologically meaningful
axis of variation that different meta models use differently:

**M1 (Canonical)**: Minimal impact. Canonical splice sites are constitutively
used — their accessibility is similar across normal and cancer tissues. Both
panels agree on these sites.

**M2 (Alternative)**: Both panels provide complementary signal. Cancer cell
lines reveal cancer-specific alternative splicing programs; primary tissues
reveal normal tissue-specific events. The model learns which accessibility
features predict alternative site usage in each context.

**M3 (Novel discovery)**: DNase is especially valuable here. "Novel" splice
sites fall into two categories: (1) sites constitutively used in normal
biology but unannotated (MANE catalogs ~19K genes vs Ensembl's ~57K), and
(2) sites that become visible as drug targets only through multimodal
evidence fusion — high conservation + tissue-specific accessibility + RBP
binding marks a legitimate target that base models alone would miss. A site
with `atac_has_peak = 1` but `dnase_has_peak = 0` is likely a cancer
artifact rather than a broadly relevant novel site — the DNase panel helps
M3 filter these out.

**M4 (Perturbation-induced)**: The cancer/normal contrast is exactly what
M4 needs. A position where `dnase_tissue_breadth = 0` (closed in all
normal tissues) but `atac_tissue_breadth = 3` (open in cancer cell lines)
is a cancer-induced accessibility event — precisely the perturbation signal
M4 is designed to detect.

By keeping ATAC and DNase as separate column groups, the model can learn
when cancer-specific accessibility matters (M4) vs when normal tissue
accessibility matters (M2, M3) — without us hard-coding the comparison.

### DNase-seq panel (5 primary tissues)

| Tissue | Experiment | File | Size |
|--------|------------|------|------|
| Brain (frontal cortex) | ENCSR000EIY | ENCFF243QQE | 662 MB |
| Heart | ENCSR989YAW | ENCFF885WJZ | 249 MB |
| Lung | ENCSR141IUS | ENCFF958NZO | 410 MB |
| Muscle (psoas) | ENCSR019MYA | ENCFF952ARY | 316 MB |
| Liver | ENCSR562FNN | ENCFF017TVV | 645 MB |

All files are read-depth normalized signal bigWig tracks aligned to GRCh38.
Note: DNase-seq does not produce fold-change-over-control tracks on ENCODE —
the processing pipeline differs from ATAC-seq.

### Why primary tissues complement cancer cell lines

The ATAC-seq panel provides cancer cell line accessibility (useful for M4
perturbation models). The DNase-seq panel adds:

- **Brain cortex**: Neuronal alternative splicing — the most complex
  alternative splicing program in the human body
- **Heart**: Cardiac-specific exon usage (e.g., titin, troponin isoforms)
- **Lung**: Respiratory tissue splicing patterns
- **Muscle**: Muscle-specific isoforms (myosin heavy chain, dystrophin)
- **Liver**: Primary hepatocyte accessibility (vs HepG2 cancer line)

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
    - chrom_access              # +12 columns (6 ATAC + 6 DNase)

modality_configs:
  chrom_access:
    # ATAC-seq (fold-change, cancer cell lines)
    cell_lines:
      - K562
      - GM12878
      - HepG2
      - A549
      - IMR-90
    window: 150                 # half-window (bp) for context_mean
    aggregation: summarized     # Strategy B (cross-tissue summary)
    signal_threshold: 2.0       # ATAC fold-change cutoff for tissue_breadth
    peak_threshold: 3.0         # ATAC fold-change cutoff for has_peak
    # DNase-seq (read-depth normalized, primary tissues)
    dnase_cell_lines:
      - brain_cortex
      - heart
      - lung
      - muscle
      - liver
    dnase_signal_threshold: 5.0   # read-depth scale (higher than ATAC)
    dnase_peak_threshold: 10.0    # read-depth scale
    remote_fallback: true
```

### Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_lines` | K562, GM12878, HepG2, A549, IMR-90 | ATAC-seq cell lines (fold-change signal) |
| `window` | 150 | Half-window (bp) for context mean |
| `aggregation` | summarized | Only 'summarized' (Strategy B) is supported |
| `signal_threshold` | 2.0 | ATAC fold-change cutoff for `atac_tissue_breadth` |
| `peak_threshold` | 3.0 | ATAC fold-change cutoff for `atac_has_peak` |
| `dnase_cell_lines` | brain_cortex, heart, lung, muscle, liver | DNase-seq primary tissues (read-depth signal) |
| `dnase_signal_threshold` | 5.0 | DNase read-depth cutoff for `dnase_tissue_breadth` |
| `dnase_peak_threshold` | 10.0 | DNase read-depth cutoff for `dnase_has_peak` |
| `cache_dir` | None | Local bigWig cache (avoids remote streaming) |
| `remote_fallback` | True | Fall back to ENCODE S3 when local files not found |

### Refreshing after config changes

When you add new cell lines or change thresholds, use `--refresh` to recompute
the chrom_access columns in existing artifacts:

```bash
python 06_multimodal_genome_workflow.py --config configs/full_stack.yaml \
    --chromosomes chr22 --refresh chrom_access
```

This drops the old `atac_*` and `dnase_*` columns and recomputes them with
the current config.

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
