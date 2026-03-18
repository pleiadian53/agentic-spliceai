# Epigenetic Marks for Splice Site Prediction

## Why Epigenetics Matters for Splicing

Base-layer models (SpliceAI, OpenSpliceAI) predict splice sites from **DNA sequence alone**.
Conservation scores add evolutionary constraint. But neither captures a critical dimension:
**which exons are actually used in a given cell type?**

Alternative splicing is fundamentally **tissue-dependent** — an exon included in brain
may be skipped in liver. Epigenetic marks, specifically histone modifications, provide
direct chromatin-level evidence of exon usage in specific cellular contexts.

Two histone marks are particularly informative for splicing:

| Mark | What it marks | Mechanism | Signal at splice sites |
|------|-------------|-----------|----------------------|
| **H3K36me3** | Active exon bodies | Deposited by SETD2 during Pol II elongation; spliceosome recruits SETD2 | High upstream (exonic), low downstream (intronic) at donor sites |
| **H3K4me3** | Active promoters / TSS | Marks transcription initiation | Enriched at 5' end of active genes |

H3K36me3 is the more directly useful signal for splice site prediction — its
enrichment on exon bodies during transcription creates a distinctive
**exon-intron boundary signature** that the meta-layer can learn.

## What the Signal Measures

Unlike conservation (evolutionary, universal) or base scores (sequence pattern),
epigenetic signal is:

1. **Cell-type-specific**: The same position has different values in K562 vs HepG2
2. **Relative**: Measured as **fold-change over input control** (dimensionless ratio)
3. **Regional**: Reflects chromatin state in a ~200bp neighborhood (nucleosome scale)

### The Measurement Pipeline

```
Wet lab (ChIP-seq)                    Computational (ENCODE pipeline)
──────────────────                    ────────────────────────────────
Crosslink DNA-protein                 Align reads → genome (BWA/Bowtie2)
Shear to ~200bp fragments             Count reads per position (pileup)
Immunoprecipitate with                Compute: ChIP pileup / Input pileup
  anti-H3K36me3 antibody                = fold-change signal
Sequence pulled-down DNA              Smooth & normalize (MACS2)
                                      Write to bigWig format
Control: same process but
  WITHOUT antibody (input)
```

### Interpreting the Values

- **Units**: Dimensionless (fold-change over input control)
- **Range**: 0–30+ (most of genome is 0–1; enriched regions: 2–15; strong peaks: 15–30+)
- **Zero**: No enrichment over background (not necessarily "no mark" — just at background level)
- **Comparison**: A fold-change of 5.0 means "5× more ChIP reads than input at this position"

### Per-Nucleotide Resolution?

BigWig files store values at **single-base resolution** — you CAN query a single position
and get a float value, just like conservation. This is critical for feature alignment.

However, the biological resolution is ~200bp (one nucleosome wraps 147bp of DNA, and
ChIP-seq fragments are 200–500bp). So the value at position p represents the **local
chromatin neighborhood**, not a nucleotide-level property. This means:

- Windowed features (±200–500bp) are more appropriate than for conservation (±10bp)
- The `exon_intron_ratio` feature (upstream vs downstream signal) is biologically
  meaningful because it captures the H3K36me3 exon-body enrichment across the boundary

## The Tissue Panel

We use ENCODE ChIP-seq fold-change-over-control bigWig files (GRCh38, replicate-pooled).
The panel spans 8 cell types representing diverse tissues:

| Cell Type | ENCODE Name | Tissue | Tier | Key Biology |
|-----------|-------------|--------|------|-------------|
| K562 | K562 | Blood (CML) | Tier 1 | Hematopoietic splicing programs |
| GM12878 | GM12878 | Lymphoblastoid | Tier 1 | Normal B-cell reference |
| H1 | H1 | Embryonic stem cells | Tier 1 | Pluripotent — many exons active |
| HepG2 | HepG2 | Liver (hepatocarcinoma) | Tier 2 | Liver-specific splicing |
| A549 | A549 | Lung (adenocarcinoma) | — | Epithelial context |
| Keratinocyte | keratinocyte | Skin (primary) | — | Primary cells (non-cancer) |
| MCF-7 | MCF-7 | Breast (adenocarcinoma) | — | Hormone-responsive |
| SK-N-SH | SK-N-SH | Brain (neuroblastoma) | — | Neural splicing programs |

### Why These Cell Types?

- **Tier 1 lines** (K562, GM12878, H1): Best characterized, highest data quality
- **Tissue diversity**: Blood, liver, lung, skin, breast, brain — maximizes the
  variance signal for detecting tissue-specific alternative splicing
- **Primary cells** (keratinocyte): Not all cancer lines — includes one primary
  cell type for comparison
- **Neural** (SK-N-SH): Brain has the most complex alternative splicing of any tissue

### Minimal Panel (5 cell types, ~3.7 GB total)

For resource-constrained environments, use: **K562, GM12878, H1, HepG2, Keratinocyte**.
These cover the Tier 1 lines plus liver and skin diversity.

## Track Registry

All URLs are ENCODE S3 direct links (stable, no redirects). Files are
fold-change-over-control, GRCh38, replicate-pooled.

### H3K36me3 (Exon Body Mark)

| Cell Type | Accession | Experiment | Size |
|-----------|-----------|------------|------|
| K562 | ENCFF163NTH | ENCSR000AKR | 459 MB |
| GM12878 | ENCFF312MUY | ENCSR000AKE | 234 MB |
| H1 | ENCFF141YAA | ENCSR476KTK | 365 MB |
| HepG2 | ENCFF414GNH | ENCSR000AMB | 541 MB |
| A549 | ENCFF473XIC | ENCSR000AUL | 2,118 MB |
| Keratinocyte | ENCFF049JNX | ENCSR000ALM | 263 MB |
| MCF-7 | ENCFF685ECA | ENCSR610IYQ | 1,881 MB |
| SK-N-SH | ENCFF681BMN | ENCSR978CNH | 2,427 MB |

### H3K4me3 (Promoter Mark)

| Cell Type | Accession | Experiment | Size |
|-----------|-----------|------------|------|
| K562 | ENCFF814IYI | ENCSR000AKU | 282 MB |
| GM12878 | ENCFF776DPQ | ENCSR000AKA | 628 MB |
| H1 | ENCFF602IAY | ENCSR443YAS | 230 MB |
| HepG2 | ENCFF284FVP | ENCSR000AMP | 541 MB |
| A549 | ENCFF242FAU | ENCSR944WVU | 1,430 MB |
| Keratinocyte | ENCFF632PUY | ENCSR970FPM | 616 MB |
| MCF-7 | ENCFF267OQQ | ENCSR985MIB | 1,548 MB |
| SK-N-SH | ENCFF755FHH | ENCSR975GZA | 2,152 MB |

> **Note on file sizes**: pyBigWig reads bigWig files via HTTP range requests — only
> the queried regions are downloaded, not the entire file. Large file sizes do not
> affect per-region query speed.

## Feature Design: Strategy B (Tissue-Summarized)

Rather than one column per cell type (Strategy A, scales poorly), we compute
**summary statistics across the tissue panel** per mark:

```python
# For each mark (e.g., h3k36me3), per splice site position:
features = {
    "h3k36me3_max_across_tissues":    max(scores_across_tissues),
    "h3k36me3_mean_across_tissues":   mean(scores_across_tissues),
    "h3k36me3_tissue_breadth":        count(score > threshold for each tissue),
    "h3k36me3_variance":              var(scores_across_tissues),
    "h3k36me3_context_mean":          mean(windowed_signal, pooled),
    "h3k36me3_exon_intron_ratio":     upstream_signal / downstream_signal,
}
```

### What Each Feature Captures

| Feature | Biological Meaning | Value for Meta-Layer |
|---------|-------------------|---------------------|
| `max_across_tissues` | Peak activity in any tissue | Detects constitutive exons |
| `mean_across_tissues` | Average activity | General exon usage level |
| `tissue_breadth` | How many tissues use this exon | High = constitutive, low = alternative |
| `variance` | Variation across tissues | High = tissue-specific splicing |
| `context_mean` | Regional chromatin state | Smoothed exon body signal |
| `exon_intron_ratio` | Exon marking at boundary | >1.0 = upstream (exon) is marked; ~1.0 = exon may be skipped |

### The Exon-Intron Ratio

This is the most biologically specific feature. At a donor site (exon → intron boundary):

```
        Exon Body              Intron
    ──────────────│─────────────────
    H3K36me3 HIGH │  H3K36me3 LOW        ← Exon is actively used
    ratio ≫ 1.0   │
                  │
    H3K36me3 ~0   │  H3K36me3 ~0         ← Exon is skipped in this tissue
    ratio ≈ 1.0   │
```

For the tissue-summarized version, we pool the ratio across tissues:
- High max ratio = exon used in at least one tissue
- High variance in ratio = tissue-specific exon inclusion

## Build Considerations

ENCODE bigWig files are **GRCh38 only**. Unlike conservation (where both hg19 and
hg38 tracks exist natively), ENCODE does not provide hg19 signal tracks.

| Build | Conservation | Epigenetic |
|-------|-------------|-----------|
| GRCh38 (OpenSpliceAI) | 100-way PhyloP/PhastCons | Full ENCODE panel |
| GRCh37 (SpliceAI) | 46-way PhyloP/PhastCons | **Not available** |

For GRCh37/SpliceAI users, the `EpigeneticModality` should:
1. Gracefully degrade: fill with NaN/null values if no tracks exist for the build
2. Log a clear warning explaining the build limitation
3. The meta-layer handles missing modalities via feature masking (NaN-aware models)

This asymmetry is another reason to prefer OpenSpliceAI (GRCh38) as the primary
base model for multimodal meta-layer training.

## Integration with the Feature Pipeline

The `EpigeneticModality` follows the same `Modality` protocol as conservation:

```python
from agentic_spliceai.splice_engine.features.modalities.epigenetic import (
    EpigeneticConfig, EpigeneticModality
)

# In YAML config (configs/full_stack.yaml):
# modality_configs:
#   epigenetic:
#     marks: [h3k36me3, h3k4me3]
#     cell_lines: [K562, GM12878, H1, HepG2, keratinocyte]  # minimal panel
#     window: 200
#     aggregation: summarized
```

Expected output columns (Strategy B, 2 marks):
- Per mark (6 columns): max, mean, breadth, variance, context_mean, exon_intron_ratio
- Total: **12 columns** for 2 marks

Combined with existing modalities:
```
base_scores:    43 columns
annotation:      3 columns
genomic:         4 columns
conservation:    9 columns
epigenetic:     12 columns
────────────────────────────
Total:          71 feature columns
```

## References

- **SETD2 and H3K36me3 in splicing**: Luco et al. (2010) Science 327:996-1000
  "Regulation of alternative splicing by histone modifications"
- **ENCODE ChIP-seq pipeline**: ENCODE Project Consortium. "An integrated encyclopedia
  of DNA elements in the human genome." Nature 489, 57-74 (2012)
- **Exon marking by H3K36me3**: Kolasinska-Zwierz et al. (2009) Nature Genetics 41:376-381
  "Differential chromatin marking of introns and expressed exons by H3K36me3"
- **Tissue-specific splicing**: Wang et al. (2008) Nature 456:470-476
  "Alternative isoform regulation in human tissue transcriptomes"
