# Cross-Build FASTA Inference Validation

## Problem

The meta-layer's FASTA inference mode (`08_evaluate_sequence_model.py --fasta`)
produces `[L, 3]` splice site predictions on arbitrary sequences.  Without
ground truth labels, these predictions cannot be evaluated quantitatively.

This document describes how to generate ground truth labels for a custom
genome build (e.g., T2T-CHM13) and use them to validate FASTA predictions.

---

## The Pipeline

```
Step 1: Generate ground truth          Step 2: Run inference
─────────────────────────              ───────────────────────
Custom GTF                             Custom FASTA
(T2T-CHM13, pangenome, etc.)           (gene sequences from same build)
       │                                       │
       ▼                                       ▼
04_generate_ground_truth.py            08_evaluate_sequence_model.py
  --gtf custom.gtf                       --fasta genes.fa
  --output data/custom/                  --checkpoint best.pt
       │                                       │
       ▼                                       ▼
splice_sites_enhanced.tsv              fasta_predictions.tsv
(chrom, position, splice_type)         (sequence_id, position, donor/acc/neither)


Step 3: Compare predictions against ground truth
────────────────────────────────────────────────
splice_sites_enhanced.tsv  +  fasta_predictions.tsv
                │
                ▼
        Concordance analysis
        (TP, FP, FN at known splice positions)
```

---

## Step 1: Generate Ground Truth Labels

Use `04_generate_ground_truth.py` with `--gtf` to extract splice sites
from any GTF annotation file:

```bash
# T2T-CHM13 example
python examples/data_preparation/04_generate_ground_truth.py \
    --gtf /path/to/chm13v2.0_RefSeq_Curated.gtf \
    --output data/t2t_chm13/ \
    --build T2T-CHM13
```

This produces `data/t2t_chm13/splice_sites_enhanced.tsv` with all splice
sites from the annotation.  See
`examples/data_preparation/docs/ground_truth_custom_genomes.md` for
details on supported formats and chromosome naming.

---

## Step 2: Extract Gene Sequences

Extract gene sequences from the custom genome's FASTA using the same
coordinates as the GTF annotation:

```python
import pyfaidx
import polars as pl

# Load ground truth to get gene coordinates
sites = pl.read_csv("data/t2t_chm13/splice_sites_enhanced.tsv", separator="\t")

# Get unique genes with their genomic spans
genes = (
    sites.group_by("gene_id", "gene_name", "chrom")
    .agg(
        pl.col("position").min().alias("start"),
        pl.col("position").max().alias("end"),
    )
)

# Extract sequences
fasta = pyfaidx.Fasta("/path/to/chm13v2.0.fa")
with open("data/t2t_chm13/genes.fa", "w") as f:
    for row in genes.iter_rows(named=True):
        chrom = row["chrom"]
        start = max(0, row["start"] - 5000)  # 5kb flanking
        end = row["end"] + 5000
        seq = str(fasta[chrom][start:end]).upper()
        f.write(f">{row['gene_id']}|{row['gene_name']}|{chrom}:{start}-{end}\n")
        f.write(f"{seq}\n")
```

The FASTA header encodes the genomic coordinates needed for Step 3.

---

## Step 3: Run FASTA Inference

```bash
python examples/meta_layer/08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --fasta data/t2t_chm13/genes.fa \
    --output-format parquet \
    --output-dir output/meta_layer/t2t_chm13_eval/ \
    --device cpu
```

This produces `fasta_predictions.parquet` with per-position probabilities.

---

## Step 4: Compare Predictions to Ground Truth

Map predicted positions back to genomic coordinates (using the FASTA header
encoding from Step 2), then check against the ground truth:

```python
import polars as pl
import numpy as np

# Load predictions
preds = pl.read_parquet("output/meta_layer/t2t_chm13_eval/fasta_predictions.parquet")

# Load ground truth
truth = pl.read_csv("data/t2t_chm13/splice_sites_enhanced.tsv", separator="\t")
truth_set = set(zip(truth["chrom"], truth["position"], truth["splice_type"]))

# Parse genomic coordinates from sequence_id
# Format: gene_id|gene_name|chrom:start-end
def parse_header(seq_id: str):
    parts = seq_id.split("|")
    coord = parts[2]
    chrom, span = coord.split(":")
    start, end = span.split("-")
    return chrom, int(start)

# Evaluate concordance
tp, fp, fn = 0, 0, 0
threshold = 0.5  # classify as splice site if max(donor, acceptor) > threshold

for seq_id in preds["sequence_id"].unique():
    chrom, gene_start = parse_header(seq_id)
    seq_preds = preds.filter(pl.col("sequence_id") == seq_id)

    for row in seq_preds.iter_rows(named=True):
        abs_pos = gene_start + row["position"]
        is_donor_pred = row["donor_prob"] > threshold
        is_acc_pred = row["acceptor_prob"] > threshold

        donor_true = (chrom, abs_pos, "donor") in truth_set
        acc_true = (chrom, abs_pos, "acceptor") in truth_set

        if is_donor_pred and donor_true:
            tp += 1
        elif is_acc_pred and acc_true:
            tp += 1
        elif is_donor_pred or is_acc_pred:
            fp += 1
        elif donor_true or acc_true:
            fn += 1

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
print(f"TP={tp}, FP={fp}, FN={fn}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}")
```

---

## What to Expect

### Model Behavior on Custom Builds

The M1-S model was trained on GRCh38/MANE.  On a different build:

- **Strong canonical sites** (GT-AG with conserved context): should be
  predicted reliably — splice motifs are sequence-level features that
  transfer across builds
- **Base score quality degrades**: FASTA mode uses uniform 1/3 prior instead
  of precomputed OpenSpliceAI scores.  The meta-layer relies more heavily
  on the DNA sequence stream
- **Multimodal features absent**: conservation, epigenetic, junction, RBP
  features are all zeroed out in FASTA mode.  These normally contribute
  significant signal for weak/alternative sites

### Improving Predictions on Custom Builds

For higher-quality predictions on a custom build, consider:

1. **Run the base model** (OpenSpliceAI) on the custom FASTA to get real
   base scores instead of uniform prior
2. **Generate conservation features** if PhyloP/PhastCons tracks exist
   for the build (bigWig files)
3. **Map to GRCh38** for the full multimodal feature stack — this is the
   Tier 2 alignment approach described in
   `examples/agentic_layer/docs/alignment_validation_tool.md`

---

## Connection to M3

The M3 model variant is designed for **novel splice site discovery** —
predicting splice sites that exist in biology but aren't in any annotation.
For M3 evaluation:

1. Generate ground truth from a **conservative** annotation (MANE)
2. Run inference on sequences from a **comprehensive** source (full genome)
3. Predicted sites NOT in the conservative annotation are M3 candidates
4. Validate candidates using junction read support (GTEx, tissue-specific RNA-seq)

The same pipeline described here (generate labels → run inference → compare)
applies to M3, with the addition of junction evidence as the validation
signal.  See `examples/meta_layer/docs/evaluation_hierarchy.md` for the
full evidence tier framework.

---

## Related Documents

- [Ground truth for custom genomes](../../data_preparation/docs/ground_truth_custom_genomes.md) — `04_generate_ground_truth.py` usage
- [Evaluation hierarchy](evaluation_hierarchy.md) — evidence tiers for prediction validation
- [Alignment validation tool](../../agentic_layer/docs/alignment_validation_tool.md) — Tier 2 alignment-based validation
- [Model variants M1-M4](../../../docs/meta_layer/methods/00_model_variants_m1_m4.md) — M3 novel site discovery
