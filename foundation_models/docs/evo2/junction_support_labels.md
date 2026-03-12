# Junction-Supported Exon Labels for Evo2 Training

**Purpose**: Explain why and how to use RNA-seq-backed splice junction support
counts to produce higher-confidence per-nucleotide exon/intron labels for
training the Evo2 exon classifier.

**Audience**: Developers and researchers building the Evo2 exon prediction pipeline.

**Status**: Design — ready to implement

---

## Background

### What are "exon labels"?

When training the exon classifier on top of Evo2 embeddings, we need a binary
label for each nucleotide in a gene sequence: **1 = exon**, **0 = intron**.

The simplest approach derives these labels directly from the GTF annotation:

```
gene_sequence_*.parquet  +  splice_sites_enhanced.tsv
        ↓
build_exon_labels(gene_id, ...)   ← implemented in utils/chunking.py
        ↓
per-nucleotide [0, 1, 1, 1, 0, 0, ...] array
```

This works, but it has a subtle weakness: GTF annotations include **all
annotated exons across all transcripts of a gene**, many of which are low-
confidence or tissue-specific.  For a classifier that we want to be useful for
*adaptive* splice site prediction (i.e. detecting which exons are actually
active), annotation-only labels are too permissive.

### What is junction support?

A **splice junction** is a (donor_position, acceptor_position) pair
representing one intron-removal event.  In RNA-seq workflows, junctions are
detected by split-read aligners (e.g. STAR, HISAT2) and each junction receives
a **read support count** — the number of RNA-seq reads that span it.

A junction with high read support is strong evidence that the corresponding
exon boundaries are real and expressed in the tissue of interest.

### Why use junction support for labels?

| Label source | Pros | Cons |
|---|---|---|
| GTF annotation only | Always available; no RNA-seq needed | Includes unannotated / theoretical exons; no tissue specificity |
| Junction support | Reflects actual splicing activity; tissue-specific; filters noise | Requires RNA-seq data; coverage-dependent |
| Combined (this doc) | Best of both: GTF as skeleton, junction support as confidence | Slightly more complex pipeline |

The combined strategy is also what we need downstream: the goal is to predict
**novel isoforms** not yet in the GTF.  A classifier trained purely on GTF
labels will never generalise to unannotated splice events.

---

## Data Sources Already Available

From the existing `data/` infrastructure:

| File | Location | Contents |
|---|---|---|
| `splice_sites_enhanced.tsv` | `data/{source}/{build}/` | Annotated donor/acceptor positions with `exon_rank`, `transcript_id`, `gene_id` |
| `junctions.tsv` | `data/ensembl/GRCh38/` | Junction-level annotations (when RNA-seq data has been processed) |
| `gene_sequence_*.parquet` | `data/ensembl/GRCh37/` | Per-gene full sequences, one row per gene |

The `junctions.tsv` format (from meta-spliceai processing) typically contains:

```
chrom  start  end  strand  support_count  gene_id  transcript_id  ...
```

Where `start` is the donor position (last base of exon) and `end` is the
acceptor position (first base of next exon), with `support_count` being the
number of RNA-seq reads supporting this junction.

---

## Implementation

### Step 1 — Load junction support counts

```python
import pandas as pd
from pathlib import Path

def load_junctions(junctions_path: str | Path) -> pd.DataFrame:
    """Load junction support file and return a clean DataFrame.

    Expected columns:
        chrom, start (donor pos), end (acceptor pos), strand,
        support_count, gene_id
    """
    df = pd.read_csv(junctions_path, sep="\t")

    # Normalise column names (different tools use different names)
    rename_map = {
        "reads": "support_count",
        "score": "support_count",
        "count": "support_count",
        "block_count": "support_count",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"chrom", "start", "end", "strand", "gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"junctions file missing columns: {missing}")

    return df
```

### Step 2 — Build a confidence-weighted label

Instead of a hard binary label, we build a **float label in [0, 1]**
representing the probability that each nucleotide is part of a *supported*
exon.  This can be used as a soft target during training.

```python
import numpy as np

def build_junction_weighted_labels(
    gene_id: str,
    gene_start: int,
    gene_sequence_length: int,
    splice_sites_df: pd.DataFrame,
    junctions_df: pd.DataFrame,
    min_support: int = 3,
    soft: bool = True,
) -> np.ndarray:
    """Per-nucleotide exon label weighted by junction read support.

    Parameters
    ----------
    gene_id:
        Ensembl / MANE gene ID.
    gene_start:
        Genomic start of the gene (0-based, from gene_sequence parquet).
    gene_sequence_length:
        Length of the gene sequence string.
    splice_sites_df:
        Loaded splice_sites_enhanced.tsv DataFrame.
    junctions_df:
        Loaded junctions.tsv DataFrame filtered to this chromosome.
    min_support:
        Minimum junction read count to count a junction as supported.
        Junctions below this threshold are treated as absent.
    soft:
        If True, return float weights normalised to [0, 1].
        If False, return a hard binary label (1 if any supported junction
        uses this nucleotide as part of its exon span, 0 otherwise).

    Returns
    -------
    np.ndarray of shape [gene_sequence_length], dtype float32 if soft,
    uint8 if hard.
    """
    labels = np.zeros(gene_sequence_length, dtype=np.float32)

    # Filter junctions for this gene
    gene_juncs = junctions_df[
        (junctions_df["gene_id"] == gene_id) &
        (junctions_df["support_count"] >= min_support)
    ].copy()

    # Filter annotation sites for this gene
    gene_sites = splice_sites_df[splice_sites_df["gene_id"] == gene_id].copy()

    if gene_sites.empty:
        return labels if soft else labels.astype(np.uint8)

    rank_col = "exon_rank" if "exon_rank" in gene_sites.columns else "exon_number"

    for tx_id, tx_sites in gene_sites.groupby("transcript_id"):
        tx_sites = tx_sites.sort_values(rank_col)
        donors = sorted(tx_sites[tx_sites["site_type"] == "donor"]["position"].tolist())
        acceptors = sorted(tx_sites[tx_sites["site_type"] == "acceptor"]["position"].tolist())
        strand = tx_sites["strand"].iloc[0]

        # Build annotated exon spans
        spans: list[tuple[int, int, float]] = []  # (start, end, weight)

        pairs = (
            zip(acceptors, donors) if strand == "+"
            else zip(donors, acceptors)
        )

        for acc, don in pairs:
            if acc > don:
                continue

            # Look up junction support for this intron
            # The intron is (don + 1, acc - 1) in genomic coords
            match = gene_juncs[
                (gene_juncs["start"] == don) &
                (gene_juncs["end"] == acc)
            ]

            if not gene_juncs.empty and len(match) == 0:
                # Junction annotation exists but no read support → down-weight
                weight = 0.1 if soft else 0
            elif len(match) > 0:
                # Supported junction → weight proportional to log(support)
                support = match["support_count"].iloc[0]
                weight = float(np.log1p(support) / np.log1p(gene_juncs["support_count"].max()))
            else:
                # No junction data at all → fall back to annotation weight
                weight = 0.5

            spans.append((acc, don, weight))

        for gstart, gend, weight in spans:
            rel_s = max(0, gstart - gene_start)
            rel_e = min(gene_sequence_length, gend - gene_start + 1)
            if rel_s < rel_e:
                labels[rel_s:rel_e] = np.maximum(labels[rel_s:rel_e], weight)

    if not soft:
        return (labels >= 0.5).astype(np.uint8)
    return labels
```

### Step 3 — Integrate into the training data pipeline

```python
from foundation_models.utils.chunking import (
    load_gene_sequences_parquet,
    make_chunks_for_gene,
    generate_labeled_windows,
)
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/ensembl/GRCh37")

def build_training_dataset(
    chromosomes: list[str] = ["1", "2", "3"],
    window_size: int = 4096,
    step_size: int = 2048,
    min_support: int = 3,
    use_junction_weights: bool = True,
):
    """Build a list of LabeledWindow objects for Evo2 exon classifier training.

    Yields windows across all genes on the specified chromosomes.
    """
    splice_sites = pd.read_csv(DATA_ROOT / "splice_sites_enhanced.tsv", sep="\t")
    junctions_df = None
    if use_junction_weights and (DATA_ROOT / "junctions.tsv").exists():
        junctions_df = pd.read_csv(DATA_ROOT / "junctions.tsv", sep="\t")

    for chrom in chromosomes:
        parquet_path = DATA_ROOT / f"gene_sequence_{chrom}.parquet"
        if not parquet_path.exists():
            print(f"  Skipping chr{chrom}: parquet not found")
            continue

        genes_df = load_gene_sequences_parquet(parquet_path)
        print(f"  chr{chrom}: {len(genes_df)} genes")

        for _, row in genes_df.iterrows():
            gene_id = row["gene_id"]
            gene_start = int(row["start"])
            seq_len = len(row["sequence"])

            if use_junction_weights and junctions_df is not None:
                labels = build_junction_weighted_labels(
                    gene_id=gene_id,
                    gene_start=gene_start,
                    gene_sequence_length=seq_len,
                    splice_sites_df=splice_sites,
                    junctions_df=junctions_df,
                    min_support=min_support,
                    soft=True,
                )
            else:
                from foundation_models.utils.chunking import build_exon_labels
                labels = build_exon_labels(
                    gene_id=gene_id,
                    gene_start=gene_start,
                    gene_sequence_length=seq_len,
                    splice_sites_df=splice_sites,
                ).astype(np.float32)

            yield from generate_labeled_windows(
                gene_sequence=row["sequence"],
                exon_labels=labels,
                gene_id=gene_id,
                chrom=str(chrom),
                gene_genomic_start=gene_start,
                window_size=window_size,
                step_size=step_size,
            )
```

---

## Soft Labels vs Hard Labels — When to Use Which

| Scenario | Recommendation |
|---|---|
| Reproducing Evo2 paper exon classifier | Hard labels from GTF annotation (`build_exon_labels`) |
| Training on well-covered RNA-seq data | Soft labels (`soft=True`, `min_support=5`) |
| Transfer learning to a new tissue | Soft labels from that tissue's RNA-seq |
| Low-coverage or noisy RNA-seq | Hard labels with junction filtering (`soft=False`, `min_support=10`) |
| No RNA-seq available | Hard labels from annotation only |

---

## Training with Soft Labels

The `ExonClassifier.fit()` method uses `BCEWithLogitsLoss`.  To train with
soft float targets, no change to the classifier is needed — `BCEWithLogitsLoss`
accepts targets in [0, 1]:

```python
from foundation_models.evo2 import Evo2Embedder, ExonClassifier
import torch

embedder = Evo2Embedder(model_size="7b", quantize=True)

# Collect embeddings and labels for a small gene set
embeddings_list, labels_list = [], []

for window in build_training_dataset(chromosomes=["22"], window_size=4096):
    emb = embedder.encode(window.sequence)          # [seq_len, 4096]
    embeddings_list.append(emb)
    labels_list.append(torch.tensor(window.labels, dtype=torch.float32))

train_embeddings = torch.stack(embeddings_list)     # [N, seq_len, 4096]
train_labels = torch.stack(labels_list)             # [N, seq_len]

classifier = ExonClassifier(input_dim=4096, architecture="mlp")
classifier.fit(
    train_embeddings=train_embeddings,
    train_labels=train_labels,
    epochs=50,
)
```

---

## Junction Confidence Tiers

A practical way to think about label quality:

```
Tier 1 — High confidence  (support ≥ 10 reads)
    → weight = log(support) / log(max_support)  ≈ 0.7–1.0
    → safe for direct training

Tier 2 — Medium confidence  (3 ≤ support < 10)
    → weight ≈ 0.3–0.7
    → useful if balanced against Tier 1 samples

Tier 3 — GTF-only, no reads  (support = 0)
    → weight = 0.1 (soft) or 0 (hard)
    → exclude from training OR use as negative-confidence examples

Tier 4 — Novel junction (not in GTF, reads present)
    → weight = log(support) / log(max_support)
    → GOLD for novel isoform discovery — label as exon even if GTF says intron
```

Tier 4 is the most scientifically interesting — a novel junction with
strong read support is a candidate novel splice event.  To detect these,
we would need to cross-reference the junction positions against the GTF and
flag any junction whose donor or acceptor falls outside annotated exon
boundaries.

---

## Connection to Adaptive Splice Site Prediction

The exon classifier trained here feeds into the meta-layer:

```
Gene sequence (full gene or ±10 kb window)
        ↓ Evo2 encoder (frozen)
        ↓ Evo2 embeddings [seq_len, 4096]
        ↓ ExonClassifier
Exon probability per nucleotide  [seq_len]
        ↓  (used as additional feature)
Meta-layer (OpenSpliceAI scores + Evo2 exon probs + conservation + ...)
        ↓
Adaptive splice site predictions
        ↓
Novel isoform candidates
```

Key insight: splice site prediction models like OpenSpliceAI only see a
short window (501 bp) around each candidate site.  Evo2's exon probability
provides **global gene-level context** — if the exon classifier says position
X is unlikely to be exonic, the meta-layer can suppress a false-positive
donor prediction there.

---

## Next Steps

- [ ] Obtain/generate `junctions.tsv` for GRCh38 MANE data (STAR alignment of GTEx or ENCODE RNA-seq)
- [ ] Run `build_training_dataset` on chr21/chr22 as a fast prototype
- [ ] Train and evaluate `ExonClassifier` (MLP vs CNN) — target AUROC ≥ 0.85
- [ ] Integrate exon probability as a meta-layer feature

---

**Last Updated**: February 22, 2026
**Status**: Design complete — awaiting RNA-seq junction data for MANE/GRCh38
