# Stage 3 — Feature Engineering (Multimodal)

**Pipeline position:** `base scoring → ` **this stage** ` → training → eval`

This stage assembles the **multimodal evidence** the meta model learns from: the base model's own
scores plus eight other modalities (sequence context, conservation, chromatin, histone marks,
splice-junction reads, RBP binding, and positional/genomic context). The output is a per-chromosome
feature table, aligned position-by-position, that both model lines draw from.

!!! abstract "Inputs → Outputs"
    **Reads:** `predictions_{chrom}.parquet` ([Stage 2](02_base_scoring.md)) + streamed
    bigWig / junction / eCLIP sources, selected by a YAML profile.
    **Writes:** `data/<source>/<build>/openspliceai_eval/analysis_sequences/analysis_sequences_{chrom}.parquet`
    (9 active modalities; 116-column parquet, 106 feature columns + metadata) + `feature_summary.json`.

---

## The command

The production pipeline is
[`examples/features/06_multimodal_genome_workflow.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/features/06_multimodal_genome_workflow.py).
For M1-S / M2-S training data, use the **`full_stack`** profile (all 9 modalities):

```bash
python examples/features/06_multimodal_genome_workflow.py \
    --config examples/features/configs/full_stack.yaml \
    --chromosomes all \
    --resume
# → data/mane/GRCh38/openspliceai_eval/analysis_sequences/analysis_sequences_chr1.parquet … chrY.parquet
```

Run it once per base-scores directory: the MANE precomputed dir for M1-S, and the Ensembl precomputed
dir for M2-S (the workflow resolves the output `analysis_sequences/` dir next to the base scores it reads).

**Variants of the workflow:**

| Script | When to use |
|--------|-------------|
| `06_multimodal_genome_workflow.py` | Default — the main pipeline. |
| `06a_ephemeral_genome_workflow.py` | Disk-bounded pods: predict → featurize → delete raw predictions per chromosome (`--ephemeral`). |
| `04_genome_scale_workflow.py` | Legacy CLI-flag-driven predecessor (no YAML). |

**Useful flags on `06`:** `--chromosomes chr22` (single chrom for a smoke run), `--resume` (continue),
`--augment` (add missing modalities to existing parquets in place), `--refresh <modality>` (drop and
recompute one modality), `--dry-run`, `--memory-limit <GB>`.

---

## Config-driven modality selection

Which modalities are computed is decided entirely by the **YAML profile's `pipeline.modalities`
list** — adding or removing a modality is a config edit, not a code change. The loader instantiates
only the listed modalities from the pipeline registry and validates each name.

| Profile | Active modalities | Purpose |
|---------|-------------------|---------|
| `default.yaml` | base_scores, annotation, genomic (3) | Fast, no external data |
| **`full_stack.yaml`** | **all 9** (adds sequence, conservation, epigenetic, junction, rbp_eclip, chrom_access) | **M1-S / M2-S training data** |
| `isoform_discovery.yaml` | 8 (junction off, wider sampling window) | Weak/competing-site capture |
| `meta_m3_novel.yaml` | 6 (junction **excluded** — it is M3's prediction target) | M3 novel-site training |

The nine modalities and their approximate column counts: `base_scores` (43), `annotation` (3),
`sequence` (3), `genomic` (4), `conservation` (9), `epigenetic` (12), `junction` (12), `rbp_eclip`
(8), `chrom_access` (6). A commented `fm_embeddings` block is the optional 10th modality (foundation-
model scalars; see [Stage 3 addendum](#optional-foundation-model-scalars)). Every column is documented
in the [Feature Catalog](../../multimodal_feature_engineering/feature_catalog.md).

!!! note "Early sampling"
    `full_stack.yaml` enables **early sampling** — positions are subsampled (keep everything above a
    probability threshold plus a small background rate) *before* the expensive external-modality
    lookups, roughly a 100× speedup with no loss of splice signal. This is why the parquet is peaky,
    not dense.

---

## How the two model lines consume this stage

The same `analysis_sequences` parquets feed two different consumers — a distinction worth pinning down
(see also the [series overview](README.md#2-there-are-two-model-lines-that-share-the-feature-stack)):

- **Position-level (`-P`, XGBoost)** reads the parquet **rows directly** — one feature vector per
  position.
- **Sequence-level (`-S`, M1-S / M2-S)** does **not** train on the parquet rows. At training time
  ([Stage 4](04_training_m1s.md)) `DenseFeatureExtractor` rebuilds the modalities as **dense
  per-position channels** over each gene window and caches them per gene as `.npz`. The `-S` models
  see a stack of channels (base scores + the eight modality channels); M1-S and M2-S use all channels,
  while M3-S drops the junction channels (junction is its target).

So the parquet's role for the `-S` line is (a) to define **which positions are sampled** and (b) to
feed the optional foundation-model scalar step — not to be the training matrix itself.

---

## Verify before moving on

Two read-only checkers guard this stage:

```bash
# Integrity: row alignment, label/score consistency, schema drift, out-of-range values
python examples/features/verify_feature_alignment.py --chromosomes all

# Coverage: which chromosomes/modalities/columns are present, missing, or all-null
python examples/features/check_modality_completeness.py --chromosomes all --suggest
```

`check_modality_completeness.py --suggest` prints the exact `06 … --augment` / `--refresh` commands to
fill any gap it finds — useful when an external data source (e.g. a stale ENCODE bigWig URL) silently
produced an all-null column.

**Reference:** the full feature-generation how-to, config recipes, and per-modality data sources are in
[`examples/features/docs/feature-engineering-guide.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/features/docs/feature-engineering-guide.md).

---

## Optional: foundation-model scalars

To add the 10th modality (`fm_embeddings`), run
[`examples/features/07_streaming_fm_scalars.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/features/07_streaming_fm_scalars.py)
**after** `06` — it streams Evo2/SpliceBERT/HyenaDNA embeddings at the sampled positions and writes a
handful of scalar columns (PCA + norm + local gradient) without persisting the multi-GB raw
embeddings, then folds them back in via `06 --augment`. This is off by default for M1-S/M2-S.

---

→ **Next: [Stage 4 — Training M1-S](04_training_m1s.md)**
