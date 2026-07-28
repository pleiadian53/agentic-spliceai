# Stage 4 — Training M1-S (Canonical)

**Pipeline position:** `features → ` **this stage** ` → evaluation → reporting`

This is where the sequence-level meta model is trained. **M1-S** is the *canonical* refiner: it learns
to sharpen OpenSpliceAI's per-nucleotide predictions at MANE splice sites using the multimodal
context from [Stage 3](03_feature_engineering.md). The trainer is a single script driven by two flags —
`--mode` (which variant) and `--arch` (which architecture).

!!! abstract "Inputs → Outputs"
    **Reads:** MANE `splice_sites_enhanced.tsv` (labels), `predictions_{chrom}.parquet` (base scores),
    reference FASTA, and the dense modality channels (built on the fly, cached per gene).
    **Writes:** `output/meta_layer/m1s_v4_cleanannot/` — `best.pt`, `final.pt`, `config.pt`,
    `best_metrics.json`, `train.log`, `gene_cache/`.

---

## The command

The trainer is
[`examples/meta_layer/07_train_sequence_model.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/07_train_sequence_model.py).
For M1-S, `--mode m1` selects the canonical variant and auto-selects MANE labels:

```bash
python examples/meta_layer/07_train_sequence_model.py \
    --mode m1 \
    --use-shards \
    --bigwig-cache <bigwig-cache-dir> \
    --output-dir output/meta_layer/m1s_v4_cleanannot
```

For a quick local check on a handful of genes before committing to a genome-scale run, add `--smoke`.
The full genome-scale run is GPU-bound — see the [GPU Pods runbook](08_gpu_pods.md).

---

## What the trainer does — two phases

`07` runs in two phases:

1. **Cache build.** For each gene it assembles the training tensors — the base-score stream, the dense
   multimodal channels (via `DenseFeatureExtractor`, pulling bigWig / junction / eCLIP values), and the
   per-position label array — and writes them to a disk-backed per-gene `.npz` cache under
   `gene_cache/{train,val}/`. With `--use-shards` these are packed into shards
   (`gene_cache/{train,val}_shards/`) for efficient loading. This phase is the expensive one; it is
   reused across re-training runs.
2. **Train.** It streams samples from the cache (`--samples-per-epoch`), trains the `MetaSpliceModel`,
   tracks validation macro PR-AUC, and checkpoints the best epoch.

### Labels { #labels }

The per-position labels are materialized here, from the [Stage 1](01_data_preparation.md) table, by
`build_splice_labels()`. It produces a `[gene_length]` array over each gene window: the internal
"chunking" convention (0 = none, 1 = acceptor, 2 = donor) is remapped to the **meta convention** the
model trains against — **0 = donor, 1 = acceptor, 2 = neither**. Every window position not listed as a
donor or acceptor in `splice_sites_enhanced.tsv` becomes "neither"; there is no separate negative
sampling. (The base-score evaluation and the `-P` line use the same labels.)

---

## Selecting the architecture

Two orthogonal choices define the model, and neither lives in a YAML:

| Flag | Default | Effect |
|------|---------|--------|
| `--mode {m1,m2,m3}` | `m1` | Variant + label source. `m1` → M1-S, MANE. |
| `--arch {v3,v4_xattn}` | `v3` | Neural architecture. `v3` = dilated-CNN with concat fusion (promoted); `v4_xattn` = cross-attention fusion (WIP, not yet promoted). |

The chosen architecture's config dataclass (`MetaSpliceConfig` for v3, `MetaSpliceXAttnConfig` for
v4_xattn) is built by the model factory and **pickled to `config.pt`** as the definitive record of how
the checkpoint was constructed. Evaluation reloads it from there.

!!! warning "`m1s_v4_cleanannot` is a data version, not `--arch v4_xattn`"
    The `v4_cleanannot` in the output directory name is an **experiment/data** tag (clean minus-strand
    annotation + neuronal-RBP union). The promoted M1-S is architecture **v3**. The two "v4"s are
    unrelated axes — see the [series overview](README.md#3-three-independent-config-levers-and-two-confusingly-named-v4s).

---

## Key hyperparameters

Sensible defaults are baked in; these are the ones you'll most often touch:

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 50 | Early stopping via `--patience` (pod runs use 10). |
| `--lr` | 1e-3 | |
| `--hidden-dim` | 32 | Frozen at 32 — wider hidden state does not help sparse splice signal. |
| `--activation` | gelu | |
| `--samples-per-epoch` | 50000 | Pod runs use 100000. |
| `--exclude-channels` | — | Drop specific dense channels (channel ablation). |
| `--use-shards` | off | Pack the `.npz` cache into shards; recommended for genome-scale. |
| `--device` | auto | `cuda` on pods, `mps`/`cpu` locally. Batch/accum auto-set (cuda 16×1, cpu/mps 4×4). |

---

## Artifacts produced

The output directory is a self-contained model bundle:

| File | Contents |
|------|----------|
| `best.pt` | Best-epoch `state_dict` (the checkpoint used for eval and promotion). |
| `final.pt` | Last-epoch `state_dict`. |
| `config.pt` | Pickled architecture config (`MetaSpliceConfig`) — the architecture record. |
| `best_metrics.json` | `{epoch, loss, accuracy, pr_aucs{donor,acceptor,neither}, macro_pr_auc}` on validation. |
| `train.log` | Full tee'd training log. |
| `gene_cache/` | Disk-backed `.npz` cache (+ shards) — reused across runs. |
| `MANIFEST.yaml` | Provenance (added at promotion, [Stage 6](07_reporting.md)). |

### Reading `best_metrics.json`

`macro_pr_auc` is the headline: the mean of the donor/acceptor/neither PR-AUCs at the best validation
epoch. The promoted M1-S reaches a validation macro PR-AUC of **0.998** (epoch 8) — a strong canonical
refiner. This is a **validation** figure; the held-out **test** comparison against the base model (the
number you report) comes from [Stage 6](06_evaluation.md), not from this file.

---

→ **Next: [Stage 5 — Training M2-S](05_training_m2s.md)**
