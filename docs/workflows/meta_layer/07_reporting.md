# Stage 7 — Reporting & Promotion

**Pipeline position:** `evaluation → ` **this stage** (the end of the loop)

The final stage turns raw evaluation output into something durable: cross-checked numbers, a
human-readable results write-up, provenance metadata, and — for a model that earns it — **promotion**
to canonical status so the rest of the system (inference, the Bio Lab UI) can find it by name.

---

## 1. Machine-readable results

Each evaluation run leaves JSON in the model's own directory:

| File | From | Contents |
|------|------|----------|
| `best_metrics.json` | training ([Stage 4](04_training_m1s.md)) | best-epoch validation metrics |
| `eval_results.json` | `08` | held-out meta-vs-base metrics |
| `m2a_eval_results.json` | `09` | M2-S overall + alternative-site metrics |
| `eval_ablation_*.json`, `eval_results_calibrated.json` | `08` diagnostics | ablation / calibration studies |

These are the source of truth. Everything below is derived from them.

## 2. Cross-check the numbers

Before quoting a metric in a write-up, reconcile it with the annotation statistics using
[`examples/meta_layer/10_verify_evaluation_stats.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/10_verify_evaluation_stats.py):

```bash
python examples/meta_layer/10_verify_evaluation_stats.py --section all
```

It recomputes gene/site counts and the Ensembl `\` MANE and GENCODE `\` MANE set differences, then
reprints the model metrics straight from the result JSONs so the tables in the results docs can't drift
from the artifacts. It writes nothing — it's a read-and-reprint guard.

## 3. Human-readable roll-ups

The curated write-ups live in
[`examples/meta_layer/results/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/results/)
as checked-in Markdown — e.g. `m1s_ablation_study.md`, `m2s_ensembl_trained_results.md`,
`alternative_site_evaluation_results.md`. A new model's results doc should state the base-vs-meta
comparison at a matched-recall / F1-optimal threshold plus PR-AUC and top-k — the
[reporting norms from Stage 6](06_evaluation.md) — and link the exact JSON it
was computed from.

## 4. Provenance

Every model directory carries a `MANIFEST.yaml` recording how it was produced:

```yaml
status: active
produced_by: 07_train_sequence_model.py --mode m1
referenced_by: settings.yaml meta_models.m1s_v4_cleanannot
```

Directory-level provenance rolls up into `output/REGISTRY.md`. Superseded models
(`m1s_v2_logit_blend/`, `m2s_v2/`, `m2s_v3_baseline_repro/`) are kept, not deleted, with their status
marked in the manifest — so a comparison can always be reproduced. The full convention is in
[Output & Artifact Management](../../system_design/output_management.md).

---

## 5. Promotion — making a model canonical

A trained checkpoint in `output/meta_layer/…` is not "the" M1-S until it is **promoted** in
[`config/settings.yaml`](https://github.com/pleiadian53/agentic-spliceai/blob/main/src/agentic_spliceai/splice_engine/config/settings.yaml).
The `meta_models:` block is a pointer registry:

```yaml
meta_models:
  m1s_v4_cleanannot:
    name: "M1-S (canonical)"
    variant: "M1-S"
    dir: "output/meta_layer/m1s_v4_cleanannot"
    base_model: "openspliceai"
  m2s_v4_cleanannot:
    name: "M2-S (alternative)"
    variant: "M2-S"
    dir: "output/meta_layer/m2s_v4_cleanannot"
    base_model: "openspliceai"
```

This block — and *only* this block — decides which directory is canonical. It carries no architecture
or hyperparameters (those live in the checkpoint's `config.pt`); it is purely "which trained bundle is
the official M1-S / M2-S." It is consumed by `model_resources.py`
(`list_available_meta_models()` / `get_meta_model_config()`), which is how the inference path and the
Bio Lab UI discover a meta model by name.

!!! success "The loop is closed"
    Once promoted, the model is reachable through the standard resource resolver — the same way every
    other stage in this series resolved its inputs. A new, better M1-S is shipped by pointing this
    block at its directory, no code change required.

---

## Where to go next

- Run the whole thing at genome scale on a GPU: **[Stage 8 — GPU Pods](08_gpu_pods.md)**.
- Understand the model internals: [Meta-Layer Architecture](../../meta_layer/ARCHITECTURE.md).
- The novel-site (M3) and perturbation (M4) variants reuse Stages 1–3 and swap the training/eval
  scripts — see the M3/M4 design notes under `examples/meta_layer/docs/`.
