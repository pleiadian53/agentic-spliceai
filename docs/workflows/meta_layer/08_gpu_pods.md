# Stage 8 — Running on GPU Pods (Optional)

**Applies to:** Stages 3–6 at genome scale.

Everything in this series runs locally on a small gene subset, which is ideal for learning and
smoke-testing. But the genome-scale runs — feature engineering across all 24 chromosomes, cache
building, and training — are GPU-bound and better suited to a cloud GPU. This project uses **RunPod**
(provisioned via **SkyPilot**) for that. This page is the orchestration layer; the model commands are
unchanged from the earlier stages, just pointed at `--device cuda`.

!!! info "Setup first"
    Cloud setup (accounts, keys, SkyPilot) is covered in
    [RunPods Setup](../../getting_started/RUNPODS_SETUP.md). This page assumes a pod is provisioned and
    the reference data is staged on its persistent volume.

---

## The pattern

Genome-scale meta-layer jobs follow the same three-step shape, wrapped by the `ops_*.sh` runners in
[`examples/meta_layer/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/):

1. **Bootstrap** — `ops_bootstrap_pod.sh` idempotently links the working tree's `data/` and `output/`
   to the pod's **persistent volume** (so large base-score parquets and the bigWig cache survive pod
   restarts) and sanity-checks the required paths. The other runners source it.
2. **Run detached** — long jobs launch under `nohup … &` with unbuffered Python (`python -u`) so they
   survive an SSH disconnect and stream a log you can tail.
3. **Collect** — results land in `output/meta_layer/…` on the volume; pull them back for
   [reporting](07_reporting.md).

## The runners

| Script | Wraps | Produces |
|--------|-------|----------|
| `ops_train_m1s_pod.sh` | `07 --mode m1 --device cuda --use-shards` (epochs 50, samples/epoch 100k, patience 10) | M1-S checkpoint |
| `ops_train_m2s_pod.sh` | `07` on Ensembl labels (`--annotation-source ensembl --base-scores-dir <ensembl>`) | M2-S checkpoint |
| `ops_eval_m1s_pod.sh` | `08 --build-cache --device cuda` | `eval_results.json` |
| `ops_eval_alt_sites_pod.sh` | `09` — parameterized `{m1s\|m2s} {ensembl\|gencode}` | `m2a` / `m2b` results |
| `ops_ablation_m1s_pod.sh`, `ops_ablation_m2s_pod.sh` | `08 --zero-channels …` looped over modality groups | `eval_ablation_*.json` |

!!! warning "Always `--device cuda` on a pod"
    The runners set it, but if you invoke the training/eval scripts directly on a pod, pass
    `--device cuda` explicitly — defaulting to CPU on a GPU box silently wastes the whole node.

---

## Data staging

The volume holds the artifacts that are expensive to regenerate: the base-score `precomputed/`
parquets ([Stage 2](02_base_scoring.md)), the `analysis_sequences/` feature parquets
([Stage 3](03_feature_engineering.md)), and the **bigWig cache** that the dense-channel extractor reads
during training. Because these persist across pods, a re-run reuses them and skips straight to
training. The top-level `ops/` package (`provision_cluster.py`, `stage_data.py`, `run_pipeline.py`) is
the newer, self-contained tooling for provisioning a cluster and staging this data.

## Teardown

Pods bill while they run. Tear the cluster down as soon as a job's outputs are safely on the volume (or
pulled back):

```bash
sky down <cluster-name> -y
```

---

→ Back to the **[series overview](README.md)**, or on to
[reporting & promotion](07_reporting.md) for what to do with the artifacts you just produced.
