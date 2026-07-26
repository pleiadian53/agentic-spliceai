# ops/ — GPU Cluster Provisioning

Programmatic provisioning of RunPod GPU clusters via [SkyPilot](https://skypilot.co/).
Spin up a pod, stage reference data, and SSH in with one command — no manual pod
setup, no manual `rsync`, no stale SSH configs.

This is infrastructure, not model code. Treat it as a top-level utility that any
task under `examples/<topic>/` or `notebooks/<topic>/` can invoke when it needs a
GPU — meta-layer training/eval, foundation-model embedding extraction, or
genome-scale feature runs.

---

## When to Use This

| Scenario | Tool |
|----------|------|
| Small tasks that fit on the laptop (a few genes, MPS inference) | Run locally (CPU or MPS) |
| Anything realistic — genome-scale eval, meta-model training, Evo2 extraction | `ops/provision_cluster.py` |
| Bigger models / more VRAM | `ops/provision_cluster.py --gpu a100` or `h100` |

Check what you're working with first:

```bash
python ops/compute_check.py                    # device, VRAM, RAM, free disk
python ops/compute_check.py --fm-requirements  # + foundation-model VRAM reference table
```

---

## Prerequisites

1. **SkyPilot + RunPod credentials** installed once:
   ```bash
   pip install "skypilot[runpod]"
   sky check runpod   # should say "enabled"
   ```
2. **A RunPod network volume** named in [`configs/gpu_config.yaml`](configs/gpu_config.yaml)
   (default: `"AI lab extension"`). Create/rename via the RunPod dashboard.
3. **Data organized** as `data/<source>/<build>/` at the project root
   (e.g., `data/mane/GRCh38/`, `data/ensembl/GRCh37/`).

---

## Common Workflows

### First time — stage a dataset to the network volume

```bash
# Default: stages data/mane/GRCh38/
python ops/provision_cluster.py --stage-data

# Stage a different build
python ops/provision_cluster.py --stage-data --data-path ensembl/GRCh37
```

This uploads the local dataset to the network volume once. Subsequent provisions
mount the volume instantly — no re-upload. You can also push extra directories to
a *running* pod at any time with `ops/stage_data.py` (see below).

### Provision a workspace cluster

```bash
# Default: A40 on RunPod, agentic-spliceai installed (bio + conservation extras), volume mounted
python ops/provision_cluster.py

# Specific GPU
python ops/provision_cluster.py --gpu a100

# With an optional model dependency profile (see gpu_config.yaml)
python ops/provision_cluster.py --model evo2
```

The cluster **stays alive** until you explicitly tear it down. This is
intentional — iterative work on a warm pod is far faster than re-provisioning
for each run.

### SSH and run jobs

```bash
ssh aspliceai-workspace        # cluster name printed by provision_cluster.py
cd ~/sky_workdir               # your repo, synced by SkyPilot's workdir sync
python examples/meta_layer/13_evaluate_m3_novel.py --output /workspace/output/
```

Or launch a one-shot job from your laptop and pull results back automatically:

```bash
# Dry-run first (prints the generated SkyPilot config, no cost)
python ops/run_pipeline.py -- python your_script.py --args

# Execute on a fresh pod (auto-downloads output/, then tears down)
python ops/run_pipeline.py --execute -- python your_script.py --args

# Reuse the warm workspace pod, keep it alive
python ops/run_pipeline.py --execute --cluster aspliceai-workspace --no-teardown \
    -- python your_script.py --args

# Smoke-test the exact same command locally first (no pod, no cost)
python ops/run_pipeline.py --local-only -- python your_script.py --args
```

### Push more data to a running pod

```bash
python ops/stage_data.py data/ensembl/GRCh37             # auto-detects the running cluster
python ops/stage_data.py --dry-run data/models/spliceai  # preview only

# --weights takes an explicit path (generic) OR a model name (resolved in-repo)
python ops/stage_data.py --weights data/models/spliceai  # explicit path — works anywhere
python ops/stage_data.py --weights openspliceai          # by name (project resource manager)
```

### Tear down

```bash
python ops/provision_cluster.py --status    # list running clusters
python ops/provision_cluster.py --down      # interactive teardown
python ops/provision_cluster.py --down-all  # nuke everything
```

**Tear down when you're done.** Pods cost money whether or not you're using them.

---

## Configuration

Defaults live in [`configs/gpu_config.yaml`](configs/gpu_config.yaml).
CLI flags override the file. The most common edits:

| Field | What it does | When to edit |
|-------|--------------|--------------|
| `gpu` | GPU type (`a40`, `a100`, `h100`, ...) | Change for more/less VRAM |
| `data_path` | Dataset subpath `<source>/<build>` | Change for a different build |
| `default_model` | Extra pip deps beyond the base package | Set to `evo2`, `hyenadna`, etc. |
| `use_volume` | Mount the network volume | Set `false` for one-off jobs |

### GPU options and rough pricing (2026)

| Key | GPU | VRAM | ~$/hr |
|-----|-----|------|-------|
| `rtx4000ada` | RTX 4000 Ada | 20 GB | 0.26 |
| `rtxa5000` | RTX A5000 | 24 GB | 0.27 |
| `l4` | L4 | 24 GB | 0.39 |
| `a40` | A40 (default) | 48 GB | 0.39 |
| `rtx4090` | RTX 4090 | 24 GB | 0.59 |
| `rtx5090` | RTX 5090 | 32 GB | 0.89 |
| `a100` | A100 | 80 GB | 1.64 |
| `h100` | H100 | 80 GB | 3.29 |

Pricing is approximate and varies with RunPod availability. Check `sky show-gpus`
for live rates before launching expensive GPUs.

### Model dependency profiles

Most splice-engine eval and meta-layer work runs on the stock PyTorch image plus
the `agentic-spliceai` package (installed with the `bio` + `conservation`
extras), so `default_model: none`. Optional genomic foundation-model profiles are
available for when a task needs them:

- **evo2** — Evo2 7b/40b causal DNA language model
- **splicebert** — BERT pre-trained on vertebrate pre-mRNAs
- **alphagenome** — 1 Mb context, splice junctions + usage
- **hyenadna** — long-range genomic model
- **pangolin** — tissue-specific splice site strength
- **dnabert** — multi-species genome foundation model

Invoke with `--model <name>`. Extend profiles by editing `models:` in
[`configs/gpu_config.yaml`](configs/gpu_config.yaml). Foundation-model work that
needs the local `foundation_models/` sub-project can add
`pip install -e ./foundation_models` via the `extra_setup:` field (or
`--extra-setup` on `run_pipeline.py`).

---

## How It Works

1. `provision_cluster.py` reads `configs/gpu_config.yaml` + CLI overrides
2. Generates a SkyPilot YAML into `configs/skypilot/generated/` (git-ignored)
3. Invokes `sky launch` — provisions the pod, runs setup (`pip install -e ".[bio,conservation]"`,
   optional model deps), mounts the volume
4. Prints SSH instructions + a reminder to tear down

The underlying Python API lives in [`gpu_runner.py`](gpu_runner.py):
`GPU_SPECS`, `InfraConfig`, `build_skypilot_config`, `launch`, `stage_data`.
Import these directly (`from ops.gpu_runner import ...`) to script a pipeline —
see the module docstring.

---

## Package Contents

| File | Purpose |
|------|---------|
| `gpu_runner.py` | SkyPilot config builder + launcher (the Python API) |
| `provision_cluster.py` | Acquire a pod, keep it alive, `--status` / `--down` / `--stage-data` |
| `run_pipeline.py` | Launch one job on a running/new cluster (dry-run by default; `--local-only` runs it here instead) |
| `stage_data.py` | rsync local directories to a running pod's volume (`--weights` takes a path or a model name) |
| `compute_check.py` | Verify the local/pod GPU environment (`--fm-requirements` prints the FM VRAM table) |
| `configs/gpu_config.yaml` | Infrastructure defaults (GPU, cloud, volume, data path) |

---

## Relationship to `runpods.example/` and per-task scripts

The repo also carries a manual, dashboard-driven RunPod workflow under
`runpods.example/` (copy to `runpods/` and customize). That still works, but
`ops/` is the recommended path going forward because it is:

- **Programmatic** — no web-dashboard clicking, no hand-written configs
- **Repeatable** — same provision every time, same data staging every time
- **Volume-aware** — stages datasets once, reuses them across runs

`ops/` handles the generic *infrastructure* (get a pod, stage data — by path or
model name via `--weights` — run a command). Task-specific run scripts still live
next to the work they drive under `examples/<topic>/` — for example
`examples/foundation_models/ops_*.py` (Evo2 embedding extraction, plus richer
staging such as data-layout manifests) and `examples/meta_layer/ops_*_pod.sh`
(meta-model training/eval on a pod). Use `ops/` to get the pod; use those
per-task scripts to drive the specific job.
