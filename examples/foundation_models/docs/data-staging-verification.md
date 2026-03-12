# Data Staging & Verification Guide

How to stage datasets and model weights to a RunPod Network Volume for GPU
training pipelines, verify the data is correctly placed, and troubleshoot
common issues.

**Prerequisites**: SkyPilot configured with RunPod
([cloud storage setup](cloud-storage-setup.md)), a network volume created via
the RunPod dashboard or `sky volumes apply`.

---

## Overview

GPU training jobs often need large input datasets (reference genomes, embeddings,
model weights). Uploading these via SkyPilot `file_mounts` on every launch is
slow and wastes time. A **network volume** provides persistent storage that
survives pod teardowns — stage data once, reuse on every run.

### Recommended Workflow

```
1. Provision    ops_provision_cluster.py [--stage-data]     Acquire pod + optional initial data
2. Stage        ops_stage_data.py <paths> [--weights MODEL] Upload additional data to running pod
3. Run          ops_run_pipeline.py --execute -- <command>   Execute your training/inference job
4. Tear down    ops_provision_cluster.py --down              Release resources when done
```

**Two ways to stage data:**

| Method | When to use |
|--------|-------------|
| `ops_provision_cluster.py --stage-data` | First time setup — stages data during provisioning (slower: SkyPilot `file_mounts` → volume) |
| **`ops_stage_data.py`** (recommended) | After pod is running — direct rsync, no SkyPilot overhead, supports arbitrary paths |

```
ops_stage_data.py:
Local machine ──rsync──> Pod volume     (one hop, fast)
                          │
               symlink ──-┘  (scripts find data at expected paths)

ops_provision_cluster.py --stage-data:
Local machine ──file_mounts──> Pod ──rsync──> Volume     (two hops, slower)
```

---

## 1. Stage Data with `ops_stage_data.py` (Recommended)

The staging script handles rsync + symlinks in one command, supporting arbitrary
datasets and model weights. It auto-detects the running cluster and preserves
the local directory structure on the volume.

### Stage reference data

```bash
# Stage GRCh37 reference data (Ensembl, for SpliceAI)
python examples/foundation_models/ops_stage_data.py data/ensembl/GRCh37

# Stage GRCh38 reference data (MANE, for OpenSpliceAI)
python examples/foundation_models/ops_stage_data.py data/mane/GRCh38

# Stage any custom dataset
python examples/foundation_models/ops_stage_data.py data/my_images/train
```

### Stage model weights

Use `--weights` with a model name — the script resolves the weights directory
via the resource manager (no need to remember paths):

```bash
# SpliceAI weights (TensorFlow .h5 files)
python examples/foundation_models/ops_stage_data.py --weights spliceai

# OpenSpliceAI weights (PyTorch .pt files)
python examples/foundation_models/ops_stage_data.py --weights openspliceai
```

### Stage everything for a model in one command

```bash
# SpliceAI: GRCh37 data + model weights
python examples/foundation_models/ops_stage_data.py \
    data/ensembl/GRCh37 --weights spliceai

# OpenSpliceAI: GRCh38 data + model weights
python examples/foundation_models/ops_stage_data.py \
    data/mane/GRCh38 --weights openspliceai
```

### Preview before transferring

```bash
python examples/foundation_models/ops_stage_data.py --dry-run \
    data/ensembl/GRCh37 --weights spliceai
```

### Target a specific cluster

If multiple clusters are running, specify which one:

```bash
python examples/foundation_models/ops_stage_data.py \
    --cluster sky-d5c6-pleiadian53 \
    data/mane/GRCh38
```

### How it works

For each path, `ops_stage_data.py`:
1. Rsyncs the local directory to the network volume (preserving structure)
2. Creates a symlink in `~/sky_workdir/` so scripts find data at expected paths

```
Local:   data/ensembl/GRCh37/
Volume:  /runpod-volume/data/ensembl/GRCh37/
Symlink: ~/sky_workdir/data/ensembl/GRCh37 -> /runpod-volume/data/ensembl/GRCh37
```

---

## 2. Stage Data During Provisioning (Alternative)

If this is your first time setting up, you can stage data as part of provisioning:

```bash
python examples/foundation_models/ops_provision_cluster.py \
    --stage-data --data-path ensembl/GRCh37 --stage-weights spliceai
```

This bundles data upload into the `sky launch` process via `file_mounts`. It's
convenient for initial setup but slower than `ops_stage_data.py` because:
- Data takes two hops: local → pod temp → volume
- The entire `sky launch` blocks until upload finishes (setup + run don't start)
- You can't stage additional datasets without re-launching

**Use `ops_stage_data.py` for subsequent data staging** — it rsyncs directly
to the running pod, is incremental, and supports multiple paths in one command.

---

## 3. Verify the Staged Data

### 3.1 Check the task logs

SkyPilot separates provisioning logs from task logs. The provisioning log
(`sky logs --provision <cluster>`) shows infrastructure setup. The **task log**
shows your actual data copy:

```bash
# Find the cluster name (SkyPilot may rename it)
sky status

# View the task log
sky logs <cluster-name>
```

Look for the rsync output followed by:

```
Data staged. Contents:
<file listing from ls -lh>
<total size from du -sh>
```

**Tip**: `sky logs --provision` only shows cluster setup (conda, Ray, packages).
Always use `sky logs` (without `--provision`) to see the staging results.

### 3.2 SSH and inspect directly

If the pod is still running:

```bash
ssh <cluster-name>

# List files with sizes
ls -lh /runpod-volume/data/my_dataset/

# Total disk usage (recurses into subdirectories)
du -sh /runpod-volume/data/my_dataset/

# Count files
find /runpod-volume/data/my_dataset/ -type f | wc -l
```

**Note on `ls` vs `du` discrepancy**: The `total` line at the top of `ls -lh`
output shows filesystem block allocation for the listed files only — it does
**not** include subdirectories. Use `du -sh` for the true recursive total. A
directory showing `total 4.0G` in `ls` but `10G` in `du` is normal if it
contains subdirectories with additional files.

### 3.3 Verify volume persistence after teardown

Network volumes persist independently of pods. After tearing down the staging
pod, confirm data survives by launching a quick check on any pod that mounts
the same volume:

```bash
# Run a one-off command on an existing cluster with the volume
sky exec <cluster-name> \
    "ls -lh /runpod-volume/data/my_dataset/ && du -sh /runpod-volume/data/my_dataset/"
```

### 3.4 Verify the symlink on execute

When you run with `--use-volume`, the GPU runner creates a symlink from the
volume path to where your scripts expect data (relative to the working
directory). The run output should show:

```
Using network volume data:
<first 5 files listed>
```

If this listing is missing or shows errors, the volume may not be mounted (see
Troubleshooting below).

To manually verify the symlink on a running pod:

```bash
ssh <cluster-name>
ls -la data/              # shows symlink → volume path
ls data/my_dataset/       # lists actual files via symlink
```

---

## 4. Running with Staged Data

Once data is staged, jobs use the network volume directly — no re-upload:

```bash
# With the pipeline runner (use_volume: true is the default in gpu_config.yaml)
python examples/foundation_models/ops_run_pipeline.py --execute \
    -- python your_script.py --your-args

# Or on an existing cluster
python examples/foundation_models/ops_run_pipeline.py --execute \
    --cluster sky-d5c6-pleiadian53 --no-teardown \
    -- python your_script.py --your-args
```

The runner mounts the volume and symlinks data to the expected relative paths.

---

## 5. Updating Staged Data

`rsync` is incremental — re-running only transfers new or changed files:

```bash
# Re-stage after local changes (only transfers diffs)
python examples/foundation_models/ops_stage_data.py data/ensembl/GRCh37
```

You can stage entirely new datasets at any time without reprovisioning:

```bash
# Add a new dataset to the running pod
python examples/foundation_models/ops_stage_data.py data/my_new_dataset

# Add weights for a new model
python examples/foundation_models/ops_stage_data.py --weights openspliceai
```

---

## Troubleshooting

### Cluster name mismatch on teardown

```
Cluster(s) not found: stage-data
```

SkyPilot may assign a different cluster name than the `name:` field in your YAML
config (e.g., `sky-f508-username` instead of `stage-data`). The GPU runner's
`_find_cluster_name()` handles this automatically, but if auto-teardown fails:

```bash
sky status                          # find the actual cluster name
sky down <actual-cluster-name> -y   # stop billing
```

### Volume is empty or path doesn't exist

1. **Check the volume mounted**: `ls /runpod-volume/` — if empty or missing,
   the volume didn't mount. Verify:
   - Volume name in `gpu_config.yaml` matches the RunPod dashboard exactly
     (case-sensitive)
   - Pod launched in the same datacenter zone as the volume
   - `sky volumes ls` confirms the volume exists

2. **Staging didn't complete**: Check `sky logs <cluster>` for rsync errors.
   Common causes:
   - Local source directory was missing when staging ran
   - Pod ran out of disk during the copy

3. **Re-stage**: Run `--stage-data` again. Rsync is idempotent.

### Wrong datacenter zone

Network volumes are tied to a specific RunPod zone. If the pod launches in a
different zone, the volume won't be available.

```bash
sky volumes ls        # check your volume's zone
```

Pin the zone in your config if needed:

```yaml
resources:
  accelerators: A40:1
  cloud: runpod
  region: CA          # must match volume's region
```

### Broken symlink

If scripts can't find data at the expected relative path:

```bash
ssh <cluster-name>
ls -la data/                                    # check symlink target
find /runpod-volume/ -maxdepth 3 -type d        # see what's actually on the volume
```

The symlink target must match `volume_data_dir` in your config.

### Provisioning log vs task log

If `sky logs <cluster>` only shows infrastructure setup (conda install, Ray
start, etc.), you're looking at the provisioning log. Use:

```bash
sky logs <cluster-name>            # task log (data copy, your script output)
sky logs --provision <cluster>     # provisioning log (infrastructure setup)
```

---

## Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `ops_provision_cluster.py` | Acquire a pod, install packages, optionally stage initial data |
| **`ops_stage_data.py`** | Stage data/weights to a running pod (recommended for ongoing use) |
| `ops_run_pipeline.py` | Execute a job on a running or new cluster |
| `ops_compute_check.py` | Verify GPU and compute environment |

### `ops_stage_data.py` flags

| Flag | Description |
|------|-------------|
| `<paths>` | Local directories to stage (positional, e.g., `data/ensembl/GRCh37`) |
| `--weights MODEL` | Stage model weights by name (repeatable) |
| `--cluster NAME` | Target cluster (auto-detects if one running) |
| `--volume-mount PATH` | Volume mount point (default: `/runpod-volume`) |
| `--data-prefix DIR` | Data prefix directory (default: `data`) |
| `--dry-run` | Preview without transferring |

### Configuration (`gpu_config.yaml`)

| Field | Default | Description |
|-------|---------|-------------|
| `use_volume` | `true` | Whether to mount the network volume |
| `volume_name` | `"AI lab extension"` | Volume name (must match RunPod dashboard) |
| `volume_mount` | `/runpod-volume` | Mount point on the pod |
| `data_prefix` | `"data"` | Top-level data directory |
| `data_path` | `"mane/GRCh38"` | Dataset subpath (compose as `{data_prefix}/{data_path}`) |

See `gpu_runner.py:InfraConfig` for the full list of fields.
