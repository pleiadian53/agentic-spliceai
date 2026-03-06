# Data Staging & Verification Guide

How to stage datasets to a RunPod Network Volume for GPU training pipelines,
verify the data is correctly placed, and troubleshoot common issues.

**Prerequisites**: SkyPilot configured with RunPod
([cloud storage setup](cloud-storage-setup.md)), a network volume created via
the RunPod dashboard or `sky volumes apply`.

---

## Overview

GPU training jobs often need large input datasets (reference genomes, embeddings,
model weights). Uploading these via SkyPilot `file_mounts` on every launch is
slow and wastes time. A **network volume** provides persistent storage that
survives pod teardowns — stage data once, reuse on every run.

```
                        Stage (one-time)
Local machine ──file_mounts──> Pod ──rsync──> Network Volume
                                                   │
                        Execute (every run)         │
                 Pod <──mount + symlink────────────-┘
```

---

## 1. Configure Your Data Paths

Data paths are defined in `gpu_config.yaml`:

```yaml
# foundation_models/configs/gpu_config.yaml

# Network volume settings
use_volume: true
volume_name: "My Volume"          # must match RunPod dashboard name exactly
volume_mount: "/runpod-volume"    # where SkyPilot mounts the volume on the pod
volume_data_dir: "/runpod-volume/data/my_dataset"   # where your data lives on the volume
```

The `volume_data_dir` is the directory on the volume where your dataset will be
staged and later read from. Choose a path that makes sense for your data:

```
/runpod-volume/data/grch38/          # genome reference
/runpod-volume/embeddings/evo2-7b/   # pre-extracted embeddings
/runpod-volume/models/checkpoints/   # model weights
```

The `stage_data()` function in `gpu_runner.py` reads from a **local source
directory** and copies to `volume_data_dir`. The local source path is currently
set to `data/mane/GRCh38` but can be customized by editing the `stage_data()`
function or the generated YAML before launching.

---

## 2. Stage Data to the Volume

```bash
python examples/foundation_models/05_run_pipeline.py --stage-data
```

This:
1. Launches a pod with both `file_mounts` (your local data) and the volume
2. Runs `rsync` from the uploaded data to the volume's `volume_data_dir`
3. Prints a directory listing and total size for verification
4. Tears down the pod (volume persists)

### What happens under the hood

```yaml
# Generated SkyPilot config (simplified)
volumes:
  /runpod-volume: "My Volume"         # mount the network volume
file_mounts:
  /tmp/upload-data: ./local/data/     # upload local data to pod
run: |
  rsync -av /tmp/upload-data/ /runpod-volume/data/my_dataset/
```

The data takes two hops: local → pod (via `file_mounts`) → volume (via `rsync`
in the `run:` block). After teardown, only the volume copy persists.

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

Once data is staged, use `--use-volume` to skip re-upload:

```bash
# With the pipeline runner
python examples/foundation_models/05_run_pipeline.py --execute --use-volume \
    -- python your_script.py --your-args

# Or set use_volume: true in gpu_config.yaml (then --use-volume is automatic)
```

This mounts the volume and symlinks `volume_data_dir` to the relative path your
scripts expect — no `file_mounts` upload needed.

---

## 5. Updating Staged Data

`rsync` is idempotent — re-running `--stage-data` only copies new or changed
files. To update your dataset after local changes:

```bash
# Re-stage (only transfers diffs)
python examples/foundation_models/05_run_pipeline.py --stage-data
```

To stage to a different path or from a different source, edit the `stage_data()`
function in `gpu_runner.py` or manually create a SkyPilot YAML:

```yaml
name: stage-custom-data
resources:
  accelerators: A40:1
  cloud: runpod
volumes:
  /runpod-volume: "My Volume"
file_mounts:
  /tmp/upload: ./path/to/local/data
run: |
  mkdir -p /runpod-volume/custom/path/
  rsync -av --progress /tmp/upload/ /runpod-volume/custom/path/
  echo "Contents:"
  ls -lh /runpod-volume/custom/path/
  du -sh /runpod-volume/custom/path/
```

```bash
sky launch stage-custom-data.yaml -y
# Verify, then tear down
sky down stage-custom-data -y
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

## Reference: Configuration Fields

| Field in `gpu_config.yaml` | Default | Description |
|---------------------------|---------|-------------|
| `use_volume` | `false` | Whether to mount the network volume |
| `volume_name` | `"AI lab extension"` | Volume name (must match RunPod dashboard) |
| `volume_mount` | `/runpod-volume` | Mount point on the pod |
| `volume_data_dir` | `/runpod-volume/data/mane/GRCh38` | Data directory on the volume |

These can be overridden per-run via CLI flags (`--use-volume`, `--no-volume`) or
by editing the config file. See `gpu_runner.py:InfraConfig` for the full list.
