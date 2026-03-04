# Remote Training Options: Modal, SkyPilot, and RunPods

**Status**: Reference
**Audience**: Developers needing GPU compute beyond your local computer
**Last Updated**: March 2026

---

## Problem

Training genomic foundation models (Evo2 7B/40B) and their downstream classifiers often requires dedicated GPU compute beyond what's available on a typical development machine. The manual RunPods workflow works but involves SSH configuration, pod environment setup, GitHub SSH keys, and rsync-based data/code transfer for every new pod instance.

**Goal**: Minimize the friction between "code works locally on small data" and "train on a real GPU" while keeping costs predictable.

---

## Development Principle: Local-First

Always develop and validate the full training workflow locally before offloading to remote GPU:

1. **Build locally** (small/synthetic data, MPS/CPU) until the pipeline runs end-to-end
2. **Test locally** with a real data subset to confirm correctness
3. **Only then** dispatch to remote GPU for production-scale training

Remote compute should be used only when the training pipeline is proven to work. This minimizes wasted GPU hours spent debugging serialization issues, data loading bugs, or training loop errors over SSH.

---

## Option 1: Modal (Recommended for Programmatic Remote Training)

**Website**: [modal.com](https://modal.com)

### How It Works

Modal lets you decorate a Python function with a GPU requirement and call it from your local machine. The function runs in a container on Modal's infrastructure with the specified GPU. No SSH, no pod setup, no environment scripts.

```python
import modal

app = modal.App("evo2-training")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers", "einops", "h5py", "peft", "scikit-learn"
)

@app.function(gpu="A100", image=image, timeout=3600)
def train_classifier(config: dict, embeddings_path: str) -> dict:
    """Runs on A100 in Modal's cloud."""
    import torch
    # ... training code identical to local version ...
    return model.state_dict()

# Called locally — weights returned automatically
weights = train_classifier.remote(config, "/data/embeddings.h5")
torch.save(weights, "output/classifier.pt")
```

### Setup (One-Time)

```bash
pip install modal
modal setup          # Authenticates via browser, stores token locally
```

### Data Transfer

- **Small data** (< 100 MB): Pass as function arguments (serialized automatically)
- **Large data** (genomic sequences, HDF5 embeddings): Use [Modal Volumes](https://modal.com/docs/guide/volumes) — persistent cloud storage mounted into containers

```python
volume = modal.Volume.from_name("genomic-data", create_if_missing=True)

@app.function(gpu="A100", volumes={"/data": volume})
def train_with_volume(config):
    # /data/ contains your uploaded HDF5 files, parquet, etc.
    embeddings = load_embeddings("/data/embeddings.h5")
    ...
```

Upload data once:
```bash
modal volume put genomic-data data/mane/GRCh38/ /mane/GRCh38/
```

### Pricing (as of March 2026)

Billed per-second of compute. No idle costs.

| GPU | Approx. $/hr | VRAM | Use Case |
|-----|-------------|------|----------|
| T4 | ~$0.60 | 16 GB | Quick tests, small classifiers |
| A10G | ~$1.10 | 24 GB | Medium models, hyperparameter search |
| A100 40GB | ~$2.10 | 40 GB | Evo2 7B embeddings, classifier training |
| A100 80GB | ~$2.50 | 80 GB | Evo2 40B (INT8), large-batch training |
| H100 | ~$3.95 | 80 GB | Fastest inference/training |

Pricing changes frequently. See [modal.com/pricing](https://modal.com/pricing) for current rates.

### Strengths

- Near-zero setup friction (pip install + browser auth)
- Same Python code locally and remotely (only the decorator/dispatch differs)
- Per-second billing, scales to zero (no idle pod costs)
- Container images are cached — cold start ~30-60s first time, warm after
- Volumes for persistent data (model weights, embeddings, training artifacts)
- Native Python SDK (no YAML, no shell scripts)
- Supports custom conda environments via `Image.from_conda`

### Limitations

- Cold-start latency (~30-60s if container hasn't been used recently)
- Slightly higher $/GPU-hour vs raw RunPod on-demand pods
- Data must be uploaded to Modal Volumes (can't mount local filesystem)
- Debugging remote failures requires reading Modal logs (no SSH into container)
- Modal is a managed service — less control than a raw VM/pod

---

## Option 2: SkyPilot (Automated Multi-Cloud Orchestration)

**Website**: [skypilot.co](https://skypilot.co) | [GitHub](https://github.com/skypilot-org/skypilot)

### How It Works

SkyPilot is an open-source orchestrator that provisions GPU VMs/pods across cloud providers (RunPod, Lambda, AWS, GCP, Azure, and more). It automates the manual steps: finding cheapest GPU, provisioning, environment setup, running your script, and teardown.

```yaml
# train.yaml
resources:
  accelerators: A100:1
  cloud: runpod         # or: lambda, aws, gcp, any

setup: |
  pip install -e .
  pip install -e ./foundation_models

run: |
  python foundation_models/examples/03_train_classifier.py \
    --config configs/evo2_7b_classifier.yaml
```

```bash
sky launch train.yaml                  # Provisions, sets up, runs
sky down train                         # Tears down when done
sky launch train.yaml --use-spot       # Use spot/preemptible for ~3x savings
```

### Setup (One-Time Per Provider)

```bash
pip install "skypilot[runpod]"         # or: skypilot[aws], skypilot[all]
sky check                              # Verifies cloud credentials
```

For RunPod: set `RUNPOD_API_KEY` environment variable.

### Data Transfer

SkyPilot uses rsync under the hood — your local `data/` directory can be synced to the VM automatically via `file_mounts` in the YAML:

```yaml
file_mounts:
  /data:
    source: ./data/mane/GRCh38/
    mode: COPY
```

Or use cloud storage (S3, GCS) for larger datasets.

### Pricing

SkyPilot itself is free (open-source). You pay the underlying cloud provider's rates:

| Provider | A40 48GB ($/hr) | A100 80GB ($/hr) | Spot Discount |
|----------|----------------|------------------|---------------|
| RunPod | ~$0.69 | ~$1.64 | N/A (on-demand only) |
| Lambda | ~$0.80 | ~$1.29 | N/A |
| AWS (p4d) | — | ~$3.10 | ~60-70% off |
| GCP (a2) | — | ~$3.67 | ~60-70% off |

SkyPilot can auto-select the cheapest available provider (`cloud: any`).

### Strengths

- Uses your existing RunPod account (and any other cloud accounts)
- Automatic price-shopping across providers
- Spot/preemptible instance support for significant cost savings
- SSH access to the VM (full debugging capability)
- File mounts sync local directories automatically
- Open-source, no vendor lock-in
- YAML-based — good for reproducible experiment configs
- [Official RunPod integration](https://docs.runpod.io/integrations/skypilot)

### Limitations

- Still VM/pod-based (boot + setup time: 2-5 minutes per launch)
- Requires cloud credentials configured per provider
- More infrastructure to manage than Modal (YAML configs, credential setup)
- No "call a Python function remotely" pattern (batch job model instead)
- Each launch is a full VM lifecycle (provision → setup → run → teardown)
- Less elegant for rapid iteration vs Modal's function-call pattern

---

## Option 3: RunPods (Current Approach — Manual Pods)

**Website**: [runpod.io](https://runpod.io)

### How It Works

Manually provision a GPU pod via the RunPod web UI or CLI, SSH in, configure the environment, rsync code and data, run training, rsync results back.

### Setup References

- First-time setup: [docs/getting_started/RUNPODS_SETUP.md](../getting_started/RUNPODS_SETUP.md)
- SSH manager scripts: `runpods.example/scripts/`
- Pod environment setup: `runpods.example/docs/POD_ENVIRONMENT_SETUP.md`
- Data transfer: `runpods.example/docs/RSYNC_QUICK_REFERENCE.md`
- GitHub SSH on pod: `runpods.example/docs/GITHUB_SSH_SETUP.md`

### Pricing

Lowest raw $/GPU-hour of the three options:

| GPU | On-Demand ($/hr) | Community Cloud ($/hr) |
|-----|------------------|----------------------|
| A40 48GB | ~$0.69 | ~$0.39 |
| A100 80GB | ~$1.64 | ~$1.04 |
| H100 80GB | ~$3.29 | ~$2.49 |

See [runpod.io/pricing](https://www.runpod.io/pricing) for current rates.

### Strengths

- Cheapest per-GPU-hour
- Full SSH access and control
- Persistent pods (keep running between sessions)
- Community cloud pricing for additional savings
- Already set up in this project (scripts, docs, environment files)

### Limitations

- Manual SSH config per pod instance
- GitHub SSH keys must be configured on each new pod
- Environment setup required per pod (conda create, pip install)
- Rsync for code/data transfer each session
- Idle pods cost money (must remember to stop/delete)
- No programmatic dispatch from local code

---

## Comparison Summary

| Feature | Modal | SkyPilot | RunPods (Manual) |
|---------|-------|----------|-----------------|
| **Setup friction** | Minimal (pip + auth) | Medium (credentials) | High (SSH, env, keys) |
| **Code pattern** | Python function call | YAML batch job | SSH + rsync |
| **Same code local/remote** | Yes (decorator only) | Mostly (YAML wraps script) | Separate workflow |
| **Data transfer** | Volumes (cloud) | rsync / cloud storage | rsync |
| **Debugging** | Logs only | SSH into VM | Full SSH |
| **Idle cost** | None (per-second) | None (teardown) | Yes (must stop pod) |
| **Raw GPU cost** | Higher (~$2-4/hr) | Varies (cheapest cloud) | Lowest (~$0.7-1.6/hr) |
| **Spot/preemptible** | No | Yes | No |
| **Vendor lock-in** | Modal only | Multi-cloud | RunPod only |

---

## Recommendation for This Project

### Near-Term (Current)

Continue using **RunPods manual workflow** for the initial Evo2 exon classifier experiments. The infrastructure is already set up, and the first priority is proving the training pipeline works end-to-end locally.

### Medium-Term (When Training Becomes Frequent)

Adopt **Modal** for the programmatic dispatch pattern. This maps directly to the small/medium/large tier strategy:

| Tier | Hardware | Method | When |
|------|----------|--------|------|
| Small | Local (MPS/CPU/CUDA) | Direct local call | Developing training logic |
| Medium | Modal T4/A10G | `train.remote(...)` | Validated pipeline, real data subset |
| Large | Modal A100/H100 | `train.remote(...)` | Production training, full genome |

### Integration with AdaptiveTrainer

The planned `AdaptiveTrainer` class (see `foundation_models/docs/training/deepspeed_training.md`) could be extended to support a `backend` parameter:

```python
trainer = AdaptiveTrainer(
    model=classifier,
    backend="auto",     # "local" | "modal" | "skypilot" | "auto"
)
# "auto" selects: local if small data + MPS/CUDA available,
#                 modal if large data or explicit GPU request
trainer.fit(train_data, val_data)
```

This keeps the training code identical regardless of where it runs.

### Long-Term (If Multi-Cloud Cost Optimization Matters)

Add **SkyPilot** as an option for spot-instance training jobs that can tolerate preemption and longer setup times, especially for large-scale sweeps across the cheapest available provider.

---

## Sources

- [Modal Pricing](https://modal.com/pricing)
- [Modal Volumes Documentation](https://modal.com/docs/guide/volumes)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [RunPod SkyPilot Integration](https://docs.runpod.io/integrations/skypilot)
- [SkyPilot GitHub](https://github.com/skypilot-org/skypilot)
- [Top Serverless GPU Clouds for 2026](https://www.runpod.io/articles/guides/top-serverless-gpu-clouds)
- [H100 Rental Price Comparison 2026](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
