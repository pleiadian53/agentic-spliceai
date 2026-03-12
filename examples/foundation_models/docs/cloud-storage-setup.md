# Cloud Storage for GPU Training Pipelines

How to configure persistent cloud storage so reference data (~10 GB FASTA + GTF)
doesn't need to be re-uploaded on every SkyPilot launch.

## Quick Summary

| Approach | Cost (10 GB) | Speed | Best For |
|----------|-------------|-------|----------|
| **Cloudflare R2** | Free (10 GB free tier, no egress) | Fast after first upload | Reference data, multi-cloud |
| RunPod Network Volume | $0.07/GB/month | Fastest (local network) | RunPod-only workflows |
| AWS S3 | ~$0.23/month + egress | Fast | If you already use AWS |
| Local rsync (current) | Free | Slow (~10 min per launch) | Quick experiments |

**Current setup**: RunPod Network Volume "AI lab extension" (150 GB, CA-MTL-1).
Use `--stage-data` once, then `--use-volume` for all future runs.

---

## Option 1: Cloudflare R2 (Recommended)

R2 is S3-compatible with zero egress fees. The free tier covers 10 GB storage,
1M Class A ops, and 10M Class B ops per month.

### One-Time Setup

#### 1. Get R2 API credentials

Go to **Cloudflare Dashboard > R2 > Manage API Tokens > Create API Token**.
Note your **Access Key ID**, **Secret Access Key**, and **Account ID**.

#### 2. Configure credentials for SkyPilot

```bash
pip install boto3

# Store R2 credentials (separate from AWS to avoid conflicts)
AWS_SHARED_CREDENTIALS_FILE=~/.cloudflare/r2.credentials \
  aws configure --profile r2
# Enter: Access Key ID, Secret Access Key, region=auto, output=json

# Store your Cloudflare Account ID
mkdir -p ~/.cloudflare
echo "YOUR_CLOUDFLARE_ACCOUNT_ID" > ~/.cloudflare/accountid
```

#### 3. Verify SkyPilot sees R2

```bash
sky check  # Should show: Cloudflare: enabled [compute, storage]
```

### Upload Reference Data (Once)

#### Method A: Let SkyPilot handle it

On first `sky launch` with this config, SkyPilot creates the R2 bucket and
uploads your data. Subsequent launches reuse the bucket (no re-upload):

```yaml
# extract_embeddings_a40.yaml
file_mounts:
  /workspace/data:
    name: agentic-spliceai-grch38       # globally unique bucket name
    source: ./data/mane/GRCh38          # local path (uploaded on first launch)
    store: r2
    persistent: true                    # keep bucket after cluster teardown
    mode: COPY                          # download to VM disk at launch
```

#### Method B: Upload manually, reference by URI

```bash
# Upload once (uses the R2 S3-compatible endpoint)
aws s3 cp --profile r2 \
  --endpoint-url https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com \
  ./data/mane/GRCh38/ \
  s3://agentic-spliceai-grch38/ \
  --recursive
```

Then reference the bucket directly in your YAML:

```yaml
file_mounts:
  /workspace/data:
    source: r2://agentic-spliceai-grch38
    mode: COPY
```

### Updated SkyPilot Config (with R2)

```yaml
# foundation_models/configs/skypilot/extract_embeddings_a40.yaml
name: extract-emb-7b

resources:
  accelerators: A40:1
  cloud: runpod

file_mounts:
  /workspace/data:
    name: agentic-spliceai-grch38
    source: ./data/mane/GRCh38
    store: r2
    persistent: true
    mode: COPY

setup: |
  pip install -e .
  pip install -e ./foundation_models
  pip install evo2

run: |
  mkdir -p /workspace/output
  python examples/foundation_models/03_embedding_extraction.py \
    --genes BRCA1 TP53 BRCA2 RB1 CFTR EGFR PTEN MLH1 MSH2 APC \
    --model evo2 --model-size 7b \
    --output /workspace/output/
```

First launch: ~5 min upload + normal setup. Subsequent launches: instant mount.

---

## Option 2: RunPod Network Volumes (Current Setup)

RunPod's native persistent storage. Faster I/O than cloud buckets but locked
to a specific datacenter zone. **This is what we use.**

Our volume: **AI lab extension** (150 GB, CA-MTL-1, ID: w92bineh2j)

### Import an Existing Volume (if created via RunPod dashboard)

```yaml
# /tmp/import-volume.yaml
name: AI lab extension
type: runpod-network-volume
infra: runpod/CA/CA-MTL-1
size: 150Gi
use_existing: true
```

```bash
sky volumes apply /tmp/import-volume.yaml
sky volumes ls   # verify it shows up
```

### Stage Reference Data

**Recommended**: Use the staging script after provisioning a cluster:

```bash
# 1. Provision a pod (no data staging — fast)
python examples/foundation_models/ops_provision_cluster.py --gpu rtxa5000

# 2. Stage data + weights to the running pod
python examples/foundation_models/ops_stage_data.py \
    data/ensembl/GRCh37 --weights spliceai

# Or for OpenSpliceAI:
python examples/foundation_models/ops_stage_data.py \
    data/mane/GRCh38 --weights openspliceai
```

This rsyncs directly to the pod (one hop) — faster than `file_mounts` staging.
See [data staging guide](data-staging-verification.md) for details.

**Alternative**: Stage during provisioning (slower, but all-in-one):

```bash
python examples/foundation_models/ops_provision_cluster.py \
    --stage-data --data-path ensembl/GRCh37 --stage-weights spliceai
```

### Use in Pipeline (Fast — No Upload)

```bash
# Run a job on the existing cluster (data already on volume)
python examples/foundation_models/ops_run_pipeline.py --execute \
    --cluster <cluster-name> --no-teardown \
    -- python your_script.py --your-args
```

### How It Works in YAML

```yaml
name: extract-emb-7b
workdir: .

resources:
  accelerators: A40:1
  cloud: runpod

# Mount the network volume (no file_mounts needed — data is already there)
volumes:
  /runpod-volume: AI lab extension

setup: |
  set -e
  pip install -e .
  pip install -e ./foundation_models
  pip install evo2

run: |
  set -e
  # Symlink volume data to the path the extraction script expects
  mkdir -p data/mane
  ln -sfn /runpod-volume/data/mane/GRCh38 data/mane/GRCh38
  python examples/foundation_models/03_embedding_extraction.py \
    --genes BRCA1 TP53 \
    --model evo2 --model-size 7b \
    --output /workspace/output/
```

**Cost**: $0.07/GB/month = ~$1.40/month for 20 GB used on a 150 GB volume.

---

## Storage Modes Reference

| Mode | Behavior | Best For |
|------|----------|----------|
| `COPY` | Downloads bucket to VM disk at launch. Writes stay local. | Reference data, model weights |
| `MOUNT` | Streams files on access via FUSE. Writes sync back to bucket. | Shared scratch, large datasets |
| `MOUNT_CACHED` | Local cache with async upload. Non-blocking writes. | Checkpoints during training |

### When to use each

```yaml
file_mounts:
  # Reference data: read-only, need fast I/O during training
  /ref_data:
    source: r2://agentic-spliceai-grch38
    mode: COPY

  # Checkpoints: written frequently, upload in background
  /checkpoints:
    name: agentic-spliceai-checkpoints
    store: s3                              # or r2
    mode: MOUNT_CACHED

  # Shared output: multiple workers need to see same files
  /shared_output:
    name: agentic-spliceai-output
    store: r2
    mode: MOUNT
```

---

## Managing Buckets

```bash
# List all SkyPilot-managed buckets
sky storage ls

# Delete a bucket (frees storage)
sky storage delete agentic-spliceai-grch38
```

---

## Cost Comparison

For 10 GB reference data accessed ~20 times/month:

| Provider | Storage | Egress (20 reads x 10 GB) | Monthly Total |
|----------|---------|---------------------------|---------------|
| Cloudflare R2 | **$0** (free tier) | **$0** | **$0** |
| RunPod Volume | $0.70 | $0 (local) | $0.70 |
| AWS S3 | $0.23 | $18.00 | $18.23 |
| GCS | $0.20 | $16.00 | $16.20 |

R2 is the clear winner for reference data that is written once and read many times.

---

## Migrating from Local Rsync

Current config (re-uploads ~10 GB every launch):
```yaml
file_mounts:
  /workspace/data: ./data/mane/GRCh38        # rsync every time
```

Migrated to R2 (uploads once, instant on subsequent launches):
```yaml
file_mounts:
  /workspace/data:
    name: agentic-spliceai-grch38
    source: ./data/mane/GRCh38
    store: r2
    persistent: true
    mode: COPY
```

The `source` field is only used on first launch (to populate the bucket).
After that, the bucket name alone is sufficient:

```yaml
file_mounts:
  /workspace/data:
    name: agentic-spliceai-grch38
    store: r2
    mode: COPY
```

---

## Troubleshooting

**`StorageNameError: Storage name must be specified`**
- You used the dict format (`source`/`mode`) without a `name` field.
  Add `name: my-bucket-name` or use the simple rsync format (`/path: ./local`).

**`StorageSourceError: paths cannot end with a slash`**
- Remove trailing `/` from source paths. Use `./data/mane/GRCh38` not `./data/mane/GRCh38/`.

**R2 not showing in `sky check`**
- Ensure `~/.cloudflare/r2.credentials` has `[r2]` profile and
  `~/.cloudflare/accountid` contains your Cloudflare account ID.
- Install boto3: `pip install boto3`.

**Slow MOUNT mode**
- FUSE mounts have higher latency than local disk. For training data that's
  read repeatedly, use `COPY` mode instead.
