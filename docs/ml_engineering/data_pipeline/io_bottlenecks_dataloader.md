# I/O Bottlenecks and DataLoader Tuning

## The Paradox

You just migrated your training workload from a local workstation to a cloud GPU cluster. The instances are up, the GPUs are spinning, and by every hardware metric you've significantly upgraded your compute. But training iteration speed is flat — or worse, slower than before.

This is one of the most common surprises when scaling to cloud infrastructure, and the diagnosis is almost never what it seems at first glance. The problem isn't the GPUs. You've essentially bought a fleet of Ferraris and are trying to fuel them through a garden hose.

---

## Why Local Training Felt Fast

On your local machine, training data sits on NVMe SSDs — sequential read speeds of 3–7 GB/s, with single-digit milliseconds of latency per access. The OS prefetches aggressively. DataLoader workers spawn on the same machine as the data. Everything is fast because everything is close.

The moment you move to the cloud, that implicit assumption breaks. Your data is almost certainly in **object storage** (S3, GCS), and object storage has a fundamentally different access model.

---

## The Three Ways Object Storage Kills Training Speed

### 1. Per-Request Latency on Small Files

Every `GET` request to S3 incurs ~10–30ms of first-byte latency *regardless of file size*. If your dataset is structured as millions of individual files — individual JPEG patches, per-gene FASTA shards, per-sample HDF5 files — you are paying that latency tax on every single read.

At 30ms per file, reading 1,000 files sequentially takes 30 seconds. Your GPU can process a batch in milliseconds. The math doesn't work.

### 2. Request Rate Throttling

The actual ceiling you hit isn't bandwidth — it's **request rate**. S3 throttles at ~5,500 GET requests per second per prefix. If your DataLoader is spawning workers that hammer S3 with individual small-file reads, you can saturate that limit before you saturate your NIC. Increasing network bandwidth doesn't help; you're not bandwidth-limited, you're request-rate-limited.

### 3. CPU-Side Preprocessing Choke

Even if you get data off S3 fast enough, your CPU workers have to unpack it: decompress archives, decode images, reverse-complement DNA sequences, extract features from raw VCF records. If this preprocessing is slower than the GPU's forward+backward cycle, the GPU sits idle waiting for the next batch. You're compute-bound locally, but the CPU becomes the bottleneck in the cloud because the preprocessing now has to happen *over the network* before it can start.

---

## Diagnosing the Bottleneck

Before touching any configuration, confirm what you're actually dealing with.

**Step 1: Check GPU utilization, not wall-clock time.**

```bash
nvidia-smi dmon -s u
```

If `sm %` is persistently 20–40% with `mem %` near zero, your GPUs are idle — waiting for data, not computing. High wall-clock time with low SM utilization is the I/O starvation signature.

**Step 2: Run the synthetic data test.**

```python
# Replace your DataLoader with random tensors
for _ in range(num_batches):
    batch = torch.randn(batch_size, *input_shape, device='cuda')
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

If GPU utilization jumps to ~100% with synthetic data, the bottleneck is definitively in your data pipeline.

**Step 3: Check disk/network from the host.**

```bash
iostat -xz 1    # disk utilization
dstat -cdnm     # CPU, disk, network, memory simultaneously
```

---

## The Fix Stack

### Fix 1: Serialize into Large Sequential Formats

The root cause of the small-file problem is that object storage is optimized for large sequential reads, not random small-file access. Pack your individual files into large contiguous shards:

**WebDataset (`.tar` shards):** Standard for image/text datasets. Each shard is a tarball of samples, streamed sequentially. S3 can deliver large sequential reads at near-full bandwidth.

**HDF5 shards:** For structured numerical data (embeddings, genomic windows), pack many samples into chunked HDF5 files. This is what `agentic-spliceai` does — the `repack_into_shards()` function packs per-gene embedding windows into large `train_shard_000.h5` files, achieving 10–50x faster I/O compared to reading millions of individual per-gene HDF5 files.

The key property: a single sequential read of a 1GB shard file incurs *one* S3 request with 30ms latency, not millions.

### Fix 2: Stage Data to Local NVMe Before Training

Don't train directly from S3. At job startup, sync data to the instance-local NVMe drive:

```bash
# In your job launch script, before starting training:
aws s3 sync s3://your-bucket/data/shards/ /mnt/nvme/shards/ --quiet
```

On AWS p3/p4 instances, local NVMe (`/dev/nvme1n1`) delivers sequential reads at 3–7 GB/s with microsecond latency — comparable to your local workstation. On RunPod or Lambda, the network volume similarly serves as the staging area. `agentic-spliceai`'s `ops_stage_data.py` handles exactly this: it rsyncs local data to the cluster's network volume before any training process starts.

### Fix 3: Tune DataLoader Prefetching

Once your data is local, squeeze out the remaining pipeline efficiency with DataLoader configuration:

```python
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,              # workers prep batches in parallel with GPU forward/backward
    pin_memory=True,            # CPU→GPU copies use DMA (avoids extra kernel copy)
    persistent_workers=True,   # workers survive between epochs (no respawn overhead)
    prefetch_factor=2,          # each worker pre-loads 2 batches ahead
)
```

**`num_workers`** is the most impactful knob. Start at 4, scale up, watch sm% in `nvidia-smi dmon`. When adding more workers stops improving SM utilization, you've found the sweet spot.

**`pin_memory=True`** enables *non-blocking* CPU→GPU transfers via DMA. Without it, the GPU stalls during each host-to-device copy. With it, the copy and the GPU compute can overlap via CUDA streams.

**`persistent_workers=True`** avoids the overhead of killing and respawning worker processes at the end of each epoch. For large datasets with slow startup, this matters.

---

## The HDF5 Fork-Safety Gotcha

One specific wrinkle when using HDF5 as your shard format: h5py file handles are **not safe to share across forked processes**. If you use the default `fork` multiprocessing context with `num_workers > 0`, you'll get silent data corruption or deadlocks.

The `agentic-spliceai` codebase handles this in two ways:

**`07_genome_scale_splice_predictor.py` (base version):** Falls back to `num_workers=0` since the dataset uses an LRU file handle cache that's not fork-safe. This sidesteps the problem but foregoes parallel prefetching.

```python
# DataLoaders — num_workers=0 because HDF5 file handles are not
# fork-safe; the LRU cache already makes single-worker I/O fast.
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available(),
)
```

**`07a_direct_shard_splice_predictor.py` (upgraded version):** Uses `spawn` instead of `fork`, which is safe because worker processes initialize their own h5py handles from scratch (no shared state from the parent process). This enables `num_workers=2` for true prefetch overlap:

```python
# ShardedWindowDataset is fork-safe (lazy per-PID handles),
# so num_workers=2 allows prefetching to overlap with GPU compute.
_nw = 2 if (use_shards and cuda_available) else 0
_mp_ctx = "spawn" if _nw > 0 else None

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=_nw, pin_memory=cuda_available,
    persistent_workers=(_nw > 0),
    multiprocessing_context=_mp_ctx,
)
```

The `spawn` context is slightly slower to start (no copy-on-write optimization), but it's safe after Polars/h5py thread pools have been initialized in the parent process — `fork` after that point can deadlock.

---

## How to Think About the Full Pipeline

Training throughput is determined by the slowest stage in this chain:

```
Disk/S3 → CPU decode/augment → RAM → PCIe → VRAM → SM (forward + backward)
```

Each optimization in this guide attacks a different stage:

| Fix | Stage addressed |
|---|---|
| Pack into large shards | Disk/S3 → CPU (eliminate per-request latency) |
| Stage to NVMe | Disk/S3 (eliminate network overhead entirely) |
| `num_workers > 0` | CPU decode → RAM (parallelize with GPU compute) |
| `pin_memory=True` | RAM → PCIe → VRAM (async DMA transfer) |
| `persistent_workers=True` | Worker respawn overhead between epochs |
| `prefetch_factor` | RAM buffering depth ahead of current batch |

Apply them in order — staging to NVMe and sharding typically give the biggest gains; DataLoader tuning gets you the last 20–30%.

---

## Key Takeaways

**The bottleneck shift is predictable.** Moving from local to cloud almost always moves the bottleneck from compute to data ingestion. Local NVMe → S3 is not a transparent swap; the access model is completely different.

**Object storage penalizes small files twice.** Once for per-request latency, once for request-rate throttling. Neither is solved by paying for more bandwidth.

**Diagnose before you optimize.** `nvidia-smi dmon` + the synthetic data test will confirm whether you're actually I/O-starved before you spend time on any of the above.

**Sharding is the structural fix.** Everything else is tuning. Packing millions of small files into large sequential shards changes the access pattern from random-small to sequential-large, which is exactly what both object storage and local SSDs are optimized for.

**HDF5 + multiple DataLoader workers requires `spawn`.** The default `fork` context will deadlock or silently corrupt data if h5py handles are open in the parent process. Use `multiprocessing_context='spawn'` and design your Dataset to open handles lazily (per-worker, not in `__init__`).
