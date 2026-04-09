# Stochastic Training Data: When Epochs Don't Mean What You Think

A guide to stochastic (on-the-fly) sampling vs. fixed-dataset training,
why each approach exists, and how the agentic-spliceai meta-layer uses
stochastic sampling while OpenSpliceAI uses fixed windows.

---

## 1. Two Approaches to Training Data

### Fixed dataset: iterate over everything once per epoch

The standard approach.  Before training, extract all examples into a
dataset.  Each epoch iterates over every example exactly once (shuffled).

```
Epoch 1:  [sample_42, sample_17, sample_3, ..., sample_99999]  (shuffled)
Epoch 2:  [sample_88, sample_5, sample_42, ..., sample_31]     (reshuffled)
Epoch 3:  [sample_12, sample_99999, sample_67, ...]            (reshuffled)
```

Every sample is seen once per epoch.  After N epochs, every sample has
been seen exactly N times.  The epoch boundary is well-defined: you've
processed all the data.

### Stochastic sampling: draw fresh random examples each step

No fixed dataset.  Each training step generates a fresh example by
randomly sampling from a large (often infinite) data space.

```
Epoch 1:  [rand_window(), rand_window(), ...] × samples_per_epoch
Epoch 2:  [rand_window(), rand_window(), ...] × samples_per_epoch
Epoch 3:  [rand_window(), rand_window(), ...]
```

Different random windows every call.  Some regions of the genome will be
sampled multiple times, others never.  The "epoch" is just a counter
that determines when to run validation — it has no relationship to
coverage of the data.

---

## 2. OpenSpliceAI: Fixed Sliding Windows

OpenSpliceAI (and the original SpliceAI) uses the **fixed dataset**
approach.  The training pipeline has two phases:

### Phase 1: Data creation (run once)

Extract all overlapping windows from all training genes and save to
HDF5.  Each gene produces ceil(gene_length / SL) windows:

```python
# From openspliceai/create_data/utils.py (simplified)
SL = 5000       # output sequence length (labels for 5000 positions)
CL_max = 10000  # context length (5000 on each side)

def create_datapoints(sequence, labels):
    """Extract all overlapping windows from a gene."""
    windows_X, windows_Y = [], []
    for i in range(len(sequence) // SL):
        start = i * SL
        x = sequence[start : start + SL + CL_max]  # 15,000 nt input
        y = labels[start : start + SL]               # 5,000 nt labels
        windows_X.append(x)
        windows_Y.append(y)
    return windows_X, windows_Y
```

A gene of 100,000 nt produces 20 windows.  The full human genome
produces ~600K windows total.  These are saved once and reused across
all training runs.

### Phase 2: Training (epoch = full pass over all windows)

Each epoch shuffles the shard order, then iterates through every shard
completely:

```python
# From openspliceai/train_base/utils.py (simplified)
def train_epoch(model, shards, ...):
    shuffled = np.random.permutation(len(shards))
    for shard_idx in shuffled:
        X, Y = load_shard(shards[shard_idx])
        loader = DataLoader(TensorDataset(X, Y), shuffle=True, ...)
        for batch_X, batch_Y in loader:
            loss = model(batch_X, batch_Y)
            loss.backward()
            ...
```

Every window is seen exactly once per epoch.  Training for 10 epochs
means every window is seen 10 times.

### How OpenSpliceAI handles class imbalance

With ~98% of positions being "neither" (non-splice), the model sees
overwhelming background in every window.  Rather than biasing the data
sampling, OpenSpliceAI uses **focal loss** during training:

```python
# gamma=2 down-weights easy (high-confidence) examples
focal_loss = -mean(y_true * log(y_pred) * (1 - y_pred)^gamma)
```

Focal loss with gamma=2 makes the model focus on hard examples (splice
sites that it gets wrong) and ignore easy background positions that it
already classifies correctly.  This is a **loss-level** solution to
class imbalance, as opposed to a **data-level** solution like
oversampling.

---

## 3. Agentic-SpliceAI Meta-Layer: Stochastic Sampling

The meta-layer's `SequenceLevelDataset` takes the opposite approach —
**no fixed dataset at all**.

### Why stochastic?

The meta-layer operates on a different problem than the base model.  The
base model processes raw DNA sequence with 10,000 nt of context — its
windows are large and fixed.  The meta-layer refines base model scores
using multimodal features, with a smaller 5,001-position output window.

The key difference: **the space of possible windows is much larger than
what can be enumerated**.

For a single gene of 100,000 positions, the meta-layer can place its
5,001-position window at any of ~95,000 starting positions.  Across
12,000 training genes spanning ~50 million nucleotides, the space of
possible windows is ~50 million.  Pre-extracting all of them would
require ~50M windows × ~60 KB each ≈ 3 TB — impractical.

Instead, each `__getitem__` call samples a fresh window:

```python
def __getitem__(self, idx):
    # idx is IGNORED — fresh random sample every call

    # 50% chance: center on a splice site (biased sampling)
    if self._rng.random() < self.splice_bias:
        gene_idx = self._rng.choice(self._genes_with_splices)
        gene = load(gene_idx)
        splice_pos = self._rng.choice(gene.splice_positions)
        jitter = self._rng.randint(-W//4, W//4 + 1)
        window_start = splice_pos - W//2 + jitter

    # 50% chance: random position in random gene
    else:
        gene_idx = self._rng.choice(self._valid_genes)
        gene = load(gene_idx)
        window_start = self._rng.randint(0, gene.length - W)

    return extract_window(gene, window_start)
```

### The `samples_per_epoch` parameter

Since there's no fixed dataset, the epoch length is a hyperparameter.
`__len__()` returns `samples_per_epoch` to tell the DataLoader when to
stop iterating.

```python
def __len__(self):
    return self.samples_per_epoch  # e.g. 100,000
```

With `samples_per_epoch=100000` and `batch_size=8`:
- One "epoch" = 12,500 batches
- After each epoch: run validation, step LR scheduler, check early stopping
- Next epoch: 100,000 **completely new** random windows

The choice of `samples_per_epoch` balances two things:
- **Too small** (e.g. 10K): validation runs too often (wasted time),
  training signal is noisy, LR scheduler steps too aggressively
- **Too large** (e.g. 1M): too long between validation checks, slow
  response to overfitting, wasted training if model has already converged

100K is a reasonable default — roughly one validation check every
8 minutes on an A40.

### Splice-bias sampling

Unlike OpenSpliceAI (which relies on focal loss for class imbalance),
the meta-layer uses **data-level biasing**: 50% of windows are centered
near a splice site, 50% are random.

This means every training batch contains roughly equal numbers of
windows with and without splice sites, regardless of their natural
frequency (~2%).  Combined with focal loss (gamma=2), the model sees
splice sites ~25x more often than their natural rate.

The jitter (±W/4) prevents the model from memorizing exact splice site
positions — the splice site appears at different offsets within the
window each time.

---

## 4. Comparison

| Property | OpenSpliceAI (fixed) | Meta-layer (stochastic) |
|----------|---------------------|------------------------|
| **Dataset creation** | Pre-extract all windows once | None — sample on-the-fly |
| **Epoch meaning** | Full pass over all data | Arbitrary sample count |
| **Same window seen twice?** | Yes, once per epoch | Essentially never |
| **Window position variety** | Fixed grid (every SL positions) | Continuous (any start position) |
| **Data augmentation** | Implicit (window overlap) | Implicit (random position + jitter) |
| **Class imbalance** | Focal loss only | Splice-biased sampling + focal loss |
| **Storage cost** | ~50 GB (pre-extracted HDF5) | ~10 GB (gene-level NPZ/shards) |
| **Reproducibility** | Deterministic per seed | Deterministic per seed (RNG state) |

### When to use which

**Fixed dataset** is appropriate when:
- The data space is small enough to enumerate
- You need exact reproducibility across runs
- You want to guarantee every example is seen
- The model is large and training is expensive (want predictable convergence)

**Stochastic sampling** is appropriate when:
- The data space is too large to enumerate
- You want implicit data augmentation (random crops, jitter)
- Class imbalance benefits from biased sampling
- The model is small and epochs are cheap (can afford some variance)

---

## 5. A Subtlety: Stochastic Sampling as Data Augmentation

Stochastic window sampling is implicitly a form of **data augmentation**.
Every splice site is seen in a different position within the window, with
different flanking context, on every encounter.  This is analogous to
random cropping in image training.

Consider a donor site at genomic position 10,000:

```
Epoch 1, sample 47:    window [7200 : 12201]  → donor at offset 2800
Epoch 1, sample 892:   window [8500 : 13501]  → donor at offset 1500
Epoch 2, sample 156:   window [9100 : 14101]  → donor at offset 900
```

The model never memorizes "donor at offset 2500" — it must learn
position-invariant features that recognize donors regardless of where
they appear in the window.

OpenSpliceAI achieves a similar (weaker) effect through overlapping
windows at a fixed grid.  A donor at position 10,000 appears in windows
starting at 5001, 10001, and 15001 — three views with different
offsets, but always the same three views.

---

## 6. Practical Implications

### For training scripts

If you're using `SequenceLevelDataset`, be aware that:

- `shuffle=True` in the DataLoader is a no-op (the dataset ignores indices)
- Increasing `samples_per_epoch` is not the same as training for more epochs
  on a fixed dataset — it changes how often you validate, not how much
  unique data the model sees
- The training/validation split is at the **gene level**, not the window
  level.  A gene's windows never appear in both train and validation.

### For evaluation

Evaluation must be deterministic.  The `infer_full_gene()` function uses
a **fixed sliding window** with 50% overlap — no randomness.  Every
position in the gene is scored exactly once (or twice in overlap
regions, where predictions are averaged).

### For reproducibility

Despite the stochastic sampling, training is fully reproducible given
the same seed.  The dataset uses `np.random.RandomState(seed)` which
produces the same sequence of random choices for the same seed.  The
key is to also fix `torch.manual_seed(seed)` and
`torch.cuda.manual_seed_all(seed)` for model weight initialization
and dropout.
