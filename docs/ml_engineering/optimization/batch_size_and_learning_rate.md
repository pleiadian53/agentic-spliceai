# Batch Size, Learning Rate, and Sparse Signals

How batch size, learning rate, and signal sparsity interact in training,
with practical guidance for genomic deep learning.

---

## 1. Batch Size Controls the Gradient Noise

Each training step computes a gradient estimate by averaging over a
mini-batch.  The batch size determines the quality of that estimate:

```
true_gradient ≈ (1/B) × Σ gradient(sample_i)    for i = 1..B
```

- **Small batch** (B=4): noisy gradient, high variance, lots of updates
  per epoch.  Acts as implicit regularization — the noise helps escape
  sharp minima.
- **Large batch** (B=256): smooth gradient, low variance, few updates
  per epoch.  Converges faster per update but may settle in sharper
  (less generalizable) minima.

Neither is inherently better.  The right batch size depends on the model,
the signal density, and the compute budget.

---

## 2. The Linear Scaling Rule

When you increase batch size by a factor k, each gradient update averages
over k times more samples and is therefore k times more precise.  To
compensate for the k-fold reduction in updates per epoch, scale the
learning rate by the same factor:

```
If batch_size doubles:  LR should (roughly) double
```

This is the **linear scaling rule** (Goyal et al., 2017).  The intuition:
a larger batch takes a more confident step, so the step size can be
proportionally larger.

| Batch size | Updates/epoch (100K samples) | LR (linear scaling) |
|-----------|------------------------------|---------------------|
| 8 | 12,500 | 5e-4 |
| 16 | 6,250 | 1e-3 |
| 32 | 3,125 | 2e-3 |
| 64 | 1,562 | 4e-3 |
| 128 | 781 | 8e-3 |

### When linear scaling breaks down

The rule assumes the loss surface is smooth and well-conditioned.  It
fails in practice when:

1. **LR gets too large** (typically >0.01 for Adam): optimization
   diverges.  Large learning rates overshoot the loss landscape,
   especially for small models with simple loss surfaces.

2. **Batch size exceeds the "critical batch size"**: beyond a certain
   point, adding more samples to the batch doesn't reduce gradient
   variance — you're already estimating the true gradient accurately.
   Larger batches just waste compute.

3. **Signal is sparse** (see Section 4): the useful gradient comes from
   a small fraction of positions.  Large batches dilute the per-position
   gradient contribution.

For most genomic models with <10M parameters, batch sizes of 16-64 work
well.  Going beyond 128 rarely helps and requires careful LR tuning.

---

## 3. Gradient Accumulation: Simulating Large Batches

When VRAM is limited, **gradient accumulation** simulates a larger batch
by splitting it across multiple forward/backward passes:

```python
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()           # accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()      # update once per effective batch
        optimizer.zero_grad()
```

With `batch_size=4, accumulation_steps=4`: each weight update uses
gradients from 16 samples, but only 4 are in VRAM at once.

**When to use it**:
- GPU memory can't fit your desired batch size
- You want the optimization dynamics of a large batch

**When NOT to use it**:
- You have plenty of VRAM.  `batch_size=16, accumulation=1` is strictly
  faster than `batch_size=8, accumulation=2` for the same effective
  batch — it does one forward/backward pass instead of two.
- The additional forward/backward passes add overhead without benefit.

### Environment-aware defaults

In agentic-spliceai, the training script auto-detects sensible defaults:

```python
# CUDA (48 GB A40): large batch, no accumulation
#   batch_size=16, accumulation_steps=1 → effective=16
# CPU/MPS (16 GB): small batch + accumulation
#   batch_size=4, accumulation_steps=4 → effective=16
```

Same effective batch size, but the CUDA path is ~2x faster because it
does one forward/backward per update instead of four.

---

## 4. Sparse Signals and Batch Size

### The problem

In genomic sequence-to-sequence models, the prediction target is
**extremely sparse**.  A typical 5001-position output window contains:

- ~50 splice sites (~1%)
- ~4,950 background positions (~99%)

The model must learn to predict the rare splice sites, but 99% of each
window's gradient comes from background positions.  The splice-site
gradient signal is diluted by 100:1.

As batch size increases, this dilution worsens:

| Batch size | Positions/update | Splice positions | Signal fraction |
|-----------|-----------------|------------------|-----------------|
| 4 | 20,004 | ~200 | 1.0% |
| 16 | 80,016 | ~800 | 1.0% |
| 128 | 640,128 | ~6,400 | 1.0% |

The signal fraction stays constant (it's a property of the data, not the
batch size), but the gradient magnitude of each splice-site position
shrinks as 1/B.  With B=128, each splice site contributes 1/640,128 of
the gradient — the optimizer may not react to it.

### Mitigations

1. **Biased sampling** (data-level): Ensure every window contains
   splice sites.  Agentic-spliceai uses `splice_bias=0.5`, centering
   50% of windows on a known splice site.  This guarantees ~2.5% splice
   sites in the batch vs ~1% without biasing.

2. **Focal loss** (loss-level): Down-weight easy (background) positions:
   ```
   focal_loss = -y * log(p) * (1 - p)^gamma
   ```
   With gamma=2, a background position with p=0.99 (correctly classified)
   contributes (0.01)^2 = 0.0001 of its normal gradient.  A misclassified
   splice site with p=0.3 contributes (0.7)^2 = 0.49.  This
   re-balances the effective gradient toward hard (splice) positions.

3. **Class weights** (loss-level): Multiply the loss by inverse class
   frequency.  Simpler than focal loss but less adaptive — it amplifies
   all splice-site losses equally, not just the hard ones.

4. **Keep batch size moderate**: For sparse targets, the sweet spot is
   usually where the effective batch contains enough positive examples
   for a stable gradient (>100 splice sites).  At `batch_size=16` with
   50% splice bias, that's ~400 splice sites per update — adequate.

---

## 5. Learning Rate Schedules

The learning rate doesn't need to be constant.  Common schedules:

### Cosine annealing

LR starts at `lr_max` and decays following a cosine curve to near zero:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

Gentle decay gives the model time to explore early, then fine-tune late.

### Step decay (used in agentic-spliceai)

LR drops by a factor every N epochs without improvement:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### Warmup

Start with a very small LR and linearly increase to the target over the
first few epochs.  Prevents instability from large initial gradients
when weights are randomly initialized.  Critical for large batch sizes
(>64) where the linear scaling rule pushes LR high.

---

## 6. Practical Guidelines

| Scenario | Batch size | LR | Accumulation |
|----------|-----------|-----|--------------|
| Local CPU (prototyping) | 4 | 1e-3 | 4 |
| M1/M2 Mac (MPS) | 4-8 | 1e-3 | 2-4 |
| Single GPU (A40/A100) | 16-64 | 1e-3 to 4e-3 | 1 |
| Multi-GPU (DDP) | 64-256 | 4e-3 to 1e-2 | 1 |

**Rule of thumb for this project**: keep the effective batch size at
16-32 for the meta-layer model (367K params, sparse splice targets).
The I/O pipeline, not the GPU, is the bottleneck — so larger batches
give diminishing returns because the DataLoader can't fill them fast
enough.

---

## 7. References

- Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet
  in 1 Hour." Establishes the linear scaling rule.
- Smith et al. (2018). "Don't Decay the Learning Rate, Increase the
  Batch Size." Shows that increasing batch size during training is
  equivalent to decaying LR.
- Hoffer et al. (2017). "Train Longer, Generalize Better: Closing the
  Generalization Gap in Large Batch Training."
- McCandlish et al. (2018). "An Empirical Model of Large-Batch Training."
  Defines the critical batch size concept.
