Temperature scaling is strictly a **post-hoc calibration** step — it's applied **after** training is complete, not during.

## Why post-training, not during training

1. **T doesn't change the ranking** — it only rescales confidences. PR-AUC, top-k accuracy, and the relative ordering of predictions are invariant to T. It only affects the decision boundary (which positions cross a given probability threshold).
2. **Training already optimizes cross-entropy** — during training, the model implicitly learns some calibration through the loss function. But modern deep nets tend to be overconfident (Guo et al. 2017), so post-hoc T fixes the residual miscalibration.
3. **T is a single scalar** — fitting it takes seconds on the validation set. There's no risk of overfitting a single parameter, so no need for nested CV or expensive search.

## The standard protocol



```
Train set (chr11-22 even, etc.)  →  train model weights + blend_alpha
Validation set (chr2,4,6,8,10)  →  fit T (minimize NLL, ~seconds)
Test set (chr1,3,5,7,9)         →  evaluate with fixed T
```

This is the same train/val/test split you already use — no extra splits or nested CV needed. The validation set is the one that the SpliceAI paper designates for hyperparameter selection.

## What T is and isn't

- **Is**: a calibration knob that adjusts confidence levels. Higher T → softer probabilities → fewer FPs at the cost of slightly more FNs.
- **Is not**: a hyperparameter like learning rate or hidden dim that affects what the model learns. The model weights are frozen when you fit T.

Think of it like choosing a classification threshold — you train the model, then pick the operating point on the PR curve that best suits your application. Temperature scaling is a more principled version of threshold tuning because it calibrates the full probability distribution, not just a single cutoff.

## For M1-S specifically

The FP doubling (7,938 → 15,883) we observed is exactly the kind of overconfidence that temperature scaling addresses. The model's probability mass is too concentrated — many "neither" positions get assigned moderate splice-site probability that crosses the argmax boundary. A T > 1 will soften those predictions back below the decision boundary.

The `--calibrate-temperature` flag we just implemented does this in one step — it runs through the validation cache, fits T in ~10 seconds, then applies it to the test evaluation. No retraining needed.