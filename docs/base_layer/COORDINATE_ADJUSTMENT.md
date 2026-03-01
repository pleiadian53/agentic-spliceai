# Coordinate Adjustment for Splice Site Prediction

**Topic**: How base models (SpliceAI, OpenSpliceAI) map predicted scores to genomic positions, and how to correct for systematic coordinate offsets.

---

## The Problem: Why Recall Can Drop to 40%

When running a base model against a ground-truth annotation set, you may observe unexpectedly low recall:

```
Gene: TP53
Ground truth:  210 splice sites
Detected:       21 sites  (threshold 0.5)
Recall:         40%  ← should be ~89%

127 sites have score = 0.0 at their annotated position
BUT: high scores exist 1–2 base pairs away!
```

**Red flag**: When exact recall ≈ windowed recall (±2 bp), the issue is not missing signal — the signal is present but displaced. This is a coordinate convention mismatch, not a model quality issue.

---

## Root Cause: Convention Differences Between Models and Annotations

Base models were trained with specific coordinate conventions that differ from standard GTF annotation conventions:

| Model | Site Type | Strand | Offset | Interpretation |
| --- | --- | --- | --- | --- |
| SpliceAI | Donor | `+` | +2 bp | Predicts 2 nt upstream of annotated position |
| SpliceAI | Donor | `-` | +1 bp | Predicts 1 nt upstream |
| SpliceAI | Acceptor | `+` | 0 bp | Exact match |
| SpliceAI | Acceptor | `-` | −1 bp | Predicts 1 nt downstream |
| OpenSpliceAI | Donor | `+` | 0 bp | Exact match |
| OpenSpliceAI | Donor | `-` | +1 bp | Predicts 1 nt upstream |
| OpenSpliceAI | Acceptor | `+` | 0 bp | Exact match |
| OpenSpliceAI | Acceptor | `-` | −1 bp | Predicts 1 nt downstream |

**Why these offsets exist**:

- GTF annotations use 1-based, inclusive coordinates
- Models were trained with 0-based, exclusive coordinates
- Exon boundary definitions vary between annotation sources

---

## The Solution: Score-Array Rolling (`np.roll`)

### The Wrong Approach — Shifting Peaks After Detection

```python
# WRONG: find peaks first, then shift positions
peaks = find_peaks(scores, threshold=0.5)
adjusted_peaks = peaks + offset  # ← cannot create new peaks!
```

This fails because if the true position has a score below threshold (e.g., 0.1), shifting an existing peak there doesn't raise its score. The true peak stays invisible.

### The Correct Approach — Roll the Score Array Before Detection

```python
# CORRECT: roll the entire score array first
adjusted_scores = np.roll(scores, shift=offset)
adjusted_scores[:offset] = 0  # zero out wrapped edge values
peaks = find_peaks(adjusted_scores, threshold=0.5)
```

**Why this works**:

```
BEFORE rolling (offset = +2):
Position:  [0]   [1]   [2]   [3]   [4]   [5]
Score:     0.1   0.2   0.9   0.3   0.1   0.0
                       ↑ peak found here
                                   ↑ true annotated site

AFTER rolling by +2:
Position:  [0]   [1]   [2]   [3]   [4]   [5]
Score:     0.0   0.0   0.1   0.2   0.9   0.3
                                   ↑ peak now aligns with annotation ✅
```

The high score (0.9) was always meant for position 4 — the model just indexed it at position 2. Rolling corrects the mapping before peak detection.

**Key principle**: Apply `np.roll()` to the score array *before* peak detection, not position shifting *after*.

---

## Implementation

### Core Adjustment Function

```python
# src/agentic_spliceai/splice_engine/base_layer/utils/coordinate_adjustment.py

def apply_custom_adjustments(
    scores: np.ndarray,
    strand: str,
    splice_type: str,
    adjustment_dict: dict,
    is_neither_prob: bool = False,
) -> np.ndarray:
    """
    Apply coordinate adjustment to a score array using np.roll().

    Parameters
    ----------
    scores : np.ndarray
        Raw score array from the base model (donor or acceptor probabilities).
    strand : str
        Gene strand ('+' or '-').
    splice_type : str
        'donor' or 'acceptor'.
    adjustment_dict : dict
        Format: {'donor': {'plus': 2, 'minus': 1}, 'acceptor': {'plus': 0, 'minus': -1}}
    is_neither_prob : bool
        If True, fill wrapped edges with 1.0 (background probability), else 0.0.

    Returns
    -------
    np.ndarray
        Adjusted score array ready for peak detection.
    """
    strand_key = 'plus' if strand == '+' else 'minus'
    offset = adjustment_dict[splice_type][strand_key]

    if offset == 0:
        return scores

    adjusted = np.roll(scores, offset)

    # Zero (or 1.0 for background) the wrapped edge to prevent artifacts
    fill = 1.0 if is_neither_prob else 0.0
    if offset > 0:
        adjusted[:offset] = fill
    else:
        adjusted[offset:] = fill

    return adjusted
```

### Integration Into the Prediction Pipeline

```python
# CRITICAL: adjustments must be applied BEFORE find_peaks()

donor_scores, acceptor_scores = model.predict(sequence)

if adjustment_dict is not None:
    donor_scores = apply_custom_adjustments(
        donor_scores, strand=strand, splice_type='donor',
        adjustment_dict=adjustment_dict
    )
    acceptor_scores = apply_custom_adjustments(
        acceptor_scores, strand=strand, splice_type='acceptor',
        adjustment_dict=adjustment_dict
    )

donor_peaks    = find_peaks(donor_scores,    threshold=0.5)
acceptor_peaks = find_peaks(acceptor_scores, threshold=0.5)
```

### Known Offsets (Hardcoded Constants)

```python
SPLICEAI_ADJUSTMENTS = {
    'donor':    {'plus': +2, 'minus': +1},
    'acceptor': {'plus':  0, 'minus': -1},
}

OPENSPLICEAI_ADJUSTMENTS = {
    'donor':    {'plus':  0, 'minus': +1},
    'acceptor': {'plus':  0, 'minus': -1},
}
```

### Automatic Detection and Caching

For new base models without known offsets, adjustments can be detected empirically from a calibration sample and cached for subsequent runs:

```python
from agentic_spliceai.splice_engine.base_layer.utils import get_or_detect_adjustments

adjustments = get_or_detect_adjustments(
    model_type='spliceai',
    build='GRCh37',
    annotation_source='ensembl',
    # On cache miss: samples ~50 genes, detects offsets (~30–60s), saves cache
)

# Cache stored at: data/ensembl/GRCh37/spliceai_coordinate_adjustments.json
```

---

## Common Mistakes

### Mistake 1: Shifting positions instead of scores

```python
# WRONG
peaks = find_peaks(scores)
adjusted = peaks + offset        # cannot recover sub-threshold signal

# CORRECT
adjusted_scores = np.roll(scores, offset)
adjusted_scores[:offset] = 0
peaks = find_peaks(adjusted_scores)
```

### Mistake 2: Wrong offset sign

If the model predicts *upstream* (too early), the scores are displaced backward — roll **forward** (`+`):

```python
# Model predicts 2 bp too early → roll forward by +2
adjusted = np.roll(scores, +2)   # ✅
adjusted = np.roll(scores, -2)   # ❌ makes it worse
```

### Mistake 3: Forgetting to zero the wrapped edges

`np.roll()` is circular — values wrap around. Always zero the edge:

```python
adjusted = np.roll(scores, +2)
adjusted[:2] = 0    # ← required, otherwise edge artifacts appear
```

---

## Expected Impact

Applying coordinate adjustments for SpliceAI on TP53 (GRCh37, Ensembl annotations):

| Metric | Without adjustment | With adjustment |
| --- | --- | --- |
| Recall | 40% | 89% |
| Precision | 86% | 97% |
| F1 | 0.55 | 0.93 |

---

## Note on Evaluation Against Canonical vs. All Transcripts

Even with correct coordinate adjustments, recall on a comprehensive annotation set
(all transcripts) will be ~40% — this is **expected and correct**. Comprehensive annotations include many alternative isoform splice sites that SpliceAI was not trained to predict.

Use `--transcript-filter canonical` to evaluate against canonical transcripts only, where recall should be ~90%.

```bash
# Canonical evaluation (expected: ~90% recall)
python examples/03_prediction_with_evaluation.py --gene TP53 --transcript-filter canonical

# All-transcript evaluation (expected: ~40% recall — not a bug)
python examples/03_prediction_with_evaluation.py --gene TP53
```

---

## Related Documentation

- [Position & Coordinate Systems](POSITION_COORDINATE_SYSTEMS.md) — Genomic coordinate conventions (0-based vs 1-based, GTF indexing)
- [Processing Architecture](PROCESSING_ARCHITECTURE.md) — How adjustments fit into the prediction pipeline
- [Base Layer Architecture](../system_design/base_layer_architecture.md) — Overall base layer design

---

**Last Updated**: February 2026
