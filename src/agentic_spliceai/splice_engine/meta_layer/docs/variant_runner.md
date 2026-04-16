# `variant_runner` — From a VCF Line to a Splice-Effect Call

This is a tutorial-style walkthrough of
`splice_engine/meta_layer/inference/variant_runner.py` — the module
that takes a single variant and produces the per-position delta
scores and high-level "events" used by the M4 variant-analysis
benchmarks.

Audience: someone comfortable with Python and ML who is new to the
biology of splice variants. Conventions and gotchas are spelled out
explicitly.

---

## 1. What the module does

Given a variant — chromosome, position, reference allele, alternate
allele — the runner answers two questions:

1. **Where in the surrounding sequence does the variant change the
   model's prediction?** Output: a per-position ``[L, 3]`` array of
   delta scores (`alt - ref`) over a window centered on the variant.
2. **What kind of effect is that?** Output: a list of human-readable
   `SpliceEvent` objects (`donor_gain`, `donor_loss`, `acceptor_gain`,
   `acceptor_loss`) annotated with magnitude and distance from the
   variant.

The pipeline orchestrates four pieces of machinery, each of which
exists for an independent reason:

```
              FASTA                         Annotation
                │                                │
       fetch_sequence(window)        infer strand + transcript
                │                                │
                └──── _to_genomic_alleles ───────┘
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
ref window                              mutated alt window
       │                                         │
       │   ─── base model (OpenSpliceAI etc.) ───▶
       │                                         │
       ▼                                         ▼
ref base scores                         alt base scores
       │                                         │
       │ + multimodal features (conservation,    │
       │   junction, RBP, chromatin) — same      │
       │   for ref and alt at SNV positions      │
       │                                         │
       ▼                                         ▼
        ────── meta-layer predict_with_delta ──────
                            │
                            ▼
                  delta [L, 3]  →  SpliceEvents
```

---

## 2. Variant notation, briefly

Three notations show up in this code path:

- **VCF-style genomic** — `chr1`, position 11802880, ref `C`, alt `T`.
  REF/ALT always refer to the **plus-strand reference genome**.
- **HGVS coding (`c.`) notation** — e.g., `NM_005957.5:c.236+1G>A`.
  The `c.` stands for "**coding sequence**" — coordinates are numbered
  along the CDS of a specific transcript, with `+N` indicating the
  Nth intronic base after a coding position and `-N` the Nth base
  before. So `c.236+1` means "first intronic base just after coding
  position 236". This is **transcript-relative**: the transcript ID
  specifies a particular mRNA (often a minus-strand gene), and `G>A`
  is the change as it appears on that transcript. For a minus-strand
  gene, the transcript is the reverse complement of the genome, so on
  the plus-strand FASTA the *same* variant looks like `C>T`.

  Other HGVS prefixes you'll encounter:
    - `g.` — genomic (numbered on the reference assembly)
    - `n.` — non-coding RNA transcript
    - `r.` — RNA (lowercase bases: `r.236+1g>a`)
    - `p.` — protein (amino acid change, e.g., `p.Arg79His`)
    - `m.` — mitochondrial DNA

- **HGVS protein (`p.`) notation** — describes the amino-acid change.
  Not used by the runner itself.

**Convention in `variant_runner.run()`:** ref/alt may be passed in
either orientation. Pass the strand explicitly, and the runner
normalizes:

```python
runner.run(chrom='chr1', position=11802880,
           ref='G', alt='A', strand='-',  # transcript orientation OK
           gene='MTHFR')

# is equivalent to (after internal RC):

runner.run(chrom='chr1', position=11802880,
           ref='C', alt='T', strand='-',  # genomic orientation OK
           gene='MTHFR')
```

The internal `_to_genomic_alleles(ref, alt, strand)` does this
conversion. Without it, the `_apply_variant` step would produce a
`Ref mismatch` warning (the FASTA at that position has `C`, not
`G`) **and** silently substitute the wrong base — making the alt
sequence the reverse complement of the intended mutation. This
class of bug was the original motivation for the helper.

---

## 3. The window and where it sits

`window_size` defaults to **5001** bases, centered on the variant.
A 5kb window is large enough to catch most splice-site changes
the spliceosome cares about within the same intron / adjacent exon.

```
                         variant
                            │
       ◄── 2500 bases ──────┼────── 2500 bases ──►
       │                                          │
       └──────────────── L = 5001 ────────────────┘
              this is the prediction window
```

The base model needs additional context on both sides (it has a
receptive field of ~10kb), so internally the sequence fetched from
the FASTA is wider than 5001. After scoring, the output is cropped
back to exactly `[5001, 3]`.

---

## 4. The three score channels and what they mean

Every position in the window has three probabilities that sum to
1.0:

| Channel | Index | Biology |
|---|:---:|---|
| `donor` | 0 | "This base is the last exonic position before an intron." |
| `acceptor` | 1 | "This base is the first exonic position after an intron." |
| `neither` | 2 | "Not a splice site." |

The base model (OpenSpliceAI/SpliceAI) outputs in the order
`[neither, acceptor, donor]`; the runner reorders to
`[donor, acceptor, neither]` for the meta layer (see
`_run_base_model` lines 327–331). External users of `DeltaResult`
should use the named fields (`max_donor_gain`, etc.) and not index
the array directly.

---

## 5. From delta scores to events

`delta = alt_probs - ref_probs` is a signed `[L, 3]` array. Sign is
the whole point — it tells you whether the variant makes a position
more or less splice-like.

Per-channel interpretation:

| Channel | Sign | Meaning |
|---|:---:|---|
| `donor` | `delta > 0` | **donor gain** — variant created/strengthened a donor here |
| `donor` | `delta < 0` | **donor loss** — variant weakened/abolished a donor here |
| `acceptor` | `delta > 0` | **acceptor gain** |
| `acceptor` | `delta < 0` | **acceptor loss** |
| `neither` | either | not surfaced as an event (uninformative on its own) |

The mapping happens in `_detect_events` (lines 570–611):

```python
for channel, splice_type in [(_DONOR, 'donor'), (_ACCEPTOR, 'acceptor')]:
    scores = delta[:, channel]
    for idx in np.where(scores > event_threshold)[0]:
        emit SpliceEvent(splice_type + '_gain', position, delta, distance)
    for idx in np.where(scores < -event_threshold)[0]:
        emit SpliceEvent(splice_type + '_loss', position, delta, distance)

events.sort(key=lambda e: abs(e.delta), reverse=True)
```

`event_threshold` defaults to **0.1**. This matches the standard
SpliceAI thresholds at which delta scores are considered
biologically meaningful:

| Threshold | Interpretation |
|---|---|
| ≥ 0.2 | "high recall" — many true positives, some false positives |
| ≥ 0.5 | "high precision" — strong evidence of a splice-altering effect |
| ≥ 0.8 | very confident call |

Events are **sorted by `|delta|`**, so `events[0]` is always the
strongest single-position effect. Multiple events per variant are
common: a variant that creates a new donor often also weakens the
canonical donor it competes with, so you'd see one `donor_gain`
near the variant plus one `donor_loss` further away.

The `neither` channel is intentionally not converted to events. A
shift in `neither` probability is the dual of donor/acceptor
shifts (the three sum to 1) and would just produce redundant
"event" labels.

---

## 6. Worked example: a known cryptic-splice variant

```python
runner = VariantRunner(
    meta_checkpoint='output/meta_layer/m2s_v2/best.pt',
    fasta_path='data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
    device='cpu',
)

# MTHFR c.236+1G>A — known to abolish a canonical donor
result = runner.run(
    chrom='chr1', position=11802880, ref='G', alt='A',
    strand='-', gene='MTHFR',
)

print(result.summary())
# Variant: chr1:11802880 G>A (MTHFR)
# Window: 11800380-11805381
# Max delta scores:
#   DS_DG (donor gain):     +0.02
#   DS_DL (donor loss):     -0.97   <-- very strong loss
#   DS_AG (acceptor gain):  +0.01
#   DS_AL (acceptor loss):  -0.05
#   Max |delta|:            0.97
#
# Detected events (3):
#   donor_loss     at chr1:11802880  Δ=-0.972  (+0bp)
#   donor_gain     at chr1:11802884  Δ=+0.183  (+4bp)
#   acceptor_loss  at chr1:11802881  Δ=-0.108  (+1bp)
```

Reading this:
- The biggest single effect is a **donor loss** of magnitude 0.97
  exactly at the variant position. That's the canonical splice
  site being abolished — a textbook splice-altering pathogenic
  variant.
- A small `donor_gain` 4bp downstream suggests the model thinks
  the spliceosome might shift to a nearby cryptic donor.
- Real diagnostic interpretation would combine this with the base
  model's delta (also returned in `DeltaResult`) and clinical
  classifiers like SpliceAI's official thresholds.

---

## 7. Three layers in `DeltaResult`

The returned object exposes scores at three fidelities:

| Layer | Field | Use |
|---|---|---|
| Raw probabilities | `ref_probs`, `alt_probs`, `[L, 3]` | full per-position output |
| Per-position deltas | `delta`, `[L, 3]` | sign + magnitude per channel |
| Discrete events | `events`, list of `SpliceEvent` | actionable summary |

For most downstream tasks (variant prioritization, M4 benchmarks),
you want **events**. For analytical work (calibration plots, error
breakdown), you want the **deltas** array. For research on the
underlying model behavior (saturation, reference-allele
sensitivity), you want the **probabilities**.

The base model has parallel fields (`base_ref_probs`,
`base_alt_probs`, `base_delta`). Comparing meta vs base deltas
helps diagnose whether a meta-layer prediction differs from the
base model because of multimodal features or because of training
on alt sites.

---

## 8. Strand handling — the full picture

**Important framing:** splice predictors (SpliceAI, OpenSpliceAI,
and our meta layer built on top of them) **live in transcript space**.
They were trained on pre-mRNA sequences in 5'→3' transcript
orientation; for minus-strand genes, that's the reverse complement
of the genomic FASTA. The model has internalized transcript-frame
biology (donor consensus preceding GT, acceptor consensus following
AG). The runner's job is to be a **translator** between the genomic
input/output the caller wants and the transcript-frame the model
needs.

Strand-awareness shows up in **three** independent places:

1. **Allele orientation** (`_to_genomic_alleles`): converts HGVS
   transcript-orientation alleles to plus-strand orientation
   before the FASTA-based mutation step. **Was the source of the
   silent ref-mismatch bug fixed in this revision.**
2. **Base-model I/O** (`_run_base_model`, lines 299–325):
   reverse-complements the genomic input sequence to transcript
   order before model inference, then reverses the output back so
   indices match genomic coordinates.
3. **Caller-facing coordinate frame**: all fields on `DeltaResult`
   are in **genomic** order — position `i` in the window maps to
   `window_start + i`, regardless of strand. The transcript-frame
   processing happens entirely inside the runner.

The asymmetry to remember: the **model thinks in transcript space**
but the **API speaks genomic**. Callers can stay in genomic
coordinates throughout as long as they pass strand correctly.

---

## 9. Common gotchas

- **Forgot to pass `strand='-'`:** the runner assumes plus-strand,
  fetches the right sequence, but mutates it as if the alleles
  were on the plus strand. For a minus-strand gene whose ref/alt
  came from HGVS, the substituted base will be wrong (you'll get
  the reverse complement of the intended mutation). No error is
  raised; the delta scores will be silently incorrect. **Always
  pass strand explicitly.**
- **Wrong FASTA build:** the runner does no genome-build
  validation. A `chr` name that exists in both GRCh37 and GRCh38
  will accept either, and the `_resolve_chrom` helper only
  toggles the `chr` prefix. Use the right FASTA for your
  meta-layer checkpoint.
- **Variants near chromosome edges:** the window is centered, so
  variants within 2500bp of a chromosome end will have a truncated
  window. The runner pads with zeros, which biases the prediction
  near the boundary.
- **Indels:** the helper `_apply_variant` handles arbitrary-length
  ref/alt by string substitution (lines 420–429). Coordinate
  bookkeeping for indels longer than a few bases is **not
  rigorously tested** — if you're scoring large insertions or
  deletions, validate independently.

---

## 10. Where this fits in M4

The runner is the engine behind every variant-analysis benchmark:

- `examples/variant_analysis/01_single_variant_delta.py` — single
  variant, full output including events table.
- `examples/variant_analysis/03_clinvar_benchmark.py` — bulk
  pathogenic-vs-benign discrimination using `max(|delta|)` per
  variant as the score.
- `examples/variant_analysis/04_mutsplicedb_benchmark.py` —
  RNA-seq-validated pathogenic variants; sensitivity-only.

For benchmarking, the typical reduction is:

```python
score = result.max_delta  # max |delta| over all positions, donor+acceptor channels
```

That single scalar is what gets compared against ClinVar
pathogenic/benign labels for AUROC / PR-AUC.

---

## 11. Related

- `variant_runner.py` source — line numbers referenced here are
  current as of 2026-04-14.
- `splice_event_detector.py` — alternative event-detection logic
  used by the consequence-prediction pipeline.
- `examples/meta_layer/docs/m3_prerequisites.md` — broader
  context on what the meta layer is for and how it relates to
  novel-site discovery.
