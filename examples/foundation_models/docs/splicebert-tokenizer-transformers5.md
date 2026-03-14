# SpliceBERT Tokenizer & Model Loading on transformers 5.x

## Overview

Loading SpliceBERT (`multimolecule/splicebert`) on the NVIDIA `pytorch:25.02-py3`
Docker image (transformers 5.3.0) requires bypassing both the `multimolecule` package
and HuggingFace's `Auto*` classes. This document covers six cascading failures and
their fixes, encountered while integrating SpliceBERT into the sparse exon classifier
pipeline (`05_sparse_exon_classifier.py`).

**TL;DR**: Use `BertModel` + `BertTokenizer` directly (not `AutoModel`/`AutoTokenizer`),
remap checkpoint keys, space-separate nucleotides, and convert T→U.

---

## Error 1: AutoTokenizer Fast Tokenizer Failure

### Signature

```
ValueError: Couldn't instantiate the backend tokenizer from one of:
(1) a `tokenizers` library serialization file,
(2) a slow tokenizer instance to convert or
(3) an equivalent slow tokenizer class to instantiate and convert.
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
```

### Root Cause

SpliceBERT's HuggingFace repo provides a sentencepiece `.model` file but **no
pre-compiled `tokenizer.json`** (which fast tokenizers need). In transformers 5.x,
`AutoTokenizer` defaults to fast tokenizers and attempts to convert the slow
sentencepiece tokenizer to a fast one. This conversion fails even when `sentencepiece`
is installed.

`use_fast=False` does NOT help — `AutoTokenizer` in transformers 5.x still routes
through the fast tokenizer path for model types it doesn't recognize natively.

### Slow vs Fast Tokenizers

| | Slow | Fast |
|---|---|---|
| Implementation | Pure Python (`BertTokenizer`) | Rust via `tokenizers` lib (`BertTokenizerFast`) |
| Vocab format | Reads sentencepiece `.model` directly | Needs compiled `tokenizer.json` or slow-to-fast conversion |
| Speed | ~10-100x slower on large batches | Optimized for throughput |
| Impact here | Negligible (hundreds of 1kb sequences) | Not worth the dependency pain |

### Fix

Use `BertTokenizer` directly instead of `AutoTokenizer`:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "multimolecule/splicebert", do_lower_case=False,
)
```

`do_lower_case=False` is critical — without it, nucleotides get lowercased and may
not match the sentencepiece vocabulary correctly.

---

## Error 2: multimolecule Package Version Incompatibility

### Signature

```python
# transformers 5.3.0 (NVIDIA Docker default)
ImportError: cannot import name 'check_model_inputs' from 'transformers.utils.generic'

# transformers 4.51.3 (downgrade attempt)
ImportError: cannot import name 'TransformersKwargs' from 'transformers.utils'
```

### Root Cause

The original wrapper used `import multimolecule` to register `splicebert` with
transformers' `AutoModel` registry. `multimolecule 0.0.9` pins a narrow transformers
window (~5.0-5.2) that is incompatible with both the NVIDIA image default (5.3.0) and
reasonable downgrades (4.51.3).

| transformers version | multimolecule 0.0.9 error |
|---|---|
| 5.3.0 (NVIDIA default) | `cannot import name 'check_model_inputs'` |
| 4.51.3 (downgrade) | `cannot import name 'TransformersKwargs'` |

### Why not downgrade transformers?

Downgrading transformers to satisfy multimolecule would risk breaking other foundation
models (Evo2, HyenaDNA) that rely on transformers 5.x features. A shared dependency
version must work for all models on the same pod.

### Fix

**Eliminate the `multimolecule` dependency entirely.** SpliceBERT is architecturally a
standard BERT — `BertModel` + `BertTokenizer` load it directly without any custom model
registration. The only pip dependency needed is `sentencepiece` (for the tokenizer).

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("multimolecule/splicebert", do_lower_case=False)
model = BertModel.from_pretrained("multimolecule/splicebert")  # works on transformers <5.x
```

> **Note**: `BertModel.from_pretrained()` with `state_dict=` fails on transformers 5.x
> (see Error 3 below). The two-step config+load approach is needed.

---

## Error 3: state_dict + model_id in transformers 5.x

### Signature

```
ValueError: You provided both a `model_id` and `state_dict`. The `state_dict` argument
cannot be passed together with a model identifier.
```

### Root Cause

SpliceBERT's checkpoint uses non-standard key names (see Error 4 below), requiring
state dict remapping before loading. In transformers <5.x, you could pass:

```python
BertModel.from_pretrained(model_id, state_dict=remapped_dict)
```

transformers 5.x removed this capability — you can't pass `state_dict` and a model ID
together.

### Fix

Load config first, create an empty model, then load the remapped state dict:

```python
from transformers import BertConfig, BertModel

bert_config = BertConfig.from_pretrained("multimolecule/splicebert")
model = BertModel(bert_config)                          # empty model
remapped = _remap_state_dict("multimolecule/splicebert") # see Error 4
model.load_state_dict(remapped, strict=False)
```

`strict=False` is needed because SpliceBERT's checkpoint doesn't include the `pooler`
layer weights (which are not needed for embedding extraction).

---

## Error 4: State Dict Key Mismatch (AUROC = 0.5)

### Symptom

Model loads without errors but produces **random embeddings** — AUROC = 0.5 (chance),
embedding variance near zero. Diagnostic logging reveals 0/101 keys matched between
the checkpoint and model.

### Root Cause

SpliceBERT's checkpoint uses a different naming convention than standard `BertModel`:

| Checkpoint key | BertModel key |
|---|---|
| `model.encoder.layer.0.attention...` | `encoder.layer.0.attention...` |
| `model.embeddings.word_embeddings.weight` | `embeddings.word_embeddings.weight` |
| `...layer_norm.weight` | `...LayerNorm.weight` |
| `lm_head.dense.weight` | *(not in BertModel)* |

Three transformations needed:
1. Strip `model.` prefix from all keys
2. Rename `layer_norm` to `LayerNorm` (capital L, capital N)
3. Skip `lm_head.*` keys (masked LM head, not part of the encoder)

### Fix

```python
@staticmethod
def _remap_state_dict(model_id: str) -> dict:
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "model.safetensors")
    raw = load_file(path)

    remapped = {}
    for key, value in raw.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        new_key = new_key.replace("layer_norm", "LayerNorm")
        if new_key.startswith("lm_head"):
            continue
        remapped[new_key] = value
    return remapped
```

After remapping: 101/101 keys matched, only `pooler.*` keys missing (expected).

### Diagnostic Pattern

Add key-matching logging after remapping to catch this class of error early:

```python
model_keys = set(model.state_dict().keys())
remapped_keys = set(remapped.keys())
matched = model_keys & remapped_keys
logger.info("State dict: %d remapped, %d model, %d matched", ...)
if len(matched) == 0:
    logger.error("NO keys matched! Sample model keys: %s", list(model_keys)[:5])
    logger.error("Sample remapped keys: %s", list(remapped_keys)[:5])
```

---

## Error 5: WordPiece Tokenization — Entire Sequence as Single [UNK]

### Symptom

With correct weights loaded (101/101 matched), model STILL produces AUROC = 0.5.
Tokenizer diagnostic reveals:

```
Tokenizer diagnostic: 'ACGTACGT' -> ids=[1, 3, 2], tokens=['<cls>', '<unk>', '<eos>']
```

The entire 8-nucleotide sequence maps to a single `<unk>` token.

### Root Cause

`BertTokenizer` uses WordPiece tokenization, which **splits on whitespace first**, then
does subword tokenization within each whitespace-delimited "word." A raw DNA sequence
like `"ACGTACGT"` is treated as a single word — and since `ACGTACGT` isn't in the
sentencepiece vocabulary, it maps to `<unk>`.

SpliceBERT expects single-nucleotide tokenization where each character (A, C, G, U) is
a separate token. The input must be space-separated.

### Fix

Space-separate nucleotides before tokenization:

```python
# Before: entire sequence -> single <unk>
inputs = tokenizer("ACGTACGT", ...)

# After: each nucleotide -> its own token
inputs = tokenizer(" ".join("ACGTACGT"), ...)
# equivalent to: tokenizer("A C G T A C G T", ...)
```

---

## Error 6: T Nucleotide Maps to `<unk>` (RNA Vocabulary)

### Symptom

After space-separation fix, A/C/G tokenize correctly but **T maps to `<unk>`**:

```
Tokenizer diagnostic: 'ACGTACGT'
  -> tokens=['<cls>', 'A', 'C', 'G', '<unk>', 'A', 'C', 'G', '<unk>', '<eos>']
```

Model achieves AUROC = 0.6993 — real signal, but degraded because ~25% of nucleotides
are losing information.

### Root Cause

SpliceBERT was trained on **pre-mRNA sequences** (2 million from 72 vertebrate species).
Pre-mRNA uses RNA convention: **U (uracil) instead of T (thymine)**. The sentencepiece
vocabulary contains `U` but not `T`.

SpliceBERT vocab (26 tokens):
```
<cls>=1, <eos>=2, <unk>=3, <pad>=0, N=5, A=6, C=7, G=8, U=9, ...
```

### Impact

| Metric | T as `<unk>` | After T→U fix | Improvement |
|---|---|---|---|
| AUROC | 0.6993 | **0.8594** | +0.16 |
| AUPRC | 0.3697 | **0.5458** | +0.18 |
| Info loss | ~25% of positions | 0% | — |

The T→U fix alone accounted for the largest single improvement. AUROC 0.8594 is within
the paper's reference range (0.82-0.99).

### Fix

Convert T→U before tokenization:

```python
spaced = [" ".join(s.replace("T", "U").replace("t", "u")) for s in sequences]
inputs = tokenizer(spaced, return_tensors="pt", padding=True, truncation=True, ...)
```

After fix, diagnostic shows all nucleotides mapped correctly:
```
Tokenizer diagnostic: 'ACGUACGU'
  -> tokens=['<cls>', 'A', 'C', 'G', 'U', 'A', 'C', 'G', 'U', '<eos>']
```

---

## Complete Working Pattern

Putting all six fixes together — the minimal correct way to load SpliceBERT on any
transformers version:

```python
from transformers import BertConfig, BertModel, BertTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

model_id = "multimolecule/splicebert"

# 1. Tokenizer: BertTokenizer (not AutoTokenizer), case-sensitive
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=False)

# 2. Model: config-first, then load remapped weights
bert_config = BertConfig.from_pretrained(model_id)
model = BertModel(bert_config)

# 3. Remap state dict keys
path = hf_hub_download(model_id, "model.safetensors")
raw = load_file(path)
remapped = {}
for key, value in raw.items():
    new_key = key
    if new_key.startswith("model."):
        new_key = new_key[len("model."):]
    new_key = new_key.replace("layer_norm", "LayerNorm")
    if new_key.startswith("lm_head"):
        continue
    remapped[new_key] = value

model.load_state_dict(remapped, strict=False)
model.eval()

# 4. Encode: T->U conversion + space-separated nucleotides
sequence = "ACGTACGT"
spaced = " ".join(sequence.replace("T", "U"))
inputs = tokenizer(spaced, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[0, 1:-1, :]  # strip [CLS] and [SEP]
# embeddings.shape: [8, 512]
```

## Dependencies

```
pip install transformers sentencepiece safetensors huggingface_hub
```

No `multimolecule` package needed.

## Environment

- Docker: `nvcr.io/nvidia/pytorch:25.02-py3`
- transformers: 5.3.0 (image default)
- sentencepiece: 0.2.1
- Python: 3.12
- GPU: NVIDIA RTX A5000 (RunPod, CA-MTL-1)

## Prevention

- For BERT-architecture models on HuggingFace, prefer explicit `BertModel`/`BertTokenizer`
  over `AutoModel`/`AutoTokenizer` when the model repo lacks `tokenizer.json`.
- Avoid wrapper packages (`multimolecule`) that pin narrow transformers version ranges —
  they break on NVIDIA Docker images that ship bleeding-edge transformers.
- Always run a tokenizer diagnostic after loading: encode a known sequence and verify
  that every nucleotide maps to a real token (not `<unk>`).
- For models trained on RNA (pre-mRNA, mRNA), check whether the vocabulary uses U or T.
  The paper or training data description usually mentions this.
- Document the transformers version shipped by each Docker image in provisioning scripts.

## Timeline

These errors were encountered and resolved in sequence over a single debugging session
(Mar 12-13, 2026) while running `05_sparse_exon_classifier.py --model splicebert` on a
RunPod RTX A5000 pod. Each fix revealed the next error — a classic "onion peeling"
debugging pattern where the surface error masks deeper issues.
