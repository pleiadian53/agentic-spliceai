# DNABERT-2: Transformers 5.x Compatibility Issues

## Symptom

```
File ".../bert_layers.py", line 42, in __init__
    padding_idx=config.pad_token_id)
               ^^^^^^^^^^^^^^^^^^^
AttributeError: 'BertConfig' object has no attribute 'pad_token_id'
```

Also emits: ``torch_dtype` is deprecated! Use `dtype` instead!``

## Root Cause

DNABERT-2's custom `bert_layers.py` (hosted on HuggingFace, downloaded at runtime via
`trust_remote_code=True`) accesses `config.pad_token_id` in `BertEmbeddings.__init__`.

In **transformers 4.x**, accessing a missing config attribute returned `None` silently.
In **transformers 5.x**, `BertConfig.__getattribute__` was made strict — it raises
`AttributeError` for attributes not explicitly set in the config.

DNABERT-2's config on HuggingFace doesn't include `pad_token_id`, so:
- transformers 4.x: `config.pad_token_id` returns `None`, `Embedding(padding_idx=None)` works
- transformers 5.x: `config.pad_token_id` raises `AttributeError`

## Fix

Load the config first, inject `pad_token_id` from the tokenizer, then pass the patched
config to `AutoModel.from_pretrained()`:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Patch config before model creation
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
    config.pad_token_id = tokenizer.pad_token_id or 0

model = AutoModel.from_pretrained(
    model_id,
    config=config,
    trust_remote_code=True,
    dtype=model_dtype,  # NOT torch_dtype (deprecated in 5.x)
)
```

---

## Error 2: ALiBi Meta-Device Tensor Mismatch

### Symptom

After fixing `pad_token_id`, the model fails during ALiBi tensor construction:

```
File ".../bert_layers.py", line 399, in rebuild_alibi_tensor
    alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
RuntimeError: Tensor on device meta is not on the expected device cpu!
```

### Root Cause

Passing `dtype=` (or the deprecated `torch_dtype=`) to `AutoModel.from_pretrained()`
triggers transformers 5.x's **meta-device initialization** — the model skeleton is
created on a virtual `meta` device for memory efficiency, then weights are loaded.

DNABERT-2's custom `rebuild_alibi_tensor()` creates real CPU tensors during `__init__`,
but the slopes tensor is on `meta` device (from the meta-device skeleton). The
multiplication fails because you can't mix `meta` and `cpu` tensors.

### Fix

Do NOT pass `dtype=` to `from_pretrained()`. Load in float32, then cast afterward:

```python
# Load in float32 (avoids meta-device initialization)
model = AutoModel.from_pretrained(
    model_id,
    config=config,
    trust_remote_code=True,
    # NO dtype= parameter here!
)

# Cast to float16 after loading (CUDA only)
if device not in ("mps", "cpu"):
    model = model.half()
```

### Why torch_dtype / dtype Is Also Wrong

Even if `dtype=` didn't trigger meta-device mode, transformers 5.x renamed the parameter
from `torch_dtype` to `dtype`. However, neither works with DNABERT-2's custom code —
the only safe approach is post-load casting.

---

## Combined Fix (Both Errors)

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Fix 1: Patch pad_token_id
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
    config.pad_token_id = tokenizer.pad_token_id or 0

# Fix 2: Load without dtype, cast after
model = AutoModel.from_pretrained(
    model_id, config=config, trust_remote_code=True,
)
if device not in ("mps", "cpu"):
    model = model.half()  # float16 on CUDA

model = model.to(device)
model.eval()
```

## Environment

- transformers 5.3.0 (shipped with `nvcr.io/nvidia/pytorch:25.02-py3`)
- DNABERT-2 HuggingFace ID: `zhihan1996/DNABERT-2-117M`

## Pattern

This is the same class of issue as SpliceBERT's tokenizer problems with transformers 5.x
— remote model code written for transformers 4.x breaks under 5.x's stricter behavior.
The general fix is always: load config/tokenizer first, patch missing attributes, then
create the model.

See also: [SpliceBERT tokenizer issues](../splicebert/splicebert-tokenizer-transformers5.md)
