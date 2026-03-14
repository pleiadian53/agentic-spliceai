# HyenaDNA: Unexpected Keyword Argument 'model_size'

## The Problem

Loading HyenaDNA via `load_embedding_model("hyenadna")` fails with:

```
TypeError: HyenaDNAModel.__init__() got an unexpected keyword argument 'model_size'
```

### Error Signature

```
load_embedding_model("hyenadna", model_size="medium-160k", device="cuda")
  -> model_cls(**kwargs)                    # base.py:291
    -> HyenaDNAModel(model_size="medium-160k", device="cuda")
      -> TypeError: unexpected keyword argument 'model_size'
```

## Root Cause

`load_embedding_model()` in `base.py` passes `**kwargs` directly to the model class
constructor. Evo2 and SpliceBERT accept `**kwargs` and forward them to their config
dataclass, but HyenaDNA's `__init__` only accepted `config: Optional[HyenaDNAConfig]`
— no `**kwargs`.

The inconsistency:
```python
# Evo2 — works: accepts **kwargs, forwards to config
class Evo2Model:
    def __init__(self, config=None, **kwargs): ...

# SpliceBERT — works: same pattern
class SpliceBERTModel:
    def __init__(self, config=None, **kwargs): ...

# HyenaDNA — broken: no **kwargs
class HyenaDNAModel:
    def __init__(self, config=None): ...  # TypeError on extra kwargs
```

## Solution

Add `**kwargs` to `HyenaDNAModel.__init__()` and forward to `HyenaDNAConfig`:

```python
def __init__(
    self,
    config: Optional[HyenaDNAConfig] = None,
    **kwargs,
) -> None:
    if config is not None:
        self.config = config
    else:
        self.config = HyenaDNAConfig(**kwargs)
```

This matches the pattern used by Evo2 and SpliceBERT.

## Prevention

All `BaseEmbeddingModel` subclasses should follow the same constructor pattern:
1. Accept `config: Optional[XConfig] = None` as the first argument.
2. Accept `**kwargs` and forward to the config dataclass when no config is provided.
3. This ensures `load_embedding_model(name, **kwargs)` works uniformly.

Consider enforcing this in `BaseEmbeddingModel` via a common `__init_subclass__` check
or documenting it as a required pattern in `base.py`.

## Environment

- Any environment (local or pod)
- Triggered when using the `load_embedding_model()` factory with keyword arguments
