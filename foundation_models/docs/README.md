# Foundation Models — Documentation

Topic-specific documentation for the `foundation_models` sub-package.

---

## `evo2/` — Evo2 Model Topics

| Document | Description |
|---|---|
| [junction_support_labels.md](evo2/junction_support_labels.md) | Why and how to use RNA-seq junction support counts as confidence-weighted per-nucleotide exon labels; connection to adaptive splice site prediction |

---

## `training/` — Training Infrastructure

| Document | Description |
|---|---|
| [deepspeed_training.md](training/deepspeed_training.md) | DeepSpeed ZeRO configurations for LoRA fine-tuning on RunPods (ZeRO-2, ZeRO-3 + CPU offload); `AdaptiveTrainer` wrapper that auto-selects between DeepSpeed and standard PyTorch |

---

## Quick orientation

```
foundation_models/
├── foundation_models/          # Python package
│   ├── evo2/                   # Evo2 model wrappers
│   │   ├── config.py           # Evo2Config (M1 Mac + GPU presets)
│   │   ├── model.py            # Evo2Model (HuggingFace AutoModel wrapper)
│   │   ├── embedder.py         # Evo2Embedder (chunking + HDF5 cache)
│   │   └── classifier.py      # ExonClassifier (Linear / MLP / CNN / LSTM)
│   └── utils/
│       ├── quantization.py     # Device detection, INT8/INT4, memory helpers
│       └── chunking.py         # Sequence chunking, embedding stitching,
│                               # exon label derivation, window generation
├── configs/                    # DeepSpeed JSON configs (to be created)
│   ├── deepspeed_zero2.json
│   └── deepspeed_zero3_offload.json
├── docs/                       # This directory
│   ├── evo2/
│   └── training/
└── examples/
    └── 01_load_evo2_local.py
```
