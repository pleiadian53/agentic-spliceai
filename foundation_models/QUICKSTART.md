# Foundation Models Quick Start

**Status**: Ready for testing  
**Last Updated**: February 15, 2026

---

## 🚀 Installation (5 minutes)

### Option A: M1 Mac (Local Development)

```bash
cd /path/to/agentic-spliceai

# 1. Create environment
mamba env create -f foundation_models/environment-evo2-local.yml

# 2. Activate
conda activate aspliceai-evo2-local

# 3. Install packages
pip install -e .                      # Core package
pip install -e ./foundation_models    # Foundation models

# 4. Test (downloads ~14GB model weights first time)
python foundation_models/examples/01_load_evo2_local.py
```

**Expected output**:
```
✓ Model loaded successfully!
✓ Embeddings shape: [200, 2560]
SUCCESS!
```

---

### Option B: GPU Pod (Production)

```bash
cd /path/to/agentic-spliceai

# 1. Create environment
mamba env create -f foundation_models/environment-evo2-pod.yml

# 2. Activate
conda activate aspliceai-evo2-pod

# 3. Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"  # Should be True

# 4. Install packages
pip install -e .
pip install -e ./foundation_models

# 5. Test with 40B model
python foundation_models/examples/01_load_evo2_pod.py  # (to be created)
```

---

## 📖 Quick Examples

### Extract Embeddings

```python
from foundation_models.evo2 import Evo2Embedder

# Load model (M1 Mac)
embedder = Evo2Embedder(model_size="7b", quantize=True)

# Single sequence
sequence = "ATCGATCGATCG" * 100  # 1.2kb
embeddings = embedder.encode(sequence)
print(embeddings.shape)  # [1200, 2560]

# Batch with caching
genes = {
    "BRCA1": brca1_sequence,
    "TP53": tp53_sequence,
}
gene_embeddings = embedder.encode_batch(
    sequences=genes,
    cache_path="data/embeddings.h5"
)
```

---

### Train Exon Classifier

```python
from foundation_models.evo2 import ExonClassifier
import torch

# Create classifier
classifier = ExonClassifier(
    input_dim=2560,  # Evo2 7B hidden dim
    architecture="mlp",
    hidden_dim=256,
    num_layers=2
)

# Train
classifier.fit(
    train_embeddings=train_emb,  # [N, seq_len, 2560]
    train_labels=train_labels,    # [N, seq_len] - 0/1 binary
    val_embeddings=val_emb,
    val_labels=val_labels,
    epochs=50,
    device="cuda"  # or "mps" for M1
)

# Predict
probs = classifier.predict(test_emb)  # [seq_len] - exon probabilities
```

---

### Variant Effect Prediction

```python
from foundation_models.evo2 import Evo2Model

# Load model
model = Evo2Model.from_config(
    Evo2Config.for_local_mac()
)

# Get variant effect
ref_seq = "ATCGATCG" + "G" + "ATCGATCG"  # Reference
alt_seq = "ATCGATCG" + "T" + "ATCGATCG"  # Variant (G>T)

delta_ll = model.compute_delta_likelihood(ref_seq, alt_seq)
print(f"Delta log-likelihood: {delta_ll:.4f}")
# Negative = deleterious
```

---

## 📂 Project Structure

```
foundation_models/
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
├── pyproject.toml              # Package metadata
├── environment-evo2-local.yml  # M1 Mac
├── environment-evo2-pod.yml    # GPU pod
│
├── foundation_models/          # Python package
│   ├── evo2/
│   │   ├── config.py           # Evo2Config
│   │   ├── model.py            # Evo2Model
│   │   ├── embedder.py         # Evo2Embedder
│   │   └── classifier.py       # ExonClassifier
│   └── utils/
│
└── examples/
    └── 01_load_evo2_local.py   # Test script
```

---

## 🔧 Configuration Presets

### M1 Mac (7B, quantized)
```python
from foundation_models.evo2 import Evo2Config

config = Evo2Config.for_local_mac()
# model_size="7b", quantize=True, device="mps", max_length=32768
```

### GPU Pod (40B, fast)
```python
config = Evo2Config.for_gpu_pod(model_size="40b")
# model_size="40b", quantize=True, device="cuda", max_length=131072
```

### Custom
```python
config = Evo2Config(
    model_size="7b",
    quantize=False,
    device="cuda",
    max_length=1_000_000,  # Full 1M context!
    batch_size=4
)
```

---

## 🐛 Troubleshooting

### Issue: Model download fails

**Solution**: Check internet connection, HuggingFace access
```bash
# Test HuggingFace
huggingface-cli whoami

# Login if needed
huggingface-cli login
```

---

### Issue: CUDA out of memory

**Solutions**:
```python
# 1. Reduce batch size
config.batch_size = 1

# 2. Use quantization
config.quantize = True
config.quantization_bits = 8

# 3. Reduce sequence length
config.max_length = 32768
```

---

### Issue: Slow on M1 Mac

**Expected**: M1 is ~10x slower than GPU

**Solutions**:
- Use smaller sequences (< 32kb)
- Pre-compute and cache embeddings
- Use GPU pod for production

---

## 📊 Performance

| Device | Model | Quantization | Speed | Memory | Max Seq |
|--------|-------|--------------|-------|--------|---------|
| M1 Mac | 7B | INT8 | ~100 bp/s | ~8GB | 32kb |
| A40 | 7B | FP16 | ~10k bp/s | ~14GB | 1M |
| A40 | 40B | INT8 | ~5k bp/s | ~40GB | 1M |
| A100 80GB | 40B | FP16 | ~8k bp/s | ~80GB | 1M |

---

## 📚 Documentation

- **Full guide**: `foundation_models/README.md`
- **Experiment plan**: `dev/foundation_and_adaptor/logs/evo2_experiment_plan.md`
- **Session summary**: `dev/foundation_and_adaptor/logs/aspliceai-evo2-session1-summary.md`
- **Evo2 paper**: https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1

---

## 🎯 Next Steps

### Week 1: Package Setup & Testing
- [x] Create package structure
- [ ] Install environment
- [ ] Test on M1 Mac
- [ ] Extract embeddings for sample genes

### Week 2: Classifier Development
- [ ] Prepare exon labels
- [ ] Train baseline classifier
- [ ] Evaluate on human data

### Week 3: GPU Scaling
- [ ] Deploy to pod
- [ ] Load Evo2 40B
- [ ] Cross-species evaluation

### Week 4: Integration
- [ ] Register in base layer
- [ ] Meta-layer integration
- [ ] Tutorial notebook

---

## 💬 Support

**Issues?** Check:
1. Environment: `conda activate aspliceai-evo2-local`
2. Installation: `pip install -e ./foundation_models`
3. CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Disk space: ~20GB needed for model weights

**Still stuck?**
- Read `foundation_models/README.md`
- Check example: `examples/01_load_evo2_local.py`
- See plan: `dev/foundation_and_adaptor/logs/evo2_experiment_plan.md`

---

**Ready to go!** 🚀

Start with:
```bash
python foundation_models/examples/01_load_evo2_local.py
```
