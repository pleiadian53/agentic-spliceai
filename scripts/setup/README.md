# Setup Scripts

**Utilities for installation verification and environment setup**

## Scripts

### `verify_setup.py`

Comprehensive setup verification script that checks:

**Imports & Dependencies**:
- Required packages (openai, fastapi, uvicorn, etc.)
- Optional packages (torch, transformers, etc.)
- Import paths for agentic_spliceai modules

**Environment**:
- Environment variables (OPENAI_API_KEY, etc.)
- Python version compatibility
- System resources (RAM, CPU)

**Data & Resources**:
- Data directory structure
- Genomic resource files (GTF, FASTA)
- Model weights availability

**Usage**:
```bash
# Run from project root
python scripts/setup/verify_setup.py

# Expected output:
# ============================================================
# Checking Required Packages...
# ============================================================
# ✓ openai (1.x.x) - OpenAI API client
# ✓ fastapi (0.x.x) - FastAPI web framework
# ...
# ============================================================
# ✅ All checks passed!
# ============================================================
```

**What it checks**:
1. **Required packages** (20+ critical dependencies)
2. **Optional packages** (for advanced features)
3. **Environment variables** (API keys, paths)
4. **Data directories** (data/, models/, output/)
5. **Import paths** (can import agentic_spliceai modules?)
6. **System resources** (sufficient RAM/CPU?)

**Exit codes**:
- `0`: All checks passed
- `1`: Some checks failed (details in output)

---

## When to Use

### After Initial Setup
```bash
# 1. Create environment
mamba env create -f environment.yml
mamba activate agentic-spliceai

# 2. Install package
pip install -e .

# 3. Verify everything works
python scripts/setup/verify_setup.py
```

### Before Running Production Code
```bash
# Quick sanity check before important runs
python scripts/setup/verify_setup.py && python examples/base_layer/01_phase1_prediction.py
```

### Debugging Installation Issues
```bash
# See what's missing or misconfigured
python scripts/setup/verify_setup.py

# Fix issues, then re-verify
python scripts/setup/verify_setup.py
```

---

## Related

- **Main setup guide**: [`../../SETUP.md`](../../SETUP.md)
- **Project structure**: [`../../docs/STRUCTURE.md`](../../docs/STRUCTURE.md)
- **Quick start**: [`../../QUICKSTART.md`](../../QUICKSTART.md)

---

## Future Scripts (Planned)

- `setup_genomic_resources.py` - Download GTF/FASTA files
- `setup_models.py` - Download pre-trained model weights
- `setup_environment.py` - Automated environment creation
- `check_compatibility.py` - Check system compatibility (GPU, RAM, etc.)
