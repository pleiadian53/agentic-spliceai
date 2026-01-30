# Notebooks: Educational & Illustrative

**Purpose**: Jupyter notebooks with step-by-step explanations, visualizations, and deep dives

**Organization**: Topic-specific directories matching the multi-layer architecture

---

## 📁 Directory Structure

```
notebooks/
├── README.md                          ← This file
│
├── base_layer/                        ← Base model prediction tutorials
│   ├── 01_phase1_basics/
│   │   ├── 01_phase1_basics.ipynb    ← Phase 1 fundamentals
│   │   └── README.md
│   └── README.md
│
├── data_preparation/                  ← Data preparation tutorials
│   ├── 01_gene_extraction/
│   │   ├── 01_gene_extraction.ipynb  ← Gene & sequence extraction
│   │   └── README.md
│   ├── 02_splice_site_extraction/
│   │   ├── 02_splice_site_extraction.ipynb  ← Splice site annotation
│   │   └── README.md
│   └── README.md
│
├── meta_layer/                        ← Meta-layer tutorials (Phase 5)
│   └── README.md
│
└── variant_analysis/                  ← Variant analysis tutorials (Phase 6)
    └── README.md
```

---

## 🎓 Learning Paths

### Getting Started Path

**For new users learning the system**:

1. **Base Layer Basics**:
   - `base_layer/01_phase1_basics/01_phase1_basics.ipynb`
   - Learn: What is splice site prediction? How does the base layer work?

2. **Data Preparation**:
   - `data_preparation/01_gene_extraction/01_gene_extraction.ipynb`
   - `data_preparation/02_splice_site_extraction/02_splice_site_extraction.ipynb`
   - Learn: How to prepare genomic data from GTF/FASTA files

3. **Advanced Topics** (coming soon):
   - Meta-layer adaptive prediction
   - Variant analysis workflows
   - Isoform discovery

---

### Developer Path

**For contributors and developers**:

1. **Architecture Overview**:
   - `base_layer/01_phase1_basics/01_phase1_basics.ipynb`
   - Understand: System architecture, data flow

2. **Implementation Details**:
   - Explore notebooks with code deep-dives
   - See how components integrate

3. **Examples**:
   - See `../examples/` for quick driver scripts
   - Use notebooks for understanding, examples for iteration

---

## 📚 Notebooks vs Examples vs Scripts

### Notebooks (This Directory)

**Purpose**: Education, explanation, visualization

**Characteristics**:
- Jupyter notebooks (.ipynb)
- Step-by-step explanations
- Inline visualizations
- Detailed documentation
- Markdown supplements (theory, QA)

**Target Audience**: New users, students, learners

**Examples**:
- `01_phase1_basics.ipynb` - Learn phase 1 with visuals
- `02_splice_site_extraction.ipynb` - Understand splice site biology

---

### Examples (`../examples/`)

**Purpose**: Quick testing, development iteration

**Characteristics**:
- Standalone Python scripts
- Command-line arguments
- Fast execution
- Console output

**Target Audience**: Developers, researchers

**Examples**:
- `examples/base_layer/01_phase1_prediction.py` - Test predictions
- `examples/data_preparation/validate_mane_metadata.py` - Quick validation

---

### Scripts (`../scripts/`)

**Purpose**: Production utilities, batch processing

**Characteristics**:
- Complex pipelines
- Production-ready
- Robust error handling
- Logging

**Target Audience**: Production users, automation

---

## 🎯 Current Status (Phase 2 Complete)

### Available Now ✅

- **Base Layer**: Phase 1 basics coming soon
- **Data Preparation**: Coming soon

### Coming Soon 🔜

- **Meta Layer**: After Phase 5 implementation
- **Variant Analysis**: After Phase 6 implementation
- **Isoform Discovery**: After Phase 8 implementation

---

## 📝 Contributing Notebooks

### Guidelines

1. **Naming**: Use numbered prefixes (e.g., `01_`, `02_`)
2. **Organization**: Create subdirectory for each topic
3. **Documentation**: 
   - Add notebook with `.ipynb` extension
   - Add README.md explaining the topic
   - Add supplemental `.md` files for theory
4. **Style**:
   - Clear explanations before code cells
   - Visualizations after results
   - Section headers for organization
5. **Test**: Ensure notebook runs top-to-bottom

### Example Structure

```
notebooks/topic_name/
├── 01_topic_basics/
│   ├── 01_topic_basics.ipynb         # Main notebook
│   ├── README.md                     # Topic overview
│   └── supplements/                  # Optional theory
│       ├── 01_background.md
│       └── 02_mathematical_details.md
```

---

## 🔗 Related Resources

- **Examples**: `../examples/` - Quick driver scripts
- **Scripts**: `../scripts/` - Production utilities
- **Tests**: `../tests/` - Integration tests
- **Docs**: `../docs/` - User documentation

---

**Last Updated**: January 30, 2026  
**Status**: Structure created, notebooks coming soon
