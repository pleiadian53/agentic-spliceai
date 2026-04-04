# Conservation Scores for Splice Site Prediction

A tutorial on evolutionary conservation as a feature modality for the
meta-layer. Covers the biology, data formats, build alignment, score
normalization, and integration with the existing feature pipeline.

**Audience**: Developers working on the meta-layer who want to understand
*why* conservation matters and *how* the data fits into our architecture.

**Prerequisites**: Familiarity with examples 01-04 (base_scores, annotation,
genomic modalities) and the feature pipeline abstractions in
`splice_engine/features/`.

---

## 1. Why Conservation Matters for Splice Prediction

### The Biological Signal

Splice sites are under strong **purifying selection** — mutations that
break splicing are usually lethal or cause disease, so evolution eliminates
them. The core splice site motifs are among the most conserved elements
in the human genome:

| Element | Consensus | Conservation |
|---------|-----------|-------------|
| Donor (5' splice site) | **GT**RAGT | GT nearly invariant across all vertebrates |
| Acceptor (3' splice site) | YYYYYYY**AG** | AG nearly invariant; upstream polypyrimidine tract moderately conserved |
| Branch point | YNYURAY | Moderately conserved (~20-50bp upstream of acceptor) |

"Nearly invariant" means PhyloP scores of 5-10+ at these positions —
far above the genome-wide median of ~0.

### What Conservation Tells the Meta-Layer

Conservation provides **orthogonal evidence** to the base model's
sequence-based prediction:

| Scenario | Base Score | Conservation | Interpretation |
|----------|-----------|-------------|----------------|
| True canonical site | High | High | Confirmed — both signals agree |
| False positive | High | Low | Suspicious — sequence looks right but not conserved |
| Alternative splice site | Moderate | Moderate | Interesting — may be lineage-specific or tissue-specific |
| Novel (meta-layer candidate) | Low | Moderate-High | Worth investigating — conserved but base model misses it |
| Random intronic | Low | Low | True negative — neither signal |

The most valuable cases for isoform discovery (Phase 8) are the
**discordant** ones — sites where conservation and base scores disagree.
These are exactly where the meta-layer adds value over the base model
alone.

### Canonical vs Alternative Sites

Not all splice sites are equally conserved:

- **Constitutive splice sites** (used in every transcript): Very high
  conservation (~PhyloP 7-10). GT/AG dinucleotides are essentially fixed.
- **Alternative splice sites** (tissue/condition-specific): Moderate
  conservation (~PhyloP 2-5). The core GT/AG is still conserved, but
  flanking regulatory sequences show more variation.
- **Minor spliceosome sites** (AT-AC introns, ~0.5% of introns): Different
  consensus, still conserved but in different positions.
- **Cryptic splice sites** (disease-activated): Low conservation. These
  are normally silent but become active due to mutations in nearby
  regulatory elements.

This gradient makes conservation a **continuous feature** rather than
binary — exactly what a meta-layer can learn from.

---

## 2. Conservation Scores: PhyloP and PhastCons

Two complementary conservation metrics are available from UCSC, derived
from multi-species whole-genome alignments:

### PhyloP (Phylogenetic P-values)

**What it measures**: Per-base test of evolutionary rate compared to
neutral expectation, computed from a phylogenetic model.

**Score interpretation**:
- **Positive scores** → conserved (slower evolution than neutral)
  - 0-2: Weak conservation (background noise range)
  - 2-5: Moderate conservation (regulatory elements, alternative sites)
  - 5-8: Strong conservation (constitutive splice sites, coding exons)
  - 8-10+: Very strong conservation (near-invariant positions like GT/AG)
- **Negative scores** → accelerated evolution (faster than neutral)
  - -1 to -3: Mildly accelerated
  - < -5: Strong positive selection (rare, biologically interesting)
- **Zero** → evolving at the neutral rate

**Statistical method**: Likelihood ratio test comparing the observed
substitution rate at each base to the neutral rate estimated from
ancestral repeats and 4-fold degenerate sites. Uses the PHAST package
(Siepel et al., 2005).

**Key property**: PhyloP scores are **independent per base** — each
position is tested individually. This makes them noisy but unbiased.
A single highly conserved base surrounded by neutral sequence will
still get a high score.

### PhastCons (Phylogenetic Conservation)

**What it measures**: Posterior probability that each base belongs to
a **conserved element** (a contiguous block of conservation).

**Score interpretation**:
- Range: [0, 1] — a probability
- 0.0: Not in a conserved element
- 0.5: Ambiguous
- 1.0: Definitely in a conserved element

**Statistical method**: Phylogenetic hidden Markov model (phylo-HMM)
with two states: "conserved" and "non-conserved". The model learns
transition probabilities between states and emission probabilities
(substitution rates) for each state. PhastCons scores are the posterior
probability of the "conserved" state at each position.

**Key property**: PhastCons scores are **smoothed across neighboring
bases** via the HMM. A single conserved base in a neutral region gets
a LOW score (the HMM doesn't transition to "conserved" for one base).
But a block of moderately conserved bases gets HIGH scores — even if
no individual base is remarkable.

### PhyloP vs PhastCons: Complementary Signals

| Property | PhyloP | PhastCons |
|----------|--------|-----------|
| Unit | Log-likelihood ratio | Probability [0,1] |
| Spatial | Per-base (independent) | Smoothed (HMM) |
| Direction | Signed (conservation + acceleration) | Unsigned (conservation only) |
| Best for | Individual critical bases (GT/AG) | Conserved blocks (exons, regulatory) |
| Noise level | Higher (per-base) | Lower (smoothed) |

**For splice sites**, both are informative:
- **PhyloP** captures the sharp conservation peak at GT/AG dinucleotides
- **PhastCons** captures the broader conserved block around splice sites
  (extended consensus, exon body conservation)

Using both together gives the meta-layer two "views" of conservation —
one zoomed in (PhyloP), one zoomed out (PhastCons).

---

## 3. Multi-Species Alignments: 100-way vs 46-way

The conservation scores are derived from whole-genome alignments of
multiple vertebrate species. The number of species affects statistical
power:

### Available Alignments

| Build | Alignment | Species | UCSC Track |
|-------|----------|---------|------------|
| GRCh38 (hg38) | 100-way vertebrate | 100 | `phyloP100way`, `phastCons100way` |
| GRCh37 (hg19) | 46-way vertebrate | 46 | `phyloP46way`, `phastCons46way` |

The **100-way alignment** (hg38) was computed natively on GRCh38. It
includes more species (especially additional mammals and birds), giving:
- Finer resolution between "moderately" and "weakly" conserved
- Better power to detect conservation at alternative splice sites
- Wider dynamic range for PhyloP scores

The **46-way alignment** (hg19) is the standard for GRCh37. It has
fewer species but is still highly informative for canonical splice sites
(GT/AG conservation is detectable even with 10 species).

### Impact on Score Distributions

Because more species provide more statistical power, the raw score
distributions differ between 100-way and 46-way:

```
PhyloP score distributions (conceptual):

100-way:  ▁▁▂▃▅▇█▇▅▃▂▁▁   wider range, sharper peaks
           -5        0        5       10

46-way:   ▁▂▃▅▇██▇▅▃▂▁     narrower range, less resolution
           -5        0        5       10
```

Conserved sites get *higher* PhyloP scores in 100-way (more evidence
for conservation), while neutral sites remain near zero in both. This
means raw scores are **not directly comparable** across alignments.

### Cross-Build Normalization Strategy

When training a meta-model that might see features from both builds
(e.g., comparing SpliceAI vs OpenSpliceAI predictions), we need the
conservation features to be comparable. Three approaches:

#### Option A: Percentile Rank Normalization (Recommended)

Convert raw scores to genome-wide percentile ranks within each build:

```python
# Conceptual (not actual implementation)
phylop_percentile = genome_wide_rank(phylop_raw) / total_positions
# Result: 0.0 = least conserved, 1.0 = most conserved
```

**Advantages**:
- Directly comparable across builds (99th percentile = "more conserved
  than 99% of the genome" regardless of alignment depth)
- Handles the power difference naturally
- Robust to outliers
- Preserves ordinal relationships

**Disadvantage**: Requires a genome-wide percentile lookup table for
each build (precomputable, ~300 MB per build).

**Implementation**: Precompute percentile bins from a genome-wide sample
of conservation scores, store as a lookup array, apply during feature
extraction. The modality config would include a flag:
`normalize: "percentile" | "raw" | "zscore"`.

#### Option B: Z-Score Normalization

Standardize to mean=0, std=1 using genome-wide statistics:

```python
phylop_z = (phylop_raw - genome_mean) / genome_std
```

**Advantages**: Simple, well-understood.
**Disadvantage**: Conservation scores are heavily skewed (most of the
genome is non-conserved), so z-scores don't map to intuitive percentiles.
A z-score of 3.0 means very different things in 100-way vs 46-way.

#### Option C: Build-Matched Tracks (No Normalization)

Use the correct track for each build and don't normalize. Let the
meta-model learn build-specific score distributions.

**Advantages**: No information loss; simplest implementation.
**Disadvantage**: The meta-model must be build-aware, or trained
separately per build.

#### Recommendation

**For single-build training** (the common case): Option C is fine.
Each meta-model trains on one build's features, and the conservation
scores are internally consistent.

**For cross-build comparison or unified models**: Option A (percentile
ranks) is the best balance of simplicity and correctness. It can be
added as an optional normalization step in the conservation modality
config.

---

## 4. Data Format and Access

### BigWig Format

Conservation scores are distributed as **bigWig** files — a compressed,
indexed binary format optimized for genomic signal tracks. Key properties:

- **Random access**: Query any genomic interval in O(log n) time
- **Compressed**: ~2 GB per genome-wide track (vs ~20 GB uncompressed)
- **Streaming**: pyBigWig can open remote HTTP URLs and fetch only the
  requested regions via range requests — no full download needed

### UCSC BigWig URLs

```python
# GRCh38 (hg38) — 100-way vertebrate alignment
CONSERVATION_URLS_HG38 = {
    "phylop": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
    "phastcons": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
}

# GRCh37 (hg19) — 46-way vertebrate alignment
# Note: hg19 uses subtree-named files (vertebrate.*), not hg19-prefixed
CONSERVATION_URLS_HG19 = {
    "phylop": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP46way/vertebrate.phyloP46way.bw",
    "phastcons": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phastCons46way/vertebrate.phastCons46way.bw",
}
```

### pyBigWig Access Pattern

```python
import pyBigWig

# Open remote file (streams on demand, caches header)
bw = pyBigWig.open("https://hgdownload.soe.ucsc.edu/.../hg38.phyloP100way.bw")

# Query a single position (returns list of 1 value)
scores = bw.values("chr17", 7579312, 7579313)  # [5.432]

# Query a window (returns list of values, one per base)
scores = bw.values("chr17", 7579300, 7579325)  # [0.1, 0.3, ..., 5.4, ...]

# Query statistics over an interval
mean = bw.stats("chr17", 7579300, 7579325, type="mean")  # [2.87]

bw.close()
```

**Performance considerations**:
- **Remote access**: ~50-200ms per query due to HTTP round-trips.
  Batching queries by chromosome region is essential.
- **Local files**: ~0.1ms per query. For genome-scale workflows, download
  the bigWig files locally first (~2 GB each).
- **Batch strategy**: For a chromosome, query the entire span once
  (`bw.values(chrom, 0, chrom_length)`) rather than per-position.
  This is 1 HTTP request vs thousands.

### Local Caching Strategy

For genome-scale pipelines (script 04), remote access is too slow.
The conservation modality should support local caching:

```
data/conservation/
├── hg38/
│   ├── hg38.phyloP100way.bw      # ~9.5 GB
│   └── hg38.phastCons100way.bw   # ~4.5 GB
└── hg19/
    ├── vertebrate.phyloP46way.bw       # ~7.5 GB
    └── vertebrate.phastCons46way.bw    # ~3.5 GB
```

The modality config would accept either a local path or fall back to
remote URLs. The resource manager can resolve the correct path based
on `base_model → build → track`.

---

## 5. Build Alignment: Matching Conservation to Base Model

### The Coordinate Problem

Our feature pipeline runs in **one coordinate system per execution**,
determined by the base model:

| Base Model | Annotation | Build | Coordinates |
|------------|-----------|-------|-------------|
| SpliceAI | Ensembl | GRCh37 | `17:7579312` (no chr prefix) |
| OpenSpliceAI | MANE | GRCh38 | `chr17:7579312` (chr prefix) |

The same genomic position (e.g., TP53 exon 7 donor site) has **different
coordinates** in GRCh37 vs GRCh38 due to:
- Sequence patches and gap closures between builds
- Centromere model changes
- Alt loci additions in GRCh38

Querying hg38 conservation bigWig with GRCh37 coordinates would return
scores for the **wrong nucleotides**.

### Solution: Build-Matched Resource Resolution

The conservation modality follows the same pattern as `annotation` and
`sequence` modalities — it accepts `base_model` in its config and
resolves the correct data source:

```
base_model="openspliceai"
    → build="GRCh38"
    → track="hg38.phyloP100way.bw"
    → query: bw.values("chr17", 7579312, 7579313)  ✓ Correct

base_model="spliceai"
    → build="GRCh37"
    → track="hg19.phyloP46way.bw"
    → query: bw.values("17", 7579312, 7579313)  ✓ Correct
```

No liftover is needed because conservation tracks exist natively in
both builds.

### Chromosome Name Handling

BigWig files use the UCSC naming convention:
- hg38: `chr1`, `chr2`, ..., `chrX`, `chrY`
- hg19: `chr1`, `chr2`, ..., `chrX`, `chrY`

Note that hg19 bigWig files also use `chr` prefix — this is UCSC
convention, independent of Ensembl's bare-number convention. So when
using SpliceAI (GRCh37/Ensembl, bare numbers), the conservation
modality must **add the `chr` prefix** when querying the bigWig:

```python
# SpliceAI feature row: chrom="17", position=7579312
# BigWig query needs: chrom="chr17"
query_chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
scores = bw.values(query_chrom, position - window, position + window + 1)
```

This is a boundary mapping internal to the conservation modality — it
does not affect the output DataFrame, which retains the original `chrom`
format for consistency with other modalities.

---

## 6. Feature Design for the Meta-Layer

### Output Columns

The conservation modality would produce these features per position:

| Column | Source | Description |
|--------|--------|-------------|
| `phylop_score` | PhyloP | Raw score at the position (or mean over small window) |
| `phastcons_score` | PhastCons | Probability of being in a conserved element |
| `phylop_context_mean` | PhyloP | Mean score in ±window around position |
| `phylop_context_max` | PhyloP | Max score in ±window (captures nearby peaks) |
| `phylop_context_std` | PhyloP | Score variability in window (sharp peak vs broad) |
| `phastcons_context_mean` | PhastCons | Mean probability in ±window |
| `conservation_contrast` | Both | phylop_score - phylop_context_mean (does this position stand out?) |

**Optional extended features** (configurable):
- `phylop_percentile`: Genome-wide percentile rank (if normalization enabled)
- `phylop_upstream_mean`, `phylop_downstream_mean`: Asymmetric context
  (splice sites have different conservation patterns on exon vs intron side)

### Window Size Considerations

The optimal window depends on the genomic element:

| Window | Captures | Use case |
|--------|----------|----------|
| 1 bp | Core dinucleotide (GT/AG) | Sharpest signal, most noise |
| 5-10 bp | Extended splice consensus | Good default for donor (6bp) |
| 20-30 bp | Polypyrimidine tract, branch point | Acceptor context |
| 50-100 bp | Exon/intron boundary region | Broader regulatory context |

For the initial implementation, a default window of 10 bp with
configurable override is reasonable. The meta-model can learn which
scale matters most.

### Interaction with Other Modalities

Conservation complements the existing modalities:

```
Position: chr17:7,579,312 (TP53 exon 7 donor)

base_scores:   donor_prob=0.95   → Sequence says "splice site"
annotation:    splice_type=donor → Ground truth confirms
genomic:       rel_position=0.42 → Mid-gene (expected for internal exon)
conservation:  phylop=8.2        → Highly conserved (expected for constitutive site)
               phastcons=0.99    → In a conserved block (exon boundary)

→ All modalities agree → High confidence
```

```
Position: chr7:117,559,590 (CFTR alternative exon 9)

base_scores:   donor_prob=0.35   → Weak sequence signal
annotation:    splice_type=donor → Known alternative site
genomic:       rel_position=0.28 → Early in gene
conservation:  phylop=3.1        → Moderate conservation
               phastcons=0.45    → Edge of conserved block

→ Moderate agreement → Alternative site (tissue-specific regulation)
→ Meta-model learns that moderate conservation + weak base score
  can still be a real site
```

```
Position: chr11:5,248,050 (HBB cryptic splice site activated by sickle cell variant)

base_scores:   donor_prob=0.02   → Base model says no
annotation:    splice_type=''    → Not in annotation
genomic:       rel_position=0.55 → Mid-gene
conservation:  phylop=0.1        → Not conserved
               phastcons=0.02    → Not in a conserved element

→ All signals say no → But with the sickle cell variant (HBB:c.20A>T),
  this site becomes active. This is where variant analysis (Phase 6)
  and RNA-seq evidence (Phase 5B) become essential.
```

---

## 7. Implementation Plan

### Modality Class Skeleton

```python
# splice_engine/features/modalities/conservation.py

@dataclass(frozen=True)
class ConservationConfig(ModalityConfig):
    """Configuration for conservation score features."""
    base_model: str = "openspliceai"
    tracks: tuple[str, ...] = ("phylop", "phastcons")
    window: int = 10          # half-window for context features
    normalize: str = "raw"    # "raw" | "percentile" | "zscore"
    cache_dir: Optional[Path] = None  # local bigWig cache
    remote_fallback: bool = True      # use UCSC URLs if local not found

class ConservationModality(Modality):
    """Evolutionary conservation features from PhyloP and PhastCons."""

    def meta(self) -> ModalityMeta:
        return ModalityMeta(
            name="conservation",
            version="0.1.0",
            output_columns=[...],        # see feature table above
            required_inputs=["chrom", "position"],
            optional_inputs=[],
            description="Evolutionary conservation from multi-species alignment",
        )

    def validate(self, available_columns: set[str]) -> bool: ...
    def transform(self, df: pl.DataFrame) -> pl.DataFrame: ...
```

### Build → Track Resolution

```python
TRACK_REGISTRY = {
    "GRCh38": {
        "phylop": {
            "alignment": "100way",
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
            "local_filename": "hg38.phyloP100way.bw",
            "chr_prefix": True,       # bigWig uses chr prefix
        },
        "phastcons": {
            "alignment": "100way",
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
            "local_filename": "hg38.phastCons100way.bw",
            "chr_prefix": True,
        },
    },
    "GRCh37": {
        "phylop": {
            "alignment": "46way",
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP46way/vertebrate.phyloP46way.bw",
            "local_filename": "vertebrate.phyloP46way.bw",
            "chr_prefix": True,       # hg19 bigWig ALSO uses chr prefix
        },
        "phastcons": {
            "alignment": "46way",
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phastCons46way/vertebrate.phastCons46way.bw",
            "local_filename": "vertebrate.phastCons46way.bw",
            "chr_prefix": True,
        },
    },
}
```

### Transform Strategy

For genome-scale efficiency, the conservation modality should:

1. **Batch by chromosome**: Open bigWig once, query all positions on
   that chromosome, close. Avoid per-position file opens.
2. **Vectorize with numpy**: `bw.values()` returns a list; convert to
   numpy array once, then compute all windowed statistics vectorially.
3. **Lazy chromosome loading**: Only open the bigWig when `transform()`
   is called, not at construction time.
4. **Local-first**: Check `cache_dir` for local bigWig before falling
   back to remote URL.

### Integration Checklist

- [ ] Implement `ConservationModality` in `modalities/conservation.py`
- [ ] Register in `modalities/__init__.py`
- [ ] Add `pyBigWig` as optional dependency in `pyproject.toml`
- [ ] Add build-specific URLs to `settings.yaml` or track registry
- [ ] Update example 02 and 04 to include conservation (if local files present)
- [ ] Add example 06 showing conservation + base_scores pipeline
- [ ] Unit tests with mock bigWig data

---

## 8. References

- Siepel A, et al. (2005). "Evolutionarily conserved elements in
  vertebrate, insect, worm, and yeast genomes." *Genome Res* 15:1034-1050.
  (PhastCons method)
- Pollard KS, et al. (2010). "Detection of nonneutral substitution rates
  on mammalian phylogenies." *Genome Res* 20:110-121. (PhyloP method)
- UCSC Genome Browser conservation tracks:
  https://genome.ucsc.edu/cgi-bin/hgTrackUi?g=cons100way
- pyBigWig documentation:
  https://github.com/deeptools/pyBigWig

---

## 9. Appendix: Expected Conservation at TP53 Splice Sites

Example output from `05_multimodal_exploration.py` (TP53, OpenSpliceAI):

```
PhyloP (100-way):
    Splice sites       mean=5.2    median=6.1    std=3.4
    Random intronic     mean=0.3    median=0.1    std=1.2
    Splice/Random ratio: ~17x

PhastCons (100-way):
    Splice sites       mean=0.82   median=0.95   std=0.25
    Random intronic     mean=0.15   median=0.03   std=0.28
    Splice/Random ratio: ~5.5x
```

These numbers illustrate the strong conservation signal at canonical
splice sites. The meta-layer will learn to use this contrast — sites
with high base scores but low conservation warrant skepticism, while
sites with moderate base scores but high conservation deserve a second
look.
