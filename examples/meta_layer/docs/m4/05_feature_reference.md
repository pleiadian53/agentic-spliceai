# M4 Tutorial 5: Feature Reference (All 229 Columns)

[Part 1](01_training_data.md) explains how one training row is built and shows a handful of its values. This part is the lookup table behind it: every column the benchmark feeds a model, with its exact name, how it is calculated, its units, which site it describes, where its data come from, what happens when the data are missing, and its value in one real row.

The running example is the row from part 1: the **ZNF706 A3SS event under AATF knockdown**, whose label is $y = +0.756$ (the long form gained 0.756 of usage).

---

## 1. Reading a column name

Every column name says which part of the event it describes.

| Pattern | Describes | Count |
|---|---|---:|
| no prefix, e.g. `alt_dist` | the event as a whole, or the regulator's eCLIP availability | 11 |
| `partner__X` | feature `X` measured at the **partner** site | 100 |
| `delta__X` | feature `X` at the **long** site minus `X` at the **short** site | 100 |
| `R_long__Y`, `R_short__Y`, `R_partner__Y` | where the knocked-down regulator binds near that site | 18 |
| | **total** | **229** |

The three sites of the example (chromosome 8, minus strand, so the RNA runs toward lower coordinates):

```text
RNA direction  ←──────────────────────────────────────────────────────
          short acceptor   long acceptor                partner donor
            101,202,447     101,202,452    ...intron...   101,205,435
                 └── 5 nt ──┘
```

The long acceptor comes first in the RNA (it has the higher coordinate on the minus strand), so choosing it keeps 5 extra exonic bases: the long form. Both acceptors compete for reads from the same partner donor.

**Why there is no `long__X` or `short__X`.** The benchmark's default *compact* set keeps only their difference, `delta__X`, because the label measures the competition between the two. The *full* set (`--feature-set full`) adds both back: 429 columns.

**Every site gets both donor and acceptor features.** `donor_score` is computed at an acceptor too (and is near 0 there); no feature looks at the site's type.

## 2. How missing values reach a model

The benchmark converts each table to a NumPy array, so a null becomes `NaN`. Nothing is imputed: XGBoost learns, at each split, which branch `NaN` should follow, and TabPFN v2 accepts `NaN` directly.

Most missing data never reach that stage as `NaN`, because the feature pipeline fills them with 0 when it builds the site table. Three consequences:

- **0 can mean "measured, nothing there" or "no data".** A site with no GTEx junction has `junction_log1p = 0`; a phyloP alignment gap reads as `phylop_score = 0`; no eCLIP peak gives `rbp_n_bound = 0`.
- **Only two site features are ever null:** `junction_psi` and `junction_psi_variance`, when no tissue has two competing junctions at the site.
- **A delta is null if either side is null.** That is why `delta__junction_psi` is null in 90% of rows.

The columns that are ever null, as a share of the 1,585,467 training rows:

| Column(s) | Null in | Why |
|---|---:|---|
| `R_*` (all 18 binding columns) | 43.7% | the knocked-down regulator has no eCLIP at all (81 of 185 regulators) |
| `delta__junction_psi`, `delta__junction_psi_variance` | 90.0%, 90.0% | null at the long or the short site |
| `partner__junction_psi`, `partner__junction_psi_variance` | 10.2%, 10.3% | null at the partner |
| `baseline_psi_loo` | 0.2% | the event was measured only within one control batch |

The other 206 columns are never null.

## 3. Event-level columns (11)

Source: `kd_events.parquet` (rMATS event coordinates) and the knockdown labels of *other* experiments; eCLIP flags from `regulator_site_features.parquet`. Code: `geometry_features`, `load_events` and `regulator_features` in `examples/meta_layer/m4/_m4_common.py`.

| Column | Meaning | Calculation | Units | Missing | Example |
|---|---|---|---|---|---:|
| `is_a5ss` | event type | 1 for A5SS (competing donors), 0 for A3SS (competing acceptors) | 0 or 1 | never | 0 |
| `alt_dist` | distance between the competing sites | \|long site − short site\| | nucleotides | never | 5 |
| `junction_span_long` | intron length if the long site is used | \|partner − long site\| | nucleotides | never | 2,983 |
| `junction_span_short` | intron length if the short site is used | \|partner − short site\| | nucleotides | never | 2,988 |
| `long_exon_len` | length of the alternative exon in its long form | end − start + 1 (rMATS coordinates) | nucleotides | never | 120 |
| `short_exon_len` | same, short form | end − start + 1 | nucleotides | never | 115 |
| `flank_exon_len` | length of the partner's exon | end − start + 1 | nucleotides | never | 298 |
| `baseline_psi_loo` | how often the long form is used normally | mean control PSI of this event over all knockdown experiments **outside this row's control batch** | fraction 0–1 | 0.2% | 0.460 |
| `n_other_kd` | how many other knockdowns measured this event | rows of the same event − 1 | count | never | 61 |
| `has_eclip` | the knocked-down regulator has eCLIP in any cell line | 1 unless its eCLIP source is "none" | 0 or 1 | never | 1 |
| `eclip_hepg2` | that eCLIP is from HepG2, the knockdown cell line | 1 if the source is HepG2 | 0 or 1 | never | 0 |

Reading the example: an A3SS event whose acceptors are 5 nt apart, so the two possible introns differ by exactly 5 nt (2,983 vs 2,988) and the two exon forms by 5 nt (120 vs 115). Normally the long form is used 46% of the time, and 61 other knockdowns measured the event. AATF has eCLIP, but from K562 (`eclip_hepg2 = 0`), so its binding features are cross-cell evidence.

`baseline_psi_loo` leaves out the whole control batch because knockdowns in a batch share control samples, and a row's own control PSI shares sampling noise with its label ([part 1, section 5](01_training_data.md#5-columns-that-must-never-enter-x)).

## 4. Regulator-binding columns (18)

Where the knocked-down regulator itself binds near each site, from ENCODE eCLIP peaks (`examples/data_preparation/m4/04_build_regulator_features.py`). HepG2 peaks are used when the regulator has them, otherwise K562, otherwise all 18 columns are null. Six measurements, each at three sites:

| Measurement `Y` | Meaning | Calculation | Units | `R_long__Y` | `R_short__Y` | `R_partner__Y` |
|---|---|---|---|---:|---:|---:|
| `bind_w50` | peaks very close to the site | number of the regulator's peaks overlapping site ± 50 nt | count | 0 | 0 | 0 |
| `bind_w300` | peaks nearby | overlapping site ± 300 nt | count | 0 | 0 | 0 |
| `bind_w1000` | peaks in the neighbourhood | overlapping site ± 1,000 nt | count | 0 | 0 | 0 |
| `bind_intron300` | peaks on the intron side | overlapping the 300 nt on the intron side (lower coordinates for a plus-strand acceptor or minus-strand donor, higher otherwise) | count | 0 | 0 | 0 |
| `bind_exon300` | peaks on the exon side | overlapping the 300 nt on the exon side | count | 0 | 0 | 0 |
| `dist` | distance to the nearest peak | 0 if a peak overlaps the site; capped at 10,000 | nucleotides | 10,000 | 10,000 | 10,000 |

These are counts of peak intervals, not read depth. In the example AATF has no peak within 10 kb of any of the three sites: all counts are 0 and every `dist` sits at the cap. That is a measurement ("no binding seen"), unlike the null of a regulator with no eCLIP ("not measured"); `has_eclip` tells the model which case it is in.

## 5. Site features (100, each used twice)

Every event site has the meta layer's 100-column multimodal vector (`examples/data_preparation/m4/03_build_site_features.py`, configuration `examples/features/configs/full_stack.yaml`). The compact set uses each feature `X` twice: as `partner__X`, and as `delta__X` = long − short. The tables give all four numbers for the example, so each delta can be checked by subtraction.

Two conventions hold throughout:

- **Neighbours are in genomic order.** `m1`/`m2` are the lower coordinates and `p1`/`p2` the higher ones, whatever the strand. On this minus-strand event, `m1` lies *downstream* in the RNA, and `context_asymmetry` changes sign relative to a plus-strand site.
- **Track values are read one base to the right.** Conservation, epigenetic and accessibility values, and eCLIP overlap, are read at genomic position p + 1, not p (a known off-by-one in the shared feature pipeline, see section 6). Base scores, junctions and genomic position use p itself.

"Splice score" `S[k]` below means max(donor, acceptor probability) at genomic offset `k` from the site; `d`, `a`, `n` are the donor, acceptor and neither probabilities at the site.

### 5.1 Base scores (43)

**Source.** OpenSpliceAI (GRCh38, trained on MANE), averaged over its checkpoints. MANE sites read the stored per-nucleotide predictions; other sites are predicted on demand from sequence.

**Missing values.** Never null. A neighbour outside the gene or the prediction span is filled with 0. Every division and log adds ε = 1e-10.

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `donor_score` | Donor probability at the site | d | probability 0–1 | 0.757 | 4.5e−8 | 1.6e−7 | −1.2e−7 |
| `acceptor_score` | Acceptor probability at the site | a | probability 0–1 | 6.2e−7 | 3.5e−5 | 1.1e−5 | 2.3e−5 |
| `neither_score` | Probability the site is not a splice site | n | probability 0–1 | 0.243 | 1.000 | 1.000 | −2.3e−5 |
| `context_score_m2` | Splice score two bases to the left | S[−2] | probability 0–1 | 7.9e−6 | 4.8e−8 | 4.8e−9 | 4.3e−8 |
| `context_score_m1` | Splice score one base to the left | S[−1] | probability 0–1 | 2.3e−5 | 2.0e−9 | 2.8e−9 | −7.8e−10 |
| `context_score_p1` | Splice score one base to the right | S[+1] | probability 0–1 | 1.5e−5 | 1.4e−9 | 1.8e−8 | −1.6e−8 |
| `context_score_p2` | Splice score two bases to the right | S[+2] | probability 0–1 | 1.3e−5 | 7.4e−8 | 2.0e−9 | 7.2e−8 |
| `relative_donor_probability` | Donor share of the two splice classes | d / (d + a) | 0–1 | 1.000 | 0.001 | 0.014 | −0.013 |
| `splice_probability` | Splice share of all three classes | (d + a) / (d + a + n) | 0–1 | 0.757 | 3.5e−5 | 1.1e−5 | 2.3e−5 |
| `donor_acceptor_diff` | Donor minus acceptor, normalised | (d − a) / max(d, a) | −1 to 1 | 1.000 | −0.999 | −0.986 | −0.013 |
| `splice_neither_diff` | Splice minus neither, normalised | (max(d, a) − n) / max(d, a, n) | −1 to 1 | 0.680 | −1.000 | −1.000 | 2.3e−5 |
| `donor_acceptor_logodds` | Log-odds of donor versus acceptor | ln d − ln a | nats (about ±23) | 14.021 | −6.636 | −4.249 | −2.387 |
| `splice_neither_logodds` | Log-odds of splice versus neither | ln(d + a) − ln n | nats | 1.138 | −10.272 | −11.377 | 1.105 |
| `probability_entropy` | How undecided the three-class output is | −Σ p ln p over d, a, n | nats, 0 to ln 3 | 0.554 | 3.9e−4 | 1.4e−4 | 2.5e−4 |
| `context_neighbor_mean` | Average splice score of the four neighbours | mean of S[−2], S[−1], S[+1], S[+2] | 0–1 | 1.5e−5 | 3.1e−8 | 6.9e−9 | 2.5e−8 |
| `context_asymmetry` | Left neighbours minus right neighbours | (S[−1] + S[−2]) − (S[+1] + S[+2]) | −2 to 2 | 2.2e−6 | −2.5e−8 | −1.2e−8 | −1.3e−8 |
| `context_max` | Highest neighbour splice score | max of the four neighbours | 0–1 | 2.3e−5 | 7.4e−8 | 1.8e−8 | 5.6e−8 |
| `donor_diff_m1` | Drop from the donor score to the left neighbour | d − S[−1] | −1 to 1 | 0.757 | 4.3e−8 | 1.6e−7 | −1.2e−7 |
| `donor_diff_m2` | Same, two bases left | d − S[−2] | −1 to 1 | 0.757 | −3.1e−9 | 1.6e−7 | −1.6e−7 |
| `donor_diff_p1` | Same, one base right | d − S[+1] | −1 to 1 | 0.757 | 4.4e−8 | 1.4e−7 | −1.0e−7 |
| `donor_diff_p2` | Same, two bases right | d − S[+2] | −1 to 1 | 0.757 | −2.9e−8 | 1.6e−7 | −1.9e−7 |
| `donor_surge_ratio` | Donor score relative to its two adjacent neighbours | d / (S[−1] + S[+1]) | ≥ 0, unbounded | 20,154 | 12.889 | 7.786 | 5.103 |
| `donor_is_local_peak` | 1 if the donor score is a local maximum | d > S[−1] and d > S[+1] and d > 0.001 | 0 or 1 | 1 | 0 | 0 | 0 |
| `donor_weighted_context` | Smoothed donor peak | 0.4·d + 0.2·(S[−1] + S[+1]) + 0.1·(S[−2] + S[+2]) | 0–1 | 0.303 | 3.1e−8 | 6.9e−8 | −3.8e−8 |
| `donor_peak_height_ratio` | Donor score over the neighbour mean | d / context_neighbor_mean | ≥ 0, unbounded | 51,626 | 1.432 | 23.153 | −21.722 |
| `donor_second_derivative` | Curvature of the donor peak | 2·d − S[−1] − S[+1] | −2 to 2 | 1.515 | 8.7e−8 | 3.0e−7 | −2.1e−7 |
| `donor_signal_strength` | Donor score above the neighbour background | d − context_neighbor_mean | −1 to 1 | 0.757 | 1.4e−8 | 1.5e−7 | −1.4e−7 |
| `donor_context_diff_ratio` | Donor score over the highest neighbour | d / context_max | ≥ 0, unbounded | 33,569 | 0.608 | 9.019 | −8.410 |
| `acceptor_diff_m1` | Drop from the acceptor score to the left neighbour | a − S[−1] | −1 to 1 | −2.2e−5 | 3.5e−5 | 1.1e−5 | 2.3e−5 |
| `acceptor_diff_m2` | Same, two bases left | a − S[−2] | −1 to 1 | −7.2e−6 | 3.4e−5 | 1.1e−5 | 2.3e−5 |
| `acceptor_diff_p1` | Same, one base right | a − S[+1] | −1 to 1 | −1.4e−5 | 3.5e−5 | 1.1e−5 | 2.3e−5 |
| `acceptor_diff_p2` | Same, two bases right | a − S[+2] | −1 to 1 | −1.3e−5 | 3.4e−5 | 1.1e−5 | 2.3e−5 |
| `acceptor_surge_ratio` | Acceptor score relative to its two adjacent neighbours | a / (S[−1] + S[+1]) | ≥ 0, unbounded | 0.016 | 9,848 | 545.607 | 9,302 |
| `acceptor_is_local_peak` | 1 if the acceptor score is a local maximum | a > S[−1] and a > S[+1] and a > 0.001 | 0 or 1 | 0 | 0 | 0 | 0 |
| `acceptor_weighted_context` | Smoothed acceptor peak | 0.4·a + 0.2·(S[−1] + S[+1]) + 0.1·(S[−2] + S[+2]) | 0–1 | 9.9e−6 | 1.4e−5 | 4.5e−6 | 9.3e−6 |
| `acceptor_peak_height_ratio` | Acceptor score over the neighbour mean | a / context_neighbor_mean | ≥ 0, unbounded | 0.042 | 1,094 | 1,623 | −528.770 |
| `acceptor_second_derivative` | Curvature of the acceptor peak | 2·a − S[−1] − S[+1] | −2 to 2 | −3.6e−5 | 6.9e−5 | 2.3e−5 | 4.7e−5 |
| `acceptor_signal_strength` | Acceptor score above the neighbour background | a − context_neighbor_mean | −1 to 1 | −1.4e−5 | 3.5e−5 | 1.1e−5 | 2.3e−5 |
| `acceptor_context_diff_ratio` | Acceptor score over the highest neighbour | a / context_max | ≥ 0, unbounded | 0.027 | 464.605 | 631.993 | −167.388 |
| `donor_acceptor_peak_ratio` | Donor versus acceptor peakedness | donor_peak_height_ratio / acceptor_peak_height_ratio | ≥ 0, unbounded | 1.2e6 | 0.001 | 0.014 | −0.013 |
| `type_signal_difference` | Donor versus acceptor signal strength | donor_signal_strength − acceptor_signal_strength | −1 to 1 | 0.757 | −3.5e−5 | −1.1e−5 | −2.3e−5 |
| `score_difference_ratio` | Donor minus acceptor over their sum | (d − a) / (d + a) | −1 to 1 | 1.000 | −0.997 | −0.972 | −0.026 |
| `signal_strength_ratio` | Ratio of the two signal strengths | donor_signal_strength / acceptor_signal_strength; the denominator can be near 0 or negative | unbounded, signed (−2.5e9 to 8.0e9 observed) | −53,891 | 4.0e−4 | 0.014 | −0.013 |

### 5.2 Genomic (4)

**Source.** Gene coordinates (MANE gene span, Ensembl gene span for non-MANE genes, or site ± 5 kb for the 160 sites outside any gene, where the three position columns are not meaningful) and the GRCh38 FASTA.

**Missing values.** Never null in the M4 table.

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `relative_gene_position` | Where the site sits along its gene, 5′ to 3′ | (p − gene_start) / (gene_end − gene_start); on the minus strand, 1 minus that | 0–1 | 0.028 | 0.374 | 0.374 | −5.8e−4 |
| `distance_to_gene_start` | Distance to the gene's lower coordinate | p − gene_start (genomic; on the minus strand this is the gene's 3′ end) | nucleotides | 8,383 | 5,400 | 5,395 | 5 |
| `distance_to_gene_end` | Distance to the gene's upper coordinate | gene_end − p (genomic) | nucleotides | 238 | 3,221 | 3,226 | −5 |
| `gc_content` | Local GC fraction | (G + C) / 100 over p − 49 … p + 50, forward strand | fraction 0–1 | 0.760 | 0.430 | 0.440 | −0.010 |

### 5.3 Conservation (9)

**Source.** UCSC hg38 phyloP 100-way and phastCons 100-way bigWigs. Window ±10 nt (21 nt) around the read position.

**Missing values.** Never null in the M4 table. Bases without alignment are read as 0, so a gap looks like neutral phyloP.

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `phylop_score` | Evolutionary constraint at the base | phyloP at the read position | phyloP score (−log10 p, signed; about −20 to 10) | −0.010 | −0.500 | −0.060 | −0.440 |
| `phylop_context_mean` | Average constraint nearby | mean over 21 nt | phyloP | 1.382 | 0.098 | −0.101 | 0.199 |
| `phylop_context_max` | Strongest constraint nearby | max over 21 nt | phyloP | 4.492 | 0.801 | 0.801 | 0 |
| `phylop_context_std` | How variable the constraint is nearby | standard deviation over 21 nt | phyloP | 1.268 | 0.403 | 0.818 | −0.416 |
| `phastcons_score` | Probability the base is in a conserved element | phastCons at the read position | probability 0–1 | 0.268 | 0 | 0.001 | −0.001 |
| `phastcons_context_mean` | Average of that probability nearby | mean over 21 nt | 0–1 | 0.744 | 0.005 | 0.003 | 0.002 |
| `phastcons_context_max` | Highest nearby | max over 21 nt | 0–1 | 1 | 0.017 | 0.017 | 0 |
| `phastcons_context_std` | Variability nearby | standard deviation over 21 nt | 0–0.5 | 0.346 | 0.005 | 0.005 | 7.4e−4 |
| `conservation_contrast` | How much the site stands out from its surroundings | phylop_score − phylop_context_mean | phyloP | −1.392 | −0.598 | 0.041 | −0.639 |

### 5.4 Epigenetic (12)

**Source.** ENCODE ChIP-seq fold-change-over-control bigWigs (GRCh38) for H3K36me3 and H3K4me3 in K562, GM12878, H1, HepG2 and keratinocyte. Window ±200 nt; exon/intron sides 500 nt.

**Missing values.** Never null in the M4 table. Uncovered bases read as 0.

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `h3k36me3_max_across_tissues` | Strongest H3K36ME3 signal in any of the 5 cell lines | max over lines of the value at the read position | fold change over control | 0.207 | 3.944 | 3.944 | 0 |
| `h3k36me3_mean_across_tissues` | Average signal over the 5 lines | mean over lines | fold change | 0.041 | 1.971 | 1.889 | 0.081 |
| `h3k36me3_tissue_breadth` | Number of lines where the mark is present | count of lines with value > 1.5 | count 0–5 | 0 | 3 | 3 | 0 |
| `h3k36me3_variance` | Spread across the 5 lines | variance over lines | fold change² | 0.007 | 1.659 | 1.612 | 0.046 |
| `h3k36me3_context_mean` | Regional level of the mark | per line, mean over 401 nt; then mean over lines | fold change | 0.046 | 1.560 | 1.556 | 0.005 |
| `h3k36me3_exon_intron_ratio` | Upstream versus downstream marking | per line, log2(U + 1) − log2(D + 1), U and D = mean over the 500 nt upstream and downstream in transcript direction; then mean over lines. A log2 difference (0 = equal), and it does not use the site type | log2 difference | −0.038 | −0.275 | −0.271 | −0.004 |
| `h3k4me3_max_across_tissues` | Strongest H3K4ME3 signal in any of the 5 cell lines | max over lines of the value at the read position | fold change over control | 33.174 | 5.279 | 5.279 | 0 |
| `h3k4me3_mean_across_tissues` | Average signal over the 5 lines | mean over lines | fold change | 14.548 | 1.988 | 1.961 | 0.028 |
| `h3k4me3_tissue_breadth` | Number of lines where the mark is present | count of lines with value > 1.5 | count 0–5 | 5 | 2 | 2 | 0 |
| `h3k4me3_variance` | Spread across the 5 lines | variance over lines | fold change² | 171.714 | 3.599 | 3.554 | 0.045 |
| `h3k4me3_context_mean` | Regional level of the mark | per line, mean over 401 nt; then mean over lines | fold change | 12.757 | 1.612 | 1.585 | 0.027 |
| `h3k4me3_exon_intron_ratio` | Upstream versus downstream marking | per line, log2(U + 1) − log2(D + 1), U and D = mean over the 500 nt upstream and downstream in transcript direction; then mean over lines. A log2 difference (0 = equal), and it does not use the site type | log2 difference | −1.115 | 0.588 | 0.588 | −5.7e−5 |

### 5.5 Junction (12)

**Source.** GTEx v8 split reads, aggregated per tissue (54 tissues; reads summed over each tissue's samples; a (junction, tissue) row needs ≥ 3 reads). A junction counts at both its donor and its acceptor.

**Missing values.** A site with no GTEx junction gets 0 in every count column. `junction_psi` and `junction_psi_variance` stay null instead (60% of all event sites, 10% of partner sites, 90% of deltas).

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `junction_log1p` | Total GTEx split-read support at the site | ln(1 + reads summed over all junctions and tissues) | ln reads | 15.275 | 7.476 | 11.938 | −4.461 |
| `junction_has_support` | 1 if any GTEx junction uses the site | 1 if at least one junction with ≥ 3 reads in a tissue | 0 or 1 | 1 | 1 | 1 | 0 |
| `junction_n_partners` | Number of partner sites joined to it | per tissue, distinct partners; then the max over tissues | count | 3 | 1 | 2 | −1 |
| `junction_max_reads` | Peak support (equals junction_tissue_max) | max over tissues of that tissue's summed reads | reads | 343,100 | 1,671 | 11,609 | −9,938 |
| `junction_entropy` | How spread the reads are | Shannon entropy over the per-(junction, tissue) read counts; mixes tissues and partners | bits | 5.496 | 0.528 | 6.282 | −5.754 |
| `junction_is_annotated` | Meant to flag GTF-annotated junctions | the source table has no annotation column, so this is always 0 | constant 0 | 0 | 0 | 0 | 0 |
| `junction_tissue_breadth` | Number of GTEx tissues using the site | count of tissues with ≥ 3 reads | count 0–54 | 54 | 22 | 54 | −32 |
| `junction_tissue_max` | Support in the strongest tissue | max over tissues of summed reads | reads | 343,100 | 1,671 | 11,609 | −9,938 |
| `junction_tissue_mean` | Average support over tissues that use the site | mean over tissues present (not over all 54) | reads | 79,692 | 80.227 | 2,832 | −2,751 |
| `junction_tissue_variance` | Variability of support across tissues | variance (ddof 1) of per-tissue reads; 0 with one tissue | reads² | 5.2e9 | 126,245 | 4.8e6 | −4.7e6 |
| `junction_psi` | How dominant the strongest junction at the site is | per tissue with ≥ 2 junction entries, max / sum; then the mean over those tissues | fraction 0–1 | 0.954 | null | 0.627 | null |
| `junction_psi_variance` | Variability of that dominance across tissues | variance of the per-tissue values; needs ≥ 2 such tissues | fraction² | 0.001 | null | 0.003 | null |

### 5.6 RBP binding, any protein (8)

**Source.** ENCODE eCLIP IDR peaks in K562 and HepG2 with −log10 p ≥ 2 (937,123 peaks, 168 RBPs). A peak counts if it covers the read position. Not specific to the knocked-down regulator; that is section 4.

**Missing values.** Never null. No overlapping peak gives 0 (65% of event sites).

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `rbp_n_bound` | How many RBPs have a peak covering the site | distinct RBPs among overlapping peaks (K562 and HepG2 pooled) | count | 2 | 2 | 2 | 0 |
| `rbp_max_signal` | Strongest binding | max peak signalValue among overlapping peaks | eCLIP enrichment (values 3 to 13 suggest log2 scale) | 5.736 | 3.978 | 3.978 | 0 |
| `rbp_max_neg_log10_pvalue` | Most significant overlapping peak | max peak −log10 p | −log10 p (capped at 400) | 29.627 | 30.298 | 30.298 | 0 |
| `rbp_has_splice_regulator` | 1 if a known splicing regulator binds | any overlapping RBP in a 29-name list of SR, hnRNP and core factors | 0 or 1 | 1 | 1 | 1 | 0 |
| `rbp_n_sr_proteins` | SR proteins bound | distinct RBPs from the SR family (SRSF1–11, TRA2A/B) | count | 0 | 0 | 0 | 0 |
| `rbp_n_hnrnps` | hnRNP proteins bound | distinct RBPs from the hnRNP family | count | 0 | 0 | 0 | 0 |
| `rbp_cell_line_breadth` | Cell lines with binding | distinct cell lines among overlapping peaks | count 0–2 | 2 | 1 | 1 | 0 |
| `rbp_mean_signal` | Average binding | mean signalValue over overlapping peaks | same as rbp_max_signal | 5.069 | 3.754 | 3.754 | 0 |

### 5.7 Chromatin accessibility (12)

**Source.** ENCODE ATAC-seq (K562, GM12878, HepG2, A549, IMR-90) and DNase-seq (brain cortex, heart, lung, muscle, liver) bigWigs. Window ±150 nt.

**Missing values.** Never null in the M4 table. Uncovered bases read as 0.

| Column `X` | Meaning | Calculation | Units | partner | long | short | `delta__X` |
|---|---|---|---|---:|---:|---:|---:|
| `atac_max_across_tissues` | Most open of the 5 ATAC-seq samples | max over samples of the value at the read position | fold change over control | 5.605 | 0.215 | 0.215 | 3.0e−5 |
| `atac_mean_across_tissues` | Average openness | mean over samples | fold change over control | 4.730 | 0.069 | 0.069 | 6.0e−6 |
| `atac_tissue_breadth` | Number of samples open at the site | count of samples with value > 2.0 | count 0–5 | 5 | 0 | 0 | 0 |
| `atac_variance` | Spread across samples | variance over samples | squared units | 0.586 | 0.006 | 0.006 | 1.7e−6 |
| `atac_context_mean` | Regional openness | per sample, mean over 301 nt; then mean over samples | fold change over control | 4.233 | 0.069 | 0.068 | 8.8e−4 |
| `atac_has_peak` | 1 if strongly open at the site | atac_max_across_tissues > 3.0 | 0 or 1 | 1 | 0 | 0 | 0 |
| `dnase_max_across_tissues` | Most open of the 5 DNase-seq samples | max over samples of the value at the read position | read-depth-normalised signal | 1.760 | 0.327 | 0.327 | 0 |
| `dnase_mean_across_tissues` | Average openness | mean over samples | read-depth-normalised signal | 1.288 | 0.097 | 0.097 | 0 |
| `dnase_tissue_breadth` | Number of samples open at the site | count of samples with value > 5.0 | count 0–5 | 0 | 0 | 0 | 0 |
| `dnase_variance` | Spread across samples | variance over samples | squared units | 0.098 | 0.014 | 0.014 | 0 |
| `dnase_context_mean` | Regional openness | per sample, mean over 301 nt; then mean over samples | read-depth-normalised signal | 1.529 | 0.066 | 0.066 | 5.7e−5 |
| `dnase_has_peak` | 1 if strongly open at the site | dnase_max_across_tissues > 10.0 | 0 or 1 | 0 | 0 | 0 | 0 |

Reading the example. The partner donor is a sharp, confident peak (`donor_score` 0.757, `donor_is_local_peak` 1). Both competing acceptors score essentially 0 in OpenSpliceAI (`acceptor_score` below 0.0001), yet GTEx shows real use of both, far more of the short one (`junction_log1p` 7.476 vs 11.938, about 1,760 vs 153,000 reads, so `delta__junction_log1p` = −4.461). The long acceptor is less conserved than the short one (`delta__phylop_score` = −0.440). Under AATF knockdown the long form nevertheless gained usage: $y = +0.756$.

## 6. Known quirks of the site features

Building this reference meant checking every site feature against the code and the data. Several do not do exactly what their names suggest. They affect M1–M4 alike, because all variants share the feature pipeline; fixes are tracked, and the effect of each on model accuracy has not been measured yet.

| Quirk | Columns affected | What it means for reading a value |
|---|---|---|
| Track values read at p + 1 | conservation, epigenetic, accessibility, `rbp_*` | the value describes the neighbouring base: intronic for plus-strand donors and minus-strand acceptors, exonic for the other two. Windowed means shift by 1 nt and hardly change |
| GTEx junctions listed once per overlapping gene | junction block | reads are counted twice where genes overlap (289,901 (junction, tissue) rows), inflating counts and adding duplicate "competitors" |
| Junction and eCLIP lookups ignore strand | junction, `rbp_*` | antisense junctions and peaks count too |
| `junction_is_annotated` has no source column | `junction_is_annotated` | always 0; carries no information |
| `junction_max_reads` equals `junction_tissue_max` | `junction_max_reads` | not the max of any single junction |
| ratio columns divide by values near 0 | `*_surge_ratio`, `*_peak_height_ratio`, `*_context_diff_ratio`, `donor_acceptor_peak_ratio`, `signal_strength_ratio` | extreme values (`signal_strength_ratio` from −2.5e9 to 8.0e9); tree models are unaffected by scale, linear models and distances are not |
| `*_exon_intron_ratio` is a log2 difference and ignores site type | `h3k36me3_exon_intron_ratio`, `h3k4me3_exon_intron_ratio` | 0 means equal; for an acceptor the upstream side is intron, so the sign means the opposite of the name |
| neighbour features in genomic order | base-score context columns | see section 5 |

## Summary

| Block | Columns | Depends on the regulator? | Ever null? |
|---|---:|---|---|
| Event geometry | 7 | no | never |
| Baseline (`baseline_psi_loo`, `n_other_kd`) | 2 | no (built from other experiments) | 0.2% |
| eCLIP flags | 2 | yes | never |
| Regulator binding `R_*` | 18 | yes | 43.7% (no eCLIP) |
| Partner site `partner__X` | 100 | no | two junction-PSI columns, about 10% |
| Long − short `delta__X` | 100 | no | two junction-PSI columns, about 90% |

Only 20 of the 229 columns describe which regulator was knocked down, and for 81 of the 185 regulators those 20 are null or constant. The other 209 describe the event and are the same under every regulator (apart from `baseline_psi_loo`, which shifts slightly with the row's control batch): a model can learn which events tend to respond, but it has little to tell regulators apart. [Part 2](02_model_architecture.md) takes up what that implies.

Previous: [Part 4, TabPFN and in-context learning](04_tabpfn_in_context_learning.md).
