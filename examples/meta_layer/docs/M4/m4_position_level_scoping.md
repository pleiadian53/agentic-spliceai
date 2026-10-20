# Scoping: a position-level M4 that finds induced splice sites

**Status:** scoping, 2026-10-19; **P0 pilot done 2026-10-20** (section 10). Data availability below was checked against the ENCODE portal API on 2026-10-19.

## 1. The problem with the current M4

The current M4 answers a closed question. You give it one specific splicing choice (a shared partner site plus two competing sites, all three positions known in advance) and it predicts which way removing a protein tips that choice.

That means it can only judge choices someone already listed. It cannot answer the open question:

> Remove protein X from the cells. **Where** in the genome does splicing change?

In particular it can never find a **cryptic site**: a cut site that is unused normally and switches on only after the knockdown. Those are the most interesting changes (TDP-43 in ALS is the classic case), and they are by definition not on any list.

The current model has this shape because of its data. Its labels come from rMATS tables, and rMATS only reports changes for pre-defined A3SS/A5SS events.

## 2. The new question

Make M4 work like M2-S, but with a protein as an extra input:

| Model | Input | Output |
|---|---|---|
| M2-S (exists) | any position p | is p a donor or acceptor? |
| **position-level M4** | any position p **and** protein X | how much does the **usage** of p change when X is removed? |

With this, you can scan a whole gene for protein X and see which positions light up, without any candidate list.

## 3. What "usage" means, with numbers

Reads that cross a splice junction tell us exactly which donor was joined to which acceptor. Take one donor D and count where its reads land.

Control cells:

| Acceptor | Reads from D | Usage |
|---|---:|---:|
| A1 | 80 | 0.80 |
| A2 | 20 | 0.20 |
| A3 | 0 | 0.00 |

After removing protein X:

| Acceptor | Reads from D | Usage | Change |
|---|---:|---:|---:|
| A1 | 30 | 0.30 | −0.50 |
| A2 | 50 | 0.50 | +0.30 |
| A3 | 20 | 0.20 | **+0.20** |

**Usage** of a site is its share of the reads from its partner. A1 and A2 are a normal alternative choice, the kind the current M4 already models. **A3 is an induced site**: unused in control, used after the knockdown. A position-level model has a label for it; the current model never sees it.

Every position gets a label this way. Positions with no reads in either condition have usage 0 before and after, so their change is 0, which is the correct answer for the vast majority of the genome. When a site has several partners, its usage is the read-weighted average over them. This is the same idea as LeafCutter's intron clusters, applied per site.

## 4. Is the data available? Yes

Checked on the ENCODE portal:

| Question | Answer |
|---|---|
| How many knockdown experiments? | **HepG2: 259 experiments, 233 proteins.** K562: 220 proteins. **197 proteins are knocked down in both.** |
| Are our current 185 proteins included? | 181 found by name; the other 4 likely use different names |
| What files are published? | Genome alignments (BAM) on GRCh38, about 3.4–4.3 GB per replicate. No ready-made junction files, so junctions must be counted from the BAMs |
| Are controls linked? | Yes. Each knockdown names its control experiment. HepG2 has **26 shared controls**, median 10 knockdowns each. This matches the 27 control batches we inferred from identical read counts, now confirmed from metadata. |
| Anchor proteins present? | U2AF1, U2AF2, QKI, PTBP1, RBFOX2, HNRNPC, SRSF1, TARDBP in both lines; SF3B1 in HepG2 only |

Two consequences:

1. **More proteins.** 233 in HepG2 instead of 185, because we are no longer limited to what the rMATS bundle covered.
2. **A second cell line.** With K562 the cell-context input of the conditional design becomes learnable, and we can test whether a model trained in one cell line predicts the other.

## 5. Building the labels

For each knockdown and each control, keeping replicates separate:

1. Download one BAM.
2. Count split reads per junction (for example with `regtools junctions extract`). The result is a small table: donor, acceptor, strand, read count.
3. Delete the BAM.

This must run on a pod: about 1.5–2 TB of BAMs for HepG2 alone, far more than the 500 GB volume holds, so the files are streamed and discarded, keeping only the junction tables (megabytes each). With 16 downloads in parallel, a rough estimate is a few hours of pod time for HepG2.

Then, per position and per knockdown: usage in knockdown, usage in its own control batch, and the read counts behind both. Keeping counts (not just usage) lets the loss weight well-measured positions more, and keeping replicates separate gives an independent control estimate (lesson from the current benchmark: never compare a knockdown to a control that shares its noise).

## 6. The model

Start from M2-S, which already predicts per position from sequence, base-model scores, and multimodal tracks. Add:

- **a protein input**: a vector describing protein X, computed from its eCLIP binding profile, its sequence motif, or a protein language-model embedding, so it exists even for proteins never trained on;
- **an interaction** between the position's representation and the protein vector (for example, the protein vector rescales the position features);
- **a bounded output**: knockdown usage = σ(logit(control usage) + δ), so a site already at 100% cannot be predicted to rise, which the current benchmark showed is the largest single source of signal;
- **a loss on read counts** (beta-binomial), so a site with 5 reads counts less than one with 500.

## 7. Evaluation

The same hiding logic as the current benchmark (held-out proteins, held-out chromosomes, control batches kept together), with position-level scores:

- **Finding changes:** for each held-out protein, rank all positions in the test genes by predicted change; how many of the real top changes are in the predicted top 100?
- **Finding induced sites:** of the sites that were unused in control and used after knockdown, how many does the model rank highly?
- **Agreement with the event model:** on the A3SS/A5SS events of the current benchmark, does the position-level model reproduce their direction?

## 8. Risks

- **Read depth.** Many genes have too few reads for usage to be measured; the loss on counts handles this, but sparse genes will contribute little.
- **Alignment artefacts.** Some split reads are mapping errors; junction counting needs filters (minimum anchor length, canonical motif or minimum support).
- **Indirect effects.** As with the current corpus, most responses are probably indirect, which a binding-based protein vector cannot explain on its own.
- **Scale.** Per-position labels for 233 proteins over all genes are large; training will need the sharded data path M1-S/M2-S already use.

## 9. Proposed phases

| Phase | What | Where | Rough cost |
|---|---|---|---|
| **P0 pilot** | Junction counts for 6 knockdowns with known biology (U2AF1, U2AF2, QKI, PTBP1, TARDBP, SF3B1) and their controls; check that known effects and TDP-43-induced cryptic sites appear | pod | ~1 hour |
| P1 labels | All HepG2 knockdowns and controls; per-position usage tables | pod | a few hours |
| P2 model | M2-S backbone + protein input + bounded output; train and evaluate as in section 7 | pod (GPU) | to be estimated after P1 |
| P3 second cell line | Add K562; test cross-cell transfer | pod | similar to P1 |

The pilot answers the two questions that decide whether this is worth doing: are the junction counts clean enough, and do known knockdown effects (including induced sites) show up in them?

## 10. P0 pilot results (2026-10-20)

Junction reads were counted (`examples/data_preparation/m4/05_extract_kd_junctions.py`) in 20 ENCODE GRCh38 BAMs: 6 HepG2 knockdowns (U2AF1, U2AF2, QKI, PTBP1, TARDBP, SF3B1), two replicates each, and their 4 control experiments (QKI and PTBP1 share one, TARDBP and SF3B1 another). 119 GB streamed and discarded in 25 minutes on a pod; the junction tables total 41 MB. Analysis: `06_pilot_junction_usage.py`.

**1. The counts are clean.** 48–124 million reads per BAM, about 190,000–200,000 junctions each, and 99.91–99.96% of junction reads carry a canonical splice motif.

**2. They reproduce the existing labels.** On each knockdown's A3SS/A5SS events, ΔPSI recomputed from junction reads alone correlates with rMATS's ΔPSI at r = 0.73–0.78 (about 5,000–6,000 events each), and the direction agrees on 82–96% of significant events. rMATS also uses exon-body reads, so exact agreement is not expected.

**3. Replicates agree moderately.** For sites whose usage changes, the change measured in replicate 1 vs replicate 2 (each against the pooled control) correlates at r = 0.73 (U2AF1, U2AF2), 0.63 (PTBP1), 0.58 (QKI), 0.55 (SF3B1), 0.43 (TARDBP). Strong regulators give reproducible position-level labels; weak ones are noisy.

**4. Induced sites: real for some regulators, noise for others.** The simple rule "usage ≤ 0.01 in control, ≥ 0.10 after knockdown, ≥ 5 reads" was also applied to two replicates of the same control, where nothing was knocked down, to measure the noise floor. At matched depth (one replicate vs one replicate):

| Knockdown | Induced sites (knockdown vs control) | Same rule, control vs control | Signal |
|---|---|---|---|
| U2AF1 | 536 / 605 | 39 / 122 | about 5× the noise floor |
| U2AF2 | 871 / 1,058 | 278 / 347 | about 3× |
| PTBP1 | 246 / 342 | 67 / 122 | about 3× |
| QKI | 229 / 251 | 67 / 122 | about 2× |
| SF3B1 | 168 / 124 | 113 / 132 | at the noise floor |
| TARDBP | 135 / 117 | 113 / 132 | at the noise floor |

Most induced sites for U2AF1/U2AF2 are in no annotation (595 of 938 and 733 of 1,255 with pooled replicates), consistent with loss of the 3′ splice-site recognition factor exposing cryptic acceptors.

**Decision.** Go, with two requirements learned here:

- **Labels need statistics, not thresholds.** A naive usage rule produces 39–347 "induced" sites between replicates of the same control. Labels and evaluation must use the read counts (beta-binomial likelihood, replicate consistency), as section 6 already proposed.
- **TDP-43 gives no induced-site signal in HepG2.** Its knockdown is indistinguishable from the noise floor here (as is SF3B1's). The ALS application will need neuronal knockdown data; HepG2 cannot validate it.

**Next (P1):** all HepG2 knockdowns. At the pilot's rate (20 BAMs in 25 minutes with 6 workers) the roughly 570 BAMs take about 12 hours with 6 workers, less with more parallel downloads.
