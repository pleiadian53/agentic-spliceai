# Stage 1 — Data Preparation (Labels)

**Pipeline position:** `raw GTF/FASTA → ` **this stage** ` → base scoring → features → training`

The goal of this stage is to turn genome annotation into the **label file** every downstream stage
depends on: a per-splice-site table called `splice_sites_enhanced.tsv`. Nothing here is model-specific
yet — you are just extracting "where are the real donor and acceptor sites, by strand" from the GTF.

!!! abstract "Inputs → Outputs"
    **Reads:** a GTF/GFF (resolved through the registry, or passed with `--gtf`).
    **Writes:** `data/<source>/<build>/splice_sites_enhanced.tsv` (14-column table).

---

## The command

The genome-wide ground-truth builder is
[`examples/data_preparation/04_generate_ground_truth.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/data_preparation/04_generate_ground_truth.py).

=== "M1-S (canonical / MANE)"

    ```bash
    python examples/data_preparation/04_generate_ground_truth.py \
        --output data/mane/GRCh38/
    # → data/mane/GRCh38/splice_sites_enhanced.tsv
    ```

=== "M2-S (alternative / Ensembl)"

    ```bash
    # M2-S needs BOTH files — MANE (from the M1-S command above) and Ensembl:
    python examples/data_preparation/04_generate_ground_truth.py \
        --output data/ensembl/GRCh38/ \
        --annotation-source ensembl
    # → data/ensembl/GRCh38/splice_sites_enhanced.tsv
    ```

The `--build` and `--annotation-source` are **inferred from the `--output` path** when omitted
(`data/mane/GRCh38/` → MANE/GRCh38), so the M1-S form needs no source flag. Other useful flags:

| Flag | Purpose |
|------|---------|
| `--gtf <path>` | Use an arbitrary GTF and bypass the registry (custom builds — T2T-CHM13, non-human). |
| `--force` | Overwrite an existing `splice_sites_enhanced.tsv` (the builder caches by default). |

Scripts `01_prepare_gene_data.py` and `02_prepare_splice_sites.py` are the **per-gene / demo**
variants (they require `--genes`); `03_full_data_pipeline.py` is the combined gene + sequence +
splice-site pipeline. For production label generation use `04` (or `03 --skip-sequences`).

---

## What it does

The splice sites are **derived from GTF exon boundaries, strand-aware** — there is no pre-existing
splice-site list. For each transcript's exons:

- **Donor** (5′ splice site) = the 3′ end of an exon in transcript direction: `exon.end` on `+`
  strand, `exon.start` on `−` strand.
- **Acceptor** (3′ splice site) = the 5′ end of an exon: `exon.start` on `+`, `exon.end` on `−`.
- Terminal exons omit the side they don't have.

Everything not listed as a donor or acceptor is implicitly **"neither"** — there is no negative
sampling at this stage. (The per-position label array is materialized later, at training time; see
[Stage 4](04_training_m1s.md#labels).)

### Output schema

`splice_sites_enhanced.tsv` has 14 columns:

```
chrom, start, end, position, strand, splice_type,
gene_id, transcript_id, gene_name, gene_biotype, transcript_biotype,
exon_id, exon_number, exon_rank
```

`splice_type` ∈ {`donor`, `acceptor`} and `position` is the exact site coordinate. This is the
canonical label artifact consumed by base-score evaluation, feature annotation, and meta training.

---

## M1-S vs M2-S: the one real difference

| | M1-S | M2-S |
|-|------|------|
| Ground-truth file | MANE `splice_sites_enhanced.tsv` | MANE **and** Ensembl `splice_sites_enhanced.tsv` |
| "Alternative sites" | n/a | Ensembl `\` MANE (set difference) |
| Extra work | — | run `04` a second time with `--annotation-source ensembl` |

The set difference itself is computed downstream (at evaluation, [Stage 6](06_evaluation.md)) — here
you just produce both tables.

---

## Verify before moving on

A quick strand-split sanity check on the extracted sites catches coordinate/strand bugs early:

- **Donor** sites should sit on a `GT` dinucleotide at the start of the intron; **acceptor** sites on
  an `AG` at its end — but only after resolving strand (the reverse-complement flips the motif you'd
  read off the `+`-strand FASTA). Always validate the GT/AG oracle **split by strand**, never pooled.
- Spot-check per-chromosome counts in the builder's summary output; MANE yields ~370K sites
  genome-wide, Ensembl ~2.8M.

**Reference:** the full label-generation guide, including custom-build handling, is in
[`examples/data_preparation/docs/ground_truth_custom_genomes.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/data_preparation/docs/ground_truth_custom_genomes.md).

---

→ **Next: [Stage 2 — Base Scoring](02_base_scoring.md)**
