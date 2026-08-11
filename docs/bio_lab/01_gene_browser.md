# Gene Browser (`/`)

A paginated table of protein-coding genes. Its job is to get you to a gene and to make the
**annotation** you are looking at explicit.

## Model and Annotation are two different axes

This trips people up, so it is worth stating directly.

- **Model** picks a base predictor. It also sets the **default** annotation, because each model was
  trained against one — that mapping is the point, not an accident. `openspliceai` was trained on
  MANE GRCh38, so selecting it shows you MANE.
- **Annotation** overrides that. Browsing genes needs no model at all; it needs a GTF.

| Annotation | Build | Protein-coding genes | Notes |
|---|---|---|---|
| MANE | GRCh38 | 19,288 | One transcript per gene. Trains M1-S and M3-S. |
| Ensembl 112 | GRCh38 | 20,089 | All transcripts. Trains **M2-S**. |
| GENCODE v47 | GRCh38 | 20,092 | Near-superset of Ensembl (+136,858 sites genome-wide, mostly non-coding). |
| Ensembl 87 | **GRCh37** | 20,356 | Legacy build, for the SpliceAI base model. |

!!! warning "Different builds are not comparable"
    Selecting a GRCh37 annotation while a GRCh38 model is chosen shows a notice and **disables the
    gene links**. Coordinates from one build mean nothing to a model trained on the other, so the
    UI refuses to hand you through to a prediction rather than quietly reconciling them.

The first load of a new annotation parses its GTF (4.5–54 s), then caches to Parquet.

## Search: name, ID, synonym, or description

Gene symbols are frequently not what a gene is *known* as. The ALS gene everyone calls **TDP-43** is
filed as `TARDBP`, and searching `TDP` returns three unrelated tyrosyl-DNA phosphodiesterases.

Search therefore covers four fields:

| Query | Matches via | Result |
|---|---|---|
| `TARDBP` | symbol | TARDBP |
| `TDP-43` | **synonym** | TARDBP |
| `TAR DNA binding` | description | TARDBP |
| `ALS10` | synonym | TARDBP |

Synonyms come from the RefSeq GFF (`gene_synonym`), which carries them for **16,890 of 19,288** MANE
genes. The GTF has none, which is why they were invisible for so long. Ensembl and GENCODE GTFs also
carry none, so they borrow the RefSeq map joined on gene name — a synonym belongs to the gene, not to
the annotation listing it.

**Results are relevance-ranked**: exact symbol → exact synonym → symbol prefix → symbol substring →
description-only. Without that, `BRCA1` returns ten rows led by `BRAP`, because BRIP1, BARD1 and
BABAM1 all mention BRCA1 in *their* descriptions.

!!! tip "Not found?"
    Literal substring matching only. `TDP-43` works because it is a recorded synonym; a functional
    query like *"genes involved in nonsense-mediated decay"* will not. Semantic search over the
    agentic layer is on the roadmap, not built.

## Columns

`n_splice_sites` is the count for that gene **in the selected annotation**, so it changes when you
switch. TARDBP: **10** in MANE, **40** in Ensembl. That gap is the entire subject of the
[Genome View](02_genome_view.md) — those 30 extra sites are what M2-S exists to find.

## Next

Click a gene name to open the [Genome View](02_genome_view.md).
