"""Gene annotation cache for fast gene browsing.

Extracts gene metadata from GTF files and caches as Parquet for sub-second
loading on subsequent requests. Each model has its own cache since they use
different GTF annotations.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from agentic_spliceai.splice_engine.resources import get_model_resources, list_available_models
from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
    extract_gene_annotations,
    extract_exon_annotations,
    extract_splice_sites_from_exons,
)
from . import config

logger = logging.getLogger(__name__)

# In-memory cache: model_name -> DataFrame
_gene_cache: Dict[str, pl.DataFrame] = {}

# Columns a cached Parquet must carry to be reusable. Bump this when a column
# is added so pre-existing caches rebuild instead of serving a stale schema.
REQUIRED_COLUMNS = ('gene_id', 'gene_name', 'description', 'aliases', 'chrom')


def _get_cache_path(model_name: str) -> Path:
    """Get Parquet cache file path for a model."""
    return config.CACHE_DIR / f"{model_name}_genes.parquet"


def annotation_for_model(model_name: str) -> str:
    """The annotation key a model implies, e.g. openspliceai -> ``mane.GRCh38``.

    This is what keeps model->dataset provenance visible: the browser's default
    annotation is always the one the selected model was built against.
    """
    r = get_model_resources(model_name)
    return f"{r.annotation_source}.{r.build}"


def available_annotations() -> list[dict]:
    """Registered annotations whose GTF is present on disk.

    Filtered by existence so the Gene Browser dropdown can only offer something
    that will actually load (see dev/tasks/lessons.md, 2026-08-02).
    """
    out = []
    for key, spec in config.ANNOTATIONS.items():
        if not Path(spec["gtf"]).exists():
            logger.debug(f"Annotation {key}: GTF missing, not offered ({spec['gtf']})")
            continue
        out.append({
            "key": key,
            "name": spec["name"],
            "source": spec["source"],
            "build": spec["build"],
            "notes": spec.get("notes", ""),
        })
    return out


def _annotation_cache_path(annotation_key: str) -> Path:
    return config.CACHE_DIR / f"annot_{annotation_key}_genes.parquet"


def get_genes_for_annotation(annotation_key: str) -> pl.DataFrame:
    """Gene DataFrame for an annotation, independent of any model.

    Same three-layer cache as :func:`get_genes` (memory -> Parquet -> GTF parse).
    """
    spec = config.ANNOTATIONS.get(annotation_key)
    if spec is None:
        raise KeyError(f"Unknown annotation: {annotation_key}")

    mem_key = f"annot:{annotation_key}"
    if mem_key in _gene_cache:
        return _gene_cache[mem_key]

    gtf_path = Path(spec["gtf"])
    if not gtf_path.exists():
        raise FileNotFoundError(f"Annotation {annotation_key}: GTF not found at {gtf_path}")

    cache_path = _annotation_cache_path(annotation_key)
    if cache_path.exists():
        try:
            if cache_path.stat().st_mtime > gtf_path.stat().st_mtime:
                df = pl.read_parquet(cache_path)
                missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if not missing:
                    _gene_cache[mem_key] = df
                    return df
                logger.info(f"Annotation cache {annotation_key} predates {missing}; rebuilding")
        except Exception as e:
            logger.warning(f"Could not load annotation cache {annotation_key}: {e}")

    df = _build_gene_dataframe_from_gtf(
        str(gtf_path), label=annotation_key, sites_parquet=spec.get("sites")
    )
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
    except Exception as e:
        logger.warning(f"Could not save annotation cache {annotation_key}: {e}")

    _gene_cache[mem_key] = df
    return df


def _load_gene_aliases(gtf_path: Path) -> Optional[pl.DataFrame]:
    """Map ``gene_name`` -> comma-joined synonyms, read from the GFF sibling.

    Returns ``None`` when no GFF is present or it carries no ``gene_synonym``
    attributes (true for the Ensembl and GENCODE GTFs), in which case the
    caller falls back to an empty alias column and search behaves as before.
    """
    candidates = [gtf_path.with_suffix(s) for s in ('.gff', '.gff3')]
    # Synonyms are a property of the GENE, not of the annotation that lists it,
    # so an annotation without its own synonyms (Ensembl and GENCODE GTFs carry
    # none) borrows the RefSeq/MANE map, joined on gene_name. Without this,
    # switching annotation silently breaks a search that just worked.
    mane_gtf = Path(config.ANNOTATIONS['mane.GRCh38']['gtf'])
    candidates += [mane_gtf.with_suffix(s) for s in ('.gff', '.gff3')]

    for gff in candidates:
        if gff.exists():
            break
    else:
        return None

    try:
        df = (
            pl.scan_csv(
                gff, separator='\t', has_header=False, comment_prefix='#',
                new_columns=['seqid', 'source', 'type', 'start', 'end',
                             'score', 'strand', 'phase', 'attributes'],
                infer_schema_length=0,
            )
            .filter(pl.col('type') == 'gene')
            .select(
                pl.col('attributes').str.extract(r'(?:^|;)gene=([^;]+)').alias('gene_name'),
                pl.col('attributes').str.extract(r'(?:^|;)gene_synonym=([^;]+)').alias('aliases'),
            )
            .filter(pl.col('gene_name').is_not_null() & pl.col('aliases').is_not_null())
            .unique(subset=['gene_name'])
            .collect()
        )
    except Exception as e:  # a malformed GFF must not break the gene browser
        logger.warning(f"Could not read gene synonyms from {gff}: {e}")
        return None

    if df.height == 0:
        return None
    logger.info(f"Loaded {df.height:,} gene synonym entries from {gff.name}")
    return df


def _build_gene_dataframe(model_name: str) -> pl.DataFrame:
    """Build gene DataFrame for a model, from that model's own GTF."""
    resources = get_model_resources(model_name)
    return _build_gene_dataframe_from_gtf(
        str(resources.get_gtf_path()), label=model_name
    )


def _splice_site_counts(sites_parquet, gene_col: str) -> Optional[pl.DataFrame]:
    """Per-gene splice-site counts from a prebuilt track parquet.

    Much cheaper than re-deriving them from exons (which re-parses the whole
    GTF). Built by ``examples/data_preparation/05_build_annotation_track_parquets.py``;
    returns ``None`` when absent so the caller falls back to exon extraction.
    """
    if sites_parquet is None or not Path(sites_parquet).exists():
        return None
    try:
        return (
            pl.scan_parquet(sites_parquet)
            .group_by('gene_name')
            .agg(pl.len().alias('n_splice_sites'))
            .rename({'gene_name': gene_col} if gene_col != 'gene_name' else {})
            .collect()
        )
    except Exception as e:
        logger.warning(f"Could not read splice-site counts from {sites_parquet}: {e}")
        return None


def _build_gene_dataframe_from_gtf(
    gtf_path: str,
    label: str,
    sites_parquet=None,
) -> pl.DataFrame:
    """Build gene DataFrame from a GTF with derived columns.

    Extracts gene annotations, filters to protein_coding, computes gene
    length, and counts splice sites per gene. *sites_parquet*, when given,
    supplies the splice-site counts directly instead of re-deriving them.
    """
    logger.info(f"Parsing GTF for {label}: {gtf_path}")

    # Gene annotations
    genes_df = extract_gene_annotations(gtf_path, verbosity=0)

    # Filter to protein_coding
    if 'gene_type' in genes_df.columns:
        pc = genes_df.filter(pl.col('gene_type') == 'protein_coding')
        if pc.height > 0:
            genes_df = pc

    # Standardize chromosome column name
    chrom_col = 'chrom' if 'chrom' in genes_df.columns else 'seqname'

    # Compute gene length
    genes_df = genes_df.with_columns(
        (pl.col('end') - pl.col('start')).alias('length')
    )

    # Count splice sites per gene — prefer the prebuilt track parquet.
    site_counts = (
        _splice_site_counts(sites_parquet, 'gene_name')
        if 'gene_name' in genes_df.columns else None
    )
    if site_counts is not None:
        genes_df = genes_df.join(site_counts, on='gene_name', how='left').with_columns(
            pl.col('n_splice_sites').fill_null(0)
        )
    else:
        try:
            exon_df = extract_exon_annotations(gtf_path, verbosity=0)
            splice_df = extract_splice_sites_from_exons(exon_df, verbosity=0)

            gene_col = 'gene_name' if 'gene_name' in splice_df.columns else 'gene_id'
            site_counts = (
                splice_df
                .group_by(gene_col)
                .agg(pl.len().alias('n_splice_sites'))
            )
            genes_df = genes_df.join(site_counts, on=gene_col, how='left')
            genes_df = genes_df.with_columns(
                pl.col('n_splice_sites').fill_null(0)
            )
        except Exception as e:
            logger.warning(f"Could not count splice sites: {e}")
            genes_df = genes_df.with_columns(
                pl.lit(0).alias('n_splice_sites')
            )

    # Rename seqname -> chrom if needed
    if chrom_col == 'seqname':
        genes_df = genes_df.rename({'seqname': 'chrom'})

    # Enrich with descriptions from GFF3 if GTF lacked them
    if 'description' not in genes_df.columns:
        gff3_path = Path(gtf_path).with_suffix('.gff')
        if gff3_path.exists():
            logger.info(f"Enriching with descriptions from GFF3: {gff3_path}")
            gff3_df = extract_gene_annotations(str(gff3_path), verbosity=0)
            if 'description' in gff3_df.columns:
                join_col = 'gene_name' if 'gene_name' in gff3_df.columns else 'gene_id'
                desc_map = gff3_df.select([join_col, 'description']).unique(subset=[join_col])
                genes_df = genes_df.join(desc_map, on=join_col, how='left')
                genes_df = genes_df.with_columns(
                    pl.col('description').fill_null('')
                )
        if 'description' not in genes_df.columns:
            genes_df = genes_df.with_columns(pl.lit('').alias('description'))

    # Enrich with gene synonyms so users can search by the name they actually
    # know. Gene symbols are frequently not the common name: the ALS gene
    # TDP-43 is filed as TARDBP, and neither its symbol nor its description
    # ("TAR DNA binding protein") contains the string "TDP-43". The RefSeq GFF
    # carries `gene_synonym=ALS10,TDP-43` on the gene feature (16,890 of the
    # MANE genes have one). The GTF has no synonyms at all, which is why this
    # reads the GFF sibling rather than the file parsed above.
    alias_map = _load_gene_aliases(Path(gtf_path))
    if alias_map is not None and 'gene_name' in genes_df.columns:
        genes_df = (
            genes_df.join(alias_map, on='gene_name', how='left')
            .with_columns(pl.col('aliases').fill_null(''))
        )
    else:
        genes_df = genes_df.with_columns(pl.lit('').alias('aliases'))

    # Select and order columns
    keep_cols = ['gene_id', 'gene_name', 'description', 'aliases', 'chrom', 'strand', 'start', 'end', 'length', 'n_splice_sites']
    available = [c for c in keep_cols if c in genes_df.columns]
    genes_df = genes_df.select(available)

    # Sort by chromosome then start position
    genes_df = genes_df.sort(['chrom', 'start'])

    logger.info(f"Built gene cache for {label}: {genes_df.height} genes")
    return genes_df


def get_genes(model_name: str) -> pl.DataFrame:
    """Get gene DataFrame for a model, using cache when available.

    Cache priority: in-memory → Parquet file → GTF parse.
    """
    # Layer 1: in-memory
    if model_name in _gene_cache:
        return _gene_cache[model_name]

    # Layer 2: Parquet file
    cache_path = _get_cache_path(model_name)
    if cache_path.exists():
        try:
            resources = get_model_resources(model_name)
            gtf_mtime = resources.get_gtf_path().stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime

            if cache_mtime > gtf_mtime:
                df = pl.read_parquet(cache_path)
                # Schema guard: mtime alone cannot detect a cache written before
                # a column was added, so a stale-schema cache is rebuilt rather
                # than silently serving a table the API expects to search.
                missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    logger.info(
                        f"Gene cache for {model_name} predates columns {missing}; rebuilding"
                    )
                else:
                    logger.info(f"Loading gene cache from Parquet: {cache_path}")
                    _gene_cache[model_name] = df
                    return df
        except Exception as e:
            logger.warning(f"Could not load Parquet cache: {e}")

    # Layer 3: GTF parse
    df = _build_gene_dataframe(model_name)

    # Save to Parquet
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        logger.info(f"Saved gene cache to: {cache_path}")
    except Exception as e:
        logger.warning(f"Could not save Parquet cache: {e}")

    _gene_cache[model_name] = df
    return df


def get_gene_stats(model_name: str) -> dict:
    """Get summary statistics for a model's gene set."""
    df = get_genes(model_name)
    resources = get_model_resources(model_name)

    per_chrom = (
        df.group_by('chrom')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
    )

    return {
        'model': model_name,
        'build': resources.build,
        'annotation_source': resources.annotation_source,
        'total_genes': df.height,
        'per_chromosome': dict(zip(
            per_chrom['chrom'].to_list(),
            per_chrom['count'].to_list()
        )),
    }


def get_chromosomes(model_name: str) -> list:
    """Get sorted list of chromosomes for a model."""
    df = get_genes(model_name)
    chroms = df['chrom'].unique().sort().to_list()
    return chroms


def clear_cache(model_name: Optional[str] = None) -> None:
    """Clear in-memory and Parquet caches."""
    if model_name:
        _gene_cache.pop(model_name, None)
        cache_path = _get_cache_path(model_name)
        if cache_path.exists():
            cache_path.unlink()
    else:
        _gene_cache.clear()
        if config.CACHE_DIR.exists():
            for f in config.CACHE_DIR.glob('*_genes.parquet'):
                f.unlink()
