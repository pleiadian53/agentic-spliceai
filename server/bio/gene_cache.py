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


def _get_cache_path(model_name: str) -> Path:
    """Get Parquet cache file path for a model."""
    return config.CACHE_DIR / f"{model_name}_genes.parquet"


def _build_gene_dataframe(model_name: str) -> pl.DataFrame:
    """Build gene DataFrame from GTF with derived columns.

    Extracts gene annotations, filters to protein_coding, computes gene
    length, and counts splice sites per gene.
    """
    resources = get_model_resources(model_name)
    gtf_path = str(resources.get_gtf_path())

    logger.info(f"Parsing GTF for {model_name}: {gtf_path}")

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

    # Count splice sites per gene
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

    # Select and order columns
    keep_cols = ['gene_id', 'gene_name', 'description', 'chrom', 'strand', 'start', 'end', 'length', 'n_splice_sites']
    available = [c for c in keep_cols if c in genes_df.columns]
    genes_df = genes_df.select(available)

    # Sort by chromosome then start position
    genes_df = genes_df.sort(['chrom', 'start'])

    logger.info(f"Built gene cache for {model_name}: {genes_df.height} genes")
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
                logger.info(f"Loading gene cache from Parquet: {cache_path}")
                df = pl.read_parquet(cache_path)
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
