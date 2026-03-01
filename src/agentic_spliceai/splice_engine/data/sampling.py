"""Cross-layer gene sampling utilities.

Provides deterministic gene sampling from the intersection of multiple models'
GTF annotations, ensuring the same gene set works for fair cross-model comparison.
"""

import polars as pl
from typing import List, Optional, Set

from ..resources import get_model_resources, list_available_models
from ..base_layer.data.genomic_extraction import extract_gene_annotations
from ..utils.dataframe import subsample_dataframe


def get_protein_coding_genes(
    model_name: str,
    gene_type: str = 'protein_coding',
) -> Set[str]:
    """Get protein-coding gene names from a model's GTF annotation.

    Parameters
    ----------
    model_name : str
        Base model name (e.g., 'spliceai', 'openspliceai').
    gene_type : str
        Gene biotype to filter for. Default 'protein_coding'.

    Returns
    -------
    set of str
        Gene names matching the specified biotype.
    """
    res = get_model_resources(model_name)
    gtf_path = res.get_gtf_path()
    df = extract_gene_annotations(str(gtf_path), verbosity=0)

    if 'gene_type' in df.columns:
        pc = df.filter(pl.col('gene_type') == gene_type)
        if pc.height > 0:
            df = pc

    return set(df['gene_name'].unique().to_list())


def sample_random_genes(
    n_genes: int,
    model_names: Optional[List[str]] = None,
    gene_type: str = 'protein_coding',
    random_seed: int = 42,
) -> List[str]:
    """Sample N random genes from the intersection of models' GTFs.

    Uses the intersection so the same gene set works across all specified
    models, enabling fair cross-model comparison with a single seed.

    Parameters
    ----------
    n_genes : int
        Number of genes to sample.
    model_names : list of str, optional
        Models whose GTFs to intersect. Defaults to all available models.
    gene_type : str
        Gene biotype to filter for. Default 'protein_coding'.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Sorted list of sampled gene names.
    """
    if model_names is None:
        model_names = list_available_models()

    gene_sets = {
        name: get_protein_coding_genes(name, gene_type)
        for name in model_names
    }

    # Intersect across all models
    common_genes = set.intersection(*gene_sets.values())
    print(f"   Gene pool: {len(common_genes)} {gene_type} genes shared across {len(model_names)} models")

    if n_genes > len(common_genes):
        print(f"   Warning: Requested {n_genes} but only {len(common_genes)} available. Using all.")
        n_genes = len(common_genes)

    # Sort for deterministic ordering, then subsample
    pool_df = pl.DataFrame({'gene_name': sorted(common_genes)})
    sampled_df = subsample_dataframe(pool_df, n=n_genes, random_state=random_seed)
    return sorted(sampled_df['gene_name'].to_list())
