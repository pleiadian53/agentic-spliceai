"""
Python API for splice site prediction.

This module provides a high-level Python API for splice site prediction,
wrapping the meta-spliceai base layer with convenient interfaces.
"""

from typing import List, Optional, Dict, Any, Union
import polars as pl
import pandas as pd


class SplicePredictionAPI:
    """
    High-level API for splice site prediction.
    
    This class provides a convenient interface to the meta-spliceai base layer,
    with methods for common prediction tasks.
    
    Examples
    --------
    >>> # Initialize API
    >>> api = SplicePredictionAPI(base_model="openspliceai")
    >>> 
    >>> # Predict for genes
    >>> results = api.predict_genes(["BRCA1", "TP53"])
    >>> positions = results["positions"]
    >>> 
    >>> # Get high-confidence predictions
    >>> high_conf = api.get_high_confidence_predictions(results, threshold=0.9)
    """
    
    def __init__(
        self,
        base_model: str = "openspliceai",
        verbosity: int = 1,
        **config_kwargs
    ):
        """
        Initialize the Splice Prediction API.
        
        Parameters
        ----------
        base_model : str, default="openspliceai"
            Base model to use: "openspliceai" or "spliceai"
        verbosity : int, default=1
            Output verbosity (0=minimal, 1=normal, 2=detailed)
        **config_kwargs
            Additional configuration parameters
        """
        self.base_model = base_model
        self.verbosity = verbosity
        self.config = config_kwargs
        
    def predict_genes(
        self,
        genes: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict splice sites for specific genes.
        
        Parameters
        ----------
        genes : List[str]
            Gene symbols or IDs to analyze
        **kwargs
            Additional prediction parameters
            
        Returns
        -------
        dict
            Results dictionary with predictions
        """
        from agentic_spliceai.splice_engine import run_base_model_predictions
        
        config = {**self.config, **kwargs}
        
        return run_base_model_predictions(
            base_model=self.base_model,
            target_genes=genes,
            verbosity=self.verbosity,
            **config
        )
    
    def predict_chromosomes(
        self,
        chromosomes: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict splice sites for specific chromosomes.
        
        Parameters
        ----------
        chromosomes : List[str]
            Chromosomes to process (e.g., ["21", "22"])
        **kwargs
            Additional prediction parameters
            
        Returns
        -------
        dict
            Results dictionary with predictions
        """
        from agentic_spliceai.splice_engine import run_base_model_predictions
        
        config = {**self.config, **kwargs}
        
        return run_base_model_predictions(
            base_model=self.base_model,
            target_chromosomes=chromosomes,
            verbosity=self.verbosity,
            **config
        )
    
    @staticmethod
    def get_high_confidence_predictions(
        results: Dict[str, Any],
        threshold: float = 0.9,
        splice_type: Optional[str] = None
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Filter predictions to high-confidence sites.
        
        Parameters
        ----------
        results : dict
            Results from predict_genes() or predict_chromosomes()
        threshold : float, default=0.9
            Minimum confidence threshold
        splice_type : str, optional
            Filter by splice type: "donor" or "acceptor"
            
        Returns
        -------
        DataFrame
            Filtered high-confidence predictions
        """
        positions = results.get("positions")
        if positions is None:
            raise ValueError("No positions found in results")
        
        # Handle both polars and pandas
        if isinstance(positions, pl.DataFrame):
            # Polars filtering
            filtered = positions.filter(
                (pl.col("donor_score") > threshold) | 
                (pl.col("acceptor_score") > threshold)
            )
            
            if splice_type:
                filtered = filtered.filter(pl.col("splice_type") == splice_type)
                
        else:  # pandas
            # Pandas filtering
            filtered = positions[
                (positions["donor_score"] > threshold) | 
                (positions["acceptor_score"] > threshold)
            ]
            
            if splice_type:
                filtered = filtered[filtered["splice_type"] == splice_type]
        
        return filtered
    
    @staticmethod
    def get_error_positions(
        results: Dict[str, Any],
        error_type: Optional[str] = None
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Get positions with prediction errors.
        
        Parameters
        ----------
        results : dict
            Results from predict_genes() or predict_chromosomes()
        error_type : str, optional
            Filter by error type: "FP" (false positive) or "FN" (false negative)
            
        Returns
        -------
        DataFrame
            Error positions
        """
        error_analysis = results.get("error_analysis")
        if error_analysis is None:
            raise ValueError("No error analysis found in results")
        
        if error_type:
            if isinstance(error_analysis, pl.DataFrame):
                return error_analysis.filter(pl.col("error_type") == error_type)
            else:
                return error_analysis[error_analysis["error_type"] == error_type]
        
        return error_analysis
    
    @staticmethod
    def export_predictions(
        results: Dict[str, Any],
        output_path: str,
        format: str = "csv"
    ):
        """
        Export predictions to file.
        
        Parameters
        ----------
        results : dict
            Results from predict_genes() or predict_chromosomes()
        output_path : str
            Output file path
        format : str, default="csv"
            Output format: "csv", "tsv", "parquet", or "json"
        """
        positions = results.get("positions")
        if positions is None:
            raise ValueError("No positions found in results")
        
        if isinstance(positions, pl.DataFrame):
            # Polars export
            if format == "csv":
                positions.write_csv(output_path)
            elif format == "tsv":
                positions.write_csv(output_path, separator="\t")
            elif format == "parquet":
                positions.write_parquet(output_path)
            elif format == "json":
                positions.write_json(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            # Pandas export
            if format == "csv":
                positions.to_csv(output_path, index=False)
            elif format == "tsv":
                positions.to_csv(output_path, sep="\t", index=False)
            elif format == "parquet":
                positions.to_parquet(output_path, index=False)
            elif format == "json":
                positions.to_json(output_path, orient="records")
            else:
                raise ValueError(f"Unsupported format: {format}")


# Convenience functions

def quick_predict(
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    base_model: str = "openspliceai",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick prediction function for interactive use.
    
    Parameters
    ----------
    genes : List[str], optional
        Gene symbols or IDs
    chromosomes : List[str], optional
        Chromosomes to process
    base_model : str, default="openspliceai"
        Base model to use
    **kwargs
        Additional parameters
        
    Returns
    -------
    dict
        Prediction results
        
    Examples
    --------
    >>> results = quick_predict(genes=["BRCA1", "TP53"])
    >>> print(f"Found {len(results['positions'])} positions")
    """
    from agentic_spliceai.splice_engine import run_base_model_predictions
    
    return run_base_model_predictions(
        base_model=base_model,
        target_genes=genes,
        target_chromosomes=chromosomes,
        **kwargs
    )


def predict_and_filter(
    genes: List[str],
    confidence_threshold: float = 0.9,
    base_model: str = "openspliceai",
    **kwargs
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Predict and return only high-confidence sites.
    
    Parameters
    ----------
    genes : List[str]
        Gene symbols or IDs
    confidence_threshold : float, default=0.9
        Minimum confidence threshold
    base_model : str, default="openspliceai"
        Base model to use
    **kwargs
        Additional parameters
        
    Returns
    -------
    DataFrame
        High-confidence predictions
        
    Examples
    --------
    >>> high_conf = predict_and_filter(["BRCA1"], confidence_threshold=0.95)
    >>> print(high_conf.head())
    """
    results = quick_predict(genes=genes, base_model=base_model, **kwargs)
    return SplicePredictionAPI.get_high_confidence_predictions(
        results, 
        threshold=confidence_threshold
    )
