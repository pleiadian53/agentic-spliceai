"""
PyTorch Dataset for meta-layer training.

Loads base layer artifacts and prepares them for multimodal training.

Ported from: meta_spliceai/splice_engine/meta_layer/data/dataset.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

from ..core.config import MetaLayerConfig
from ..core.artifact_loader import ArtifactLoader
from ..core.feature_schema import (
    FeatureSchema, 
    LABEL_ENCODING, 
    DEFAULT_SCHEMA
)

logger = logging.getLogger(__name__)


class MetaLayerDataset(Dataset):
    """
    PyTorch Dataset for multimodal meta-layer training.
    
    Loads base layer artifacts and provides:
    - DNA sequences (501nt windows)
    - Base model score features (50+ features)
    - Labels (donor, acceptor, neither)
    - Optional sample weights (for SpliceVarDB integration)
    
    Parameters
    ----------
    data : Union[pl.DataFrame, Path, str]
        Either a Polars DataFrame or path to a parquet/TSV file.
    schema : FeatureSchema, optional
        Feature schema defining column categories.
    tokenizer : callable, optional
        Function to tokenize DNA sequences. If None, returns raw sequences.
    normalize_features : bool
        Whether to z-score normalize numeric features.
    max_seq_length : int
        Maximum sequence length (for padding/truncation).
    
    Examples
    --------
    >>> from agentic_spliceai.splice_engine.meta_layer.data import MetaLayerDataset
    >>> 
    >>> dataset = MetaLayerDataset('data/meta_training/training_data.parquet')
    >>> sample = dataset[0]
    >>> print(sample['sequence'].shape)  # Tokenized sequence
    >>> print(sample['features'].shape)  # Score features
    >>> print(sample['label'])           # 0, 1, or 2
    """
    
    def __init__(
        self,
        data: Union[pl.DataFrame, Path, str],
        schema: Optional[FeatureSchema] = None,
        tokenizer: Optional[callable] = None,
        normalize_features: bool = True,
        max_seq_length: int = 512,
        weight_column: Optional[str] = None,
        check_leakage: bool = True,
        leakage_threshold: float = 0.95,
        extra_exclude_cols: Optional[List[str]] = None
    ):
        self.schema = schema or DEFAULT_SCHEMA
        self.tokenizer = tokenizer
        self.normalize_features = normalize_features
        self.max_seq_length = max_seq_length
        self.weight_column = weight_column
        self.check_leakage = check_leakage
        self.leakage_threshold = leakage_threshold
        self.extra_exclude_cols = extra_exclude_cols or []
        
        # Load data
        if isinstance(data, pl.DataFrame):
            self.df = data
        else:
            self.df = self._load_data(Path(data))
        
        # Validate required columns
        self._validate_columns()
        
        # Check for leakage columns in data
        self._check_leakage_columns()
        
        # Get feature columns (excluding leakage and metadata)
        self.feature_cols = self._get_feature_columns()
        
        # Optional: Check for correlation-based leakage
        if check_leakage:
            self._check_correlation_leakage()
        
        # Compute normalization statistics if needed
        if self.normalize_features:
            self._compute_normalization_stats()
        
        logger.info(f"MetaLayerDataset initialized with {len(self)} samples")
        logger.info(f"  Feature columns: {len(self.feature_cols)}")
        logger.info(f"  Sequence column: {self.schema.SEQUENCE_COL}")
    
    def _load_data(self, path: Path) -> pl.DataFrame:
        """Load data from file."""
        path = Path(path)
        
        if path.suffix == '.parquet':
            return pl.read_parquet(path)
        elif path.suffix in ['.tsv', '.csv']:
            sep = '\t' if path.suffix == '.tsv' else ','
            return pl.read_csv(path, separator=sep)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _validate_columns(self):
        """Validate that required columns exist."""
        required = [self.schema.SEQUENCE_COL] + self.schema.LABEL_COLS[:1]  # At least splice_type
        
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for at least some score features
        score_cols = self.schema.BASE_SCORE_COLS
        available_scores = [c for c in score_cols if c in self.df.columns]
        if not available_scores:
            raise ValueError(
                f"No base score columns found. Expected at least one of: {score_cols}"
            )
    
    def _check_leakage_columns(self):
        """Check for and warn about leakage columns in the data."""
        # Get all columns that should be excluded
        excluded_cols = set(self.schema.get_excluded_cols() + self.extra_exclude_cols)
        
        # Check which leakage columns are present
        present_leakage = [c for c in self.schema.LEAKAGE_COLS if c in self.df.columns]
        present_metadata = [c for c in self.schema.METADATA_COLS if c in self.df.columns]
        
        if present_leakage:
            logger.warning(
                f"⚠️  LEAKAGE COLUMNS DETECTED in data: {present_leakage}. "
                "These will be excluded from features."
            )
        
        if present_metadata:
            logger.debug(f"Metadata columns present (will be excluded): {present_metadata}")
    
    def _check_correlation_leakage(self):
        """Check for correlation-based feature leakage with the label."""
        try:
            # Get labels
            label_col = 'splice_type'
            if label_col not in self.df.columns:
                return
            
            # Encode labels numerically
            labels = self.df[label_col].map_dict(LABEL_ENCODING, default=2).to_numpy()
            
            # Check correlation with each feature
            leaky_features = []
            for col in self.feature_cols[:50]:  # Sample first 50 features
                try:
                    values = self.df[col].drop_nulls().to_numpy()
                    if len(values) < 100:
                        continue
                    
                    # Compute correlation
                    labels_subset = labels[:len(values)]
                    corr = np.abs(np.corrcoef(values, labels_subset)[0, 1])
                    
                    if corr >= self.leakage_threshold:
                        leaky_features.append((col, corr))
                except Exception:
                    continue
            
            if leaky_features:
                logger.warning(
                    f"⚠️  POTENTIAL LEAKAGE: Features with correlation >= {self.leakage_threshold}:"
                )
                for feat, corr in leaky_features:
                    logger.warning(f"    - {feat}: correlation = {corr:.3f}")
                logger.warning(
                    "Consider adding these to extra_exclude_cols or investigating further."
                )
        except Exception as e:
            logger.debug(f"Correlation leakage check failed: {e}")
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns to use (excluding leakage and metadata)."""
        # Define allowed feature categories
        all_feature_cols = (
            self.schema.BASE_SCORE_COLS +
            self.schema.CONTEXT_SCORE_COLS +
            self.schema.PROBABILITY_FEATURE_COLS +
            self.schema.CONTEXT_PATTERN_COLS +
            self.schema.DONOR_PATTERN_COLS +
            self.schema.ACCEPTOR_PATTERN_COLS +
            self.schema.COMPARATIVE_COLS
        )
        
        # Get columns to exclude (leakage + metadata + extra)
        excluded = set(self.schema.get_excluded_cols() + self.extra_exclude_cols)
        
        # Filter to existing columns, excluding leakage/metadata
        available = []
        excluded_count = 0
        for c in all_feature_cols:
            if c not in self.df.columns:
                continue
            if c in excluded:
                excluded_count += 1
                logger.debug(f"Excluding column '{c}' (in exclusion list)")
                continue
            available.append(c)
        
        logger.info(f"Using {len(available)}/{len(all_feature_cols)} feature columns")
        if excluded_count > 0:
            logger.info(f"  Excluded {excluded_count} columns (leakage/metadata)")
        
        return available
    
    def _compute_normalization_stats(self):
        """Compute mean and std for feature normalization."""
        feature_df = self.df.select(self.feature_cols)
        
        self.feature_means = {}
        self.feature_stds = {}
        
        for col in self.feature_cols:
            values = feature_df[col].drop_nulls()
            self.feature_means[col] = float(values.mean())
            std = float(values.std())
            # Avoid division by zero
            self.feature_stds[col] = std if std > 1e-8 else 1.0
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'sequence': Tokenized sequence tensor [L] or raw string
            - 'features': Feature tensor [F]
            - 'label': Label tensor (scalar)
            - 'weight': Sample weight (scalar, optional)
        """
        row = self.df.row(idx, named=True)
        
        # Get sequence
        sequence = row[self.schema.SEQUENCE_COL]
        if self.tokenizer is not None:
            sequence = self.tokenizer(sequence, max_length=self.max_seq_length)
        else:
            # One-hot encode for CNN
            sequence = self._one_hot_encode(sequence)
        
        # Get features
        features = self._extract_features(row)
        
        # Get label
        label = self._encode_label(row.get('splice_type', ''))
        
        result = {
            'sequence': sequence,
            'features': features,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Add weight if available
        if self.weight_column and self.weight_column in row:
            weight = row[self.weight_column]
            result['weight'] = torch.tensor(weight if weight is not None else 1.0, dtype=torch.float32)
        else:
            result['weight'] = torch.tensor(1.0, dtype=torch.float32)
        
        return result
    
    def _one_hot_encode(self, sequence: str) -> torch.Tensor:
        """
        One-hot encode DNA sequence.
        
        Returns tensor of shape [4, L] for CNN processing.
        """
        # Mapping: A=0, C=1, G=2, T=3
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        
        sequence = sequence.upper()[:self.max_seq_length]
        
        # Pad if needed
        if len(sequence) < self.max_seq_length:
            sequence = sequence + 'N' * (self.max_seq_length - len(sequence))
        
        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
        
        for i, base in enumerate(sequence):
            idx = mapping.get(base, 0)
            encoded[idx, i] = 1.0
        
        return torch.from_numpy(encoded)
    
    def _extract_features(self, row: Dict) -> torch.Tensor:
        """Extract and normalize feature values."""
        features = []
        
        for col in self.feature_cols:
            value = row.get(col, 0.0)
            
            # Handle None values
            if value is None:
                value = 0.0
            
            # Normalize if enabled
            if self.normalize_features:
                value = (value - self.feature_means[col]) / self.feature_stds[col]
            
            features.append(value)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_label(self, splice_type: str) -> int:
        """Encode splice type to integer label."""
        if splice_type is None:
            splice_type = ''
        return LABEL_ENCODING.get(splice_type.lower(), LABEL_ENCODING['neither'])
    
    @property
    def num_features(self) -> int:
        """Number of feature columns."""
        return len(self.feature_cols)
    
    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return 3  # donor, acceptor, neither
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        
        Returns
        -------
        torch.Tensor
            Tensor of shape [3] with weights for each class.
        """
        # Count labels
        label_col = 'splice_type'
        if label_col not in self.df.columns:
            return torch.ones(3)
        
        counts = self.df.group_by(label_col).count()
        
        # Map to class indices
        class_counts = {0: 0, 1: 0, 2: 0}
        for row in counts.iter_rows(named=True):
            label = self._encode_label(row[label_col])
            class_counts[label] = row['count']
        
        total = sum(class_counts.values())
        if total == 0:
            return torch.ones(3)
        
        # Inverse frequency weighting
        weights = []
        for i in range(3):
            count = class_counts[i]
            if count > 0:
                weights.append(total / (3 * count))
            else:
                weights.append(1.0)
        
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    dataset: MetaLayerDataset,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from a dataset.
    
    Parameters
    ----------
    dataset : MetaLayerDataset
        The dataset to split.
    train_split : float
        Fraction for training.
    val_split : float
        Fraction for validation.
    test_split : float
        Fraction for testing.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Number of worker processes.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6
    
    # Set seed
    torch.manual_seed(seed)
    
    # Compute sizes
    n = len(dataset)
    train_size = int(n * train_split)
    val_size = int(n * val_split)
    test_size = n - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={train_size}, val={val_size}, test={test_size}")
    
    return train_loader, val_loader, test_loader


def prepare_training_data(
    config: MetaLayerConfig,
    output_dir: Optional[Path] = None,
    chromosomes: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    balance_classes: bool = True
) -> pl.DataFrame:
    """
    Prepare training data from base layer artifacts.
    
    Parameters
    ----------
    config : MetaLayerConfig
        Configuration specifying base model and paths.
    output_dir : Path, optional
        Directory to save processed data.
    chromosomes : list of str, optional
        Specific chromosomes to include.
    max_samples : int, optional
        Maximum number of samples to include.
    balance_classes : bool
        Whether to balance class distribution.
    
    Returns
    -------
    pl.DataFrame
        Prepared training data.
    """
    logger.info("Preparing training data...")
    
    # Load artifacts
    loader = ArtifactLoader(config)
    df = loader.load_analysis_sequences(chromosomes=chromosomes)
    
    logger.info(f"Loaded {len(df)} positions from artifacts")
    
    # Encode labels
    df = df.with_columns(
        pl.col('splice_type')
        .fill_null('')
        .str.to_lowercase()
        .replace(LABEL_ENCODING)
        .alias('label')
    )
    
    # Balance classes if requested
    if balance_classes:
        df = _balance_classes(df, max_samples)
    elif max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, seed=config.seed)
    
    logger.info(f"Prepared {len(df)} training samples")
    
    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'training_data.parquet'
        df.write_parquet(output_path)
        logger.info(f"Saved training data to {output_path}")
    
    return df


def _balance_classes(
    df: pl.DataFrame, 
    max_samples: Optional[int] = None
) -> pl.DataFrame:
    """Balance class distribution via undersampling."""
    # Count per class
    label_col = 'splice_type'
    counts = df.group_by(label_col).count()
    
    # Find minimum class size
    min_count = counts['count'].min()
    
    if max_samples:
        # Limit per-class samples
        per_class = max_samples // 3
        min_count = min(min_count, per_class)
    
    # Sample from each class
    balanced_dfs = []
    
    for label in ['donor', 'acceptor', '', None]:
        if label is None:
            subset = df.filter(pl.col(label_col).is_null())
        elif label == '':
            subset = df.filter(pl.col(label_col) == '')
        else:
            subset = df.filter(pl.col(label_col) == label)
        
        if len(subset) > min_count:
            subset = subset.sample(n=min_count)
        
        if len(subset) > 0:
            balanced_dfs.append(subset)
    
    return pl.concat(balanced_dfs).sample(fraction=1.0)  # Shuffle









