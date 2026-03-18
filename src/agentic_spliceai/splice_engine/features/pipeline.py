"""Feature pipeline — orchestrates modalities into a composable transform.

The FeaturePipeline composes multiple Modality instances sequentially,
resolving dependencies via topological sort, validating inputs/outputs,
and optionally resolving resource paths (FASTA, GTF) from the registry.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import polars as pl

from .modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    """Configuration for the feature engineering pipeline.

    Attributes
    ----------
    base_model : str
        Base model name for resource resolution (e.g., 'openspliceai').
    modalities : list of str
        Ordered list of modality names to apply.
    modality_configs : dict
        Per-modality configuration overrides. Keys are modality names,
        values are typed ModalityConfig subclass instances.
        Modalities not listed here use their ``default_config()``.
    output_format : str
        Output serialization format ('parquet' or 'tsv').
    verbosity : int
        Logging verbosity (0=quiet, 1=normal, 2=debug).
    """

    base_model: str = "openspliceai"
    modalities: List[str] = field(
        default_factory=lambda: ["base_scores", "annotation", "sequence"]
    )
    modality_configs: Dict[str, ModalityConfig] = field(default_factory=dict)
    output_format: str = "parquet"
    verbosity: int = 1


class FeaturePipeline:
    """Compose modalities into a sequential feature engineering pipeline.

    The pipeline maintains a class-level registry of available modalities.
    Built-in modalities are registered by ``features.modalities.__init__``.
    External modalities can be registered via ``register()``.

    Examples
    --------
    >>> config = FeaturePipelineConfig(modalities=['base_scores'])
    >>> pipeline = FeaturePipeline(config)
    >>> enriched_df = pipeline.transform(predictions_df)
    """

    # Class-level registry: name → (ModalityClass, DefaultConfigClass)
    _REGISTRY: Dict[str, Tuple[Type[Modality], Type[ModalityConfig]]] = {}

    def __init__(self, config: FeaturePipelineConfig) -> None:
        self.config = config
        self._modalities: List[Modality] = []
        self._build_modality_chain()

    # ------------------------------------------------------------------
    # Registry (class methods)
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        modality_cls: Type[Modality],
        config_cls: Type[ModalityConfig],
    ) -> None:
        """Register a modality for use in pipelines.

        Parameters
        ----------
        name : str
            Registry key (e.g., 'base_scores', 'epigenetic').
        modality_cls : type
            Modality subclass.
        config_cls : type
            Corresponding ModalityConfig subclass.
        """
        cls._REGISTRY[name] = (modality_cls, config_cls)
        logger.debug("Registered modality: %s", name)

    @classmethod
    def available_modalities(cls) -> List[str]:
        """Return names of all registered modalities."""
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def get_modality_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get info about a registered modality."""
        if name not in cls._REGISTRY:
            return None
        mod_cls, cfg_cls = cls._REGISTRY[name]
        instance = mod_cls(mod_cls.default_config())
        meta = instance.meta
        return {
            "name": meta.name,
            "version": meta.version,
            "description": meta.description,
            "output_columns": list(meta.output_columns),
            "required_inputs": sorted(meta.required_inputs),
            "config_class": cfg_cls.__name__,
        }

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_modality_chain(self) -> None:
        """Instantiate and order modalities based on config."""
        for name in self.config.modalities:
            if name not in self._REGISTRY:
                raise ValueError(
                    f"Unknown modality '{name}'. "
                    f"Available: {self.available_modalities()}"
                )

            mod_cls, cfg_cls = self._REGISTRY[name]

            # Use user config if provided, otherwise default
            if name in self.config.modality_configs:
                cfg = self.config.modality_configs[name]
            else:
                cfg = mod_cls.default_config()

            # Propagate pipeline-level base_model to modality configs
            if hasattr(cfg, "base_model") and cfg.base_model != self.config.base_model:
                cfg.base_model = self.config.base_model

            if not cfg.enabled:
                logger.info("Modality '%s' is disabled, skipping.", name)
                continue

            self._modalities.append(mod_cls(cfg))

        # Topological sort by dependencies
        self._modalities = self._topological_sort(self._modalities)

        if self.config.verbosity >= 1:
            names = [m.meta.name for m in self._modalities]
            logger.info("Pipeline modalities (ordered): %s", names)

    def _topological_sort(self, modalities: List[Modality]) -> List[Modality]:
        """Sort modalities so dependencies come before dependents.

        A modality depends on another if its required_inputs overlap
        with the other's output_columns.
        """
        output_map: Dict[str, str] = {}  # column → modality name that produces it
        for m in modalities:
            for col in m.meta.output_columns:
                output_map[col] = m.meta.name

        # Build adjacency: modality name → set of modality names it depends on
        # Include both required and optional inputs — if a modality CAN use
        # another's output, it should run after it.
        deps: Dict[str, Set[str]] = {}
        name_to_mod: Dict[str, Modality] = {}
        for m in modalities:
            name_to_mod[m.meta.name] = m
            deps[m.meta.name] = set()
            all_inputs = m.meta.required_inputs | m.meta.optional_inputs
            for req in all_inputs:
                if req in output_map and output_map[req] != m.meta.name:
                    deps[m.meta.name].add(output_map[req])

        # Kahn's algorithm
        sorted_names: List[str] = []
        in_degree = {n: len(d) for n, d in deps.items()}
        queue = [n for n, d in in_degree.items() if d == 0]

        while queue:
            # Stable sort: process alphabetically for determinism
            queue.sort()
            node = queue.pop(0)
            sorted_names.append(node)
            for n, d in deps.items():
                if node in d:
                    in_degree[n] -= 1
                    if in_degree[n] == 0:
                        queue.append(n)

        if len(sorted_names) != len(modalities):
            missing = set(name_to_mod.keys()) - set(sorted_names)
            raise ValueError(f"Circular dependency among modalities: {missing}")

        return [name_to_mod[n] for n in sorted_names]

    # ------------------------------------------------------------------
    # Validation & transform
    # ------------------------------------------------------------------

    def validate(self, input_columns: Set[str]) -> None:
        """Validate the pipeline against available input columns.

        Raises
        ------
        ValueError
            If any modality has unmet prerequisites.
        """
        available = set(input_columns)
        all_errors: List[str] = []

        for mod in self._modalities:
            errors = mod.validate(available)
            if errors:
                all_errors.extend(errors)
            else:
                # After validation, this modality's outputs become available
                available.update(mod.meta.output_columns)

        if all_errors:
            raise ValueError(
                "Pipeline validation failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all modalities sequentially to the DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame with base-layer prediction columns.

        Returns
        -------
        pl.DataFrame
            Enriched DataFrame with all modality features added.
        """
        self.validate(set(df.columns))

        for mod in self._modalities:
            n_before = df.width
            if self.config.verbosity >= 1:
                logger.info(
                    "Applying modality '%s' (%d input cols)...",
                    mod.meta.name,
                    n_before,
                )

            df = mod.transform(df)
            n_added = df.width - n_before

            if self.config.verbosity >= 1:
                logger.info(
                    "  → added %d columns (total: %d)", n_added, df.width
                )

        return df

    def get_output_schema(self) -> Dict[str, List[str]]:
        """Get the output column schema grouped by modality.

        Returns
        -------
        dict
            Maps modality name → list of output column names.
        """
        return {m.meta.name: list(m.meta.output_columns) for m in self._modalities}

    @property
    def modalities(self) -> List[Modality]:
        """Return the ordered list of active modality instances."""
        return list(self._modalities)

    def __repr__(self) -> str:
        names = [m.meta.name for m in self._modalities]
        return f"FeaturePipeline(modalities={names})"
