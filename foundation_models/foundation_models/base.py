"""
Base classes and registry for foundation embedding models.

Provides:
- ``BaseEmbeddingModel`` — abstract base class that all embedding-producing
  foundation models (Evo2, HyenaDNA, SpliceBERT, etc.) must inherit from.
- ``ModelMetadata`` — frozen dataclass describing model capabilities.
- ``load_embedding_model()`` — factory that loads a model by name.
- ``get_model_metadata()`` — get metadata without loading model weights.
- ``list_available_models()`` — list registered model names.
"""

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMetadata:
    """Static metadata describing a foundation embedding model.

    Attributes:
        name: Human-readable model name (e.g. "evo2-7b", "splicebert").
        model_type: Architecture family — determines extraction strategy.
            "causal": last-position embedding is richest (Evo2, HyenaDNA).
            "bidirectional": all positions see full context (SpliceBERT, DNABERT-2).
        hidden_dim: Embedding dimension per position.
        max_context: Maximum input sequence length in nucleotides.
        tokenization: How the model tokenizes DNA.
            "character": 1 token per nucleotide (Evo2, HyenaDNA, SpliceBERT).
            "bpe": byte-pair encoding, tokens span variable nucleotides (DNABERT-2).
        supports_layer_selection: Whether an internal layer can be specified
            for embedding extraction (e.g. Evo2 ``blocks.26``).
    """

    name: str
    model_type: Literal["causal", "bidirectional"]
    hidden_dim: int
    max_context: int
    tokenization: Literal["character", "bpe"] = "character"
    supports_layer_selection: bool = False

    @property
    def feature_dim_for_classifier(self) -> int:
        """Feature dimension produced by the sparse exon classifier pipeline.

        Causal models use dual-strand extraction (forward + reverse) →
        ``2 × hidden_dim``.  Bidirectional models use a single centered
        window → ``1 × hidden_dim``.
        """
        if self.model_type == "causal":
            return self.hidden_dim * 2
        return self.hidden_dim


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseEmbeddingModel(ABC):
    """Abstract base class for all foundation embedding models.

    Subclasses must implement:
    - ``encode()`` — produce per-position embeddings from DNA sequences.
    - ``metadata()`` — return a ``ModelMetadata`` describing the model.
    - ``hidden_dim`` property — embedding dimension.
    - ``device`` property — current torch device.

    Optional overrides:
    - ``forward_trainable()`` — gradient-enabled forward pass for fine-tuning.

    Concrete methods provided:
    - ``feature_dim_for_classifier`` — classifier input dim (model-type aware).
    - ``__repr__`` — human-readable string.
    """

    @abstractmethod
    def encode(
        self,
        sequences: Union[str, List[str]],
        layer: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode DNA sequences to per-position embeddings.

        Args:
            sequences: One or more DNA sequences (uppercase ACGTN).
            layer: Optional internal layer to extract from.  Only meaningful
                for models where ``metadata().supports_layer_selection`` is
                True.  Ignored otherwise.

        Returns:
            ``[seq_len, hidden_dim]`` for a single sequence, or
            ``[batch, seq_len, hidden_dim]`` for multiple sequences.
        """
        ...

    def forward_trainable(
        self,
        sequences: Union[str, List[str]],
    ) -> torch.Tensor:
        """Gradient-enabled forward pass for fine-tuning.

        Same interface as :meth:`encode` but without ``torch.no_grad()``,
        allowing gradients to flow back through the model's parameters.

        Override this in subclasses that support fine-tuning. The default
        raises ``NotImplementedError``.

        Args:
            sequences: One or more DNA sequences (uppercase ACGTN).

        Returns:
            ``[seq_len, hidden_dim]`` for a single sequence, or
            ``[batch, seq_len, hidden_dim]`` for multiple sequences.

        Raises:
            NotImplementedError: If the model does not support fine-tuning.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support fine-tuning. "
            "Override forward_trainable() to enable gradient flow."
        )

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return metadata describing this model instance."""
        ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Embedding dimension per position."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Torch device the model is loaded on."""
        ...

    # -- concrete helpers ----------------------------------------------------

    @property
    def feature_dim_for_classifier(self) -> int:
        """Feature dimension for the sparse exon classifier pipeline."""
        return self.metadata().feature_dim_for_classifier

    def __repr__(self) -> str:
        meta = self.metadata()
        return (
            f"{self.__class__.__name__}("
            f"name={meta.name!r}, type={meta.model_type}, "
            f"hidden_dim={meta.hidden_dim}, device={self.device})"
        )


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

# Maps model name → (module_path, class_name, metadata_factory)
# metadata_factory: callable(**kwargs) → ModelMetadata (lightweight, no GPU)
_REGISTRY: Dict[str, Tuple[str, str, Callable[..., ModelMetadata]]] = {}


def _register(
    name: str,
    module_path: str,
    class_name: str,
    metadata_factory: Callable[..., ModelMetadata],
) -> None:
    """Register a model in the global registry."""
    _REGISTRY[name] = (module_path, class_name, metadata_factory)


# -- Evo2 metadata (no import needed) --------------------------------------

def _evo2_metadata(**kwargs) -> ModelMetadata:
    size = kwargs.get("model_size", "7b")
    hidden = {"7b": 4096, "40b": 8192}.get(size, 4096)
    return ModelMetadata(
        name=f"evo2-{size}",
        model_type="causal",
        hidden_dim=hidden,
        max_context=kwargs.get("max_length", 32768),
        tokenization="character",
        supports_layer_selection=True,
    )


_register("evo2", "foundation_models.evo2.model", "Evo2Model", _evo2_metadata)


# -- HyenaDNA metadata (no import needed) ----------------------------------

_HYENADNA_VARIANTS = {
    "tiny-1k": (1_024, 128),
    "small-32k": (32_768, 256),
    "medium-160k": (160_000, 256),
    "medium-450k": (450_000, 256),
    "large-1m": (1_000_000, 256),
}


def _hyenadna_metadata(**kwargs) -> ModelMetadata:
    size = kwargs.get("model_size", "medium-160k")
    max_ctx, hidden = _HYENADNA_VARIANTS.get(size, (160_000, 256))
    return ModelMetadata(
        name=f"hyenadna-{size}",
        model_type="causal",
        hidden_dim=hidden,
        max_context=max_ctx,
        tokenization="character",
        supports_layer_selection=False,
    )


_register(
    "hyenadna",
    "foundation_models.hyenadna.model",
    "HyenaDNAModel",
    _hyenadna_metadata,
)


# -- SpliceBERT metadata (no import needed) ---------------------------------

_SPLICEBERT_VARIANTS = {
    "splicebert": (1024, 512),
    "splicebert.510nt": (510, 512),
}


def _splicebert_metadata(**kwargs) -> ModelMetadata:
    variant = kwargs.get("model_variant", "splicebert")
    max_ctx, hidden = _SPLICEBERT_VARIANTS.get(variant, (1024, 512))
    # Avoid stutter: "splicebert-splicebert" → just "splicebert"
    name = variant if variant != "splicebert" else "splicebert"
    return ModelMetadata(
        name=name,
        model_type="bidirectional",
        hidden_dim=hidden,
        max_context=max_ctx,
        tokenization="character",
        supports_layer_selection=False,
    )


_register(
    "splicebert",
    "foundation_models.splicebert.model",
    "SpliceBERTModel",
    _splicebert_metadata,
)


# -- DNABERT-2 metadata (no import needed) ---------------------------------

def _dnabert2_metadata(**kwargs) -> ModelMetadata:
    return ModelMetadata(
        name="dnabert2-117M",
        model_type="bidirectional",
        hidden_dim=768,
        max_context=kwargs.get("max_length", 4000),
        tokenization="bpe",
        supports_layer_selection=False,
    )


_register(
    "dnabert",
    "foundation_models.dnabert.model",
    "DNABERT2Model",
    _dnabert2_metadata,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_available_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


def get_model_metadata(name: str, **kwargs) -> ModelMetadata:
    """Get model metadata without loading weights.

    Args:
        name: Registered model name (e.g. "evo2", "splicebert").
        **kwargs: Model-specific options (e.g. ``model_size="7b"``).

    Returns:
        ModelMetadata instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    _, _, meta_fn = _REGISTRY[name]
    return meta_fn(**kwargs)


def load_embedding_model(name: str, **kwargs) -> "BaseEmbeddingModel":
    """Load a foundation embedding model by name.

    Uses lazy imports so that only the requested model's dependencies are
    loaded (e.g. selecting SpliceBERT does not trigger Evo2's CUDA check).

    Args:
        name: Registered model name (e.g. "evo2", "splicebert", "hyenadna").
        **kwargs: Passed to the model's config constructor.  Common options:
            ``model_size`` (str), ``device`` (str), ``embedding_layer`` (str).

    Returns:
        Model instance satisfying the ``BaseEmbeddingModel`` interface.

    Raises:
        ValueError: If *name* is not registered.
        ImportError: If the model's dependencies are not installed.
    """
    if name not in _REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    module_path, class_name, _ = _REGISTRY[name]
    logger.info("Loading model '%s' from %s.%s", name, module_path, class_name)

    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    return model_cls(**kwargs)
