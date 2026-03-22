"""
End-to-end fine-tuning wrapper for foundation models + splice classifier.

Composes a foundation embedding model (e.g., SpliceBERT) with a
:class:`SpliceClassifier` head into a single differentiable ``nn.Module``.
Supports three freezing strategies:

- ``last_n``: Freeze all but the last N encoder layers (default).
- ``lora``: Low-rank adaptation of attention weights via ``peft``.
- ``full``: Unfreeze everything (only practical for small models).

Usage::

    from foundation_models.classifiers.finetune import SpliceFineTuneModel

    model = SpliceFineTuneModel(
        foundation_model=splicebert_model,
        classifier_head=splice_classifier,
        strategy="last_n",
        n_unfreeze=2,
    )

    # Forward: raw DNA → splice logits
    logits = model(["ATCGATCG...", "GCTAGCTA..."])
    # logits shape: [batch, 3, seq_len]
"""

import enum
import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FreezingStrategy(enum.Enum):
    """Available parameter freezing strategies."""

    LAST_N = "last_n"
    LORA = "lora"
    FULL = "full"


def apply_last_n_freeze(model: nn.Module, n_unfreeze: int = 2) -> int:
    """Freeze all encoder layers except the last *n_unfreeze*.

    Works with any model that has ``encoder.layer`` (HuggingFace BERT pattern).

    Args:
        model: The underlying HuggingFace model (e.g., ``BertModel``).
        n_unfreeze: Number of final encoder layers to keep trainable.

    Returns:
        Total number of trainable parameters after freezing.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N encoder layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        n_layers = len(model.encoder.layer)
        start = max(0, n_layers - n_unfreeze)
        for i in range(start, n_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True
        logger.info(
            "Unfroze layers %d-%d of %d (strategy=last_n, n_unfreeze=%d)",
            start, n_layers - 1, n_layers, n_unfreeze,
        )
    else:
        logger.warning(
            "Model has no encoder.layer attribute — freezing everything. "
            "Override freezing manually for non-BERT architectures."
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to attention layers via ``peft``.

    Args:
        model: The underlying HuggingFace model.
        rank: LoRA rank (lower = fewer params, less capacity).
        alpha: LoRA alpha (scaling factor, typically 2x rank).
        dropout: LoRA dropout probability.
        target_modules: Which modules to adapt. Defaults to
            ``["query", "value"]`` (BERT attention projections).

    Returns:
        The PeftModel wrapper with LoRA adapters injected.

    Raises:
        ImportError: If ``peft`` is not installed.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError(
            "LoRA fine-tuning requires the `peft` package. "
            "Install with: pip install peft>=0.4.0"
        )

    if target_modules is None:
        target_modules = ["query", "value"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, config)

    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "Applied LoRA (rank=%d, alpha=%d): %d/%d trainable (%.1f%%)",
        rank, alpha, n_trainable, n_total, 100 * n_trainable / max(n_total, 1),
    )
    return peft_model


def get_param_groups(
    fine_tune_model: "SpliceFineTuneModel",
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-3,
) -> List[Dict]:
    """Build optimizer parameter groups with differential learning rates.

    Args:
        fine_tune_model: The ``SpliceFineTuneModel`` instance.
        encoder_lr: Learning rate for trainable encoder parameters.
        head_lr: Learning rate for classifier head parameters.

    Returns:
        List of param group dicts for ``torch.optim.AdamW``.
    """
    encoder_params = [
        p for p in fine_tune_model.encoder_trainable_params() if p.requires_grad
    ]
    head_params = [
        p for p in fine_tune_model.head.parameters() if p.requires_grad
    ]

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})

    logger.info(
        "Param groups: encoder=%d params (lr=%.1e), head=%d params (lr=%.1e)",
        sum(p.numel() for p in encoder_params),
        encoder_lr,
        sum(p.numel() for p in head_params),
        head_lr,
    )
    return groups


class SpliceFineTuneModel(nn.Module):
    """End-to-end wrapper: foundation model + splice classifier head.

    Composes a foundation embedding model with a
    :class:`~foundation_models.classifiers.splice_classifier.SpliceClassifier`
    into a single differentiable module.

    Args:
        foundation_model: A foundation model implementing ``forward_trainable()``.
        classifier_head: A ``SpliceClassifier`` instance.
        strategy: Freezing strategy (``"last_n"``, ``"lora"``, ``"full"``).
        n_unfreeze: Layers to unfreeze (for ``last_n`` strategy).
        lora_rank: LoRA rank (for ``lora`` strategy).
        lora_alpha: LoRA alpha (for ``lora`` strategy).
        lora_dropout: LoRA dropout (for ``lora`` strategy).
        lora_targets: LoRA target modules (for ``lora`` strategy).
    """

    def __init__(
        self,
        foundation_model: "BaseEmbeddingModel",
        classifier_head: nn.Module,
        strategy: str = "last_n",
        n_unfreeze: int = 2,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self._foundation = foundation_model
        self.head = classifier_head
        self.strategy = FreezingStrategy(strategy)

        # Access the underlying HuggingFace model (e.g., BertModel)
        # Foundation model wrappers store it as self.model
        self._hf_model = getattr(foundation_model, "model", None)
        if self._hf_model is None:
            raise ValueError(
                f"Foundation model {type(foundation_model).__name__} has no "
                "'model' attribute — cannot access underlying HuggingFace model."
            )

        # Apply freezing strategy
        if self.strategy == FreezingStrategy.LAST_N:
            n_encoder_trainable = apply_last_n_freeze(self._hf_model, n_unfreeze)
            logger.info("Encoder trainable params: %d", n_encoder_trainable)

        elif self.strategy == FreezingStrategy.LORA:
            self._hf_model = apply_lora(
                self._hf_model, lora_rank, lora_alpha, lora_dropout, lora_targets,
            )
            # Update the foundation model's reference so forward_trainable uses LoRA
            foundation_model.model = self._hf_model

        elif self.strategy == FreezingStrategy.FULL:
            # Unfreeze everything
            for param in self._hf_model.parameters():
                param.requires_grad = True
            n_trainable = sum(
                p.numel() for p in self._hf_model.parameters() if p.requires_grad
            )
            logger.info("Full fine-tuning: %d encoder params trainable", n_trainable)

        # Log total param summary
        n_head = sum(p.numel() for p in self.head.parameters())
        n_head_trainable = sum(
            p.numel() for p in self.head.parameters() if p.requires_grad
        )
        n_encoder_total = sum(p.numel() for p in self._hf_model.parameters())
        n_encoder_train = sum(
            p.numel() for p in self._hf_model.parameters() if p.requires_grad
        )
        logger.info(
            "SpliceFineTuneModel: encoder=%d/%d trainable, head=%d/%d trainable, "
            "total=%d trainable",
            n_encoder_train, n_encoder_total,
            n_head_trainable, n_head,
            n_encoder_train + n_head_trainable,
        )

    def forward(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """End-to-end forward: DNA strings → splice logits.

        Args:
            sequences: One or more raw DNA strings.

        Returns:
            Logits of shape ``[batch, 3, seq_len]``.
        """
        # Gradient-enabled encoding
        embeddings = self._foundation.forward_trainable(sequences)

        # Handle single sequence: add batch dim
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        # Classifier head
        logits = self.head(embeddings)  # [batch, 3, seq_len]
        return logits

    def encoder_trainable_params(self):
        """Yield trainable parameters from the encoder (not head)."""
        for param in self._hf_model.parameters():
            if param.requires_grad:
                yield param

    def load_head_from_frozen(self, checkpoint_path: str) -> None:
        """Load a pre-trained classifier head from a frozen-embedding checkpoint.

        Useful for warm-starting fine-tuning with a head that was already
        trained on frozen embeddings (from 07_*.py).

        Args:
            checkpoint_path: Path to a ``best_model.pt`` file from 07_*.py.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Load into head (strict=False to handle potential architecture diffs)
        result = self.head.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning("Head load missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Head load unexpected keys: %s", result.unexpected_keys)
        logger.info(
            "Loaded head from frozen checkpoint: %s (%d/%d keys matched)",
            checkpoint_path,
            len(state_dict) - len(result.unexpected_keys),
            len(state_dict),
        )

    def save_checkpoint(
        self,
        path: str,
        epoch: int = -1,
        metrics: Optional[Dict] = None,
    ) -> None:
        """Save full fine-tuning checkpoint (encoder + head + metadata).

        Args:
            path: Output file path.
            epoch: Current epoch number.
            metrics: Validation metrics dict.
        """
        ckpt = {
            "strategy": self.strategy.value,
            "foundation_model": type(self._foundation).__name__,
            "encoder_state_dict": {
                k: v.cpu() for k, v in self._hf_model.state_dict().items()
            },
            "head_state_dict": {
                k: v.cpu() for k, v in self.head.state_dict().items()
            },
            "epoch": epoch,
            "metrics": metrics or {},
        }
        torch.save(ckpt, path)
        logger.info("Saved fine-tune checkpoint: %s (epoch %d)", path, epoch)
