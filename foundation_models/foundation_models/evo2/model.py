"""
Evo2 Model Wrapper

Uses the official ``evo2`` Python package from Arc Institute.
Requires CUDA (Linux + GPU). For MPS/CPU, use HyenaDNA instead.

Installation:
    pip install flash-attn==2.8.0.post2 --no-build-isolation
    pip install evo2
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from foundation_models.evo2.config import Evo2Config

logger = logging.getLogger(__name__)


class Evo2Model:
    """Wrapper for Evo2 foundation model via the official ``evo2`` package.

    Handles model loading, tokenization, and embedding extraction.
    Requires CUDA — will raise a clear error on MPS/CPU systems.

    Example::

        >>> config = Evo2Config(model_size="7b")
        >>> model = Evo2Model(config)
        >>> embeddings = model.encode("ATCGATCG...")
        >>> print(embeddings.shape)  # [seq_len, hidden_dim]
    """

    def __init__(self, config: Optional[Evo2Config] = None) -> None:
        self.config = config or Evo2Config()
        self._evo2 = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Evo2 model via the official evo2 package."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Evo2 requires CUDA (Linux + GPU). CUDA is not available on "
                "this system.\n\n"
                "Options:\n"
                "  1. Use HyenaDNA for local MPS/CPU inference:\n"
                "       from foundation_models.hyenadna import HyenaDNAModel\n"
                "  2. Run on a GPU machine (RunPod, SkyPilot):\n"
                "       sky launch foundation_models/configs/skypilot/"
                "extract_embeddings_a40.yaml\n"
            )

        try:
            from evo2 import Evo2
        except ImportError:
            raise ImportError(
                "The 'evo2' package is required. Install with:\n"
                "  pip install flash-attn==2.8.0.post2 --no-build-isolation\n"
                "  pip install evo2"
            )

        checkpoint = self.config.checkpoint_name
        logger.info("Loading Evo2 %s model...", self.config.model_size)
        print(f"Loading Evo2 {self.config.model_size} model...")
        print(f"  Checkpoint: {checkpoint}")

        self._evo2 = Evo2(checkpoint)

        logger.info("Evo2 %s loaded successfully", self.config.model_size)
        print(f"  Loaded Evo2 {self.config.model_size} successfully")

    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension of model."""
        return self.config.hidden_dim

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return torch.device("cuda:0")

    def encode(
        self,
        sequences: Union[str, List[str]],
        layer: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode DNA sequences to per-nucleotide embeddings.

        Args:
            sequences: DNA sequence(s) to encode.
            layer: Internal layer to extract from. Defaults to
                ``config.embedding_layer``.

        Returns:
            Tensor of shape ``[seq_len, hidden_dim]`` for a single sequence,
            or ``[batch, seq_len, hidden_dim]`` for multiple sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            single_input = True
        else:
            single_input = False

        layer_name = layer or self.config.embedding_layer

        all_embeddings = []
        for seq in sequences:
            tokens = self._evo2.tokenizer.tokenize(seq)
            input_ids = (
                torch.tensor(tokens, dtype=torch.int)
                .unsqueeze(0)
                .to("cuda:0")
            )

            with torch.no_grad():
                _, embeddings = self._evo2(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[layer_name],
                )

            emb = embeddings[layer_name][0]  # [seq_len, hidden_dim]
            all_embeddings.append(emb)

        if single_input:
            return all_embeddings[0]

        return torch.stack(all_embeddings)

    def get_likelihood(self, sequence: str) -> float:
        """Get sequence log-likelihood for zero-shot variant effect prediction.

        Args:
            sequence: DNA sequence.

        Returns:
            Log-likelihood of the sequence under the model.
        """
        tokens = self._evo2.tokenizer.tokenize(sequence)
        input_ids = (
            torch.tensor(tokens, dtype=torch.int)
            .unsqueeze(0)
            .to("cuda:0")
        )

        with torch.no_grad():
            outputs, _ = self._evo2(input_ids)
            logits = outputs[0]  # [batch, seq_len, vocab]

        # Shift for autoregressive: predict token t+1 from position t
        shift_logits = logits[0, :-1, :]  # [seq_len-1, vocab]
        shift_targets = input_ids[0, 1:]  # [seq_len-1]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            1, shift_targets.unsqueeze(-1).long()
        ).squeeze(-1)

        return token_log_probs.sum().item()

    def compute_delta_likelihood(self, ref_seq: str, alt_seq: str) -> float:
        """Compute delta log-likelihood for variant effect prediction.

        Args:
            ref_seq: Reference sequence.
            alt_seq: Alternate sequence (with variant).

        Returns:
            Change in log-likelihood (alt - ref).
            Negative delta = variant predicted to be deleterious.
        """
        ref_ll = self.get_likelihood(ref_seq)
        alt_ll = self.get_likelihood(alt_seq)
        return alt_ll - ref_ll

    def __repr__(self) -> str:
        return (
            f"Evo2Model(size={self.config.model_size}, "
            f"checkpoint={self.config.checkpoint_name})"
        )


def load_evo2_model(
    model_size: str = "7b",
    **kwargs,
) -> Evo2Model:
    """Convenience function to load Evo2 model.

    Args:
        model_size: "7b" or "40b".
        **kwargs: Additional config parameters.

    Returns:
        Evo2Model instance.
    """
    config = Evo2Config(model_size=model_size, **kwargs)
    return Evo2Model(config)
