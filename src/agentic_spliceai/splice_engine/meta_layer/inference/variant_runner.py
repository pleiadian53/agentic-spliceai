"""Variant effect prediction pipeline for M4.

Orchestrates the full ref/alt delta computation:
  1. Fetch ref/alt DNA sequences from FASTA
  2. Run base model (OpenSpliceAI) on both → base_scores
  3. Extract dense multimodal features → mm_features
  4. Run meta-layer predict_with_delta() → refined deltas

Produces per-position delta scores [L, 3] with donor/acceptor
gain/loss classification following the SpliceAI convention.

Usage::

    runner = VariantRunner(
        meta_checkpoint="output/meta_layer/m1s/best.pt",
        fasta_path="data/mane/GRCh38/hg38.fa",
    )
    result = runner.run("chr17", 43094464, "A", "C", gene="BRCA1")
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SpliceAI convention: donor=col0, acceptor=col1, neither=col2
_DONOR = 0
_ACCEPTOR = 1
_NEITHER = 2


@dataclass
class SpliceEvent:
    """A single donor/acceptor gain or loss event."""

    event_type: str  # "donor_gain", "donor_loss", "acceptor_gain", "acceptor_loss"
    position: int  # absolute genomic position
    delta: float  # signed delta score at this position
    distance_from_variant: int  # distance in bp from the input variant

    @property
    def is_gain(self) -> bool:
        return "gain" in self.event_type

    @property
    def splice_type(self) -> str:
        return "donor" if "donor" in self.event_type else "acceptor"


@dataclass
class DeltaResult:
    """Full result from variant delta prediction.

    Attributes
    ----------
    chrom : str
        Chromosome.
    position : int
        Variant position (1-based).
    ref : str
        Reference allele.
    alt : str
        Alternate allele.
    gene : str
        Gene name (if provided).
    window_start : int
        Absolute genomic start of the prediction window.
    ref_probs : np.ndarray
        Reference probabilities ``[L, 3]``.
    alt_probs : np.ndarray
        Alternate probabilities ``[L, 3]``.
    delta : np.ndarray
        Per-position deltas ``[L, 3]`` (alt - ref).
    base_ref_probs : np.ndarray
        Base model ref probabilities ``[L, 3]``.
    base_alt_probs : np.ndarray
        Base model alt probabilities ``[L, 3]``.
    base_delta : np.ndarray
        Base model deltas ``[L, 3]``.
    events : list of SpliceEvent
        Detected splice gain/loss events (sorted by |delta|).
    """

    chrom: str
    position: int
    ref: str
    alt: str
    gene: str
    window_start: int
    ref_probs: np.ndarray
    alt_probs: np.ndarray
    delta: np.ndarray
    base_ref_probs: np.ndarray
    base_alt_probs: np.ndarray
    base_delta: np.ndarray
    events: List[SpliceEvent] = field(default_factory=list)

    @property
    def window_length(self) -> int:
        return len(self.delta)

    @property
    def max_donor_gain(self) -> float:
        return float(self.delta[:, _DONOR].max())

    @property
    def max_donor_loss(self) -> float:
        return float(-self.delta[:, _DONOR].min())

    @property
    def max_acceptor_gain(self) -> float:
        return float(self.delta[:, _ACCEPTOR].max())

    @property
    def max_acceptor_loss(self) -> float:
        return float(-self.delta[:, _ACCEPTOR].min())

    @property
    def max_delta(self) -> float:
        """Maximum absolute delta across all channels (SpliceAI-style)."""
        return max(
            self.max_donor_gain, self.max_donor_loss,
            self.max_acceptor_gain, self.max_acceptor_loss,
        )

    def summary(self) -> str:
        """Human-readable summary of the variant effect."""
        lines = [
            f"Variant: {self.chrom}:{self.position} {self.ref}>{self.alt}"
            + (f" ({self.gene})" if self.gene else ""),
            f"Window: {self.window_start}-{self.window_start + self.window_length}",
            f"Max delta scores:",
            f"  DS_DG (donor gain):     {self.max_donor_gain:+.4f}",
            f"  DS_DL (donor loss):     {-self.max_donor_loss:+.4f}",
            f"  DS_AG (acceptor gain):  {self.max_acceptor_gain:+.4f}",
            f"  DS_AL (acceptor loss):  {-self.max_acceptor_loss:+.4f}",
            f"  Max |delta|:            {self.max_delta:.4f}",
        ]
        if self.events:
            lines.append(f"\nDetected events ({len(self.events)}):")
            for e in self.events[:10]:  # show top 10
                lines.append(
                    f"  {e.event_type:<16} at {self.chrom}:{e.position}"
                    f"  Δ={e.delta:+.4f}  ({e.distance_from_variant:+d}bp)"
                )
        return "\n".join(lines)


class VariantRunner:
    """End-to-end variant effect prediction pipeline.

    Parameters
    ----------
    meta_checkpoint : Path or str
        Path to trained M1-S (or M4-S) checkpoint (best.pt).
    fasta_path : Path or str
        Path to reference genome FASTA.
    config_path : Path or str, optional
        Path to model config.pt (default: same dir as checkpoint).
    bigwig_cache_dir : Path or str, optional
        Local BigWig cache for conservation/epigenetic features.
    base_model : str
        Base model for ref/alt scoring.  Any model that follows the
        per-nucleotide 3-class splice site protocol (``[L, 3]`` output)
        can be used.  Default ``"openspliceai"`` (GRCh38/MANE).
        Resolved via ``load_spliceai_models()`` in the resource manager.
        Set to ``"none"`` to skip base model scoring (uniform 1/3 prior).
    device : str
        Inference device ('cpu', 'cuda', 'mps').
    window_size : int
        Prediction window size (default 5001).
    event_threshold : float
        Minimum |delta| to report as a splice event (default 0.1).
    """

    def __init__(
        self,
        meta_checkpoint: str | Path,
        fasta_path: str | Path,
        config_path: Optional[str | Path] = None,
        bigwig_cache_dir: Optional[str | Path] = None,
        base_model: str = "openspliceai",
        device: str = "cpu",
        window_size: int = 5001,
        event_threshold: float = 0.1,
    ) -> None:
        import torch
        from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
            MetaSpliceConfig,
            MetaSpliceModel,
        )

        self.fasta_path = Path(fasta_path)
        self.base_model_name = base_model
        self.device = torch.device(device)
        self.window_size = window_size
        self.event_threshold = event_threshold

        # Load meta model
        meta_checkpoint = Path(meta_checkpoint)
        config_path = Path(config_path) if config_path else meta_checkpoint.parent / "config.pt"
        torch.serialization.add_safe_globals([MetaSpliceConfig])
        self.cfg = torch.load(config_path, map_location="cpu", weights_only=True)
        self.model = MetaSpliceModel(self.cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(meta_checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "VariantRunner: %s (%s, %s params) on %s",
            self.cfg.variant, meta_checkpoint.name, f"{n_params:,}", self.device,
        )

        # Open FASTA
        import pyfaidx
        self.fasta = pyfaidx.Fasta(str(self.fasta_path))

        # Base model predictor (lazy init)
        self._base_predictor = None

        # Dense feature extractor (lazy init)
        self._extractor = None
        self._bigwig_cache_dir = Path(bigwig_cache_dir) if bigwig_cache_dir else None

    def _get_base_models(self) -> Optional[list]:
        """Lazy-load base model ensemble via the resource manager.

        Supports any model that follows the per-nucleotide 3-class splice
        site protocol.  Resolved via ``load_spliceai_models()`` which
        handles path resolution through the genomic registry.

        Returns list of PyTorch models, or None if unavailable or disabled.
        """
        if self._base_predictor is None:
            if self.base_model_name.lower() == "none":
                self._base_predictor = False
                return None
            try:
                from agentic_spliceai.splice_engine.base_layer.prediction.core import (
                    load_spliceai_models,
                )
                models = load_spliceai_models(
                    model_type=self.base_model_name,
                    build="GRCh38",
                    verbosity=0,
                )
                self._base_predictor = models
                logger.info(
                    "Base model (%s) loaded: %d models",
                    self.base_model_name, len(models),
                )
            except Exception as e:
                logger.info(
                    "Base model '%s' not available (%s: %s). "
                    "Using uniform 1/3 prior — delta scores will reflect "
                    "sequence CNN contribution only.",
                    self.base_model_name, type(e).__name__, e,
                )
                self._base_predictor = False
        return self._base_predictor if self._base_predictor is not False else None

    def _run_base_model(
        self, sequence: str, strand: str = "+",
    ) -> Optional[np.ndarray]:
        """Run base model on a DNA sequence, strand-aware.

        The base model expects 5'→3' transcript-order input.  For
        minus-strand genes, the sequence is reverse-complemented before
        prediction, then the output is reversed back to genomic
        coordinate order — matching how the precomputed prediction
        parquets were generated during genome-scale evaluation.

        Returns ``[L, 3]`` in meta-model column order
        ``[donor, acceptor, neither]`` in **genomic coordinate order**,
        or None if base model unavailable.
        """
        from agentic_spliceai.splice_engine.base_layer.prediction.core import (
            prepare_input_sequence,
            predict_with_model,
            SPLICEAI_CONTEXT,
            SPLICEAI_BLOCK_SIZE,
        )

        models = self._get_base_models()
        if models is None:
            return None

        seq_len = len(sequence)

        # For minus-strand genes, RC to transcript order for the base model
        model_seq = self._reverse_complement(sequence) if strand == "-" else sequence

        blocks = prepare_input_sequence(model_seq, context=SPLICEAI_CONTEXT)

        # Average predictions across ensemble
        all_preds = np.mean(
            [predict_with_model(m, blocks) for m in models],
            axis=0,
        )
        # all_preds: [num_blocks, block_size, 3] with [neither=0, acceptor=1, donor=2]

        # Reassemble from blocks to full sequence
        num_blocks = all_preds.shape[0]
        full_len = num_blocks * SPLICEAI_BLOCK_SIZE
        probs = np.zeros((full_len, 3), dtype=np.float32)
        for i in range(num_blocks):
            s = i * SPLICEAI_BLOCK_SIZE
            probs[s:s + SPLICEAI_BLOCK_SIZE] = all_preds[i, :SPLICEAI_BLOCK_SIZE]
        probs = probs[:seq_len]

        if strand == "-":
            # Reverse output back to genomic coordinate order.
            # Following OpenSpliceAI convention: do NOT swap donor/acceptor
            # channels.  The model's channel semantics (neither=0, acceptor=1,
            # donor=2) are invariant to strand — only positions are reversed.
            probs = probs[::-1].copy()

        # Reorder to meta-model convention: [donor, acceptor, neither]
        reordered = np.empty_like(probs)
        reordered[:, _DONOR] = probs[:, 2]     # donor
        reordered[:, _ACCEPTOR] = probs[:, 1]  # acceptor
        reordered[:, _NEITHER] = probs[:, 0]   # neither
        return reordered

    def _get_base_scores(
        self,
        chrom: str,
        window_start: int,
        window_size: int,
        variant_0based: int,
        ref_allele: str,
        alt_allele: str,
        strand: str = "+",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get base model [W, 3] scores for ref and alt sequences.

        Fetches the genomic window, runs the base model (which internally
        handles padding and block-splitting for any sequence length), then
        crops output to the meta-model window.

        Falls back to uniform 1/3 if base model isn't available.
        """
        W = window_size

        if self._get_base_models() is None:
            return (
                np.full((W, 3), 1.0 / 3, dtype=np.float32),
                np.full((W, 3), 1.0 / 3, dtype=np.float32),
            )

        # Fetch the window region for the base model
        ref_seq = self._fetch_sequence(chrom, window_start, window_start + W)
        variant_offset = variant_0based - window_start
        alt_seq = self._mutate_sequence(ref_seq, variant_offset, ref_allele, alt_allele)

        ref_scores = self._run_base_model(ref_seq, strand=strand)  # [W, 3]
        alt_scores = self._run_base_model(alt_seq, strand=strand)

        if ref_scores is None or alt_scores is None:
            return (
                np.full((W, 3), 1.0 / 3, dtype=np.float32),
                np.full((W, 3), 1.0 / 3, dtype=np.float32),
            )

        # Ensure exact window size (pad or trim)
        def _fit_to_window(arr: np.ndarray) -> np.ndarray:
            if len(arr) >= W:
                return arr[:W].astype(np.float32)
            padded = np.full((W, 3), 1.0 / 3, dtype=np.float32)
            padded[:len(arr)] = arr
            return padded

        return _fit_to_window(ref_scores), _fit_to_window(alt_scores)

    def _get_extractor(self):
        """Lazy-init the dense feature extractor."""
        if self._extractor is None:
            from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
                DenseFeatureConfig,
                DenseFeatureExtractor,
            )
            feat_config = DenseFeatureConfig(
                build="GRCh38",
                bigwig_cache_dir=self._bigwig_cache_dir,
            )
            self._extractor = DenseFeatureExtractor(feat_config)
            logger.info("Dense feature extractor initialized (%d channels)", self._extractor.n_channels)
        return self._extractor

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Reverse complement a DNA sequence."""
        comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
        return seq.translate(comp)[::-1]

    def _resolve_chrom(self, chrom: str) -> str:
        """Resolve chromosome name to match FASTA index (handles chr prefix)."""
        if chrom in self.fasta:
            return chrom
        # Try toggling chr prefix
        alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in self.fasta:
            return alt
        raise KeyError(f"Chromosome {chrom} (and {alt}) not found in FASTA")

    def _fetch_sequence(self, chrom: str, start: int, end: int) -> str:
        """Fetch DNA sequence from FASTA (0-based half-open)."""
        resolved = self._resolve_chrom(chrom)
        return str(self.fasta[resolved][start:end]).upper()

    def _mutate_sequence(self, sequence: str, variant_offset: int, ref: str, alt: str) -> str:
        """Apply a variant to a sequence at the given offset."""
        # Verify ref matches
        actual = sequence[variant_offset:variant_offset + len(ref)]
        if actual != ref:
            logger.warning(
                "Ref mismatch at offset %d: expected %s, found %s",
                variant_offset, ref, actual,
            )
        return sequence[:variant_offset] + alt + sequence[variant_offset + len(ref):]

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence → [4, L]."""
        from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
            _one_hot_encode,
        )
        return _one_hot_encode(sequence)

    def run(
        self,
        chrom: str,
        position: int,
        ref: str,
        alt: str,
        gene: Optional[str] = None,
        strand: str = "+",
        use_multimodal: bool = True,
    ) -> DeltaResult:
        """Run variant effect prediction.

        Parameters
        ----------
        chrom : str
            Chromosome (e.g., 'chr17').
        position : int
            1-based variant position.
        ref : str
            Reference allele (plus-strand).
        alt : str
            Alternate allele (plus-strand).
        gene : str, optional
            Gene name (for logging).
        strand : str
            Gene strand ('+' or '-').  For minus-strand genes, sequences
            are reverse-complemented before model input and results are
            mapped back to genomic coordinates.
        use_multimodal : bool
            If True, extract dense multimodal features. If False, use zeros
            (faster, for quick screening).

        Returns
        -------
        DeltaResult
            Per-position delta scores with event detection.
        """
        import torch

        W = self.window_size
        ctx = self.cfg.context_padding

        # Compute genomic window centered on variant
        variant_0based = position - 1  # convert to 0-based
        window_start = variant_0based - W // 2
        window_end = window_start + W

        # Sequence with extra context for the CNN
        seq_start = max(0, window_start - ctx // 2)
        seq_end = window_end + (ctx - ctx // 2)

        # Fetch reference sequence
        ref_seq_full = self._fetch_sequence(chrom, seq_start, seq_end)

        # Apply variant to get alt sequence
        variant_offset_in_seq = variant_0based - seq_start
        alt_seq_full = self._mutate_sequence(ref_seq_full, variant_offset_in_seq, ref, alt)

        # One-hot encode both
        ref_onehot = self._one_hot_encode(ref_seq_full)  # [4, L_seq]
        alt_onehot = self._one_hot_encode(alt_seq_full)  # [4, L_seq]

        # Ensure consistent length (pad if needed)
        total_len = W + ctx
        if ref_onehot.shape[1] < total_len:
            pad_ref = np.zeros((4, total_len), dtype=np.float32)
            pad_alt = np.zeros((4, total_len), dtype=np.float32)
            off = (total_len - ref_onehot.shape[1]) // 2
            pad_ref[:, off:off + ref_onehot.shape[1]] = ref_onehot
            pad_alt[:, off:off + alt_onehot.shape[1]] = alt_onehot
            ref_onehot = pad_ref
            alt_onehot = pad_alt

        # Base model scores for ref and alt sequences
        base_scores_ref, base_scores_alt = self._get_base_scores(
            chrom, window_start, W, variant_0based, ref, alt, strand=strand,
        )

        # Multimodal features (same for ref and alt — epigenomic tracks
        # don't change with a point mutation)
        mm_channels = self.cfg.mm_channels
        if use_multimodal:
            try:
                extractor = self._get_extractor()
                mm_features = extractor.extract_window(chrom, window_start, window_end)
                # Transpose to [C, W] for the model
                mm_ref = mm_features.T.astype(np.float32)
            except Exception as e:
                logger.warning("Feature extraction failed: %s. Using zeros.", e)
                mm_ref = np.zeros((mm_channels, W), dtype=np.float32)
        else:
            mm_ref = np.zeros((mm_channels, W), dtype=np.float32)
        mm_alt = mm_ref.copy()  # identical for SNVs

        # Run meta model predict_with_delta
        with torch.no_grad():
            ref_seq_t = torch.from_numpy(ref_onehot).unsqueeze(0).to(self.device)
            alt_seq_t = torch.from_numpy(alt_onehot).unsqueeze(0).to(self.device)
            ref_base_t = torch.from_numpy(base_scores_ref).unsqueeze(0).to(self.device)
            alt_base_t = torch.from_numpy(base_scores_alt).unsqueeze(0).to(self.device)
            ref_mm_t = torch.from_numpy(mm_ref).unsqueeze(0).to(self.device)
            alt_mm_t = torch.from_numpy(mm_alt).unsqueeze(0).to(self.device)

            ref_probs, alt_probs, delta = self.model.predict_with_delta(
                ref_seq_t, alt_seq_t,
                ref_base_t, alt_base_t,
                ref_mm_t, alt_mm_t,
            )

        ref_probs_np = ref_probs[0].cpu().numpy()  # [W, 3]
        alt_probs_np = alt_probs[0].cpu().numpy()
        delta_np = delta[0].cpu().numpy()

        # Detect splice events
        events = self._detect_events(delta_np, window_start, variant_0based)

        return DeltaResult(
            chrom=chrom,
            position=position,
            ref=ref,
            alt=alt,
            gene=gene or "",
            window_start=window_start,
            ref_probs=ref_probs_np,
            alt_probs=alt_probs_np,
            delta=delta_np,
            base_ref_probs=base_scores_ref,
            base_alt_probs=base_scores_alt,
            base_delta=base_scores_alt - base_scores_ref,
            events=events,
        )

    def _detect_events(
        self,
        delta: np.ndarray,
        window_start: int,
        variant_0based: int,
    ) -> List[SpliceEvent]:
        """Detect donor/acceptor gain/loss events from delta scores.

        Finds positions where |delta| exceeds threshold, classifies as
        gain or loss for donor or acceptor channels.
        """
        events = []
        t = self.event_threshold

        for channel, splice_type in [(_DONOR, "donor"), (_ACCEPTOR, "acceptor")]:
            scores = delta[:, channel]

            # Gains: delta > threshold
            gain_mask = scores > t
            for idx in np.where(gain_mask)[0]:
                abs_pos = window_start + int(idx)
                events.append(SpliceEvent(
                    event_type=f"{splice_type}_gain",
                    position=abs_pos,
                    delta=float(scores[idx]),
                    distance_from_variant=abs_pos - variant_0based,
                ))

            # Losses: delta < -threshold
            loss_mask = scores < -t
            for idx in np.where(loss_mask)[0]:
                abs_pos = window_start + int(idx)
                events.append(SpliceEvent(
                    event_type=f"{splice_type}_loss",
                    position=abs_pos,
                    delta=float(scores[idx]),
                    distance_from_variant=abs_pos - variant_0based,
                ))

        # Sort by magnitude (strongest events first)
        events.sort(key=lambda e: abs(e.delta), reverse=True)
        return events

    def close(self) -> None:
        """Release resources."""
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
