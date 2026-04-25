"""Foundation-model-derived base predictor.

Exposes a trained ``SpliceClassifier`` (from the ``foundation_models``
sub-project) through the ``BasePredictor`` protocol. The full inference
pipeline is:

    gene sequence  ──►  Foundation model (e.g. SpliceBERT)  ──►
        per-nt embeddings  ──►  SpliceClassifier head  ──►
            per-nt 3-class probabilities (neither / acceptor / donor)

The classifier checkpoint is registered in ``settings.yaml`` under
``base_models.splicebert_classifier`` (weights symlinked to
``data/models/splicebert_classifier/``). The foundation model is loaded
via ``foundation_models.base.load_embedding_model``.

This adapter is **optional**: the ``foundation_models`` sub-package has
heavy GPU dependencies and is installed separately. If it is not
available, :func:`make_splicebert_classifier` raises a descriptive
``ImportError`` at adapter instantiation time — the registry gracefully
logs and skips it.

Status
------
Adapter structure mirrors ``examples/foundation_models/07a`` live-
inference path. The checkpoint is tested via that script; the adapter
itself should be validated with::

    agentic-spliceai-base predict \\
        --predictor splicebert_classifier --genes BRCA1

on a machine with ``foundation_models`` installed. Any issues surface
immediately rather than lurking behind indirection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..protocol import PredictionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory (registered via predictors.yaml manifest)
# ---------------------------------------------------------------------------


def make_splicebert_classifier() -> "FoundationModelPredictor":
    """Factory for the registered ``splicebert_classifier`` entry."""
    return FoundationModelPredictor(
        predictor_name="splicebert_classifier",
        foundation_model_name="splicebert",
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class _CheckpointConfig:
    """Resolved config for a foundation-model-derived predictor."""

    predictor_name: str
    foundation_model_name: str
    training_build: str
    annotation_source: str
    training_annotation: Optional[str]
    classifier_architecture: str
    weights_path: Path
    temperature_path: Optional[Path]
    notes: Optional[str]


class FoundationModelPredictor:
    """BasePredictor adapter for foundation-model + trained classifier head.

    The class is generic: ``make_splicebert_classifier`` instantiates it
    with ``foundation_model_name="splicebert"``, and the same class can
    serve other foundation models (Evo2, HyenaDNA) by registering
    additional factories.
    """

    def __init__(
        self,
        predictor_name: str,
        foundation_model_name: str,
    ) -> None:
        self._cfg = _resolve_config(
            predictor_name=predictor_name,
            foundation_model_name=foundation_model_name,
        )

        # Mirror the BasePredictor protocol fields.
        self.name: str = self._cfg.predictor_name
        self.training_build: str = self._cfg.training_build
        self.annotation_source: str = self._cfg.annotation_source

        # Lazily initialised on first predict call.
        self._fm = None
        self._classifier = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # BasePredictor protocol
    # ------------------------------------------------------------------

    def predict_genes(
        self,
        genes: List[str],
        *,
        threshold: float = 0.5,
        verbosity: int = 1,
    ) -> PredictionResult:
        t0 = time.time()

        try:
            self._ensure_loaded(verbosity=verbosity)
        except ImportError as exc:
            return PredictionResult(
                predictor_name=self.name,
                positions=pl.DataFrame(),
                error=str(exc),
                runtime_seconds=time.time() - t0,
            )

        # Prepare gene data via the core library.
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            prepare_gene_data,
        )

        gene_df = prepare_gene_data(
            genes=list(genes),
            build=self.training_build,
            annotation_source=self.annotation_source,
        )

        if gene_df is None or len(gene_df) == 0:
            return PredictionResult(
                predictor_name=self.name,
                positions=pl.DataFrame(),
                missing_genes=set(genes),
                error="No genes found in annotations.",
                runtime_seconds=time.time() - t0,
            )

        processed: set = set()
        missing: set = set(genes)
        frames: List[pl.DataFrame] = []

        for row in _iter_rows(gene_df):
            gene_symbol = row.get("gene_name") or row.get("gene_id")
            chrom = row.get("chrom") or row.get("seqname")
            start = int(row.get("start") or row.get("gene_start") or 0)
            end = int(row.get("end") or row.get("gene_end") or 0)
            sequence = row.get("sequence")

            if not sequence or end <= start:
                continue

            try:
                df = self._predict_gene(
                    gene_name=str(gene_symbol),
                    chrom=str(chrom),
                    start=start,
                    sequence=str(sequence),
                    verbosity=verbosity,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to predict %s: %s", gene_symbol, exc)
                continue

            if df is None or df.height == 0:
                continue
            frames.append(df)
            processed.add(str(gene_symbol))
            missing.discard(str(gene_symbol))

        positions = pl.concat(frames) if frames else pl.DataFrame()

        return PredictionResult(
            predictor_name=self.name,
            positions=positions,
            processed_genes=processed,
            missing_genes=missing,
            runtime_seconds=time.time() - t0,
            metadata={
                "threshold": threshold,
                "foundation_model": self._cfg.foundation_model_name,
                "weights_path": str(self._cfg.weights_path),
                "architecture": self._cfg.classifier_architecture,
            },
        )

    def predict_chromosomes(
        self,
        chromosomes: List[str],
        *,
        threshold: float = 0.5,
        verbosity: int = 1,
    ) -> PredictionResult:
        # Resolve chromosomes -> genes, then delegate.
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            filter_by_chromosomes,
            load_gene_annotations,
        )

        gene_df = load_gene_annotations(
            build=self.training_build,
            annotation_source=self.annotation_source,
        )
        filtered = filter_by_chromosomes(gene_df, chromosomes, build=self.training_build)
        gene_col = "gene_name" if "gene_name" in filtered.columns else "gene_id"
        genes = sorted({str(g) for g in filtered[gene_col].to_list() if g})
        return self.predict_genes(
            genes=genes, threshold=threshold, verbosity=verbosity,
        )

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "training_build": self.training_build,
            "annotation_source": self.annotation_source,
            "training_annotation": self._cfg.training_annotation,
            "foundation_model": self._cfg.foundation_model_name,
            "classifier_architecture": self._cfg.classifier_architecture,
            "weights_path": str(self._cfg.weights_path),
            "notes": self._cfg.notes,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self, *, verbosity: int) -> None:
        """Load foundation model + classifier on first use."""
        if self._classifier is not None:
            return

        try:
            from foundation_models.base import load_embedding_model
            from foundation_models.classifiers.splice_classifier import (
                SpliceClassifier,
            )
        except ImportError as exc:
            raise ImportError(
                f"The {self.name!r} predictor requires the "
                f"'foundation_models' sub-package: `pip install -e "
                f"./foundation_models`. Original error: {exc}"
            ) from exc

        import torch

        self._device = _pick_device()
        if verbosity >= 1:
            logger.info(
                "Loading foundation model %r on %s...",
                self._cfg.foundation_model_name, self._device,
            )

        self._fm = load_embedding_model(self._cfg.foundation_model_name)
        fm_meta = self._fm.metadata()
        fm_hidden = fm_meta.hidden_dim

        # Load checkpoint — training scripts save a full dict including
        # architecture hyperparams; we prefer those over the settings.yaml
        # defaults since they describe the *actual* trained head.
        ckpt = torch.load(self._cfg.weights_path, map_location=self._device)
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Unexpected checkpoint format at {self._cfg.weights_path}: "
                f"expected a dict, got {type(ckpt).__name__}."
            )

        # Extract the state_dict under either the new or the legacy key.
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Raw state_dict (keys look like "head.0.weight" etc.).
            state_dict = ckpt

        # Pull the architecture config from the checkpoint when available;
        # fall back to settings.yaml / protocol defaults otherwise.
        architecture = ckpt.get("architecture", self._cfg.classifier_architecture)
        input_dim = ckpt.get("input_dim", fm_hidden)
        if input_dim != fm_hidden:
            logger.warning(
                "Checkpoint input_dim=%d does not match foundation model "
                "hidden_dim=%d — predictions will likely be garbage.",
                input_dim, fm_hidden,
            )
        hidden_dim = ckpt.get("hidden_dim", 128)
        num_blocks = ckpt.get("num_blocks", 3)
        kernel_size = ckpt.get("kernel_size", 11)
        dilations = ckpt.get("dilations", None)
        dropout = ckpt.get("dropout", 0.1)

        if verbosity >= 1:
            logger.info(
                "Building classifier head (arch=%s, input_dim=%d, "
                "hidden_dim=%d, num_blocks=%d)...",
                architecture, input_dim, hidden_dim, num_blocks,
            )
        classifier = SpliceClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            architecture=architecture,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dilations=list(dilations) if dilations is not None else None,
            dropout=dropout,
        )

        # The trained state_dict may contain a calibrated ``temperature``
        # parameter that SpliceClassifier does not register at init (set
        # later by ``calibrate()``). Pop it and apply after the weight load.
        state_dict = dict(state_dict)  # copy so we can mutate safely
        temp_from_state = state_dict.pop("temperature", None)

        classifier.load_state_dict(state_dict)

        # Temperature scaling priority:
        #   1. value inline in the state_dict (pop'd above)
        #   2. top-level ``temperature`` in the checkpoint dict
        #   3. separate temperature.pt file
        temp = temp_from_state
        if temp is None:
            temp = ckpt.get("temperature")
        if temp is None and self._cfg.temperature_path and Path(self._cfg.temperature_path).exists():
            temp_state = torch.load(
                self._cfg.temperature_path, map_location=self._device,
            )
            if isinstance(temp_state, dict) and "temperature" in temp_state:
                temp = temp_state["temperature"]
            else:
                temp = temp_state

        if temp is not None:
            classifier.temperature = torch.nn.Parameter(
                torch.as_tensor(temp).float(), requires_grad=False,
            )
            if verbosity >= 1:
                logger.info("Loaded temperature calibration.")

        classifier = classifier.to(self._device).eval()
        self._classifier = classifier

    def _predict_gene(
        self,
        *,
        gene_name: str,
        chrom: str,
        start: int,
        sequence: str,
        verbosity: int,
    ) -> Optional[pl.DataFrame]:
        """Run live inference for a single gene."""
        import torch
        from foundation_models.utils.chunking import chunk_sequence, stitch_embeddings

        fm = self._fm
        classifier = self._classifier
        device = self._device or "cpu"
        hidden_dim = fm.metadata().hidden_dim
        max_context = int(getattr(fm.metadata(), "max_context", 512))

        gene_len = len(sequence)
        if gene_len < 16:
            return None

        # Per-nt embeddings: either direct encode or chunked+stitched.
        if gene_len <= max_context:
            with torch.no_grad():
                emb = fm.encode(sequence)
            if hasattr(emb, "cpu"):
                emb = emb.detach().cpu().float().numpy()
            if emb.shape[0] < gene_len:
                padded = np.zeros((gene_len, hidden_dim), dtype=np.float32)
                padded[: emb.shape[0]] = emb
                emb = padded
            elif emb.shape[0] > gene_len:
                emb = emb[:gene_len]
        else:
            overlap = min(128, max_context // 8)
            chunks = chunk_sequence(
                sequence, chunk_size=max_context, overlap=overlap,
            )
            chunk_embeddings = []
            for chunk in chunks:
                with torch.no_grad():
                    c_emb = fm.encode(chunk.sequence)
                if hasattr(c_emb, "cpu"):
                    c_emb = c_emb.detach().cpu().float().numpy()
                c_len = len(chunk.sequence)
                if c_emb.shape[0] < c_len:
                    padded = np.zeros((c_len, hidden_dim), dtype=np.float32)
                    padded[: c_emb.shape[0]] = c_emb
                    c_emb = padded
                elif c_emb.shape[0] > c_len:
                    c_emb = c_emb[:c_len]
                chunk_embeddings.append(c_emb)
            emb = stitch_embeddings(chunks, chunk_embeddings, gene_len, hidden_dim)

        # Run classifier in windows. MPS caps Conv1d output channels at
        # 65536, so a single forward pass fails on genes larger than that.
        # We use non-overlapping windows of CLASSIFIER_WINDOW bp; edge
        # positions get one classifier pass, no stitching needed.
        CLASSIFIER_WINDOW = 16384

        donor_parts: List[np.ndarray] = []
        acceptor_parts: List[np.ndarray] = []
        neither_parts: List[np.ndarray] = []

        pos = 0
        while pos < gene_len:
            end_pos = min(pos + CLASSIFIER_WINDOW, gene_len)
            win_emb = emb[pos:end_pos]
            if win_emb.shape[0] == 0:
                break
            emb_t = torch.as_tensor(win_emb, dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = classifier.predict(emb_t)
            donor_parts.append(np.asarray(preds["donor_prob"]).flatten())
            acceptor_parts.append(np.asarray(preds["acceptor_prob"]).flatten())
            if "neither_prob" in preds:
                neither_parts.append(np.asarray(preds["neither_prob"]).flatten())
            pos = end_pos

        if not donor_parts:
            return None

        donor = np.concatenate(donor_parts)
        acceptor = np.concatenate(acceptor_parts)
        if neither_parts:
            neither = np.concatenate(neither_parts)
        else:
            neither = (1.0 - donor - acceptor).astype(np.float32)

        n = int(min(donor.shape[0], acceptor.shape[0], gene_len))
        positions = np.arange(start, start + n, dtype=np.int64)

        return pl.DataFrame({
            "chrom": [chrom] * n,
            "position": positions,
            "gene": [gene_name] * n,
            "donor_prob": donor[:n].astype(np.float32),
            "acceptor_prob": acceptor[:n].astype(np.float32),
            "neither_prob": neither[:n].astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_config(
    *,
    predictor_name: str,
    foundation_model_name: str,
) -> _CheckpointConfig:
    """Look up the predictor entry in settings.yaml and resolve paths."""
    from agentic_spliceai.splice_engine.resources.model_resources import (
        get_model_info,
        get_model_resources,
    )

    info = get_model_info(predictor_name)
    resources = get_model_resources(predictor_name)

    weights_dir_raw = info.get("weights_dir") or f"data/models/{predictor_name}/model"
    weights_dir = Path(weights_dir_raw)
    if not weights_dir.is_absolute():
        weights_dir = _project_root() / weights_dir

    weights_path = weights_dir / "best_model.pt"
    temperature_path = weights_dir / "temperature.pt"
    if not temperature_path.exists():
        temperature_path = None  # type: ignore[assignment]

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found for {predictor_name}: {weights_path}. "
            f"Expected settings.yaml base_models.{predictor_name}.weights_dir "
            f"to point at the directory containing best_model.pt."
        )

    return _CheckpointConfig(
        predictor_name=predictor_name,
        foundation_model_name=info.get("foundation_model", foundation_model_name),
        training_build=resources.build,
        annotation_source=resources.annotation_source,
        training_annotation=info.get("training_annotation"),
        classifier_architecture=info.get("classifier_architecture", "dilated_cnn"),
        weights_path=weights_path,
        temperature_path=temperature_path,
        notes=info.get("notes"),
    )


def _project_root() -> Path:
    """Locate the project root (parent of ``src/``).

    File layout:
      src/agentic_spliceai/applications/base_layer/predictors/from_checkpoint.py
      └─0── └─1──────────── └─2────────── └─3──────── └─4──────── └─file

    parents[5] is project root.
    """
    return Path(__file__).resolve().parents[5]


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _iter_rows(gene_df):
    """Iterate rows of a pandas or polars DataFrame as plain dicts."""
    if hasattr(gene_df, "iter_rows"):
        for row in gene_df.iter_rows(named=True):
            yield dict(row)
    else:
        # pandas
        for _, row in gene_df.iterrows():
            yield row.to_dict()
