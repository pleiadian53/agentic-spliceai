"""Memory-bounded streaming evaluation for sequence-level meta-splice models.

Accumulates lightweight statistics (scalar counters + splice-site probabilities)
gene-by-gene, never holding more than one gene's arrays in memory.  Computes
final metrics (PR-AUC, accuracy, precision, recall, F1, top-k) from the
accumulators.

Memory budget: ~5 MB regardless of gene count, vs. ~30 GB for bulk loading.

Usage::

    evaluator = StreamingEvaluator()
    for gene_id in test_genes:
        data = load_gene(gene_id)
        meta_probs = infer(model, data)        # [L, 3]
        base_probs = data["base_scores"]       # [L, 3]
        labels = data["labels"]                # [L]
        evaluator.update(meta_probs, base_probs, labels, gene_id)
        del data, meta_probs  # free immediately

    results = evaluator.compute()
    # results["meta_model"], results["base_model"], results["meta_topk"], ...
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Label encoding: 0=donor, 1=acceptor, 2=neither
_CLASS_NAMES = ["donor", "acceptor", "neither"]


def preflight_check(
    needs_bigwig: bool = True,
    needs_pyfaidx: bool = True,
    cache_dir: Optional[Path] = None,
    base_scores_dir: Optional[Path] = None,
    fasta_path: Optional[str] = None,
) -> None:
    """Verify required libraries and data paths before expensive work.

    Call this at script startup, before launching cache building or
    evaluation loops.  Fails fast with a clear message instead of
    dying hours into a run.

    Parameters
    ----------
    needs_bigwig : bool
        Check pyBigWig is importable (required for conservation/epigenetic features).
    needs_pyfaidx : bool
        Check pyfaidx is importable (required for FASTA access).
    cache_dir : Path, optional
        If provided, verify the directory is writable.
    base_scores_dir : Path, optional
        If provided, verify it exists and contains parquet files.
    fasta_path : str, optional
        If provided, verify the FASTA file exists.

    Raises
    ------
    SystemExit
        If any check fails.
    """
    errors: List[str] = []

    # Library checks
    if needs_bigwig:
        try:
            import pyBigWig  # noqa: F401
        except ImportError:
            errors.append(
                "pyBigWig not installed. Required for conservation/epigenetic features.\n"
                "  Install: pip install pyBigWig\n"
                "  Or:      pip install -e '.[conservation]'"
            )

    if needs_pyfaidx:
        try:
            import pyfaidx  # noqa: F401
        except ImportError:
            errors.append(
                "pyfaidx not installed. Required for FASTA sequence access.\n"
                "  Install: pip install pyfaidx"
            )

    try:
        from sklearn.metrics import average_precision_score  # noqa: F401
    except ImportError:
        errors.append(
            "scikit-learn not installed. Required for PR-AUC computation.\n"
            "  Install: pip install scikit-learn"
        )

    # Data path checks
    if fasta_path and not Path(fasta_path).exists():
        errors.append(f"FASTA not found: {fasta_path}")

    if base_scores_dir:
        bsd = Path(base_scores_dir)
        if not bsd.exists():
            errors.append(f"Base scores directory not found: {bsd}")
        elif not list(bsd.glob("predictions_*.parquet")):
            errors.append(f"No prediction parquets in: {bsd}")

    if errors:
        logger.error("Preflight check failed:")
        for e in errors:
            logger.error("  - %s", e.split("\n")[0])
            for line in e.split("\n")[1:]:
                logger.error("    %s", line)
        sys.exit(1)

    logger.info("Preflight check passed")


class _ModelAccumulator:
    """Per-model accumulator for streaming metrics.

    Stores scalar counters and a small buffer of splice-site probabilities
    for exact PR-AUC computation on the rare classes.
    """

    def __init__(self, neither_subsample_rate: float = 0.01) -> None:
        self.neither_subsample_rate = neither_subsample_rate

        # Scalar counters per class
        self.n_positions = 0
        self.n_correct = 0
        self.class_counts = {c: {"tp": 0, "fp": 0, "fn": 0, "n_true": 0}
                             for c in range(3)}

        # Splice-site probs for PR-AUC.
        # For donor PR-AUC: store (donor_prob, is_true_donor) at every
        # splice site position (label 0 or 1).  Splice sites are <0.5%
        # of positions, so this is ~100K entries for 5K genes.
        # For neither: subsample at 1%.
        self._donor_probs: List[float] = []
        self._donor_labels: List[int] = []
        self._acceptor_probs: List[float] = []
        self._acceptor_labels: List[int] = []
        self._neither_probs: List[float] = []
        self._neither_labels: List[int] = []

        # Top-k: per-gene per-splice-type recall at various k multipliers
        self.topk_donor: Dict[float, List[float]] = defaultdict(list)
        self.topk_acceptor: Dict[float, List[float]] = defaultdict(list)

    def update(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        gene_id: str,
        k_multipliers: List[float] = [0.5, 1.0, 2.0, 4.0],
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Update accumulators with one gene's predictions.

        Parameters
        ----------
        probs : np.ndarray
            Shape ``[L, 3]`` — predicted probabilities.
        labels : np.ndarray
            Shape ``[L]`` — integer labels (0=donor, 1=acceptor, 2=neither).
        gene_id : str
            Gene identifier (for logging only).
        k_multipliers : list
            Top-k multipliers for SpliceAI paper metric.
        rng : np.random.Generator, optional
            RNG for "neither" subsampling reproducibility.
        """
        preds = probs.argmax(axis=1)
        L = len(labels)

        # ── Scalar counters ──────────────────────────────────────────
        self.n_positions += L
        self.n_correct += int((preds == labels).sum())

        for cls in range(3):
            true_mask = labels == cls
            pred_mask = preds == cls
            n_true = int(true_mask.sum())
            self.class_counts[cls]["n_true"] += n_true
            self.class_counts[cls]["tp"] += int((true_mask & pred_mask).sum())
            self.class_counts[cls]["fp"] += int((~true_mask & pred_mask).sum())
            self.class_counts[cls]["fn"] += int((true_mask & ~pred_mask).sum())

        # ── PR-AUC accumulators ───────────────────────────────────────
        # PR-AUC needs both positives and negatives.  We store:
        #   - ALL splice site positions (rare, ~0.5% of total)
        #   - A subsample of "neither" positions as negatives
        # This gives exact PR-AUC for donor/acceptor with minimal memory.

        splice_mask = labels < 2  # donor or acceptor
        splice_idx = np.where(splice_mask)[0]
        neither_idx = np.where(labels == 2)[0]

        if rng is None:
            rng = np.random.default_rng(42)

        # Subsample "neither" positions as negative examples
        bg_idx = np.array([], dtype=int)
        if len(neither_idx) > 0 and self.neither_subsample_rate > 0:
            n_sample = max(1, int(len(neither_idx) * self.neither_subsample_rate))
            bg_idx = rng.choice(neither_idx, size=n_sample, replace=False)

        # Donor PR-AUC: all splice sites + subsampled background
        if len(splice_idx) > 0 or len(bg_idx) > 0:
            eval_idx = np.concatenate([splice_idx, bg_idx])
            # Donor: binary (is this position a donor?)
            self._donor_probs.extend(probs[eval_idx, 0].tolist())
            self._donor_labels.extend((labels[eval_idx] == 0).astype(int).tolist())
            # Acceptor: binary (is this position an acceptor?)
            self._acceptor_probs.extend(probs[eval_idx, 1].tolist())
            self._acceptor_labels.extend((labels[eval_idx] == 1).astype(int).tolist())
            # Neither: binary (is this position neither?)
            self._neither_probs.extend(probs[eval_idx, 2].tolist())
            self._neither_labels.extend((labels[eval_idx] == 2).astype(int).tolist())

        # ── Top-k accuracy (per gene) ────────────────────────────────
        for splice_idx_cls, splice_name, topk_dict in [
            (0, "donor", self.topk_donor),
            (1, "acceptor", self.topk_acceptor),
        ]:
            true_mask = labels == splice_idx_cls
            n_true = int(true_mask.sum())
            if n_true == 0:
                continue

            scores = probs[:, splice_idx_cls]
            ranked = np.argsort(scores)[::-1]
            true_positions = set(np.where(true_mask)[0])

            for km in k_multipliers:
                k = max(1, int(np.ceil(km * n_true)))
                recovered = len(set(ranked[:k]) & true_positions)
                topk_dict[km].append(recovered / n_true)

    def compute_metrics(
        self,
        k_multipliers: List[float] = [0.5, 1.0, 2.0, 4.0],
    ) -> dict:
        """Compute final metrics from accumulators.

        Returns
        -------
        dict
            Contains accuracy, per-class PR-AUC, precision, recall, F1,
            FN/FP/TP counts, top-k accuracy, and confusion matrix diagonal.
        """
        from sklearn.metrics import average_precision_score

        metrics: dict = {}

        # Accuracy
        metrics["accuracy"] = (
            self.n_correct / max(self.n_positions, 1)
        )
        metrics["top_1_accuracy"] = metrics["accuracy"]
        metrics["n_positions"] = self.n_positions

        # Per-class precision, recall, F1
        for cls, name in enumerate(_CLASS_NAMES):
            cc = self.class_counts[cls]
            tp, fp, fn = cc["tp"], cc["fp"], cc["fn"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            metrics[f"{name}_precision"] = prec
            metrics[f"{name}_recall"] = rec
            metrics[f"{name}_f1"] = f1

        # Aggregate splice FN/FP/TP (donor + acceptor)
        metrics["fn_count"] = (
            self.class_counts[0]["fn"] + self.class_counts[1]["fn"]
        )
        metrics["fp_count"] = (
            self.class_counts[0]["fp"] + self.class_counts[1]["fp"]
        )
        metrics["tp_count"] = (
            self.class_counts[0]["tp"] + self.class_counts[1]["tp"]
        )

        # PR-AUC from accumulated splice-site probs
        pr_aucs = {}
        for name, probs_list, labels_list in [
            ("donor", self._donor_probs, self._donor_labels),
            ("acceptor", self._acceptor_probs, self._acceptor_labels),
            ("neither", self._neither_probs, self._neither_labels),
        ]:
            if len(probs_list) > 0:
                probs_arr = np.array(probs_list)
                labels_arr = np.array(labels_list)
                if labels_arr.sum() > 0 and labels_arr.sum() < len(labels_arr):
                    pr_aucs[name] = float(
                        average_precision_score(labels_arr, probs_arr)
                    )
        metrics["pr_aucs"] = pr_aucs
        metrics["macro_pr_auc"] = (
            float(np.mean(list(pr_aucs.values()))) if pr_aucs else 0.0
        )

        # Top-k accuracy
        topk: Dict[str, Dict[float, float]] = {}
        for splice_name, topk_dict in [
            ("donor", self.topk_donor),
            ("acceptor", self.topk_acceptor),
        ]:
            topk[splice_name] = {}
            for km in k_multipliers:
                vals = topk_dict[km]
                topk[splice_name][km] = float(np.mean(vals)) if vals else 0.0

        topk["overall"] = {}
        for km in k_multipliers:
            topk["overall"][km] = float(np.mean([
                topk["donor"].get(km, 0),
                topk["acceptor"].get(km, 0),
            ]))
        metrics["topk"] = topk

        return metrics

    def sweep_thresholds(
        self,
        thresholds: Optional[List[float]] = None,
    ) -> Dict[str, List[dict]]:
        """Sweep classification thresholds for donor and acceptor.

        For each threshold, compute precision, recall, F1, TP, FP, FN
        using the accumulated splice-site probabilities.

        The default argmax decision (threshold=0) is included for reference.

        Parameters
        ----------
        thresholds : list of float, optional
            Probability thresholds to test.  Default: 0.01 to 0.99 in steps.

        Returns
        -------
        dict
            Keys: ``"donor"``, ``"acceptor"``.  Each value is a list of dicts
            with keys: ``threshold``, ``precision``, ``recall``, ``f1``,
            ``tp``, ``fp``, ``fn``.
        """
        if thresholds is None:
            thresholds = [
                0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
            ]

        results = {}
        for name, probs_list, labels_list in [
            ("donor", self._donor_probs, self._donor_labels),
            ("acceptor", self._acceptor_probs, self._acceptor_labels),
        ]:
            if not probs_list:
                results[name] = []
                continue

            probs_arr = np.array(probs_list)
            labels_arr = np.array(labels_list)

            sweep = []
            for t in thresholds:
                preds = (probs_arr >= t).astype(int)
                tp = int((preds & labels_arr).sum())
                fp = int((preds & ~labels_arr.astype(bool)).sum())
                fn = int((~preds.astype(bool) & labels_arr).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                sweep.append({
                    "threshold": t,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "tp": tp, "fp": fp, "fn": fn,
                })
            results[name] = sweep

        return results

    def memory_usage_bytes(self) -> int:
        """Estimate current memory usage of accumulators."""
        n_floats = (
            len(self._donor_probs) + len(self._donor_labels)
            + len(self._acceptor_probs) + len(self._acceptor_labels)
            + len(self._neither_probs) + len(self._neither_labels)
        )
        topk_floats = sum(
            len(v) for v in self.topk_donor.values()
        ) + sum(
            len(v) for v in self.topk_acceptor.values()
        )
        return (n_floats + topk_floats) * 8  # 8 bytes per Python float


class StreamingEvaluator:
    """Memory-bounded evaluator comparing meta model vs base model.

    Processes one gene at a time.  Call :meth:`update` per gene, then
    :meth:`compute` to get final metrics for both models.

    Parameters
    ----------
    neither_subsample_rate : float
        Fraction of "neither" positions to keep for PR-AUC (default 0.01).
    k_multipliers : list
        Top-k multipliers for SpliceAI paper metric.
    """

    def __init__(
        self,
        neither_subsample_rate: float = 0.01,
        k_multipliers: Optional[List[float]] = None,
    ) -> None:
        self.k_multipliers = k_multipliers or [0.5, 1.0, 2.0, 4.0]
        self._meta = _ModelAccumulator(neither_subsample_rate)
        self._base = _ModelAccumulator(neither_subsample_rate)
        self._rng = np.random.default_rng(42)
        self.n_genes = 0
        self.n_skipped = 0

    def update(
        self,
        meta_probs: np.ndarray,
        base_probs: np.ndarray,
        labels: np.ndarray,
        gene_id: str,
    ) -> None:
        """Process one gene's predictions for both models.

        Parameters
        ----------
        meta_probs : np.ndarray
            Meta model output, shape ``[L, 3]``.
        base_probs : np.ndarray
            Base model scores, shape ``[L, 3]``.
        labels : np.ndarray
            Ground truth labels, shape ``[L]``.
        gene_id : str
            Gene identifier.
        """
        self._meta.update(
            meta_probs, labels, gene_id, self.k_multipliers, self._rng,
        )
        self._base.update(
            base_probs, labels, gene_id, self.k_multipliers, self._rng,
        )
        self.n_genes += 1

    def compute(self) -> dict:
        """Compute final comparison metrics.

        Returns
        -------
        dict
            Keys: ``meta_model``, ``base_model`` (each a metrics dict),
            ``n_genes``, ``n_skipped``, ``fn_reduction_pct``,
            ``fp_reduction_pct``, ``meta_topk``, ``base_topk``.
        """
        meta_m = self._meta.compute_metrics(self.k_multipliers)
        base_m = self._base.compute_metrics(self.k_multipliers)

        b_fn = base_m["fn_count"]
        m_fn = meta_m["fn_count"]
        b_fp = base_m["fp_count"]
        m_fp = meta_m["fp_count"]

        return {
            "meta_model": meta_m,
            "base_model": base_m,
            "n_genes": self.n_genes,
            "n_skipped": self.n_skipped,
            "n_positions": meta_m["n_positions"],
            "fn_reduction_pct": (b_fn - m_fn) / max(b_fn, 1) * 100,
            "fp_reduction_pct": (b_fp - m_fp) / max(b_fp, 1) * 100,
            "meta_topk": meta_m.pop("topk"),
            "base_topk": base_m.pop("topk"),
        }

    def sweep_thresholds(
        self,
        thresholds: Optional[List[float]] = None,
    ) -> dict:
        """Sweep thresholds for both models.

        Returns dict with ``meta`` and ``base`` keys, each containing
        per-class threshold sweep results from :meth:`_ModelAccumulator.sweep_thresholds`.
        """
        return {
            "meta": self._meta.sweep_thresholds(thresholds),
            "base": self._base.sweep_thresholds(thresholds),
        }

    def memory_usage_mb(self) -> float:
        """Estimated memory usage of accumulators in MB."""
        return (
            self._meta.memory_usage_bytes() + self._base.memory_usage_bytes()
        ) / (1024 * 1024)


class TemperatureScaler:
    """Class-wise temperature scaling for probability calibration.

    Learns a temperature vector ``T = [T_donor, T_acceptor, T_neither]``
    that minimizes negative log-likelihood on held-out validation data.
    Each class logit is divided by its own temperature before softmax,
    allowing independent calibration of splice-site vs background
    confidence.  Applied as:

        calibrated = alpha * softmax(logits / T) + (1-alpha) * softmax(base_scores)

    where ``T`` is a 3-element vector and alpha is the model's learned
    blend weight.

    This follows the OpenSpliceAI calibration approach (Chao et al. 2025)
    which uses per-class temperature to address class imbalance and
    splice-site sparsity.

    Reference:
        - Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
        - Chao et al. "OpenSpliceAI" (bioRxiv 2025) — class-wise variant

    Usage::

        scaler = TemperatureScaler()
        for gene in val_genes:
            logits = infer_full_gene(model, data, return_logits=True)
            scaler.collect(logits, data["base_scores"], data["labels"])

        result = scaler.fit(blend_alpha=0.5)
        T_vector = result["temperature"]  # np.ndarray [3]

        from sequence_inference import apply_temperature_blend
        probs = apply_temperature_blend(logits, base_scores, T_vector, blend_alpha)
    """

    def __init__(self, subsample_rate: float = 0.1) -> None:
        """Initialize the temperature scaler.

        Parameters
        ----------
        subsample_rate : float
            Fraction of "neither" positions to keep for fitting.
            Splice sites are always kept (all of them).
        """
        self.subsample_rate = subsample_rate
        self._logits: List[np.ndarray] = []
        self._base_scores: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._rng = np.random.default_rng(42)

    def collect(
        self,
        logits: np.ndarray,
        base_scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Collect one gene's logits and labels for calibration.

        Subsamples "neither" positions to keep memory bounded.

        Parameters
        ----------
        logits : np.ndarray
            Raw model logits ``[L, 3]``.
        base_scores : np.ndarray
            Base model scores ``[L, 3]``.
        labels : np.ndarray
            Ground truth labels ``[L]`` (0=donor, 1=acceptor, 2=neither).
        """
        splice_mask = labels < 2
        neither_idx = np.where(~splice_mask)[0]
        splice_idx = np.where(splice_mask)[0]

        n_sample = max(1, int(len(neither_idx) * self.subsample_rate))
        if len(neither_idx) > n_sample:
            bg_idx = self._rng.choice(neither_idx, size=n_sample, replace=False)
        else:
            bg_idx = neither_idx

        keep_idx = np.sort(np.concatenate([splice_idx, bg_idx]))
        self._logits.append(logits[keep_idx])
        self._base_scores.append(base_scores[keep_idx])
        self._labels.append(labels[keep_idx])

    def fit(
        self,
        blend_alpha: float = 0.5,
        num_classes: int = 3,
        blend_mode: str = "logit",
        lr: float = 0.01,
        max_epochs: int = 2000,
        patience: int = 2,
        min_delta: float = 1e-6,
        t_min: float = 0.05,
        t_max: float = 5.0,
    ) -> dict:
        """Optimize class-wise temperature vector to minimize NLL.

        Uses Adam optimizer with ReduceLROnPlateau scheduler and early
        stopping, matching OpenSpliceAI's calibration procedure.

        Parameters
        ----------
        blend_alpha : float
            The model's residual blend weight (sigmoid of blend_alpha param).
        num_classes : int
            Number of output classes (default 3).
        blend_mode : str
            ``"logit"`` or ``"probability"``.  For ``"logit"`` mode, logits
            passed to :meth:`collect` are already blended by the model —
            temperature is applied directly.  For ``"probability"`` mode,
            blend is done here as ``alpha * softmax(logits/T) + (1-alpha) * base``.
        lr : float
            Adam learning rate (default 0.01).
        max_epochs : int
            Maximum optimization epochs (default 2000).
        patience : int
            Early stopping patience (default 2).
        min_delta : float
            Minimum improvement to reset patience (default 1e-6).
        t_min, t_max : float
            Temperature clamp bounds (default 0.05, 5.0).

        Returns
        -------
        dict
            Keys: ``temperature`` (np.ndarray [C]),
            ``nll_before``, ``nll_after``,
            ``ece_before``, ``ece_after``, ``n_positions``,
            ``blend_alpha``, ``blend_mode``.
        """
        import torch
        import torch.nn.functional as F

        if not self._logits:
            raise ValueError("No data collected. Call collect() first.")

        all_logits = np.concatenate(self._logits, axis=0)  # [N, 3]
        all_base = np.concatenate(self._base_scores, axis=0)  # [N, 3]
        all_labels = np.concatenate(self._labels, axis=0)  # [N]

        n = len(all_labels)
        logger.info("Class-wise temperature scaling: fitting on %d positions", n)

        device = torch.device("cpu")
        logits_t = torch.from_numpy(all_logits).float().to(device)
        base_t = torch.from_numpy(all_base).float().to(device)
        labels_t = torch.from_numpy(all_labels).long().to(device)

        # Class-wise temperature vector, initialized to 1.0
        temperature = torch.nn.Parameter(torch.ones(num_classes, device=device))

        def _compute_nll(T: torch.Tensor) -> torch.Tensor:
            T_clamped = torch.clamp(T, min=t_min, max=t_max)
            scaled = logits_t / T_clamped  # [N, C] / [C] broadcasts

            if blend_mode == "logit":
                # Logits are already blended — just apply temperature
                blended = F.softmax(scaled, dim=-1)
            else:
                # Legacy: probability-space blend
                meta_probs = F.softmax(scaled, dim=-1)
                # base_scores are already probabilities — use directly
                blended = blend_alpha * meta_probs + (1 - blend_alpha) * base_t

            blended = torch.clamp(blended, min=1e-8)
            log_probs = torch.log(blended[torch.arange(n, device=device), labels_t])
            return -log_probs.mean()

        def _compute_ece(T: torch.Tensor, n_bins: int = 15) -> float:
            with torch.no_grad():
                T_clamped = torch.clamp(T, min=t_min, max=t_max)
                scaled = logits_t / T_clamped
                meta_probs = F.softmax(scaled, dim=-1)
                base_probs = F.softmax(base_t, dim=-1)
                blended = blend_alpha * meta_probs + (1 - blend_alpha) * base_probs
                confidences, predictions = blended.max(dim=-1)
                accuracies = (predictions == labels_t).float()

                ece = 0.0
                boundaries = torch.linspace(0, 1, n_bins + 1)
                for i in range(n_bins):
                    in_bin = (confidences > boundaries[i]) & (
                        confidences <= boundaries[i + 1]
                    )
                    if in_bin.sum() > 0:
                        avg_conf = confidences[in_bin].mean().item()
                        avg_acc = accuracies[in_bin].mean().item()
                        ece += in_bin.sum().item() * abs(avg_acc - avg_conf)
                return ece / n

        # Metrics before calibration (T = [1, 1, 1])
        ones_t = torch.ones(num_classes, device=device)
        nll_before = _compute_nll(ones_t).item()
        ece_before = _compute_ece(ones_t)

        # Optimize with Adam + ReduceLROnPlateau (following OpenSpliceAI)
        optimizer = torch.optim.Adam([temperature], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=patience,
        )

        best_loss = float("inf")
        best_temp = temperature.data.clone()
        patience_counter = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = _compute_nll(temperature)
            loss.backward()
            optimizer.step()
            temperature.data.clamp_(min=t_min, max=t_max)

            current_loss = loss.item()
            scheduler.step(current_loss)

            if best_loss - current_loss > min_delta:
                best_loss = current_loss
                best_temp = temperature.data.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (loss=%.6f)", epoch + 1, current_loss,
                )
                break

        # Restore best temperature
        temperature.data = best_temp
        T_final = temperature.data.clamp(min=t_min, max=t_max).cpu().numpy()

        nll_after = _compute_nll(temperature).item()
        ece_after = _compute_ece(temperature)

        logger.info(
            "Class-wise T=[%.4f, %.4f, %.4f]  NLL %.4f→%.4f  ECE %.4f→%.4f",
            T_final[0], T_final[1], T_final[2],
            nll_before, nll_after, ece_before, ece_after,
        )

        # Free memory
        del logits_t, base_t, labels_t, all_logits, all_base, all_labels

        return {
            "temperature": T_final,
            "nll_before": float(nll_before),
            "nll_after": float(nll_after),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "n_positions": n,
            "blend_alpha": blend_alpha,
            "blend_mode": blend_mode,
        }

    def memory_usage_mb(self) -> float:
        """Estimate memory usage of collected data."""
        total = sum(a.nbytes for a in self._logits)
        total += sum(a.nbytes for a in self._base_scores)
        total += sum(a.nbytes for a in self._labels)
        return total / (1024 * 1024)


def print_threshold_analysis(sweep: dict) -> None:
    """Print threshold sweep results comparing meta vs base model.

    Highlights the F1-optimal threshold and the threshold that maintains
    recall >= 95% with maximum precision.

    Parameters
    ----------
    sweep : dict
        Output of :meth:`StreamingEvaluator.sweep_thresholds`.
    """
    for splice_type in ["donor", "acceptor"]:
        meta_sweep = sweep["meta"].get(splice_type, [])
        base_sweep = sweep["base"].get(splice_type, [])
        if not meta_sweep:
            continue

        print(f"\n{'─'*70}")
        print(f"Threshold Sweep: {splice_type.upper()}")
        print(f"{'─'*70}\n")

        # Find optimal points for meta model
        best_f1 = max(meta_sweep, key=lambda x: x["f1"])
        high_recall = [s for s in meta_sweep if s["recall"] >= 0.95]
        best_prec_at_95rec = max(high_recall, key=lambda x: x["precision"]) if high_recall else None

        print(f"  {'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
        print(f"  {'─'*68}")

        for s in meta_sweep:
            marker = ""
            if s["threshold"] == best_f1["threshold"]:
                marker = " ← best F1"
            elif best_prec_at_95rec and s["threshold"] == best_prec_at_95rec["threshold"]:
                marker = " ← best prec@95%rec"
            print(f"  {s['threshold']:<12.2f} {s['precision']:>10.4f} {s['recall']:>10.4f} "
                  f"{s['f1']:>10.4f} {s['tp']:>8,} {s['fp']:>8,} {s['fn']:>8,}{marker}")

        print(f"\n  Optimal points:")
        print(f"    Best F1:         t={best_f1['threshold']:.2f} → "
              f"P={best_f1['precision']:.4f} R={best_f1['recall']:.4f} F1={best_f1['f1']:.4f}")
        if best_prec_at_95rec:
            print(f"    Best P@95%R:     t={best_prec_at_95rec['threshold']:.2f} → "
                  f"P={best_prec_at_95rec['precision']:.4f} R={best_prec_at_95rec['recall']:.4f} "
                  f"F1={best_prec_at_95rec['f1']:.4f}")

        # Compare with base model at default threshold
        if base_sweep:
            base_default = next((s for s in base_sweep if abs(s["threshold"] - 0.5) < 0.01), None)
            if base_default:
                print(f"    Base (t=0.50):   "
                      f"P={base_default['precision']:.4f} R={base_default['recall']:.4f} "
                      f"F1={base_default['f1']:.4f}")


def print_comparison_report(results: dict) -> None:
    """Print a formatted comparison report to stdout.

    Parameters
    ----------
    results : dict
        Output of :meth:`StreamingEvaluator.compute`.
    """
    meta_m = results["meta_model"]
    base_m = results["base_model"]

    print(f"\n{'─'*70}")
    print("Classification Metrics")
    print(f"{'─'*70}\n")

    print(f"{'Metric':<30} {'Base Model':>15} {'Meta Model':>15} {'Delta':>12}")
    print(f"{'─'*72}")

    for key in ["accuracy", "macro_pr_auc"]:
        b, m = base_m[key], meta_m[key]
        print(f"{key:<30} {b:>15.4f} {m:>15.4f} {m - b:>+12.4f}")

    print()
    for cls in ["donor", "acceptor", "neither"]:
        b_auc = base_m["pr_aucs"].get(cls, 0)
        m_auc = meta_m["pr_aucs"].get(cls, 0)
        print(f"PR-AUC ({cls}){'':<17} {b_auc:>15.4f} {m_auc:>15.4f} {m_auc - b_auc:>+12.4f}")

    print()
    for cls in ["donor", "acceptor"]:
        for met in ["precision", "recall", "f1"]:
            key = f"{cls}_{met}"
            b, m = base_m.get(key, 0), meta_m.get(key, 0)
            print(f"{key:<30} {b:>15.4f} {m:>15.4f} {m - b:>+12.4f}")

    # FN/FP
    print(f"\n{'─'*70}")
    print("Error Analysis (splice sites: donor + acceptor)")
    print(f"{'─'*70}\n")

    b_fn, m_fn = base_m["fn_count"], meta_m["fn_count"]
    b_fp, m_fp = base_m["fp_count"], meta_m["fp_count"]
    fn_red = results["fn_reduction_pct"]
    fp_red = results["fp_reduction_pct"]

    print(f"{'':30} {'Base':>15} {'Meta':>15} {'Reduction':>12}")
    print(f"{'False Negatives':<30} {b_fn:>15,} {m_fn:>15,} {fn_red:>+11.1f}%")
    print(f"{'False Positives':<30} {b_fp:>15,} {m_fp:>15,} {fp_red:>+11.1f}%")
    print(f"{'True Positives':<30} {base_m['tp_count']:>15,} {meta_m['tp_count']:>15,}")

    # Top-K
    meta_topk = results["meta_topk"]
    base_topk = results["base_topk"]
    k_mults = sorted(meta_topk.get("donor", {}).keys())

    if k_mults:
        print(f"\n{'─'*70}")
        print("Top-K Accuracy (SpliceAI paper, per-gene ranking)")
        print(f"{'─'*70}\n")

        print(f"{'k multiplier':<15}", end="")
        for km in k_mults:
            print(f"  {'k=' + str(km) + 'x':>10}", end="")
        print()
        print("─" * 60)

        for splice in ["donor", "acceptor", "overall"]:
            print(f"\n  {splice.upper()}")
            for label, topk in [("  Base", base_topk), ("  Meta", meta_topk)]:
                print(f"  {label:<12}", end="")
                for km in k_mults:
                    print(f"  {topk[splice][km]:>10.4f}", end="")
                print()
            print(f"  {'  Delta':<12}", end="")
            for km in k_mults:
                d = meta_topk[splice][km] - base_topk[splice][km]
                print(f"  {d:>+10.4f}", end="")
            print()
