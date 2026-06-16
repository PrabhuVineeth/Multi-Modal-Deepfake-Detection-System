"""Evaluation metrics for the Deepfake Forensic Detection System."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve.

    Args:
        labels: Binary ground truth labels (0 or 1).
        scores: Predicted scores / probabilities.

    Returns:
        AUC-ROC score.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    The EER is the point where False Positive Rate == False Negative Rate.

    Args:
        labels: Binary ground truth labels.
        scores: Predicted scores.

    Returns:
        Tuple of (eer_value, eer_threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    # Find intersection of FPR and FNR curves
    try:
        eer = float(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        # Fallback: find the threshold where |FPR - FNR| is minimized
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2)
        eer_threshold = float(thresholds[idx])

    return eer, eer_threshold


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual correctness.

    Args:
        confidences: Predicted confidence values (max softmax probability).
        predictions: Predicted class labels.
        labels: Ground truth labels.
        n_bins: Number of confidence bins.

    Returns:
        ECE score (lower is better; 0 = perfectly calibrated).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total = len(labels)

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > lower) & (confidences <= upper)
        count = mask.sum()
        if count == 0:
            continue
        bin_accuracy = (predictions[mask] == labels[mask]).mean()
        bin_confidence = confidences[mask].mean()
        ece += (count / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(labels, predictions))


def compute_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    average: str = "binary",
) -> float:
    """Compute F1 score."""
    return float(f1_score(labels, predictions, average=average, zero_division=0))


def compute_precision_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    average: str = "binary",
) -> Tuple[float, float]:
    """Compute precision and recall."""
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return float(precision), float(recall)


def compute_boundary_f1(
    pred_tags: np.ndarray,
    true_tags: np.ndarray,
    boundary_tag: int = 2,
) -> Dict[str, float]:
    """
    Compute F1 specifically for boundary detection in TFBD.

    Evaluates how well the model identifies REAL/FAKE/BOUNDARY segments.

    Args:
        pred_tags: Predicted tag sequence.
        true_tags: Ground truth tag sequence.
        boundary_tag: Tag ID for BOUNDARY class.

    Returns:
        Dictionary with per-class and overall F1 scores.
    """
    # Per-class F1
    precision, recall, f1, support = precision_recall_fscore_support(
        true_tags, pred_tags, labels=[0, 1, 2], average=None, zero_division=0
    )

    tag_names = {0: "real", 1: "fake", 2: "boundary"}
    results = {}
    for i in range(3):
        name = tag_names[i]
        results[f"{name}_precision"] = float(precision[i])
        results[f"{name}_recall"] = float(recall[i])
        results[f"{name}_f1"] = float(f1[i])
        results[f"{name}_support"] = int(support[i])

    # Overall (macro)
    results["macro_f1"] = float(f1_score(true_tags, pred_tags, average="macro", zero_division=0))
    results["weighted_f1"] = float(f1_score(true_tags, pred_tags, average="weighted", zero_division=0))

    # Boundary-specific accuracy
    boundary_mask = true_tags == boundary_tag
    if boundary_mask.sum() > 0:
        results["boundary_accuracy"] = float(
            (pred_tags[boundary_mask] == true_tags[boundary_mask]).mean()
        )
    else:
        results["boundary_accuracy"] = 0.0

    return results


def compute_temporal_iou(
    pred_tags: np.ndarray,
    true_tags: np.ndarray,
    fake_tag: int = 1,
) -> float:
    """Compute frame-level IoU for predicted vs. true fake regions."""
    pred_mask = pred_tags == fake_tag
    true_mask = true_tags == fake_tag
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float(intersection / union)


def compute_mean_timestamp_error(
    pred_tags: np.ndarray,
    true_tags: np.ndarray,
    fps: float,
    fake_tag: int = 1,
) -> float:
    """Compute mean start/end timestamp error in seconds."""
    def bounds(tags: np.ndarray) -> Optional[Tuple[int, int]]:
        fake_idx = np.where(tags == fake_tag)[0]
        if len(fake_idx) == 0:
            return None
        return int(fake_idx[0]), int(fake_idx[-1])

    pred_bounds = bounds(pred_tags)
    true_bounds = bounds(true_tags)
    if pred_bounds is None or true_bounds is None:
        return 0.0 if pred_bounds == true_bounds else float(len(true_tags) / max(fps, 1.0))

    start_err = abs(pred_bounds[0] - true_bounds[0]) / max(fps, 1.0)
    end_err = abs(pred_bounds[1] - true_bounds[1]) / max(fps, 1.0)
    return float((start_err + end_err) / 2)


def select_performance_metrics(
    metrics: Dict[str, float],
    iou: float = 0.0,
    mte: float = 0.0,
) -> Dict[str, float]:
    """Return only the seven requested performance metrics."""
    return {
        "accuracy": metrics.get("accuracy", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1_score": metrics.get("f1", 0.0),
        "iou": iou,
        "mean_timestamp_error": mte,
        "ece": metrics.get("ece", 0.0),
    }


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a full suite of binary classification metrics.

    Args:
        labels: Ground truth binary labels.
        scores: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        Dictionary of all metric values.
    """
    predictions = (scores >= threshold).astype(int)
    confidences = np.where(predictions == 1, scores, 1 - scores)

    auc_val = compute_auc(labels, scores)
    eer_val, eer_thresh = compute_eer(labels, scores)
    ece_val = compute_ece(confidences, predictions, labels)
    acc_val = compute_accuracy(labels, predictions)
    f1_val = compute_f1(labels, predictions)
    prec, rec = compute_precision_recall(labels, predictions)

    return {
        "auc_roc": auc_val,
        "eer": eer_val,
        "eer_threshold": eer_thresh,
        "ece": ece_val,
        "accuracy": acc_val,
        "f1": f1_val,
        "precision": prec,
        "recall": rec,
        "threshold": threshold,
    }
