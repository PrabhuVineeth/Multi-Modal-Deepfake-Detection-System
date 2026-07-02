"""
Evaluation script for the Deepfake Forensic Detection System.

Computes performance metrics:
  - Accuracy, precision, recall, F1-score, IoU, MTE, ECE
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger

from config import ModelConfig, get_device, model_config, preprocess_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint, save_json
from utils.logger import setup_logger
from utils.metrics import (
    compute_all_metrics,
    compute_mean_timestamp_error,
    compute_temporal_iou,
    select_performance_metrics,
)
from utils.visualization import (
    plot_score_distribution,
    plot_reliability_diagram,
)


@torch.no_grad()
def evaluate_dataset(
    model: DeepfakeForensicModel,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str = "",
    visual_only: bool = False,
) -> Dict[str, any]:
    """
    Evaluate model on a single dataset.

    Returns:
        Dictionary with all metrics and raw predictions.
    """
    model.eval()
    all_labels = []
    all_scores = []
    all_pred_tags = []
    all_true_tags = []

    for batch in dataloader:
        if batch is None:
            logger.warning("All samples failed in evaluation batch; skipping batch.")
            continue
        audio = batch["audio"].to(device)
        if visual_only:
            audio = torch.zeros_like(audio)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"]

        output = model(audio, faces, mouths)

        # Classification scores
        probs = torch.sigmoid(output.logits.squeeze(-1))
        all_labels.extend(labels.numpy())
        all_scores.extend(probs.cpu().numpy())

        # Boundary predictions (if available)
        if output.boundary_tags is not None:
            pred_tags = output.boundary_tags.cpu().numpy()
            all_pred_tags.extend(pred_tags.flatten())

            if "boundary_tags" in batch:
                true_tags = batch["boundary_tags"].numpy()
                all_true_tags.extend(true_tags.flatten())

    labels = np.array(all_labels)
    scores = np.array(all_scores)

    # Core classification metrics
    if len(np.unique(labels)) > 1:
        raw_metrics = compute_all_metrics(labels, scores)
    else:
        raw_metrics = {
            "accuracy": 0.0, "f1": 0.0, "precision": 0.0,
            "recall": 0.0, "ece": 0.0,
        }

    iou = 0.0
    mte = 0.0
    if all_pred_tags and all_true_tags:
        pred_arr = np.array(all_pred_tags)
        true_arr = np.array(all_true_tags)
        iou = compute_temporal_iou(pred_arr, true_arr)
        mte = compute_mean_timestamp_error(
            pred_arr, true_arr, fps=preprocess_config.target_fps
        )

    metrics = select_performance_metrics(raw_metrics, iou=iou, mte=mte)

    metrics["dataset"] = dataset_name
    metrics["num_samples"] = len(labels)
    metrics["num_real"] = int((labels == 0).sum())
    metrics["num_fake"] = int((labels == 1).sum())

    return {
        "metrics": metrics,
        "labels": labels.tolist(),
        "scores": scores.tolist(),
    }


def compute_metrics_at_t(labels, scores, t):
    preds = (scores >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(
    checkpoint_path: str,
    datasets: Dict[str, DataLoader],
    output_dir: str,
    model_cfg: Optional[ModelConfig] = None,
    visual_only: bool = False,
    threshold: Optional[float] = None,
):
    """
    Full evaluation pipeline across multiple datasets.

    Args:
        checkpoint_path: Path to model checkpoint.
        datasets: Dict mapping dataset name → DataLoader.
        output_dir: Directory to save evaluation results.
        model_cfg: Model configuration.
        visual_only: Whether to zero-out audio waveform.
        threshold: Custom decision threshold to evaluate and report.
    """
    model_cfg = model_cfg or model_config
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logger(log_dir=str(output_path / "logs"))

    # Load model
    model = DeepfakeForensicModel(config=model_cfg)
    load_checkpoint(checkpoint_path, model, device=str(device))
    model.to(device)
    model.eval()

    all_results = {}

    for name, dataloader in datasets.items():
        logger.info(f"Evaluating on: {name}")

        result = evaluate_dataset(model, dataloader, device, name, visual_only=visual_only)
        all_results[name] = result

        labels = np.array(result["labels"])
        scores = np.array(result["scores"])

        metrics = result["metrics"]
        logger.info(
            f"  {name} (T=0.50): Acc={metrics.get('accuracy', 0):.4f} | "
            f"Precision={metrics.get('precision', 0):.4f} | "
            f"Recall={metrics.get('recall', 0):.4f} | "
            f"F1={metrics.get('f1_score', 0):.4f}"
        )

        m_050 = compute_metrics_at_t(labels, scores, 0.50)
        logger.info(f"    Confusion Matrix T=0.50: TP={m_050['tp']} | TN={m_050['tn']} | FP={m_050['fp']} | FN={m_050['fn']}")

        if threshold is not None:
            m_cust = compute_metrics_at_t(labels, scores, threshold)
            logger.info(
                f"  {name} (T={threshold:.2f}): Acc={m_cust['accuracy']:.4f} | "
                f"Precision={m_cust['precision']:.4f} | "
                f"Recall={m_cust['recall']:.4f} | "
                f"F1={m_cust['f1']:.4f}"
            )
            logger.info(f"    Confusion Matrix T={threshold:.2f}: TP={m_cust['tp']} | TN={m_cust['tn']} | FP={m_cust['fp']} | FN={m_cust['fn']}")

        # Generate plots
        if len(np.unique(labels)) > 1:
            real_scores = scores[labels == 0]
            fake_scores = scores[labels == 1]

            plot_score_distribution(
                real_scores, fake_scores,
                title=f"Score Distribution - {name}",
                save_path=str(output_path / f"scores_{name}.png"),
            )

            predictions = (scores >= (threshold if threshold is not None else 0.50)).astype(int)
            confidences = np.where(predictions == 1, scores, 1 - scores)
            accuracies = (predictions == labels).astype(float)

            plot_reliability_diagram(
                confidences, accuracies,
                title=f"Reliability Diagram - {name}",
                save_path=str(output_path / f"reliability_{name}.png"),
            )

    # Save all results
    save_json(
        {name: r["metrics"] for name, r in all_results.items()},
        str(output_path / "evaluation_results.json"),
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<16} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'IoU':>7} {'MTE':>7} {'ECE':>7}")
    print("-" * 80)
    for name, result in all_results.items():
        m = result["metrics"]
        print(
            f"{name:<20} "
            f"{m.get('accuracy', 0):>7.4f} "
            f"{m.get('precision', 0):>7.4f} "
            f"{m.get('recall', 0):>7.4f} "
            f"{m.get('f1_score', 0):>7.4f} "
            f"{m.get('iou', 0):>7.4f} "
            f"{m.get('mean_timestamp_error', 0):>7.4f} "
            f"{m.get('ece', 0):>7.4f}"
        )
    print("=" * 80)

    logger.info(f"Evaluation results saved to: {output_path}")



def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Forensic Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name(s), comma-separated")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="output/eval",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached preprocessed tensors (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Disable caching and run on-the-fly preprocessing")
    parser.add_argument("--cache-dir", default="output/cache",
                        help="Directory where preprocessed tensors are saved/loaded")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Custom decision threshold (default: None)")
    parser.add_argument("--visual-only", action="store_true", default=False,
                        help="Evaluate in visual-only mode (zero audio)")
    parser.add_argument("--disable-audio", dest="visual_only", action="store_true",
                        help="Alias for --visual-only")
    args = parser.parse_args()

    from datasets import (
        FaceForensicsDataset, FakeAVCelebDataset,
        LAVDFDataset,
    )
    from train import forensic_collate_fn

    dataset_map = {
        "faceforensics": FaceForensicsDataset,
        "fakeavceleb": FakeAVCelebDataset,
        "lavdf": LAVDFDataset,
    }

    datasets = {}
    for name in args.dataset.split(","):
        name = name.strip()
        DatasetClass = dataset_map.get(name)
        if DatasetClass is None:
            logger.warning(f"Unknown dataset: {name}")
            continue
        ds = DatasetClass(
            args.data_root, split="test", max_samples=args.max_samples,
            use_cache=args.use_cache, cache_dir=args.cache_dir
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=forensic_collate_fn
        )
        datasets[name] = loader

    evaluate(
        args.checkpoint, datasets, args.output_dir,
        visual_only=args.visual_only, threshold=args.threshold
    )



if __name__ == "__main__":
    main()
