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
        audio = batch["audio"].to(device)
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


def evaluate(
    checkpoint_path: str,
    datasets: Dict[str, DataLoader],
    output_dir: str,
    model_cfg: Optional[ModelConfig] = None,
):
    """
    Full evaluation pipeline across multiple datasets.

    Args:
        checkpoint_path: Path to model checkpoint.
        datasets: Dict mapping dataset name → DataLoader.
        output_dir: Directory to save evaluation results.
        model_cfg: Model configuration.
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

        result = evaluate_dataset(model, dataloader, device, name)
        all_results[name] = result

        metrics = result["metrics"]
        logger.info(
            f"  {name}: Acc={metrics.get('accuracy', 0):.4f} | "
            f"Precision={metrics.get('precision', 0):.4f} | "
            f"Recall={metrics.get('recall', 0):.4f} | "
            f"F1={metrics.get('f1_score', 0):.4f}"
        )

        # Generate plots
        labels = np.array(result["labels"])
        scores = np.array(result["scores"])

        if len(np.unique(labels)) > 1:
            real_scores = scores[labels == 0]
            fake_scores = scores[labels == 1]

            plot_score_distribution(
                real_scores, fake_scores,
                title=f"Score Distribution - {name}",
                save_path=str(output_path / f"scores_{name}.png"),
            )

            predictions = (scores >= 0.5).astype(int)
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
    args = parser.parse_args()

    from datasets import (
        FaceForensicsDataset, FakeAVCelebDataset,
        LAVDFDataset,
    )

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
            args.data_root, split="test", max_samples=args.max_samples
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        datasets[name] = loader

    evaluate(args.checkpoint, datasets, args.output_dir)


if __name__ == "__main__":
    main()
