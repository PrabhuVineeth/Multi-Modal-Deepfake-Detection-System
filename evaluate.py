"""
Evaluation script for the Deepfake Forensic Detection System.

Computes comprehensive metrics:
  - AUC-ROC, accuracy, precision, recall, F1, EER
  - Per-dataset evaluation tables
  - Calibration analysis (ECE, reliability diagrams)
  - Boundary detection metrics (frame-level TFBD F1)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger

from config import ModelConfig, InferenceConfig, get_device, model_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint, save_json
from utils.logger import setup_logger
from utils.metrics import (
    compute_all_metrics,
    compute_boundary_f1,
    compute_ece,
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
        metrics = compute_all_metrics(labels, scores)
    else:
        metrics = {
            "auc_roc": 0.5, "accuracy": 0.0, "f1": 0.0,
            "precision": 0.0, "recall": 0.0, "eer": 1.0,
        }

    # Boundary detection metrics
    if all_pred_tags and all_true_tags:
        pred_arr = np.array(all_pred_tags)
        true_arr = np.array(all_true_tags)
        boundary_metrics = compute_boundary_f1(pred_arr, true_arr)
        metrics["boundary"] = boundary_metrics

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
            f"  {name}: AUC={metrics.get('auc_roc', 0):.4f} | "
            f"Acc={metrics.get('accuracy', 0):.4f} | "
            f"F1={metrics.get('f1', 0):.4f} | "
            f"EER={metrics.get('eer', 0):.4f}"
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
    print(f"{'Dataset':<20} {'AUC':>8} {'Acc':>8} {'F1':>8} {'EER':>8} {'Samples':>8}")
    print("-" * 80)
    for name, result in all_results.items():
        m = result["metrics"]
        print(
            f"{name:<20} "
            f"{m.get('auc_roc', 0):>8.4f} "
            f"{m.get('accuracy', 0):>8.4f} "
            f"{m.get('f1', 0):>8.4f} "
            f"{m.get('eer', 0):>8.4f} "
            f"{m.get('num_samples', 0):>8d}"
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
        FaceForensicsDataset, DFDCDataset, CelebDFDataset,
        FakeAVCelebDataset, ForgeryNetDataset,
    )

    dataset_map = {
        "faceforensics": FaceForensicsDataset,
        "dfdc": DFDCDataset,
        "celebdf": CelebDFDataset,
        "fakeavceleb": FakeAVCelebDataset,
        "forgerynet": ForgeryNetDataset,
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
