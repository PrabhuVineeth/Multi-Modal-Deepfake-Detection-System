"""
Threshold Calibration Script for Deepfake Forensic Detection.
1. Performs validation inference and saves labels + scores.
2. Sweeps thresholds from 0.01 to 0.99.
3. Computes accuracy, precision, recall, F1, balanced accuracy, and TP/TN/FP/FN.
4. Selects optimal threshold from the validation split (by F1 or balanced accuracy).
5. Evaluates the test split once using the selected threshold.
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from loguru import logger

from config import get_device, model_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint
from utils.logger import setup_logger
from train import forensic_collate_fn


def compute_metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float):
    preds = (scores >= threshold).astype(int)
    
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    
    # Balanced Accuracy = (TPR + TNR) / 2
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0
    
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device, visual_only: bool = False):
    model.eval()
    all_labels = []
    all_scores = []
    
    for batch in dataloader:
        audio = batch["audio"].to(device)
        if visual_only:
            audio = torch.zeros_like(audio)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"]
        
        output = model(audio, faces, mouths)
        probs = torch.sigmoid(output.logits.squeeze(-1))
        
        all_labels.extend(labels.numpy())
        all_scores.extend(probs.cpu().numpy())
        
    return np.array(all_labels), np.array(all_scores)


def main():
    parser = argparse.ArgumentParser(description="Perform Threshold Calibration and Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["fakeavceleb", "faceforensics", "lavdf"])
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="output/calibration")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-dir", default="output/cache")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to load")
    parser.add_argument("--selection-metric", type=str, default="f1", choices=["f1", "balanced_acc"],
                        help="Metric to select the best threshold on validation split")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visual-only", action="store_true", default=False,
                        help="Calibrate in visual-only mode (zero audio)")
    parser.add_argument("--disable-audio", dest="visual_only", action="store_true",
                        help="Alias for --visual-only")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_path / "logs"))
    
    # ── Load Dataset and Split Deterministically ──
    from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset
    dataset_map = {
        "faceforensics": FaceForensicsDataset,
        "fakeavceleb": FakeAVCelebDataset,
        "lavdf": LAVDFDataset,
    }
    
    DatasetClass = dataset_map[args.dataset]
    logger.info(f"Loading {args.dataset} split=all...")
    dataset = DatasetClass(
        args.data_root, split="all",
        use_cache=args.use_cache, cache_dir=args.cache_dir,
        max_samples=args.max_samples
    )

    
    all_labels = [dataset.samples[i].label for i in range(len(dataset))]
    all_idx = list(range(len(dataset)))
    
    # Separate real (label 0) and fake (label 1) indices
    real_idx = [i for i in all_idx if all_labels[i] == 0]
    fake_idx = [i for i in all_idx if all_labels[i] == 1]
    
    # Deterministic shuffling split to match train.py exactly
    import random
    rng = random.Random(42)
    real_idx_shuffled = real_idx.copy()
    rng.shuffle(real_idx_shuffled)
    
    n_real = len(real_idx_shuffled)
    train_real_end = int(n_real * 0.8)
    val_real_end = train_real_end + int(n_real * 0.1)
    
    val_real_idx = real_idx_shuffled[train_real_end:val_real_end]
    test_real_idx = real_idx_shuffled[val_real_end:]
    
    val_fake_count = min(len(val_real_idx), len(fake_idx))
    test_fake_count = min(len(test_real_idx), len(fake_idx) - val_fake_count)
    
    fake_idx_shuffled = fake_idx.copy()
    rng.shuffle(fake_idx_shuffled)
    
    val_fake_idx = fake_idx_shuffled[:val_fake_count]
    test_fake_idx = fake_idx_shuffled[val_fake_count:val_fake_count + test_fake_count]
    
    val_idx = val_real_idx + val_fake_idx
    test_idx = test_real_idx + test_fake_idx
    
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=forensic_collate_fn, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=forensic_collate_fn, num_workers=args.num_workers
    )
    
    # ── Load Model ──
    device = get_device()
    model = DeepfakeForensicModel(config=model_config)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.to(device)
    
    # ── Part 1: Validation Split Inference & Threshold Sweep ──
    logger.info(f"Running inference on validation split ({len(val_dataset)} samples)...")
    val_labels, val_scores = collect_predictions(model, val_loader, device, visual_only=args.visual_only)
    
    sweep_results = []
    best_val_f1_metrics = None
    best_val_bal_acc_metrics = None
    
    for t in np.arange(0.01, 1.0, 0.01):
        metrics = compute_metrics_at_threshold(val_labels, val_scores, float(t))
        sweep_results.append(metrics)
        
        if best_val_f1_metrics is None or metrics["f1"] > best_val_f1_metrics["f1"]:
            best_val_f1_metrics = metrics
        if best_val_bal_acc_metrics is None or metrics["balanced_accuracy"] > best_val_bal_acc_metrics["balanced_accuracy"]:
            best_val_bal_acc_metrics = metrics
            
    selected_threshold = (
        best_val_f1_metrics["threshold"]
        if args.selection_metric == "f1"
        else best_val_bal_acc_metrics["threshold"]
    )
    
    logger.info(f"Selected optimal validation threshold: {selected_threshold:.2f} (based on {args.selection_metric})")
    
    # ── Part 2: Test Split Inference & Selected Threshold Evaluation ──
    logger.info(f"Running inference on test split ({len(test_dataset)} samples)...")
    test_labels, test_scores = collect_predictions(model, test_loader, device, visual_only=args.visual_only)

    
    test_metrics_050 = compute_metrics_at_threshold(test_labels, test_scores, 0.50)
    test_metrics_selected = compute_metrics_at_threshold(test_labels, test_scores, selected_threshold)
    test_metrics_best_f1 = compute_metrics_at_threshold(test_labels, test_scores, best_val_f1_metrics["threshold"])
    test_metrics_best_bal = compute_metrics_at_threshold(test_labels, test_scores, best_val_bal_acc_metrics["threshold"])
    
    # ── Part 3: Write outputs ──
    # Save CSV
    csv_path = output_path / f"threshold_sweep_val_{args.dataset}.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_results)
        
    # Save JSON Report
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "selection_metric": args.selection_metric,
        "optimal_threshold": float(selected_threshold),
        "validation_best_f1_threshold": float(best_val_f1_metrics["threshold"]),
        "validation_best_balanced_acc_threshold": float(best_val_bal_acc_metrics["threshold"]),
        "test_results_at_0.50": test_metrics_050,
        "test_results_at_selected_t": test_metrics_selected,
        "test_results_at_best_val_f1_t": test_metrics_best_f1,
        "test_results_at_best_val_bal_acc_t": test_metrics_best_bal,
    }
    
    json_path = output_path / f"calibration_report_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # ── Part 4: Display Summary ──
    print("\n" + "="*80)
    print(f"THRESHOLD CALIBRATION REPORT: {args.dataset.upper()}")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val split size: {len(val_dataset)} | Test split size: {len(test_dataset)}")
    print(f"Optimal Threshold selected on Val ({args.selection_metric}): {selected_threshold:.2f}")
    print("-"*80)
    print("VALIDATION SWEEP BESTS:")
    print(f"  Best F1 threshold: {best_val_f1_metrics['threshold']:.2f} (F1: {best_val_f1_metrics['f1']:.4f}, Acc: {best_val_f1_metrics['accuracy']:.4f})")
    print(f"  Best Balanced Acc threshold: {best_val_bal_acc_metrics['threshold']:.2f} (BalAcc: {best_val_bal_acc_metrics['balanced_accuracy']:.4f})")
    print("-"*80)
    print("TEST SET METRICS:")
    print(f"  At T = 0.50:")
    print(f"    Acc: {test_metrics_050['accuracy']:.4f} | Prec: {test_metrics_050['precision']:.4f} | Recall: {test_metrics_050['recall']:.4f} | F1: {test_metrics_050['f1']:.4f} | BalAcc: {test_metrics_050['balanced_accuracy']:.4f}")
    print(f"    Confusion Matrix: TP={test_metrics_050['tp']} | TN={test_metrics_050['tn']} | FP={test_metrics_050['fp']} | FN={test_metrics_050['fn']}")
    print(f"  At Selected T = {selected_threshold:.2f} (Best Val {args.selection_metric.upper()}):")
    print(f"    Acc: {test_metrics_selected['accuracy']:.4f} | Prec: {test_metrics_selected['precision']:.4f} | Recall: {test_metrics_selected['recall']:.4f} | F1: {test_metrics_selected['f1']:.4f} | BalAcc: {test_metrics_selected['balanced_accuracy']:.4f}")
    print(f"    Confusion Matrix: TP={test_metrics_selected['tp']} | TN={test_metrics_selected['tn']} | FP={test_metrics_selected['fp']} | FN={test_metrics_selected['fn']}")
    print("="*80)
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
