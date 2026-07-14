"""
Calibration and Threshold Sweep Utility for MDDS.
Sweeps threshold from 0.01 to 0.99 to find the optimal F1 score threshold.
Computes full test metrics, confusion matrix, and balanced subset metrics.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger

from config import get_device, model_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint, save_json
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
    
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def get_balanced_subset_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, seed: int = 42):
    real_indices = np.where(labels == 0)[0]
    fake_indices = np.where(labels == 1)[0]
    
    num_balanced = min(len(real_indices), len(fake_indices))
    
    np.random.seed(seed)
    balanced_real = np.random.choice(real_indices, num_balanced, replace=False)
    balanced_fake = np.random.choice(fake_indices, num_balanced, replace=False)
    
    balanced_indices = np.concatenate([balanced_real, balanced_fake])
    balanced_labels = labels[balanced_indices]
    balanced_scores = scores[balanced_indices]
    
    return compute_metrics_at_threshold(balanced_labels, balanced_scores, threshold)


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []
    
    for batch in dataloader:
        audio = batch["audio"].to(device)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"]
        
        output = model(audio, faces, mouths)
        probs = torch.sigmoid(output.logits.squeeze(-1))
        
        all_labels.extend(labels.numpy())
        all_scores.extend(probs.cpu().numpy())
        
    return np.array(all_labels), np.array(all_scores)


def main():
    parser = argparse.ArgumentParser(description="Calibrate and Sweep Thresholds")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["fakeavceleb", "faceforensics", "lavdf"])
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="output/calibration")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-dir", default="output/cache_full")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_path / "logs"))
    
    from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset
    dataset_map = {
        "faceforensics": FaceForensicsDataset,
        "fakeavceleb": FakeAVCelebDataset,
        "lavdf": LAVDFDataset,
    }
    
    DatasetClass = dataset_map[args.dataset]
    ds = DatasetClass(
        args.data_root, split="test",
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )
    
    if args.max_samples is not None and args.max_samples > 0:
        from torch.utils.data import Subset
        import random
        indices = list(range(len(ds)))
        random.seed(42)
        random.sample(indices, min(len(indices), args.max_samples)) # dry run to match seed
        indices = indices[:args.max_samples]
        ds = Subset(ds, indices)
        logger.info(f"Subsetting calibration dataset to max_samples: {len(ds)}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=forensic_collate_fn
    )
    
    device = get_device()
    model = DeepfakeForensicModel(config=model_config)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.to(device)
    
    logger.info(f"Collecting predictions on {args.dataset} split=test ({len(ds)} samples)...")
    labels, scores = collect_predictions(model, loader, device)
    
    # ── Step 1: Perform threshold sweep to find best F1 threshold ──
    best_f1 = -1.0
    best_t = 0.5
    sweep_results = []
    
    for t in np.arange(0.01, 1.0, 0.01):
        metrics = compute_metrics_at_threshold(labels, scores, float(t))
        sweep_results.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(t)
            
    # ── Step 2: Extract reports ──
    metrics_050 = compute_metrics_at_threshold(labels, scores, 0.50)
    metrics_best = compute_metrics_at_threshold(labels, scores, best_t)
    
    balanced_050 = get_balanced_subset_metrics(labels, scores, 0.50)
    balanced_best = get_balanced_subset_metrics(labels, scores, best_t)
    
    report = {
        "dataset": args.dataset,
        "num_samples": len(labels),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "best_threshold": best_t,
        "metrics_at_0.50": metrics_050,
        "metrics_at_best_t": metrics_best,
        "balanced_metrics_at_0.50": balanced_050,
        "balanced_metrics_at_best_t": balanced_best
    }
    
    save_json(report, str(output_path / f"calibration_report_{args.dataset}.json"))
    
    # ── Step 3: Print summary ──
    print("\n" + "="*80)
    print(f"CALIBRATION & THRESHOLD SWEEP REPORT: {args.dataset.upper()} TEST SET")
    print("="*80)
    print(f"Total Samples: {len(labels)} | Real: {report['num_real']} | Fake: {report['num_fake']}")
    print(f"Optimal F1 Threshold: {best_t:.2f}")
    print("-"*80)
    print("FULL TEST SET METRICS:")
    print(f"  At T = 0.50: Acc={metrics_050['accuracy']:.4f} | Prec={metrics_050['precision']:.4f} | Recall={metrics_050['recall']:.4f} | F1={metrics_050['f1']:.4f}")
    print(f"    Confusion Matrix: TP={metrics_050['tp']} | TN={metrics_050['tn']} | FP={metrics_050['fp']} | FN={metrics_050['fn']}")
    print(f"  At T = {best_t:.2f}: Acc={metrics_best['accuracy']:.4f} | Prec={metrics_best['precision']:.4f} | Recall={metrics_best['recall']:.4f} | F1={metrics_best['f1']:.4f}")
    print(f"    Confusion Matrix: TP={metrics_best['tp']} | TN={metrics_best['tn']} | FP={metrics_best['fp']} | FN={metrics_best['fn']}")
    print("-"*80)
    print(f"BALANCED SUBSET METRICS (N = {report['num_real'] * 2} samples):")
    print(f"  At T = 0.50: Acc={balanced_050['accuracy']:.4f} | Prec={balanced_050['precision']:.4f} | Recall={balanced_050['recall']:.4f} | F1={balanced_050['f1']:.4f}")
    print(f"    Confusion Matrix: TP={balanced_050['tp']} | TN={balanced_050['tn']} | FP={balanced_050['fp']} | FN={balanced_050['fn']}")
    print(f"  At T = {best_t:.2f}: Acc={balanced_best['accuracy']:.4f} | Prec={balanced_best['precision']:.4f} | Recall={balanced_best['recall']:.4f} | F1={balanced_best['f1']:.4f}")
    print(f"    Confusion Matrix: TP={balanced_best['tp']} | TN={balanced_best['tn']} | FP={balanced_best['fp']} | FN={balanced_best['fn']}")
    print("="*80)


if __name__ == "__main__":
    main()
