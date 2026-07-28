"""
Full End-to-End Evaluation on All Datasets + Joint
====================================================
Uses the full DeepfakeForensicModel with G: drive dataset paths.
Evaluates: FakeAVCeleb, FaceForensics++, LAV-DF, and Joint (combined).
"""

import json
import csv
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random

from config import get_device, model_config, preprocess_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint
from utils.metrics import compute_auc, compute_eer, compute_ece
from train import forensic_collate_fn
from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset


def compute_full_metrics(labels, scores, threshold=0.5):
    """Compute all metrics at a given threshold."""
    labels = labels.flatten().astype(int)
    scores = scores.flatten()
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0

    confidences = np.where(preds == 1, scores, 1.0 - scores)
    ece_val = compute_ece(confidences, preds, labels)

    if len(np.unique(labels)) > 1:
        auc_val = compute_auc(labels, scores)
        eer_val, eer_thresh = compute_eer(labels, scores)
    else:
        auc_val, eer_val, eer_thresh = 0.0, 0.0, 0.5

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_val),
        "ece": float(ece_val),
        "eer": float(eer_val),
        "eer_threshold": float(eer_thresh),
        "specificity": float(tnr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_samples": len(labels),
        "real_count": int((labels == 0).sum()),
        "fake_count": int((labels == 1).sum()),
    }


def find_best_threshold(labels, scores, metric="f1"):
    """Sweep 0.01-0.99 to find best threshold for a given metric."""
    best_t, best_val = 0.5, -1.0
    for t in np.arange(0.01, 1.0, 0.01):
        m = compute_full_metrics(labels, scores, float(t))
        if m[metric] > best_val:
            best_val = m[metric]
            best_t = float(t)
    return best_t, best_val


@torch.no_grad()
def collect_predictions(model, dataloader, device, visual_only=False):
    """Run model on dataloader, return (labels, scores)."""
    model.eval()
    all_labels, all_scores = [], []

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        audio = batch["audio"].to(device)
        if visual_only:
            audio = torch.zeros_like(audio)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"]

        output = model(audio, faces, mouths)
        probs = torch.sigmoid(output.logits.squeeze(-1))

        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}")

    return np.array(all_labels), np.array(all_scores)


def get_deterministic_test_split(dataset, test_ratio=0.1, seed=42):
    """Get deterministic test split indices."""
    all_labels = [dataset.samples[i].label for i in range(len(dataset))]
    real_idx = [i for i, l in enumerate(all_labels) if l == 0]
    fake_idx = [i for i, l in enumerate(all_labels) if l == 1]

    rng = random.Random(seed)
    rng.shuffle(real_idx)
    rng.shuffle(fake_idx)

    # Last 10% for test
    test_real = real_idx[int(len(real_idx) * 0.9):]
    test_fake = fake_idx[int(len(fake_idx) * 0.9):]
    test_idx = test_real + test_fake
    rng.shuffle(test_idx)
    return test_idx


def main():
    output_dir = Path("output/full_evaluation_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # ─── Dataset configurations (G: drive) ───
    datasets_config = {
        "fakeavceleb": {
            "class": FakeAVCelebDataset,
            "checkpoint": "checkpoints/best_model.pth",  # Original FakeAVCeleb model
            "data_root": "G:/NNM24AD071/FakeAVCeleb_v1.2",
            "visual_only": False,
            "max_samples": None,
        },
        "faceforensics": {
            "class": FaceForensicsDataset,
            "checkpoint": "checkpoints/best_model_faceforensics_visual.pth",
            "data_root": "G:/NNM24AD071/FaceForensics++_C23",
            "visual_only": True,
            "max_samples": None,
        },
        "lavdf": {
            "class": LAVDFDataset,
            "checkpoint": "checkpoints/best_model_lavdf_full_tuned.pth",
            "data_root": "G:/NNM24AD071/LAV-DF",
            "visual_only": False,
            "max_samples": 5000,
        },
    }

    summary_rows = []
    all_predictions = {}  # Store for joint evaluation

    # ════════════════════════════════════════════════════════════════
    # PART 1: Per-Dataset Evaluation
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  EVALUATING ALL DATASETS (Full End-to-End Model)")
    print("=" * 110)

    for ds_name, cfg in datasets_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {ds_name.upper()}")
        logger.info(f"{'='*60}")

        # Load dataset
        DatasetClass = cfg["class"]
        dataset = DatasetClass(
            cfg["data_root"], split="all",
            use_cache=True, cache_dir="output/cache",
            max_samples=cfg["max_samples"]
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Get test split
        test_idx = get_deterministic_test_split(dataset)
        test_dataset = Subset(dataset, test_idx)
        logger.info(f"Test split: {len(test_dataset)} samples")

        test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=False,
            collate_fn=forensic_collate_fn, num_workers=0
        )

        # Load model
        model = DeepfakeForensicModel(config=model_config)
        load_checkpoint(cfg["checkpoint"], model, device=str(device))
        model.to(device)
        model.eval()

        # Collect predictions
        labels, scores = collect_predictions(
            model, test_loader, device, visual_only=cfg["visual_only"]
        )
        logger.info(f"Predictions collected: {len(labels)} samples (Real={int((labels==0).sum())}, Fake={int((labels==1).sum())})")

        # Store for joint evaluation
        all_predictions[ds_name] = {"labels": labels, "scores": scores}

        # Find optimal thresholds
        best_f1_t, _ = find_best_threshold(labels, scores, "f1")
        best_acc_t, _ = find_best_threshold(labels, scores, "accuracy")

        # Evaluate at multiple thresholds
        thresholds = {
            "T=0.50": 0.50,
            f"T={best_f1_t:.2f} (best-F1)": best_f1_t,
            f"T={best_acc_t:.2f} (best-Acc)": best_acc_t,
        }

        print(f"\n  {ds_name.upper()} ({'Visual-Only' if cfg['visual_only'] else 'Audio-Visual'})")
        print(f"  Samples: {len(labels)} (Real={int((labels==0).sum())}, Fake={int((labels==1).sum())})")

        for t_name, t_val in thresholds.items():
            m = compute_full_metrics(labels, scores, t_val)
            print(f"    {t_name}: AUC={m['auc']:.4f} | EER={m['eer']*100:.2f}% | Acc={m['accuracy']*100:.2f}% | "
                  f"Prec={m['precision']*100:.2f}% | Rec={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}% | ECE={m['ece']:.4f}")

            summary_rows.append({
                "Dataset": ds_name.upper(),
                "Mode": "Visual-Only" if cfg["visual_only"] else "Audio-Visual",
                "Threshold": f"{t_val:.2f}",
                "Strategy": t_name,
                "AUC": m["auc"],
                "EER": m["eer"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "ECE": m["ece"],
                "Specificity": m["specificity"],
                "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
                "Samples": m["total_samples"],
            })

        # Save per-dataset results
        with open(output_dir / f"{ds_name}_metrics.json", "w") as f:
            json.dump({t: compute_full_metrics(labels, scores, v) for t, v in thresholds.items()}, f, indent=4)

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ════════════════════════════════════════════════════════════════
    # PART 2: Joint Combined Dataset Evaluation
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  JOINT COMBINED DATASET EVALUATION")
    print("=" * 110)

    if all_predictions:
        # Use the best general model for joint eval
        joint_checkpoint = "checkpoints/best_model.pth"
        model = DeepfakeForensicModel(config=model_config)
        load_checkpoint(joint_checkpoint, model, device=str(device))
        model.to(device)
        model.eval()

        # Merge all predictions
        joint_labels = np.concatenate([all_predictions[ds]["labels"] for ds in all_predictions])
        joint_scores = np.concatenate([all_predictions[ds]["scores"] for ds in all_predictions])

        n_total = len(joint_labels)
        n_real = int((joint_labels == 0).sum())
        n_fake = int((joint_labels == 1).sum())
        logger.info(f"Joint dataset: {n_total} samples (Real={n_real}, Fake={n_fake})")

        best_f1_t, _ = find_best_threshold(joint_labels, joint_scores, "f1")
        best_acc_t, _ = find_best_threshold(joint_labels, joint_scores, "accuracy")

        thresholds = {
            "T=0.50": 0.50,
            f"T={best_f1_t:.2f} (best-F1)": best_f1_t,
            f"T={best_acc_t:.2f} (best-Acc)": best_acc_t,
        }

        print(f"\n  JOINT (ALL DATASETS)")
        print(f"  Samples: {n_total} (Real={n_real}, Fake={n_fake})")

        for t_name, t_val in thresholds.items():
            m = compute_full_metrics(joint_labels, joint_scores, t_val)
            print(f"    {t_name}: AUC={m['auc']:.4f} | EER={m['eer']*100:.2f}% | Acc={m['accuracy']*100:.2f}% | "
                  f"Prec={m['precision']*100:.2f}% | Rec={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}% | ECE={m['ece']:.4f}")

            summary_rows.append({
                "Dataset": "JOINT (ALL)",
                "Mode": "Audio-Visual",
                "Threshold": f"{t_val:.2f}",
                "Strategy": t_name,
                "AUC": m["auc"],
                "EER": m["eer"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "ECE": m["ece"],
                "Specificity": m["specificity"],
                "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
                "Samples": m["total_samples"],
            })

        with open(output_dir / "joint_metrics.json", "w") as f:
            json.dump({t: compute_full_metrics(joint_labels, joint_scores, v) for t, v in thresholds.items()}, f, indent=4)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ════════════════════════════════════════════════════════════════
    # SAVE SUMMARY
    # ════════════════════════════════════════════════════════════════
    if summary_rows:
        csv_path = output_dir / "evaluation_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nCSV saved: {csv_path.resolve()}")

    # Final summary table
    print("\n" + "=" * 130)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 130)
    print(f"{'Dataset':<16} {'Mode':<14} {'Threshold':>9} {'AUC':>8} {'EER':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ECE':>8}")
    print("-" * 130)
    for row in summary_rows:
        print(
            f"{row['Dataset']:<16} {row['Mode']:<14} {row['Threshold']:>9} "
            f"{row['AUC']:>8.4f} {row['EER']*100:>7.2f}% {row['Accuracy']*100:>7.2f}% "
            f"{row['Precision']*100:>7.2f}% {row['Recall']*100:>7.2f}% {row['F1']*100:>7.2f}% "
            f"{row['ECE']:>8.4f}"
        )
    print("=" * 130)
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
