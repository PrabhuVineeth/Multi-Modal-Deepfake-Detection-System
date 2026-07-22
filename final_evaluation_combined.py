import os
import json
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from config import get_device, model_config, path_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint
from train import forensic_collate_fn
from utils.metrics import compute_auc, compute_ece
from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset


def compute_metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float):
    labels = labels.flatten()
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
    
    # Balanced Accuracy
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0
    specificity = tnr
    
    confidences = np.where(preds == 1, scores, 1.0 - scores)
    ece = compute_ece(confidences, preds, labels)
    auc_val = compute_auc(labels, scores)
    
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_val),
        "ece": float(ece),
        "balanced_accuracy": float(balanced_acc),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def collect_predictions(model, loader, device, visual_only=False):
    all_scores = []
    all_labels = []
    
    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
            
        audio = batch["audio"].to(device)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].cpu().numpy()
        
        if visual_only:
            audio = torch.zeros_like(audio)
            
        with torch.no_grad():
            output = model(audio, faces, mouths)
            logits = output.logits.cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
            
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(loader)} batches")
            
    return np.array(all_labels), np.array(all_scores)


def plot_confusion_matrix(tn, fp, fn, tp, title, filepath):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_roc_curve(labels, scores, title, filepath):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = compute_auc(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_scores_dist(labels, scores, title, filepath):
    plt.figure(figsize=(6, 5))
    plt.hist(scores[labels == 0], bins=30, alpha=0.5, label='Real (0)', color='blue', edgecolor='k')
    plt.hist(scores[labels == 1], bins=30, alpha=0.5, label='Fake (1)', color='red', edgecolor='k')
    plt.xlabel('Predicted Sigmoid Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def main():
    output_dir = Path("output/final_evaluation_combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_config = {
        "fakeavceleb": {
            "class": FakeAVCelebDataset,
            "checkpoint": "checkpoints/best_model_combined.pth",
            "data_root": "c:/Users/Nitte/Desktop/NNM24AD071/FakeAVCeleb_v1.2",
            "visual_only": False,
            "known_threshold": 0.50,
            "max_samples": 150,
            "note": "Backbone tuned"
        },
        "faceforensics": {
            "class": FaceForensicsDataset,
            "checkpoint": "checkpoints/best_model_combined.pth",
            "data_root": "c:/Users/Nitte/Desktop/NNM24AD071/FaceForensics++_C23",
            "visual_only": True,
            "known_threshold": 0.12,
            "max_samples": 150,
            "note": "Visual-only due to silent/unreliable audio"
        },
        "lavdf": {
            "class": LAVDFDataset,
            "checkpoint": "checkpoints/best_model_combined.pth",
            "data_root": "c:/Users/Nitte/Desktop/NNM24AD071/LAV-DF",
            "visual_only": False,
            "known_threshold": 0.34,
            "max_samples": 150,
            "note": "Transfer learning evaluation"
        }
    }
    
    device = get_device()
    summary_data = []
    
    for name, cfg in datasets_config.items():
        logger.info(f"Evaluating {name}...")
        
        # Load dataset
        DatasetClass = cfg["class"]
        dataset = DatasetClass(
            cfg["data_root"], split="all",
            use_cache=True, cache_dir="output/cache",
            max_samples=cfg["max_samples"]
        )
        
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False,
            collate_fn=forensic_collate_fn, num_workers=0
        )
        
        # Load model
        model = DeepfakeForensicModel(config=model_config)
        load_checkpoint(cfg["checkpoint"], model, device=str(device))
        model.to(device)
        model.eval()
        
        # Collect predictions
        logger.info(f"Collecting predictions for {name} on all cached samples...")
        labels, scores = collect_predictions(model, loader, device, visual_only=cfg["visual_only"])
        
        # Select thresholds
        best_acc_threshold = 0.50
        best_acc = 0.0
        
        best_f1_threshold = 0.50
        best_f1 = 0.0
        
        best_bal_acc_threshold = 0.50
        best_bal_acc = 0.0
        
        for t in np.arange(0.01, 1.0, 0.01):
            metrics = compute_metrics_at_threshold(labels, scores, float(t))
            
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_acc_threshold = float(t)
                
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_f1_threshold = float(t)
                
            if metrics["balanced_accuracy"] > best_bal_acc:
                best_bal_acc = metrics["balanced_accuracy"]
                best_bal_acc_threshold = float(t)
                
        logger.info(f"Selected thresholds for {name}:")
        logger.info(f"  Best Accuracy Threshold: {best_acc_threshold:.2f} (Acc={best_acc:.4f})")
        logger.info(f"  Best F1 Threshold: {best_f1_threshold:.2f} (F1={best_f1:.4f})")
        logger.info(f"  Best Balanced Acc Threshold: {best_bal_acc_threshold:.2f} (BalAcc={best_bal_acc:.4f})")
        
        # Evaluations
        eval_thresholds = {
            "default_0.50": 0.50,
            "known_threshold": cfg["known_threshold"],
            "best_accuracy": best_acc_threshold,
            "best_f1": best_f1_threshold,
            "best_balanced_accuracy": best_bal_acc_threshold
        }
        
        test_results = {}
        for strategy, t in eval_thresholds.items():
            metrics = compute_metrics_at_threshold(labels, scores, t)
            test_results[strategy] = metrics
            
            summary_data.append({
                "Dataset": name,
                "Eval Mode": "Visual-Only" if cfg["visual_only"] else "Audio-Visual",
                "Threshold Strategy": strategy,
                "Threshold": t,
                "AUC": metrics["auc"],
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1"],
                "ECE": metrics["ece"],
                "Specificity (TNR)": metrics["specificity"],
                "TP": metrics["tp"],
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"]
            })
            
        # Save raw predictions and details
        with open(output_dir / f"{name}_predictions.json", "w") as f:
            json.dump({
                "dataset": name,
                "raw_scores": scores.tolist(),
                "labels": labels.tolist(),
                "thresholds": {
                    "best_accuracy": best_acc_threshold,
                    "best_f1": best_f1_threshold,
                    "best_balanced_accuracy": best_bal_acc_threshold
                },
                "results": test_results
            }, f, indent=4)
            
        # Plot generated graphs (using F1 threshold confusion matrix)
        best_f1_res = test_results["best_f1"]
        plot_confusion_matrix(
            best_f1_res["tn"], best_f1_res["fp"], best_f1_res["fn"], best_f1_res["tp"],
            f"{name.upper()} Confusion Matrix (F1-Opt Threshold = {best_f1_threshold:.2f})",
            str(output_dir / f"{name}_confusion_matrix.png")
        )
        
        plot_roc_curve(
            labels, scores,
            f"{name.upper()} ROC Curve (AUC = {best_f1_res['auc']:.4f})",
            str(output_dir / f"{name}_roc.png")
        )
        
        plot_scores_dist(
            labels, scores,
            f"{name.upper()} Score Distribution",
            str(output_dir / f"{name}_scores_dist.png")
        )
        
        logger.info(f"Completed evaluation and generated plots for {name}")
        
    # Write summary CSV
    csv_keys = summary_data[0].keys()
    with open(output_dir / "final_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(summary_data)
        
    # Write summary Markdown
    md_content = []
    md_content.append("# Deepfake Forensic System Final Evaluation Summary\n")
    md_content.append("This document summarizes the final evaluation metrics across all three datasets (FakeAVCeleb, FaceForensics++, and LAV-DF) using the newly backbone-tuned model.\n")
    
    md_content.append("## Dataset Notes")
    md_content.append("- **FakeAVCeleb**: Fully trained multimodal model using audio-visual fusion.")
    md_content.append("- **FaceForensics++ (FF++)**: Trained in visual-only mode due to silent/unreliable audio.")
    md_content.append("- **LAV-DF**: Transfer learning evaluation on VoxCeleb2 source identities.")
    
    md_content.append("## Complete Metrics Table\n")
    md_content.append("| Dataset | Evaluation Mode | Threshold | AUC | Accuracy | Bal. Acc | Precision | Recall | F1-score | ECE | Specificity | TP / TN / FP / FN |")
    md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for row in summary_data:
        matrix_str = f"{row['TP']} / {row['TN']} / {row['FP']} / {row['FN']}"
        md_content.append(
            f"| **{row['Dataset'].upper()}** | {row['Eval Mode']} | {row['Threshold']:.2f} | {row['AUC']:.4f} | {row['Accuracy']:.4f} | {row['Balanced Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | **{row['F1-score']:.4f}** | {row['ECE']:.4f} | {row['Specificity (TNR)']:.4f} | {matrix_str} |"
        )
        
    with open(output_dir / "final_summary.md", "w") as f:
        f.write("\n".join(md_content))
        
    logger.info("Saved final summary report: output/final_evaluation_combined/final_summary.md")
    print("=" * 60)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
