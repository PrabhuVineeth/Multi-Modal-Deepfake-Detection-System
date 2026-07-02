"""
mine_hard_examples.py — Hard-negative mining for deepfake detection.

Runs inference on a dataset split, identifies misclassified samples
(false positives and false negatives) at a given threshold, and saves them
to JSON + CSV for inspection or use in subsequent hard-example fine-tuning.

Usage:
    python mine_hard_examples.py \
        --checkpoint checkpoints/best_model_faceforensics_visual.pth \
        --dataset faceforensics \
        --data-root c:/Users/Nitte/Desktop/NNM24AD071/FaceForensics++_C23 \
        --split val \
        --threshold 0.12 \
        --visual-only \
        --output-dir output/hard_examples \
        --use-cache --cache-dir output/cache
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader, Subset

from config import get_device, model_config
from models.full_model import DeepfakeForensicModel
from utils.io_utils import load_checkpoint
from utils.logger import setup_logger
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────

def _get_video_path(base_dataset, global_idx: int) -> str:
    try:
        return base_dataset.samples[global_idx].video_path
    except Exception:
        return f"<unknown idx={global_idx}>"


@torch.no_grad()
def run_inference(model, dataloader, device, visual_only: bool = False):
    """Return (video_paths, labels, scores)."""
    model.eval()
    all_vpaths, all_labels, all_scores = [], [], []

    for batch in dataloader:
        if batch is None:
            continue
        audio = batch["audio"].to(device)
        if visual_only:
            audio = torch.zeros_like(audio)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].cpu().numpy().tolist()

        output = model(audio, faces, mouths)
        probs = torch.sigmoid(output.logits.squeeze(-1)).float().cpu().numpy().tolist()

        meta_list = batch.get("metadata", [])
        for i, (lbl, score) in enumerate(zip(labels, probs)):
            vp = ""
            if meta_list and i < len(meta_list):
                m = meta_list[i]
                vp = m.get("video_path", "") if isinstance(m, dict) else getattr(m, "video_path", "")
            all_vpaths.append(vp)
            all_labels.append(int(lbl))
            all_scores.append(float(score))

    return all_vpaths, all_labels, all_scores


def collect_hard_examples(video_paths, labels, scores, threshold):
    results = []
    for vp, lbl, score in zip(video_paths, labels, scores):
        pred = 1 if score >= threshold else 0
        if pred == lbl:
            continue
        results.append({
            "video_path": vp,
            "label": lbl,
            "score": round(score, 6),
            "prediction": pred,
            "threshold": threshold,
            "error_type": "FP" if pred == 1 else "FN",
        })
    return results


def save_results(hard: List[dict], output_dir: Path, dataset_name: str, split: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hard_examples_{dataset_name}_{split}"

    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hard, f, indent=2)
    logger.info(f"JSON saved: {json_path}")

    csv_path = output_dir / f"{stem}.csv"
    if hard:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=hard[0].keys())
            writer.writeheader()
            writer.writerows(hard)
    logger.info(f"CSV saved: {csv_path}")
    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────

def _make_splits(base_dataset, max_samples=None):
    """Replicate the deterministic 80/10/10 split from train.py (seed=42)."""
    import random
    all_labels = [base_dataset.samples[i].label for i in range(len(base_dataset))]
    real_idx = [i for i, l in enumerate(all_labels) if l == 0]
    fake_idx = [i for i, l in enumerate(all_labels) if l == 1]
    rng = random.Random(42)
    real_shuffled = real_idx.copy(); rng.shuffle(real_shuffled)
    fake_shuffled = fake_idx.copy(); rng.shuffle(fake_shuffled)

    n_real = len(real_shuffled)
    train_real_end = int(n_real * 0.8)
    val_real_end   = train_real_end + int(n_real * 0.1)
    train_real = real_shuffled[:train_real_end]
    val_real   = real_shuffled[train_real_end:val_real_end]
    test_real  = real_shuffled[val_real_end:]

    val_fake_count  = min(len(val_real),  len(fake_shuffled))
    test_fake_count = min(len(test_real), len(fake_shuffled) - val_fake_count)
    val_fake   = fake_shuffled[:val_fake_count]
    test_fake  = fake_shuffled[val_fake_count:val_fake_count + test_fake_count]
    train_fake = fake_shuffled[val_fake_count + test_fake_count:]

    train_idx = train_real + train_fake; rng.shuffle(train_idx)
    val_idx   = val_real   + val_fake;   rng.shuffle(val_idx)
    test_idx  = test_real  + test_fake;  rng.shuffle(test_idx)

    return {"train": train_idx, "val": val_idx, "test": test_idx}, all_labels


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard examples (FP/FN) from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["faceforensics", "fakeavceleb", "lavdf"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test", "all"],
                        help="Split to mine. Use 'val' for legitimate model selection.")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--output-dir", default="output/hard_examples")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--cache-dir", default="output/cache")
    parser.add_argument("--visual-only", action="store_true", default=False)
    parser.add_argument("--disable-audio", dest="visual_only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logger(log_dir=str(output_dir / "logs"))

    # ── Dataset ──────────────────────────────────────────────────────────────
    from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset
    from train import forensic_collate_fn

    DatasetClass = {"faceforensics": FaceForensicsDataset,
                    "fakeavceleb": FakeAVCelebDataset,
                    "lavdf": LAVDFDataset}[args.dataset]

    base_dataset = DatasetClass(
        args.data_root, split="all",
        max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir,
    )

    if args.split == "all":
        dataset = base_dataset
        indices = list(range(len(base_dataset)))
        split_labels = [base_dataset.samples[i].label for i in indices]
    else:
        splits, all_labels = _make_splits(base_dataset)
        indices = splits[args.split]
        dataset = Subset(base_dataset, indices)
        split_labels = [all_labels[i] for i in indices]

    n_real = sum(1 for l in split_labels if l == 0)
    n_fake = sum(1 for l in split_labels if l == 1)
    logger.info(f"Split '{args.split}': {len(indices)} samples (real={n_real}, fake={n_fake})")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=forensic_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = get_device()
    model = DeepfakeForensicModel(config=model_config)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.to(device)
    logger.info(f"Checkpoint loaded: {args.checkpoint}")
    if args.visual_only:
        logger.info("Visual-only mode: audio zeroed.")

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info(f"Running inference ({len(dataset)} samples)...")
    vpaths, labels, scores = run_inference(model, loader, device, visual_only=args.visual_only)

    # Resolve missing video paths from dataset samples
    resolved = []
    for i, vp in enumerate(vpaths):
        if vp:
            resolved.append(vp)
        else:
            global_idx = indices[i] if isinstance(dataset, Subset) else i
            resolved.append(_get_video_path(base_dataset, global_idx))

    # ── Collect & report ──────────────────────────────────────────────────────
    hard = collect_hard_examples(resolved, labels, scores, args.threshold)
    n_fp = sum(1 for h in hard if h["error_type"] == "FP")
    n_fn = sum(1 for h in hard if h["error_type"] == "FN")

    logger.info("=" * 70)
    logger.info(f"HARD EXAMPLE MINING: {args.dataset.upper()} / split={args.split}")
    logger.info(f"  Threshold : {args.threshold}")
    logger.info(f"  Total     : {len(labels)}")
    logger.info(f"  Correct   : {len(labels) - len(hard)}")
    logger.info(f"  Errors    : {len(hard)}  (FP={n_fp}, FN={n_fn})")
    logger.info("=" * 70)

    save_results(hard, output_dir, args.dataset, args.split)

    # ── Save index file (used by --hard-examples-file in train.py) ────────────
    index_path = output_dir / f"hard_indices_{args.dataset}_{args.split}.json"
    hard_vpaths = [h["video_path"] for h in hard]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "dataset": args.dataset,
            "split": args.split,
            "threshold": args.threshold,
            "hard_video_paths": hard_vpaths,
            "n_fp": n_fp,
            "n_fn": n_fn,
        }, f, indent=2)
    logger.info(f"Index file saved: {index_path}")
    logger.info(
        f"\nTo fine-tune on hard examples add to train.py command:\n"
        f"  --hard-examples-file {index_path}"
    )


if __name__ == "__main__":
    main()
