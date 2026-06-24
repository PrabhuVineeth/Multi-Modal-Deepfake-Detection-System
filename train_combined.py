"""
Combined training script — trains on FakeAVCeleb + FaceForensics++ together.

Usage:
    C:\\Users\\Nitte\\miniconda3\\python.exe train_combined.py
    C:\\Users\\Nitte\\miniconda3\\python.exe train_combined.py --resume checkpoints/best_model.pth
    C:\\Users\\Nitte\\miniconda3\\python.exe train_combined.py --epochs 30 --batch-size 4
"""

import argparse
from pathlib import Path
from torch.utils.data import ConcatDataset, random_split

from config import TrainingConfig, path_config
from datasets import FakeAVCelebDataset, FaceForensicsDataset
from train import train


def main():
    parser = argparse.ArgumentParser(description="Train on FakeAVCeleb + FaceForensics++")
    parser.add_argument("--fakeavceleb-root", default=str(path_config.fakeavceleb_root))
    parser.add_argument("--faceforensics-root", default=str(path_config.faceforensics_root))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (useful for quick testing)")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached preprocessed tensors (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Disable caching and run on-the-fly preprocessing")
    parser.add_argument("--cache-dir", default="output/cache",
                        help="Directory where preprocessed tensors are saved/loaded")
    args = parser.parse_args()

    t_cfg = TrainingConfig()
    if args.epochs:
        t_cfg.max_epochs = args.epochs
    t_cfg.batch_size = args.batch_size

    print("=" * 60)
    print("  Multi-Dataset Training: FakeAVCeleb + FaceForensics++")
    print("  GPU: NVIDIA RTX 4070")
    print("=" * 60)

    # ── Load datasets ──
    print("\n[1/3] Loading FakeAVCeleb...")
    fav_train = FakeAVCelebDataset(
        args.fakeavceleb_root, split="train", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )
    fav_val = FakeAVCelebDataset(
        args.fakeavceleb_root, split="val", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )

    print("[2/3] Loading FaceForensics++...")
    ff_train = FaceForensicsDataset(
        args.faceforensics_root, split="train", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )
    ff_val = FaceForensicsDataset(
        args.faceforensics_root, split="val", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )

    # ── Combine ──
    print("[3/3] Combining datasets...")
    train_dataset = ConcatDataset([fav_train, ff_train])
    val_dataset = ConcatDataset([fav_val, ff_val])

    print(f"\n  Train samples : {len(train_dataset):,}  "
          f"(FakeAVCeleb: {len(fav_train):,} + FF++: {len(ff_train):,})")
    print(f"  Val samples   : {len(val_dataset):,}  "
          f"(FakeAVCeleb: {len(fav_val):,} + FF++: {len(ff_val):,})")
    print(f"  Batch size    : {t_cfg.batch_size}")
    print(f"  Max epochs    : {t_cfg.max_epochs}")
    print(f"  Cache enabled : {args.use_cache}")
    print(f"  Cache dir     : {args.cache_dir}")
    print(f"  Checkpoints   : checkpoints/\n")

    # ── Train ──
    train(
        train_dataset, val_dataset, train_cfg=t_cfg, resume_from=args.resume,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
