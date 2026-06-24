"""
Training script for the Deepfake Forensic Detection System.

Multi-task training with:
  L = λ₁·L_cls + λ₂·L_lip + λ₃·L_id + λ₄·L_temp + λ₅·L_sync + λ₆·L_boundary

Supports:
  - AdamW optimizer with cosine/step schedulers
  - Gradient accumulation for large effective batch sizes
  - Mixed precision training (AMP)
  - Checkpoint saving with best-model tracking (by AUC)
  - Early stopping
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

from config import (
    ModelConfig,
    PreprocessConfig,
    TrainingConfig,
    PathConfig,
    get_device,
    model_config,
    training_config,
    path_config,
)
from models.full_model import DeepfakeForensicModel, ForensicOutput
from utils.io_utils import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.metrics import compute_all_metrics


def forensic_collate_fn(batch):
    """Custom collate: stack tensors normally, keep metadata as list (avoids None collation errors)."""
    from torch.utils.data._utils.collate import default_collate
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k == "metadata":
            result[k] = [d[k] for d in batch]  # list of dicts, not stacked
        else:
            result[k] = default_collate([d[k] for d in batch])
    return result


def create_optimizer(
    model: nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with separate LR for backbone vs. heads."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name and "projection" not in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.learning_rate * 0.1},
            {"params": head_params, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
):
    """Create learning rate scheduler."""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=1e-7
        )
    elif config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.step_gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def compute_multitask_loss(
    output: ForensicOutput,
    labels: torch.Tensor,
    boundary_tags: Optional[torch.Tensor],
    config: TrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the multi-task loss.

    Args:
        output: ForensicOutput from the model.
        labels: Ground truth labels [B] (0=REAL, 1=FAKE).
        boundary_tags: Optional per-frame boundary tags [B, T].
        config: Training configuration with loss weights.

    Returns:
        Tuple of (total_loss, loss_dict).
    """
    losses = {}

    # 1. Classification loss (BCE with logits)
    cls_loss = F.binary_cross_entropy_with_logits(
        output.logits.squeeze(-1), labels.float()
    )
    losses["cls"] = cls_loss.item()

    # 2. Lip sync loss
    lip_target = labels.float().unsqueeze(-1)  # FAKE=1 → high score
    lip_loss = F.mse_loss(output.lip_sync_score, lip_target)
    losses["lip_sync"] = lip_loss.item()

    # 3. Identity loss
    id_loss = F.mse_loss(output.identity_score, lip_target)
    losses["identity"] = id_loss.item()

    # 4. Temporal loss
    temp_loss = F.mse_loss(output.temporal_score, lip_target)
    losses["temporal"] = temp_loss.item()

    # 5. AV sync loss
    sync_loss = F.mse_loss(output.av_sync_score, lip_target)
    losses["av_sync"] = sync_loss.item()

    # 6. Boundary loss (TFBD CRF)
    boundary_loss = torch.tensor(0.0, device=labels.device)
    if output.boundary_loss is not None:
        boundary_loss = output.boundary_loss
    losses["boundary"] = boundary_loss.item()

    # Weighted sum
    total = (
        config.lambda_cls * cls_loss
        + config.lambda_lip * lip_loss
        + config.lambda_id * id_loss
        + config.lambda_temp * temp_loss
        + config.lambda_sync * sync_loss
        + config.lambda_boundary * boundary_loss
    )
    losses["total"] = total.item()

    return total, losses


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].to(device)
        boundary_tags = batch.get("boundary_tags")
        if boundary_tags is not None:
            boundary_tags = boundary_tags.to(device)

        # Forward pass
        if config.use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                output = model(
                    audio, faces, mouths,
                    boundary_tags=boundary_tags,
                )
                loss, loss_dict = compute_multitask_loss(
                    output, labels, boundary_tags, config
                )
                loss = loss / config.gradient_accumulation_steps
        else:
            output = model(
                audio, faces, mouths,
                boundary_tags=boundary_tags,
            )
            loss, loss_dict = compute_multitask_loss(
                output, labels, boundary_tags, config
            )
            loss = loss / config.gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
            optimizer.zero_grad()

        # Accumulate losses
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            avg_loss = epoch_losses["total"] / num_batches
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f}"
            )

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Validate the model.

    Returns:
        Tuple of (loss_dict, metrics_dict).
    """
    model.eval()
    all_labels = []
    all_scores = []
    epoch_losses = {}
    num_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].to(device)

        if config.use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                output = model(audio, faces, mouths)
                _, loss_dict = compute_multitask_loss(output, labels, None, config)
        else:
            output = model(audio, faces, mouths)
            _, loss_dict = compute_multitask_loss(output, labels, None, config)

        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v
        num_batches += 1

        # Collect predictions
        probs = torch.sigmoid(output.logits.squeeze(-1))
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    # Compute metrics
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    predictions_arr = (scores_arr >= 0.5).astype(int)

    # Logging diagnostic stats
    real_count = int((labels_arr == 0).sum())
    fake_count = int((labels_arr == 1).sum())
    mean_confidence = float(scores_arr.mean())
    
    logger.info("=== Validation Diagnostics ===")
    logger.info(f"  Total validation samples: {len(labels_arr)}")
    logger.info(f"  Real (class 0) count    : {real_count}")
    logger.info(f"  Fake (class 1) count    : {fake_count}")
    logger.info(f"  Mean predicted prob     : {mean_confidence:.4f}")
    logger.info(f"  First 20 labels         : {labels_arr[:20].tolist()}")
    logger.info(f"  First 20 predictions    : {predictions_arr[:20].tolist()}")
    logger.info(f"  First 20 probabilities  : {[f'{x:.4f}' for x in scores_arr[:20]]}")
    logger.info("==============================")

    if len(np.unique(labels_arr)) > 1:
        metrics = compute_all_metrics(labels_arr, scores_arr)
    else:
        # Fallback for single-class validation split (e.g. tiny max_samples)
        acc_val = float((predictions_arr == labels_arr).mean())
        metrics = {"auc_roc": 0.5, "accuracy": acc_val, "f1": 0.0}

    return epoch_losses, metrics


def train(
    train_dataset,
    val_dataset,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    path_cfg: Optional[PathConfig] = None,
    resume_from: Optional[str] = None,
    use_cache: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Main training loop.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        model_cfg: Model config.
        train_cfg: Training config.
        path_cfg: Path config.
        resume_from: Checkpoint path to resume from.
    """
    model_cfg = model_cfg or model_config
    train_cfg = train_cfg or training_config
    path_cfg = path_cfg or path_config

    # Setup
    setup_logger(log_dir=str(path_cfg.output_dir / "logs"))
    device = get_device()
    path_cfg.ensure_dirs()

    logger.info(f"Training on device: {device}")
    logger.info(f"Training config: {train_cfg}")

    # Model
    model = DeepfakeForensicModel(config=model_cfg)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = create_optimizer(model, train_cfg)
    scheduler = create_scheduler(optimizer, train_cfg)

    # AMP scaler
    scaler = None
    if train_cfg.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # Resume
    start_epoch = 0
    best_auc = 0.0
    if resume_from:
        loaded_epoch, metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler, device=str(device)
        )
        start_epoch = loaded_epoch + 1
        best_auc = metrics.get("auc_roc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")

    # Enable caching in datasets if requested
    for dataset in [train_dataset, val_dataset]:
        if hasattr(dataset, "datasets"):  # ConcatDataset
            for sub_ds in dataset.datasets:
                sub_ds.use_cache = use_cache
                sub_ds.cache_dir = Path(cache_dir) if cache_dir else None
        else:
            dataset.use_cache = use_cache
            dataset.cache_dir = Path(cache_dir) if cache_dir else None

    # Resolve DataLoader workers and optimization settings
    import sys
    num_workers = train_cfg.num_workers
    
    # Decord CPU frames extraction is not process-safe on Windows without cache
    if sys.platform.startswith("win") and not use_cache:
        logger.warning("Windows detected without caching. Forcing num_workers=0 to prevent multiprocessing crashes.")
        num_workers = 0

    # Build loader kwargs dynamically
    loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "pin_memory": train_cfg.pin_memory,
        "collate_fn": forensic_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    else:
        loader_kwargs["num_workers"] = 0

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, train_cfg.max_epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            train_cfg, epoch, scaler
        )

        # Validate
        val_losses, val_metrics = validate(model, val_loader, device, train_cfg)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_auc = val_metrics.get("auc_roc", 0.0)

        logger.info(
            f"Epoch {epoch}/{train_cfg.max_epochs} "
            f"({epoch_time:.1f}s) | "
            f"Train loss: {train_losses['total']:.4f} | "
            f"Val loss: {val_losses['total']:.4f} | "
            f"Val AUC: {current_auc:.4f} | "
            f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
        )

        # Save checkpoint
        checkpoint_path = str(
            path_cfg.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        )
        save_checkpoint(
            model, optimizer, epoch,
            {**val_losses, **val_metrics},
            checkpoint_path, scheduler
        )

        # Best model tracking
        if current_auc > best_auc + train_cfg.min_delta:
            best_auc = current_auc
            patience_counter = 0
            best_path = str(path_cfg.checkpoint_dir / "best_model.pth")
            save_checkpoint(
                model, optimizer, epoch,
                {**val_losses, **val_metrics},
                best_path, scheduler
            )
            logger.info(f"★ New best model: AUC={best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={train_cfg.patience})"
                )
                break

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train Deepfake Forensic Model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (faceforensics, fakeavceleb, lavdf)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for testing)")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached preprocessed tensors (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Disable caching and run on-the-fly preprocessing")
    parser.add_argument("--cache-dir", default="output/cache",
                        help="Directory where preprocessed tensors are saved/loaded")
    args = parser.parse_args()

    # Override config if specified
    t_cfg = TrainingConfig()
    if args.epochs:
        t_cfg.max_epochs = args.epochs
    if args.batch_size:
        t_cfg.batch_size = args.batch_size
    if args.lr:
        t_cfg.learning_rate = args.lr

    # Create datasets
    from datasets import (
        FaceForensicsDataset, FakeAVCelebDataset,
        LAVDFDataset,
    )

    dataset_map = {
        "faceforensics": FaceForensicsDataset,
        "fakeavceleb": FakeAVCelebDataset,
        "lavdf": LAVDFDataset,
    }

    DatasetClass = dataset_map.get(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset = DatasetClass(
        args.data_root, split="train", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )
    val_dataset = DatasetClass(
        args.data_root, split="val", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )

    train(
        train_dataset, val_dataset, train_cfg=t_cfg, resume_from=args.resume,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
