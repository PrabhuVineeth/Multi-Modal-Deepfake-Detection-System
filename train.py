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
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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
    """Custom collate: skip failed samples, stack tensors, and keep metadata as a list."""
    from torch.utils.data._utils.collate import default_collate
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k == "metadata":
            result[k] = [d[k] for d in batch]  # list of dicts, not stacked
        else:
            result[k] = default_collate([d[k] for d in batch])
    return result


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary Focal Loss computed from raw logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Uses BCEWithLogits for numerical stability (no sigmoid + BCE separately).

    Args:
        logits:  Raw model logits [B] or [B, 1].
        targets: Binary labels [B], float (0.0 or 1.0).
        gamma:   Focusing exponent. 0 → standard BCE.
        alpha:   Optional scalar balance weight for the positive class.
                 None → no alpha balancing (plain focal only).
        weight:  Optional per-sample weight tensor [B] (e.g. from class-weight re-weighting).

    Returns:
        Scalar mean focal loss.
    """
    # Standard BCE with logits (per-sample, no reduction)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    # Compute p_t = sigmoid(logit) when y=1, else 1 - sigmoid(logit)
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)

    # Focal modulation: (1 - p_t)^gamma
    focal_weight = (1.0 - p_t).pow(gamma)

    loss = focal_weight * bce

    # Optional alpha balancing
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    # Optional per-sample weight (from class imbalance re-weighting)
    if weight is not None:
        loss = loss * weight

    return loss.mean()


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


def get_scheduler(optimizer, config, total_steps, args):
    """Create scheduler with CLI override support."""
    scheduler_type = args.scheduler if args and args.scheduler else config.scheduler
    warmup_steps = args.warmup_steps if args and args.warmup_steps is not None else config.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)  # Linear warmup 0→1
        if scheduler_type == "none":
            return 1.0  # Constant LR
        elif scheduler_type == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
        else:
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_multitask_loss(
    output: ForensicOutput,
    labels: torch.Tensor,
    boundary_tags: Optional[torch.Tensor],
    config: TrainingConfig,
    weight: Optional[torch.Tensor] = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the multi-task loss.

    Args:
        output: ForensicOutput from the model.
        labels: Ground truth labels [B] (0=REAL, 1=FAKE).
        boundary_tags: Optional per-frame boundary tags [B, T].
        config: Training configuration with loss weights.
        weight: Configurable weight for per-sample/class scaling.

    Returns:
        Tuple of (total_loss, loss_dict).
    """
    losses = {}
    # Cast tensors to float32 for high precision dynamic range (prevents FP16 overflow/NaN under weighted loss)
    logits = output.logits.squeeze(-1).float()
    labels = labels.float()
    weight_fp32 = weight.float() if weight is not None else None

    # Clamp logits to prevent exponential overflow in FP16/FP32
    logits = torch.clamp(logits, min=-15.0, max=15.0)

    # 1. Classification loss (BCE with logits, or Focal loss)
    if use_focal:
        cls_loss = focal_loss_with_logits(
            logits, labels,
            gamma=focal_gamma,
            alpha=focal_alpha,
            weight=weight_fp32,
        )
    else:
        cls_loss = F.binary_cross_entropy_with_logits(
            logits, labels, weight=weight_fp32
        )

    # 2. Lip sync loss
    lip_target = labels.unsqueeze(-1)  # FAKE=1 → high score
    lip_loss = F.mse_loss(output.lip_sync_score.float(), lip_target)

    # 3. Identity loss
    id_loss = F.mse_loss(output.identity_score.float(), lip_target)

    # 4. Temporal loss
    temp_loss = F.mse_loss(output.temporal_score.float(), lip_target)

    # 5. AV sync loss
    sync_loss = F.mse_loss(output.av_sync_score.float(), lip_target)

    # 6. Boundary loss (TFBD CRF)
    boundary_loss = torch.tensor(0.0, dtype=torch.float32, device=labels.device)
    if output.boundary_loss is not None:
        boundary_loss = output.boundary_loss.float()

    # Weighted sum
    total = (
        config.lambda_cls * cls_loss
        + config.lambda_lip * lip_loss
        + config.lambda_id * id_loss
        + config.lambda_temp * temp_loss
        + config.lambda_sync * sync_loss
        + config.lambda_boundary * boundary_loss
    )
    
    losses = {
        "cls": cls_loss.item(),
        "lip": lip_loss.item(),
        "id": id_loss.item(),
        "temp": temp_loss.item(),
        "sync": sync_loss.item(),
        "boundary": boundary_loss.item(),
        "total": total.item()
    }
    
    if torch.isnan(total):
        logger.error(
            f"NaN Loss detected! "
            f"logits_has_nan={torch.isnan(logits).any().item()}, "
            f"cls={cls_loss.item():.4f}, "
            f"lip={lip_loss.item():.4f}, "
            f"id={id_loss.item():.4f}, "
            f"temp={temp_loss.item():.4f}, "
            f"sync={sync_loss.item():.4f}, "
            f"boundary={boundary_loss.item():.4f}"
        )

    return total, losses


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    real_weight: Optional[torch.Tensor] = None,
    fake_weight: Optional[torch.Tensor] = None,
    scheduler = None,
    use_bf16: bool = False,
    visual_only: bool = False,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            logger.warning(f"All samples failed in batch {batch_idx+1}; skipping batch.")
            continue
        audio = batch["audio"].to(device)
        if visual_only:
            audio = torch.zeros_like(audio)

        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].to(device)
        boundary_tags = batch.get("boundary_tags")
        if boundary_tags is not None:
            boundary_tags = boundary_tags.to(device)

        # NaN Guard: Check inputs
        if torch.isnan(audio).any() or torch.isnan(faces).any() or torch.isnan(mouths).any():
            logger.warning(f"NaN detected in batch inputs at batch {batch_idx+1}! Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if real_weight is not None and fake_weight is not None:
            sample_weights = torch.where(labels == 0, real_weight, fake_weight)
        else:
            sample_weights = None

        # Forward pass
        if config.use_amp and device.type == "cuda":
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.amp.autocast('cuda', dtype=dtype):
                output = model(
                    audio, faces, mouths,
                    boundary_tags=boundary_tags,
                )
                
                # NaN Guard: Check outputs
                if torch.isnan(output.logits).any():
                    logger.warning(f"NaN detected in logits at batch {batch_idx+1}! Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
                loss, loss_dict = compute_multitask_loss(
                    output, labels, boundary_tags, config, weight=sample_weights,
                    use_focal=use_focal, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
                )
                
                # NaN Guard: Check loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx+1}! Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
                loss = loss / config.gradient_accumulation_steps
        else:
            output = model(
                audio, faces, mouths,
                boundary_tags=boundary_tags,
            )
            
            # NaN Guard: Check outputs
            if torch.isnan(output.logits).any():
                logger.warning(f"NaN detected in logits at batch {batch_idx+1}! Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
                
            loss, loss_dict = compute_multitask_loss(
                output, labels, boundary_tags, config, weight=sample_weights,
                use_focal=use_focal, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            )
            
            # NaN Guard: Check loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx+1}! Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
                
            loss = loss / config.gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # NaN Guard: Check gradients before optimizer step
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                logger.warning(f"NaN detected in gradients at batch {batch_idx+1}! Skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                continue
                
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
            optimizer.zero_grad(set_to_none=True)

            # Step scheduler only when optimizer steps
            if scheduler is not None:
                scheduler.step()

        # Log learning rate at batch 1 and every 50 batches (reduce log/sync overhead)
        if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Accumulate losses
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Loss breakdown: cls={loss_dict['cls']:.4f} | lip={loss_dict['lip']:.4f} | id={loss_dict['id']:.4f} | temp={loss_dict['temp']:.4f} | sync={loss_dict['sync']:.4f} | boundary={loss_dict['boundary']:.4f}")

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    use_bf16: bool = False,
    visual_only: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Validate the model.

    Returns:
        Tuple of (loss_dict, metrics_dict).
    """
    model.eval()
    all_labels = []
    all_scores = []
    all_logits = []
    epoch_losses = {}
    num_batches = 0

    for batch in dataloader:
        if batch is None:
            logger.warning("All samples failed in validation batch; skipping batch.")
            continue
        audio = batch["audio"].to(device)
        if visual_only:
            audio = torch.zeros_like(audio)
        faces = batch["face_frames"].to(device)
        mouths = batch["mouth_rois"].to(device)
        labels = batch["label"].to(device)


        if config.use_amp and device.type == "cuda":
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.amp.autocast('cuda', dtype=dtype):
                output = model(audio, faces, mouths)
                _, loss_dict = compute_multitask_loss(output, labels, None, config, weight=None)
        else:
            output = model(audio, faces, mouths)
            _, loss_dict = compute_multitask_loss(output, labels, None, config, weight=None)

        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v
        num_batches += 1

        # Collect predictions
        logits_val = output.logits.squeeze(-1)
        probs = torch.sigmoid(logits_val)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.float().cpu().numpy())
        all_logits.extend(logits_val.reshape(-1).float().cpu().numpy().tolist())

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
    pred_real_count = int((predictions_arr == 0).sum())
    pred_fake_count = int((predictions_arr == 1).sum())
    mean_confidence = float(scores_arr.mean())
    
    logger.info("=== Validation Diagnostics ===")
    logger.info(f"  Total validation samples: {len(labels_arr)}")
    logger.info(f"  Real (class 0) count    : {real_count}")
    logger.info(f"  Fake (class 1) count    : {fake_count}")
    logger.info(f"  Predicted Real count    : {pred_real_count}")
    logger.info(f"  Predicted Fake count    : {pred_fake_count}")
    logger.info(f"  Mean predicted prob     : {mean_confidence:.4f}")
    logger.info(f"  First 20 labels         : {labels_arr[:20].tolist()}")
    logger.info(f"  First 20 predictions    : {predictions_arr[:20].tolist()}")
    logger.info(f"  First 20 probabilities  : {[f'{x:.4f}' for x in scores_arr[:20]]}")
    logger.info("==============================")

    # 5. Log prediction distribution in validate()
    pred_labels = (torch.sigmoid(torch.tensor(all_logits)) > 0.5).int().tolist()
    n_pred_real = pred_labels.count(0)
    n_pred_fake = pred_labels.count(1)
    print(f"  Prediction distribution → Real: {n_pred_real} | Fake: {n_pred_fake}")

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
    args = None,
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
    # AMP is enabled via config.use_amp (True by default for RTX 4070)
    path_cfg = path_cfg or path_config

    # Setup
    setup_logger(log_dir=str(path_cfg.output_dir / "logs"))
    device = get_device()
    path_cfg.ensure_dirs()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmarking enabled for optimized kernel selection")

    logger.info(f"Training on device: {device}")
    logger.info(f"Training config: {train_cfg}")

    # Enable caching in datasets if requested
    for dataset in [train_dataset, val_dataset]:
        if isinstance(dataset, Subset):
            ds = dataset.dataset
        else:
            ds = dataset
            
        if hasattr(ds, "datasets"):  # ConcatDataset
            for sub_ds in ds.datasets:
                sub_ds.use_cache = use_cache
                sub_ds.cache_dir = Path(cache_dir) if cache_dir else None
        else:
            ds.use_cache = use_cache
            ds.cache_dir = Path(cache_dir) if cache_dir else None

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
        loader_kwargs["prefetch_factor"] = 2  # Set to 2 to minimize RAM footprint/Windows spawn lag
    else:
        loader_kwargs["num_workers"] = 0

    # Extract labels from train_dataset
    if isinstance(train_dataset, Subset):
        train_labels = [train_dataset.dataset.samples[i].label for i in train_dataset.indices]
    elif hasattr(train_dataset, "datasets"):
        train_labels = []
        for sub_ds in train_dataset.datasets:
            train_labels.extend([s.label for s in sub_ds.samples])
    else:
        train_labels = [s.label for s in train_dataset.samples]
        
    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    
    n_real_safe = max(n_real, 1)
    n_fake_safe = max(n_fake, 1)
    
    # Compute per-sample loss weights (capped at 3.0)
    raw_weight = n_fake_safe / n_real_safe
    real_weight = torch.tensor(min(raw_weight, 3.0), dtype=torch.float32).to(device)
    fake_weight = torch.tensor(1.0, dtype=torch.float32).to(device)
    logger.info(f"Per-sample weights: Real={real_weight.item():.2f}x, Fake=1.0x")

    logger.info(f"Auto-computed training split label counts: real={n_real}, fake={n_fake}")

    # Optional WeightedRandomSampler (--balanced-sampler flag)
    use_balanced_sampler = bool(args and getattr(args, "balanced_sampler", False))
    sampler = None
    if use_balanced_sampler:
        class_counts = [n_real_safe, n_fake_safe]
        sampler_weights_list = [1.0 / class_counts[int(l)] for l in train_labels]
        sampler_weights_tensor = torch.tensor(sampler_weights_list, dtype=torch.float32)
        sampler = WeightedRandomSampler(
            sampler_weights_tensor, num_samples=len(sampler_weights_tensor), replacement=True
        )
        logger.info(
            f"Balanced sampler ENABLED | class_counts=[real={n_real_safe}, fake={n_fake_safe}] "
            f"| sampler_weight[real]={1.0/n_real_safe:.6f}, sampler_weight[fake]={1.0/n_fake_safe:.6f}"
        )
    else:
        logger.info("Balanced sampler DISABLED (using shuffle=True). Pass --balanced-sampler to enable.")

    # Focal loss configuration
    use_focal = bool(args and getattr(args, "focal_loss", False))
    focal_gamma = float(getattr(args, "focal_gamma", 2.0)) if args else 2.0
    focal_alpha = getattr(args, "focal_alpha", None) if args else None
    if use_focal:
        logger.info(
            f"Focal loss ENABLED | gamma={focal_gamma:.2f} | "
            f"alpha={'%.4f' % focal_alpha if focal_alpha is not None else 'None (no alpha balancing)'}"
        )
    else:
        logger.info("Focal loss DISABLED (using standard BCE). Pass --focal-loss to enable.")

    # Data loaders
    if sampler is not None:
        # sampler is mutually exclusive with shuffle
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            drop_last=True,
            **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **loader_kwargs
        )
    # Subsetting validation dataset if max_val_samples is specified
    if args and hasattr(args, "max_val_samples") and args.max_val_samples is not None:
        if isinstance(val_dataset, Subset):
            val_dataset = Subset(val_dataset.dataset, val_dataset.indices[:args.max_val_samples])
        else:
            val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), args.max_val_samples))))
        logger.info(f"Subsetting validation dataset to max_val_samples: {len(val_dataset)}")

    val_batch_size = min(train_cfg.batch_size * 2, 16)
    if args and hasattr(args, "val_batch_size") and args.val_batch_size is not None:
        val_batch_size = args.val_batch_size
        logger.info(f"Overriding validation batch size: {val_batch_size}")

    val_loader_kwargs = {
        **loader_kwargs,
        "batch_size": val_batch_size,
        "num_workers": 0,
        "persistent_workers": False
    }
    if "prefetch_factor" in val_loader_kwargs:
        del val_loader_kwargs["prefetch_factor"]

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **val_loader_kwargs
    )

    # Model
    model = DeepfakeForensicModel(config=model_cfg)
    model.to(device)

    # Optimizer
    visual_only = bool(args and getattr(args, "visual_only", False))
    optimizer = create_optimizer(model, train_cfg)

    # Scheduler
    total_train_steps = (len(train_loader) // train_cfg.gradient_accumulation_steps) * train_cfg.max_epochs
    scheduler = get_scheduler(optimizer, train_cfg, total_train_steps, args)

    logger.info(f"Scheduler: {args.scheduler or train_cfg.scheduler if args else train_cfg.scheduler}")
    logger.info(f"Warmup steps: {args.warmup_steps or train_cfg.warmup_steps if args else train_cfg.warmup_steps}")
    logger.info(f"Total train steps: {total_train_steps}")

    # AMP configuration (bfloat16 auto-detection for RTX 4070)
    scaler = None
    use_bf16 = False
    if train_cfg.use_amp and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            logger.info("NVIDIA RTX 4070 supports native bfloat16. Enabling BF16 AMP (No GradScaler needed)")
            use_bf16 = True
        else:
            logger.info("BF16 not supported. Enabling FP16 AMP with GradScaler")
            scaler = torch.amp.GradScaler('cuda')

    # Resume checkpoint
    start_epoch = 0
    best_auc = 0.0
    patience_counter = 0
    if resume_from:
        fine_tune = bool(args and getattr(args, "fine_tune", False))
        if fine_tune:
            loaded_epoch, metrics, _ = load_checkpoint(
                resume_from, model, optimizer=None, scheduler=None,
                device=str(device), scaler=None
            )
            logger.info(
                "Loaded checkpoint weights for fine-tuning; "
                "resetting epoch, optimizer, scheduler, best AUC, and patience."
            )
        else:
            loaded_epoch, metrics, patience_counter = load_checkpoint(
                resume_from, model, optimizer, scheduler, device=str(device), scaler=scaler
            )
        # Override optimizer learning rate with --lr if specified
        if args and args.lr:
            new_lr = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[0]['lr'] = new_lr * 0.1
                optimizer.param_groups[1]['lr'] = new_lr
                if scheduler is not None:
                    scheduler.base_lrs = [new_lr * 0.1, new_lr]
                logger.info(f"Overrode optimizer learning rate with --lr: backbone={new_lr*0.1:.2e}, head={new_lr:.2e}")
            else:
                optimizer.param_groups[0]['lr'] = new_lr
                if scheduler is not None:
                    scheduler.base_lrs = [new_lr]
                logger.info(f"Overrode optimizer learning rate with --lr: {new_lr:.2e}")
        if fine_tune:
            start_epoch = 0
            best_auc = 0.0
            patience_counter = 0
            # Freeze backbone encoders — only train fusion/head layers
            frozen = 0
            for name, param in model.named_parameters():
                if any(k in name for k in ["audio_encoder.backbone", "video_encoder.backbone", "mouth_encoder"]):
                    param.requires_grad = False
                    frozen += 1
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Fine-tune mode: froze {frozen} backbone params. Trainable params: {trainable:,}")

        else:
            start_epoch = loaded_epoch + 1
            best_auc = metrics.get("auc_roc", 0.0)
            best_epoch = loaded_epoch

            # Load historical best AUC from this run's best checkpoint if it exists.
            best_checkpoint_name = getattr(args, "best_checkpoint_name", "best_model.pth") if args else "best_model.pth"
            best_model_path = path_cfg.checkpoint_dir / best_checkpoint_name
            if best_model_path.exists():
                try:
                    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    historical_best = best_checkpoint.get("metrics", {}).get("auc_roc", 0.0)
                    if historical_best > best_auc:
                        best_auc = historical_best
                        best_epoch = best_checkpoint.get("epoch", loaded_epoch)
                        logger.info(
                            f"Loaded historical best AUC from {best_checkpoint_name}: "
                            f"{best_auc:.4f} (epoch {best_epoch})"
                        )
                except Exception as e:
                    logger.warning(f"Could not load historical best AUC from {best_checkpoint_name}: {e}")

            # Correct patience_counter to account for epochs since the historical best
            patience_counter = max(0, start_epoch - 1 - best_epoch)
        logger.info(f"Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}, patience={patience_counter}")

    # ── Configuration summary ─────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**2) if device.type == "cuda" else 0
    logger.info("=" * 60)
    logger.info("  TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Batch size       : {train_cfg.batch_size}")
    logger.info(f"  AMP enabled      : {train_cfg.use_amp} (dtype={'bfloat16' if use_bf16 else 'float16'})")
    logger.info(f"  DataLoader workers: {num_workers}")
    logger.info(f"  GPU name         : {gpu_name}")
    logger.info(f"  GPU memory (total): {gpu_mem} MB")
    logger.info(f"  PyTorch version  : {torch.__version__}")
    logger.info(f"  CUDA version     : {torch.version.cuda}")
    logger.info("=" * 60)
    # ─────────────────────────────────────────────────────────────────────

    # Training loop
    for epoch in range(start_epoch, train_cfg.max_epochs):
        epoch_start = time.time()

        # ── GPU memory at epoch start ──────────────────────────────────────
        if device.type == "cuda":
            alloc  = torch.cuda.memory_allocated(device) / (1024**2)
            reserv = torch.cuda.memory_reserved(device)  / (1024**2)
            max_al = torch.cuda.max_memory_allocated(device) / (1024**2)
            logger.info(
                f"[GPU] Epoch {epoch} start | "
                f"Allocated: {alloc:.0f} MB | "
                f"Reserved: {reserv:.0f} MB | "
                f"Max Allocated: {max_al:.0f} MB"
            )
        # ──────────────────────────────────────────────────────────────────

        # Log both backbone and head LR at epoch start
        backbone_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        logger.info(f"Epoch {epoch} start | Backbone LR={backbone_lr:.2e} | Head LR={head_lr:.2e}")

        train_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            train_cfg, epoch, scaler,
            real_weight=real_weight, fake_weight=fake_weight,
            scheduler=scheduler, use_bf16=use_bf16,
            visual_only=visual_only,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )

        # Validate
        val_every = args.val_every if args and hasattr(args, "val_every") and args.val_every is not None else 1
        is_val_epoch = (epoch + 1) % val_every == 0 or epoch == train_cfg.max_epochs - 1

        if is_val_epoch:
            val_losses, val_metrics = validate(
                model, val_loader, device, train_cfg,
                use_bf16=use_bf16, visual_only=visual_only
            )

            epoch_time = time.time() - epoch_start
            avg_batch_time = epoch_time / max(len(train_loader), 1)
            epochs_left = train_cfg.max_epochs - epoch - 1
            eta_seconds = avg_batch_time * len(train_loader) * epochs_left
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            current_auc = val_metrics.get("auc_roc", 0.0)

            logger.info(
                f"Epoch {epoch}/{train_cfg.max_epochs} "
                f"({epoch_time:.1f}s) | "
                f"Avg batch: {avg_batch_time:.2f}s | "
                f"ETA: {eta_h}h {eta_m}m | "
                f"Train loss: {train_losses['total']:.4f} | "
                f"Val loss: {val_losses['total']:.4f} | "
                f"Val AUC: {current_auc:.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
            )

            # Best model tracking based on validation AUC
            val_auc = current_auc
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                best_checkpoint_name = getattr(args, "best_checkpoint_name", "best_model.pth") if args else "best_model.pth"
                best_path = str(path_cfg.checkpoint_dir / best_checkpoint_name)
                save_checkpoint(
                    model, optimizer, epoch,
                    {**val_losses, **val_metrics},
                    best_path, scheduler,
                    scaler=scaler, patience_counter=patience_counter
                )
                print(f"  Best model saved (AUC: {best_auc:.4f})")
                logger.info(f"[NEW BEST] New best model: AUC={best_auc:.4f}")
            else:
                patience_counter += 1

            # Save checkpoint with updated patience_counter
            checkpoint_path = str(
                path_cfg.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                model, optimizer, epoch,
                {**val_losses, **val_metrics},
                checkpoint_path, scheduler,
                scaler=scaler, patience_counter=patience_counter
            )
        else:
            epoch_time = time.time() - epoch_start
            avg_batch_time = epoch_time / max(len(train_loader), 1)
            epochs_left = train_cfg.max_epochs - epoch - 1
            eta_seconds = avg_batch_time * len(train_loader) * epochs_left
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            
            logger.info(
                f"Epoch {epoch}/{train_cfg.max_epochs} "
                f"({epoch_time:.1f}s) | "
                f"Avg batch: {avg_batch_time:.2f}s | "
                f"ETA: {eta_h}h {eta_m}m | "
                f"Train loss: {train_losses['total']:.4f} | "
                f"Validation skipped (val_every={val_every})"
            )
            
            # Save periodic checkpoint without altering validation metrics or patience_counter
            checkpoint_path = str(
                path_cfg.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                model, optimizer, epoch,
                train_losses,
                checkpoint_path, scheduler,
                scaler=scaler, patience_counter=patience_counter
            )

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
    parser.add_argument("--fine-tune", action="store_true", default=False,
                        help="Load --resume weights only and start a fresh fine-tuning run")
    parser.add_argument("--best-checkpoint-name", type=str, default="best_model.pth",
                        help="Filename for the best checkpoint saved under checkpoints/")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for testing)")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached preprocessed tensors (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Disable caching and run on-the-fly preprocessing")
    parser.add_argument("--cache-dir", default="output/cache",
                        help="Directory where preprocessed tensors are saved/loaded")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Number of steps for learning rate warmup")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Perform validation every N epochs (default: 1)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit validation dataset to a maximum number of samples")
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="Override validation batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override default number of DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", default=False,
                        help="Disable Automatic Mixed Precision (AMP) training")
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cosine", "step", "none"],
                        help="Override config scheduler. none=constant LR after warmup.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--visual-only", action="store_true", default=False,
                        help="Train/evaluate in visual-only mode (zero audio, disable sync losses)")
    parser.add_argument("--disable-audio", dest="visual_only", action="store_true",
                        help="Alias for --visual-only")
    parser.add_argument("--balanced-sampler", action="store_true", default=False,
                        help="Enable WeightedRandomSampler to balance class 0/1 training batches")
    parser.add_argument("--focal-loss", action="store_true", default=False,
                        help="Replace BCE classification loss with Binary Focal Loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss focusing exponent gamma (default: 2.0)")
    parser.add_argument("--focal-alpha", type=float, default=None,
                        help="Focal loss alpha for positive class (optional, e.g. 0.25). None=disabled.")
    args = parser.parse_args()


    # Set random seed if specified
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Override config if specified
    t_cfg = TrainingConfig()
    if args.epochs:
        t_cfg.max_epochs = args.epochs
    if args.batch_size:
        t_cfg.batch_size = args.batch_size
    if args.lr:
        t_cfg.learning_rate = args.lr
    if args.grad_accum:
        t_cfg.gradient_accumulation_steps = args.grad_accum
    if args.warmup_steps is not None:
        t_cfg.warmup_steps = args.warmup_steps
    elif args.max_samples == 500:
        t_cfg.warmup_steps = 100
    if args.scheduler is not None:
        t_cfg.scheduler = args.scheduler
    if args.num_workers is not None:
        t_cfg.num_workers = args.num_workers
    if getattr(args, "visual_only", False):
        t_cfg.lambda_sync = 0.0
        t_cfg.lambda_lip = 0.0
        t_cfg.lambda_id = 0.0
        logger.info("Visual-only mode enabled: zeroing sync, lip, and id losses")
    if args.no_amp:
        t_cfg.use_amp = False

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

    # Use stratified split on a single dataset to guarantee class distribution
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    dataset = DatasetClass(
        args.data_root, split="all", max_samples=args.max_samples,
        use_cache=args.use_cache, cache_dir=args.cache_dir
    )

    all_labels = [dataset.samples[i].label for i in range(len(dataset))]
    all_idx = list(range(len(dataset)))

    # Balance validation and test splits: equal real and fake samples
    # Separate real (label 0) and fake (label 1) indices
    real_idx = [i for i in all_idx if all_labels[i] == 0]
    fake_idx = [i for i in all_idx if all_labels[i] == 1]

    # Split real samples: 80% train, 10% val, 10% test
    import random
    rng = random.Random(42)
    real_idx_shuffled = real_idx.copy()
    rng.shuffle(real_idx_shuffled)

    n_real = len(real_idx_shuffled)
    train_real_end = int(n_real * 0.8)
    val_real_end = train_real_end + int(n_real * 0.1)

    train_real_idx = real_idx_shuffled[:train_real_end]
    val_real_idx = real_idx_shuffled[train_real_end:val_real_end]
    test_real_idx = real_idx_shuffled[val_real_end:]

    # Match fake samples for val and test to balance them
    val_fake_count = min(len(val_real_idx), len(fake_idx))
    test_fake_count = min(len(test_real_idx), len(fake_idx) - val_fake_count)

    fake_idx_shuffled = fake_idx.copy()
    rng.shuffle(fake_idx_shuffled)

    val_fake_idx = fake_idx_shuffled[:val_fake_count]
    test_fake_idx = fake_idx_shuffled[val_fake_count:val_fake_count + test_fake_count]
    train_fake_idx = fake_idx_shuffled[val_fake_count + test_fake_count:]

    # Combine real and fake indices
    train_idx = train_real_idx + train_fake_idx
    val_idx = val_real_idx + val_fake_idx
    test_idx = test_real_idx + test_fake_idx

    # Shuffle lists to mix real and fake samples within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    val_real_count = sum(1 for i in val_idx if all_labels[i] == 0)
    logger.info(f"Val split: {len(val_idx)} samples ({val_real_count} real)")

    train(
        train_dataset, val_dataset, train_cfg=t_cfg, resume_from=args.resume,
        use_cache=args.use_cache, cache_dir=args.cache_dir, args=args
    )


if __name__ == "__main__":
    main()
