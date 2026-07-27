"""
Modality Ablation Training script.

Loads pre-extracted features (e.g. from RetinaFace+ViT-16) and trains 
three classifier configurations under the identical 5-fold cross-validation
protocol:
  1. Audio-Only Model
  2. Visual-Only Model (Face + Mouth)
  3. Full Fusion Model

Calculates overall AUC/EER/Acc/F1 and per-manipulation-type AUC (reals vs each fake type).
"""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import get_device, model_config
from train_fast_fusion import FastFusionClassifier
from train_joint_balanced import FocalLoss

SEED = 42
BACKBONE_DIM = 768
FAKE_TYPES = ["face-swap", "lip-sync", "both"]

def setup_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0) * 100.0

# ─────────────────────────── Models ───────────────────────────

class AudioOnlyModel(nn.Module):
    def __init__(self, in_dim=BACKBONE_DIM, hidden_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, a, f, m):
        # f and m are ignored
        return self.classifier(self.proj(a))


class VisualOnlyModel(nn.Module):
    def __init__(self, in_dim=BACKBONE_DIM, hidden_dim=256):
        super().__init__()
        self.proj_f = nn.Linear(in_dim, hidden_dim)
        self.proj_m = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, a, f, m):
        # a is ignored
        x = torch.cat([self.proj_f(f), self.proj_m(m)], dim=-1)
        return self.classifier(x)


class FullFusionModel(nn.Module):
    def __init__(self, in_dim=BACKBONE_DIM, cfg=model_config):
        super().__init__()
        d = cfg.fusion_hidden_dim
        self.proj_a = nn.Linear(in_dim, d)
        self.proj_f = nn.Linear(in_dim, d)
        self.proj_m = nn.Linear(in_dim, d)
        self.head = FastFusionClassifier(cfg)

    def forward(self, a, f, m):
        return self.head(self.proj_a(a), self.proj_f(f), self.proj_m(m))

# ─────────────────────────── Training ───────────────────────────

def run_fold(model_class, tr, te, epochs, device):
    """Train the head on one fold. `tr`/`te` are (audio, face, mouth, y)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Scalers fit on training fold only
    scalers = [StandardScaler().fit(tr[i]) for i in range(3)]
    tr_x = [scalers[i].transform(tr[i]) for i in range(3)]
    te_x = [scalers[i].transform(te[i]) for i in range(3)]

    tr_ds = TensorDataset(*[torch.tensor(x, dtype=torch.float32) for x in tr_x],
                          torch.tensor(tr[3], dtype=torch.float32))
    te_ds = TensorDataset(*[torch.tensor(x, dtype=torch.float32) for x in te_x],
                          torch.tensor(te[3], dtype=torch.float32))

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = FocalLoss(alpha=0.5, gamma=2.0)

    model.train()
    for _ in range(epochs):
        for a, f, m, y in tr_loader:
            a, f, m, y = a.to(device), f.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(a, f, m).squeeze(-1), y)
            loss.backward()
            opt.step()

    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for a, f, m, y in te_loader:
            p = torch.sigmoid(model(a.to(device), f.to(device), m.to(device)).squeeze(-1))
            scores += p.cpu().numpy().tolist()
            targets += y.numpy().tolist()

    scores, targets = np.asarray(scores), np.asarray(targets)
    preds = (scores >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(targets, scores) * 100.0,
        "eer": calculate_eer(targets, scores),
        "acc": accuracy_score(targets, preds) * 100.0,
        "bal_acc": balanced_accuracy_score(targets, preds) * 100.0,
        "f1": f1_score(targets, preds, zero_division=0) * 100.0,
    }
    return metrics, scores, targets


def evaluate_configuration(model_class, cache, backbone, folds, epochs, device):
    X_f = cache[f"face_{backbone}"]
    X_m = cache[f"mouth_{backbone}"]
    X_a = cache["audio"]
    y = cache["labels"]
    groups = cache["identities"]
    manip = cache["manip"]

    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)

    per_fold = []
    pooled_scores, pooled_targets, pooled_manip = [], [], []
    for k, (tr_i, te_i) in enumerate(splitter.split(X_f, y, groups)):
        metrics, sc, tg = run_fold(
            model_class,
            (X_a[tr_i], X_f[tr_i], X_m[tr_i], y[tr_i]),
            (X_a[te_i], X_f[te_i], X_m[te_i], y[te_i]),
            epochs, device,
        )
        per_fold.append(metrics)
        pooled_scores.append(sc)
        pooled_targets.append(tg)
        pooled_manip.append(manip[te_i])

    agg = {}
    for metric in ("auc", "eer", "acc", "bal_acc", "f1"):
        vals = [f[metric] for f in per_fold]
        agg[f"{metric}_mean"] = float(np.mean(vals))
        agg[f"{metric}_std"] = float(np.std(vals))

    ps = np.concatenate(pooled_scores)
    pt = np.concatenate(pooled_targets)
    pm = np.concatenate(pooled_manip)
    per_cat = {}
    for c in FAKE_TYPES:
        mask = (pt == 0) | (pm == c)
        yt, ys = pt[mask], ps[mask]
        n_fake = int((pm[mask] == c).sum())
        entry = {"n_fake": n_fake, "n_real": int((yt == 0).sum())}
        entry["auc"] = float(roc_auc_score(yt, ys) * 100.0) if (n_fake > 0 and len(np.unique(yt)) == 2) else None
        per_cat[c] = entry
    agg["per_category_auc"] = per_cat

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="output/ablation_fakeavceleb_stratified/features_retinaface.npz",
                    help="path to cached NPZ features file")
    ap.add_argument("--backbone", type=str, default="ViT-16", help="backbone to evaluate")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    setup_determinism()
    device = get_device()

    logger.info(f"Loading cached features from {args.features}...")
    cache = dict(np.load(args.features, allow_pickle=True))

    configs = {
        "Audio-Only": AudioOnlyModel,
        "Visual-Only": VisualOnlyModel,
        "Full Fusion": FullFusionModel,
    }

    results = {}
    for name, model_class in configs.items():
        logger.info(f"Training and evaluating configuration: {name}...")
        results[name] = evaluate_configuration(
            model_class, cache, args.backbone, args.folds, args.epochs, device
        )

    # Output formatted tables
    print("\n" + "=" * 90)
    print(f"MODALITY ABLATION STUDY RESULTS ({args.backbone}, 5-fold identity-disjoint)")
    print("=" * 90)
    print(f"{'Configuration':<15} | {'Mean AUC%':<15} | {'Mean EER%':<15} | {'Mean F1%':<15}")
    print("-" * 90)
    for name in configs:
        r = results[name]
        print(f"{name:<15} | {r['auc_mean']:6.2f}+-{r['auc_std']:<6.2f} | {r['eer_mean']:5.2f}+-{r['eer_std']:<6.2f} | {r['f1_mean']:6.2f}+-{r['f1_std']:<6.2f}")

    print("\n" + "=" * 90)
    print("PER-MANIPULATION-TYPE AUC BREAKDOWN (Reals vs One Fake Type)")
    print("=" * 90)
    print(f"{'Configuration':<15} | {'Face-Swap%':<15} | {'Lip-Sync% (Audio-only)':<23} | {'Both% (AV)':<15}")
    print("-" * 90)
    for name in configs:
        r = results[name]["per_category_auc"]
        print(f"{name:<15} | {r['face-swap']['auc']:10.2f}     | {r['lip-sync']['auc']:10.2f}              | {r['both']['auc']:10.2f}")
    print("=" * 90)

    # Save to json file
    out_dir = Path(args.features).parent
    out_path = out_dir / "modality_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
