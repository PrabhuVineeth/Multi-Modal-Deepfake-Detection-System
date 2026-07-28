"""
Balanced, reproducible evaluation across FakeAVCeleb, FaceForensics++, LAV-DF,
and the Joint (all-combined) dataset.

Why this exists
---------------
Every prior "all-dataset" evaluation was invalid for two independent reasons:
  1. The cached embeddings held only 3 / 18 / 40 REAL samples (an unbalanced
     random draw of 150), so the models scored "all fake" and accuracy just
     measured class prevalence.
  2. The per-dataset deployed checkpoints (best_model_fakeavceleb.pth,
     best_model_lavdf_quick_fixed.pth) were deleted and are not in version
     control, so their numbers cannot be reproduced.

This script fixes both: it extracts a BALANCED (N real + N fake) embedding set
per dataset straight from raw video, then runs a deterministic 5-fold
cross-validated train+eval of the fusion head. No deleted checkpoint is needed,
train/test never overlap, and the numbers are reproducible.

Outputs one internally-consistent table (AUC, EER, balanced accuracy,
precision, recall, F1, ECE) to output/balanced_eval/.
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset

from config import get_device, model_config
from datasets import FaceForensicsDataset, FakeAVCelebDataset, LAVDFDataset
from models.audio_encoder import AudioEncoder
from models.video_encoder import MouthEncoder, VideoEncoder
from train import forensic_collate_fn
from train_fast_fusion import FastFusionClassifier
from train_joint_balanced import FocalLoss

SEED = 42
DATASETS = {
    "fakeavceleb": ("G:/NNM24AD071/FakeAVCeleb_v1.2", FakeAVCelebDataset),
    "faceforensics": ("G:/NNM24AD071/FaceForensics++_C23", FaceForensicsDataset),
    "lavdf": ("G:/NNM24AD071/LAV-DF", LAVDFDataset),
}


def setup_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def calculate_eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2.0) * 100.0


def expected_calibration_error(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s > lo) & (s <= hi)
        if m.sum() == 0:
            continue
        conf = s[m].mean()
        acc = (y[m] == (s[m] >= 0.5)).mean()
        ece += (m.sum() / len(s)) * abs(conf - acc)
    return float(ece)


@torch.no_grad()
def extract_balanced(name, root, cls, per_class, device):
    """Extract 512-d audio/face/mouth embeddings for a balanced real/fake subset."""
    ds = cls(root, split="all", use_cache=True, cache_dir="output/cache")
    real = [i for i, s in enumerate(ds.samples) if s.label == 0]
    fake = [i for i, s in enumerate(ds.samples) if s.label == 1]
    rng = random.Random(SEED)
    rng.shuffle(real)
    rng.shuffle(fake)
    n = min(per_class, len(real), len(fake))
    idx = real[:n] + fake[:n]
    rng.shuffle(idx)
    logger.info(f"[{name}] balanced subset: {len(idx)} videos ({n} real / {n} fake)")

    loader = DataLoader(Subset(ds, idx), batch_size=8, shuffle=False,
                        collate_fn=forensic_collate_fn, num_workers=0)
    aenc = AudioEncoder(projection_dim=model_config.fusion_hidden_dim).to(device).eval()
    venc = VideoEncoder(model_name=model_config.vit_model_name,
                        projection_dim=model_config.fusion_hidden_dim).to(device).eval()
    menc = MouthEncoder(model_name=model_config.vit_model_name,
                        projection_dim=model_config.fusion_hidden_dim).to(device).eval()

    A, F, M, Y = [], [], [], []
    kept = 0
    for bi, b in enumerate(loader):
        if b is None:
            continue
        a = aenc(b["audio"].to(device))
        f = venc(b["face_frames"].to(device))
        m = menc(b["mouth_rois"].to(device))
        A.append((a.mean(1) if a.ndim == 3 else a).cpu().numpy())
        F.append((f.mean(1) if f.ndim == 3 else f).cpu().numpy())
        M.append((m.mean(1) if m.ndim == 3 else m).cpu().numpy())
        Y.append(b["label"].cpu().numpy())
        kept += len(b["label"])
        if (bi + 1) % 10 == 0:
            logger.info(f"[{name}] {kept}/{len(idx)} embedded")
    out = {
        "audio": np.concatenate(A), "face": np.concatenate(F),
        "mouth": np.concatenate(M), "labels": np.concatenate(Y).astype(int),
    }
    r = int((out["labels"] == 0).sum()); fk = int((out["labels"] == 1).sum())
    logger.info(f"[{name}] extracted {len(out['labels'])} embeddings ({r} real / {fk} fake)")
    return out


def cv_evaluate(name, data, folds, epochs, device):
    """Deterministic k-fold train+eval of the fusion head on balanced embeddings."""
    Xa, Xf, Xm, y = data["audio"], data["face"], data["mouth"], data["labels"]
    if len(np.unique(y)) < 2:
        logger.warning(f"[{name}] single class; skipping")
        return None

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    per_fold = []
    for k, (tr, te) in enumerate(skf.split(Xf, y)):
        torch.manual_seed(SEED)
        sca = [StandardScaler().fit(X[tr]) for X in (Xa, Xf, Xm)]
        trX = [s.transform(X[tr]) for s, X in zip(sca, (Xa, Xf, Xm))]
        teX = [s.transform(X[te]) for s, X in zip(sca, (Xa, Xf, Xm))]
        trds = TensorDataset(*[torch.tensor(x, dtype=torch.float32) for x in trX],
                             torch.tensor(y[tr], dtype=torch.float32))
        loader = DataLoader(trds, batch_size=32, shuffle=True)

        model = FastFusionClassifier(model_config).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        crit = FocalLoss(alpha=0.5, gamma=2.0)
        model.train()
        for _ in range(epochs):
            for a, f, m, yb in loader:
                opt.zero_grad()
                loss = crit(model(a.to(device), f.to(device), m.to(device)).squeeze(-1), yb.to(device))
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(
                torch.tensor(teX[0], dtype=torch.float32, device=device),
                torch.tensor(teX[1], dtype=torch.float32, device=device),
                torch.tensor(teX[2], dtype=torch.float32, device=device),
            ).squeeze(-1)).cpu().numpy()
        yt = y[te]
        p = (s >= 0.5).astype(int)
        per_fold.append({
            "auc": roc_auc_score(yt, s) * 100, "eer": calculate_eer(yt, s),
            "acc": accuracy_score(yt, p) * 100, "bal_acc": balanced_accuracy_score(yt, p) * 100,
            "precision": precision_score(yt, p, zero_division=0) * 100,
            "recall": recall_score(yt, p, zero_division=0) * 100,
            "f1": f1_score(yt, p, zero_division=0) * 100,
            "ece": expected_calibration_error(yt, s),
        })

    agg = {}
    for mkey in ("auc", "eer", "acc", "bal_acc", "precision", "recall", "f1", "ece"):
        v = [f[mkey] for f in per_fold]
        agg[mkey] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    agg["n_samples"] = int(len(y))
    agg["folds"] = per_fold
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150, help="real (=fake) videos per dataset")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=str, default="output/balanced_eval")
    args = ap.parse_args()

    setup_determinism()
    device = get_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── extract balanced embeddings per dataset (cache to disk) ──
    caches = {}
    for name, (root, cls) in DATASETS.items():
        npz = out_dir / f"emb_{name}.npz"
        if npz.exists():
            z = np.load(npz)
            caches[name] = {k: z[k] for k in z.files}
            logger.info(f"[{name}] loaded cached embeddings ({len(caches[name]['labels'])})")
        else:
            caches[name] = extract_balanced(name, root, cls, args.per_class, device)
            np.savez_compressed(npz, **caches[name])
            logger.info(f"[{name}] cached -> {npz}")

    # ── joint = concatenation of the three balanced sets ──
    caches["joint"] = {
        k: np.concatenate([caches[d][k] for d in DATASETS]) for k in ("audio", "face", "mouth", "labels")
    }
    logger.info(f"[joint] {len(caches['joint']['labels'])} videos "
                f"({int((caches['joint']['labels']==0).sum())} real / "
                f"{int((caches['joint']['labels']==1).sum())} fake)")

    # ── cross-validated evaluation ──
    results = {}
    for name in list(DATASETS) + ["joint"]:
        logger.info(f"evaluating {name} ({args.folds}-fold)")
        results[name] = cv_evaluate(name, caches[name], args.folds, args.epochs, device)

    # ── table ──
    print("\n" + "=" * 100)
    print(f"BALANCED CROSS-VALIDATED EVALUATION ({args.folds}-fold, deterministic)")
    print("=" * 100)
    print(f"{'Dataset':<14} | {'N':<6} | {'AUC%':<13} | {'EER%':<12} | {'Bal.Acc%':<13} | {'F1%':<13} | {'ECE':<8}")
    print("-" * 100)
    for name in list(DATASETS) + ["joint"]:
        r = results[name]
        if r is None:
            continue
        print(f"{name:<14} | {r['n_samples']:<6} | "
              f"{r['auc']['mean']:5.2f}+-{r['auc']['std']:<5.2f} | "
              f"{r['eer']['mean']:5.2f}+-{r['eer']['std']:<4.2f} | "
              f"{r['bal_acc']['mean']:5.2f}+-{r['bal_acc']['std']:<5.2f} | "
              f"{r['f1']['mean']:5.2f}+-{r['f1']['std']:<5.2f} | "
              f"{r['ece']['mean']:.3f}")
    print("=" * 100)

    payload = {
        "protocol": f"balanced {args.per_class}+{args.per_class} per dataset; "
                    f"{args.folds}-fold StratifiedKFold; fresh FastFusionClassifier per fold; "
                    "512-d embeddings from pretrained encoders; deterministic",
        "caveats": [
            "Random (not identity-disjoint) split: for FakeAVCeleb/FF++ this can be "
            "optimistic; the identity-disjoint FakeAVCeleb number is 93.97% AUC (see results/ablation).",
            "FF++ audio is unreliable (near-silent); its audio embeddings carry little signal.",
        ],
        "results": results,
    }
    with open(out_dir / "balanced_eval_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten to {out_dir / 'balanced_eval_results.json'}")


if __name__ == "__main__":
    main()
