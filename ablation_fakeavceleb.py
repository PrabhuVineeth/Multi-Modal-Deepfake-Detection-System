"""
FakeAVCeleb 2x3 ablation: face detectors x visual backbones.

Answers two separate questions, sharing RetinaFace+ViT-16 (the deployed system)
as the anchor cell:

  Table 1 - detector arm  (ViT-16 fixed):   RetinaFace vs MTCNN
  Table 2 - backbone arm  (RetinaFace fixed): ViT-16 vs ViT-32 vs Swin-Tiny

Protocol
--------
Frozen pretrained backbones (linear-probe style): only the fusion head trains,
so each cell measures the quality of the backbone's representation rather than
how well it fine-tunes.

Face detection depends on the detector, not the backbone, so each video is
preprocessed exactly ONCE per detector and all three backbones encode the
identical crops in the same pass. This is 3x cheaper than re-detecting per cell
and, more importantly, removes face detection as a confound in the backbone arm.

Evaluation is 5-fold stratified group k-fold grouped on FakeAVCeleb identity
(id00XXX). FakeAVCeleb has exactly one real video per identity and generates
every fake from it, so a video-level split puts a test fake's source real video
into training. That leakage pushes all cells to ceiling and erases the very
differences this study is meant to measure.

Usage:
    python ablation_fakeavceleb.py [--max-videos N] [--folds K] [--out DIR]
"""

import argparse
import json
import os
import re
import random
import time

# Must be set before the first CUDA/cuBLAS call for deterministic GEMMs; setting
# it at import time guarantees that regardless of when the device is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from collections import Counter, defaultdict
from pathlib import Path

import cv2
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

from config import PreprocessConfig, get_device, model_config
from datasets.fakeavceleb import FakeAVCelebDataset
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder
from preprocessing.pipeline import PreprocessingPipeline
from train_fast_fusion import FastFusionClassifier
from train_joint_balanced import FocalLoss

FAKEAVCELEB_ROOT = "G:/NNM24AD071/FakeAVCeleb_v1.2"
SEED = 42

DETECTORS = ["retinaface", "mtcnn"]
BACKBONES = {
    # google/vit-base-patch32-224 does not exist on the Hub; the -in21k variant is
    # the real ViT-B/32 release. Recorded in the output so the pretraining
    # asymmetry vs ViT-16 (in21k->in1k) is visible in the writeup.
    "ViT-16": "google/vit-base-patch16-224",
    "ViT-32": "google/vit-base-patch32-224-in21k",
    "Swin-Tiny": "microsoft/swin-tiny-patch4-window7-224",
}
ANCHOR = ("retinaface", "ViT-16")

MAX_FRAMES = 32
BACKBONE_DIM = 768


# ─────────────────────────── data ───────────────────────────

def identity_of(video_path: str) -> str:
    """FakeAVCeleb encodes identity in the path: .../<race>/<gender>/id00076/x.mp4"""
    m = re.search(r"(id\d+)", str(video_path).replace("\\", "/"))
    return m.group(1) if m else f"UNK::{Path(video_path).stem}"


# The three fake sub-categories, in a fixed order so remainders are assigned
# deterministically. lip-sync (RealVideo-FakeAudio) is the audio-only class and
# the whole reason to stratify: an unstratified draw from the pooled 21,066 gives
# it ~2.4% share, leaving the audio branch untestable.
FAKE_TYPES = ["face-swap", "lip-sync", "both"]


def load_samples(fake_total=500, stratify=True, max_videos=None):
    """Real + fake subset, deterministic under SEED.

    Args:
        fake_total: number of fake videos (real is always the full 500).
        stratify: draw fakes evenly across the three sub-categories rather than
            proportionally from the pool. Equal split so the audio-only class is
            represented; falls back to proportional if a category runs short.
    """
    dataset = FakeAVCelebDataset(FAKEAVCELEB_ROOT, split="all")
    real = [s for s in dataset.samples if s.label == 0]
    by_type = defaultdict(list)
    for s in dataset.samples:
        if s.label == 1:
            by_type[s.manipulation_type].append(s)
    logger.info(
        f"FakeAVCeleb full: {len(real)} real / {sum(len(v) for v in by_type.values())} fake "
        f"({ {t: len(by_type[t]) for t in FAKE_TYPES} })"
    )

    rng = random.Random(SEED)
    rng.shuffle(real)
    for t in FAKE_TYPES:
        rng.shuffle(by_type[t])

    n_real = min(len(real), 500)
    if max_videos:
        n_real = min(n_real, max_videos // 2)
        fake_total = min(fake_total, max_videos - n_real)
    real_subset = real[:n_real]

    if stratify:
        # Even split, remainder to the first types; cap each by availability.
        base, rem = divmod(fake_total, len(FAKE_TYPES))
        want = {t: base + (1 if i < rem else 0) for i, t in enumerate(FAKE_TYPES)}
        fake_subset = []
        for t in FAKE_TYPES:
            take = min(want[t], len(by_type[t]))
            if take < want[t]:
                logger.warning(f"{t}: wanted {want[t]} but only {len(by_type[t])} available")
            fake_subset += by_type[t][:take]
    else:
        pool = [s for t in FAKE_TYPES for s in by_type[t]]
        rng.shuffle(pool)
        fake_subset = pool[:fake_total]

    combined = real_subset + fake_subset
    rng.shuffle(combined)

    got = Counter(s.manipulation_type for s in fake_subset)
    ids = {identity_of(s.video_path) for s in combined}
    logger.info(
        f"Subset: {len(combined)} videos ({len(real_subset)} real / {len(fake_subset)} fake), "
        f"{len(ids)} identities | fake by type: {dict(got)} | stratify={stratify}"
    )
    return combined


def setup_determinism():
    """Make the run reproducible. Two runs of an identical config previously
    differed by ~1.09 AUC points; this removes that drift so between-cell gaps
    can be read directly rather than only via effect size across repeats."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: some vision ops (e.g. bilinear upsample backward) lack deterministic
    # kernels; warn instead of crashing rather than losing the whole run.
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Determinism enabled (cudnn.deterministic, seeded, CUBLAS_WORKSPACE_CONFIG)")


# ─────────────────────── feature extraction ───────────────────────

def _crops_to_tensor(crops, mean, std, size, device):
    """BGR uint8 HWC crops -> normalized RGB float CHW batch.

    The previous runner passed raw 0-255 BGR straight into the encoders; both the
    channel order and the scale were wrong for every HuggingFace vision model.
    """
    out = []
    for c in crops:
        if c is None:
            continue
        if c.shape[:2] != size:
            c = cv2.resize(c, (size[1], size[0]))
        out.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    if not out:
        return None
    arr = np.stack(out).astype(np.float32) / 255.0          # [T,H,W,C]
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)            # [T,C,H,W]
    return t.unsqueeze(0).to(device)                         # [1,T,C,H,W]


def build_detector_cache(detector, samples, device, detect_pad):
    """Preprocess every video once with `detector`, encode with all backbones.

    Returns a dict with per-backbone face/mouth features, shared audio features,
    labels, identities, and detector-native diagnostics.
    """
    logger.info(f"=== [{detector.upper()}] preprocessing {len(samples)} videos ===")
    prep = PreprocessingPipeline(
        config=PreprocessConfig(
            max_frames=MAX_FRAMES,
            face_detection_backend=detector,
            face_detect_pad_ratio=detect_pad,
        )
    )

    encoders, norms = {}, {}
    for name, mid in BACKBONES.items():
        enc = VideoEncoder(model_name=mid, projection_dim=512).to(device).eval()
        enc._load_backbone()
        if not enc.pretrained:
            raise RuntimeError(
                f"{name} ({mid}) fell back to random init - refusing to benchmark it."
            )
        proc = enc._processor
        mean = np.array(getattr(proc, "image_mean", [0.5] * 3), dtype=np.float32)
        std = np.array(getattr(proc, "image_std", [0.5] * 3), dtype=np.float32)
        encoders[name], norms[name] = enc, (mean, std)
        logger.info(f"  {name}: pretrained OK, norm mean={mean.tolist()}")

    audio_enc = AudioEncoder(model_name=model_config.wav2vec2_model_name).to(device).eval()
    audio_enc._load_backbone()

    face = {n: [] for n in BACKBONES}
    mouth = {n: [] for n in BACKBONES}
    audio_feats, labels, idents, manips = [], [], [], []
    det_rates, latencies = [], []
    failures, silent_audio = [], 0

    t_start = time.time()
    for i, sample in enumerate(samples):
        try:
            t0 = time.time()
            pre = prep.process(sample.video_path)
            latencies.append((time.time() - t0) / max(1, pre.num_frames) * 1000.0)

            if not pre.face_crops:
                failures.append((str(sample.video_path), "no face crops"))
                continue

            # The real per-frame detection rate. The pipeline substitutes the full
            # frame when detection fails, so len(face_crops) is always num_frames
            # and cannot be used as a detection metric.
            det_rates.append(float(pre.face_detection_rate))

            wav = pre.audio_waveform
            if wav is None or len(wav) == 0:
                silent_audio += 1
                a_t = torch.zeros(1, 16000, device=device)
            else:
                a_t = torch.as_tensor(np.asarray(wav), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                a_hidden = audio_enc._encoder(a_t).last_hidden_state  # [1,T,768]
                a_vec = a_hidden.mean(dim=1).squeeze(0).cpu().numpy()

                ok = True
                f_vecs, m_vecs = {}, {}
                for name, enc in encoders.items():
                    mean, std = norms[name]
                    f_in = _crops_to_tensor(pre.face_crops, mean, std, (224, 224), device)
                    m_in = _crops_to_tensor(pre.mouth_rois, mean, std, (224, 224), device)
                    if f_in is None or m_in is None:
                        ok = False
                        break
                    # mean over frames -> one 768-d vector per video per stream
                    f_vecs[name] = enc.extract_backbone_features(f_in).mean(dim=1).squeeze(0).cpu().numpy()
                    m_vecs[name] = enc.extract_backbone_features(m_in).mean(dim=1).squeeze(0).cpu().numpy()

            if not ok:
                failures.append((str(sample.video_path), "empty crop tensor"))
                det_rates.pop()
                continue

            for name in BACKBONES:
                face[name].append(f_vecs[name])
                mouth[name].append(m_vecs[name])
            audio_feats.append(a_vec)
            labels.append(sample.label)
            idents.append(identity_of(sample.video_path))
            manips.append(sample.manipulation_type)

        except Exception as e:  # recorded, never silently swallowed
            failures.append((str(sample.video_path), f"{type(e).__name__}: {e}"))

        if (i + 1) % 100 == 0 or (i + 1) == len(samples):
            logger.info(f"  [{detector}] {i+1}/{len(samples)} | kept={len(labels)} | failed={len(failures)}")

    elapsed = time.time() - t_start
    logger.info(
        f"[{detector}] done in {elapsed/60:.1f} min | kept {len(labels)}/{len(samples)} "
        f"| failures {len(failures)} | silent audio {silent_audio}"
    )

    return {
        "detector": detector,
        "face": {n: np.asarray(v, dtype=np.float32) for n, v in face.items()},
        "mouth": {n: np.asarray(v, dtype=np.float32) for n, v in mouth.items()},
        "audio": np.asarray(audio_feats, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "identities": np.asarray(idents),
        "manip": np.asarray(manips),
        "diagnostics": {
            "videos_attempted": len(samples),
            "videos_kept": len(labels),
            "videos_failed": len(failures),
            "silent_audio_videos": silent_audio,
            "mean_face_detection_rate_pct": float(np.mean(det_rates) * 100) if det_rates else 0.0,
            "median_face_detection_rate_pct": float(np.median(det_rates) * 100) if det_rates else 0.0,
            "videos_below_30pct_detection": int(np.sum(np.asarray(det_rates) < 0.3)) if det_rates else 0,
            "mean_latency_ms_per_frame": float(np.mean(latencies)) if latencies else 0.0,
            "total_preprocess_minutes": elapsed / 60.0,
            "failure_examples": failures[:10],
        },
    }


# ─────────────────────────── model ───────────────────────────

class AblationModel(nn.Module):
    """Trainable 768->512 projections feeding the project's fusion head.

    The encoders' own projection heads are randomly initialized and never trained
    under a frozen-backbone protocol, so they are bypassed in favour of these.
    """

    def __init__(self, in_dim=BACKBONE_DIM, cfg=model_config):
        super().__init__()
        d = cfg.fusion_hidden_dim
        self.proj_a = nn.Linear(in_dim, d)
        self.proj_f = nn.Linear(in_dim, d)
        self.proj_m = nn.Linear(in_dim, d)
        self.head = FastFusionClassifier(cfg)

    def forward(self, a, f, m):
        return self.head(self.proj_a(a), self.proj_f(f), self.proj_m(m))


def calculate_eer(y_true, y_scores):
    """EER at the fpr/fnr crossover (the previous version returned fpr alone)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0) * 100.0


def run_fold(tr, te, epochs, device):
    """Train the fusion head on one fold. `tr`/`te` are (audio, face, mouth, y)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Scalers fit on the training fold only.
    scalers = [StandardScaler().fit(tr[i]) for i in range(3)]
    tr_x = [scalers[i].transform(tr[i]) for i in range(3)]
    te_x = [scalers[i].transform(te[i]) for i in range(3)]

    tr_ds = TensorDataset(*[torch.tensor(x, dtype=torch.float32) for x in tr_x],
                          torch.tensor(tr[3], dtype=torch.float32))
    te_ds = TensorDataset(*[torch.tensor(x, dtype=torch.float32) for x in te_x],
                          torch.tensor(te[3], dtype=torch.float32))

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = AblationModel().to(device)
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
        # balanced accuracy is prevalence-robust; matters once real:fake != 1:1
        "bal_acc": balanced_accuracy_score(targets, preds) * 100.0,
        "f1": f1_score(targets, preds, zero_division=0) * 100.0,
    }
    # scores/targets returned separately (not JSON-dumped) for pooled per-category AUC
    return metrics, scores, targets


def evaluate_cell(cache, backbone, folds, epochs, device):
    """Identity-disjoint k-fold evaluation of one (detector, backbone) cell."""
    X_f, X_m, X_a = cache["face"][backbone], cache["mouth"][backbone], cache["audio"]
    y, groups = cache["labels"], cache["identities"]
    manip = cache.get("manip")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    except ImportError:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=folds)
        logger.warning("StratifiedGroupKFold unavailable; folds may be label-imbalanced")

    per_fold = []
    pooled_scores, pooled_targets, pooled_manip = [], [], []
    for k, (tr_i, te_i) in enumerate(splitter.split(X_f, y, groups)):
        overlap = set(groups[tr_i]) & set(groups[te_i])
        assert not overlap, f"identity leak in fold {k}: {sorted(overlap)[:5]}"
        if len(np.unique(y[te_i])) < 2:
            logger.warning(f"fold {k} is single-class; skipped")
            continue
        metrics, sc, tg = run_fold(
            (X_a[tr_i], X_f[tr_i], X_m[tr_i], y[tr_i]),
            (X_a[te_i], X_f[te_i], X_m[te_i], y[te_i]),
            epochs, device,
        )
        per_fold.append(metrics)
        # te_loader has shuffle=False, so sc aligns with te_i order.
        pooled_scores.append(sc)
        pooled_targets.append(tg)
        if manip is not None:
            pooled_manip.append(manip[te_i])

    agg = {}
    for metric in ("auc", "eer", "acc", "bal_acc", "f1"):
        vals = [f[metric] for f in per_fold]
        agg[f"{metric}_mean"] = float(np.mean(vals))
        agg[f"{metric}_std"] = float(np.std(vals))
    agg["folds"] = per_fold
    agg["n_samples"] = int(len(y))
    agg["n_identities"] = int(len(set(groups)))

    # Per-manipulation-type AUC: reals vs fakes of one type, pooled over folds
    # (every video is tested exactly once, so this is one prediction per video).
    # This is what makes the audio-only class interpretable.
    if manip is not None and pooled_scores:
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


# ─────────────────────────── report ───────────────────────────

def print_tables(results, caches):
    def cell(det, bb):
        return results[f"{det}+{bb}"]

    print("\n" + "=" * 104)
    print("TABLE 1 - FACE DETECTOR ARM   (backbone fixed: ViT-16)")
    print("=" * 104)
    print(f"{'Detector':<12} | {'FaceDet%':<9} | {'<30% det':<9} | {'ms/frame':<9} | "
          f"{'AUC%':<14} | {'EER%':<13} | {'Acc%':<14}")
    print("-" * 104)
    for det in DETECTORS:
        d, r = caches[det]["diagnostics"], cell(det, "ViT-16")
        print(f"{det.upper():<12} | {d['mean_face_detection_rate_pct']:7.1f}   | "
              f"{d['videos_below_30pct_detection']:>4}      | {d['mean_latency_ms_per_frame']:7.1f}   | "
              f"{r['auc_mean']:6.2f}+-{r['auc_std']:<5.2f} | {r['eer_mean']:5.2f}+-{r['eer_std']:<5.2f} | "
              f"{r['acc_mean']:6.2f}+-{r['acc_std']:<5.2f}")

    print("\n" + "=" * 104)
    print("TABLE 2 - VISUAL BACKBONE ARM   (detector fixed: RetinaFace)")
    print("=" * 104)
    print(f"{'Backbone':<12} | {'AUC%':<14} | {'EER%':<13} | {'BalAcc%':<14} | {'F1%':<14}")
    print("-" * 104)
    for bb in BACKBONES:
        r = cell("retinaface", bb)
        print(f"{bb:<12} | {r['auc_mean']:6.2f}+-{r['auc_std']:<5.2f} | "
              f"{r['eer_mean']:5.2f}+-{r['eer_std']:<5.2f} | {r['bal_acc_mean']:6.2f}+-{r['bal_acc_std']:<5.2f} | "
              f"{r['f1_mean']:6.2f}+-{r['f1_std']:<5.2f}")

    print("\n" + "=" * 104)
    print("FULL 2x3 GRID (AUC% mean+-std over identity-disjoint folds)")
    print("=" * 104)
    print(f"{'Detector':<12} | " + " | ".join(f"{b:<16}" for b in BACKBONES))
    print("-" * 104)
    for det in DETECTORS:
        row = " | ".join(f"{cell(det,b)['auc_mean']:6.2f}+-{cell(det,b)['auc_std']:<5.2f}  " for b in BACKBONES)
        print(f"{det.upper():<12} | {row}")

    # Per-manipulation-type AUC on the anchor cell. This is the point of the
    # stratified draw: it separates whether the model catches audio-only forgeries
    # (lip-sync, the modality the audio branch exists for) from video forgeries.
    anchor = cell(ANCHOR[0], ANCHOR[1])
    if anchor.get("per_category_auc"):
        print("\n" + "=" * 104)
        print(f"PER-MANIPULATION-TYPE AUC   (anchor cell {ANCHOR[0].upper()}+{ANCHOR[1]}, reals vs one fake type)")
        print("=" * 104)
        print(f"{'Manipulation':<26} | {'AUC%':<9} | {'#fake':<7} | {'#real':<7}")
        print("-" * 104)
        label = {"face-swap": "face-swap (video)", "lip-sync": "lip-sync (AUDIO-only)", "both": "both"}
        for c in FAKE_TYPES:
            e = anchor["per_category_auc"][c]
            auc = f"{e['auc']:6.2f}" if e["auc"] is not None else "  n/a "
            print(f"{label[c]:<26} | {auc}   | {e['n_fake']:<7} | {e['n_real']:<7}")
    print("=" * 104)
    print(f"\nAnchor cell (deployed system): {ANCHOR[0].upper()} + {ANCHOR[1]}")
    print("Protocol: frozen pretrained backbones, trainable fusion head,")
    print("          identity-disjoint stratified group k-fold on FakeAVCeleb id00XXX.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=None, help="cap total videos (debug)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--detect-pad", type=float, default=0.5,
                    help="pre-detection frame padding ratio; applied identically to "
                         "both detectors. 0.5 measured to lift RetinaFace from 54%% "
                         "to 100%% frame detection on FakeAVCeleb's 224x224 crops")
    ap.add_argument("--fake-total", type=int, default=500,
                    help="number of fake videos; real is always the full 500. "
                         "500 keeps a 1:1 balance (167/167/166 across fake types); "
                         "1500 gives 500/500/500 at the cost of 1:3 imbalance.")
    ap.add_argument("--no-stratify", action="store_true",
                    help="draw fakes proportionally from the pool instead of evenly "
                         "across the three types (leaves audio-only at ~2.4%%)")
    ap.add_argument("--out", type=str, default="output/ablation_fakeavceleb_stratified")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    setup_determinism()

    samples = load_samples(
        fake_total=args.fake_total, stratify=not args.no_stratify, max_videos=args.max_videos
    )

    caches = {}
    for det in DETECTORS:
        caches[det] = build_detector_cache(det, samples, device, args.detect_pad)
        np.savez_compressed(
            out_dir / f"features_{det}.npz",
            audio=caches[det]["audio"], labels=caches[det]["labels"],
            identities=caches[det]["identities"], manip=caches[det]["manip"],
            **{f"face_{n}": v for n, v in caches[det]["face"].items()},
            **{f"mouth_{n}": v for n, v in caches[det]["mouth"].items()},
        )
        logger.info(f"cached features -> {out_dir / f'features_{det}.npz'}")

    results = {}
    for det in DETECTORS:
        for bb in BACKBONES:
            logger.info(f"evaluating {det.upper()} + {bb} ({args.folds}-fold)")
            results[f"{det}+{bb}"] = evaluate_cell(caches[det], bb, args.folds, args.epochs, device)

    payload = {
        "config": {
            "dataset": "FakeAVCeleb v1.2",
            "videos": len(samples),
            "max_frames": MAX_FRAMES,
            "folds": args.folds,
            "epochs": args.epochs,
            "seed": SEED,
            "deterministic": True,
            "detect_pad_ratio": args.detect_pad,
            "fake_total": args.fake_total,
            "stratified_fakes": not args.no_stratify,
            "real_fake_ratio": f"500:{args.fake_total}",
            "backbones": BACKBONES,
            "detectors": DETECTORS,
            "anchor_cell": f"{ANCHOR[0]}+{ANCHOR[1]}",
            "protocol": "frozen pretrained backbones; trainable fusion head; "
                        "identity-disjoint StratifiedGroupKFold on FakeAVCeleb id00XXX",
            "notes": [
                "ViT-32 uses google/vit-base-patch32-224-in21k: google/vit-base-patch32-224 "
                "does not exist on the Hub. ViT-16 is in21k->in1k, ViT-32 is in21k-only.",
                "Backbones are frozen, so results measure representation quality, "
                "not fine-tuning behaviour.",
                f"detect_pad_ratio={args.detect_pad} pads the frame before detection. "
                "FakeAVCeleb ships 224x224 pre-cropped faces that fill the frame, "
                "which falls outside RetinaFace's anchor scales: unpadded it detects "
                "~34-54% of frames vs MTCNN's 100%, purely as a configuration "
                "artifact. Padding is applied identically to both detectors.",
                f"Fakes drawn {'evenly across' if not args.no_stratify else 'proportionally from'} "
                "the three sub-categories (face-swap / lip-sync / both). Stratifying lifts "
                "the audio-only (lip-sync) class from ~2.4% to a testable share; see "
                "per_category_auc for the audio-branch result.",
                "All videos span the same 500 identities regardless of count, so more "
                "videos add coverage, not statistical power.",
            ],
        },
        "detector_diagnostics": {d: caches[d]["diagnostics"] for d in DETECTORS},
        "results": results,
    }
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print_tables(results, caches)
    print(f"\nResults written to {out_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
