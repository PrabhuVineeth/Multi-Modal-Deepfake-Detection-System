"""
Corrected cross-dataset evaluation (FakeAVCeleb, FaceForensics++, LAV-DF).

Why this replaces evaluate_balanced_all.py
-------------------------------------------
evaluate_balanced_all.py fed features through the encoders' *randomly
initialized, never-trained* 768->512 projection heads (VideoEncoder.projection).
That random nonlinear map scrambled the backbone representation, so its numbers
(FakeAVCeleb 74.9, FF++ 56.0, LAV-DF 68.5, joint degenerate) are an INSTRUMENT
ARTIFACT, not the model's real performance. The committed FakeAVCeleb ablation
explicitly bypasses those heads (ablation_fakeavceleb.py:329) and uses the RAW
768-d backbone features via extract_backbone_features -> 93.97% AUC.

This script applies that exact, proven method uniformly to all three datasets:
  * raw 768-d backbone features (ViT-16 face+mouth, wav2vec2 audio), mean-pooled
  * AblationModel: trainable 768->512 projections + FastFusionClassifier head
  * identity-disjoint 5-fold StratifiedGroupKFold (no source-video leakage)
    - FakeAVCeleb : id00XXX from path
    - FF++        : leading source-sequence id (000 from 000_003.mp4)
    - LAV-DF      : the `original` (source real video) field in metadata.min.json
  * deterministic (seeded, CUBLAS workspace pinned)

Running FakeAVCeleb here too is a self-consistency check: if the fix is right it
should land far above 74.9 and near the committed ablation, validating the method
before we trust the FF++/LAV-DF rows.

Outputs one internally-consistent table to output/crossdataset_eval/.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold

from config import PreprocessConfig, get_device, model_config
from datasets.fakeavceleb import FakeAVCelebDataset
from datasets.faceforensics import FaceForensicsDataset
from datasets.lavdf import LAVDFDataset
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder
from preprocessing.pipeline import PreprocessingPipeline

# Reuse the committed ablation's proven, bug-free components verbatim.
from ablation_fakeavceleb import (
    SEED,
    MAX_FRAMES,
    BACKBONE_DIM,
    setup_determinism,
    _crops_to_tensor,
    AblationModel,
    run_fold,
    identity_of as identity_of_fav,
)

VIT = "google/vit-base-patch16-224"
DATASETS = {
    "fakeavceleb": ("G:/NNM24AD071/FakeAVCeleb_v1.2", FakeAVCelebDataset),
    "faceforensics": ("G:/NNM24AD071/FaceForensics++_C23", FaceForensicsDataset),
    "lavdf": ("G:/NNM24AD071/LAV-DF", LAVDFDataset),
}


# ─────────────────────────── identity ───────────────────────────

def _ident_ff(path, root):
    """FF++ groups on the source-sequence id: 000.mp4 and 000_003.mp4 -> '000'."""
    return Path(str(path)).stem.split("_")[0]


_LAVDF_META = None


def _lavdf_meta(root):
    """Map normalized 'split/stem' -> source-video group id (original if fake)."""
    global _LAVDF_META
    if _LAVDF_META is None:
        norm = lambda x: str(x).replace("\\", "/").rsplit(".", 1)[0]
        with open(Path(root) / "metadata.min.json", encoding="utf-8") as f:
            md = json.load(f)
        _LAVDF_META = {}
        for e in md:
            key = norm(e["file"])
            _LAVDF_META[key] = norm(e["original"]) if e.get("original") else key
        logger.info(f"[lavdf] metadata loaded: {len(_LAVDF_META)} entries")
    return _LAVDF_META


def _ident_lavdf(path, root):
    meta = _lavdf_meta(root)
    p = str(path).replace("\\", "/")
    rel = "/".join(p.rsplit("/", 2)[-2:]).rsplit(".", 1)[0]  # e.g. test/000001
    return meta.get(rel, Path(p).stem)


IDENT = {
    "fakeavceleb": lambda p, root: identity_of_fav(p),
    "faceforensics": _ident_ff,
    "lavdf": _ident_lavdf,
}


# ─────────────────────────── metrics ───────────────────────────

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


# ─────────────────────────── sampling ───────────────────────────

def balanced_samples(name, cls, root, per_class):
    ds = cls(root, split="all")
    real = [s for s in ds.samples if s.label == 0]
    fake = [s for s in ds.samples if s.label == 1]
    rng = random.Random(SEED)
    rng.shuffle(real)
    rng.shuffle(fake)
    n = min(per_class, len(real), len(fake))
    sub = real[:n] + fake[:n]
    rng.shuffle(sub)
    logger.info(f"[{name}] balanced sample: {len(sub)} videos ({n} real / {n} fake)")
    return sub, n


# ─────────────────────── feature extraction ───────────────────────

@torch.no_grad()
def extract(name, samples, root, device):
    """RAW 768-d backbone features (the fix): bypass the random projection heads."""
    prep = PreprocessingPipeline(
        config=PreprocessConfig(max_frames=MAX_FRAMES, face_detection_backend="retinaface")
    )
    venc = VideoEncoder(model_name=VIT, projection_dim=512).to(device).eval()
    venc._load_backbone()
    if not venc.pretrained:
        raise RuntimeError(f"{VIT} fell back to random init - refusing to benchmark.")
    aenc = AudioEncoder(model_name=model_config.wav2vec2_model_name).to(device).eval()
    aenc._load_backbone()

    proc = venc._processor
    mean = np.array(getattr(proc, "image_mean", [0.5] * 3), dtype=np.float32)
    std = np.array(getattr(proc, "image_std", [0.5] * 3), dtype=np.float32)

    F, M, A, Y, G, MAN = [], [], [], [], [], []
    fails = 0
    for i, s in enumerate(samples):
        try:
            pre = prep.process(s.video_path)
            if not pre.face_crops:
                fails += 1
                continue
            wav = pre.audio_waveform
            if wav is None or len(wav) == 0:
                a_t = torch.zeros(1, 16000, device=device)
            else:
                a_t = torch.as_tensor(np.asarray(wav), dtype=torch.float32, device=device).unsqueeze(0)
            a_vec = aenc._encoder(a_t).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

            f_in = _crops_to_tensor(pre.face_crops, mean, std, (224, 224), device)
            m_in = _crops_to_tensor(pre.mouth_rois, mean, std, (224, 224), device)
            if f_in is None or m_in is None:
                fails += 1
                continue
            f_vec = venc.extract_backbone_features(f_in).mean(dim=1).squeeze(0).cpu().numpy()
            m_vec = venc.extract_backbone_features(m_in).mean(dim=1).squeeze(0).cpu().numpy()

            F.append(f_vec)
            M.append(m_vec)
            A.append(a_vec)
            Y.append(int(s.label))
            G.append(str(IDENT[name](s.video_path, root)))
            MAN.append(getattr(s, "manipulation_type", ""))
        except Exception as e:
            fails += 1
            logger.warning(f"[{name}] fail {s.video_path}: {type(e).__name__}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            logger.info(f"[{name}] {i+1}/{len(samples)} | kept={len(Y)} | fail={fails}")

    out = {
        "face": np.asarray(F, np.float32), "mouth": np.asarray(M, np.float32),
        "audio": np.asarray(A, np.float32), "labels": np.asarray(Y, np.int64),
        "identities": np.asarray(G), "manip": np.asarray(MAN),
    }
    r = int((out["labels"] == 0).sum()); fk = int((out["labels"] == 1).sum())
    logger.info(f"[{name}] extracted {len(out['labels'])} ({r} real / {fk} fake), "
                f"{len(set(out['identities'].tolist()))} identities, {fails} failed")
    return out


# ─────────────────────────── evaluation ───────────────────────────

def evaluate(name, cache, folds, epochs, device):
    Xf, Xm, Xa = cache["face"], cache["mouth"], cache["audio"]
    y, g = cache["labels"], cache["identities"]
    if len(np.unique(y)) < 2:
        logger.warning(f"[{name}] single class; skipping")
        return None

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    per = []
    for k, (tr, te) in enumerate(skf.split(Xf, y, g)):
        assert not (set(g[tr]) & set(g[te])), f"[{name}] identity leak fold {k}"
        metrics, scores, targets = run_fold(
            (Xa[tr], Xf[tr], Xm[tr], y[tr]),
            (Xa[te], Xf[te], Xm[te], y[te]),
            epochs, device,
        )
        metrics["ece"] = expected_calibration_error(targets, scores)
        per.append(metrics)
        logger.info(f"[{name}] fold {k}: AUC={metrics['auc']:.1f} "
                    f"bal_acc={metrics['bal_acc']:.1f} f1={metrics['f1']:.1f}")

    agg = {}
    for mk in ("auc", "eer", "acc", "bal_acc", "f1", "ece"):
        v = [p[mk] for p in per]
        agg[mk] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    agg["n_samples"] = int(len(y))
    agg["n_identities"] = int(len(set(g.tolist())))
    agg["fake_by_type"] = dict(Counter(cache["manip"][y == 1].tolist()))
    agg["folds"] = per
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--datasets", type=str, default="fakeavceleb,faceforensics,lavdf")
    ap.add_argument("--out", type=str, default="output/crossdataset_eval")
    args = ap.parse_args()

    setup_determinism()
    device = get_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    caches = {}
    for name in names:
        root, cls = DATASETS[name]
        npz = out_dir / f"feat_{name}.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            caches[name] = {k: z[k] for k in z.files}
            logger.info(f"[{name}] loaded cached features ({len(caches[name]['labels'])})")
        else:
            samples, _ = balanced_samples(name, cls, root, args.per_class)
            caches[name] = extract(name, samples, root, device)
            np.savez_compressed(npz, **caches[name])
            logger.info(f"[{name}] cached -> {npz}")

    results = {}
    for name in names:
        logger.info(f"=== evaluating {name} ({args.folds}-fold identity-disjoint) ===")
        results[name] = evaluate(name, caches[name], args.folds, args.epochs, device)

    print("\n" + "=" * 104)
    print(f"CORRECTED CROSS-DATASET EVAL (raw backbone features, identity-disjoint {args.folds}-fold, deterministic)")
    print("=" * 104)
    print(f"{'Dataset':<14} | {'N':<5} | {'#ID':<5} | {'AUC%':<13} | {'EER%':<12} | {'Bal.Acc%':<13} | {'F1%':<13} | {'ECE':<6}")
    print("-" * 104)
    for name in names:
        r = results[name]
        if r is None:
            continue
        print(f"{name:<14} | {r['n_samples']:<5} | {r['n_identities']:<5} | "
              f"{r['auc']['mean']:5.2f}+-{r['auc']['std']:<5.2f} | "
              f"{r['eer']['mean']:5.2f}+-{r['eer']['std']:<4.2f} | "
              f"{r['bal_acc']['mean']:5.2f}+-{r['bal_acc']['std']:<5.2f} | "
              f"{r['f1']['mean']:5.2f}+-{r['f1']['std']:<5.2f} | {r['ece']['mean']:.3f}")
    print("=" * 104)

    payload = {
        "protocol": f"balanced {args.per_class}+{args.per_class} per dataset; raw 768-d backbone "
                    f"features (ViT-16 face+mouth, wav2vec2 audio); AblationModel "
                    f"(trainable 768->512 proj + FastFusionClassifier); {args.folds}-fold "
                    f"identity-disjoint StratifiedGroupKFold; FocalLoss; deterministic",
        "fixes": "Bypasses the random untrained projection heads that made "
                 "evaluate_balanced_all.py's numbers an artifact (see module docstring).",
        "caveats": [
            "FF++ audio is near-silent; its audio stream carries little signal (fusion ~ visual).",
            "LAV-DF grouping is source-video-level (metadata has no speaker id), not speaker-disjoint.",
            "FakeAVCeleb row here uses a random (non-stratified) fake draw; the committed ablation "
            "reports stratified per-category AUC separately.",
        ],
        "results": results,
    }
    with open(out_dir / "crossdataset_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten to {out_dir / 'crossdataset_results.json'}")


if __name__ == "__main__":
    main()
