"""
Late fusion + temperature calibration on the corrected cross-dataset features.

Consumes output/crossdataset_eval/feat_{dataset}.npz (raw 768-d backbone
features produced by evaluate_crossdataset.py) and, on the SAME identity-disjoint
folds, trains three heads per fold:
  * Audio-Only, Visual-Only  -> their mean is the late-fusion score
  * Full Fusion (end-to-end) -> the baseline + the model we calibrate

Reports, per dataset:
  * Full Fusion vs Late-Fusion(mean): AUC / EER / bal_acc / F1 / ECE
  * paired t-test (per-fold AUC) late vs full  [the committed FakeAVCeleb ablation
    found late fusion +5.8 AUC, p=0.04 -- this checks whether it generalizes]
  * temperature scaling of the Full-Fusion out-of-fold scores: ECE before -> after

No re-extraction: reuses the cached features, so this runs in minutes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import get_device
from ablation_fakeavceleb import setup_determinism, SEED
from evaluate_modality_ablation import (
    AudioOnlyModel,
    VisualOnlyModel,
    FullFusionModel,
    run_fold,
    calculate_eer,
)

DATASETS = ["fakeavceleb", "faceforensics", "lavdf"]


def ece(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s > lo) & (s <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / len(s)) * abs(s[m].mean() - (y[m] == (s[m] >= 0.5)).mean())
    return float(e)


def metrics_of(y, s):
    p = (s >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, s) * 100.0,
        "eer": calculate_eer(y, s),
        "bal_acc": balanced_accuracy_score(y, p) * 100.0,
        "f1": f1_score(y, p, zero_division=0) * 100.0,
        "ece": ece(y, s),
    }


def temperature_scale(scores, labels):
    """Fit T>0 minimizing BCE on logit(scores)/T; return T and calibrated probs."""
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = torch.tensor(np.log(s / (1 - s)), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / T.abs().clamp(min=1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    Tv = float(T.detach().abs().clamp(min=1e-3))
    cal = 1.0 / (1.0 + np.exp(-(np.log(s / (1 - s)) / Tv)))
    return Tv, cal


def evaluate(name, cache, folds, epochs, device):
    Xf, Xm, Xa = cache["face"], cache["mouth"], cache["audio"]
    y, g = cache["labels"], cache["identities"]
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)

    configs = {"Audio-Only": AudioOnlyModel, "Visual-Only": VisualOnlyModel, "Full Fusion": FullFusionModel}
    per_fold = {c: [] for c in configs}
    per_fold["Late-Fusion (mean)"] = []
    fold_auc = {**{c: [] for c in configs}, "Late-Fusion (mean)": []}

    # pooled out-of-fold full-fusion scores for calibration
    oof_full = np.full(len(y), np.nan)
    oof_y = y.copy()

    for k, (tr, te) in enumerate(skf.split(Xf, y, g)):
        assert not (set(g[tr]) & set(g[te])), f"[{name}] identity leak fold {k}"
        scores = {}
        tgt = None
        for c, mc in configs.items():
            _, s, t = run_fold(mc, (Xa[tr], Xf[tr], Xm[tr], y[tr]),
                               (Xa[te], Xf[te], Xm[te], y[te]), epochs, device)
            scores[c] = s
            tgt = t
        late = (scores["Audio-Only"] + scores["Visual-Only"]) / 2.0
        scores["Late-Fusion (mean)"] = late
        oof_full[te] = scores["Full Fusion"]

        for c in list(configs) + ["Late-Fusion (mean)"]:
            m = metrics_of(tgt, scores[c])
            per_fold[c].append(m)
            fold_auc[c].append(m["auc"])
        logger.info(f"[{name}] fold {k}: full={fold_auc['Full Fusion'][-1]:.1f} "
                    f"late={fold_auc['Late-Fusion (mean)'][-1]:.1f}")

    agg = {}
    for c in list(configs) + ["Late-Fusion (mean)"]:
        agg[c] = {mk: {"mean": float(np.mean([f[mk] for f in per_fold[c]])),
                       "std": float(np.std([f[mk] for f in per_fold[c]]))}
                  for mk in ("auc", "eer", "bal_acc", "f1", "ece")}

    # paired significance: late vs full (per-fold AUC, same folds)
    t = stats.ttest_rel(fold_auc["Late-Fusion (mean)"], fold_auc["Full Fusion"])
    agg["late_vs_full"] = {
        "delta_auc": float(np.mean(fold_auc["Late-Fusion (mean)"]) - np.mean(fold_auc["Full Fusion"])),
        "p_value": float(t.pvalue),
        "significant": bool(t.pvalue < 0.05),
    }

    # calibration of full-fusion OOF scores
    mask = ~np.isnan(oof_full)
    T, cal = temperature_scale(oof_full[mask], oof_y[mask])
    agg["calibration"] = {
        "temperature": T,
        "ece_before": ece(oof_y[mask], oof_full[mask]),
        "ece_after": ece(oof_y[mask], cal),
        "note": "T fit on pooled out-of-fold full-fusion scores; ECE reported on same (post-hoc estimate).",
    }
    agg["n_samples"] = int(len(y))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=str, default="output/crossdataset_eval")
    ap.add_argument("--datasets", type=str, default="fakeavceleb,faceforensics,lavdf")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    setup_determinism()
    device = get_device()
    feat_dir = Path(args.feat_dir)
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    results = {}
    for name in names:
        npz = feat_dir / f"feat_{name}.npz"
        if not npz.exists():
            logger.warning(f"[{name}] {npz} not found; skipping")
            continue
        cache = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
        logger.info(f"=== late fusion + calibration: {name} ===")
        results[name] = evaluate(name, cache, args.folds, args.epochs, device)

    # Save JSON BEFORE printing so a console-encoding hiccup can never lose results.
    out = feat_dir / "latefusion_calibration_results.json"
    json.dump({"protocol": "late fusion (mean of audio-only + visual-only), same identity-disjoint "
                           "folds as evaluate_crossdataset; temperature scaling on OOF full-fusion scores",
               "results": results}, open(out, "w"), indent=2)
    print(f"Saved results -> {out}")

    print("\n" + "=" * 108)
    print("LATE FUSION vs FULL FUSION + CALIBRATION (corrected raw features, identity-disjoint)")
    print("=" * 108)
    print(f"{'Dataset':<14} | {'Full AUC%':<14} | {'Late AUC%':<14} | {'dAUC':<7} | {'p':<7} | "
          f"{'Late F1%':<13} | {'ECE bef->aft':<12}")
    print("-" * 108)
    for name in names:
        if name not in results:
            continue
        r = results[name]
        f_auc = r["Full Fusion"]["auc"]
        l_auc = r["Late-Fusion (mean)"]["auc"]
        lvf = r["late_vs_full"]
        cal = r["calibration"]
        print(f"{name:<14} | {f_auc['mean']:5.2f}±{f_auc['std']:<6.2f} | "
              f"{l_auc['mean']:5.2f}±{l_auc['std']:<6.2f} | {lvf['delta_auc']:+5.2f}  | "
              f"{lvf['p_value']:.3f}  | {r['Late-Fusion (mean)']['f1']['mean']:5.2f}±{r['Late-Fusion (mean)']['f1']['std']:<5.2f} | "
              f"{cal['ece_before']:.3f}→{cal['ece_after']:.3f}")
    print("=" * 108)

    print("\nDone.")


if __name__ == "__main__":
    main()
