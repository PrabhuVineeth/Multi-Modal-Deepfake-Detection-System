"""
Late-fusion diagnostic for the modality-ablation finding.

The modality ablation showed end-to-end (Full Fusion) scoring WORSE than the
single-modality models on audio-only forgeries (lip-sync: Full 61.5 vs
Audio-only 78.0), i.e. negative transfer. This script runs the decisive test:
does combining the two single-modality *scores* (late fusion) beat the joint
end-to-end model?

  - Late fusion >> Full  =>  the features are fine and the JOINT ARCHITECTURE is
    the defect; late fusion is an immediate fix.
  - Late fusion also fails =>  the problem is deeper than the fusion stage.

Late fusion here is the unweighted mean of the two single-modality sigmoid
scores: zero extra training, no combiner fit, so no possibility of leakage.

Everything reuses the exact models / training loop / folds of
evaluate_modality_ablation.py, so numbers are directly comparable. Unlike that
script this one keeps per-fold and per-fold-per-category AUC arrays, enabling
paired t-tests (the earlier run reported only means/stds).

Usage:
    python evaluate_late_fusion.py [--features NPZ] [--backbone ViT-16] [--folds 5]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import get_device
from evaluate_modality_ablation import (
    FAKE_TYPES,
    SEED,
    AudioOnlyModel,
    VisualOnlyModel,
    FullFusionModel,
    calculate_eer,
    run_fold,
    setup_determinism,
)

CONFIGS = ["Audio-Only", "Visual-Only", "Full Fusion", "Late-Fusion (mean)"]


def cat_auc(scores, targets, manip, cat):
    """AUC of reals vs fakes of one manipulation type."""
    mask = (targets == 0) | (manip == cat)
    yt, ys = targets[mask], scores[mask]
    if len(np.unique(yt)) < 2 or (manip[mask] == cat).sum() == 0:
        return None
    return float(roc_auc_score(yt, ys) * 100.0)


def summarize(fold_scores, fold_targets, fold_manip):
    """Aggregate a config's per-fold test scores into overall + per-category AUC.

    Returns per-fold overall AUC (for paired tests), per-fold per-category AUC
    (for per-category paired tests), and pooled per-category AUC (the headline,
    one prediction per video since folds are disjoint)."""
    per_fold_auc = [roc_auc_score(t, s) * 100.0 for s, t in zip(fold_scores, fold_targets)]
    per_fold_cat = {c: [] for c in FAKE_TYPES}
    for s, t, m in zip(fold_scores, fold_targets, fold_manip):
        for c in FAKE_TYPES:
            a = cat_auc(s, t, m, c)
            if a is not None:
                per_fold_cat[c].append(a)

    ps = np.concatenate(fold_scores)
    pt = np.concatenate(fold_targets)
    pm = np.concatenate(fold_manip)
    pooled_cat = {c: cat_auc(ps, pt, pm, c) for c in FAKE_TYPES}

    return {
        "auc_mean": float(np.mean(per_fold_auc)),
        "auc_std": float(np.std(per_fold_auc)),
        "eer_mean": float(np.mean([calculate_eer(t, s) for s, t in zip(fold_scores, fold_targets)])),
        "per_fold_auc": per_fold_auc,
        "per_fold_category_auc": per_fold_cat,
        "pooled_category_auc": pooled_cat,
    }


def paired(a, b):
    """Paired t-test of two per-fold AUC lists (same folds)."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return {"mean_diff": float(d.mean()), "p_value": float(p),
            "significant": bool(p < 0.05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str,
                    default="output/ablation_fakeavceleb_stratified/features_retinaface.npz")
    ap.add_argument("--backbone", type=str, default="ViT-16")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    setup_determinism()
    device = get_device()
    cache = dict(np.load(args.features, allow_pickle=True))
    X_f = cache[f"face_{args.backbone}"]
    X_m = cache[f"mouth_{args.backbone}"]
    X_a = cache["audio"]
    y, groups, manip = cache["labels"], cache["identities"], cache["manip"]

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    # Collected per config: list (over folds) of test-score arrays, aligned to te order.
    coll = {c: {"s": [], "t": [], "m": []} for c in CONFIGS}

    for k, (tr_i, te_i) in enumerate(splitter.split(X_f, y, groups)):
        assert not (set(groups[tr_i]) & set(groups[te_i])), f"identity leak fold {k}"
        tr = (X_a[tr_i], X_f[tr_i], X_m[tr_i], y[tr_i])
        te = (X_a[te_i], X_f[te_i], X_m[te_i], y[te_i])

        _, p_a, tg = run_fold(AudioOnlyModel, tr, te, args.epochs, device)
        _, p_v, _ = run_fold(VisualOnlyModel, tr, te, args.epochs, device)
        _, p_full, _ = run_fold(FullFusionModel, tr, te, args.epochs, device)
        p_late = (p_a + p_v) / 2.0  # unweighted mean; no combiner training -> no leakage

        mt = manip[te_i]
        for cfg, sc in zip(CONFIGS, [p_a, p_v, p_full, p_late]):
            coll[cfg]["s"].append(sc)
            coll[cfg]["t"].append(tg)
            coll[cfg]["m"].append(mt)
        logger.info(f"fold {k}: AUC audio={roc_auc_score(tg,p_a)*100:.1f} "
                    f"visual={roc_auc_score(tg,p_v)*100:.1f} full={roc_auc_score(tg,p_full)*100:.1f} "
                    f"late={roc_auc_score(tg,p_late)*100:.1f}")

    results = {c: summarize(coll[c]["s"], coll[c]["t"], coll[c]["m"]) for c in CONFIGS}

    # ── tables ──
    print("\n" + "=" * 92)
    print(f"LATE-FUSION DIAGNOSTIC ({args.backbone}, {args.folds}-fold identity-disjoint, deterministic)")
    print("=" * 92)
    print(f"{'Configuration':<20} | {'Overall AUC%':<16} | {'face-swap':<10} | {'lip-sync(AUD)':<14} | {'both':<8}")
    print("-" * 92)
    for c in CONFIGS:
        r = results[c]
        pc = r["pooled_category_auc"]
        fmt = lambda v: f"{v:6.2f}" if v is not None else "  n/a "
        print(f"{c:<20} | {r['auc_mean']:6.2f}+-{r['auc_std']:<7.2f} | {fmt(pc['face-swap'])}   | "
              f"{fmt(pc['lip-sync'])}        | {fmt(pc['both'])}")
    print("=" * 92)

    # ── the decisive paired tests ──
    print("\nPAIRED SIGNIFICANCE (same folds):")
    tests = {
        "Late vs Full  (overall)": paired(results["Late-Fusion (mean)"]["per_fold_auc"],
                                          results["Full Fusion"]["per_fold_auc"]),
        "Late vs Full  (lip-sync)": paired(results["Late-Fusion (mean)"]["per_fold_category_auc"]["lip-sync"],
                                           results["Full Fusion"]["per_fold_category_auc"]["lip-sync"]),
        "Audio vs Full (lip-sync)": paired(results["Audio-Only"]["per_fold_category_auc"]["lip-sync"],
                                           results["Full Fusion"]["per_fold_category_auc"]["lip-sync"]),
        "Visual vs Full(overall)": paired(results["Visual-Only"]["per_fold_auc"],
                                          results["Full Fusion"]["per_fold_auc"]),
    }
    for name, t in tests.items():
        print(f"  {name:<26} diff={t['mean_diff']:+6.2f}  p={t['p_value']:.4f}  "
              f"{'SIGNIFICANT' if t['significant'] else 'n.s.'}")

    out = Path(args.features).parent / "late_fusion_results.json"
    with open(out, "w") as f:
        json.dump({"config": {"backbone": args.backbone, "folds": args.folds,
                              "epochs": args.epochs, "seed": SEED,
                              "late_fusion": "unweighted mean of audio/visual sigmoid scores"},
                   "results": results, "paired_tests": tests}, f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
