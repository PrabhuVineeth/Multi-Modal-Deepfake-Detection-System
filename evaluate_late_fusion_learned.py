"""
Learned late fusion (the optimal-combiner follow-up to evaluate_late_fusion.py).

Mean late fusion already beat end-to-end fusion, but averaging in chance-level
audio drags face-swap down (84.9 vs Visual-Only's 89.2). A learned combiner
should down-weight audio where it is uninformative and keep the lip-sync gain.

The combiner is a logistic regression on [p_audio, p_visual]. To avoid leakage
its weights are fit ONLY on out-of-fold predictions generated inside each outer
training fold via an identity-disjoint inner CV — the outer test fold never
touches the combiner fit:

  outer fold:
    inner StratifiedGroupKFold on the training identities
      -> OOF p_audio, p_visual for every training video
    fit LogisticRegression(OOF scores -> label)
    retrain audio & visual on the full training fold, predict the outer test
    combiner(test scores) = learned late-fusion prediction

Compares against the mean/full/audio/visual per-fold arrays already saved by
evaluate_late_fusion.py, on the identical folds (same seeded splitter).

Usage:
    python evaluate_late_fusion_learned.py [--features NPZ] [--backbone ViT-16]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import get_device
from evaluate_late_fusion import cat_auc, paired, summarize
from evaluate_modality_ablation import (
    FAKE_TYPES,
    SEED,
    AudioOnlyModel,
    VisualOnlyModel,
    run_fold,
    setup_determinism,
)


def modality_scores(model_class, tr, te, epochs, device):
    """Test-fold sigmoid scores from a single-modality model (order == te order)."""
    _, scores, _ = run_fold(model_class, tr, te, epochs, device)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str,
                    default="output/ablation_fakeavceleb_stratified/features_retinaface.npz")
    ap.add_argument("--backbone", type=str, default="ViT-16")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    setup_determinism()
    device = get_device()
    cache = dict(np.load(args.features, allow_pickle=True))
    Xf, Xm, Xa = cache[f"face_{args.backbone}"], cache[f"mouth_{args.backbone}"], cache["audio"]
    y, groups, manip = cache["labels"], cache["identities"], cache["manip"]

    def bundle(idx):
        return (Xa[idx], Xf[idx], Xm[idx], y[idx])

    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_scores, fold_targets, fold_manip = [], [], []

    for k, (tr_i, te_i) in enumerate(outer.split(Xf, y, groups)):
        assert not (set(groups[tr_i]) & set(groups[te_i])), f"outer identity leak fold {k}"

        # ── inner OOF predictions on the training fold (combiner training signal) ──
        g_tr = groups[tr_i]
        oof_a = np.zeros(len(tr_i))
        oof_v = np.zeros(len(tr_i))
        inner = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True, random_state=SEED)
        for itr, iva in inner.split(Xf[tr_i], y[tr_i], g_tr):
            a_idx, v_idx = tr_i[itr], tr_i[iva]
            assert not (set(groups[a_idx]) & set(groups[v_idx])), "inner identity leak"
            oof_a[iva] = modality_scores(AudioOnlyModel, bundle(a_idx), bundle(v_idx), args.epochs, device)
            oof_v[iva] = modality_scores(VisualOnlyModel, bundle(a_idx), bundle(v_idx), args.epochs, device)

        combiner = LogisticRegression(max_iter=1000)
        combiner.fit(np.column_stack([oof_a, oof_v]), y[tr_i])

        # ── retrain on full training fold, apply combiner to the outer test fold ──
        p_a = modality_scores(AudioOnlyModel, bundle(tr_i), bundle(te_i), args.epochs, device)
        p_v = modality_scores(VisualOnlyModel, bundle(tr_i), bundle(te_i), args.epochs, device)
        learned = combiner.predict_proba(np.column_stack([p_a, p_v]))[:, 1]

        fold_scores.append(learned)
        fold_targets.append(y[te_i])
        fold_manip.append(manip[te_i])
        logger.info(f"fold {k}: learned AUC={roc_auc_score(y[te_i], learned)*100:.1f} "
                    f"(combiner w_audio={combiner.coef_[0][0]:+.2f} w_visual={combiner.coef_[0][1]:+.2f})")

    learned_res = summarize(fold_scores, fold_targets, fold_manip)

    # ── load the earlier configs for a shared table + paired tests (same folds) ──
    prev_path = Path(args.features).parent / "late_fusion_results.json"
    prev = json.load(open(prev_path))["results"] if prev_path.exists() else {}

    print("\n" + "=" * 92)
    print(f"LEARNED LATE FUSION ({args.backbone}, {args.folds}-fold, nested inner CV, deterministic)")
    print("=" * 92)
    print(f"{'Configuration':<22} | {'Overall AUC%':<16} | {'face-swap':<10} | {'lip-sync(AUD)':<14} | {'both':<8}")
    print("-" * 92)
    fmt = lambda v: f"{v:6.2f}" if v is not None else "  n/a "
    order = ["Audio-Only", "Visual-Only", "Full Fusion", "Late-Fusion (mean)"]
    for c in order:
        if c in prev:
            r = prev[c]; pc = r["pooled_category_auc"]
            print(f"{c:<22} | {r['auc_mean']:6.2f}+-{r['auc_std']:<7.2f} | {fmt(pc['face-swap'])}   | "
                  f"{fmt(pc['lip-sync'])}        | {fmt(pc['both'])}")
    pc = learned_res["pooled_category_auc"]
    print(f"{'Late-Fusion (learned)':<22} | {learned_res['auc_mean']:6.2f}+-{learned_res['auc_std']:<7.2f} | "
          f"{fmt(pc['face-swap'])}   | {fmt(pc['lip-sync'])}        | {fmt(pc['both'])}")
    print("=" * 92)

    if prev:
        print("\nPAIRED SIGNIFICANCE vs learned late fusion (same folds):")
        cmps = {
            "Learned vs Full   (overall)": ("Full Fusion", "per_fold_auc", None),
            "Learned vs Full   (lip-sync)": ("Full Fusion", "per_fold_category_auc", "lip-sync"),
            "Learned vs Mean   (overall)": ("Late-Fusion (mean)", "per_fold_auc", None),
            "Learned vs Mean   (face-swap)": ("Late-Fusion (mean)", "per_fold_category_auc", "face-swap"),
            "Learned vs Visual (face-swap)": ("Visual-Only", "per_fold_category_auc", "face-swap"),
        }
        for name, (cfg, key, cat) in cmps.items():
            lhs = learned_res[key][cat] if cat else learned_res[key]
            rhs = prev[cfg][key][cat] if cat else prev[cfg][key]
            t = paired(lhs, rhs)
            print(f"  {name:<30} diff={t['mean_diff']:+6.2f}  p={t['p_value']:.4f}  "
                  f"{'SIGNIFICANT' if t['significant'] else 'n.s.'}")

    out = Path(args.features).parent / "late_fusion_learned_results.json"
    json.dump({"config": {"backbone": args.backbone, "folds": args.folds,
                          "inner_folds": args.inner_folds, "seed": SEED,
                          "combiner": "LogisticRegression on [p_audio,p_visual], nested inner-CV OOF fit"},
               "learned": learned_res}, open(out, "w"), indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
