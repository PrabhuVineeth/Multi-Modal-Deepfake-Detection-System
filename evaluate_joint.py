"""
Joint (all-three-datasets) evaluation, done two honest ways.

1. Macro-average: the mean of the per-dataset metrics -- the correct "overall
   across three benchmarks" number. Read from the per-dataset late-fusion JSON.

2. Pooled joint: ONE classifier trained across all three datasets at BALANCED
   equal contribution (per_class from each, so LAV-DF's larger pool can't
   dominate), evaluated identity-disjoint with dataset-namespaced identities so a
   FakeAVCeleb id and an FF++ id are never conflated. Full-fusion AND late-fusion.

Why this matters: the earlier "Joint 0.952" was not reproducible, and the buggy
balanced_eval joint degenerated (AUC 51.9) from the random-projection bug + LAV-DF
domination. This re-tests on the corrected raw features with balanced pooling. A
pooled cross-dataset classifier is genuinely harder (heterogeneous domains/fakes),
so the honest expectation is at/below the per-dataset numbers -- not above them.

Reuses evaluate_crossdataset_latefusion.evaluate verbatim on the pooled cache, so
the joint gets the identical full-vs-late comparison + temperature calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from config import get_device
from ablation_fakeavceleb import setup_determinism, SEED
from evaluate_crossdataset_latefusion import evaluate

FEATS = {
    "fakeavceleb": "output/crossdataset_eval_scaleup/feat_fakeavceleb.npz",
    "faceforensics": "output/crossdataset_eval_scaleup/feat_faceforensics.npz",
    "lavdf": "output/crossdataset_eval_scaleup/feat_lavdf.npz",
}
KEYS = ("face", "mouth", "audio", "labels", "identities", "manip")


def load_balanced(name, path, per_class):
    """Balanced per-class subsample with dataset-namespaced identities."""
    z = np.load(path, allow_pickle=True)
    y = z["labels"]
    idx_r = np.where(y == 0)[0]
    idx_f = np.where(y == 1)[0]
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx_r)
    rng.shuffle(idx_f)
    n = min(per_class, len(idx_r), len(idx_f))
    take = np.concatenate([idx_r[:n], idx_f[:n]])
    out = {k: z[k][take] for k in ("face", "mouth", "audio", "labels", "manip")}
    out["identities"] = np.array([f"{name}::{g}" for g in z["identities"][take]])
    logger.info(f"[{name}] pooled contribution: {2*n} ({n}/class), "
                f"{len(set(out['identities'].tolist()))} identities")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500, help="per-class per dataset in the pool")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-dataset-json", type=str,
                    default="output/crossdataset_eval_scaleup/latefusion_calibration_results.json")
    ap.add_argument("--out", type=str, default="output/crossdataset_eval_scaleup/joint_results.json")
    args = ap.parse_args()

    setup_determinism()
    device = get_device()

    # ── pooled joint ──
    caches = [load_balanced(n, p, args.per_class) for n, p in FEATS.items()]
    pooled = {k: np.concatenate([c[k] for c in caches]) for k in KEYS}
    r = int((pooled["labels"] == 0).sum())
    f = int((pooled["labels"] == 1).sum())
    logger.info(f"[joint] pooled {len(pooled['labels'])} ({r} real / {f} fake), "
                f"{len(set(pooled['identities'].tolist()))} namespaced identities")
    joint = evaluate("joint", pooled, args.folds, args.epochs, device)

    # ── macro-average across the per-dataset late-fusion results ──
    macro = None
    pj = Path(args.per_dataset_json)
    if pj.exists():
        per = json.load(open(pj))["results"]
        macro = {}
        for cfg in ("Full Fusion", "Late-Fusion (mean)"):
            aucs = [per[d][cfg]["auc"]["mean"] for d in FEATS]
            f1s = [per[d][cfg]["f1"]["mean"] for d in FEATS]
            macro[cfg] = {"auc_macro": float(np.mean(aucs)), "f1_macro": float(np.mean(f1s)),
                          "per_dataset_auc": {d: per[d][cfg]["auc"]["mean"] for d in FEATS}}

    print("\n" + "=" * 96)
    print("JOINT EVALUATION (all three datasets)")
    print("=" * 96)
    if macro:
        print("1) MACRO-AVERAGE across datasets (overall-across-benchmarks):")
        for cfg in ("Full Fusion", "Late-Fusion (mean)"):
            print(f"   {cfg:<20} AUC(macro)={macro[cfg]['auc_macro']:.2f}  F1(macro)={macro[cfg]['f1_macro']:.2f}")
    print("\n2) POOLED JOINT (single classifier, balanced equal contribution, identity-disjoint):")
    print(f"   N={joint['n_samples']}")
    for cfg in ("Full Fusion", "Late-Fusion (mean)"):
        x = joint[cfg]
        print(f"   {cfg:<20} AUC={x['auc']['mean']:5.2f}±{x['auc']['std']:.2f}  "
              f"bal_acc={x['bal_acc']['mean']:.2f}  F1={x['f1']['mean']:.2f}  ECE={x['ece']['mean']:.3f}")
    lvf = joint["late_vs_full"]
    cal = joint["calibration"]
    print(f"   d(late-full)={lvf['delta_auc']:+.2f} p={lvf['p_value']:.3f} "
          f"{'SIG' if lvf['significant'] else 'n.s.'} | calib T={cal['temperature']:.2f} "
          f"ECE {cal['ece_before']:.3f}->{cal['ece_after']:.3f}")
    print("=" * 96)

    json.dump({"pooled_joint": joint, "macro_average": macro,
               "protocol": f"pooled: {args.per_class}/class from each dataset, dataset-namespaced "
                           f"identity-disjoint {args.folds}-fold; macro: mean of per-dataset late-fusion"},
              open(args.out, "w"), indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
