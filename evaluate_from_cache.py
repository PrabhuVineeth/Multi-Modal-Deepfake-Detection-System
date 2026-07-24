"""
Re-run the ablation's k-fold evaluation from cached features, without redoing
face detection or backbone encoding.

`ablation_fakeavceleb.py` writes `features_<detector>.npz` as soon as that
detector's extraction pass finishes. This script consumes whatever caches exist,
so a run that is cut short still yields every table its features support:

  - features_retinaface.npz alone  -> Table 2 (backbone arm) is complete
  - both caches                    -> Table 1, Table 2 and the full 2x3 grid

Extraction is ~95% of the wall time, so re-scoring here takes a few minutes and
can be repeated with different fold counts or epochs at no extraction cost.

Usage:
    python evaluate_from_cache.py [--folds 5] [--epochs 30] [--dir DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from ablation_fakeavceleb import (
    ANCHOR,
    BACKBONES,
    DETECTORS,
    SEED,
    evaluate_cell,
)
from config import get_device


def load_cache(path: Path, detector: str):
    """Rebuild the in-memory cache structure evaluate_cell() expects."""
    z = np.load(path, allow_pickle=True)
    return {
        "detector": detector,
        "face": {n: z[f"face_{n}"] for n in BACKBONES},
        "mouth": {n: z[f"mouth_{n}"] for n in BACKBONES},
        "audio": z["audio"],
        "labels": z["labels"],
        "identities": z["identities"],
        "diagnostics": None,  # only written to the final JSON by the main runner
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="output/ablation_fakeavceleb")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON name (default: <dir>/ablation_results_from_cache.json)")
    args = ap.parse_args()

    cache_dir = Path(args.dir)
    device = get_device()

    caches = {}
    for det in DETECTORS:
        p = cache_dir / f"features_{det}.npz"
        if p.exists():
            caches[det] = load_cache(p, det)
            c = caches[det]
            logger.info(
                f"loaded {p.name}: {len(c['labels'])} videos, "
                f"{len(set(c['identities']))} identities, "
                f"{int((c['labels']==0).sum())} real / {int((c['labels']==1).sum())} fake"
            )
        else:
            logger.warning(f"missing {p.name} - cells for {det.upper()} will be skipped")

    if not caches:
        raise SystemExit(f"no feature caches found in {cache_dir}")

    results = {}
    for det, cache in caches.items():
        for bb in BACKBONES:
            logger.info(f"evaluating {det.upper()} + {bb} ({args.folds}-fold)")
            results[f"{det}+{bb}"] = evaluate_cell(cache, bb, args.folds, args.epochs, device)

    # ── tables ──
    def cell(det, bb):
        return results.get(f"{det}+{bb}")

    if "retinaface" in caches:
        print("\n" + "=" * 92)
        print("TABLE 2 - VISUAL BACKBONE ARM   (detector fixed: RetinaFace)")
        print("=" * 92)
        print(f"{'Backbone':<12} | {'AUC%':<14} | {'EER%':<14} | {'Acc%':<14} | {'F1%':<14}")
        print("-" * 92)
        for bb in BACKBONES:
            r = cell("retinaface", bb)
            print(f"{bb:<12} | {r['auc_mean']:6.2f}+-{r['auc_std']:<5.2f} | "
                  f"{r['eer_mean']:6.2f}+-{r['eer_std']:<5.2f} | "
                  f"{r['acc_mean']:6.2f}+-{r['acc_std']:<5.2f} | "
                  f"{r['f1_mean']:6.2f}+-{r['f1_std']:<5.2f}")

    if len(caches) == len(DETECTORS):
        print("\n" + "=" * 92)
        print("TABLE 1 - FACE DETECTOR ARM   (backbone fixed: ViT-16)")
        print("  detector-native metrics (detection rate, latency) live in the main runner's JSON")
        print("=" * 92)
        print(f"{'Detector':<12} | {'AUC%':<14} | {'EER%':<14} | {'Acc%':<14} | {'F1%':<14}")
        print("-" * 92)
        for det in DETECTORS:
            r = cell(det, "ViT-16")
            print(f"{det.upper():<12} | {r['auc_mean']:6.2f}+-{r['auc_std']:<5.2f} | "
                  f"{r['eer_mean']:6.2f}+-{r['eer_std']:<5.2f} | "
                  f"{r['acc_mean']:6.2f}+-{r['acc_std']:<5.2f} | "
                  f"{r['f1_mean']:6.2f}+-{r['f1_std']:<5.2f}")

    print("\n" + "=" * 92)
    print("GRID (AUC% mean+-std, identity-disjoint folds)")
    print("=" * 92)
    print(f"{'Detector':<12} | " + " | ".join(f"{b:<16}" for b in BACKBONES))
    print("-" * 92)
    for det in caches:
        row = " | ".join(f"{cell(det,b)['auc_mean']:6.2f}+-{cell(det,b)['auc_std']:<5.2f}  " for b in BACKBONES)
        print(f"{det.upper():<12} | {row}")
    print("=" * 92)

    missing = [d for d in DETECTORS if d not in caches]
    if missing:
        print(f"\nINCOMPLETE: no feature cache for {', '.join(m.upper() for m in missing)}. "
              f"Tables above cover only {', '.join(d.upper() for d in caches)}.")
    print(f"Anchor cell (deployed system): {ANCHOR[0].upper()} + {ANCHOR[1]}")

    out = Path(args.out) if args.out else cache_dir / "ablation_results_from_cache.json"
    with open(out, "w") as f:
        json.dump({
            "source": "evaluate_from_cache.py",
            "detectors_available": list(caches),
            "detectors_missing": missing,
            "folds": args.folds,
            "epochs": args.epochs,
            "seed": SEED,
            "results": results,
        }, f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
