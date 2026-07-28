# Final Consolidated Evaluation Metrics

**Status: reproducible, identity-disjoint, internally consistent.** This is the
authoritative results set. It **supersedes** both the pre-deletion reported table
(`results/baseline-pre-deletion/`, not reproducible) and the buggy
`output/balanced_eval_150/` run (instrument artifact — see its README).

Generated 2026-07-28.

## 1. Headline results

Balanced real/fake, **frozen pretrained backbones** (ViT-16 face+mouth, wav2vec2
audio), trainable fusion head, **identity-disjoint 5-fold StratifiedGroupKFold**,
deterministic. "Late-Fusion" = mean of independently-trained audio-only and
visual-only heads.

| Dataset | N | #ID | Full-Fusion AUC | **Late-Fusion AUC** | F1 | Bal.Acc | ECE |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FakeAVCeleb | 1,000 | 500 | 85.9 | **92.6 ± 1.9** | 86.4 | 86.6 | 0.31 |
| FaceForensics++ | 2,000 | 1,026 | 76.0 | **89.3 ± 0.8** | 81.4 | 81.9 | 0.20 |
| LAV-DF | 4,000 | 3,821 | 81.8 | **89.0 ± 0.9** | 83.3 | 83.5 | 0.24 |
| **Macro-average** | — | — | 81.2 | **90.3** | 83.7 | — | — |
| **Joint (pooled, 1 model)** | 3,000 | 2,208 | 56.4 | **83.9 ± 1.8** | 76.1 | 77.3 | 0.31 |

- Macro-average = mean of the three per-dataset numbers ("overall across benchmarks").
- Joint (pooled) = a single classifier trained across all three datasets at
  balanced equal contribution (500/class each), identity-disjoint with
  dataset-namespaced identities. Genuinely harder than per-dataset — the honest
  cross-dataset number.

## 2. The late-fusion contribution (the headline finding)

End-to-end (full) fusion suffers **negative transfer**; late fusion of
independently-trained streams recovers it. Paired per-fold t-test (same folds):

| Comparison | Δ AUC | p | |
| :-- | :-: | :-: | :-- |
| Late vs Full — FakeAVCeleb | +6.8 | 0.092 | n.s. (borderline) |
| Late vs Full — FaceForensics++ | +13.3 | **0.039** | **significant** |
| Late vs Full — LAV-DF | +7.1 | 0.134 | n.s. |
| Late vs Full — **Joint (pooled)** | **+27.6** | **0.001** | **significant** |

On the heterogeneous pooled joint, end-to-end fusion collapses to near-chance
(56.4) while late fusion holds at 83.9 — the clearest demonstration of the effect.

## 3. Calibration

Temperature scaling on out-of-fold scores. ECE remains high (0.20–0.31): the
focal-loss heads are overconfident on ~1–4k samples, and post-hoc scaling only
nudges it (FF++ 0.209→0.204, LAV-DF 0.257→0.243; FakeAVCeleb did not improve).
**Reported honestly as a known limitation**, not hidden.

## 4. Comparison to the earlier (unreproducible) reported table

| Dataset | Reported (pre-deletion, NOT reproducible) | **This work (reproducible)** |
| :-- | :-: | :-: |
| FakeAVCeleb | 0.913 | **0.926** |
| FaceForensics++ | 0.753 | **0.893** |
| LAV-DF | 0.806 | **0.890** |
| Joint | 0.952 (inflated/impossible) | **0.839** (pooled) / 0.903 (macro) |

The per-dataset late-fusion numbers **match or exceed** the old reported figures —
but these are identity-disjoint and fully reproducible. The old "Joint 0.952" was
higher than any per-dataset number (backwards for a harder task) and is not
reproducible; the honest pooled joint is 0.839.

## 5. Reproduce

```bash
# per-dataset corrected features + full-fusion (identity-disjoint)
python evaluate_crossdataset.py --per-class 500 --datasets fakeavceleb --out output/crossdataset_eval
python evaluate_crossdataset.py --per-class 2000 --datasets faceforensics,lavdf --out output/crossdataset_eval_scaleup
# (FF++ auto-caps at its 1000-real limit; LAV-DF uses 2000/class)

# late fusion + calibration on the cached features
python evaluate_crossdataset_latefusion.py --feat-dir output/crossdataset_eval_scaleup

# joint (macro + pooled)
python evaluate_joint.py --per-class 500
```

Artifacts: `output/crossdataset_eval*/crossdataset_results.json`,
`latefusion_calibration_results.json`, `joint_results.json`.

## 6. Protocol & caveats

- **Frozen linear-probe**, not end-to-end fine-tuned: measures backbone
  representation quality + fusion, reproducibly. Higher numbers than this would
  require fine-tuning (which collapsed at 1–2 epochs in the deleted checkpoints).
- **FakeAVCeleb caps at 500 identities** (500 real videos, one per identity); more
  samples cannot raise its statistical power. FF++ ≈ 1,000 source ids; LAV-DF
  grouped at source-video level (no speaker id in metadata), not speaker-disjoint.
- **FF++ audio is near-silent** — its audio branch carries little signal; FF++
  performance is essentially the visual stream.
- The committed FakeAVCeleb detector×backbone ablation (`results/ablation-stratified/`)
  is consistent with these numbers (ViT-16 anchor).

## 7. Deployed UI

The full `DeepfakeForensicModel` checkpoints on disk are degenerate (undertrained,
collapsed to one class) and the working general checkpoint was deleted. The UI now
runs the **validated late-fusion head** (`checkpoints/latefusion_head.pt`, this
table's FakeAVCeleb config) via `models/inference/latefusion_pipeline.py` — verified
real→REAL, fake→FAKE. The head is FakeAVCeleb-trained (best on talking-head
content); arbitrary web video adds domain shift, and hard lip-sync fakes are the
weak spot.
