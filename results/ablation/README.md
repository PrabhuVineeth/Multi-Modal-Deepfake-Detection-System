# FakeAVCeleb 2x3 Ablation — Face Detectors x Visual Backbones

Produced by [`ablation_fakeavceleb.py`](../../ablation_fakeavceleb.py) on 2026-07-24.

| File | What it is |
|---|---|
| `ablation_results.json` | **Primary result.** 5-fold, both tables, detector diagnostics, full config. |
| `results_10fold.json` | Robustness check at 10 folds. Not the headline — see below. |
| `results_retinaface_partial.json` | Independent 5-fold repeat of the RetinaFace arm; used to measure run-to-run noise. |

Feature caches (`features_*.npz`, ~20 MB each) and the full run log stay under the
gitignored `output/ablation_fakeavceleb/`. Re-score without re-extracting via
[`evaluate_from_cache.py`](../../evaluate_from_cache.py).

## Protocol

- 1000 videos: 500 real (**all** of RealVideo-RealAudio) + 500 fake drawn from the
  pooled 21,066, spanning 500 identities.
- Frozen pretrained backbones (linear-probe); only the fusion head trains.
- Face detection runs **once per detector**; all three backbones encode identical
  crops, so the backbone arm has no detection confound.
- 5-fold `StratifiedGroupKFold` grouped on identity (`id00XXX`), asserted zero
  train/test identity overlap per fold.
- `face_detect_pad_ratio=0.5`, applied identically to both detectors.

## Headline numbers (5-fold)

| Backbone (RetinaFace) | AUC % | EER % | Acc % | Params |
|---|---|---|---|---|
| ViT-16 | 93.97 ± 1.00 | 9.11 ± 1.45 | 91.30 ± 0.63 | 86M |
| ViT-32 | 88.64 ± 3.46 | 16.32 ± 3.89 | 82.37 ± 6.07 | 88M |
| Swin-Tiny | 93.28 ± 0.47 | 10.80 ± 1.98 | 88.40 ± 2.28 | 27.5M |

| Detector (ViT-16) | Face det. % | ms/frame | AUC % |
|---|---|---|---|
| RetinaFace | 99.9 | **9.7** | 93.97 ± 1.00 |
| MTCNN | 99.9 | 25.0 | 93.33 ± 1.86 |

## Read these caveats before quoting anything

**Quote effect sizes, not p-values.** Two runs of the *identical* RetinaFace+ViT-16
config differ by **1.09 AUC points** (92.88 vs 93.97) from GPU nondeterminism
(cuDNN backward kernels; `torch.use_deterministic_algorithms(True)` is not set).
That is the measurement noise floor.

- **ViT-16 > ViT-32 is real**: +5.35 / +5.33 / +4.63 across three independent runs,
  ~5x the noise floor. But p ranged 0.026–0.107 depending on run and fold count —
  do not stake the claim on p<0.05.
- **ViT-16 vs Swin-Tiny is a tie** (0.69–0.98 pts, never significant). Swin-Tiny
  matches it at **1/3 the parameters**.
- **RetinaFace vs MTCNN accuracy is noise — the sign flips** between runs
  (+0.64 at 5-fold, −0.80 at 10-fold). RetinaFace's justification is **2.6x lower
  latency** at identical 99.9% detection, not accuracy.
- **10-fold is not "better"**: ~100 test videos/fold made per-fold AUC noisier
  (ViT-16 ±1.00 → ±3.13) and p-values worse. Use the 5-fold as primary.

## Known limitations

1. **Only 14 of 500 fakes (2.8%) are audio-only** (RealVideo-FakeAudio), because the
   fake half is an unstratified draw from the pooled 21,066. This study therefore
   says **nothing** about the audio branch or AV-sync analyzers. A stratified
   500/500/500 redraw would fix it — RealVideo-FakeAudio has exactly 500 available.
2. **500 identities is a hard ceiling.** All 21,566 FakeAVCeleb videos belong to the
   same 500 people, so identity-disjoint evaluation caps independent test units at
   ~100/fold regardless of video count. More videos will not buy statistical power;
   only another dataset will.
3. ViT-32 uses `google/vit-base-patch32-224-in21k` (in21k only) — `google/vit-base-patch32-224`
   does not exist on the Hub. ViT-16 is in21k→in1k. Disclose this asymmetry.

## Superseded

`train_and_eval_backbones.py` and `eval_saved_checkpoints.py` produced invalid
results and must not be used: they randomly initialized ViT-32/Swin, split at video
level (65% identity leakage), synthesized the mouth stream as `face_feats*0.9+noise`,
and fed raw 0-255 BGR to the encoders. The `checkpoints/best_model_{detector}_{backbone}.pth`
files are unusable — they store only the fusion head, while the random feature
extractor they were trained against was never saved.
