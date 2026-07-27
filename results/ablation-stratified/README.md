# FakeAVCeleb Stratified Ablation + Modality Analysis

Detector × backbone ablation on FakeAVCeleb v1.2 with **stratified balanced
sampling**, followed by a modality ablation and a late-fusion diagnostic that
together explain — and fix — the multi-modal system's weakness on audio-only
forgeries.

## Configuration
- **Dataset:** FakeAVCeleb v1.2
- **Samples:** 1,000 videos (500 real / 500 fake), all 500 dataset identities
- **Split:** 5-fold identity-disjoint `StratifiedGroupKFold` on `id00XXX` (asserted zero train/test identity overlap)
- **Stratified fake half:** 167 face-swap (video-only) / 167 lip-sync (audio-only) / 166 both (AV)
- **Protocol:** frozen pretrained backbones (linear-probe), trainable head
- **Determinism:** `use_deterministic_algorithms`, seeded, CUBLAS workspace pinned — verified bit-identical across two re-scores. Gaps below are exact, not noisy.

## 1. Face-detector arm (backbone fixed: ViT-16)
| Detector | Face Det % | ms/frame | AUC % | EER % |
| :-- | :-: | :-: | :-: | :-: |
| RetinaFace | 99.9 | **8.5** | 81.11 ± 4.99 | 25.90 ± 4.82 |
| MTCNN | 99.9 | 25.1 | 81.73 ± 3.24 | 26.32 ± 6.12 |

Accuracy tied (RetinaFace −0.6 AUC, p=0.74, n.s.); RetinaFace is **~3× faster**. Detector choice is a speed decision, not an accuracy one. NB both detectors see FakeAVCeleb's 224×224 pre-cropped faces, so 99.9% detection reflects a trivial detection task — see the deferred full-scene test.

## 2. Visual-backbone arm (detector fixed: RetinaFace)
| Backbone | AUC % | EER % | Balanced Acc % | Params |
| :-- | :-: | :-: | :-: | :-: |
| ViT-16 | 81.11 ± 4.99 | 25.90 ± 4.82 | 73.90 | 86M |
| ViT-32 | 76.30 ± 3.47 | 30.01 ± 3.58 | 68.29 | 88M |
| Swin-Tiny | 77.27 ± 3.98 | 29.20 ± 2.78 | 70.30 | 27.5M |

Effect sizes: ViT-16 vs ViT-32 +4.81 (p=0.093), vs Swin-Tiny +3.83 (p=0.066) — **none significant** at 5 folds (500-identity ceiling bounds power). Only **ViT-16 > ViT-32 is stable** (+5.35/+5.33/+4.63/+4.81 across four runs). ViT-16 vs Swin-Tiny was a tie (+0.7) in the 1:1 run and +3.8 here — **composition-dependent, treat as inconclusive**, and note Swin matches within noise at ⅓ the parameters.

## 3. Modality ablation (RetinaFace + ViT-16)
| Configuration | Overall AUC % | face-swap | **lip-sync (audio)** | both |
| :-- | :-: | :-: | :-: | :-: |
| Audio-Only | 69.97 ± 1.41 | 47.54 | 77.99 | 84.32 |
| Visual-Only | 84.15 ± 2.46 | 89.23 | 65.93 | 97.59 |
| Full Fusion (end-to-end) | 81.11 ± 4.99 | 86.03 | **61.54** | 94.26 |
| **Late-Fusion (mean of scores)** | **86.88 ± 1.88** | 84.93 | **77.65** | 98.00 |

Validity check: Audio-Only scores 47.5 (chance) on face-swap — correct, since face-swap has real audio. The ablation is wired correctly.

## 4. The headline finding: negative transfer, and its fix

**End-to-end fusion suppresses the audio signal; late fusion recovers it.**

| Paired test (same folds) | Δ AUC | p | |
| :-- | :-: | :-: | :-- |
| Late-Fusion vs Full Fusion (overall) | +5.77 | 0.043 | **significant** |
| Late-Fusion vs Full Fusion (lip-sync) | +15.97 | 0.032 | **significant** |
| Audio-Only vs Full Fusion (lip-sync) | +16.01 | 0.071 | n.s. (borderline) |
| Visual-Only vs Full Fusion (overall) | +3.05 | 0.121 | n.s. |

- On lip-sync, late fusion recovers to **77.65** (≈ Audio-Only's 78.0) while end-to-end sits at **61.54**. The audio features carry the forgery; the joint architecture threw it away.
- Late fusion (86.88) beats even Visual-Only (84.15) and has the **lowest variance** (±1.88 vs Full's ±4.99) — Full Fusion was both worse *and* unstable/overfitting on 800 samples.
- Late fusion wins in **all 5 folds individually**, which underpins the borderline p-values.

> **Contribution:** Late fusion of independently-trained audio and visual streams significantly outperforms end-to-end cross-modal fusion on FakeAVCeleb (86.9 vs 81.1 AUC, p=0.04), recovering audio-only-forgery detection that the joint model suppresses (77.7 vs 61.5, p=0.03). End-to-end fusion exhibits negative transfer and higher variance, consistent with a visually-dominant loss landscape overwhelming the audio branch.

## Caveats (must accompany any use of these numbers)
1. **Overall AUC ~81 here vs ~94 in the 1:1 run is NOT a regression.** The test set is deliberately loaded with the hard audio-only class (167 vs 14 samples); 81 ≈ mean of the three per-category AUCs. Report per-category; do not place 81 next to 94 without this note.
2. **Visual-Only's 65.9 on lip-sync is a confound, not detection.** Frame-diffing 499 lip-sync clips against their source real videos: mouth-region diff ≈ full-frame diff ≈ 1.9 (uniform), so the mouth is *not* locally modified — the video track is essentially the original, re-encoded. The residual visual separability is a global compression/preprocessing difference between folders (a known FakeAVCeleb bias). The late-fusion lip-sync gain comes from the **audio** branch (genuine fake audio), not vision.
3. **Mean late fusion is a floor, not optimal.** Averaging in chance-level audio (47.5) drags face-swap down (84.9 vs Visual's 89.2). A learned/weighted combiner should recover it — see the learned late-fusion follow-up.
4. **500-identity ceiling:** all FakeAVCeleb videos span the same 500 identities, so power cannot be bought with more videos — only another dataset raises it.

## Files
- `ablation_results.json` — full 2×3 grid, detector diagnostics, per-category AUC, config
- `modality_ablation_results.json` — audio / visual / full-fusion per-category
- `late_fusion_results.json` — late-fusion diagnostic + per-fold arrays + paired tests
- `run.log` — full stratified run log
- `modality_ablation_analysis.md` — narrative write-up

Reproduce from cached features (no re-extraction) with `evaluate_from_cache.py`,
`evaluate_modality_ablation.py`, `evaluate_late_fusion.py`.
