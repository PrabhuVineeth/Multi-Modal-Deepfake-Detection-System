# Pre-Deletion Reported Evaluation Metrics — ARCHIVAL RECORD

**Status: reported numbers, NOT currently reproducible.** This file preserves the
per-dataset + joint metrics as they were reported *before* the deletion of the
per-dataset checkpoints. It is an archival record of what was previously stated,
kept so the values are not lost — it is **not** a verified or reproducible result
set. See the provenance notes before citing any figure here.

## Reported table (verbatim, as provided 2026-07-28)

| Dataset / Route | Evaluation Mode | Optimal Threshold (T) | ROC-AUC | EER | Accuracy | Precision | Recall | F1-Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FakeAVCeleb | Audio-Visual | 0.96 | 0.9130 | 11.50% | 82.00% | 74.24% | 98.00% | 84.48% | 0.0287 |
| FaceForensics++ | Visual-Only | 0.52 | 0.7530 | 23.10% | 67.50% | 62.24% | 89.00% | 73.25% | 0.0636 |
| LAV-DF | Audio-Visual | 0.40 | 0.8060 | 19.10% | 73.13% | 70.03% | 80.87% | 75.06% | 0.2180 |
| Joint (Combined) | Audio-Visual | 0.50 | 0.9520 | 5.80% | 91.20% | 92.50% | 93.10% | 92.80% | 0.1579 |

## Why these are not reproducible from the current repository

1. **Source checkpoints were deleted.** `best_model_fakeavceleb.pth` and
   `best_model_lavdf_quick_fixed.pth` (the checkpoints that produced the
   FakeAVCeleb and LAV-DF rows) were deleted during a disk cleanup. They are
   **not in version control** (`checkpoints/` is gitignored) and **not in the
   Windows Recycle Bin** — so they cannot be restored.
2. **The evaluation embeddings are unusable for a balanced metric.** The cached
   embeddings hold only 3 real / 147 fake (FakeAVCeleb), 18/132 (FF++), 40/110
   (LAV-DF). On such sets a model that always predicts "fake" reaches
   98/88/73 % accuracy, so the accuracy/precision/recall/F1 columns above cannot
   be reproduced or trusted without a balanced test set.

## Known inconsistencies in the table itself

- **The Joint row conflicts with the project's own presentation.** This table
  lists Joint ROC-AUC **0.9520** / F1 **0.9280**; `presentation_slides.html`
  (slide 07) lists Joint AUC **0.8610** / F1 **0.8490**. Two different joint
  numbers exist in the project's materials; the source of 0.9520 is not on disk.
- **The ECE column matches a degenerate run.** The ECE values
  (0.0287 / 0.0636 / 0.2180) are exactly those of
  `output/final_evaluation_combined/` — the all-fake run — suggesting this table
  was assembled from more than one evaluation.
- The three per-dataset **ROC-AUC** values (0.9130 / 0.7530 / 0.8060) do match
  `presentation_slides.html`, so those are the presentation's canonical AUCs.

## Reproducible alternatives (use these for any new claim)

- **FakeAVCeleb (identity-disjoint, committed):** 93.97 % AUC — `results/ablation/`.
  More rigorous than the 0.9130 above and fully reproducible.
- **Balanced cross-validated eval (all datasets + joint):** run
  `evaluate_balanced_all.py`, which extracts balanced embeddings from raw video
  and trains fresh fusion heads under 5-fold CV — no deleted checkpoint required.
- To restore the exact deployed per-dataset models, retrain with the existing
  training scripts and raw data on the `G:` drive.
