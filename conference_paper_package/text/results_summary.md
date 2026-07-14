# Results Summary Notes

## Deep MDDS Checkpoints
See `tables/checkpoint_inventory.csv` for available trained checkpoints. Important final-stage checkpoints include `best_model_fakeavceleb.pth`, `best_model_faceforensics_visual.pth`, `best_model_lavdf_full.pth`, and `best_model_combined.pth`.

## Calibration Evidence
Calibration reports and threshold sweeps are copied under `evidence/calibration/`, with a consolidated table in `tables/calibration_reports_summary.csv` when JSON fields are available.

## Classical ML Baseline
The sample-size 500 and 750 FakeAVCeleb classical ML baseline files are copied under `evidence/ml_baseline/` and `tables/`. The high-level best-model summary is in `tables/ml_baseline_best_summary.csv`.

## Key Known Finding From ML Baseline
Across the 500/class and 750/class runs, Fusion + XGBoost was the strongest classical ML baseline. This supports the claim that combined audio-video evidence is more useful than single-modality features, while also showing that audio-only features are weaker and less stable.
