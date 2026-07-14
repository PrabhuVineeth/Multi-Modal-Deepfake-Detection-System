# Conference Paper Package - MDDS

Generated: 2026-07-14T09:48:44

This folder contains high-signal artifacts for writing an IEEE conference paper about the Multimodal Deepfake Detection System.

## Folder Map

- `tables/` - CSV tables for checkpoint inventory, dataset summary, architecture components, calibration summaries, and ML baseline results.
- `figures/` - Mermaid architecture diagram source and any copied visual assets.
- `evidence/calibration/` - Raw calibration JSON/CSV evidence copied from `output/calibration`.
- `evidence/ml_baseline/` - FakeAVCeleb classical ML baseline outputs for sample sizes 500 and 750.
- `evidence/demo_reports/` - Recent demo JSON/HTML reports and heatmaps when small enough to copy.
- `evidence/source_snapshots/` - Source files needed to describe and reproduce the architecture.
- `text/` - Paper outline, methodology notes, limitations, and result summary notes.
- `logs/` - Reserved for short logs or command records.

Model checkpoint `.pth` files are not copied because they are large. Their inventory is in `tables/checkpoint_inventory.csv`.
