# Limitations and Threats to Validity

- Dataset distribution varies substantially across FakeAVCeleb, FaceForensics++, and LAV-DF; direct cross-dataset comparison must be interpreted cautiously.
- FaceForensics++ routing is visual-only because audio in the available setup is silent or unreliable.
- Some classical ML samples were skipped when face/audio feature extraction failed, causing post-extraction class imbalance.
- Thresholds should be selected on validation splits only; test-set threshold tuning is not appropriate for final reporting.
- Heatmaps are frame-level anomaly overlays, not true pixel-level causal attribution.
- Model behavior on arbitrary internet videos may differ from benchmark test-set behavior due to compression, resolution, codec, lighting, and domain shift.
