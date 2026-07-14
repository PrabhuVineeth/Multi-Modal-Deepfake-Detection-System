# Methodology Notes for IEEE Paper

## Proposed System
The project implements a multimodal deepfake forensic detection system that combines audio, facial, mouth-region, temporal, and cross-modal evidence. The system accepts a video input, extracts audio and visual streams, detects faces and mouth regions, encodes speech with a Wav2Vec2-based audio branch, encodes face/mouth information with ViT-based visual branches, fuses modalities through cross-attention, and produces a calibrated REAL/FAKE prediction with interpretable evidence scores.

## Project-Specific Elements
- Three cross-modal attention pathways: speech-lip, voice-identity, and audio-video timing consistency.
- Evidence aggregation over lip-sync, identity, temporal, and AV-sync analyzer heads.
- Temporal forgery boundary detection for frame/segment-level interpretability.
- Cross-modal heatmap and HTML/JSON forensic report generation.
- Dataset-aware inference routing with calibrated thresholds and visual-only routing for FaceForensics++.

## Classical ML Baseline
A separate feature-based ML baseline was evaluated on FakeAVCeleb using extracted audio and video features. It reports Audio-only, Video-only, and Fusion modes using Logistic Regression, SVM, Random Forest, and XGBoost. Thresholds were selected on validation splits and evaluated on held-out test splits.
