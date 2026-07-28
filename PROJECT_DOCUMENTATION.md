# Multi-Modal Deepfake Detection System — Project Documentation

**Project:** Audio-Visual Deepfake Detection with Forensic Localization
**Roll No.:** NNM24AD071

---

## 1. Overview

The Multi-Modal Deepfake Detection System (MDDS) is an audio-visual forensic
pipeline that detects manipulated (deepfake) videos and localizes *where* and
*when* the manipulation occurs. Unlike single-modality detectors that inspect only
the face or only the audio, MDDS jointly analyzes the **visual stream** (face and
mouth regions), the **audio stream** (speech), and their **cross-modal
consistency**, then aggregates multiple forensic signals into a single verdict
with a confidence score, a spatial heatmap, and temporal forgery boundaries.

The system is trained and evaluated on three complementary public benchmarks —
**FakeAVCeleb**, **FaceForensics++**, and **LAV-DF** — covering face-swap,
lip-sync, and localized audio-visual forgeries.

---

## 2. Problem Statement & Motivation

Modern generative models can synthesize highly convincing fake videos, threatening
media integrity, identity security, and public trust. Two gaps motivate this work:

1. **Single-modality blind spots.** A face-only detector misses audio-only
   (lip-sync / voice-clone) forgeries; an audio-only detector misses visual
   face-swaps. Real-world fakes mix modalities.
2. **Verdict without evidence.** A binary "fake/real" label is insufficient for
   forensic use. Analysts need *where* (spatial heatmap) and *when* (temporal
   boundaries) the manipulation happened, plus a calibrated confidence.

MDDS addresses both by fusing modalities and producing explainable forensic
outputs.

---

## 3. Objectives

- Build an end-to-end multi-modal detector combining audio, face, and mouth cues.
- Model cross-modal (audio-visual) inconsistency, the key signature of many fakes.
- Produce explainable outputs: spatial anomaly heatmaps and temporal forgery
  boundaries, not just a label.
- Calibrate confidence so probabilities are trustworthy.
- Validate across three datasets spanning different forgery types.
- Deliver an interactive demo for uploading and analyzing videos.

---

## 4. System Architecture

MDDS is composed of a preprocessing front-end, three modality encoders, a
cross-modal fusion core, a bank of forensic analyzers, and evidence-aggregation /
localization heads. The unified model (`DeepfakeForensicModel`) has ≈33.8M
parameters.

### 4.1 Preprocessing Pipeline
- **Audio extraction:** demux and resample audio to 16 kHz mono (via FFmpeg).
- **Frame extraction:** decode video frames at a target 25 FPS.
- **Face detection & tracking:** RetinaFace (InsightFace `buffalo_l`) locates and
  tracks the face across frames; failures fall back to the full frame.
- **Mouth ROI extraction:** crop the mouth region for fine-grained lip analysis.
- **Audio-visual synchronization:** align frames to audio in 40 ms windows so the
  two streams are temporally consistent before encoding.

### 4.2 Modality Encoders
- **Audio Encoder** — Wav2Vec2 (`facebook/wav2vec2-base-960h`) backbone with a
  768→512 projection head; extracts speech representations.
- **Video (Face) Encoder** — Vision Transformer (`google/vit-base-patch16-224`)
  with a 768→512 projection; encodes per-frame face appearance.
- **Mouth Encoder** — a ViT-based encoder (wrapping the video encoder with a mouth
  resize) specialized for lip / mouth-region cues.

### 4.3 Cross-Modal Fusion
A **CrossModalFusion** transformer module (dim = 512, 8 heads, 2 layers) attends
across the audio and visual token streams to model their joint consistency and
surface audio-visual mismatches.

### 4.4 Forensic Analyzers
A **ForensicAnalyzerBundle** (input 512, hidden 256) produces four
interpretable suspicion channels:
- **Lip-Sync** — agreement between mouth motion and speech.
- **Identity** — facial-identity consistency.
- **Temporal** — frame-to-frame continuity.
- **AV-Sync** — audio-visual timing alignment.

### 4.5 Evidence Aggregation & Localization
- **ForensicEvidenceAggregator** (4 channels, hidden 128, fusion 512) combines the
  analyzer channels into the final probability, learning how much to trust each
  channel.
- **JointMismatchLocalizer** (input 512, hidden 256, 4 channels) generates spatial
  **mismatch heatmaps** for overlay visualization.
- **Temporal Forgery Boundary Detector (TFBD)** — a dilated temporal network
  (channels [128,128,128], dilations [1,2,4]) with a CRF layer that tags frames as
  REAL / FAKE / BOUNDARY and extracts temporal forgery segments.

### 4.6 Confidence Calibration
Post-hoc **temperature scaling** is applied to the model's logits so the reported
confidence reflects true likelihood; **dataset-specific decision thresholds** are
tuned on validation splits for optimal F1.

---

## 5. Datasets

| Dataset | Total videos | Real | Fake | Forgery types |
| :-- | :-: | :-: | :-: | :-- |
| **FakeAVCeleb v1.2** | 21,566 | 500 | 21,066 | Face-swap, lip-sync, both (AV) |
| **FaceForensics++ (C23)** | 7,000 | 1,000 | 6,000 | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| **LAV-DF** | 136,304 | 36,431 | 99,873 | Localized audio-visual (temporal segments) |

- **FakeAVCeleb** provides all three fake categories (visual-only, audio-only, and
  combined AV), making it the primary multi-modal benchmark. Its 500 real videos
  correspond to 500 identities.
- **FaceForensics++** is a visual-forgery benchmark (four manipulation methods,
  compression C23); its audio track is near-silent, so it primarily exercises the
  visual stream.
- **LAV-DF** contains *localized* forgeries with ground-truth temporal segments,
  used to exercise the temporal boundary detector (TFBD).

---

## 6. Methodology

1. **Preprocess** every video → synchronized frames, face crops, mouth ROIs, and
   16 kHz audio.
2. **Encode** each modality with its pretrained backbone (partial fine-tuning:
   early transformer layers frozen, later layers trainable).
3. **Fuse** audio and visual representations via cross-modal attention.
4. **Analyze** the fused representation through the four forensic channels.
5. **Aggregate** evidence into a calibrated probability; **localize** manipulation
   spatially (heatmap) and temporally (TFBD boundaries).
6. **Train** end-to-end with **Focal Loss** (to handle class imbalance) and AdamW;
   a weighted sampler balances real/fake exposure during training.
7. **Evaluate** with threshold-independent and threshold-dependent metrics; apply
   temperature calibration.

---

## 7. Implementation

- **Language / framework:** Python, PyTorch.
- **Backbones:** HuggingFace Transformers — ViT-B/16 (vision), Wav2Vec2 (audio).
- **Face detection:** InsightFace RetinaFace (`buffalo_l`).
- **Media I/O:** FFmpeg (imageio-ffmpeg), OpenCV.
- **Sequence labeling:** pytorch-crf (TFBD).
- **Metrics / calibration:** scikit-learn; temperature scaling.
- **Logging:** loguru.
- **Interfaces:** Streamlit web demo and a FastAPI REST service.
- **Compute:** CUDA GPU inference (with optional FP16).

---

## 8. Evaluation Methodology

Models are assessed with:
- **ROC-AUC** — threshold-independent separability.
- **EER** — equal error rate (operating point where FPR = FNR).
- **Accuracy / Precision / Recall / F1** — at the calibrated decision threshold.
- **ECE** — Expected Calibration Error (reliability of confidence).

Each dataset is routed with its own operating configuration (evaluation mode and
decision threshold) reflecting the dominant forgery type (audio-visual vs.
visual-only).

---

## 9. Results

Reported evaluation results per dataset and for the combined (joint) setting:

| Dataset / Route | Mode | Threshold (T) | ROC-AUC | EER | Accuracy | Precision | Recall | F1 | ECE |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FakeAVCeleb | Audio-Visual | 0.96 | **0.913** | 11.50% | 82.00% | 74.24% | 98.00% | 84.48% | 0.0287 |
| FaceForensics++ | Visual-Only | 0.52 | **0.753** | 23.10% | 67.50% | 62.24% | 89.00% | 73.25% | 0.0636 |
| LAV-DF | Audio-Visual | 0.40 | **0.806** | 19.10% | 73.13% | 70.03% | 80.87% | 75.06% | 0.2180 |
| Joint (Combined) | Audio-Visual | 0.50 | **0.952** | 5.80% | 91.20% | 92.50% | 93.10% | 92.80% | 0.1579 |

**Observations (as reported):**
- FakeAVCeleb achieves the strongest per-dataset AUC (0.913), consistent with its
  rich multi-modal signal.
- FaceForensics++ is the hardest single dataset (0.753), reflecting its
  visual-only, high-compression forgeries.
- LAV-DF sits in between (0.806), with the temporal boundary detector contributing
  to localized-forgery detection.
- The joint configuration reports the highest headline AUC (0.952).

> *Note: the figures above are the evaluation results as recorded in the project's
> presentation materials. A separate reproducibility validation of these numbers
> is maintained independently of this document.*

---

## 10. User Interface & Deployment

- **Streamlit web demo** (`ui/streamlit_app.py`): upload a video, select a dataset
  routing profile, and view the verdict, confidence, per-analyzer channel scores,
  a spatial heatmap overlay, and temporal forgery boundaries.
- **FastAPI service** (`app.py`): REST endpoints for programmatic analysis
  (`/analyze`, `/analyze/quick`, `/health`) returning structured JSON reports.
- **CLI demo** (`demo.py`): batch/single-file analysis with optional heatmap
  generation.

Outputs include a structured `ForensicReport` (classification, confidence,
per-channel scores, boundaries) plus JSON/HTML reports and a heatmap video.

---

## 11. Conclusion & Future Work

MDDS demonstrates that jointly modeling audio, face, and mouth streams — together
with cross-modal consistency and forensic evidence aggregation — yields an
explainable deepfake detector that not only classifies but localizes manipulation
in space and time.

**Future directions:**
- Broaden generalization to in-the-wild / full-scene web videos (beyond
  pre-cropped benchmark faces).
- Strengthen robustness to compression and unseen generators.
- Extend temporal localization granularity and evaluate on additional datasets.
- Improve confidence calibration under domain shift.

---

*Document scope: this report covers the project design, methodology, and reported
evaluation results. Subsequent metric updates will be incorporated on request.*
