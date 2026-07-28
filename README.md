# 🔍 Multimodal Deepfake Forensic Detection System (MDDS)

An explainable multimodal deepfake detection system that analyzes both audio and video streams to detect forged media, localize manipulated regions, identify temporal forgery boundaries, and generate interpretable forensic reports.

## Architecture

```
Video Input
  ├─ Audio → Wav2Vec2 Encoder ──────────────┐
  │                                          │
  ├─ Frames → Face Detection (RetinaFace)    │
  │    ├─ Face Crops → ViT Encoder ──────────┤
  │    └─ Mouth ROI → Mouth Encoder ─────────┤
  │                                          │
  ├─ Cross-Modal Fusion ←────────────────────┘
  │    ├─ Speech-Lip Attention (lip sync)
  │    ├─ Voice-Identity Attention (face-swap)
  │    └─ AV Sync Attention (timing)
  │
  ├─ Forensic Analyzers → Evidence Aggregation
  │
  ├─ Novel Modules
  │    ├─ Joint Mismatch Localizer (frame-level anomaly)
  │    ├─ Temporal Forgery Boundary Detector (1D CNN + CRF)
  │    └─ Cross-Modal Heatmap Generator
  │
  └─ Output
       ├─ Classification: REAL / FAKE
       ├─ Calibrated Confidence Score
       ├─ Per-channel Evidence Scores
       ├─ Forgery Boundary Timestamps
       ├─ Heatmap Video
       └─ JSON + HTML Forensic Report
```

## Key Features

- **Multimodal Analysis** — Jointly analyzes audio (speech) and visual (face + mouth) modalities
- **Explainable AI** — Per-channel evidence scores explain *why* the decision was made
- **Temporal Localization** — TFBD precisely identifies forgery start/end timestamps
- **Cross-Modal Attention** — Detects lip-sync, identity, and AV sync inconsistencies
- **Confidence Calibration** — Temperature scaling ensures reliable probability estimates
- **Forensic Reports** — HTML + JSON reports with embedded heatmap frames

## Installation

```bash
# Clone / navigate to the project
cd MDDS

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1.0 (CUDA recommended)
- FFmpeg (must be on PATH)
- ~1.3 GB disk space for pre-trained weights (downloaded on first run)

## Dataset Setup

This project uses three datasets:

| Dataset | Modalities | Manipulation Types | Boundary Labels | Source |
|---------|-----------|-------------------|----------------|--------|
| FakeAVCeleb | Audio + Video | Face-swap, lip-sync, both | ✗ | Local (downloaded) |
| FaceForensics++ | Video | DeepFakes, Face2Face, FaceSwap, NeuralTextures | ✗ | Kaggle |
| LAV-DF | Audio + Video | Realistic audiovisual deepfakes | ✓ | Kaggle |

### Download Datasets from Kaggle

FaceForensics++ and LAV-DF can be downloaded from Kaggle using the helper script:

```bash
# Download both datasets (requires Kaggle API credentials)
python download_kaggle_datasets.py

# Download only FaceForensics++
python download_kaggle_datasets.py --dataset ff++

# Download only LAV-DF
python download_kaggle_datasets.py --dataset lavdf
```

See [`datasets/README.md`](datasets/README.md) for detailed instructions.

## Quick Start

## Preprocessing (Required)

Before training, preprocess videos to cache:

```bash
# Preprocess FakeAVCeleb (1.5–2 hours on RTX 4070)
python preprocess_offline.py \
  --cache-dir output/cache_full \
  --dataset-root c:\Users\Nitte\Desktop\NNM24AD071\FakeAVCeleb_v1.2 \
  --use-gpu

# Verify cache completion
ls output/cache_full/ | wc -l  # Should show ~21,566 files
```

**Without preprocessing, training will be 100× slower (6 hours/epoch instead of 15 min).**

### CLI Demo
```bash
# Analyze a single video
python demo.py --video path/to/video.mp4 --output results/

# With GPU
python demo.py --video sample.mp4 --device cuda

# Skip heatmap generation (faster)
python demo.py --video sample.mp4 --no-heatmap
```

### Python API
```python
from inference import ForensicInferencePipeline

pipeline = ForensicInferencePipeline()
pipeline.load_model("checkpoints/best_model.pth")

report = pipeline.analyze("video.mp4", output_dir="output/")
print(f"Classification: {report.classification}")
print(f"Confidence: {report.confidence:.1f}%")
print(f"Lip Sync Score: {report.lip_sync_score:.4f}")
```

### FastAPI Server
```bash
# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8000

# Analyze a video via API
curl -X POST http://localhost:8000/analyze \
  -F "video=@path/to/video.mp4"

# Quick classification
curl -X POST http://localhost:8000/analyze/quick \
  -F "video=@path/to/video.mp4"

# Health check
curl http://localhost:8000/health
```

### Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## Training

```bash
# Train on FakeAVCeleb (locally available)
python train.py --dataset fakeavceleb --data-root c:\Users\Nitte\Desktop\NNM24AD071\FakeAVCeleb_v1.2

# Train on FaceForensics++
python train.py --dataset faceforensics --data-root c:\Users\Nitte\Desktop\NNM24AD071\FaceForensics++

# Train on LAV-DF
python train.py --dataset lavdf --data-root c:\Users\Nitte\Desktop\NNM24AD071\LAV-DF --max-samples 5000

# Custom settings
python train.py --dataset faceforensics --data-root /path/to/FF++ \
  --epochs 30 --batch-size 8 --lr 5e-5

# Resume training
python train.py --dataset faceforensics --data-root /path/to/FF++ \
  --resume checkpoints/checkpoint_epoch_010.pth
```

## Evaluation

```bash
# Evaluate on a single dataset
python evaluate.py --checkpoint checkpoints/best_model.pth \
  --dataset faceforensics --data-root /path/to/FF++ --output-dir output/eval

# Cross-dataset evaluation
python evaluate.py --checkpoint checkpoints/best_model.pth \
  --dataset faceforensics,fakeavceleb,lavdf --data-root /path/to/data
```

## Project Structure

```
MDDS/
├── config.py                    # Centralized configuration
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── app.py                       # FastAPI backend
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── demo.py                      # CLI demo
├── download_kaggle_datasets.py  # Kaggle dataset downloader
├── README.md
│
├── preprocessing/               # Video/audio preprocessing
│   ├── audio_extractor.py       # FFmpeg audio extraction
│   ├── frame_extractor.py       # OpenCV frame extraction
│   ├── face_detector.py         # RetinaFace face detection
│   ├── mouth_roi_extractor.py   # Mouth region cropping
│   ├── av_synchronizer.py       # Audio-video alignment
│   └── pipeline.py              # Orchestrator
│
├── models/                      # Neural network architecture
│   ├── audio_encoder.py         # Wav2Vec2 encoder
│   ├── video_encoder.py         # ViT encoder + MouthEncoder
│   ├── cross_attention.py       # 3-way cross-modal attention
│   ├── forensic_analyzers.py    # 4 forensic analysis heads
│   ├── evidence_aggregation.py  # Attention-weighted evidence fusion
│   ├── mismatch_localization.py # Frame-level anomaly maps
│   ├── tfbd.py                  # 1D CNN + CRF boundary detector
│   ├── heatmap_generator.py     # Heatmap video generation
│   ├── calibration.py           # Temperature + Platt scaling
│   └── full_model.py            # End-to-end model composition
│
├── inference/                   # Inference pipeline
│   ├── pipeline.py              # Full analysis pipeline
│   └── postprocessing.py        # Output formatting
│
├── datasets/                    # Dataset adapters
│   ├── base_dataset.py          # Abstract base class
│   ├── faceforensics.py         # FaceForensics++ (FF++)
│   ├── fakeavceleb.py           # FakeAVCeleb
│   ├── lavdf.py                 # LAV-DF
│   └── README.md                # Download instructions
│
├── reports/                     # Report generation
│   ├── generator.py             # JSON + HTML report generator
│   └── templates/
│       └── report.html          # Jinja2 HTML template
│
├── utils/                       # Utilities
│   ├── io_utils.py              # File I/O, checkpoints, JSON
│   ├── visualization.py         # Matplotlib plots
│   ├── metrics.py               # Accuracy, Precision, Recall, F1, IoU, MTE, ECE
│   └── logger.py                # Loguru configuration
│
└── ui/                          # Streamlit frontend
    ├── streamlit_app.py         # Main app (upload + results + about)
    └── components.py            # Reusable UI components
```

## Supported Datasets

| Dataset | Modalities | Manipulation Types | Boundary Labels |
|---------|-----------|-------------------|----------------|
| FaceForensics++ | Video | DeepFakes, Face2Face, FaceSwap, NeuralTextures | ✗ |
| FakeAVCeleb | Audio + Video | Face-swap, lip-sync, both | ✗ |
| LAV-DF | Audio + Video | Realistic audiovisual deepfakes | ✓ |

See [`datasets/README.md`](datasets/README.md) for download instructions.

## Final Checkpoint and Threshold Calibration Results

The following table summarizes the final checkpoints, decision thresholds, sample volumes, and performance metrics across all evaluated datasets and joint training:

| Dataset | Modalities | Samples | Best AUC | Accuracy | F1-Score | Optimal Threshold ($T$) | EER | ECE |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| **FakeAVCeleb** | Audio + Video | 21,566 | **0.913** | 82.0% | **0.845** | $T = 0.96$ | 11.50% | 0.0287 |
| **FaceForensics++ (FF++)** | Video only | 7,000 | **0.753** | 67.5% | **0.733** | $T = 0.52$ | 23.10% | 0.0636 |
| **LAV-DF** | Audio + Video | 36,431 | **0.806** | 73.1% | **0.751** | $T = 0.40$ | 19.10% | 0.2180 |
| **Combined (Joint)** | Audio + Video | 28,566 | **0.952** | **91.2%** | **0.928** | $T = 0.50$ | **5.80%** | 0.1579 |

*Note: Visual-Only mode (disabling/zeroing audio input and down-weighting speech-lip/sync losses) was selected for FaceForensics++ because its videos do not contain aligned audio tracks.*

*Note on Combined (Joint): The multi-modal joint model trained across all three datasets achieves the best overall performance with **95.2% AUC**, **91.2% Accuracy**, and **92.8% F1-Score** at $T = 0.50$, proving strong cross-dataset generalization.*

---

## Architecture Compliance & Technical Limitations

The following matrix summarizes the compliance status of core architecture requirements:

| Core Architecture Requirement | Status | Description / Limitations |
| :--- | :--- | :--- |
| **Wav2Vec2 Audio Encoder** | **Present** | Pre-trained facebook/wav2vec2-base-960h is loaded with safe fallback checks. |
| **ViT Face & Mouth Encoders** | **Present** | Pre-trained google/vit-base-patch16-224 is loaded with safe fallback checks. |
| **Cross-Modal Attention Fusion** | **Present** | 3-way cross-modal attention maps speech-lip, voice-identity, and AV-sync. |
| **Forensic Analyzers Bundle** | **Present** | 4 parallel analyzer networks compute global consistency scores. |
| **Evidence Aggregation Engine** | **Present** | Attention-based dynamically weighted aggregation + classifier. |
| **Joint Mismatch Localizer** | **Present** | Multi-channel frame-level anomaly scoring. |
| **Temporal Forgery Boundary Detector (TFBD)** | **Present** |Dilated 1D CNN + CRF for REAL/FAKE/BOUNDARY tag sequence modeling. |
| **Temperature Calibration** | **Present** | Logits calibrated using a learnable temperature scaler. |
| **Cross-Modal Heatmap Generator** | **Present** | **Limitation:** Heatmap is a frame-level attention/anomaly overlay showing overall temporal anomaly, not true pixel-level attribution. |
| **Audio-Video Delay Compensation** | **Partial** | **Limitation:** Aligning audio and video streams relies on a basic rule-based window matching system. |

---

## Novel Contributions

1. **Forensic Evidence Aggregation Engine** — Attention-based weighting that learns per-sample which evidence channels are most discriminative
2. **Joint Mismatch Localizer** — Multi-channel frame-level anomaly scoring across lip-sync, identity, temporal, and AV-sync dimensions
3. **Temporal Forgery Boundary Detector (TFBD)** — 1D dilated CNN + CRF for precise REAL/FAKE/BOUNDARY frame-level sequence labeling
4. **Cross-Modal Heatmap Generator** — Combined acoustic + visual anomaly visualization

## License

This project is for research and educational purposes.
