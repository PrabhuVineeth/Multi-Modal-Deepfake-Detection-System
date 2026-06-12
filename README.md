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

## Quick Start

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
# Train on FaceForensics++
python train.py --dataset faceforensics --data-root /path/to/FF++

# Train on DFDC (first 10 chunks)
python train.py --dataset dfdc --data-root /path/to/dfdc --max-samples 5000

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
  --dataset faceforensics,celebdf,dfdc --data-root /path/to/data
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
│   ├── dfdc.py                  # DFDC
│   ├── celebdf.py               # Celeb-DF v2
│   ├── fakeavceleb.py           # FakeAVCeleb
│   ├── forgerynet.py            # ForgeryNet (boundary labels)
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
│   ├── metrics.py               # AUC, EER, ECE, F1
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
| DFDC | Audio + Video | Various | ✗ |
| Celeb-DF v2 | Video | Celebrity deepfakes | ✗ |
| FakeAVCeleb | Audio + Video | Face-swap, lip-sync, both | ✗ |
| ForgeryNet | Video | Various | ✓ |

See [`datasets/README.md`](datasets/README.md) for download instructions.

## Novel Contributions

1. **Forensic Evidence Aggregation Engine** — Attention-based weighting that learns per-sample which evidence channels are most discriminative
2. **Joint Mismatch Localizer** — Multi-channel frame-level anomaly scoring across lip-sync, identity, temporal, and AV-sync dimensions
3. **Temporal Forgery Boundary Detector (TFBD)** — 1D dilated CNN + CRF for precise REAL/FAKE/BOUNDARY frame-level sequence labeling
4. **Cross-Modal Heatmap Generator** — Combined acoustic + visual anomaly visualization

## License

This project is for research and educational purposes.
