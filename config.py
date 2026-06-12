"""
Centralized configuration for the Deepfake Forensic Detection System.

All hyperparameters, model settings, paths, and preprocessing parameters
are defined here using dataclasses for type safety and easy modification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    # Encoder dimensions
    audio_embed_dim: int = 768          # Wav2Vec2 base output dimension
    visual_embed_dim: int = 768         # ViT base output dimension
    fusion_hidden_dim: int = 512        # Cross-attention / fusion hidden size
    projection_dim: int = 256           # Shared projection space

    # Cross-attention
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    num_cross_attention_layers: int = 2

    # Forensic analyzers
    analyzer_hidden_dim: int = 256
    analyzer_dropout: float = 0.2

    # Evidence aggregation
    evidence_dim: int = 128
    num_evidence_heads: int = 4

    # TFBD (Temporal Forgery Boundary Detector)
    tfbd_num_tags: int = 3              # REAL=0, FAKE=1, BOUNDARY=2
    tfbd_cnn_channels: List[int] = field(default_factory=lambda: [128, 128, 128])
    tfbd_kernel_size: int = 3
    tfbd_dilations: List[int] = field(default_factory=lambda: [1, 2, 4])

    # Backbone model names (HuggingFace)
    wav2vec2_model_name: str = "facebook/wav2vec2-base-960h"
    vit_model_name: str = "google/vit-base-patch16-224"

    # Calibration
    temperature_init: float = 1.5

    # Freezing strategy
    freeze_audio_layers: int = 8        # Freeze first N transformer layers of Wav2Vec2
    freeze_visual_layers: int = 8       # Freeze first N transformer layers of ViT


@dataclass
class PreprocessConfig:
    """Preprocessing pipeline parameters."""

    # Video
    target_fps: int = 25
    max_frames: int = 300               # Max frames to process per video (~12s at 25fps)
    frame_size: Tuple[int, int] = (720, 1280)  # H, W for initial resize

    # Face detection
    face_crop_size: Tuple[int, int] = (224, 224)
    face_detection_threshold: float = 0.8
    face_detection_backend: str = "retinaface"  # "retinaface" or "mtcnn"
    max_faces: int = 1                  # Max faces to track per video
    face_padding: float = 0.3           # Padding ratio around detected face

    # Mouth ROI
    mouth_roi_size: Tuple[int, int] = (96, 96)
    mouth_padding: float = 0.4          # Padding ratio around mouth region

    # Audio
    audio_sample_rate: int = 16000
    audio_n_fft: int = 400
    audio_hop_length: int = 160
    audio_max_duration: float = 15.0    # Max audio duration in seconds

    # Synchronization
    sync_window_ms: int = 40            # Audio window per frame (ms)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_epochs: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Loss weights (multi-task)
    lambda_cls: float = 1.0             # Classification loss
    lambda_lip: float = 0.5             # Lip sync loss
    lambda_id: float = 0.5              # Identity loss
    lambda_temp: float = 0.5            # Temporal loss
    lambda_sync: float = 0.5            # AV sync loss
    lambda_boundary: float = 1.0        # TFBD CRF loss

    # Scheduler
    scheduler: str = "cosine"           # "cosine" or "step"
    step_size: int = 10
    step_gamma: float = 0.5

    # Mixed precision
    use_amp: bool = True

    # Early stopping
    patience: int = 7
    min_delta: float = 1e-4

    # Data
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class InferenceConfig:
    """Inference pipeline parameters."""

    confidence_threshold: float = 0.5
    batch_size: int = 8
    device: str = "auto"                # "auto", "cuda", "cpu"
    use_fp16: bool = True               # Use half precision for inference

    # Heatmap generation
    heatmap_colormap: str = "jet"
    heatmap_alpha: float = 0.4          # Overlay transparency
    heatmap_fps: int = 25

    # Report
    generate_html_report: bool = True
    generate_json_report: bool = True
    report_heatmap_frames: int = 10     # Number of key heatmap frames in report


@dataclass
class PathConfig:
    """File system paths."""

    # Project root
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)

    # Model checkpoints
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "checkpoints")
    best_model_path: Optional[Path] = None

    # Outputs
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    report_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output" / "reports")
    heatmap_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output" / "heatmaps")
    debug_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output" / "debug")

    # Dataset roots (to be configured by user)
    faceforensics_root: Optional[Path] = None
    dfdc_root: Optional[Path] = None
    celebdf_root: Optional[Path] = None
    fakeavceleb_root: Optional[Path] = None
    forgerynet_root: Optional[Path] = None

    # Report templates
    template_dir: Path = field(default_factory=lambda: Path(__file__).parent / "reports" / "templates")

    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for dir_path in [
            self.checkpoint_dir,
            self.output_dir,
            self.report_dir,
            self.heatmap_dir,
            self.debug_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Tag Constants for TFBD
# ──────────────────────────────────────────────
TFBD_TAGS = {
    "REAL": 0,
    "FAKE": 1,
    "BOUNDARY": 2,
}
TFBD_TAG_NAMES = {v: k for k, v in TFBD_TAGS.items()}


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device string to a torch.device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ──────────────────────────────────────────────
# Default instances
# ──────────────────────────────────────────────
model_config = ModelConfig()
preprocess_config = PreprocessConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
path_config = PathConfig()
