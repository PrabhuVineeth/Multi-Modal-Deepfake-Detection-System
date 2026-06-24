"""
I/O utility functions for the Deepfake Forensic Detection System.

Handles: checkpoint saving/loading, video metadata reading, FFmpeg checks,
and JSON serialization of forensic results.
"""

import json
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger


# ──────────────────────────────────────────────
# FFmpeg
# ──────────────────────────────────────────────

def ensure_ffmpeg() -> str:
    """
    Verify FFmpeg is installed and return its path.
    Falls back to the imageio-ffmpeg bundled binary if not on PATH.

    Returns:
        Path to the ffmpeg executable.

    Raises:
        RuntimeError: If FFmpeg is not found anywhere.
    """
    import os
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        # Try imageio-ffmpeg bundled binary
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            # Add its directory to PATH so subprocess calls also work
            ffmpeg_dir = str(Path(ffmpeg_path).parent)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            logger.debug(f"FFmpeg found via imageio-ffmpeg: {ffmpeg_path}")
        except Exception:
            raise RuntimeError(
                "FFmpeg not found. Install it via:\n"
                "  pip install imageio-ffmpeg   (easiest)\n"
                "  Windows: download from https://ffmpeg.org/download.html\n"
                "  Linux:   sudo apt install ffmpeg"
            )
    logger.debug(f"FFmpeg found at: {ffmpeg_path}")
    return ffmpeg_path



def read_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Read video metadata using OpenCV (as primary/fallback) and FFprobe (if available).

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with keys: duration, fps, width, height, has_audio, codec.
    """
    video_path = str(video_path)
    metadata: Dict[str, Any] = {
        "duration": 0.0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "has_audio": False,
        "video_codec": "unknown",
        "audio_codec": "unknown",
    }

    # Use OpenCV as robust cross-platform fallback for visual streams
    import cv2
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata["fps"] = fps
        metadata["width"] = width
        metadata["height"] = height
        if fps > 0:
            metadata["duration"] = frame_count / fps
        cap.release()

    # Use FFprobe to check for audio streams if ffprobe is installed globally
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is not None:
        try:
            cmd = [
                ffprobe_path, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            fmt = probe_data.get("format", {})
            if "duration" in fmt:
                metadata["duration"] = float(fmt.get("duration", 0))
            
            for stream in probe_data.get("streams", []):
                codec_type = stream.get("codec_type", "")
                if codec_type == "video":
                    metadata["width"] = int(stream.get("width", 0))
                    metadata["height"] = int(stream.get("height", 0))
                    metadata["video_codec"] = stream.get("codec_name", "unknown")
                    fps_str = stream.get("avg_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        metadata["fps"] = float(num) / max(float(den), 1)
                elif codec_type == "audio":
                    metadata["has_audio"] = True
                    metadata["audio_codec"] = stream.get("codec_name", "unknown")
        except Exception as e:
            logger.debug(f"ffprobe execution failed, relying on OpenCV: {e}")
            
    # Default to has_audio = True if ffprobe not available, since all training clips have audio streams
    if ffprobe_path is None:
        metadata["has_audio"] = True

    return metadata


# ──────────────────────────────────────────────
# Checkpoints
# ──────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: Optimizer state (optional).
        epoch: Current epoch number.
        metrics: Dictionary of current metrics (loss, auc, etc.).
        path: File path to save the checkpoint.
        scheduler: Learning rate scheduler state (optional).
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Tuple[int, Dict[str, float]]:
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state (optional).
        scheduler: Scheduler to restore state (optional).
        device: Device to map tensors to.

    Returns:
        Tuple of (epoch, metrics_dict).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Force load of lazy-initialized layers if present before loading state_dict
    if hasattr(model, "audio_encoder") and hasattr(model.audio_encoder, "_load_backbone"):
        model.audio_encoder._load_backbone()
    if hasattr(model, "video_encoder") and hasattr(model.video_encoder, "_load_backbone"):
        model.video_encoder._load_backbone()
    if hasattr(model, "mouth_encoder") and hasattr(model.mouth_encoder, "encoder") and hasattr(model.mouth_encoder.encoder, "_load_backbone"):
        model.mouth_encoder.encoder._load_backbone()
    if hasattr(model, "tfbd") and hasattr(model.tfbd, "_init_crf"):
        model.tfbd._init_crf(device=torch.device(device))

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    logger.info(f"Checkpoint loaded: {path} (epoch {epoch})")
    return epoch, metrics


# ──────────────────────────────────────────────
# JSON Serialization
# ──────────────────────────────────────────────

class ForensicJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays, torch tensors, and dataclasses."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file using the forensic encoder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=ForensicJSONEncoder, indent=indent)
    logger.debug(f"JSON saved: {path}")


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
