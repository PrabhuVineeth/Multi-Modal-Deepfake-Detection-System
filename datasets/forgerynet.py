"""
ForgeryNet dataset adapter.

Provides temporal boundary labels for TFBD training with
per-frame REAL/FAKE/BOUNDARY annotations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from config import PreprocessConfig
from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata


class ForgeryNetDataset(BaseDeepfakeDataset):
    """
    ForgeryNet dataset loader.

    Key feature: provides per-frame temporal boundary labels
    (REAL=0, FAKE=1, BOUNDARY=2) for TFBD training.

    Expected structure:
        root_dir/
        ├── videos/
        │   └── *.mp4
        ├── annotations/
        │   ├── train.json
        │   ├── val.json
        │   └── test.json
        └── boundary_labels/
            └── *.json  (per-video frame-level labels)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        config: Optional[PreprocessConfig] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir, split, config, max_samples)

    def _load_samples(self) -> None:
        """Parse ForgeryNet annotations."""
        anno_file = self.root_dir / "annotations" / f"{self.split}.json"

        if anno_file.exists():
            with open(anno_file) as f:
                annotations = json.load(f)

            for entry in annotations:
                video_name = entry.get("video", "")
                video_path = self.root_dir / "videos" / video_name

                if not video_path.exists():
                    continue

                label = entry.get("label", 0)
                manip_type = entry.get("manipulation_type", "unknown")

                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=label,
                    dataset_name="forgerynet",
                    manipulation_type=manip_type,
                    split=self.split,
                ))
        else:
            # Fallback: scan video directory
            video_dir = self.root_dir / "videos"
            if video_dir.exists():
                for video_path in sorted(video_dir.rglob("*.mp4")):
                    self.samples.append(SampleMetadata(
                        video_path=str(video_path),
                        label=0,  # Unknown label
                        dataset_name="forgerynet",
                        split=self.split,
                    ))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Override to include boundary labels when available."""
        result = super().__getitem__(idx)

        # Load boundary labels if available
        sample = self.samples[idx]
        boundary_labels = self._load_boundary_labels(sample.video_path)
        if boundary_labels is not None:
            result["boundary_tags"] = boundary_labels
        else:
            # Generate default: all REAL (0) or all FAKE (1)
            T = result["face_frames"].shape[0]
            result["boundary_tags"] = torch.full(
                (T,), sample.label, dtype=torch.long
            )

        return result

    def _load_boundary_labels(
        self, video_path: str
    ) -> Optional[torch.Tensor]:
        """
        Load per-frame boundary labels for a video.

        Args:
            video_path: Path to the video.

        Returns:
            Tensor of frame-level tags [T], or None if not available.
        """
        video_name = Path(video_path).stem
        label_file = self.root_dir / "boundary_labels" / f"{video_name}.json"

        if not label_file.exists():
            return None

        try:
            with open(label_file) as f:
                data = json.load(f)
            frame_labels = data.get("frame_labels", [])
            return torch.tensor(frame_labels, dtype=torch.long)
        except Exception as e:
            logger.warning(f"Failed to load boundary labels: {e}")
            return None
