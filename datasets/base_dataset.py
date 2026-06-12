"""
Base dataset class for deepfake detection.

Provides a common interface for loading video samples with labels,
preprocessing on-the-fly or from cache, and yielding model-ready
(audio, face_frames, mouth_rois, label, metadata) tuples.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from config import PreprocessConfig, preprocess_config
from preprocessing.pipeline import PreprocessingPipeline, PreprocessedData


@dataclass
class SampleMetadata:
    """Metadata for a single dataset sample."""

    video_path: str = ""
    label: int = 0               # 0=REAL, 1=FAKE
    dataset_name: str = ""
    manipulation_type: str = ""
    split: str = ""              # "train", "val", "test"
    original_video: str = ""     # Path to original for fake samples


class BaseDeepfakeDataset(Dataset, ABC):
    """
    Abstract base class for deepfake detection datasets.

    Subclasses must implement:
      - _load_samples(): Parse dataset directory and populate self.samples
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        config: Optional[PreprocessConfig] = None,
        max_samples: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Root directory of the dataset.
            split: Dataset split ("train", "val", "test").
            config: Preprocessing configuration.
            max_samples: Maximum number of samples to load.
            use_cache: Whether to use pre-cached preprocessed data.
            cache_dir: Directory for cached preprocessed data.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.config = config or preprocess_config
        self.max_samples = max_samples
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Preprocessing pipeline
        self.pipeline = PreprocessingPipeline(config=self.config)

        # Load sample list
        self.samples: List[SampleMetadata] = []
        self._load_samples()

        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]

        logger.info(
            f"{self.__class__.__name__}: loaded {len(self.samples)} samples "
            f"(split={split})"
        )

    @abstractmethod
    def _load_samples(self) -> None:
        """Parse the dataset directory and populate self.samples."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with keys:
              - 'audio': [num_samples] float32 waveform
              - 'face_frames': [T, C, H, W] float32 face crops
              - 'mouth_rois': [T, C, H, W] float32 mouth ROIs
              - 'label': int (0=REAL, 1=FAKE)
              - 'metadata': SampleMetadata dict
        """
        sample = self.samples[idx]

        # Try loading from cache first
        if self.use_cache and self.cache_dir:
            cached = self._load_from_cache(idx)
            if cached is not None:
                return cached

        # Preprocess on-the-fly
        try:
            preprocessed = self.pipeline.process(sample.video_path)
        except Exception as e:
            logger.warning(f"Failed to preprocess {sample.video_path}: {e}")
            return self._get_dummy_sample(sample)

        # Convert to tensors
        result = self._preprocessed_to_tensors(preprocessed, sample)

        # Save to cache if enabled
        if self.use_cache and self.cache_dir:
            self._save_to_cache(idx, result)

        return result

    def _preprocessed_to_tensors(
        self, preprocessed: PreprocessedData, sample: SampleMetadata
    ) -> Dict[str, Any]:
        """Convert PreprocessedData to tensor dict."""
        # Audio
        audio = torch.tensor(preprocessed.audio_waveform, dtype=torch.float32)

        # Face frames: [T, H, W, C] BGR → [T, C, H, W] RGB float
        faces = np.stack(preprocessed.face_crops) if preprocessed.face_crops else np.zeros((1, 224, 224, 3))
        faces = faces[..., ::-1].copy()  # BGR → RGB
        faces = torch.tensor(faces, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        # Mouth ROIs
        mouths = np.stack(preprocessed.mouth_rois) if preprocessed.mouth_rois else np.zeros((1, 96, 96, 3))
        mouths = mouths[..., ::-1].copy()
        mouths = torch.tensor(mouths, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        return {
            "audio": audio,
            "face_frames": faces,
            "mouth_rois": mouths,
            "label": sample.label,
            "metadata": {
                "video_path": sample.video_path,
                "dataset_name": sample.dataset_name,
                "manipulation_type": sample.manipulation_type,
                "split": sample.split,
            },
        }

    def _get_dummy_sample(self, sample: SampleMetadata) -> Dict[str, Any]:
        """Return a dummy sample when preprocessing fails."""
        return {
            "audio": torch.zeros(16000, dtype=torch.float32),
            "face_frames": torch.zeros(1, 3, 224, 224, dtype=torch.float32),
            "mouth_rois": torch.zeros(1, 3, 96, 96, dtype=torch.float32),
            "label": sample.label,
            "metadata": {
                "video_path": sample.video_path,
                "dataset_name": sample.dataset_name,
                "error": True,
            },
        }

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, Any]]:
        """Try loading a preprocessed sample from cache."""
        cache_path = self.cache_dir / f"sample_{idx:06d}.pt"
        if cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception:
                return None
        return None

    def _save_to_cache(self, idx: int, data: Dict[str, Any]) -> None:
        """Save a preprocessed sample to cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"sample_{idx:06d}.pt"
            torch.save(data, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the count of REAL vs FAKE samples."""
        real_count = sum(1 for s in self.samples if s.label == 0)
        fake_count = sum(1 for s in self.samples if s.label == 1)
        return {"real": real_count, "fake": fake_count, "total": len(self.samples)}
