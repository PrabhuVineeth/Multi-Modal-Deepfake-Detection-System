"""
DFDC (Deepfake Detection Challenge) dataset adapter.

Handles the chunk-based DFDC structure with metadata.json manifests.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from config import PreprocessConfig
from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata


class DFDCDataset(BaseDeepfakeDataset):
    """
    DFDC dataset loader.

    Expected structure:
        root_dir/
        ├── dfdc_train_part_0/
        │   ├── metadata.json
        │   ├── aagfhgtpmv.mp4
        │   └── ...
        ├── dfdc_train_part_1/
        └── ...
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_chunks: Optional[int] = None,
        config: Optional[PreprocessConfig] = None,
        max_samples: Optional[int] = None,
    ):
        self.num_chunks = num_chunks
        super().__init__(root_dir, split, config, max_samples)

    def _load_samples(self) -> None:
        """Parse DFDC chunk directories with metadata.json."""
        chunk_dirs = sorted(self.root_dir.glob("dfdc_train_part_*"))

        if self.num_chunks:
            chunk_dirs = chunk_dirs[:self.num_chunks]

        for chunk_dir in chunk_dirs:
            meta_file = chunk_dir / "metadata.json"
            if not meta_file.exists():
                logger.warning(f"No metadata.json in {chunk_dir}")
                continue

            with open(meta_file) as f:
                metadata = json.load(f)

            for filename, info in metadata.items():
                video_path = chunk_dir / filename
                if not video_path.exists():
                    continue

                label = 1 if info.get("label") == "FAKE" else 0
                original = info.get("original", "")

                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=label,
                    dataset_name="dfdc",
                    manipulation_type="dfdc",
                    split=self.split,
                    original_video=original,
                ))
