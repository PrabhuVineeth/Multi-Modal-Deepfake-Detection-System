"""
FakeAVCeleb dataset adapter.

Multimodal deepfake dataset with face-swap, lip-sync, and
combined audio-visual forgeries.
"""

import csv
from pathlib import Path
from typing import Optional

from loguru import logger

from config import PreprocessConfig
from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata


class FakeAVCelebDataset(BaseDeepfakeDataset):
    """
    FakeAVCeleb dataset loader.

    Expected structure:
        root_dir/
        ├── RealVideo-RealAudio/
        │   └── *.mp4
        ├── FakeVideo-RealAudio/
        │   └── *.mp4
        ├── RealVideo-FakeAudio/
        │   └── *.mp4
        ├── FakeVideo-FakeAudio/
        │   └── *.mp4
        └── meta_data.csv
    """

    FORGERY_CATEGORIES = {
        "RealVideo-RealAudio": ("original", 0),
        "FakeVideo-RealAudio": ("face-swap", 1),
        "RealVideo-FakeAudio": ("lip-sync", 1),
        "FakeVideo-FakeAudio": ("both", 1),
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        forgery_types: Optional[list] = None,
        config: Optional[PreprocessConfig] = None,
        max_samples: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.forgery_types = forgery_types  # Filter by forgery category
        super().__init__(root_dir, split, config, max_samples, use_cache, cache_dir)

    def _load_samples(self) -> None:
        """Parse FakeAVCeleb directory structure and partition splits deterministically and randomly."""
        # Try metadata CSV first
        meta_file = self.root_dir / "meta_data.csv"
        if meta_file.exists():
            self._load_from_csv(meta_file)
        else:
            self._load_from_dirs()

        # Sort samples first to ensure baseline consistency across environments
        self.samples.sort(key=lambda s: s.video_path)

        # Deterministic shuffle using a fixed random seed
        import random
        rng = random.Random(42)
        rng.shuffle(self.samples)

        # Partition the samples deterministically into train/val/test splits
        # Use ratios: 80% train, 10% val, 10% test
        num_samples = len(self.samples)
        train_end = int(num_samples * 0.8)
        val_end = train_end + int(num_samples * 0.1)

        if self.split == "train":
            self.samples = self.samples[:train_end]
        elif self.split == "val":
            self.samples = self.samples[train_end:val_end]
        elif self.split == "test":
            self.samples = self.samples[val_end:]

        # Update the split attribute on the metadata objects to match the partitioned split
        for s in self.samples:
            s.split = self.split

    def _load_from_csv(self, meta_file: Path) -> None:
        """Load samples from metadata CSV."""
        with open(meta_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The CSV has an unnamed column ('') with the full relative path
                # e.g. "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076"
                # and 'path' contains just the filename e.g. "00109.mp4"
                full_rel = row.get("") or row.get("full_path", "")
                filename = row.get("path", "")

                if full_rel:
                    # Strip leading "FakeAVCeleb/" prefix if present
                    full_rel = full_rel.replace("FakeAVCeleb/", "", 1)
                    video_path = self.root_dir / full_rel / filename
                else:
                    # Fallback: type-based subdirectory + filename
                    type_dir = row.get("type", "")
                    video_path = self.root_dir / type_dir / filename

                if not video_path.exists():
                    continue

                category = row.get("type", row.get("category", ""))
                manip_type, label = self.FORGERY_CATEGORIES.get(
                    category, ("unknown", 1)
                )

                if self.forgery_types and manip_type not in self.forgery_types:
                    continue

                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=label,
                    dataset_name="fakeavceleb",
                    manipulation_type=manip_type,
                    split=self.split,
                ))


    def _load_from_dirs(self) -> None:
        """Load samples by scanning forgery category directories."""
        for dir_name, (manip_type, label) in self.FORGERY_CATEGORIES.items():
            category_dir = self.root_dir / dir_name
            if not category_dir.exists():
                continue

            if self.forgery_types and manip_type not in self.forgery_types:
                continue

            for video_path in sorted(category_dir.rglob("*.mp4")):
                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=label,
                    dataset_name="fakeavceleb",
                    manipulation_type=manip_type,
                    split=self.split,
                ))
