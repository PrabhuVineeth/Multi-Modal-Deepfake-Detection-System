"""
FaceForensics++ dataset adapter.

Handles the FF++ directory structure with four manipulation types
(DeepFakes, Face2Face, FaceSwap, NeuralTextures) and three
compression levels (c0, c23, c40).
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from config import PreprocessConfig
from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata


class FaceForensicsDataset(BaseDeepfakeDataset):
    """
    FaceForensics++ dataset loader.

    Expected directory structure:
        root_dir/
        ├── original_sequences/
        │   └── youtube/
        │       └── c23/
        │           └── videos/
        │               ├── 000.mp4
        │               └── ...
        ├── manipulated_sequences/
        │   ├── Deepfakes/
        │   │   └── c23/
        │   │       └── videos/
        │   │           ├── 000_003.mp4
        │   │           └── ...
        │   ├── Face2Face/
        │   ├── FaceSwap/
        │   └── NeuralTextures/
        └── splits/
            ├── train.json
            ├── val.json
            └── test.json
    """

    MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        compression: str = "c23",
        manipulation_types: Optional[list] = None,
        config: Optional[PreprocessConfig] = None,
        max_samples: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Path to FaceForensics++ root.
            split: Dataset split.
            compression: Compression level ('c0', 'c23', 'c40').
            manipulation_types: Which types to include (None = all).
            config: Preprocessing config.
            max_samples: Max samples.
            use_cache: Whether to use pre-cached preprocessed data.
            cache_dir: Directory for cached preprocessed data.
        """
        self.compression = compression
        self.manipulation_types = manipulation_types or self.MANIPULATION_TYPES
        super().__init__(root_dir, split, config, max_samples, use_cache, cache_dir)

    def _load_samples(self) -> None:
        """Parse FF++ directory structure — supports standard and flat Kaggle layout."""
        # Detect layout
        standard_real = (
            self.root_dir / "original_sequences" / "youtube"
            / self.compression / "videos"
        )
        flat_real = self.root_dir / "original"

        if standard_real.exists():
            self._load_standard(standard_real)
        elif flat_real.exists():
            self._load_flat()
        else:
            logger.warning(f"FF++ directory structure not recognised at {self.root_dir}")
            return

        # Exclude corrupt files
        bad_files_path = Path("output/dataset_health/faceforensics_bad_files.txt")
        if bad_files_path.exists():
            try:
                with open(bad_files_path, "r", encoding="utf-8") as f:
                    bad_files = {line.strip() for line in f if line.strip()}
                if bad_files:
                    before_count = len(self.samples)
                    self.samples = [
                        s for s in self.samples
                        if Path(s.video_path).resolve().as_posix() not in {Path(b).resolve().as_posix() for b in bad_files}
                    ]
                    removed = before_count - len(self.samples)
                    if removed > 0:
                        logger.info(f"Integrity check: filtered out {removed} corrupt videos from {self.split} loading.")
            except Exception as e:
                logger.warning(f"Failed to read bad files log: {e}")

        if flat_real.exists():
            # Sort and shuffle deterministically to ensure consistent partitioning
            self.samples.sort(key=lambda s: s.video_path)
            import random
            rng = random.Random(42)
            rng.shuffle(self.samples)
            
            num_samples = len(self.samples)
            train_end = int(num_samples * 0.8)
            val_end = train_end + int(num_samples * 0.1)
            
            if self.split == "train":
                self.samples = self.samples[:train_end]
            elif self.split == "val":
                self.samples = self.samples[train_end:val_end]
            elif self.split == "test":
                self.samples = self.samples[val_end:]
                
            for s in self.samples:
                s.split = self.split


    def _load_standard(self, real_dir: Path) -> None:
        """Load from official FF++ hierarchy: original_sequences/youtube/c23/videos/."""
        split_ids = self._read_split_ids()

        for video_path in sorted(real_dir.glob("*.mp4")):
            if split_ids and video_path.stem not in split_ids:
                continue
            self.samples.append(SampleMetadata(
                video_path=str(video_path), label=0,
                dataset_name="faceforensics", manipulation_type="original",
                split=self.split,
            ))

        for manip_type in self.manipulation_types:
            fake_dir = (
                self.root_dir / "manipulated_sequences" / manip_type
                / self.compression / "videos"
            )
            if not fake_dir.exists():
                logger.warning(f"FF++ directory not found: {fake_dir}")
                continue
            for video_path in sorted(fake_dir.glob("*.mp4")):
                vid_id = video_path.stem.split("_")[0]
                if split_ids and vid_id not in split_ids:
                    continue
                self.samples.append(SampleMetadata(
                    video_path=str(video_path), label=1,
                    dataset_name="faceforensics", manipulation_type=manip_type,
                    split=self.split,
                ))

    def _load_flat(self) -> None:
        """Load from flat Kaggle layout: original/, Deepfakes/, Face2Face/, etc. under root."""
        # Real videos
        real_dir = self.root_dir / "original"
        for video_path in sorted(real_dir.rglob("*.mp4")):
            self.samples.append(SampleMetadata(
                video_path=str(video_path), label=0,
                dataset_name="faceforensics", manipulation_type="original",
                split=self.split,
            ))

        # Fake videos — map folder names to manipulation types
        folder_map = {
            "Deepfakes": "Deepfakes",
            "Face2Face": "Face2Face",
            "FaceSwap": "FaceSwap",
            "NeuralTextures": "NeuralTextures",
            "FaceShifter": "FaceShifter",
            "DeepFakeDetection": "DeepFakeDetection",
        }
        for folder, manip_type in folder_map.items():
            if manip_type not in self.manipulation_types and manip_type not in ["FaceShifter", "DeepFakeDetection"]:
                continue
            fake_dir = self.root_dir / folder
            if not fake_dir.exists():
                continue
            for video_path in sorted(fake_dir.rglob("*.mp4")):
                self.samples.append(SampleMetadata(
                    video_path=str(video_path), label=1,
                    dataset_name="faceforensics", manipulation_type=manip_type,
                    split=self.split,
                ))

    def _read_split_ids(self) -> set:
        """Read train/val/test split IDs from splits JSON if present."""
        split_file = self.root_dir / "splits" / f"{self.split}.json"
        if not split_file.exists():
            return set()
        import json
        with open(split_file) as f:
            pairs = json.load(f)
        ids = set()
        for pair in pairs:
            ids.update(str(x) for x in pair)
        return ids

