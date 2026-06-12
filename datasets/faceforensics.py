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
    ):
        """
        Args:
            root_dir: Path to FaceForensics++ root.
            split: Dataset split.
            compression: Compression level ('c0', 'c23', 'c40').
            manipulation_types: Which types to include (None = all).
            config: Preprocessing config.
            max_samples: Max samples.
        """
        self.compression = compression
        self.manipulation_types = manipulation_types or self.MANIPULATION_TYPES
        super().__init__(root_dir, split, config, max_samples)

    def _load_samples(self) -> None:
        """Parse FF++ directory structure."""
        # Load split file
        split_file = self.root_dir / "splits" / f"{self.split}.json"
        split_ids = set()
        if split_file.exists():
            import json
            with open(split_file) as f:
                pairs = json.load(f)
                for pair in pairs:
                    split_ids.update(str(x) for x in pair)

        # Real videos
        real_dir = (
            self.root_dir / "original_sequences" / "youtube"
            / self.compression / "videos"
        )
        if real_dir.exists():
            for video_path in sorted(real_dir.glob("*.mp4")):
                vid_id = video_path.stem
                if split_ids and vid_id not in split_ids:
                    continue
                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=0,
                    dataset_name="faceforensics",
                    manipulation_type="original",
                    split=self.split,
                ))

        # Fake videos
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
                    video_path=str(video_path),
                    label=1,
                    dataset_name="faceforensics",
                    manipulation_type=manip_type,
                    split=self.split,
                ))
