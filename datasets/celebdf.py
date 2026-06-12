"""
Celeb-DF v2 dataset adapter.

Handles Celeb-real, Celeb-synthesis, and YouTube-real splits.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from config import PreprocessConfig
from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata


class CelebDFDataset(BaseDeepfakeDataset):
    """
    Celeb-DF v2 dataset loader.

    Expected structure:
        root_dir/
        ├── Celeb-real/
        │   ├── id0_0000.mp4
        │   └── ...
        ├── Celeb-synthesis/
        │   ├── id0_id1_0000.mp4
        │   └── ...
        ├── YouTube-real/
        │   ├── 00000.mp4
        │   └── ...
        └── List_of_testing_videos.txt
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
        """Parse Celeb-DF directory structure."""
        # Load test split list
        test_file = self.root_dir / "List_of_testing_videos.txt"
        test_videos = set()
        if test_file.exists():
            with open(test_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            test_videos.add(parts[1].strip())

        # Real videos
        for real_dir_name in ["Celeb-real", "YouTube-real"]:
            real_dir = self.root_dir / real_dir_name
            if not real_dir.exists():
                continue

            for video_path in sorted(real_dir.glob("*.mp4")):
                rel_path = f"{real_dir_name}/{video_path.name}"
                is_test = rel_path in test_videos

                if self.split == "test" and not is_test:
                    continue
                if self.split != "test" and is_test:
                    continue

                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=0,
                    dataset_name="celebdf",
                    manipulation_type="original",
                    split=self.split,
                ))

        # Fake videos
        synth_dir = self.root_dir / "Celeb-synthesis"
        if synth_dir.exists():
            for video_path in sorted(synth_dir.glob("*.mp4")):
                rel_path = f"Celeb-synthesis/{video_path.name}"
                is_test = rel_path in test_videos

                if self.split == "test" and not is_test:
                    continue
                if self.split != "test" and is_test:
                    continue

                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=1,
                    dataset_name="celebdf",
                    manipulation_type="celeb-synthesis",
                    split=self.split,
                ))
