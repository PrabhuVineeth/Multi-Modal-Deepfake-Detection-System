"""
LAV-DF dataset adapter.

Supports audiovisual deepfake samples with optional temporal forgery
segments used by the TFBD boundary detector.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from loguru import logger

from config import PreprocessConfig, TFBD_TAGS
from datasets.base_dataset import BaseDeepfakeDataset, SampleMetadata


class LAVDFDataset(BaseDeepfakeDataset):
    """
    LAV-DF dataset loader.

    Expected flexible structure:
        root_dir/
        ├── videos/ or data/
        │   └── *.mp4/*.avi/*.mov
        └── annotations/
            ├── train.json or train.csv
            ├── val.json or val.csv
            └── test.json or test.csv

    Annotation rows may contain:
        video/path/file, label/is_fake, manipulation_type,
        forgery_start/start, forgery_end/end, or segments/fake_periods.
    """

    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        config: Optional[PreprocessConfig] = None,
        max_samples: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.boundary_segments: Dict[str, List[Tuple[float, float]]] = {}
        super().__init__(root_dir, split, config, max_samples, use_cache, cache_dir)

    def _load_samples(self) -> None:
        if self.split == "all":
            splits_to_load = ["train", "val", "test"]
        else:
            splits_to_load = [self.split]
            
        loaded_any = False
        for s in splits_to_load:
            anno_base = self.root_dir / "annotations" / s
            if anno_base.with_suffix(".json").exists():
                self._load_from_json(anno_base.with_suffix(".json"))
                loaded_any = True
            elif anno_base.with_suffix(".csv").exists():
                self._load_from_csv(anno_base.with_suffix(".csv"))
                loaded_any = True
                
        if not loaded_any:
            self._load_from_dirs()

    def _load_from_json(self, anno_file: Path) -> None:
        with open(anno_file, encoding="utf-8") as f:
            data = json.load(f)
        entries = data.values() if isinstance(data, dict) else data
        for entry in entries:
            if isinstance(entry, dict):
                self._add_entry(entry)

    def _load_from_csv(self, anno_file: Path) -> None:
        with open(anno_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._add_entry(row)

    def _load_from_dirs(self) -> None:
        for video_dir in [self.root_dir / "videos", self.root_dir / "data", self.root_dir]:
            if not video_dir.exists():
                continue
            for video_path in self._iter_videos(video_dir):
                label = 1 if "fake" in video_path.as_posix().lower() else 0
                self.samples.append(SampleMetadata(
                    video_path=str(video_path),
                    label=label,
                    dataset_name="lavdf",
                    manipulation_type="audiovisual" if label else "original",
                    split=self.split,
                ))
            if self.samples:
                return

    def _add_entry(self, entry: Dict[str, Any]) -> None:
        rel_path = (
            entry.get("video") or entry.get("path") or entry.get("file")
            or entry.get("filename") or entry.get("video_path") or ""
        )
        video_path = self._resolve_video_path(str(rel_path))
        if video_path is None:
            logger.warning(f"LAV-DF video not found: {rel_path}")
            return

        label = self._parse_label(entry)
        segments = self._parse_segments(entry)
        if segments:
            self.boundary_segments[str(video_path)] = segments

        start = segments[0][0] if segments else None
        end = segments[-1][1] if segments else None
        self.samples.append(SampleMetadata(
            video_path=str(video_path),
            label=label,
            dataset_name="lavdf",
            manipulation_type=entry.get("manipulation_type", "audiovisual" if label else "original"),
            split=self.split,
            forgery_start=start,
            forgery_end=end,
        ))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = super().__getitem__(idx)
        sample = self.samples[idx]
        segments = self.boundary_segments.get(sample.video_path)
        if segments:
            frame_count = result["face_frames"].shape[0]
            result["boundary_tags"] = self._segments_to_tags(segments, frame_count)
        return result

    def _resolve_video_path(self, rel_path: str) -> Optional[Path]:
        candidates = [
            self.root_dir / rel_path,
            self.root_dir / "videos" / rel_path,
            self.root_dir / "data" / rel_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        stem = Path(rel_path).stem
        for video_path in self._iter_videos(self.root_dir):
            if video_path.stem == stem:
                return video_path
        return None

    def _iter_videos(self, root: Path) -> Iterable[Path]:
        for ext in self.VIDEO_EXTENSIONS:
            yield from sorted(root.rglob(f"*{ext}"))

    def _parse_label(self, entry: Dict[str, Any]) -> int:
        raw = entry.get("label", entry.get("is_fake", entry.get("fake", 0)))
        if isinstance(raw, str):
            return 1 if raw.strip().lower() in {"1", "fake", "true", "yes"} else 0
        return int(bool(raw))

    def _parse_segments(self, entry: Dict[str, Any]) -> List[Tuple[float, float]]:
        raw_segments = entry.get("segments") or entry.get("fake_periods") or entry.get("forgery_segments")
        if isinstance(raw_segments, str):
            try:
                raw_segments = json.loads(raw_segments)
            except json.JSONDecodeError:
                raw_segments = None

        segments: List[Tuple[float, float]] = []
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if isinstance(seg, dict):
                    start, end = seg.get("start"), seg.get("end")
                else:
                    start, end = seg[0], seg[1]
                segments.append((float(start), float(end)))

        start = entry.get("forgery_start", entry.get("start"))
        end = entry.get("forgery_end", entry.get("end"))
        if start not in (None, "") and end not in (None, ""):
            segments.append((float(start), float(end)))
        return segments

    def _segments_to_tags(self, segments: List[Tuple[float, float]], frame_count: int) -> torch.Tensor:
        tags = torch.full((frame_count,), TFBD_TAGS["REAL"], dtype=torch.long)
        fps = max(self.config.target_fps, 1)
        for start, end in segments:
            start_idx = max(int(start * fps), 0)
            end_idx = min(int(end * fps), frame_count - 1)
            if start_idx <= end_idx:
                tags[start_idx:end_idx + 1] = TFBD_TAGS["FAKE"]
                tags[start_idx] = TFBD_TAGS["BOUNDARY"]
                tags[end_idx] = TFBD_TAGS["BOUNDARY"]
        return tags
