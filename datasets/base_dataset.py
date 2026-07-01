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
    forgery_start: Optional[float] = None
    forgery_end: Optional[float] = None


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

        # Preprocessing pipeline (lazy init to avoid worker VRAM bloat)
        self._pipeline = None

        # Pre-load manifest in memory once at startup
        self._manifest_samples = {}
        if self.use_cache and self.cache_dir:
            raw = self._load_manifest()
            self._manifest_samples = raw.get("samples", {})
            logger.info(f"{self.__class__.__name__}: Preloaded manifest cache with {len(self._manifest_samples)} entries.")

        # Load sample list
        self.samples: List[SampleMetadata] = []
        self._load_samples()

        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]

        logger.info(
            f"{self.__class__.__name__}: loaded {len(self.samples)} samples "
            f"(split={split})"
        )

    @property
    def pipeline(self) -> PreprocessingPipeline:
        """Get or initialize the preprocessing pipeline lazily."""
        if self._pipeline is None:
            self._pipeline = PreprocessingPipeline(config=self.config)
        return self._pipeline

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
            cached = self._load_from_cache(sample.video_path)
            if cached is not None:
                # Pad/truncate audio & video sequences dynamically to config lengths
                max_frames = self.config.max_frames
                audio_len = int(self.config.audio_sample_rate * self.config.audio_max_duration)
                
                cached["audio"] = self._fix_len_1d(cached["audio"], audio_len)
                cached["face_frames"] = self._fix_len_seq(cached["face_frames"], max_frames)
                cached["mouth_rois"] = self._fix_len_seq(cached["mouth_rois"], max_frames)
                
                cached["metadata"] = {
                    "video_path": sample.video_path,
                    "dataset_name": sample.dataset_name,
                    "manipulation_type": sample.manipulation_type,
                    "split": sample.split,
                    "forgery_start": sample.forgery_start,
                    "forgery_end": sample.forgery_end,
                }
                return cached

        # Preprocess on-the-fly
        try:
            preprocessed = self.pipeline.process(sample.video_path)
        except Exception as e:
            logger.warning(f"Failed to preprocess {sample.video_path}: {e}")
            return None

        # Convert to raw unpadded tensors
        result = self._preprocessed_to_tensors(preprocessed, sample)

        # Save to cache if enabled (it saves the unpadded uint8/int16 tensors to disk)
        if self.use_cache and self.cache_dir:
            self._save_to_cache(sample.video_path, result)

        # Pad/truncate audio & video sequences dynamically for model training
        max_frames = self.config.max_frames
        audio_len = int(self.config.audio_sample_rate * self.config.audio_max_duration)
        
        result["audio"] = self._fix_len_1d(result["audio"], audio_len)
        result["face_frames"] = self._fix_len_seq(result["face_frames"], max_frames)
        result["mouth_rois"] = self._fix_len_seq(result["mouth_rois"], max_frames)

        return result

    def _preprocessed_to_tensors(
        self, preprocessed: PreprocessedData, sample: SampleMetadata
    ) -> Dict[str, Any]:
        """Convert PreprocessedData to raw unpadded tensor dict."""
        # Audio — unpadded
        audio = torch.tensor(preprocessed.audio_waveform, dtype=torch.float32)

        # Face frames: [T, H, W, C] BGR → [T, C, H, W] RGB float, unpadded
        faces = np.stack(preprocessed.face_crops) if preprocessed.face_crops else np.zeros((1, 224, 224, 3))
        faces = faces[..., ::-1].copy()  # BGR → RGB
        faces = torch.tensor(faces, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        # Mouth ROIs — unpadded
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
                "forgery_start": sample.forgery_start,
                "forgery_end": sample.forgery_end,
            },
        }

    @staticmethod
    def _fix_len_seq(t: torch.Tensor, n: int) -> torch.Tensor:
        """Truncate or zero-pad [T, ...] tensor to exactly n frames."""
        if t.shape[0] >= n:
            return t[:n]
        pad = torch.zeros((n - t.shape[0], *t.shape[1:]), dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    @staticmethod
    def _fix_len_1d(t: torch.Tensor, n: int) -> torch.Tensor:
        """Truncate or zero-pad 1-D audio tensor to exactly n samples."""
        if t.shape[0] >= n:
            return t[:n]
        return torch.nn.functional.pad(t, (0, n - t.shape[0]))

    def _get_dummy_sample(self, sample: SampleMetadata) -> Dict[str, Any]:
        """Return a dummy sample (fixed lengths) when preprocessing fails."""
        max_frames = self.config.max_frames
        audio_len  = int(self.config.audio_sample_rate * self.config.audio_max_duration)
        return {
            "audio": torch.zeros(audio_len, dtype=torch.float32),
            "face_frames": torch.zeros(max_frames, 3, 224, 224, dtype=torch.float32),
            "mouth_rois": torch.zeros(max_frames, 3, 96, 96, dtype=torch.float32),
            "label": sample.label,
            "metadata": {
                "video_path": sample.video_path,
                "dataset_name": sample.dataset_name,
                "error": True,
            },
        }

    def _get_cache_key(self, video_path: str) -> str:
        """Generate a unique SHA256 hash for the video path to use as a cache key."""
        import hashlib
        path_str = Path(video_path).resolve().as_posix()
        # Ensure Windows drive letter is always uppercase for hash consistency
        if len(path_str) > 1 and path_str[1] == ":" and path_str[0].islower():
            path_str = path_str[0].upper() + path_str[1:]
        return hashlib.sha256(path_str.encode("utf-8")).hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load cache manifest file."""
        if not self.cache_dir:
            return {}
        manifest_path = self.cache_dir / "cache_manifest.json"
        if manifest_path.exists():
            import json
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache manifest: {e}")
        return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save cache manifest file."""
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / "cache_manifest.json"
        import json
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache manifest: {e}")

    @staticmethod
    def _get_sha256_checksum(file_path: Path) -> str:
        """Calculate the SHA256 checksum of a file."""
        import hashlib
        sha = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception:
            return ""

    def validate_cache(self, check_checksums: bool = False) -> Dict[str, Any]:
        """
        Validate all cache entries on disk against the manifest.
        Deletes corrupted cache files and updates the manifest.
        """
        if not self.cache_dir:
            return {"status": "no_cache_dir", "healthy": True}
            
        manifest = self._load_manifest()
        samples = manifest.get("samples", {})
        
        healthy_count = 0
        corrupt_count = 0
        missing_count = 0
        corrupt_keys = []
        
        logger.info(f"Validating {len(samples)} cache entries in {self.cache_dir}...")
        
        for key, info in list(samples.items()):
            cache_file = self.cache_dir / info.get("cache_file", "")
            if not cache_file.exists():
                missing_count += 1
                corrupt_keys.append(key)
                continue
                
            # Verify file size
            expected_size = info.get("file_size", 0)
            actual_size = cache_file.stat().st_size
            
            is_corrupt = False
            if expected_size > 0 and actual_size != expected_size:
                is_corrupt = True
            elif check_checksums:
                actual_checksum = self._get_sha256_checksum(cache_file)
                if actual_checksum != info.get("sha256_checksum", ""):
                    is_corrupt = True
                    
            if is_corrupt:
                corrupt_count += 1
                corrupt_keys.append(key)
                try:
                    cache_file.unlink()
                    logger.warning(f"Deleted corrupt cache file: {cache_file}")
                except Exception as e:
                    logger.error(f"Failed to delete corrupt cache file {cache_file}: {e}")
            else:
                healthy_count += 1
                
        # Clean manifest entries for missing/corrupt items
        if corrupt_keys:
            for key in corrupt_keys:
                samples.pop(key, None)
            manifest["samples"] = samples
            
            stats = manifest.get("dataset_statistics", {})
            stats["total_samples"] = len(samples)
            # Recompute total size
            stats["total_size_bytes"] = sum(
                (self.cache_dir / info["cache_file"]).stat().st_size 
                for info in samples.values() 
                if (self.cache_dir / info["cache_file"]).exists()
            )
            manifest["dataset_statistics"] = stats
            self._save_manifest(manifest)
            
        report = {
            "status": "completed",
            "total_manifest_entries": len(samples) + len(corrupt_keys),
            "healthy": healthy_count,
            "corrupt_deleted": corrupt_count,
            "missing_removed": missing_count,
            "is_fully_healthy": corrupt_count == 0 and missing_count == 0,
        }
        logger.info(f"Cache validation report: {report}")
        return report

    def _load_from_cache(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Try loading a preprocessed sample from cache, converting visual/audio back to float32."""
        if not self.cache_dir:
            return None
        key = self._get_cache_key(video_path)
        cache_path = self.cache_dir / f"{key}.pt"
        
        # Verify in manifest
        sample_info = self._manifest_samples.get(key)
        if not sample_info:
            if cache_path.exists() and cache_path.stat().st_size > 10000:
                sample_info = {"file_size": cache_path.stat().st_size}
            else:
                return None
            
        if cache_path.exists():
            try:
                # Basic size verification to prevent loading corrupted files
                expected_size = sample_info.get("file_size", 0)
                if expected_size > 0 and cache_path.stat().st_size != expected_size:
                    logger.warning(f"Cache size mismatch for {video_path}. Removing corrupt file.")
                    try:
                        cache_path.unlink()
                    except Exception:
                        pass
                    return None
                    
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.load(cache_path, map_location="cpu", weights_only=False)
                
                # Convert visual tensors from uint8 to float32 [0, 1] range
                if "face_frames" in data and data["face_frames"].dtype == torch.uint8:
                    data["face_frames"] = data["face_frames"].to(torch.float32) / 255.0
                if "mouth_rois" in data and data["mouth_rois"].dtype == torch.uint8:
                    data["mouth_rois"] = data["mouth_rois"].to(torch.float32) / 255.0
                    
                # Convert audio waveform from int16 (PCM16) back to float32 [-1.0, 1.0] range
                if "audio" in data and data["audio"].dtype == torch.int16:
                    data["audio"] = data["audio"].to(torch.float32) / 32767.0
                    
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {video_path}: {e}")
                return None
        return None

    def _save_to_cache(self, video_path: str, data: Dict[str, Any], update_manifest: bool = False) -> None:
        """Save a preprocessed sample to cache, converting visual to uint8, audio to PCM16, and updating manifest."""
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._get_cache_key(video_path)
            cache_path = self.cache_dir / f"{key}.pt"
            
            # Make a copy to avoid mutating data in memory during training
            save_data = {k: v for k, v in data.items() if k != "metadata"}
            save_data["metadata"] = data["metadata"]
            
            # Cast visual tensors to uint8 to save 75% of space and speed up disk reading
            if isinstance(save_data.get("face_frames"), torch.Tensor):
                if save_data["face_frames"].dtype != torch.uint8:
                    save_data["face_frames"] = (save_data["face_frames"] * 255.0).to(torch.uint8)
            if isinstance(save_data.get("mouth_rois"), torch.Tensor):
                if save_data["mouth_rois"].dtype != torch.uint8:
                    save_data["mouth_rois"] = (save_data["mouth_rois"] * 255.0).to(torch.uint8)
                    
            # Cast audio waveform to int16 (PCM16 representation) to save 50% space
            if isinstance(save_data.get("audio"), torch.Tensor):
                if save_data["audio"].dtype != torch.int16:
                    clipped = torch.clamp(save_data["audio"], -1.0, 1.0)
                    save_data["audio"] = (clipped * 32767.0).to(torch.int16)
                    
            torch.save(save_data, cache_path)
            
            if update_manifest:
                # Update manifest
                file_size = cache_path.stat().st_size
                checksum = self._get_sha256_checksum(cache_path)
                
                manifest = self._load_manifest()
                if "samples" not in manifest:
                    manifest["samples"] = {}
                manifest["samples"][key] = {
                    "video_path": video_path,
                    "cache_file": f"{key}.pt",
                    "file_size": file_size,
                    "sha256_checksum": checksum,
                    "preprocessing_version": "1.0",
                    "metadata": {
                        "dataset_name": data["metadata"].get("dataset_name", ""),
                        "label": data.get("label", 0),
                        "split": data["metadata"].get("split", ""),
                    }
                }
                # Update statistics
                stats = manifest.get("dataset_statistics", {})
                stats["total_samples"] = len(manifest["samples"])
                stats["total_size_bytes"] = stats.get("total_size_bytes", 0) + file_size
                import datetime
                stats["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
                manifest["dataset_statistics"] = stats
                
                self._save_manifest(manifest)
        except Exception as e:
            logger.warning(f"Failed to save cache for {video_path}: {e}")

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the count of REAL vs FAKE samples."""
        real_count = sum(1 for s in self.samples if s.label == 0)
        fake_count = sum(1 for s in self.samples if s.label == 1)
        return {"real": real_count, "fake": fake_count, "total": len(self.samples)}
