"""
Preprocessing pipeline orchestrator.

Runs the full preprocessing flow: audio extraction, frame extraction,
face detection, mouth ROI cropping, and audio-video synchronization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config import preprocess_config, PreprocessConfig
from preprocessing.audio_extractor import AudioExtractor
from preprocessing.av_synchronizer import AVSynchronizer
from preprocessing.face_detector import FaceDetection, FaceDetector
from preprocessing.frame_extractor import FrameExtractor
from preprocessing.mouth_roi_extractor import MouthROIExtractor


@dataclass
class PreprocessedData:
    """Container for all preprocessed data from a single video."""

    # Audio
    audio_waveform: np.ndarray = field(default_factory=lambda: np.array([]))
    audio_sample_rate: int = 16000

    # Video frames
    frames: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Face data
    face_detections: List[List[FaceDetection]] = field(default_factory=list)
    face_crops: List[np.ndarray] = field(default_factory=list)

    # Mouth ROI
    mouth_rois: List[np.ndarray] = field(default_factory=list)

    # Synchronized pairs
    synced_audio_chunks: List[np.ndarray] = field(default_factory=list)

    # Metadata
    video_path: str = ""
    num_frames: int = 0
    duration: float = 0.0
    has_audio: bool = True
    has_faces: bool = True

    @property
    def is_valid(self) -> bool:
        """Check if preprocessing produced usable data."""
        return (
            self.num_frames > 0
            and len(self.face_crops) > 0
            and len(self.audio_waveform) > 0
        )


class PreprocessingPipeline:
    """Orchestrates the full video preprocessing flow."""

    def __init__(
        self,
        config: Optional[PreprocessConfig] = None,
        debug: bool = False,
        debug_dir: Optional[str] = None,
    ):
        """
        Args:
            config: Preprocessing configuration. Uses defaults if None.
            debug: If True, saves intermediate outputs.
            debug_dir: Directory for debug outputs.
        """
        self.config = config or preprocess_config
        self.debug = debug
        self.debug_dir = debug_dir

        # Initialize sub-modules
        self.audio_extractor = AudioExtractor(
            sample_rate=self.config.audio_sample_rate,
        )
        self.frame_extractor = FrameExtractor(
            target_fps=self.config.target_fps,
            max_frames=self.config.max_frames,
        )
        self.face_detector = FaceDetector(
            backend=self.config.face_detection_backend,
            detection_threshold=self.config.face_detection_threshold,
            crop_size=self.config.face_crop_size,
            max_faces=self.config.max_faces,
            padding=self.config.face_padding,
        )
        self.mouth_extractor = MouthROIExtractor(
            roi_size=self.config.mouth_roi_size,
            padding=self.config.mouth_padding,
        )
        self.av_synchronizer = AVSynchronizer(
            window_ms=self.config.sync_window_ms,
            sample_rate=self.config.audio_sample_rate,
        )

    def process(self, video_path: str) -> PreprocessedData:
        """
        Run the full preprocessing pipeline on a single video.

        Pipeline stages:
          1. Extract audio waveform
          2. Extract video frames at target FPS
          3. Detect and track faces across frames
          4. Crop face regions and mouth ROIs
          5. Synchronize audio chunks with video frames

        Args:
            video_path: Path to the input video file.

        Returns:
            PreprocessedData containing all extracted data.
        """
        logger.info(f"Starting preprocessing pipeline for: {video_path}")
        result = PreprocessedData(video_path=str(video_path))

        # ── Stage 1: Audio extraction ──
        try:
            waveform, sr = self.audio_extractor.extract(video_path)
            result.audio_waveform = waveform
            result.audio_sample_rate = sr
            result.has_audio = True
            logger.info(f"Audio: {len(waveform)} samples, {len(waveform)/sr:.2f}s")
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            # Generate silent waveform placeholder
            result.audio_waveform = np.zeros(16000 * 5, dtype=np.float32)
            result.audio_sample_rate = 16000
            result.has_audio = False

        # ── Stage 2: Frame extraction ──
        try:
            frames_ts = self.frame_extractor.extract_with_timestamps(video_path)
            result.frames = [f for _, f in frames_ts]
            result.timestamps = [t for t, _ in frames_ts]
            result.num_frames = len(result.frames)
            result.duration = result.timestamps[-1] if result.timestamps else 0.0
            logger.info(f"Frames: {result.num_frames} extracted, {result.duration:.2f}s")
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise RuntimeError(f"Cannot extract frames from: {video_path}") from e

        if result.num_frames == 0:
            logger.error("No frames extracted")
            return result

        # ── Stage 3: Face detection & tracking ──
        try:
            result.face_detections = self.face_detector.detect_and_track(
                result.frames
            )

            # Extract face crops (use primary face per frame)
            result.face_crops = []
            result.mouth_rois = []
            valid_frame_indices = []

            for i, detections in enumerate(result.face_detections):
                if detections:
                    primary = detections[0]  # Highest confidence
                    result.face_crops.append(primary.face_crop)

                    # Stage 4: Mouth ROI extraction
                    mouth_roi = self.mouth_extractor.extract_from_detection(
                        result.frames[i], primary
                    )
                    result.mouth_rois.append(mouth_roi)
                    valid_frame_indices.append(i)
                else:
                    # No face detected — use blank placeholder
                    blank_face = np.zeros(
                        (*self.config.face_crop_size, 3), dtype=np.uint8
                    )
                    blank_mouth = np.zeros(
                        (*self.config.mouth_roi_size, 3), dtype=np.uint8
                    )
                    result.face_crops.append(blank_face)
                    result.mouth_rois.append(blank_mouth)

            face_rate = len(valid_frame_indices) / max(result.num_frames, 1)
            result.has_faces = face_rate > 0.3  # At least 30% of frames have faces
            logger.info(
                f"Faces: detected in {len(valid_frame_indices)}/{result.num_frames} "
                f"frames ({face_rate:.1%})"
            )

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            result.has_faces = False
            # Fill with blank crops
            result.face_crops = [
                np.zeros((*self.config.face_crop_size, 3), dtype=np.uint8)
            ] * result.num_frames
            result.mouth_rois = [
                np.zeros((*self.config.mouth_roi_size, 3), dtype=np.uint8)
            ] * result.num_frames

        # ── Stage 5: AV synchronization ──
        try:
            _, result.synced_audio_chunks = self.av_synchronizer.synchronize(
                result.frames,
                result.timestamps,
                result.audio_waveform,
                result.audio_sample_rate,
            )
        except Exception as e:
            logger.warning(f"AV synchronization failed: {e}")
            result.synced_audio_chunks = []

        # ── Debug output ──
        if self.debug and self.debug_dir:
            self._save_debug_outputs(result)

        logger.info(
            f"Preprocessing complete: {result.num_frames} frames, "
            f"{len(result.face_crops)} face crops, "
            f"{len(result.mouth_rois)} mouth ROIs, "
            f"{len(result.synced_audio_chunks)} audio chunks"
        )
        return result

    def _save_debug_outputs(self, data: PreprocessedData) -> None:
        """Save intermediate outputs for debugging."""
        debug_path = Path(self.debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

        # Save frames
        self.frame_extractor.save_frames(
            data.frames[:10],  # Save first 10
            str(debug_path / "frames"),
        )

        # Save audio
        if data.has_audio:
            self.audio_extractor.save_audio(
                data.audio_waveform,
                data.audio_sample_rate,
                str(debug_path / "audio.wav"),
            )

        # Save face crops
        import cv2
        face_dir = debug_path / "faces"
        face_dir.mkdir(exist_ok=True)
        for i, crop in enumerate(data.face_crops[:10]):
            cv2.imwrite(str(face_dir / f"face_{i:04d}.jpg"), crop)

        # Save mouth ROIs
        mouth_dir = debug_path / "mouths"
        mouth_dir.mkdir(exist_ok=True)
        for i, roi in enumerate(data.mouth_rois[:10]):
            cv2.imwrite(str(mouth_dir / f"mouth_{i:04d}.jpg"), roi)

        logger.debug(f"Debug outputs saved to: {debug_path}")
