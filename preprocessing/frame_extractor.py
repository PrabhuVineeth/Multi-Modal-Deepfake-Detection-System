"""
Video frame extraction module.

Extracts frames from video files at a configurable FPS using OpenCV,
with support for timestamp tracking and variable frame rate videos.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class FrameExtractor:
    """Extracts video frames using OpenCV."""

    def __init__(self, target_fps: int = 25, max_frames: int = 300):
        """
        Args:
            target_fps: Target frames per second for extraction.
            max_frames: Maximum number of frames to extract.
        """
        self.target_fps = target_fps
        self.max_frames = max_frames

    def extract(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from a video at the target FPS.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of BGR numpy arrays (H, W, 3).
        """
        frames_with_ts = self.extract_with_timestamps(video_path)
        return [frame for _, frame in frames_with_ts]

    def extract_with_timestamps(
        self, video_path: str
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames with their timestamps.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of (timestamp_seconds, frame_bgr) tuples.

        Raises:
            RuntimeError: If video cannot be opened.
        """
        video_path = str(video_path)
        logger.info(f"Extracting frames from: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if source_fps <= 0:
                logger.warning("Could not detect FPS, defaulting to 25")
                source_fps = 25.0

            # Calculate frame sampling interval
            # If source FPS > target, we skip frames; if less, we take all
            frame_interval = max(1, int(round(source_fps / self.target_fps)))

            frames: List[Tuple[float, np.ndarray]] = []
            frame_idx = 0

            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frames.append((timestamp, frame))

                frame_idx += 1

            logger.info(
                f"Extracted {len(frames)} frames from {total_frames} total "
                f"(source: {source_fps:.1f}fps, target: {self.target_fps}fps)"
            )
            return frames

        finally:
            cap.release()

    def extract_specific_frames(
        self, video_path: str, frame_indices: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract specific frames by index.

        Args:
            video_path: Path to the video.
            frame_indices: List of frame indices to extract.

        Returns:
            List of (frame_index, frame_bgr) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            frames = []
            sorted_indices = sorted(set(frame_indices))

            for idx in sorted_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append((idx, frame))
                else:
                    logger.warning(f"Could not read frame {idx}")

            return frames
        finally:
            cap.release()

    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame",
    ) -> List[str]:
        """
        Save extracted frames to disk for debugging.

        Args:
            frames: List of frame arrays.
            output_dir: Directory to save frames.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved = []

        for i, frame in enumerate(frames):
            fp = str(out_path / f"{prefix}_{i:05d}.jpg")
            cv2.imwrite(fp, frame)
            saved.append(fp)

        logger.debug(f"Saved {len(saved)} frames to {output_dir}")
        return saved
