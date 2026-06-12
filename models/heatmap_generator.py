"""
Cross-modal heatmap generator (Novel Module 4).

Combines acoustic mismatch scores and visual anomaly scores to produce
colored overlay frames highlighting manipulated regions, and assembles
them into a heatmap video.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger


class CrossModalHeatmapGenerator:
    """
    Generates visual heatmap overlays showing manipulation evidence.

    Combines multi-channel anomaly scores into per-frame color overlays
    and produces an output heatmap video.
    """

    def __init__(
        self,
        colormap: str = "jet",
        alpha: float = 0.4,
        fps: int = 25,
    ):
        """
        Args:
            colormap: OpenCV colormap name ('jet', 'hot', 'turbo', etc.).
            alpha: Overlay transparency (0 = invisible, 1 = opaque).
            fps: Output video frame rate.
        """
        self.alpha = alpha
        self.fps = fps

        # Map string colormap names to OpenCV constants
        colormap_map = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "turbo": cv2.COLORMAP_TURBO,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
        }
        self.colormap = colormap_map.get(colormap, cv2.COLORMAP_JET)

    def generate(
        self,
        frames: List[np.ndarray],
        anomaly_scores: List[float],
        mismatch_maps: Optional[Dict[str, List[float]]] = None,
    ) -> List[np.ndarray]:
        """
        Generate heatmap overlay frames.

        Args:
            frames: List of BGR frame arrays (original video frames).
            anomaly_scores: Per-frame combined anomaly scores [0, 1].
            mismatch_maps: Optional per-channel scores for multi-layer heatmaps.
                Keys: 'lip_sync', 'identity', 'temporal', 'av_sync'.
                Values: List of per-frame scores [0, 1].

        Returns:
            List of BGR frames with heatmap overlays.
        """
        logger.info(f"Generating heatmap overlays for {len(frames)} frames")

        heatmap_frames = []
        for i, (frame, score) in enumerate(zip(frames, anomaly_scores)):
            overlay = self._create_overlay(frame, score)
            heatmap_frames.append(overlay)

        return heatmap_frames

    def _create_overlay(
        self,
        frame: np.ndarray,
        anomaly_score: float,
    ) -> np.ndarray:
        """
        Create a single heatmap overlay on a frame.

        Args:
            frame: BGR frame array.
            anomaly_score: Anomaly intensity [0, 1].

        Returns:
            BGR frame with colored overlay.
        """
        h, w = frame.shape[:2]

        # Create a uniform intensity map (can be spatial in future)
        intensity = np.full((h, w), int(anomaly_score * 255), dtype=np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(intensity, self.colormap)

        # Blend with original frame
        # More anomalous → more overlay; less anomalous → more original
        effective_alpha = self.alpha * anomaly_score
        blended = cv2.addWeighted(frame, 1 - effective_alpha, heatmap, effective_alpha, 0)

        # Add score text
        score_text = f"Anomaly: {anomaly_score:.2f}"
        color = (0, 0, 255) if anomaly_score > 0.5 else (0, 255, 0)
        cv2.putText(
            blended, score_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

        return blended

    def generate_video(
        self,
        frames: List[np.ndarray],
        anomaly_scores: List[float],
        output_path: str,
        mismatch_maps: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        """
        Generate and save a heatmap overlay video.

        Args:
            frames: Original video frames.
            anomaly_scores: Per-frame anomaly scores.
            output_path: Path to save the output MP4.
            mismatch_maps: Optional multi-channel scores.

        Returns:
            Path to the saved video.
        """
        logger.info(f"Generating heatmap video: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        heatmap_frames = self.generate(frames, anomaly_scores, mismatch_maps)

        if not heatmap_frames:
            logger.warning("No frames to write")
            return output_path

        h, w = heatmap_frames[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        for frame in heatmap_frames:
            writer.write(frame)

        writer.release()
        logger.info(
            f"Heatmap video saved: {output_path} "
            f"({len(heatmap_frames)} frames, {len(heatmap_frames)/self.fps:.1f}s)"
        )
        return output_path

    def generate_key_frames(
        self,
        frames: List[np.ndarray],
        anomaly_scores: List[float],
        n_frames: int = 10,
    ) -> List[np.ndarray]:
        """
        Extract the N most anomalous frames with heatmap overlays.

        Useful for report generation (embed key frames in HTML report).

        Args:
            frames: Original video frames.
            anomaly_scores: Per-frame anomaly scores.
            n_frames: Number of key frames to extract.

        Returns:
            List of heatmap overlay frames (sorted by anomaly score descending).
        """
        if not frames or not anomaly_scores:
            return []

        # Find indices of most anomalous frames
        scores = np.array(anomaly_scores)
        top_indices = np.argsort(scores)[-n_frames:][::-1]

        key_frames = []
        for idx in top_indices:
            overlay = self._create_overlay(frames[idx], float(scores[idx]))
            # Add frame number annotation
            cv2.putText(
                overlay, f"Frame #{idx}", (10, overlay.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            key_frames.append(overlay)

        return key_frames
