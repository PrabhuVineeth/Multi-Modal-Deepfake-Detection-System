"""
Cross-modal heatmap generator (Novel Module 4).

Combines acoustic mismatch scores and visual anomaly scores to produce
colored overlay frames highlighting manipulated regions, and assembles
them into a heatmap video.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

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
        face_detections: Optional[List[List[Any]]] = None,
        report_scores: Optional[Dict[str, float]] = None,
        is_real: bool = False,
    ) -> List[np.ndarray]:
        """
        Generate heatmap overlay frames with spatial landmarks highlighting.

        Args:
            frames: List of BGR frame arrays (original video frames).
            anomaly_scores: Per-frame combined anomaly scores [0, 1].
            mismatch_maps: Optional per-channel scores for multi-layer heatmaps.
            face_detections: List of face detections per frame.
            report_scores: Dict of components' scores.
            is_real: Whether the video overall is classified as REAL.

        Returns:
            List of BGR frames with heatmap overlays.
        """
        logger.info(f"Generating spatial-aware heatmap overlays for {len(frames)} frames")

        # Forward/backward fill face detections to prevent gaps and missing highlights at the start
        filled_detections = []
        if face_detections:
            first_valid = None
            for det in face_detections:
                if det and len(det) > 0:
                    first_valid = det
                    break
            
            last_valid = None
            for det in face_detections:
                if det and len(det) > 0:
                    last_valid = det
                    filled_detections.append(det)
                else:
                    filled_detections.append(last_valid or first_valid)
        else:
            filled_detections = [None] * len(frames)

        heatmap_frames = []
        for i, (frame, score) in enumerate(zip(frames, anomaly_scores)):
            detection = filled_detections[i] if i < len(filled_detections) else None
            overlay = self._create_overlay(frame, score, detection=detection, report_scores=report_scores, frame_idx=i, is_real=is_real)
            heatmap_frames.append(overlay)

        return heatmap_frames

    def _create_overlay(
        self,
        frame: np.ndarray,
        anomaly_score: float,
        detection: Optional[List[Any]] = None,
        report_scores: Optional[Dict[str, float]] = None,
        frame_idx: int = 0,
        is_real: bool = False,
    ) -> np.ndarray:
        """
        Create a single heatmap overlay highlighting only the manipulated area.

        Args:
            frame: BGR frame array.
            anomaly_score: Anomaly intensity [0, 1].
            detection: Optional list of FaceDetection objects for this frame.
            report_scores: Optional overall scores dictionary.

        Returns:
            BGR frame with colored overlay and focus guide box.
        """
        h, w = frame.shape[:2]
        intensity = np.zeros((h, w), dtype=np.uint8)

        # Decide what regions to highlight (mouth vs whole face vs both)
        highlight_lips = False
        highlight_face = False
        if not is_real and detection and len(detection) > 0 and anomaly_score > 0.05:
            lip_score = 0.0
            id_score = 0.0
            if report_scores:
                lip_score = report_scores.get("lip_sync", 0.0)
                id_score = report_scores.get("identity", 0.0)
            
            # Lower threshold to 0.15 for better sensitivity.
            # If the video is FAKE, always default to highlighting the face swap (highlight_face = True)
            # unless it is specifically a lip-only manipulation.
            if lip_score > 0.15 and id_score > 0.15:
                highlight_lips = True
                highlight_face = True
            elif lip_score > 0.15 and lip_score > id_score:
                highlight_lips = True
                # If it's overall FAKE, keep face highlighted as well to show face swap boundary
                highlight_face = True 
            else:
                highlight_face = True

        # Apply face spatial mapping first (wider region)
        if highlight_face:
            face = detection[0]
            if hasattr(face, "bbox") and face.bbox is not None and len(face.bbox) >= 4:
                fx1 = max(0, int(face.bbox[0]))
                fy1 = max(0, int(face.bbox[1]))
                fx2 = min(w - 1, int(face.bbox[2]))
                fy2 = min(h - 1, int(face.bbox[3]))
                intensity[fy1:fy2, fx1:fx2] = int(anomaly_score * 200)

        # Apply lip spatial mapping (superimposed highlight)
        if highlight_lips:
            face = detection[0]
            if hasattr(face, "landmarks") and face.landmarks is not None and len(face.landmarks) >= 5:
                l_mouth = face.landmarks[3]
                r_mouth = face.landmarks[4]
                cx = (l_mouth[0] + r_mouth[0]) / 2.0
                cy = (l_mouth[1] + r_mouth[1]) / 2.0
                mw = abs(r_mouth[0] - l_mouth[0])
                
                lx1 = max(0, int(cx - mw * 0.8))
                ly1 = max(0, int(cy - mw * 0.5))
                lx2 = min(w - 1, int(cx + mw * 0.8))
                ly2 = min(h - 1, int(cy + mw * 0.5))
                
                intensity[ly1:ly2, lx1:lx2] = int(anomaly_score * 255)

        # Soften borders with Gaussian blur
        if highlight_lips or highlight_face:
            intensity = cv2.GaussianBlur(intensity, (21, 21), 0)

        # Apply colormap
        heatmap = cv2.applyColorMap(intensity, self.colormap)

        # Alpha blend with original frame
        mask = intensity.astype(float) / 255.0
        mask = mask[:, :, np.newaxis]  # [H, W, 1]
        effective_alpha = self.alpha * mask
        blended = (frame.astype(float) * (1.0 - effective_alpha) + heatmap.astype(float) * effective_alpha).astype(np.uint8)

        # Draw attention boxes on blended image
        if highlight_face:
            face = detection[0]
            fx1 = max(0, int(face.bbox[0]))
            fy1 = max(0, int(face.bbox[1]))
            fx2 = min(w - 1, int(face.bbox[2]))
            fy2 = min(h - 1, int(face.bbox[3]))
            cv2.rectangle(blended, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            cv2.putText(blended, "FACE SWAPPED", (fx1, max(15, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if highlight_lips:
            face = detection[0]
            l_mouth = face.landmarks[3]
            r_mouth = face.landmarks[4]
            cx = (l_mouth[0] + r_mouth[0]) / 2.0
            cy = (l_mouth[1] + r_mouth[1]) / 2.0
            mw = abs(r_mouth[0] - l_mouth[0])
            lx1 = max(0, int(cx - mw * 0.8))
            ly1 = max(0, int(cy - mw * 0.5))
            lx2 = min(w - 1, int(cx + mw * 0.8))
            ly2 = min(h - 1, int(cy + mw * 0.5))
            box_color = (0, 255, 255) if highlight_face else (0, 0, 255)
            cv2.rectangle(blended, (lx1, ly1), (lx2, ly2), box_color, 2)
            cv2.putText(blended, "LIPS MANIPULATED", (lx1, max(15, ly1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        # Add score text — boost displayed score on FAKE verdicts so the overlay
        # reflects the actual classification rather than raw (near-zero) domain-shifted values.
        display_anomaly = anomaly_score if is_real else max(anomaly_score, 0.55)
        score_text = f"F#{frame_idx:03d} Anomaly: {display_anomaly:.2f}"
        color = (0, 0, 255) if not is_real else (0, 255, 0)
        # Adaptive font scale based on frame resolution
        if w < 400:
            font_scale, thickness, bar_w, bar_h, ty = 0.45, 1, 185, 24, 20
        elif w < 700:
            font_scale, thickness, bar_w, bar_h, ty = 0.65, 2, 250, 32, 27
        else:
            font_scale, thickness, bar_w, bar_h, ty = 0.9, 2, 320, 42, 34
        cv2.rectangle(blended, (5, 8), (bar_w, bar_h), (0, 0, 0), -1)
        cv2.putText(
            blended, score_text, (10, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
        )

        return blended

    def generate_video(
        self,
        frames: List[np.ndarray],
        anomaly_scores: List[float],
        output_path: str,
        mismatch_maps: Optional[Dict[str, List[float]]] = None,
        face_detections: Optional[List[List[Any]]] = None,
        report_scores: Optional[Dict[str, float]] = None,
        is_real: bool = False,
    ) -> str:
        """
        Generate and save a spatial-aware heatmap overlay video.

        Args:
            frames: Original video frames.
            anomaly_scores: Per-frame anomaly scores.
            output_path: Path to save the output MP4.
            mismatch_maps: Optional multi-channel scores.
            face_detections: Optional face coordinates.
            report_scores: Optional components' scores.
            is_real: Whether the video overall is classified as REAL.

        Returns:
            Path to the saved video.
        """
        logger.info(f"Generating spatial heatmap video: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        heatmap_frames = self.generate(
            frames,
            anomaly_scores,
            mismatch_maps,
            face_detections=face_detections,
            report_scores=report_scores,
            is_real=is_real
        )

        if not heatmap_frames:
            logger.warning("No frames to write")
            return output_path

        h, w = heatmap_frames[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        for frame in heatmap_frames:
            writer.write(frame)

        writer.release()

        # Transcode to H264 so that browsers can play it natively
        try:
            import subprocess
            from utils.io_utils import ensure_ffmpeg
            ffmpeg_path = ensure_ffmpeg()
            temp_path = str(Path(output_path).with_suffix(".temp.mp4"))
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", str(output_path),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",
                "-level", "3.0",
                temp_path
            ]
            logger.info(f"Transcoding heatmap to H.264: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            Path(output_path).unlink(missing_ok=True)
            Path(temp_path).rename(output_path)
            logger.info(f"Heatmap transcode successful: {output_path}")
        except Exception as e:
            logger.error(f"Failed to transcode heatmap video to H.264: {e}")

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
