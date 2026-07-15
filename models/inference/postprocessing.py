"""
Post-processing module.

Aggregates per-frame model outputs into video-level results,
applies confidence calibration, and formats everything into
a structured ForensicReport.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from models.tfbd import ForgeryBoundary


@dataclass
class ForensicReport:
    """Structured forensic analysis report."""

    # Classification
    classification: str = "UNKNOWN"           # "REAL" or "FAKE"
    confidence: float = 0.0                    # 0-100 percentage
    raw_probability: float = 0.0               # 0-1 sigmoid output

    # Per-analyzer scores (0-1, higher = more suspicious)
    lip_sync_score: float = 0.0
    identity_score: float = 0.0
    temporal_score: float = 0.0
    av_sync_score: float = 0.0

    # Evidence channel weights (which evidence contributed most)
    channel_weights: Dict[str, float] = field(default_factory=dict)

    # Temporal boundaries
    boundaries: List[Dict[str, Any]] = field(default_factory=list)
    has_forgery_boundaries: bool = False

    # Per-frame anomaly scores
    frame_anomaly_scores: List[float] = field(default_factory=list)
    raw_frame_anomaly_scores: List[float] = field(default_factory=list)

    # Metadata
    video_path: str = ""
    duration: float = 0.0
    num_frames: int = 0
    processing_time: float = 0.0
    model_version: str = "1.0.0"
    timestamp: str = ""
    json_report_path: Optional[str] = None
    html_report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "classification": self.classification,
            "confidence": round(self.confidence, 2),
            "raw_probability": round(self.raw_probability, 4),
            "scores": {
                "lip_sync": round(self.lip_sync_score, 4),
                "identity": round(self.identity_score, 4),
                "temporal": round(self.temporal_score, 4),
                "av_sync": round(self.av_sync_score, 4),
            },
            "channel_weights": {
                k: round(v, 4) for k, v in self.channel_weights.items()
            },
            "boundaries": self.boundaries,
            "has_forgery_boundaries": self.has_forgery_boundaries,
            "frame_anomaly_scores": [
                round(s, 4) for s in self.frame_anomaly_scores
            ],
            "json_report_path": self.json_report_path,
            "html_report_path": self.html_report_path,
            "metadata": {
                "video_path": self.video_path,
                "duration": round(self.duration, 2),
                "num_frames": self.num_frames,
                "processing_time": round(self.processing_time, 3),
                "model_version": self.model_version,
                "timestamp": self.timestamp,
            },
        }


class PostProcessor:
    """
    Post-processes model outputs into a structured forensic report.

    Aggregates per-frame scores, applies calibration, interprets
    TFBD outputs, and formats everything for display.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Classification threshold (> threshold = FAKE).
        """
        self.threshold = threshold

    def process(
        self,
        forensic_output,
        video_path: str = "",
        duration: float = 0.0,
        num_frames: int = 0,
        timestamps: Optional[List[float]] = None,
        processing_time: float = 0.0,
    ) -> ForensicReport:
        """
        Convert model ForensicOutput to a ForensicReport.

        Args:
            forensic_output: ForensicOutput from the model.
            video_path: Source video path.
            duration: Video duration in seconds.
            num_frames: Number of processed frames.
            timestamps: Frame timestamps.
            processing_time: Total processing time.

        Returns:
            Structured ForensicReport.
        """
        report = ForensicReport(
            video_path=video_path,
            duration=duration,
            num_frames=num_frames,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
        )

        # Classification
        prob = self._to_float(forensic_output.probability)
        report.raw_probability = prob
        report.classification = "FAKE" if prob >= self.threshold else "REAL"

        # Per-analyzer scores
        report.lip_sync_score = self._to_float(forensic_output.lip_sync_score)
        report.identity_score = self._to_float(forensic_output.identity_score)
        report.temporal_score = self._to_float(forensic_output.temporal_score)
        report.av_sync_score = self._to_float(forensic_output.av_sync_score)

        # Fallback override: if any individual component analyzer detects manipulation with extremely high confidence
        if report.classification == "REAL" and max(report.lip_sync_score, report.identity_score, report.av_sync_score) > 0.85:
            report.classification = "FAKE"
            prob = max(prob, 0.75)  # Boost probability above threshold
            report.raw_probability = prob

        # Calibrate confidence relative to the decision threshold
        if report.classification == "FAKE":
            report.confidence = 50.0 + 50.0 * (prob - self.threshold) / (1.0 - self.threshold + 1e-6)
        else:
            report.confidence = 50.0 + 50.0 * (self.threshold - prob) / (self.threshold + 1e-6)

        # Channel weights
        if forensic_output.channel_weights is not None:
            weights = forensic_output.channel_weights.detach().cpu().numpy()
            if weights.ndim > 1:
                weights = weights[0]  # Take first batch element
            channel_names = ["lip_sync", "identity", "temporal", "av_sync"]
            report.channel_weights = {
                name: float(w) for name, w in zip(channel_names, weights)
            }

        # Frame anomaly scores from mismatch maps
        if forensic_output.mismatch_maps and "combined" in forensic_output.mismatch_maps:
            combined = forensic_output.mismatch_maps["combined"]
            scores = combined.detach().cpu().numpy()
            if scores.ndim > 2:
                scores = scores[0]  # First batch element
            raw_scores = scores.flatten().tolist()
            
            # Calibrate frame scores relative to the global video prediction.
            # If the overall video probability is below threshold (REAL), scale down
            # the frame scores proportionally to prevent false-alarm red blocks on real videos.
            factor = min(1.0, prob / self.threshold) if prob < self.threshold else 1.0
            report.frame_anomaly_scores = [float(s * factor) for s in raw_scores]
            report.raw_frame_anomaly_scores = list(report.frame_anomaly_scores)

        # Temporal boundaries from TFBD
        if forensic_output.boundary_tags is not None and timestamps:
            from models.tfbd import TemporalForgeryBoundaryDetector

            tfbd = TemporalForgeryBoundaryDetector()
            tags = forensic_output.boundary_tags
            if tags.dim() > 1:
                tags = tags[0]  # First batch

            # If the overall video is classified as REAL, override boundary tags to all REAL (0)
            if report.classification == "REAL":
                tags = torch.zeros_like(tags)

            boundaries = tfbd.extract_boundaries(tags, timestamps)
            report.boundaries = [
                {
                    "start_time": b.start_time,
                    "end_time": b.end_time,
                    "start_frame": b.start_frame,
                    "end_frame": b.end_frame,
                    "tag": b.tag,
                    "confidence": b.confidence,
                }
                for b in boundaries
            ]
            report.has_forgery_boundaries = any(
                b.tag == "FAKE" for b in boundaries
            )

            # Align frame_anomaly_scores with CRF tags to prevent visual contradictions
            tag_list = tags.cpu().numpy().flatten().tolist()
            for idx in range(min(len(report.frame_anomaly_scores), len(tag_list))):
                tag_val = int(tag_list[idx])
                if tag_val == 0:  # REAL
                    report.frame_anomaly_scores[idx] = min(0.09, report.frame_anomaly_scores[idx])
                elif tag_val == 2:  # BOUNDARY
                    report.frame_anomaly_scores[idx] = 0.42
                elif tag_val == 1:  # FAKE
                    report.frame_anomaly_scores[idx] = max(0.55, report.frame_anomaly_scores[idx])

        logger.info(
            f"Report generated: {report.classification} "
            f"(confidence={report.confidence:.1f}%)"
        )
        return report

    @staticmethod
    def _to_float(tensor) -> float:
        """Safely extract a scalar float from a tensor."""
        if tensor is None:
            return 0.0
        if isinstance(tensor, (int, float)):
            return float(tensor)
        if isinstance(tensor, torch.Tensor):
            return float(tensor.detach().cpu().item())
        if isinstance(tensor, np.ndarray):
            return float(tensor.flat[0])
        return 0.0
