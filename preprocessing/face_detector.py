"""
Face detection module.

Detects and aligns faces using InsightFace (RetinaFace) with
MTCNN fallback. Provides face bounding boxes, 5-point landmarks,
and cropped + aligned face images.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class FaceDetection:
    """A single detected face."""

    bbox: np.ndarray            # [x1, y1, x2, y2]
    confidence: float           # Detection confidence
    landmarks: np.ndarray       # 5-point landmarks: (5, 2)
    face_crop: np.ndarray       # Cropped & aligned face image
    track_id: int = -1          # Tracking ID across frames


class FaceDetector:
    """
    Detects faces using InsightFace (RetinaFace) with MTCNN fallback.

    Face tracking across frames uses IoU-based matching to maintain
    consistent identity assignments.
    """

    def __init__(
        self,
        backend: str = "retinaface",
        detection_threshold: float = 0.8,
        crop_size: Tuple[int, int] = (224, 224),
        max_faces: int = 1,
        padding: float = 0.3,
    ):
        """
        Args:
            backend: Detection backend ("retinaface" or "mtcnn").
            detection_threshold: Minimum confidence threshold.
            crop_size: Output size for cropped faces (H, W).
            max_faces: Maximum number of faces to detect per frame.
            padding: Padding ratio around the detected bounding box.
        """
        self.backend = backend
        self.detection_threshold = detection_threshold
        self.crop_size = crop_size
        self.max_faces = max_faces
        self.padding = padding

        self._detector = None
        self._next_track_id = 0
        self._prev_detections: List[FaceDetection] = []

    def _init_detector(self):
        """Lazy initialization of the face detector."""
        if self._detector is not None:
            return

        if self.backend == "retinaface":
            try:
                import os
                import sys
                import torch
                # On Windows, explicitly add PyTorch's bundled DLL library to the path so ONNX Runtime can find CUDA/cuDNN
                if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
                    if os.path.exists(torch_lib_dir):
                        os.add_dll_directory(torch_lib_dir)

                from insightface.app import FaceAnalysis

                self._detector = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection"],
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._detector.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("RetinaFace detector initialized (InsightFace)")
                return
            except ImportError:
                logger.warning("InsightFace not available, falling back to OpenCV DNN")
            except Exception as e:
                logger.warning(f"RetinaFace init failed: {e}, falling back to OpenCV DNN")

        # Fallback: OpenCV DNN face detector
        self._detector = "opencv_dnn"
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        logger.info("Using OpenCV Haar cascade face detector (fallback)")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a single frame.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of FaceDetection objects, sorted by confidence descending.
        """
        self._init_detector()

        if isinstance(self._detector, str) and self._detector == "opencv_dnn":
            return self._detect_opencv(frame)
        else:
            return self._detect_insightface(frame)

    def _detect_insightface(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect using InsightFace RetinaFace."""
        faces = self._detector.get(frame)

        detections = []
        for face in faces:
            if face.det_score < self.detection_threshold:
                continue

            bbox = face.bbox.astype(int)
            landmarks = face.kps  # (5, 2)
            confidence = float(face.det_score)

            # Crop face with padding
            face_crop = self._crop_face(frame, bbox)

            detections.append(FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
                face_crop=face_crop,
            ))

        # Sort by confidence, limit to max_faces
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = detections[:self.max_faces]

        return detections

    def _detect_opencv(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect using OpenCV Haar cascade (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        detections = []
        for (x, y, w, h) in faces:
            bbox = np.array([x, y, x + w, y + h])
            confidence = 0.9  # Haar doesn't give confidence

            # Estimate 5-point landmarks from bounding box
            landmarks = self._estimate_landmarks_from_bbox(bbox)

            face_crop = self._crop_face(frame, bbox)

            detections.append(FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
                face_crop=face_crop,
            ))

        detections = detections[:self.max_faces]
        return detections

    def _estimate_landmarks_from_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Estimate approximate 5-point landmarks from a bounding box."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        # Approximate positions: left_eye, right_eye, nose, left_mouth, right_mouth
        landmarks = np.array([
            [x1 + 0.3 * w, y1 + 0.35 * h],   # Left eye
            [x1 + 0.7 * w, y1 + 0.35 * h],   # Right eye
            [x1 + 0.5 * w, y1 + 0.55 * h],   # Nose
            [x1 + 0.35 * w, y1 + 0.78 * h],  # Left mouth corner
            [x1 + 0.65 * w, y1 + 0.78 * h],  # Right mouth corner
        ], dtype=np.float32)

        return landmarks

    def _crop_face(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop and resize a face region with padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        bw, bh = x2 - x1, y2 - y1
        pad_w = int(bw * self.padding)
        pad_h = int(bh * self.padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((*self.crop_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.crop_size[1], self.crop_size[0]))

        return crop

    def detect_and_track(
        self, frames: List[np.ndarray]
    ) -> List[List[FaceDetection]]:
        """
        Detect and track faces across multiple frames.

        Uses IoU-based matching to maintain consistent track IDs.

        Args:
            frames: List of BGR frame arrays.

        Returns:
            List of List[FaceDetection] for each frame.
        """
        logger.info(f"Detecting and tracking faces across {len(frames)} frames")
        all_detections = []
        self._prev_detections = []
        self._next_track_id = 0

        for i, frame in enumerate(frames):
            detections = self.detect(frame)
            self._assign_track_ids(detections)
            all_detections.append(detections)

            if (i + 1) % 50 == 0:
                logger.debug(f"Face detection: {i + 1}/{len(frames)} frames processed")

        return all_detections

    def _assign_track_ids(self, detections: List[FaceDetection]) -> None:
        """Assign track IDs using IoU matching with previous frame."""
        if not self._prev_detections:
            for det in detections:
                det.track_id = self._next_track_id
                self._next_track_id += 1
        else:
            # Compute IoU matrix
            used_prev = set()
            for det in detections:
                best_iou = 0.0
                best_prev_id = -1
                for prev in self._prev_detections:
                    if prev.track_id in used_prev:
                        continue
                    iou = self._compute_iou(det.bbox, prev.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_prev_id = prev.track_id

                if best_iou > 0.3 and best_prev_id >= 0:
                    det.track_id = best_prev_id
                    used_prev.add(best_prev_id)
                else:
                    det.track_id = self._next_track_id
                    self._next_track_id += 1

        self._prev_detections = detections

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / max(union, 1e-6)
