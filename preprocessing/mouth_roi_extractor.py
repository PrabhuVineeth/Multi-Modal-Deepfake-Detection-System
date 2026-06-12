"""
Mouth ROI extraction module.

Crops the mouth region from face frames using facial landmarks,
with configurable padding for lip-sync analysis input.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class MouthROIExtractor:
    """Extracts mouth Region of Interest from face frames using landmarks."""

    def __init__(
        self,
        roi_size: Tuple[int, int] = (96, 96),
        padding: float = 0.4,
    ):
        """
        Args:
            roi_size: Output size for mouth ROI (H, W).
            padding: Padding ratio around the estimated mouth bounding box.
        """
        self.roi_size = roi_size
        self.padding = padding

    def extract(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        Extract mouth ROI from a frame using 5-point facial landmarks.

        The 5-point landmark model provides:
          [0] left eye, [1] right eye, [2] nose,
          [3] left mouth corner, [4] right mouth corner.

        The mouth ROI is estimated from landmarks 3 and 4 (mouth corners)
        with vertical extent inferred from the face geometry.

        Args:
            frame: BGR image array (H, W, 3).
            landmarks: 5-point landmarks array of shape (5, 2).

        Returns:
            Cropped and resized mouth ROI of shape (roi_size[0], roi_size[1], 3).
        """
        h, w = frame.shape[:2]

        # Mouth corners
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        # Estimate mouth center and dimensions
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        mouth_width = np.linalg.norm(right_mouth - left_mouth)

        # Estimate mouth height from face proportions
        # Use distance from nose to mouth corners as reference
        nose = landmarks[2]
        nose_to_mouth = np.linalg.norm(nose - np.array([mouth_center_x, mouth_center_y]))
        mouth_height = max(mouth_width * 0.6, nose_to_mouth * 0.8)

        # Apply padding
        padded_w = mouth_width * (1 + self.padding)
        padded_h = mouth_height * (1 + self.padding)

        # Calculate bounding box
        x1 = int(max(0, mouth_center_x - padded_w / 2))
        y1 = int(max(0, mouth_center_y - padded_h / 2))
        x2 = int(min(w, mouth_center_x + padded_w / 2))
        y2 = int(min(h, mouth_center_y + padded_h / 2))

        # Crop
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning("Empty mouth crop, returning blank ROI")
            return np.zeros((*self.roi_size, 3), dtype=np.uint8)

        # Resize to target size
        roi = cv2.resize(crop, (self.roi_size[1], self.roi_size[0]))
        return roi

    def extract_batch(
        self,
        frames: list,
        landmarks_list: list,
    ) -> list:
        """
        Extract mouth ROIs from multiple frames.

        Args:
            frames: List of BGR frame arrays.
            landmarks_list: List of corresponding 5-point landmark arrays.

        Returns:
            List of mouth ROI arrays.
        """
        rois = []
        for frame, landmarks in zip(frames, landmarks_list):
            roi = self.extract(frame, landmarks)
            rois.append(roi)
        return rois

    def extract_from_detection(
        self, frame: np.ndarray, face_detection
    ) -> np.ndarray:
        """
        Extract mouth ROI directly from a FaceDetection object.

        Args:
            frame: BGR image array.
            face_detection: FaceDetection dataclass with landmarks attribute.

        Returns:
            Mouth ROI array.
        """
        return self.extract(frame, face_detection.landmarks)
