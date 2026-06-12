"""
Audio-Video synchronization module.

Aligns audio segments to corresponding video frames based on timestamps,
producing frame-aligned audio chunks for the cross-attention module.
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


class AVSynchronizer:
    """Synchronizes audio waveforms with video frame timestamps."""

    def __init__(
        self,
        window_ms: int = 40,
        sample_rate: int = 16000,
    ):
        """
        Args:
            window_ms: Audio window size per frame in milliseconds.
            sample_rate: Audio sample rate in Hz.
        """
        self.window_ms = window_ms
        self.sample_rate = sample_rate
        self.samples_per_window = int(sample_rate * window_ms / 1000)

    def synchronize(
        self,
        frames: list,
        timestamps: List[float],
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align audio segments to corresponding video frames.

        Each frame gets a chunk of audio centered around its timestamp.

        Args:
            frames: List of video frame arrays.
            timestamps: List of frame timestamps in seconds.
            waveform: Audio waveform array (mono).
            sample_rate: Audio sample rate.

        Returns:
            Tuple of (synced_frames, synced_audio_chunks).
            synced_audio_chunks[i] corresponds to frames[i].
        """
        logger.info(
            f"Synchronizing {len(frames)} frames with "
            f"{len(waveform)} audio samples ({len(waveform)/sample_rate:.2f}s)"
        )

        audio_duration = len(waveform) / sample_rate
        video_duration = timestamps[-1] if timestamps else 0.0

        if abs(audio_duration - video_duration) > 1.0:
            logger.warning(
                f"Audio/video duration mismatch: "
                f"audio={audio_duration:.2f}s, video={video_duration:.2f}s"
            )

        synced_frames = []
        synced_audio = []
        half_window = self.samples_per_window // 2

        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            # Center sample index for this timestamp
            center_sample = int(ts * sample_rate)

            # Extract audio window centered on this frame's timestamp
            start = max(0, center_sample - half_window)
            end = min(len(waveform), center_sample + half_window)

            chunk = waveform[start:end]

            # Pad if chunk is too short (near boundaries)
            if len(chunk) < self.samples_per_window:
                pad_total = self.samples_per_window - len(chunk)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                chunk = np.pad(chunk, (pad_left, pad_right), mode="constant")

            synced_frames.append(frame)
            synced_audio.append(chunk)

        logger.info(
            f"Synchronized {len(synced_frames)} frame-audio pairs "
            f"(window: {self.window_ms}ms = {self.samples_per_window} samples)"
        )
        return synced_frames, synced_audio

    def get_contiguous_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        max_duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get a contiguous audio segment (optionally truncated) for the
        full-sequence audio encoder.

        Args:
            waveform: Full audio waveform.
            sample_rate: Audio sample rate.
            max_duration: Maximum duration in seconds (None = full).

        Returns:
            Contiguous waveform array.
        """
        if max_duration is not None:
            max_samples = int(max_duration * sample_rate)
            if len(waveform) > max_samples:
                logger.debug(
                    f"Truncating audio from {len(waveform)/sample_rate:.2f}s "
                    f"to {max_duration:.2f}s"
                )
                waveform = waveform[:max_samples]
        return waveform

    def compute_sync_offset(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        timestamps: List[float],
    ) -> float:
        """
        Estimate the audio-video sync offset in seconds.

        Uses audio energy peaks and frame timestamps to detect
        potential sync drift. Positive = audio ahead of video.

        Args:
            waveform: Audio waveform.
            sample_rate: Audio sample rate.
            timestamps: Frame timestamps.

        Returns:
            Estimated offset in seconds.
        """
        # Simple energy-based offset estimation
        # Compute short-time energy
        hop = self.samples_per_window
        n_frames = len(waveform) // hop
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop
            end = start + hop
            segment = waveform[start:end]
            energy[i] = np.sum(segment ** 2) / len(segment)

        # Detect onset (first significant energy)
        if energy.max() == 0:
            return 0.0

        threshold = energy.max() * 0.1
        audio_onset_frame = np.argmax(energy > threshold)
        audio_onset_time = audio_onset_frame * hop / sample_rate

        # Compare with first frame timestamp
        video_onset_time = timestamps[0] if timestamps else 0.0
        offset = audio_onset_time - video_onset_time

        logger.debug(f"Estimated AV sync offset: {offset:.3f}s")
        return offset
