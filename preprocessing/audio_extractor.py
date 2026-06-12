"""
Audio extraction module.

Extracts audio tracks from video files using FFmpeg, converts to 16kHz
mono WAV format suitable for Wav2Vec2 processing.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

from utils.io_utils import ensure_ffmpeg


class AudioExtractor:
    """Extracts audio waveforms from video files using FFmpeg."""

    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        """
        Args:
            sample_rate: Target audio sample rate in Hz.
            mono: Whether to convert to mono channel.
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self._ffmpeg = ensure_ffmpeg()

    def extract(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio waveform from a video file.

        Args:
            video_path: Path to the input video file.

        Returns:
            Tuple of (waveform_array, sample_rate).
            waveform_array shape: (num_samples,) for mono.

        Raises:
            RuntimeError: If audio extraction fails.
        """
        video_path = str(video_path)
        logger.info(f"Extracting audio from: {video_path}")

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # FFmpeg command: extract audio, convert to target format
            cmd = [
                "ffmpeg", "-y",           # Overwrite output
                "-i", video_path,         # Input video
                "-vn",                    # No video
                "-acodec", "pcm_s16le",   # 16-bit PCM WAV
                "-ar", str(self.sample_rate),  # Sample rate
            ]
            if self.mono:
                cmd.extend(["-ac", "1"])  # Mono
            cmd.append(tmp_path)

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                # Check if video has no audio stream
                if "does not contain any stream" in result.stderr:
                    logger.warning(f"No audio stream in video: {video_path}")
                    return self._generate_silent_waveform(video_path), self.sample_rate
                raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")

            # Read the extracted WAV
            waveform, sr = sf.read(tmp_path, dtype="float32")
            logger.info(
                f"Audio extracted: {len(waveform)} samples, "
                f"{len(waveform) / sr:.2f}s, {sr}Hz"
            )
            return waveform, sr

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Audio extraction timed out for: {video_path}")

        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    def _generate_silent_waveform(self, video_path: str) -> np.ndarray:
        """Generate a silent waveform matching the video duration."""
        from utils.io_utils import read_video_metadata

        metadata = read_video_metadata(video_path)
        duration = metadata.get("duration", 5.0)
        num_samples = int(duration * self.sample_rate)
        logger.warning(f"Generating silent waveform: {duration:.2f}s, {num_samples} samples")
        return np.zeros(num_samples, dtype=np.float32)

    def save_audio(
        self, waveform: np.ndarray, sample_rate: int, path: str
    ) -> None:
        """
        Save a waveform to a WAV file (for debugging).

        Args:
            waveform: Audio waveform array.
            sample_rate: Sample rate.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, waveform, sample_rate)
        logger.debug(f"Audio saved: {path}")
