"""
Forensic analyzer heads.

Four specialized analysis heads that examine specific aspects of
potential deepfake manipulation:
  1. LipSyncAnalyzer — lip-speech consistency
  2. IdentityAnalyzer — voice-face identity consistency
  3. TemporalAnalyzer — temporal coherence
  4. AVSyncAnalyzer — audio-video synchronization
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger


class LipSyncAnalyzer(nn.Module):
    """
    Analyzes lip-speech synchronization consistency.

    Operates on the speech-lip cross-attention features to produce
    a per-frame lip sync score and a global score.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        # Per-frame analysis
        self.frame_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Global pooling + score
        self.global_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Speech-lip cross-attention features [B, T, D].

        Returns:
            Tuple of:
              - per_frame_evidence: [B, T, 1] per-frame scores
              - global_score: [B, 1] video-level lip sync score
        """
        per_frame = self.frame_head(features)         # [B, T, 1]

        # Global score from mean-pooled features
        pooled = features.mean(dim=1)                  # [B, D]
        global_score = self.global_head(pooled)        # [B, 1]

        return per_frame, global_score


class IdentityAnalyzer(nn.Module):
    """
    Analyzes voice-face identity consistency.

    Detects mismatches between the speaker's voice characteristics
    and facial identity (common in face-swap deepfakes).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.frame_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.global_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Voice-identity cross-attention features [B, T, D].

        Returns:
            per_frame_evidence [B, T, 1], global_score [B, 1].
        """
        per_frame = self.frame_head(features)
        pooled = features.mean(dim=1)
        global_score = self.global_head(pooled)
        return per_frame, global_score


class TemporalAnalyzer(nn.Module):
    """
    Analyzes temporal coherence of visual features.

    Uses 1D convolutions to detect temporal discontinuities and
    unnatural transitions between frames.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        # Temporal convolution stack (operates on T dimension)
        self.temporal_conv = nn.Sequential(
            # [B, D, T] → [B, hidden, T]
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Per-frame score from conv output
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Global score
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Fused multimodal features [B, T, D].

        Returns:
            per_frame_evidence [B, T, 1], global_score [B, 1].
        """
        # Transpose for conv1d: [B, T, D] → [B, D, T]
        x = features.transpose(1, 2)
        x = self.temporal_conv(x)  # [B, hidden, T]
        x = x.transpose(1, 2)     # [B, T, hidden]

        per_frame = self.frame_head(x)        # [B, T, 1]

        # Global from temporal average
        pooled = x.mean(dim=1)                 # [B, hidden]
        global_score = self.global_head(pooled)  # [B, 1]

        return per_frame, global_score


class AVSyncAnalyzer(nn.Module):
    """
    Analyzes audio-video temporal synchronization.

    Detects timing mismatches between audio events and corresponding
    visual events (e.g., lip movements lagging behind speech).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.frame_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.global_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: AV sync cross-attention features [B, T, D].

        Returns:
            per_frame_evidence [B, T, 1], global_score [B, 1].
        """
        per_frame = self.frame_head(features)
        pooled = features.mean(dim=1)
        global_score = self.global_head(pooled)
        return per_frame, global_score


class ForensicAnalyzerBundle(nn.Module):
    """
    Convenience wrapper that bundles all four forensic analyzers.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.lip_sync = LipSyncAnalyzer(input_dim, hidden_dim, dropout)
        self.identity = IdentityAnalyzer(input_dim, hidden_dim, dropout)
        self.temporal = TemporalAnalyzer(input_dim, hidden_dim, dropout)
        self.av_sync = AVSyncAnalyzer(input_dim, hidden_dim, dropout)

        logger.info(
            f"ForensicAnalyzerBundle initialized "
            f"(input_dim={input_dim}, hidden_dim={hidden_dim})"
        )

    def forward(
        self,
        speech_lip_features: torch.Tensor,
        voice_identity_features: torch.Tensor,
        fused_features: torch.Tensor,
        av_sync_features: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run all four forensic analyzers.

        Returns:
            Dict mapping analyzer name → (per_frame_evidence, global_score).
        """
        return {
            "lip_sync": self.lip_sync(speech_lip_features),
            "identity": self.identity(voice_identity_features),
            "temporal": self.temporal(fused_features),
            "av_sync": self.av_sync(av_sync_features),
        }
