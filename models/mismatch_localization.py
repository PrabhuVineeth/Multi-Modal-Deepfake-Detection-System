"""
Joint mismatch localization module (Novel Module 2).

Produces spatial mismatch maps that highlight which frames exhibit
inconsistencies across different forensic dimensions. Enables
frame-level localization of manipulated segments.
"""

from typing import Dict

import torch
import torch.nn as nn
from loguru import logger


class JointMismatchLocalizer(nn.Module):
    """
    Localizes manipulated frames using multi-channel evidence vectors.

    Produces per-frame anomaly scores across four dimensions:
    lip inconsistencies, identity mismatches, temporal breaks,
    and AV sync issues.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_channels: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Dimension of fused features.
            hidden_dim: Hidden layer dimension.
            num_channels: Number of anomaly map channels.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_channels = num_channels
        self.channel_names = ["lip_sync", "identity", "temporal", "av_sync"]

        # Shared temporal encoder (1D conv across frames)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Per-channel anomaly heads
        self.anomaly_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            for name in self.channel_names
        })

        # Combined anomaly map
        self.combiner = nn.Sequential(
            nn.Linear(num_channels, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"JointMismatchLocalizer initialized "
            f"(input={input_dim}, hidden={hidden_dim}, channels={num_channels})"
        )

    def forward(
        self,
        unified_evidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Produce per-frame mismatch maps.

        Args:
            unified_evidence: [B, T, input_dim] per-frame evidence from
                the evidence aggregator.

        Returns:
            Dict mapping channel name to anomaly map [B, T, 1]:
              - 'lip_sync': lip inconsistency scores
              - 'identity': identity mismatch scores
              - 'temporal': temporal discontinuity scores
              - 'av_sync': AV sync anomaly scores
              - 'combined': overall anomaly score per frame
        """
        # Temporal encoding
        x = unified_evidence.transpose(1, 2)  # [B, D, T]
        x = self.temporal_encoder(x)           # [B, hidden, T]
        x = x.transpose(1, 2)                 # [B, T, hidden]

        # Per-channel anomaly maps
        maps = {}
        channel_scores = []
        for name in self.channel_names:
            score = self.anomaly_heads[name](x)  # [B, T, 1]
            maps[name] = score
            channel_scores.append(score)

        # Combined anomaly map
        stacked = torch.cat(channel_scores, dim=-1)  # [B, T, 4]
        maps["combined"] = self.combiner(stacked)      # [B, T, 1]

        return maps
