"""
Temporal Forgery Boundary Detector (Novel Module 3).

Uses multi-scale dilated 1D convolutions followed by a CRF layer
to sequence-label video frames as REAL (0), FAKE (1), or BOUNDARY (2).
Enables precise temporal localization of forgery transitions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class ForgeryBoundary:
    """A detected forgery boundary segment."""

    start_time: float    # Seconds
    end_time: float      # Seconds
    start_frame: int     # Frame index
    end_frame: int       # Frame index
    tag: str             # "REAL", "FAKE", or "BOUNDARY"
    confidence: float    # Average confidence


class TemporalForgeryBoundaryDetector(nn.Module):
    """
    TFBD: 1D CNN + CRF for frame-level forgery tagging.

    Architecture:
      Input features [B, T, D]
      → Multi-scale dilated 1D convolutions (captures local+global patterns)
      → Emission scores [B, T, num_tags]
      → CRF layer (enforces valid tag sequences)
      → Optimal tag sequence via Viterbi decoding
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_tags: int = 3,
        cnn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension.
            num_tags: Number of CRF tags (3: REAL, FAKE, BOUNDARY).
            cnn_channels: Hidden channels for each dilated conv layer.
            kernel_size: Convolution kernel size.
            dilations: Dilation rates for multi-scale receptive field.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_tags = num_tags

        if cnn_channels is None:
            cnn_channels = [128, 128, 128]
        if dilations is None:
            dilations = [1, 2, 4]

        assert len(cnn_channels) == len(dilations), (
            "cnn_channels and dilations must have same length"
        )

        # Multi-scale dilated 1D CNN stack
        layers = []
        in_channels = input_dim
        for channels, dilation in zip(cnn_channels, dilations):
            padding = (kernel_size - 1) * dilation // 2
            layers.extend([
                nn.Conv1d(
                    in_channels, channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                ),
                nn.GELU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
            ])
            in_channels = channels

        self.cnn = nn.Sequential(*layers)

        # Emission score projection
        self.emission = nn.Conv1d(in_channels, num_tags, kernel_size=1)

        # CRF layer
        self._crf = None  # Lazy init to avoid import issues
        self._crf_initialized = False

        logger.info(
            f"TFBD initialized (input={input_dim}, tags={num_tags}, "
            f"channels={cnn_channels}, dilations={dilations})"
        )

    def _init_crf(self):
        """Lazy initialize CRF layer."""
        if self._crf_initialized:
            return

        try:
            from torchcrf import CRF
            self._crf = CRF(self.num_tags, batch_first=True)
            self._crf_initialized = True
            logger.info("CRF layer initialized (pytorch-crf)")
        except ImportError:
            logger.warning(
                "pytorch-crf not installed. TFBD will use softmax fallback. "
                "Install with: pip install pytorch-crf"
            )
            self._crf = None
            self._crf_initialized = True

    def forward(
        self,
        features: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through 1D CNN + CRF.

        Args:
            features: Input features [B, T, D].
            tags: Ground truth tag sequence [B, T] (for training).
            mask: Boolean mask [B, T] (True = valid frame).

        Returns:
            Tuple of:
              - If training (tags provided): (loss, emissions)
                loss is the negative log-likelihood from CRF.
              - If inference (no tags): (best_tags, emissions)
                best_tags [B, T] is the Viterbi-decoded sequence.
        """
        self._init_crf()

        # CNN expects [B, D, T]
        x = features.transpose(1, 2)         # [B, D, T]
        x = self.cnn(x)                       # [B, channels, T]
        emissions = self.emission(x)          # [B, num_tags, T]
        emissions = emissions.transpose(1, 2)  # [B, T, num_tags]

        if self._crf is not None:
            if mask is None:
                mask = torch.ones(
                    emissions.shape[:2], dtype=torch.bool, device=emissions.device
                )

            if tags is not None:
                # Training: compute CRF loss (negative log-likelihood)
                loss = -self._crf(emissions, tags, mask=mask, reduction="mean")
                return loss, emissions
            else:
                # Inference: Viterbi decoding
                best_tags = self._crf.decode(emissions, mask=mask)
                # Convert list of lists to tensor
                best_tags = torch.tensor(best_tags, device=emissions.device)
                return best_tags, emissions
        else:
            # Fallback: softmax classification (no CRF)
            if tags is not None:
                loss = nn.functional.cross_entropy(
                    emissions.reshape(-1, self.num_tags),
                    tags.reshape(-1),
                    ignore_index=-1,
                )
                return loss, emissions
            else:
                best_tags = emissions.argmax(dim=-1)
                return best_tags, emissions

    def extract_boundaries(
        self,
        tag_sequence: torch.Tensor,
        timestamps: List[float],
    ) -> List[ForgeryBoundary]:
        """
        Convert a predicted tag sequence into forgery boundary segments.

        Args:
            tag_sequence: [T] tensor of predicted tags.
            timestamps: List of frame timestamps in seconds.

        Returns:
            List of ForgeryBoundary dataclasses.
        """
        tag_names = {0: "REAL", 1: "FAKE", 2: "BOUNDARY"}
        tags = tag_sequence.cpu().numpy()
        boundaries = []

        if len(tags) == 0:
            return boundaries

        current_tag = int(tags[0])
        segment_start = 0

        for i in range(1, len(tags)):
            if int(tags[i]) != current_tag:
                # Segment ended
                boundaries.append(ForgeryBoundary(
                    start_time=timestamps[segment_start],
                    end_time=timestamps[i - 1],
                    start_frame=segment_start,
                    end_frame=i - 1,
                    tag=tag_names.get(current_tag, "UNKNOWN"),
                    confidence=1.0,  # Will be refined by emissions
                ))
                current_tag = int(tags[i])
                segment_start = i

        # Last segment
        boundaries.append(ForgeryBoundary(
            start_time=timestamps[segment_start],
            end_time=timestamps[-1],
            start_frame=segment_start,
            end_frame=len(tags) - 1,
            tag=tag_names.get(current_tag, "UNKNOWN"),
            confidence=1.0,
        ))

        return boundaries
