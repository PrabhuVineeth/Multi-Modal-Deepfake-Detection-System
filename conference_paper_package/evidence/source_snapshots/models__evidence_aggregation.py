"""
Evidence aggregation engine (Novel Module 1).

Attention-based weighted fusion of forensic evidence from all four
analyzers. Learns which evidence channels are most informative per
sample, and produces a unified evidence representation for
classification and boundary detection.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ForensicEvidenceAggregator(nn.Module):
    """
    Attention-based aggregation of multi-channel forensic evidence.

    Each evidence channel (lip_sync, identity, temporal, av_sync) has
    learned importance weights that adapt per-sample, making the model
    robust to cases where only some evidence is discriminative.
    """

    def __init__(
        self,
        num_channels: int = 4,
        evidence_dim: int = 1,
        hidden_dim: int = 128,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_channels: Number of forensic evidence channels (4).
            evidence_dim: Dimension of per-frame evidence from each analyzer.
            hidden_dim: Hidden dimension for channel attention.
            fusion_dim: Dimension of the fused multimodal features.
            num_heads: Number of attention heads for evidence weighting.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_channels = num_channels

        # Project each evidence channel to a common dimension
        self.channel_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(evidence_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_channels)
        ])

        # Channel-level attention: learn which evidence type matters most
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)

        # Combine fused features with aggregated evidence
        self.combiner = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Classification head on aggregated evidence
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary: real vs fake
        )

        logger.info(
            f"ForensicEvidenceAggregator initialized "
            f"(channels={num_channels}, hidden={hidden_dim}, fusion={fusion_dim})"
        )

    def forward(
        self,
        evidence_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        fused_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate forensic evidence and produce classification logits.

        Args:
            evidence_dict: Dict from ForensicAnalyzerBundle.forward().
                Keys: 'lip_sync', 'identity', 'temporal', 'av_sync'.
                Values: (per_frame_evidence [B,T,1], global_score [B,1]).
            fused_features: [B, T, fusion_dim] from CrossModalFusion.

        Returns:
            Tuple of:
              - logits: [B, 1] classification logits (pre-sigmoid)
              - unified_evidence: [B, T, fusion_dim] per-frame evidence
              - channel_weights: [B, num_channels] attention weights per channel
        """
        channel_names = ["lip_sync", "identity", "temporal", "av_sync"]

        # Collect global scores from each channel
        global_scores = []  # [B, 1] each
        per_frame_evidence = []  # [B, T, 1] each

        for name in channel_names:
            pf, gs = evidence_dict[name]
            global_scores.append(gs)
            per_frame_evidence.append(pf)

        # Stack global scores: [B, num_channels, 1]
        global_stack = torch.stack(global_scores, dim=1)  # [B, 4, 1]

        # Project each channel
        projected = []
        for i, proj in enumerate(self.channel_projections):
            projected.append(proj(global_stack[:, i, :]))  # [B, hidden_dim]

        # Stack projected: [B, num_channels, hidden_dim]
        projected_stack = torch.stack(projected, dim=1)

        # Self-attention across channels to learn importance (FP32 forced under AMP for numerical safety unless weights are FP16)
        dtype = self.channel_attention.in_proj_weight.dtype if hasattr(self.channel_attention, "in_proj_weight") and self.channel_attention.in_proj_weight is not None else projected_stack.dtype
        with torch.amp.autocast('cuda', enabled=False):
            p_stack_dtype = projected_stack.to(dtype)
            attended, attn_weights = self.channel_attention(
                p_stack_dtype, p_stack_dtype, p_stack_dtype,
                need_weights=True,
                average_attn_weights=True,
            )
        attended = attended.to(projected_stack.dtype)
        attn_weights = attn_weights.to(projected_stack.dtype)
        
        # attn_weights: [B, num_channels, num_channels]
        attended = self.channel_norm(projected_stack + attended)

        # Pool across channels: [B, hidden_dim]
        channel_pooled = attended.mean(dim=1)

        # Extract channel weights (average attention received)
        channel_weights = attn_weights.mean(dim=1)  # [B, num_channels]

        # Combine with fused features (per-frame)
        fused_pooled = fused_features.mean(dim=1)  # [B, fusion_dim]
        combined = torch.cat([fused_pooled, channel_pooled], dim=-1)  # [B, fusion+hidden]
        unified = self.combiner(combined)  # [B, fusion_dim]

        # Per-frame unified evidence
        # Expand unified to each frame and add per-frame evidence
        T = fused_features.size(1)
        unified_expanded = unified.unsqueeze(1).expand(-1, T, -1)  # [B, T, fusion]
        unified_evidence = fused_features + unified_expanded  # Residual

        # Classification logits
        logits = self.classifier(unified)  # [B, 1]

        return logits, unified_evidence, channel_weights
