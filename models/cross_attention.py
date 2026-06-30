"""
Cross-modal attention module.

Implements three specialized cross-attention mechanisms for multimodal
deepfake forensics:
  1. Speech-Lip attention (audio ↔ mouth visual)
  2. Voice-Identity attention (audio ↔ face visual)
  3. AV Sync attention (audio temporal ↔ video temporal)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class MultimodalRepresentation:
    """Bundled output of the cross-modal attention fusion."""

    # Fused representations
    speech_lip_features: torch.Tensor = None     # [B, T, D]
    voice_identity_features: torch.Tensor = None  # [B, T, D]
    av_sync_features: torch.Tensor = None         # [B, T, D]
    fused_features: torch.Tensor = None            # [B, T, D]

    # Attention weights (for explainability)
    speech_lip_attn: torch.Tensor = None           # [B, heads, T_q, T_k]
    voice_identity_attn: torch.Tensor = None       # [B, heads, T_q, T_k]
    av_sync_attn: torch.Tensor = None              # [B, heads, T_q, T_k]


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with residual connection and layer norm.

    Implements: output = LayerNorm(query + MultiHeadAttention(query, key, value))
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_k, D]
            key_padding_mask: [B, T_k] mask for padded positions.

        Returns:
            Tuple of (output [B, T_q, D], attention_weights [B, num_heads, T_q, T_k]).
        """
        # Force MultiheadAttention dot-product to run in FP32 to prevent FP16 overflow/NaN under AMP
        with torch.amp.autocast('cuda', enabled=False):
            attn_out, attn_weights = self.attention(
                query.float(), key.float(), value.float(),
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # Return per-head weights
            )
        
        # Cast attention outputs back to the query tensor dtype
        attn_out = attn_out.to(query.dtype)
        attn_weights = attn_weights.to(query.dtype)
        
        x = self.layer_norm(query + self.dropout(attn_out))

        # FFN with residual
        x = self.ffn_norm(x + self.ffn(x))

        return x, attn_weights


class SpeechLipAttention(nn.Module):
    """
    Cross-attention between audio (speech) and mouth visual features.

    Query: audio features → attend to mouth region features.
    Detects lip-sync inconsistencies in deepfakes.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        audio_features: torch.Tensor,
        mouth_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: [B, T_a, D] audio embeddings.
            mouth_features: [B, T_v, D] mouth ROI embeddings.

        Returns:
            Tuple of (fused_features [B, T_a, D], attention_weights).
        """
        x = audio_features
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(
                query=x,
                key=mouth_features,
                value=mouth_features,
            )
        return x, attn_weights


class VoiceIdentityAttention(nn.Module):
    """
    Cross-attention between audio (voice) and face visual features.

    Query: audio features → attend to face region features.
    Detects voice-face identity mismatches in deepfakes.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        audio_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: [B, T_a, D] audio embeddings.
            face_features: [B, T_v, D] face visual embeddings.

        Returns:
            Tuple of (fused_features [B, T_a, D], attention_weights).
        """
        x = audio_features
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(
                query=x,
                key=face_features,
                value=face_features,
            )
        return x, attn_weights


class AVSyncAttention(nn.Module):
    """
    Cross-attention for temporal audio-video synchronization analysis.

    Bidirectional: audio → video AND video → audio, then combined.
    Detects temporal synchronization anomalies.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Audio queries video
        self.a2v_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Video queries audio
        self.v2a_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Combine bidirectional features
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: [B, T_a, D]
            visual_features: [B, T_v, D]

        Returns:
            Tuple of (sync_features [B, T, D], attention_weights).
            T = min(T_a, T_v) after alignment.
        """
        # Audio → Video cross-attention
        a2v = audio_features
        a2v_attn = None
        for layer in self.a2v_layers:
            a2v, a2v_attn = layer(
                query=a2v,
                key=visual_features,
                value=visual_features,
            )

        # Video → Audio cross-attention
        v2a = visual_features
        v2a_attn = None
        for layer in self.v2a_layers:
            v2a, v2a_attn = layer(
                query=v2a,
                key=audio_features,
                value=audio_features,
            )

        # Align temporal dimensions
        min_len = min(a2v.size(1), v2a.size(1))
        a2v = a2v[:, :min_len, :]
        v2a = v2a[:, :min_len, :]

        # Combine bidirectional features
        combined = torch.cat([a2v, v2a], dim=-1)  # [B, T, 2D]
        sync_features = self.combine(combined)     # [B, T, D]

        return sync_features, a2v_attn


class CrossModalFusion(nn.Module):
    """
    Full cross-modal fusion module.

    Orchestrates all three cross-attention mechanisms and produces
    a unified multimodal representation.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Three cross-attention modules
        self.speech_lip = SpeechLipAttention(embed_dim, num_heads, num_layers, dropout)
        self.voice_identity = VoiceIdentityAttention(embed_dim, num_heads, num_layers, dropout)
        self.av_sync = AVSyncAttention(embed_dim, num_heads, num_layers, dropout)

        # Fusion of the three cross-attention outputs
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        logger.info(
            f"CrossModalFusion initialized "
            f"(dim={embed_dim}, heads={num_heads}, layers={num_layers})"
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        face_features: torch.Tensor,
        mouth_features: torch.Tensor,
    ) -> MultimodalRepresentation:
        """
        Fuse audio, face, and mouth features through cross-attention.

        Args:
            audio_features: [B, T_a, D] from AudioEncoder.
            face_features: [B, T_v, D] from VideoEncoder.
            mouth_features: [B, T_v, D] from MouthEncoder.

        Returns:
            MultimodalRepresentation with all fused features and attention maps.
        """
        # 1. Speech-Lip: audio queries mouth
        speech_lip_out, speech_lip_attn = self.speech_lip(
            audio_features, mouth_features
        )

        # 2. Voice-Identity: audio queries face
        voice_id_out, voice_id_attn = self.voice_identity(
            audio_features, face_features
        )

        # 3. AV Sync: bidirectional audio ↔ visual
        av_sync_out, av_sync_attn = self.av_sync(
            audio_features, face_features
        )

        # Align temporal dimensions for fusion
        min_len = min(
            speech_lip_out.size(1),
            voice_id_out.size(1),
            av_sync_out.size(1),
        )
        sl = speech_lip_out[:, :min_len, :]
        vi = voice_id_out[:, :min_len, :]
        avs = av_sync_out[:, :min_len, :]

        # Concatenate and fuse
        combined = torch.cat([sl, vi, avs], dim=-1)  # [B, T, 3D]
        fused = self.fusion(combined)                  # [B, T, D]

        return MultimodalRepresentation(
            speech_lip_features=sl,
            voice_identity_features=vi,
            av_sync_features=avs,
            fused_features=fused,
            speech_lip_attn=speech_lip_attn,
            voice_identity_attn=voice_id_attn,
            av_sync_attn=av_sync_attn,
        )
