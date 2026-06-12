"""
Full end-to-end Deepfake Forensic Detection Model.

Composes all sub-modules into a single nn.Module:
  AudioEncoder → VideoEncoder/MouthEncoder → CrossModalFusion
  → ForensicAnalyzers → EvidenceAggregator → MismatchLocalizer → TFBD
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from config import ModelConfig, model_config


@dataclass
class ForensicOutput:
    """Bundled output from the full forensic model."""

    # Classification
    logits: torch.Tensor = None            # [B, 1]
    probability: torch.Tensor = None       # [B, 1] (calibrated)
    prediction: int = 0                     # 0=REAL, 1=FAKE

    # Per-analyzer scores
    lip_sync_score: torch.Tensor = None    # [B, 1]
    identity_score: torch.Tensor = None    # [B, 1]
    temporal_score: torch.Tensor = None    # [B, 1]
    av_sync_score: torch.Tensor = None     # [B, 1]

    # Per-frame evidence
    per_frame_evidence: torch.Tensor = None  # [B, T, D]
    channel_weights: torch.Tensor = None     # [B, 4]

    # Mismatch maps
    mismatch_maps: Dict[str, torch.Tensor] = field(default_factory=dict)

    # TFBD
    boundary_tags: torch.Tensor = None       # [B, T]
    boundary_emissions: torch.Tensor = None  # [B, T, num_tags]
    boundary_loss: Optional[torch.Tensor] = None

    # Attention maps (for explainability)
    speech_lip_attn: torch.Tensor = None
    voice_identity_attn: torch.Tensor = None
    av_sync_attn: torch.Tensor = None


class DeepfakeForensicModel(nn.Module):
    """
    Complete multimodal deepfake forensic detection model.

    End-to-end architecture:
      Audio Waveform → AudioEncoder → audio_features
      Face Frames    → VideoEncoder → face_features
      Mouth ROIs     → MouthEncoder → mouth_features
      (audio, face, mouth) → CrossModalFusion → fused representation
      fused → ForensicAnalyzers → per-channel evidence + scores
      evidence → EvidenceAggregator → classification + unified evidence
      evidence → MismatchLocalizer → spatial anomaly maps
      evidence → TFBD → temporal boundary tags
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or model_config

        # Import here to avoid circular imports
        from models.audio_encoder import AudioEncoder
        from models.video_encoder import VideoEncoder, MouthEncoder
        from models.cross_attention import CrossModalFusion
        from models.forensic_analyzers import ForensicAnalyzerBundle
        from models.evidence_aggregation import ForensicEvidenceAggregator
        from models.mismatch_localization import JointMismatchLocalizer
        from models.tfbd import TemporalForgeryBoundaryDetector
        from models.calibration import TemperatureScaler

        # ── Encoders ──
        self.audio_encoder = AudioEncoder(
            model_name=self.config.wav2vec2_model_name,
            output_dim=self.config.audio_embed_dim,
            projection_dim=self.config.fusion_hidden_dim,
            freeze_layers=self.config.freeze_audio_layers,
            dropout=self.config.attention_dropout,
        )
        self.video_encoder = VideoEncoder(
            model_name=self.config.vit_model_name,
            output_dim=self.config.visual_embed_dim,
            projection_dim=self.config.fusion_hidden_dim,
            freeze_layers=self.config.freeze_visual_layers,
            dropout=self.config.attention_dropout,
        )
        self.mouth_encoder = MouthEncoder(
            model_name=self.config.vit_model_name,
            output_dim=self.config.visual_embed_dim,
            projection_dim=self.config.fusion_hidden_dim,
            freeze_layers=self.config.freeze_visual_layers,
            dropout=self.config.attention_dropout,
        )

        # ── Fusion ──
        self.fusion = CrossModalFusion(
            embed_dim=self.config.fusion_hidden_dim,
            num_heads=self.config.num_attention_heads,
            num_layers=self.config.num_cross_attention_layers,
            dropout=self.config.attention_dropout,
        )

        # ── Forensic Analyzers ──
        self.analyzers = ForensicAnalyzerBundle(
            input_dim=self.config.fusion_hidden_dim,
            hidden_dim=self.config.analyzer_hidden_dim,
            dropout=self.config.analyzer_dropout,
        )

        # ── Evidence Aggregation ──
        self.evidence_aggregator = ForensicEvidenceAggregator(
            num_channels=4,
            evidence_dim=1,
            hidden_dim=self.config.evidence_dim,
            fusion_dim=self.config.fusion_hidden_dim,
            num_heads=self.config.num_evidence_heads,
            dropout=self.config.analyzer_dropout,
        )

        # ── Mismatch Localization ──
        self.mismatch_localizer = JointMismatchLocalizer(
            input_dim=self.config.fusion_hidden_dim,
            hidden_dim=self.config.analyzer_hidden_dim,
            dropout=self.config.analyzer_dropout,
        )

        # ── TFBD ──
        self.tfbd = TemporalForgeryBoundaryDetector(
            input_dim=self.config.fusion_hidden_dim,
            num_tags=self.config.tfbd_num_tags,
            cnn_channels=self.config.tfbd_cnn_channels,
            kernel_size=self.config.tfbd_kernel_size,
            dilations=self.config.tfbd_dilations,
            dropout=self.config.analyzer_dropout,
        )

        # ── Calibration ──
        self.temperature_scaler = TemperatureScaler(
            init_temperature=self.config.temperature_init,
        )

        self._log_model_size()

    def _log_model_size(self):
        """Log total parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"DeepfakeForensicModel: {total:,} total params, "
            f"{trainable:,} trainable ({trainable/max(total,1):.1%})"
        )

    def forward(
        self,
        audio_waveform: torch.Tensor,
        face_frames: torch.Tensor,
        mouth_rois: torch.Tensor,
        boundary_tags: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> ForensicOutput:
        """
        Full forward pass through the forensic detection model.

        Args:
            audio_waveform: Raw audio [B, num_samples].
            face_frames: Face crops [B, T, C, H, W].
            mouth_rois: Mouth ROI crops [B, T, C, H, W].
            boundary_tags: Optional TFBD ground truth [B, T] (training only).
            frame_mask: Optional frame validity mask [B, T].

        Returns:
            ForensicOutput with all predictions and evidence.
        """
        output = ForensicOutput()

        # ── Step 1: Feature extraction ──
        audio_features = self.audio_encoder(audio_waveform)     # [B, T_a, D]
        face_features = self.video_encoder(face_frames)          # [B, T_v, D]
        mouth_features = self.mouth_encoder(mouth_rois)          # [B, T_v, D]

        # ── Step 2: Cross-modal fusion ──
        multimodal = self.fusion(audio_features, face_features, mouth_features)

        output.speech_lip_attn = multimodal.speech_lip_attn
        output.voice_identity_attn = multimodal.voice_identity_attn
        output.av_sync_attn = multimodal.av_sync_attn

        # ── Step 3: Forensic analysis ──
        evidence_dict = self.analyzers(
            speech_lip_features=multimodal.speech_lip_features,
            voice_identity_features=multimodal.voice_identity_features,
            fused_features=multimodal.fused_features,
            av_sync_features=multimodal.av_sync_features,
        )

        # Extract global scores
        output.lip_sync_score = evidence_dict["lip_sync"][1]
        output.identity_score = evidence_dict["identity"][1]
        output.temporal_score = evidence_dict["temporal"][1]
        output.av_sync_score = evidence_dict["av_sync"][1]

        # ── Step 4: Evidence aggregation + classification ──
        logits, unified_evidence, channel_weights = self.evidence_aggregator(
            evidence_dict, multimodal.fused_features
        )
        output.logits = logits
        output.probability = self.temperature_scaler.calibrate(logits)
        output.per_frame_evidence = unified_evidence
        output.channel_weights = channel_weights

        # ── Step 5: Mismatch localization ──
        output.mismatch_maps = self.mismatch_localizer(unified_evidence)

        # ── Step 6: Temporal boundary detection ──
        if boundary_tags is not None:
            # Training: compute CRF loss
            boundary_loss, emissions = self.tfbd(
                unified_evidence, tags=boundary_tags, mask=frame_mask
            )
            output.boundary_loss = boundary_loss
            output.boundary_emissions = emissions
        else:
            # Inference: Viterbi decoding
            best_tags, emissions = self.tfbd(
                unified_evidence, mask=frame_mask
            )
            output.boundary_tags = best_tags
            output.boundary_emissions = emissions

        return output
