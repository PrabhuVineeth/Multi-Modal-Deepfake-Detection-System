"""
Video encoder module.

Uses Vision Transformer (ViT) from HuggingFace Transformers to extract
visual embeddings from face-cropped frames and mouth ROI crops.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


class VideoEncoder(nn.Module):
    """
    ViT-based visual feature encoder for face frames.

    Extracts per-frame visual embeddings. Supports partial fine-tuning
    by freezing early transformer layers.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        output_dim: int = 768,
        projection_dim: int = 512,
        freeze_layers: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_name: HuggingFace model name for ViT.
            output_dim: ViT hidden size (768 for base).
            projection_dim: Projection head output dimension.
            freeze_layers: Number of transformer layers to freeze.
            dropout: Dropout rate for the projection head.
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.freeze_layers = freeze_layers

        self._encoder = None
        self._processor = None

        # Projection head
        self.projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

        logger.info(
            f"VideoEncoder initialized (model={model_name}, "
            f"projection={output_dim}→{projection_dim})"
        )

    def _load_backbone(self):
        """Lazy load the ViT backbone from HuggingFace."""
        if self._encoder is not None:
            return

        from transformers import ViTModel, ViTImageProcessor

        logger.info(f"Loading ViT backbone: {self.model_name}")
        self._processor = ViTImageProcessor.from_pretrained(self.model_name)
        self._encoder = ViTModel.from_pretrained(self.model_name)

        # Freeze embeddings
        for param in self._encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze first N layers
        if self.freeze_layers > 0:
            for i, layer in enumerate(self._encoder.encoder.layer):
                if i < self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            trainable = sum(
                p.numel() for p in self._encoder.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self._encoder.parameters())
            logger.info(
                f"ViT: froze {self.freeze_layers} layers, "
                f"{trainable}/{total} params trainable ({trainable/total:.1%})"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract visual embeddings from face frame images.

        Args:
            pixel_values: Preprocessed face frames [B, T, C, H, W] or [B, C, H, W].
                          If 5D, T frames are processed individually and stacked.

        Returns:
            Visual embeddings [B, T, projection_dim] or [B, projection_dim].
        """
        self._load_backbone()

        if pixel_values.dim() == 5:
            # Batch of frame sequences: [B, T, C, H, W]
            B, T, C, H, W = pixel_values.shape
            # Reshape to [B*T, C, H, W] for efficient processing
            flat = pixel_values.reshape(B * T, C, H, W)
            embeddings = self._encode_single(flat)  # [B*T, projection_dim]
            return embeddings.reshape(B, T, -1)  # [B, T, projection_dim]
        else:
            # Single batch of frames: [B, C, H, W]
            return self._encode_single(pixel_values)  # [B, projection_dim]

    def _encode_single(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of single images.

        Args:
            pixel_values: [B, C, H, W] tensor.

        Returns:
            Embeddings [B, projection_dim] using CLS token.
        """
        outputs = self._encoder(
            pixel_values=pixel_values,
            output_hidden_states=False,
        )

        # Use the CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Project
        projected = self.projection(cls_embedding)  # [B, projection_dim]
        return projected

    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get full patch-level embeddings (for spatial analysis).

        Args:
            pixel_values: [B, C, H, W] tensor.

        Returns:
            Patch embeddings [B, num_patches+1, projection_dim] including CLS.
        """
        self._load_backbone()

        outputs = self._encoder(
            pixel_values=pixel_values,
            output_hidden_states=False,
        )

        # All tokens including CLS: [B, num_patches+1, 768]
        all_tokens = outputs.last_hidden_state
        projected = self.projection(all_tokens)
        return projected


class MouthEncoder(nn.Module):
    """
    Specialized encoder for mouth ROI crops.

    Uses the same ViT backbone but applies additional preprocessing
    suitable for the smaller 96×96 mouth region images.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        output_dim: int = 768,
        projection_dim: int = 512,
        freeze_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim

        # Upscale small mouth ROI to ViT input size
        self.resize = nn.Upsample(
            size=(224, 224), mode="bilinear", align_corners=False
        )

        # Use the same VideoEncoder internally
        self.encoder = VideoEncoder(
            model_name=model_name,
            output_dim=output_dim,
            projection_dim=projection_dim,
            freeze_layers=freeze_layers,
            dropout=dropout,
        )

        logger.info("MouthEncoder initialized (wraps VideoEncoder with resize)")

    def forward(self, mouth_rois: torch.Tensor) -> torch.Tensor:
        """
        Encode mouth ROI images.

        Args:
            mouth_rois: [B, T, C, H, W] or [B, C, H, W] mouth region crops.
                       H, W may be 96×96 (will be upscaled to 224×224).

        Returns:
            Mouth embeddings [B, T, projection_dim] or [B, projection_dim].
        """
        if mouth_rois.dim() == 5:
            B, T, C, H, W = mouth_rois.shape
            flat = mouth_rois.reshape(B * T, C, H, W)
            resized = self.resize(flat)
            embeddings = self.encoder._encode_single(resized)  # [B*T, proj]
            return embeddings.reshape(B, T, -1)
        else:
            resized = self.resize(mouth_rois)
            return self.encoder._encode_single(resized)
