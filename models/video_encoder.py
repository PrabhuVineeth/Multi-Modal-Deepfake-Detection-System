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
        # False if _load_backbone had to fall back to random init. Benchmarks must
        # check this -- a randomly initialized backbone silently invalidates results.
        self.pretrained = False

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
        """Lazy load the vision backbone (ViT / Swin) from HuggingFace."""
        if self._encoder is not None:
            return

        from transformers import AutoModel, AutoImageProcessor, AutoConfig

        logger.info(f"Loading vision backbone: {self.model_name}")
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load AutoImageProcessor for {self.model_name}: {e}")

        # Move encoder to the device of the projection parameters
        device = next(self.projection.parameters()).device
        try:
            self._encoder = AutoModel.from_pretrained(self.model_name).to(device)
            self.pretrained = True
            logger.info(f"Pretrained weights loaded for {self.model_name}")
        except Exception as e:
            # Random init used to happen silently here, which made every benchmark
            # run against this backbone meaningless. Keep the fallback so inference
            # still runs, but make it loud and record it on self.pretrained.
            logger.error(
                f"Could not load pretrained weights for {self.model_name}: {e}. "
                f"Falling back to RANDOM INITIALIZATION - any benchmark using this "
                f"backbone is invalid."
            )
            from transformers import ViTConfig, ViTModel
            config = ViTConfig(image_size=224, patch_size=16, hidden_size=768)
            self._encoder = ViTModel(config).to(device)
            self.pretrained = False

        # Freeze embeddings if present
        if hasattr(self._encoder, "embeddings"):
            for param in self._encoder.embeddings.parameters():
                param.requires_grad = False

        # Freeze first N layers
        if self.freeze_layers > 0:
            layers_container = getattr(self._encoder, "encoder", self._encoder)
            layer_list = getattr(layers_container, "layer", getattr(layers_container, "layers", None))
            if layer_list is not None:
                for i, layer in enumerate(layer_list):
                    if i < self.freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

            trainable = sum(
                p.numel() for p in self._encoder.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self._encoder.parameters())
            logger.info(
                f"Vision backbone: froze {self.freeze_layers} layers, "
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
            
            # Chunk processing to prevent VRAM overflow/thrashing
            chunk_size = 32
            embeddings_list = []
            for i in range(0, B * T, chunk_size):
                chunk = flat[i : i + chunk_size]
                embeddings_list.append(self._encode_single(chunk))
            
            embeddings = torch.cat(embeddings_list, dim=0)
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
            Embeddings [B, projection_dim].
        """
        self._load_backbone()
        outputs = self._encoder(
            pixel_values=pixel_values,
            output_hidden_states=False,
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat_embedding = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state.ndim == 3:
            if "swin" in self.model_name.lower():
                feat_embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                feat_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            feat_embedding = outputs.last_hidden_state

        # Project to target dimension (512)
        projected = self.projection(feat_embedding)  # [B, projection_dim]
        return projected

    def extract_backbone_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Raw pretrained backbone embedding, bypassing the pooler and projection head.

        `_encode_single` reads `pooler_output` and then applies `self.projection`.
        Neither carries pretrained weights: the ViT pooler is absent from the
        HuggingFace checkpoints and the projection head is only learned during
        end-to-end training. For frozen-backbone benchmarking both are random,
        so this method reads the CLS token (ViT) or mean-pooled tokens (Swin)
        directly off `last_hidden_state` instead.

        Args:
            pixel_values: [B, T, C, H, W] or [B, C, H, W].

        Returns:
            [B, T, hidden_size] or [B, hidden_size] (hidden_size is 768 for
            ViT-Base and Swin-Tiny alike).
        """
        self._load_backbone()

        squeeze_time = pixel_values.dim() == 4
        if squeeze_time:
            pixel_values = pixel_values.unsqueeze(1)

        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * T, C, H, W)

        chunk_size = 32
        feats = []
        is_swin = "swin" in self.model_name.lower()
        for i in range(0, B * T, chunk_size):
            outputs = self._encoder(
                pixel_values=flat[i : i + chunk_size], output_hidden_states=False
            )
            hidden = outputs.last_hidden_state  # [b, tokens, hidden]
            # Swin has no CLS token; ViT keeps it at index 0.
            feats.append(hidden.mean(dim=1) if is_swin else hidden[:, 0, :])

        embeddings = torch.cat(feats, dim=0).reshape(B, T, -1)
        return embeddings.squeeze(1) if squeeze_time else embeddings

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
            
            # Chunk processing to prevent VRAM overflow/thrashing
            chunk_size = 32
            embeddings_list = []
            for i in range(0, B * T, chunk_size):
                chunk = resized[i : i + chunk_size]
                embeddings_list.append(self.encoder._encode_single(chunk))
            
            embeddings = torch.cat(embeddings_list, dim=0)
            return embeddings.reshape(B, T, -1)
        else:
            resized = self.resize(mouth_rois)
            return self.encoder._encode_single(resized)
