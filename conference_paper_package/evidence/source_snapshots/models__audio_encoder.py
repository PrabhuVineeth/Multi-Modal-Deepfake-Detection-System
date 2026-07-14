"""
Audio encoder module.

Uses Wav2Vec2 from HuggingFace Transformers to extract contextualized
audio embeddings from speech waveforms.
"""

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class AudioEncoder(nn.Module):
    """
    Wav2Vec2-based audio feature encoder.

    Extracts contextualized audio embeddings from raw waveforms.
    Supports partial fine-tuning by freezing early transformer layers.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        output_dim: int = 768,
        projection_dim: int = 512,
        freeze_layers: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_name: HuggingFace model name for Wav2Vec2.
            output_dim: Wav2Vec2 hidden size (768 for base).
            projection_dim: Projection head output dimension.
            freeze_layers: Number of transformer layers to freeze.
            dropout: Dropout rate for the projection head.
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.freeze_layers = freeze_layers

        # Lazy load — model is loaded on first forward pass
        self._encoder = None
        self._processor = None

        # Projection head: map Wav2Vec2 output to fusion dimension
        self.projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

        logger.info(
            f"AudioEncoder initialized (model={model_name}, "
            f"projection={output_dim}→{projection_dim}, "
            f"freeze_layers={freeze_layers})"
        )

    def _load_backbone(self):
        """Lazy load the Wav2Vec2 backbone from HuggingFace."""
        if self._encoder is not None:
            return

        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        logger.info(f"Loading Wav2Vec2 backbone: {self.model_name}")
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        # Move encoder to the device of the projection parameters
        device = next(self.projection.parameters()).device
        try:
            self._encoder = Wav2Vec2Model.from_pretrained(self.model_name).to(device)
            logger.info("Pretrained Wav2Vec2 model loaded successfully from HuggingFace.")
        except Exception as e:
            logger.error(f"Failed to load pretrained Wav2Vec2 model: {e}")
            logger.info("Initializing fallback Wav2Vec2Model with random weights.")
            from transformers import Wav2Vec2Config
            config = Wav2Vec2Config.from_pretrained(self.model_name)
            self._encoder = Wav2Vec2Model(config).to(device)

        # Freeze feature extractor (CNN) layers — always frozen
        self._encoder.feature_extractor._freeze_parameters()

        # Freeze the first N transformer layers
        if self.freeze_layers > 0:
            for i, layer in enumerate(self._encoder.encoder.layers):
                if i < self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            trainable = sum(
                p.numel() for p in self._encoder.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self._encoder.parameters())
            logger.info(
                f"Wav2Vec2: froze {self.freeze_layers} layers, "
                f"{trainable}/{total} params trainable "
                f"({trainable/total:.1%})"
            )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract audio embeddings from raw waveform.

        Args:
            waveform: Raw audio waveform tensor [B, T] (float32, normalized).
            attention_mask: Optional mask for padded sequences [B, T].

        Returns:
            Projected audio embeddings [B, T_a, projection_dim].
            T_a is the compressed temporal dimension from Wav2Vec2's CNN.
        """
        self._load_backbone()

        # Wav2Vec2 expects float32 input
        waveform = waveform.float()

        # Extract features through Wav2Vec2
        outputs = self._encoder(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # last_hidden_state: [B, T_a, 768]
        hidden = outputs.last_hidden_state

        # Project to fusion dimension
        projected = self.projection(hidden)  # [B, T_a, projection_dim]

        return projected

    def get_output_length(self, input_length: int) -> int:
        """
        Compute output sequence length given input waveform length.

        Wav2Vec2's CNN compresses the temporal dimension by ~320x.

        Args:
            input_length: Number of input audio samples.

        Returns:
            Number of output time steps.
        """
        # Wav2Vec2 feature extractor downsamples by factor of ~320
        # (7 CNN layers: strides 5,2,2,2,2,2,2 → 5*2^6 = 320)
        return max(1, (input_length - 400) // 320 + 1)
