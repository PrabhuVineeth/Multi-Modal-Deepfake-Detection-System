"""
Confidence calibration module.

Implements temperature scaling and Platt scaling for calibrating
model output probabilities to match true correctness rates.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class TemperatureScaler(nn.Module):
    """
    Learned temperature scaling for confidence calibration.

    Divides logits by a single scalar temperature parameter,
    optimized on a validation set to minimize NLL.
    """

    def __init__(self, init_temperature: float = 1.5):
        """
        Args:
            init_temperature: Initial temperature value (> 1 = soften,
                < 1 = sharpen predictions).
        """
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model logits [B, ...].

        Returns:
            Temperature-scaled logits [B, ...].
        """
        return logits / self.temperature.clamp(min=0.01)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrate logits and convert to probabilities.

        Args:
            logits: Raw logits [B, 1] (binary) or [B, C] (multi-class).

        Returns:
            Calibrated probabilities.
        """
        scaled = self.forward(logits)
        if scaled.shape[-1] == 1:
            return torch.sigmoid(scaled)
        else:
            return F.softmax(scaled, dim=-1)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Optimize temperature on validation data using LBFGS.

        Args:
            val_logits: Validation set logits [N, 1].
            val_labels: Validation set labels [N].
            lr: Learning rate for LBFGS.
            max_iter: Maximum optimization iterations.

        Returns:
            Final NLL loss value.
        """
        logger.info("Fitting temperature scaler on validation data...")
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        val_logits = val_logits.detach()
        val_labels = val_labels.detach().float()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            if scaled.shape[-1] == 1:
                loss = F.binary_cross_entropy_with_logits(
                    scaled.squeeze(), val_labels
                )
            else:
                loss = F.cross_entropy(scaled, val_labels.long())
            loss.backward()
            return loss

        optimizer.step(closure)

        final_loss = closure().item()
        logger.info(
            f"Temperature calibration complete: "
            f"T={self.temperature.item():.4f}, NLL={final_loss:.4f}"
        )
        return final_loss


class PlattScaler:
    """
    Platt scaling (logistic regression) calibration.

    Scikit-learn based fallback when LBFGS temperature scaling
    is not suitable.
    """

    def __init__(self):
        self._model = None

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
    ) -> None:
        """
        Fit logistic regression on validation logits.

        Args:
            val_logits: [N, 1] array of logits.
            val_labels: [N] array of binary labels.
        """
        from sklearn.linear_model import LogisticRegression

        if val_logits.ndim == 1:
            val_logits = val_logits.reshape(-1, 1)

        self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._model.fit(val_logits, val_labels)

        logger.info("Platt scaler fitted on validation data")

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to produce calibrated probabilities.

        Args:
            logits: [N, 1] array of logits.

        Returns:
            [N] array of calibrated probabilities.
        """
        if self._model is None:
            raise RuntimeError("PlattScaler has not been fitted. Call fit() first.")

        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)

        return self._model.predict_proba(logits)[:, 1]


class ConfidenceCalibrator:
    """
    High-level calibration interface that selects the best method.
    """

    def __init__(
        self,
        method: str = "temperature",
        init_temperature: float = 1.5,
    ):
        """
        Args:
            method: Calibration method ('temperature' or 'platt').
            init_temperature: Initial temperature for temperature scaling.
        """
        self.method = method

        if method == "temperature":
            self.scaler = TemperatureScaler(init_temperature)
        elif method == "platt":
            self.scaler = PlattScaler()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        logger.info(f"ConfidenceCalibrator initialized (method={method})")

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> None:
        """Fit calibration parameters on validation data."""
        if self.method == "temperature":
            self.scaler.fit(val_logits, val_labels)
        else:
            self.scaler.fit(
                val_logits.detach().cpu().numpy(),
                val_labels.detach().cpu().numpy(),
            )

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrate model logits to produce well-calibrated probabilities.

        Args:
            logits: Raw model logits [B, 1].

        Returns:
            Calibrated probabilities [B, 1].
        """
        if self.method == "temperature":
            return self.scaler.calibrate(logits)
        else:
            np_logits = logits.detach().cpu().numpy()
            np_probs = self.scaler.calibrate(np_logits)
            return torch.tensor(np_probs, dtype=torch.float32, device=logits.device)
