"""PatchTST for forecasting-based time-series anomaly detection.

Custom PyTorch implementation of PatchTST (Nie et al., 2023).

Architecture:
    Input (batch, seq_len, num_channels)
        → Channel-independent: each channel processed separately
        → Patch embedding (unfold into patches + linear projection)
        → Positional encoding (learnable)
        → Transformer encoder (N layers of multi-head self-attention)
        → Flatten + Linear head → forecast next H steps per channel
    Output (batch, forecast_horizon, num_channels)

Anomaly logic:
    anomaly_score = MSE(actual_future, predicted_future) per sample
    is_anomaly = anomaly_score > threshold

Data setup for training:
    Given a window of length seq_len + forecast_horizon:
        input  = window[:seq_len]           → model input
        target = window[seq_len:]           → ground truth for forecast
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def load_patchtst_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load PatchTST config from model_config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["patchtst"]


# ──────────────────────────────────────────────
# Patch Embedding
# ──────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    """Unfold time series into patches and project to d_model.

    Input:  (batch * num_channels, seq_len)
    Output: (batch * num_channels, num_patches, d_model)
    """

    def __init__(
        self,
        patch_length: int,
        stride: int,
        d_model: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = (seq_len - patch_length) // stride + 1

        # Linear projection from patch to d_model
        self.projection = nn.Linear(patch_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create and project patches.

        Args:
            x: (batch * num_channels, seq_len)

        Returns:
            patches: (batch * num_channels, num_patches, d_model)
        """
        # Unfold into patches: (batch * num_channels, num_patches, patch_length)
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        # Project to d_model
        patches = self.projection(patches)
        return patches


# ──────────────────────────────────────────────
# Positional Encoding (Learnable)
# ──────────────────────────────────────────────


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for patch sequences."""

    def __init__(self, num_patches: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: (batch * num_channels, num_patches, d_model)

        Returns:
            (batch * num_channels, num_patches, d_model)
        """
        return self.dropout(x + self.position_embedding)


# ──────────────────────────────────────────────
# Transformer Encoder Block
# ──────────────────────────────────────────────


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder layer: multi-head attention + feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: (batch * num_channels, num_patches, d_model)

        Returns:
            (batch * num_channels, num_patches, d_model)
        """
        # Multi-head self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out

        # Feed-forward with pre-norm
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


# ──────────────────────────────────────────────
# Full PatchTST Model
# ──────────────────────────────────────────────


class PatchTST(nn.Module):
    """PatchTST for multivariate time-series forecasting.

    Channel-independent design: each sensor channel is processed
    independently through shared Transformer weights.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_horizon: int,
        patch_length: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            seq_len=seq_len,
        )
        num_patches = self.patch_embedding.num_patches

        # Positional encoding
        self.positional_encoding = LearnablePositionalEncoding(
            num_patches=num_patches,
            d_model=d_model,
            dropout=dropout,
        )

        # Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_encoder_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Forecast head: flatten patches → predict forecast_horizon steps
        self.head = nn.Linear(num_patches * d_model, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast future values from input sequence.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            forecast: (batch, forecast_horizon, input_dim)
        """
        batch_size = x.shape[0]

        # Channel-independent: reshape to (batch * input_dim, seq_len)
        # Each channel is treated as an independent sample
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = x.reshape(batch_size * self.input_dim, self.seq_len)

        # Patch embedding: (batch * input_dim, num_patches, d_model)
        x = self.patch_embedding(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        # Flatten patches and forecast
        x = x.reshape(batch_size * self.input_dim, -1)  # (batch * input_dim, num_patches * d_model)
        x = self.head(x)  # (batch * input_dim, forecast_horizon)

        # Reshape back: (batch, input_dim, forecast_horizon) → (batch, forecast_horizon, input_dim)
        x = x.reshape(batch_size, self.input_dim, self.forecast_horizon)
        x = x.permute(0, 2, 1)

        return x

    def get_forecast_error(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Compute per-sample MSE forecast error.

        Processes data in batches to avoid MPS/GPU memory exhaustion
        (channel-independent design multiplies effective batch by input_dim).

        Args:
            x: (batch, seq_len, input_dim) — input window
            y: (batch, forecast_horizon, input_dim) — actual future values
            batch_size: Number of windows per forward pass.

        Returns:
            errors: (batch,) — mean squared error per sample
        """
        self.eval()
        all_errors = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                forecast = self.forward(x_batch)
                errors = torch.mean((y_batch - forecast) ** 2, dim=(1, 2))
                all_errors.append(errors)
        return torch.cat(all_errors, dim=0)

    def predict_anomaly(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        threshold: float,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomalies based on forecast error threshold.

        Args:
            x: (batch, seq_len, input_dim) — input window
            y: (batch, forecast_horizon, input_dim) — actual future values
            threshold: Forecast error threshold.
            batch_size: Number of windows per forward pass.

        Returns:
            Tuple of:
                anomaly_scores: (batch,) — forecast errors
                is_anomaly:     (batch,) — binary predictions
        """
        errors = self.get_forecast_error(x, y, batch_size=batch_size)
        anomaly_scores = errors.cpu().numpy()
        is_anomaly = (anomaly_scores > threshold).astype(np.int64)
        return anomaly_scores, is_anomaly


# ──────────────────────────────────────────────
# Data preparation for PatchTST
# ──────────────────────────────────────────────


def create_forecast_windows(
    windows: np.ndarray,
    forecast_horizon: int | None = None,
    config_path: Path = MODEL_CONFIG_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sliding windows into input/target pairs for forecasting.

    Takes overlapping windows of length seq_len and creates pairs where:
        input  = window[i][:seq_len - forecast_horizon]
        target = window[i][seq_len - forecast_horizon:]

    Args:
        windows: (N, seq_len, num_features) — from feature engineering
        forecast_horizon: Number of future steps to predict.
        config_path: Path to model_config.yaml.

    Returns:
        Tuple of:
            inputs:  (N, seq_len - forecast_horizon, num_features)
            targets: (N, forecast_horizon, num_features)
    """
    if forecast_horizon is None:
        config = load_patchtst_config(config_path)
        forecast_horizon = config["forecast_horizon"]

    seq_len = windows.shape[1]

    if forecast_horizon >= seq_len:
        raise ValueError(
            f"forecast_horizon ({forecast_horizon}) must be < seq_len ({seq_len})"
        )

    inputs = windows[:, :seq_len - forecast_horizon, :]
    targets = windows[:, seq_len - forecast_horizon:, :]

    logger.info(
        "Created forecast pairs: inputs=%s, targets=%s",
        inputs.shape,
        targets.shape,
    )
    return inputs.astype(np.float32), targets.astype(np.float32)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def build_patchtst(
    input_dim: int,
    seq_len: int,
    config_path: Path = MODEL_CONFIG_PATH,
) -> PatchTST:
    """Build PatchTST model from config.

    Args:
        input_dim: Number of features per timestep (from feature engineering).
        seq_len: Input sequence length (seq_len - forecast_horizon).
        config_path: Path to model_config.yaml.

    Returns:
        PatchTST model instance.
    """
    config = load_patchtst_config(config_path)

    model = PatchTST(
        input_dim=input_dim,
        seq_len=seq_len,
        forecast_horizon=config["forecast_horizon"],
        patch_length=config["patch_length"],
        stride=config["stride"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_patches = model.patch_embedding.num_patches
    logger.info(
        "Built PatchTST: input_dim=%d, seq_len=%d, forecast_horizon=%d, "
        "patches=%d (len=%d, stride=%d), d_model=%d, heads=%d, layers=%d, "
        "params=%d (trainable=%d)",
        input_dim,
        seq_len,
        config["forecast_horizon"],
        num_patches,
        config["patch_length"],
        config["stride"],
        config["d_model"],
        config["n_heads"],
        config["num_encoder_layers"],
        total_params,
        trainable_params,
    )
    return model
