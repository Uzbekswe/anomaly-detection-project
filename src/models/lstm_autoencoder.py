"""LSTM Autoencoder for time-series anomaly detection.

Architecture:
    Input (batch, seq_len, input_dim)
        → LSTM Encoder → hidden states
        → Bottleneck (linear projection to latent_dim)
        → RepeatVector (expand latent to seq_len)
        → LSTM Decoder → hidden states
        → Linear output → reconstructed input (batch, seq_len, input_dim)

Anomaly logic:
    reconstruction_error = MSE(input, output) per sample
    is_anomaly = reconstruction_error > threshold
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


def load_lstm_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load LSTM autoencoder config from model_config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["lstm_autoencoder"]


# ──────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────


class Encoder(nn.Module):
    """LSTM encoder that compresses a sequence into a latent vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project final hidden state to latent space
        self.fc_latent = nn.Linear(hidden_dim * self.num_directions, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent vector.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            latent: (batch, latent_dim)
        """
        # output: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        _, (h_n, _) = self.lstm(x)

        # Take the last layer's hidden state from all directions
        if self.num_directions == 2:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # (batch, hidden_dim)
            h_backward = h_n[-1]  # (batch, hidden_dim)
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = h_n[-1]  # (batch, hidden_dim)

        latent = self.fc_latent(h_last)  # (batch, latent_dim)
        return latent


# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────


class Decoder(nn.Module):
    """LSTM decoder that reconstructs a sequence from a latent vector."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Expand latent to LSTM input dimension
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project LSTM output back to original feature dimension
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to sequence.

        Args:
            latent: (batch, latent_dim)

        Returns:
            reconstructed: (batch, seq_len, output_dim)
        """
        # Expand latent and repeat across sequence length
        expanded = self.fc_expand(latent)  # (batch, hidden_dim)
        repeated = expanded.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)

        # Decode
        decoded, _ = self.lstm(repeated)  # (batch, seq_len, hidden_dim)
        output = self.fc_output(decoded)  # (batch, seq_len, output_dim)
        return output


# ──────────────────────────────────────────────
# Full Autoencoder
# ──────────────────────────────────────────────


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time-series anomaly detection.

    Reconstructs the input sequence. High reconstruction error = anomaly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
        seq_len: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            reconstructed: (batch, seq_len, input_dim)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            errors: (batch,) — mean squared error per sample
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE per sample: mean over seq_len and input_dim
            errors = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return errors

    def predict_anomaly(
        self,
        x: torch.Tensor,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomalies based on reconstruction error threshold.

        Args:
            x: (batch, seq_len, input_dim)
            threshold: Reconstruction error threshold.

        Returns:
            Tuple of:
                anomaly_scores: (batch,) — reconstruction errors
                is_anomaly:     (batch,) — binary predictions
        """
        errors = self.get_reconstruction_error(x)
        anomaly_scores = errors.cpu().numpy()
        is_anomaly = (anomaly_scores > threshold).astype(np.int64)
        return anomaly_scores, is_anomaly


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def build_lstm_autoencoder(
    input_dim: int,
    seq_len: int,
    config_path: Path = MODEL_CONFIG_PATH,
) -> LSTMAutoencoder:
    """Build LSTM Autoencoder from config.

    Args:
        input_dim: Number of features per timestep (determined by feature engineering).
        seq_len: Sequence length (window_size from config).
        config_path: Path to model_config.yaml.

    Returns:
        LSTMAutoencoder model instance.
    """
    config = load_lstm_config(config_path)

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        seq_len=seq_len,
        bidirectional=config["bidirectional"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built LSTMAutoencoder: input_dim=%d, seq_len=%d, "
        "hidden=%d, latent=%d, layers=%d, params=%d (trainable=%d)",
        input_dim,
        seq_len,
        config["hidden_dim"],
        config["latent_dim"],
        config["num_layers"],
        total_params,
        trainable_params,
    )
    return model
