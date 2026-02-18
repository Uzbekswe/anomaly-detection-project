"""Unit tests for src/serving/predictor.py and model definitions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.isolation_forest import (
    IsolationForestAnomalyDetector,
    build_isolation_forest,
    flatten_windows,
)
from src.models.lstm_autoencoder import Decoder, Encoder, LSTMAutoencoder, build_lstm_autoencoder
from src.models.patchtst import PatchTST, create_forecast_windows
from src.serving.predictor import AnomalyPredictor

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_windows() -> np.ndarray:
    """Sample 3D windows: (N=20, seq_len=30, features=14)."""
    np.random.seed(42)
    return np.random.randn(20, 30, 14).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Binary labels for 20 samples."""
    return np.array([0] * 15 + [1] * 5, dtype=np.int64)


# ──────────────────────────────────────────────
# LSTM Autoencoder
# ──────────────────────────────────────────────


class TestEncoder:
    def test_output_shape(self) -> None:
        encoder = Encoder(input_dim=14, hidden_dim=64, latent_dim=32,
                          num_layers=2, dropout=0.1, bidirectional=False)
        x = torch.randn(4, 30, 14)
        latent = encoder(x)
        assert latent.shape == (4, 32)

    def test_bidirectional_output(self) -> None:
        encoder = Encoder(input_dim=14, hidden_dim=64, latent_dim=32,
                          num_layers=2, dropout=0.1, bidirectional=True)
        x = torch.randn(4, 30, 14)
        latent = encoder(x)
        assert latent.shape == (4, 32)


class TestDecoder:
    def test_output_shape(self) -> None:
        decoder = Decoder(latent_dim=32, hidden_dim=64, output_dim=14,
                          num_layers=2, dropout=0.1, seq_len=30)
        latent = torch.randn(4, 32)
        output = decoder(latent)
        assert output.shape == (4, 30, 14)


class TestLSTMAutoencoder:
    def test_forward_shape(self) -> None:
        model = LSTMAutoencoder(input_dim=14, hidden_dim=64, latent_dim=32,
                                num_layers=2, dropout=0.1, seq_len=30)
        x = torch.randn(4, 30, 14)
        output = model(x)
        assert output.shape == x.shape

    def test_reconstruction_error_shape(self) -> None:
        model = LSTMAutoencoder(input_dim=14, hidden_dim=64, latent_dim=32,
                                num_layers=2, dropout=0.1, seq_len=30)
        x = torch.randn(4, 30, 14)
        errors = model.get_reconstruction_error(x)
        assert errors.shape == (4,)
        assert (errors >= 0).all()

    def test_predict_anomaly(self) -> None:
        model = LSTMAutoencoder(input_dim=14, hidden_dim=64, latent_dim=32,
                                num_layers=2, dropout=0.1, seq_len=30)
        x = torch.randn(4, 30, 14)
        scores, preds = model.predict_anomaly(x, threshold=0.5)
        assert scores.shape == (4,)
        assert preds.shape == (4,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_build_from_config(self) -> None:
        model = build_lstm_autoencoder(input_dim=14, seq_len=30)
        assert isinstance(model, LSTMAutoencoder)
        assert model.input_dim == 14
        assert model.seq_len == 30


# ──────────────────────────────────────────────
# Isolation Forest
# ──────────────────────────────────────────────


class TestFlattenWindows:
    def test_output_shape(self, sample_windows: np.ndarray) -> None:
        flat = flatten_windows(sample_windows)
        assert flat.shape == (20, 14 * 2)  # mean + std per feature

    def test_output_finite(self, sample_windows: np.ndarray) -> None:
        flat = flatten_windows(sample_windows)
        assert np.isfinite(flat).all()


class TestIsolationForestAnomalyDetector:
    def test_fit_and_predict(self, sample_windows: np.ndarray) -> None:
        detector = IsolationForestAnomalyDetector(n_estimators=50, contamination=0.1, random_state=42)
        detector.fit(sample_windows)
        assert detector._is_fitted

        scores, preds = detector.predict_anomaly(sample_windows)
        assert scores.shape == (20,)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_scores_normalized(self, sample_windows: np.ndarray) -> None:
        detector = IsolationForestAnomalyDetector(n_estimators=50, contamination=0.1, random_state=42)
        detector.fit(sample_windows)
        scores = detector.get_anomaly_scores(sample_windows)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_threshold_override(self, sample_windows: np.ndarray) -> None:
        detector = IsolationForestAnomalyDetector(n_estimators=50, contamination=0.1, random_state=42)
        detector.fit(sample_windows)
        # Very high threshold → no anomalies
        _, preds = detector.predict_anomaly(sample_windows, threshold=999.0)
        assert preds.sum() == 0

    def test_build_from_config(self) -> None:
        detector = build_isolation_forest(contamination=0.05)
        assert isinstance(detector, IsolationForestAnomalyDetector)
        assert detector.contamination == 0.05


# ──────────────────────────────────────────────
# PatchTST
# ──────────────────────────────────────────────


class TestPatchTST:
    def test_forward_shape(self) -> None:
        model = PatchTST(
            input_dim=14, seq_len=20, forecast_horizon=10,
            patch_length=8, stride=4, d_model=32,
            n_heads=4, num_encoder_layers=2, d_ff=64, dropout=0.1,
        )
        x = torch.randn(4, 20, 14)
        output = model(x)
        assert output.shape == (4, 10, 14)

    def test_forecast_error_shape(self) -> None:
        model = PatchTST(
            input_dim=14, seq_len=20, forecast_horizon=10,
            patch_length=8, stride=4, d_model=32,
            n_heads=4, num_encoder_layers=2, d_ff=64, dropout=0.1,
        )
        x = torch.randn(4, 20, 14)
        y = torch.randn(4, 10, 14)
        errors = model.get_forecast_error(x, y)
        assert errors.shape == (4,)

    def test_predict_anomaly(self) -> None:
        model = PatchTST(
            input_dim=14, seq_len=20, forecast_horizon=10,
            patch_length=8, stride=4, d_model=32,
            n_heads=4, num_encoder_layers=2, d_ff=64, dropout=0.1,
        )
        x = torch.randn(4, 20, 14)
        y = torch.randn(4, 10, 14)
        scores, preds = model.predict_anomaly(x, y, threshold=0.5)
        assert scores.shape == (4,)
        assert set(np.unique(preds)).issubset({0, 1})


class TestCreateForecastWindows:
    def test_output_shapes(self, sample_windows: np.ndarray) -> None:
        inputs, targets = create_forecast_windows(sample_windows, horizon=10)
        assert inputs.shape == (20, 20, 14)   # seq_len=30-10=20
        assert targets.shape == (20, 10, 14)

    def test_horizon_too_large_raises(self, sample_windows: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must be < seq_len"):
            create_forecast_windows(sample_windows, horizon=30)

    def test_inputs_targets_contiguous(self, sample_windows: np.ndarray) -> None:
        """inputs + targets should reconstruct original window."""
        inputs, targets = create_forecast_windows(sample_windows, horizon=10)
        reconstructed = np.concatenate([inputs, targets], axis=1)
        np.testing.assert_array_almost_equal(reconstructed, sample_windows)


