"""Model loading and inference for the serving layer.

Loads the trained model (and scaler) from MLflow registry or local artifacts,
runs inference using the SAME feature transforms as training.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from src.features.engineer import MinMaxScaler, StandardScaler, transform_single_window
from src.models.lstm_autoencoder import LSTMAutoencoder, build_lstm_autoencoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG_PATH = PROJECT_ROOT / "configs" / "serving_config.yaml"

# Regex to match ${VAR:-default} patterns in YAML values
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def resolve_env_vars(obj: dict | list | str | int | float | bool | None) -> dict | list | str | int | float | bool | None:
    """Recursively resolve ${VAR:-default} patterns in a loaded YAML config.

    Works on strings, dicts, and lists. Non-string values pass through unchanged.
    """
    if isinstance(obj, str):
        def _replace(match: re.Match) -> str:
            env_key = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_key, default_val)
        return _ENV_VAR_PATTERN.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def load_serving_config(config_path: Path = SERVING_CONFIG_PATH) -> dict:
    """Load serving configuration with environment variable resolution."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return resolve_env_vars(config)


class AnomalyPredictor:
    """Loads a trained model and runs inference.

    Handles model loading from local artifacts (saved during training)
    or from MLflow model registry.
    """

    def __init__(self) -> None:
        self.model: LSTMAutoencoder | None = None
        self.scaler: MinMaxScaler | StandardScaler | None = None
        self.threshold: float = 0.65
        self.model_version: str = "not_loaded"
        self.feature_columns: list[str] = []
        self.device = torch.device("cpu")
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_from_artifact(self, artifact_path: Path) -> None:
        """Load model from a local .pt artifact saved during training.

        Args:
            artifact_path: Path to the .pt file containing model state,
                threshold, scaler, and feature columns.
        """
        logger.info("Loading model from artifact: %s", artifact_path)

        checkpoint = torch.load(artifact_path, map_location=self.device, weights_only=False)

        # Build model with saved dimensions
        input_dim = checkpoint["input_dim"]
        seq_len = checkpoint["seq_len"]
        self.model = build_lstm_autoencoder(input_dim=input_dim, seq_len=seq_len)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load scaler
        scaler_data = checkpoint["scaler"]
        if "min_vals" in scaler_data:
            self.scaler = MinMaxScaler.from_dict(scaler_data)
        else:
            self.scaler = StandardScaler.from_dict(scaler_data)

        self.threshold = checkpoint["threshold"]
        self.feature_columns = checkpoint["feature_columns"]
        self.model_version = "lstm_autoencoder_local"
        self._is_loaded = True

        logger.info(
            "Model loaded: input_dim=%d, seq_len=%d, threshold=%.4f, features=%d",
            input_dim,
            seq_len,
            self.threshold,
            len(self.feature_columns),
        )

    def load_from_mlflow(
        self,
        model_name: str | None = None,
        model_stage: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Load model from MLflow model registry.

        Args:
            model_name: Registered model name.
            model_stage: Model stage (Production, Staging, etc.).
            tracking_uri: MLflow tracking URI.
        """
        import mlflow

        config = load_serving_config()
        model_config = config["model"]

        model_name = model_name or model_config["name"]
        model_stage = model_stage or model_config["stage"]
        tracking_uri = tracking_uri or model_config["mlflow_tracking_uri"]

        # Config values are already resolved by load_serving_config()
        mlflow.set_tracking_uri(tracking_uri)

        # Try to load from registry, fall back to latest run artifact
        try:
            model_uri = f"models:/{model_name}/{model_stage}"
            logger.info("Loading model from MLflow registry: %s", model_uri)
            loaded = mlflow.pytorch.load_model(model_uri)
            self.model = loaded
            self.model.to(self.device)
            self.model.eval()
            self.model_version = f"{model_name}_{model_stage}"
            self._is_loaded = True
        except Exception as e:
            logger.warning("MLflow registry load failed: %s. Trying artifact fallback.", e)
            self._load_from_latest_run(model_name, tracking_uri)

    def _load_from_latest_run(self, model_name: str, tracking_uri: str) -> None:
        """Fall back to loading from the latest MLflow run artifact."""
        artifact_path = PROJECT_ROOT / "data" / "artifacts" / "lstm_autoencoder.pt"
        if artifact_path.exists():
            self.load_from_artifact(artifact_path)
        else:
            raise FileNotFoundError(
                f"No model found in MLflow registry or at {artifact_path}. "
                "Run training first: make train"
            )

    def load(self) -> None:
        """Auto-load model: try local artifact first, then MLflow.

        This is the default loading strategy for the serving layer.
        """
        # Try local artifact first (fastest, no network dependency)
        artifact_path = PROJECT_ROOT / "data" / "artifacts" / "lstm_autoencoder.pt"
        if artifact_path.exists():
            self.load_from_artifact(artifact_path)
            return

        # Try MLflow
        try:
            self.load_from_mlflow()
        except Exception as e:
            raise RuntimeError(
                f"Could not load model from any source: {e}. "
                "Run training first: make train"
            ) from e

    def predict(
        self,
        window: np.ndarray | list[list[float]],
    ) -> tuple[float, bool, float]:
        """Run inference on a single sensor window.

        Uses the SAME scaler as training (loaded from artifact).

        Args:
            window: Raw sensor data of shape (seq_len, num_raw_sensors).
                Can be numpy array or nested list.

        Returns:
            Tuple of (anomaly_score, is_anomaly, confidence).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to numpy if needed
        if isinstance(window, list):
            window = np.array(window, dtype=np.float32)

        # Validate window dimensions match model expectations
        if self.model is not None:
            expected_seq_len = self.model.seq_len
            expected_input_dim = self.model.input_dim
            if window.shape[0] != expected_seq_len:
                raise ValueError(
                    f"Window has {window.shape[0]} timesteps, "
                    f"expected {expected_seq_len}"
                )
            # Only check input_dim if no scaler (raw sensors may differ from model input)
            if self.scaler is None and window.shape[1] != expected_input_dim:
                raise ValueError(
                    f"Window has {window.shape[1]} features, "
                    f"expected {expected_input_dim}"
                )

        # Apply the same normalization as training
        if self.scaler is not None and self.feature_columns:
            window = transform_single_window(window, self.scaler, self.feature_columns)

        # Convert to tensor and add batch dimension
        x = torch.FloatTensor(window).unsqueeze(0).to(self.device)

        # Get anomaly score
        scores, preds = self.model.predict_anomaly(x, self.threshold)
        anomaly_score = float(scores[0])
        is_anomaly = bool(preds[0])

        # Confidence: how far the score is from the threshold (normalized)
        distance = abs(anomaly_score - self.threshold)
        confidence = min(1.0, distance / self.threshold) if self.threshold > 0 else 0.5

        return anomaly_score, is_anomaly, confidence

    def predict_batch(
        self,
        windows: list[np.ndarray | list[list[float]]],
    ) -> list[tuple[float, bool, float]]:
        """Run inference on a batch of windows.

        Args:
            windows: List of raw sensor windows.

        Returns:
            List of (anomaly_score, is_anomaly, confidence) tuples.
        """
        return [self.predict(w) for w in windows]
