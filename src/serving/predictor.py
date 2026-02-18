"""Model loading and inference for the serving layer.

Loads a trained model and its artifacts (scaler, threshold, feature columns)
from either local artifact files or an MLflow run, then exposes a unified
prediction interface for all model types.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from src.features.engineer import MinMaxScaler, StandardScaler
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.patchtst import PatchTST, create_forecast_windows

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

# Type aliases
AnyModel = LSTMAutoencoder | PatchTST | IsolationForestAnomalyDetector
AnyScaler = MinMaxScaler | StandardScaler


@dataclass
class Model:
    """Container for a loaded model and all its inference artifacts."""

    model: AnyModel
    scaler: AnyScaler
    threshold: float
    feature_columns: list[str]
    model_version: str
    model_type: str


# ──────────────────────────────────────────────
# Scaler helpers
# ──────────────────────────────────────────────


def _load_scaler_from_dict(scaler_dict: dict) -> AnyScaler:
    """Reconstruct a scaler from its serialised dictionary."""
    if "min_vals" in scaler_dict:
        return MinMaxScaler.from_dict(scaler_dict)
    return StandardScaler.from_dict(scaler_dict)


def _load_scaler_from_file(path: Path) -> AnyScaler:
    """Load a scaler from a JSON file on disk."""
    with open(path) as f:
        return _load_scaler_from_dict(json.load(f))


# ──────────────────────────────────────────────
# PyTorch model reconstruction
# ──────────────────────────────────────────────


def _build_torch_model(model_type: str, checkpoint: dict) -> torch.nn.Module:
    """Reconstruct a PyTorch model architecture from checkpoint metadata."""
    from src.models.lstm_autoencoder import build_lstm_autoencoder
    from src.models.patchtst import build_patchtst

    if model_type == "lstm_ae":
        return build_lstm_autoencoder(
            input_dim=checkpoint["input_dim"],
            seq_len=checkpoint["seq_len"],
        )
    if model_type == "patchtst":
        return build_patchtst(
            input_dim=checkpoint["input_dim"],
            seq_len=checkpoint["seq_len"],
            horizon=checkpoint["forecast_horizon"],
        )
    raise ValueError(f"Unknown torch model type: {model_type}")


# ──────────────────────────────────────────────
# Loading from local artifacts
# ──────────────────────────────────────────────


def load_model_from_local(
    model_type: str,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Model:
    """Load a model from locally-saved artifact files.

    Supports:
      - lstm_ae / lstm_autoencoder →  lstm_autoencoder.pt  (torch checkpoint)
      - patchtst                   →  patchtst.pt          (torch checkpoint)
      - isolation_forest           →  isolation_forest.joblib

    Each artifact embeds its own scaler, threshold and feature_columns.
    A standalone ``scaler.json`` is used as fallback if the artifact does
    not contain one.
    """
    # Normalise common aliases
    _aliases = {"lstm_autoencoder": "lstm_ae"}
    canonical = _aliases.get(model_type, model_type)
    logger.info("Loading model '%s' from %s", canonical, artifact_dir)

    if canonical in ("lstm_ae", "patchtst"):
        filename = "lstm_autoencoder.pt" if canonical == "lstm_ae" else "patchtst.pt"
        checkpoint = torch.load(artifact_dir / filename, map_location="cpu")

        model = _build_torch_model(canonical, checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        scaler = _load_scaler_from_dict(checkpoint["scaler"])
        threshold = checkpoint["threshold"]
        feature_columns = checkpoint["feature_columns"]

    elif canonical == "isolation_forest":
        data = joblib.load(artifact_dir / "isolation_forest.joblib")
        model = data["model"]
        scaler = _load_scaler_from_dict(data["scaler"])
        threshold = data["contamination"]
        feature_columns = data["feature_columns"]

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info("Loaded '%s' — threshold=%.4f, features=%d", canonical, threshold, len(feature_columns))

    return Model(
        model=model,
        scaler=scaler,
        threshold=threshold,
        feature_columns=feature_columns,
        model_version=f"{canonical}:local",
        model_type=canonical,
    )


# ──────────────────────────────────────────────
# Loading from MLflow
# ──────────────────────────────────────────────


def load_model_from_run(run_id: str) -> Model:
    """Load a model and artifacts from a specific MLflow run.

    The run must have been produced by ``src.models.train`` which logs
    scaler.json and the model checkpoint / joblib artifact.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    model_type = run.data.tags.get("model_type", "unknown")
    logger.info(
        "Loading model '%s' (type: %s) from run %s",
        run.data.tags.get("mlflow.runName"),
        model_type,
        run_id,
    )

    artifact_path = Path(client.download_artifacts(run_id, "."))

    # Scaler
    scaler = _load_scaler_from_file(artifact_path / "scaler.json")

    # Model-specific artefacts
    if model_type == "isolation_forest":
        data = joblib.load(artifact_path / "isolation_forest.joblib")
        model = data["model"]
        threshold = data["contamination"]
        feature_columns = data["feature_columns"]

    elif model_type in ("lstm_ae", "patchtst"):
        pt_path = next(artifact_path.glob("*.pt"))
        checkpoint = torch.load(pt_path, map_location="cpu")

        model = _build_torch_model(model_type, checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        threshold = checkpoint["threshold"]
        feature_columns = checkpoint["feature_columns"]
    else:
        raise ValueError(f"Unknown model type '{model_type}' in run {run_id}")

    logger.info("Loaded model from MLflow run %s", run_id)

    return Model(
        model=model,
        scaler=scaler,
        threshold=threshold,
        feature_columns=feature_columns,
        model_version=f"{model_type}:{run_id[:8]}",
        model_type=model_type,
    )


def find_latest_run_id(model_name: str, model_stage: str) -> str:
    """Find the latest MLflow model-registry run ID for a given name/stage."""
    client = MlflowClient()
    try:
        latest = client.get_latest_versions(name=model_name, stages=[model_stage])[0]
        return latest.run_id
    except IndexError:
        raise ValueError(f"No model named '{model_name}' found in stage '{model_stage}'")


# ──────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────


class AnomalyPredictor:
    """Manages a loaded model and exposes a unified prediction interface."""

    def __init__(self, model: Model | None = None):
        self.model_container = model
        self.device = torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self.model_container is not None

    # ── Inference ────────────────────────────

    def predict(self, window: np.ndarray | list[list[float]]) -> dict:
        """Run inference on a single sensor window.

        Args:
            window: Sensor data of shape ``(seq_len, num_features)``.

        Returns:
            Dict with keys: anomaly_score, is_anomaly, confidence, model_version.
        """
        if not self.is_loaded or self.model_container is None:
            raise RuntimeError("Model not loaded.")

        mc = self.model_container

        # 1. Pre-process
        window_np = np.array(window, dtype=np.float32)
        norm = self._transform_window(window_np, mc.scaler, mc.feature_columns)

        # 2. Predict
        if mc.model_type == "isolation_forest":
            anomaly_score, is_anomaly = self._predict_isolation_forest(mc, norm)
        elif mc.model_type == "lstm_ae":
            anomaly_score, is_anomaly = self._predict_lstm_ae(mc, norm)
        elif mc.model_type == "patchtst":
            anomaly_score, is_anomaly = self._predict_patchtst(mc, norm)
        else:
            raise TypeError(f"Unsupported model type: {mc.model_type}")

        # 3. Confidence
        distance = abs(anomaly_score - mc.threshold)
        confidence = min(1.0, distance / mc.threshold) if mc.threshold > 0 else 0.5

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "model_version": mc.model_version,
        }

    def predict_batch(self, windows: list[np.ndarray | list[list[float]]]) -> list[dict]:
        """Run inference on a batch of windows."""
        return [self.predict(w) for w in windows]

    # ── Per-model prediction helpers ─────────

    def _predict_isolation_forest(self, mc: Model, norm: np.ndarray) -> tuple[float, bool]:
        """IF expects 3D (N, seq_len, features); flatten_windows is handled internally."""
        window_3d = norm[np.newaxis, ...]  # (1, seq_len, features)
        scores = mc.model.get_anomaly_scores(window_3d)
        anomaly_score = float(scores[0])
        is_anomaly = bool(anomaly_score > mc.threshold)
        return anomaly_score, is_anomaly

    def _predict_lstm_ae(self, mc: Model, norm: np.ndarray) -> tuple[float, bool]:
        x = torch.FloatTensor(norm).unsqueeze(0).to(self.device)
        scores, preds = mc.model.predict_anomaly(x, mc.threshold)
        return float(scores[0]), bool(preds[0])

    def _predict_patchtst(self, mc: Model, norm: np.ndarray) -> tuple[float, bool]:
        # create_forecast_windows is a module-level function, not a model method
        window_3d = norm[np.newaxis, ...]  # (1, seq_len, features)
        inputs, targets = create_forecast_windows(window_3d, horizon=mc.model.forecast_horizon)
        x = torch.FloatTensor(inputs).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        scores, preds = mc.model.predict_anomaly(x, y, mc.threshold)
        return float(scores[0]), bool(preds[0])

    # ── Pre-processing ───────────────────────

    @staticmethod
    def _transform_window(
        raw_window: np.ndarray,
        scaler: AnyScaler,
        feature_columns: list[str],
    ) -> np.ndarray:
        """Apply the same normalisation as training to a single window."""
        df = pd.DataFrame(raw_window, columns=feature_columns)
        df_transformed = scaler.transform(df)
        return df_transformed.values.astype(np.float32)
