"""Model loading and inference for the serving layer.

This module is responsible for loading a trained model and its associated
artifacts (scaler, threshold, etc.) from an MLflow run and performing inference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow.tracking import MlflowClient

from src.features.engineer import MinMaxScaler, StandardScaler
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.patchtst import PatchTST

logger = logging.getLogger(__name__)

# Define a union type for all possible model classes
AnyModel = LSTMAutoencoder | PatchTST | IsolationForestAnomalyDetector
AnyScaler = MinMaxScaler | StandardScaler


@dataclass
class Model:
    """A container for all components of a loaded model."""
    model: AnyModel
    scaler: AnyScaler
    threshold: float
    feature_columns: list[str]
    model_version: str
    model_type: str


def load_model_from_run(run_id: str) -> Model:
    """Load a model, scaler, and other artifacts from a specific MLflow run.

    Args:
        run_id: The ID of the MLflow run to load from.

    Returns:
        A Model object containing all necessary components for inference.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    model_type = run.data.tags.get("model_type", "unknown")
    logger.info("Loading model '%s' (type: %s) from run_id: %s", run.data.tags.get("mlflow.runName"), model_type, run_id)

    # Download all artifacts
    artifact_path = client.download_artifacts(run_id, ".")

    # 1. Load scaler
    with open(Path(artifact_path) / "scaler.json") as f:
        scaler_dict = json.load(f)
    if "min_vals" in scaler_dict:
        scaler = MinMaxScaler.from_dict(scaler_dict)
    else:
        scaler = StandardScaler.from_dict(scaler_dict)
    logger.info("Loaded scaler from scaler.json")

    # 2. Load model-specific artifacts
    if model_type == "isolation_forest":
        model_artifact = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        threshold = model_artifact["contamination"]
        model = model_artifact["model"]
        feature_columns = model_artifact["feature_columns"]
    elif model_type in ["lstm_ae", "patchtst"]:
        model_artifact_path = next(Path(artifact_path).glob("*.pt"))
        checkpoint = torch.load(model_artifact_path, map_location="cpu")
        
        # Dynamically build model from its own artifacts
        model = _build_torch_model(model_type, checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        threshold = checkpoint["threshold"]
        feature_columns = checkpoint["feature_columns"]
    else:
        raise ValueError(f"Unknown model type '{model_type}' for run_id {run_id}")

    logger.info("Successfully loaded model and artifacts for run_id: %s", run_id)
    
    return Model(
        model=model,
        scaler=scaler,
        threshold=threshold,
        feature_columns=feature_columns,
        model_version=f"{model_type}:{run_id[:8]}",
        model_type=model_type,
    )

def _build_torch_model(model_type: str, checkpoint: dict) -> torch.nn.Module:
    """Helper to reconstruct a PyTorch model from its checkpoint."""
    if model_type == "lstm_ae":
        from src.models.lstm_autoencoder import build_lstm_autoencoder
        return build_lstm_autoencoder(
            input_dim=checkpoint["input_dim"], 
            seq_len=checkpoint["seq_len"]
        )
    elif model_type == "patchtst":
        from src.models.patchtst import build_patchtst
        return build_patchtst(
            input_dim=checkpoint["input_dim"],
            seq_len=checkpoint["seq_len"],
            forecast_horizon=checkpoint["forecast_horizon"],
        )
    raise ValueError(f"Unknown torch model type: {model_type}")


class AnomalyPredictor:
    """Manages a loaded model and exposes a prediction interface."""

    def __init__(self, model: Model | None = None):
        self.model_container = model
        self.device = torch.device("cpu") # Serving on CPU for simplicity

    @property
    def is_loaded(self) -> bool:
        return self.model_container is not None

    def predict(self, window: np.ndarray | list[list[float]]) -> dict:
        """Run inference on a single sensor window.

        Args:
            window: Raw sensor data of shape (seq_len, num_raw_sensors).

        Returns:
            A dictionary containing the prediction results.
        """
        if not self.is_loaded or self.model_container is None:
            raise RuntimeError("Model not loaded.")

        mc = self.model_container
        
        # 1. Pre-process: transform the raw window
        window_np = np.array(window, dtype=np.float32)
        normalized_window = self._transform_window(window_np, mc.scaler, mc.feature_columns)

        # 2. Predict based on model type
        if mc.model_type == "isolation_forest":
            window_flat = normalized_window.reshape(1, -1)
            anomaly_score = float(mc.model.predict_anomaly_score(window_flat)[0])
            is_anomaly = bool(anomaly_score < mc.threshold) # IF score is inverted
        elif mc.model_type in ["lstm_ae", "patchtst"]:
            x = torch.FloatTensor(normalized_window).unsqueeze(0).to(self.device)
            
            if mc.model_type == "lstm_ae":
                scores, preds = mc.model.predict_anomaly(x, mc.threshold)
            else: # PatchTST
                # Need to create forecast horizon for target
                inputs, targets = mc.model.create_forecast_windows(x, mc.model.forecast_horizon)
                scores, preds = mc.model.predict_anomaly(inputs, targets, mc.threshold)
                
            anomaly_score = float(scores[0])
            is_anomaly = bool(preds[0])
        else:
            raise TypeError(f"Unsupported model type for prediction: {mc.model_type}")

        # 3. Post-process: calculate confidence
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

    @staticmethod
    def _transform_window(
        raw_window: np.ndarray,
        scaler: AnyScaler,
        feature_columns: list[str],
    ) -> np.ndarray:
        """Apply the same normalization as training to a single window."""
        # This assumes the raw_window has the feature columns in the correct order
        df = pd.DataFrame(raw_window, columns=feature_columns)
        df_transformed = scaler.transform(df)
        return df_transformed.values.astype(np.float32)

def find_latest_run_id(model_name: str, model_stage: str) -> str:
    """Find the latest run ID for a given model name and stage."""
    client = MlflowClient()
    try:
        latest_version = client.get_latest_versions(name=model_name, stages=[model_stage])[0]
        return latest_version.run_id
    except IndexError:
        raise ValueError(f"No model named '{model_name}' found in stage '{model_stage}'")
