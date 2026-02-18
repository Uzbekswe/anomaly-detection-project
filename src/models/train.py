"""Unified training entry point for all three anomaly detection models.

Trains LSTM Autoencoder, Isolation Forest, and PatchTST under a single
MLflow experiment. Each model is logged as a separate run with params,
metrics, and artifacts.

Usage:
    python src/models/train.py --config configs/training_config.yaml
    python src/models/train.py --config configs/training_config.yaml --model lstm_autoencoder
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.ingest import ingest_train_data
from src.features.engineer import build_features, load_feature_config
from src.models.isolation_forest import build_isolation_forest
from src.models.lstm_autoencoder import build_lstm_autoencoder
from src.models.patchtst import build_patchtst, create_forecast_windows

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────


def load_training_config(config_path: Path) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# ──────────────────────────────────────────────
# Data splitting
# ──────────────────────────────────────────────


def split_data(
    windows: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    validation_size: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split windows into train/val/test sets preserving temporal order.

    No shuffling — time series data must maintain order.

    Returns:
        Dict with keys "train", "val", "test", each containing (windows, labels).
    """
    n = len(windows)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - validation_size / (1 - test_size)))

    splits = {
        "train": (windows[:val_start], labels[:val_start]),
        "val": (windows[val_start:test_start], labels[val_start:test_start]),
        "test": (windows[test_start:], labels[test_start:]),
    }

    for name, (w, labels_split) in splits.items():
        logger.info(
            "Split '%s': %d windows, anomaly_rate=%.1f%%",
            name,
            len(w),
            100 * labels_split.mean() if len(labels_split) > 0 else 0,
        )
    return splits


# ──────────────────────────────────────────────
# Threshold search
# ──────────────────────────────────────────────


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    search_min: float,
    search_max: float,
    search_steps: int,
) -> tuple[float, dict[str, float]]:
    """Search for threshold that maximizes F1 score.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Ground truth binary labels.
        search_min: Minimum threshold to try.
        search_max: Maximum threshold to try.
        search_steps: Number of thresholds to evaluate.

    Returns:
        Tuple of (best_threshold, best_metrics_dict).
    """
    thresholds = np.linspace(search_min, search_max, search_steps)
    best_f1 = -1.0
    best_threshold = search_min
    best_metrics: dict[str, float] = {}

    for t in thresholds:
        preds = (scores > t).astype(int)

        if preds.sum() == 0 or preds.sum() == len(preds):
            continue

        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {
                "f1": f1,
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
            }

    # AUC-ROC (threshold-independent)
    if len(np.unique(labels)) > 1:
        best_metrics["auc_roc"] = roc_auc_score(labels, scores)

    logger.info(
        "Optimal threshold=%.4f → F1=%.4f, Precision=%.4f, Recall=%.4f",
        best_threshold,
        best_metrics.get("f1", 0),
        best_metrics.get("precision", 0),
        best_metrics.get("recall", 0),
    )
    return best_threshold, best_metrics


# ──────────────────────────────────────────────
# MLflow helpers
# ──────────────────────────────────────────────


def setup_mlflow(config: dict) -> str:
    """Configure MLflow tracking and return experiment ID."""
    mlflow_config = config["mlflow"]

    tracking_uri = mlflow_config["tracking_uri"]
    # Resolve env var placeholder
    if tracking_uri.startswith("${"):
        env_key = tracking_uri.split(":-")[0].lstrip("${")
        default_val = tracking_uri.split(":-")[1].rstrip("}")
        tracking_uri = os.environ.get(env_key, default_val)

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(mlflow_config["experiment_name"])
    logger.info("MLflow tracking URI: %s, experiment: %s", tracking_uri, experiment.name)
    return experiment.experiment_id


def log_model_size(model: nn.Module) -> float:
    """Compute and return model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


# ──────────────────────────────────────────────
# Train: LSTM Autoencoder
# ──────────────────────────────────────────────


def train_lstm_autoencoder(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    training_config: dict,
    feature_config: dict,
    device: torch.device,
    scaler_dict: dict,
    feature_columns: list[str],
) -> None:
    """Train LSTM Autoencoder and log to MLflow."""
    config = training_config["lstm_autoencoder"]
    train_windows, train_labels = splits["train"]
    val_windows, val_labels = splits["val"]
    test_windows, test_labels = splits["test"]

    input_dim = train_windows.shape[2]
    seq_len = train_windows.shape[1]

    with mlflow.start_run(run_name="lstm_autoencoder_v1", tags={"model_type": "lstm_ae"}):
        # Log params — read from model_config.yaml, never hardcode (Rule #1)
        model_cfg = load_feature_config()  # loads model_config.yaml
        lstm_cfg = model_cfg.get("lstm_autoencoder", {})
        mlflow.log_params({
            "model_type": "lstm_autoencoder",
            "input_dim": input_dim,
            "seq_len": seq_len,
            "window_size": feature_config["window_size"],
            "hidden_dim": lstm_cfg.get("hidden_dim", 64),
            "latent_dim": lstm_cfg.get("latent_dim", 32),
            "num_layers": lstm_cfg.get("num_layers", 2),
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "train_samples": len(train_windows),
            "val_samples": len(val_windows),
            "test_samples": len(test_windows),
        })

        # Build model
        model = build_lstm_autoencoder(input_dim=input_dim, seq_len=seq_len)
        model = model.to(device)

        # Prepare data loaders (train on normal data only for autoencoder)
        normal_mask = train_labels == 0
        train_normal = torch.FloatTensor(train_windows[normal_mask]).to(device)
        train_dataset = TensorDataset(train_normal, train_normal)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        val_tensor = torch.FloatTensor(val_windows).to(device)

        # Optimizer & scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config["scheduler"]["patience"],
            factor=config["scheduler"]["factor"],
            min_lr=config["scheduler"]["min_lr"],
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        start_time = time.time()

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0

            for batch_x, batch_target in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_normal)

            # Validation loss (on all validation data, including anomalies)
            model.eval()
            with torch.no_grad():
                val_output = model(val_tensor)
                val_loss = criterion(val_output, val_tensor).item()

            scheduler.step(val_loss)

            # Log epoch metrics
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )

            if epoch % 10 == 0 or epoch == config["epochs"] - 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                    epoch + 1,
                    config["epochs"],
                    train_loss,
                    val_loss,
                )

            # Early stopping
            if val_loss < best_val_loss - config["early_stopping"]["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping"]["patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        training_time = time.time() - start_time

        # Restore best model
        model.load_state_dict(best_state)
        model.eval()

        # Threshold search on validation set
        val_scores = model.get_reconstruction_error(val_tensor).cpu().numpy()
        threshold, val_metrics = find_optimal_threshold(
            val_scores,
            val_labels,
            config["threshold"]["search_min"],
            config["threshold"]["search_max"],
            config["threshold"]["search_steps"],
        )

        # Evaluate on test set
        test_tensor = torch.FloatTensor(test_windows).to(device)
        test_scores, test_preds = model.predict_anomaly(test_tensor, threshold)

        test_metrics = {
            "test_f1": f1_score(test_labels, test_preds, zero_division=0),
            "test_precision": precision_score(test_labels, test_preds, zero_division=0),
            "test_recall": recall_score(test_labels, test_preds, zero_division=0),
        }
        if len(np.unique(test_labels)) > 1:
            test_metrics["test_auc_roc"] = roc_auc_score(test_labels, test_scores)

        # Log all metrics
        model_size = log_model_size(model)
        mlflow.log_metrics({
            "threshold": threshold,
            "training_time_sec": training_time,
            "model_size_mb": model_size,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **test_metrics,
        })

        # Save artifacts
        model_path = PROJECT_ROOT / "data" / "artifacts" / "lstm_autoencoder.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "input_dim": input_dim,
            "seq_len": seq_len,
            "scaler": scaler_dict,
            "feature_columns": feature_columns,
        }, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(
            "LSTM-AE training complete: test_f1=%.4f, threshold=%.4f, time=%.1fs",
            test_metrics["test_f1"],
            threshold,
            training_time,
        )


# ──────────────────────────────────────────────
# Train: Isolation Forest
# ──────────────────────────────────────────────


def train_isolation_forest(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    training_config: dict,
    feature_config: dict,
    scaler_dict: dict,
    feature_columns: list[str],
) -> None:
    """Train Isolation Forest with contamination grid search, log to MLflow."""
    contamination_grid = training_config["isolation_forest"]["contamination_grid"]
    train_windows, train_labels = splits["train"]
    val_windows, val_labels = splits["val"]
    test_windows, test_labels = splits["test"]

    best_f1 = -1.0
    best_contamination = contamination_grid[0]
    best_detector = None

    with mlflow.start_run(
        run_name="isolation_forest_v1", tags={"model_type": "isolation_forest"}
    ):
        for contamination in contamination_grid:
            start_time = time.time()

            detector = build_isolation_forest(contamination=contamination)
            detector.fit(train_windows)

            # Evaluate on validation set
            val_scores, val_preds = detector.predict_anomaly(val_windows)
            f1 = f1_score(val_labels, val_preds, zero_division=0)

            training_time = time.time() - start_time

            logger.info(
                "IsolationForest contamination=%.3f → val_f1=%.4f (%.1fs)",
                contamination,
                f1,
                training_time,
            )

            mlflow.log_metrics(
                {f"val_f1_c{contamination:.2f}": f1},
            )

            if f1 > best_f1:
                best_f1 = f1
                best_contamination = contamination
                best_detector = detector

        # Log best params
        mlflow.log_params({
            "model_type": "isolation_forest",
            "best_contamination": best_contamination,
            "n_estimators": best_detector.model.n_estimators,
            "window_size": feature_config["window_size"],
            "train_samples": len(train_windows),
            "val_samples": len(val_windows),
            "test_samples": len(test_windows),
        })

        # Evaluate best model on test set
        test_scores, test_preds = best_detector.predict_anomaly(test_windows)
        test_metrics = {
            "test_f1": f1_score(test_labels, test_preds, zero_division=0),
            "test_precision": precision_score(test_labels, test_preds, zero_division=0),
            "test_recall": recall_score(test_labels, test_preds, zero_division=0),
        }
        if len(np.unique(test_labels)) > 1:
            test_metrics["test_auc_roc"] = roc_auc_score(test_labels, test_scores)

        # Model size estimate (serialize to measure actual model size)
        import io

        import joblib as _joblib
        buf = io.BytesIO()
        _joblib.dump(best_detector, buf)
        model_size_mb = buf.tell() / (1024 * 1024)

        mlflow.log_metrics({
            "best_contamination": best_contamination,
            "model_size_mb": model_size_mb,
            **test_metrics,
        })

        # Save artifacts
        import joblib

        artifact_path = PROJECT_ROOT / "data" / "artifacts" / "isolation_forest.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": best_detector,
            "contamination": best_contamination,
            "scaler": scaler_dict,
            "feature_columns": feature_columns,
        }, artifact_path)
        mlflow.log_artifact(str(artifact_path))

        logger.info(
            "IsolationForest training complete: best_contamination=%.3f, test_f1=%.4f",
            best_contamination,
            test_metrics["test_f1"],
        )


# ──────────────────────────────────────────────
# Train: PatchTST
# ──────────────────────────────────────────────


def train_patchtst(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    training_config: dict,
    feature_config: dict,
    device: torch.device,
    scaler_dict: dict,
    feature_columns: list[str],
) -> None:
    """Train PatchTST forecasting model and log to MLflow."""
    config = training_config["patchtst"]
    train_windows, train_labels = splits["train"]
    val_windows, val_labels = splits["val"]
    test_windows, test_labels = splits["test"]

    # Split windows into input/target pairs for forecasting
    train_inputs, train_targets = create_forecast_windows(train_windows)
    val_inputs, val_targets = create_forecast_windows(val_windows)
    test_inputs, test_targets = create_forecast_windows(test_windows)

    input_dim = train_inputs.shape[2]
    seq_len = train_inputs.shape[1]

    with mlflow.start_run(run_name="patchtst_v1", tags={"model_type": "patchtst"}):
        # Log params
        mlflow.log_params({
            "model_type": "patchtst",
            "input_dim": input_dim,
            "seq_len": seq_len,
            "forecast_horizon": train_targets.shape[1],
            "window_size": feature_config["window_size"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "train_samples": len(train_inputs),
            "val_samples": len(val_inputs),
            "test_samples": len(test_inputs),
        })

        # Build model
        model = build_patchtst(input_dim=input_dim, seq_len=seq_len)
        model = model.to(device)

        # Prepare data loaders
        train_x = torch.FloatTensor(train_inputs).to(device)
        train_y = torch.FloatTensor(train_targets).to(device)
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        val_x = torch.FloatTensor(val_inputs).to(device)
        val_y = torch.FloatTensor(val_targets).to(device)

        # Optimizer & scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["scheduler"]["T_max"],
            eta_min=config["scheduler"]["eta_min"],
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        start_time = time.time()

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                forecast = model(batch_x)
                loss = criterion(forecast, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_inputs)
            scheduler.step()

            # Validation loss (batched to avoid MPS memory exhaustion)
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                eval_batch_size = 256
                for vi in range(0, len(val_x), eval_batch_size):
                    vx_b = val_x[vi : vi + eval_batch_size]
                    vy_b = val_y[vi : vi + eval_batch_size]
                    vf_b = model(vx_b)
                    val_loss += criterion(vf_b, vy_b).item() * len(vx_b)
                val_loss /= len(val_x)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )

            if epoch % 10 == 0 or epoch == config["epochs"] - 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                    epoch + 1,
                    config["epochs"],
                    train_loss,
                    val_loss,
                )

            # Early stopping
            if val_loss < best_val_loss - config["early_stopping"]["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping"]["patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        training_time = time.time() - start_time

        # Restore best model
        model.load_state_dict(best_state)
        model.eval()

        # Threshold search on validation set
        val_errors = model.get_forecast_error(val_x, val_y).cpu().numpy()
        threshold, val_metrics = find_optimal_threshold(
            val_errors,
            val_labels,
            config["threshold"]["search_min"],
            config["threshold"]["search_max"],
            config["threshold"]["search_steps"],
        )

        # Evaluate on test set
        test_x = torch.FloatTensor(test_inputs).to(device)
        test_y = torch.FloatTensor(test_targets).to(device)
        test_scores, test_preds = model.predict_anomaly(test_x, test_y, threshold)

        test_metrics = {
            "test_f1": f1_score(test_labels, test_preds, zero_division=0),
            "test_precision": precision_score(test_labels, test_preds, zero_division=0),
            "test_recall": recall_score(test_labels, test_preds, zero_division=0),
        }
        if len(np.unique(test_labels)) > 1:
            test_metrics["test_auc_roc"] = roc_auc_score(test_labels, test_scores)

        # Log metrics
        model_size = log_model_size(model)
        mlflow.log_metrics({
            "threshold": threshold,
            "training_time_sec": training_time,
            "model_size_mb": model_size,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **test_metrics,
        })

        # Save artifacts
        model_path = PROJECT_ROOT / "data" / "artifacts" / "patchtst.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "input_dim": input_dim,
            "seq_len": seq_len,
            "forecast_horizon": train_targets.shape[1],
            "scaler": scaler_dict,
            "feature_columns": feature_columns,
        }, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(
            "PatchTST training complete: test_f1=%.4f, threshold=%.4f, time=%.1fs",
            test_metrics["test_f1"],
            threshold,
            training_time,
        )


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────


TRAINERS = {
    "lstm_autoencoder": train_lstm_autoencoder,
    "isolation_forest": train_isolation_forest,
    "patchtst": train_patchtst,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Model Training")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training_config.yaml",
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm_autoencoder", "isolation_forest", "patchtst", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CMAPSS data before training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    # Load configs
    training_config = load_training_config(args.config)
    feature_config = load_feature_config()
    device = resolve_device(training_config.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Setup MLflow
    setup_mlflow(training_config)

    # Ingest data
    logger.info("Ingesting CMAPSS data...")
    df = ingest_train_data(download=args.download)

    # Feature engineering
    logger.info("Building features...")
    windows, labels, metadata, scaler, feature_columns = build_features(df)
    scaler_dict = scaler.to_dict()

    logger.info(
        "Data ready: %d windows, shape=%s, anomaly_rate=%.1f%%",
        len(windows),
        windows.shape,
        100 * labels.mean(),
    )

    # Split data
    splits = split_data(
        windows,
        labels,
        test_size=training_config["data"]["test_size"],
        validation_size=training_config["data"]["validation_size"],
    )

    # Determine which models to train
    models_to_train = list(TRAINERS.keys()) if args.model == "all" else [args.model]

    # Train models
    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info("Training: %s", model_name)
        logger.info("=" * 60)

        trainer = TRAINERS[model_name]

        if model_name == "isolation_forest":
            trainer(
                splits=splits,
                training_config=training_config,
                feature_config=feature_config,
                scaler_dict=scaler_dict,
                feature_columns=feature_columns,
            )
        else:
            trainer(
                splits=splits,
                training_config=training_config,
                feature_config=feature_config,
                device=device,
                scaler_dict=scaler_dict,
                feature_columns=feature_columns,
            )

    logger.info("=" * 60)
    logger.info("All training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
