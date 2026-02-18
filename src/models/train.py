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


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


# ──────────────────────────────────────────────
# Data splitting (Unit-aware)
# ──────────────────────────────────────────────


def split_data_unit_aware(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Split DataFrame into train/val/test sets, ensuring units are not split.

    Args:
        df: The full DataFrame to split.
        test_size: Proportion of units to allocate to the test set.
        validation_size: Proportion of the remaining units for validation.
        random_state: Seed for the random split.

    Returns:
        A dictionary with "train", "val", and "test" DataFrames.
    """
    unit_ids = df["unit_id"].unique()
    n_units = len(unit_ids)
    np.random.RandomState(random_state).shuffle(unit_ids)

    test_split_idx = int(n_units * (1 - test_size))
    val_split_idx = int(test_split_idx * (1 - validation_size / (1 - test_size)))

    train_units = unit_ids[:val_split_idx]
    val_units = unit_ids[val_split_idx:test_split_idx]
    test_units = unit_ids[test_split_idx:]

    splits = {
        "train": df[df["unit_id"].isin(train_units)].copy(),
        "val": df[df["unit_id"].isin(val_units)].copy(),
        "test": df[df["unit_id"].isin(test_units)].copy(),
    }

    logger.info(
        "Data split (unit-aware): train=%d units, val=%d units, test=%d units",
        len(train_units), len(val_units), len(test_units)
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
    data_splits: dict[str, dict],
    training_config: dict,
    model_config: dict,
    device: torch.device,
    scaler_dict: dict,
    feature_columns: list[str],
) -> None:
    """Train LSTM Autoencoder and log to MLflow."""
    config = training_config["lstm_autoencoder"]
    lstm_cfg = model_config["lstm_autoencoder"]
    
    train_windows = data_splits["train"]["windows"]
    train_labels = data_splits["train"]["labels"]
    val_windows = data_splits["val"]["windows"]
    val_labels = data_splits["val"]["labels"]
    test_windows = data_splits["test"]["windows"]
    test_labels = data_splits["test"]["labels"]

    # Get input_dim and seq_len from the data and config
    input_dim = model_config["model_features"]["num_features"]
    seq_len = model_config["data"]["window_size"]

    with mlflow.start_run(run_name="lstm_autoencoder_v1", tags={"model_type": "lstm_ae"}):
        # Log params
        mlflow.log_params({
            "model_type": "lstm_autoencoder",
            "input_dim": input_dim,
            "seq_len": seq_len,
            "hidden_dim": lstm_cfg["hidden_dim"],
            "latent_dim": lstm_cfg["latent_dim"],
            "num_layers": lstm_cfg["num_layers"],
            "dropout": lstm_cfg["dropout"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
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
    data_splits: dict[str, dict],
    training_config: dict,
    model_config: dict,
    scaler_dict: dict,
    feature_columns: list[str],
) -> None:
    """Train Isolation Forest with contamination grid search, log to MLflow."""
    contamination_grid = training_config["isolation_forest"]["contamination_grid"]
    
    train_windows = data_splits["train"]["windows"]
    val_windows = data_splits["val"]["windows"]
    val_labels = data_splits["val"]["labels"]
    test_windows = data_splits["test"]["windows"]
    test_labels = data_splits["test"]["labels"]

    # Reshape windows for sklearn model
    train_windows_flat = train_windows.reshape(train_windows.shape[0], -1)
    val_windows_flat = val_windows.reshape(val_windows.shape[0], -1)
    test_windows_flat = test_windows.reshape(test_windows.shape[0], -1)

    best_f1 = -1.0
    best_contamination = contamination_grid[0]
    best_detector = None

    with mlflow.start_run(
        run_name="isolation_forest_v1", tags={"model_type": "isolation_forest"}
    ):
        for contamination in contamination_grid:
            start_time = time.time()

            detector = build_isolation_forest(contamination=contamination)
            detector.fit(train_windows_flat)

            # Evaluate on validation set
            val_scores, val_preds = detector.predict_anomaly(val_windows_flat)
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
        test_scores, test_preds = best_detector.predict_anomaly(test_windows_flat)
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
    data_splits: dict[str, dict],
    training_config: dict,
    model_config: dict,
    device: torch.device,
    scaler_dict: dict,
    feature_columns: list[str],
) -> None:
    """Train PatchTST forecasting model and log to MLflow."""
    config = training_config["patchtst"]
    patchtst_cfg = model_config["patchtst"]
    
    train_windows = data_splits["train"]["windows"]
    val_windows = data_splits["val"]["windows"]
    val_labels = data_splits["val"]["labels"]
    test_windows = data_splits["test"]["windows"]
    test_labels = data_splits["test"]["labels"]

    # Split windows into input/target pairs for forecasting
    train_inputs, train_targets = create_forecast_windows(train_windows, horizon=patchtst_cfg["forecast_horizon"])
    val_inputs, val_targets = create_forecast_windows(val_windows, horizon=patchtst_cfg["forecast_horizon"])
    test_inputs, test_targets = create_forecast_windows(test_windows, horizon=patchtst_cfg["forecast_horizon"])

    # Get input_dim and seq_len from the data and config
    input_dim = model_config["model_features"]["num_features"]
    seq_len = train_inputs.shape[1]

    with mlflow.start_run(run_name="patchtst_v1", tags={"model_type": "patchtst"}):
        # Log params
        mlflow.log_params({
            "model_type": "patchtst",
            "input_dim": input_dim,
            "seq_len": seq_len,
            "forecast_horizon": patchtst_cfg["forecast_horizon"],
            "patch_length": patchtst_cfg["patch_length"],
            "stride": patchtst_cfg["stride"],
            "d_model": patchtst_cfg["d_model"],
            "n_heads": patchtst_cfg["n_heads"],
            "num_encoder_layers": patchtst_cfg["num_encoder_layers"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
        })

        # Build model
        model = build_patchtst(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_horizon=patchtst_cfg["forecast_horizon"],
            patch_len=patchtst_cfg["patch_length"],
            stride=patchtst_cfg["stride"],
            d_model=patchtst_cfg["d_model"],
            n_heads=patchtst_cfg["n_heads"],
            num_encoder_layers=patchtst_cfg["num_encoder_layers"],
            d_ff=patchtst_cfg["d_ff"],
            dropout=patchtst_cfg["dropout"],
        )
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
    model_config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    device = resolve_device(training_config.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Set seed for reproducibility
    set_seed(training_config["data"]["random_state"])

    # Setup MLflow
    setup_mlflow(training_config)

    # Ingest data
    logger.info("Ingesting CMAPSS data...")
    df = ingest_train_data(download=args.download)

    # 1. Split data at the unit level BEFORE feature engineering
    logger.info("Splitting data (unit-aware)...")
    df_splits = split_data_unit_aware(
        df,
        test_size=training_config["data"]["test_size"],
        validation_size=training_config["data"]["validation_size"],
        random_state=training_config["data"]["random_state"],
    )

    # 2. Fit scaler ONCE on the training data
    logger.info("Fitting feature scaler on training data...")
    scaler = fit_scaler(df_splits["train"], method=feature_config["normalization"])
    scaler_dict = scaler.to_dict()

    # Log scaler as a separate artifact in a parent run
    with mlflow.start_run(run_name="feature_engineering", nested=True) as parent_run:
        mlflow.log_dict(scaler_dict, "scaler.json")
        mlflow.log_params({"normalization_method": feature_config["normalization"]})
        parent_run_id = parent_run.info.run_id
        logger.info("Logged scaler and feature engineering params to parent run: %s", parent_run_id)

    # 3. Build features for each split using the SAME fitted scaler
    data_splits = {}
    for split_name, split_df in df_splits.items():
        logger.info("Building features for '%s' split...", split_name)
        windows, labels, metadata, feature_columns = build_features(split_df, scaler)
        data_splits[split_name] = {
            "windows": windows,
            "labels": labels,
            "metadata": metadata,
        }
        logger.info(
            "'%s' split ready: %d windows, shape=%s, anomaly_rate=%.1f%%",
            split_name,
            len(windows),
            windows.shape,
            100 * labels.mean() if len(labels) > 0 else 0,
        )

    # Determine which models to train
    models_to_train = list(TRAINERS.keys()) if args.model == "all" else [args.model]

    # Train models
    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info("Training: %s", model_name)
        logger.info("=" * 60)

        trainer = TRAINERS[model_name]

        # Each trainer function will receive the full data_splits dictionary
        # and the feature_columns list for artifact logging.
        # We also pass the full model_config for dynamic parameter access.
        trainer_args = {
            "data_splits": data_splits,
            "training_config": training_config,
            "model_config": model_config,
            "scaler_dict": scaler_dict,
            "feature_columns": feature_columns,
        }
        if model_name != "isolation_forest":
            trainer_args["device"] = device

        trainer(**trainer_args)

    logger.info("=" * 60)
    logger.info("All training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
