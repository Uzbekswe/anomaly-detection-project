"""Feature engineering for CMAPSS sensor data.

CRITICAL: This module is the SINGLE SOURCE OF TRUTH for feature transforms.
It is imported by BOTH training (src/models/train.py) and serving
(src/serving/predictor.py). Never duplicate this logic elsewhere.

Pipeline:
    raw sensors → drop low-variance → rolling statistics → normalize → sliding windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────


def load_feature_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load data + features sections from model_config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {
        "sensor_columns": config["data"]["sensor_columns"],
        "drop_sensors": config["data"]["drop_sensors"],
        "window_size": config["data"]["window_size"],
        "rolling_window_sizes": config["features"]["rolling_window_sizes"],
        "rolling_statistics": config["features"]["rolling_statistics"],
        "normalization": config["features"]["normalization"],
    }


# ──────────────────────────────────────────────
# Scaler (serializable for serving)
# ──────────────────────────────────────────────


@dataclass
class MinMaxScaler:
    """Min-max scaler that can be serialized for identical serving transforms.

    Stores per-column min/max values fitted on training data.
    """

    min_vals: dict[str, float] = field(default_factory=dict)
    max_vals: dict[str, float] = field(default_factory=dict)
    feature_range: tuple[float, float] = (0.0, 1.0)

    def fit(self, df: pd.DataFrame) -> MinMaxScaler:
        """Fit scaler on training data."""
        for col in df.columns:
            self.min_vals[col] = float(df[col].min())
            self.max_vals[col] = float(df[col].max())
        logger.info("MinMaxScaler fitted on %d features", len(df.columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted min/max values."""
        df = df.copy()
        lo, hi = self.feature_range
        for col in df.columns:
            col_min = self.min_vals[col]
            col_max = self.max_vals[col]
            col_range = col_max - col_min
            if col_range == 0:
                df[col] = 0.0
            else:
                df[col] = lo + (df[col] - col_min) / col_range * (hi - lo)
        return df

    def to_dict(self) -> dict:
        """Serialize for MLflow artifact logging."""
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "feature_range": list(self.feature_range),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MinMaxScaler:
        """Deserialize from MLflow artifact."""
        return cls(
            min_vals=data["min_vals"],
            max_vals=data["max_vals"],
            feature_range=tuple(data["feature_range"]),
        )


@dataclass
class StandardScaler:
    """Standard (z-score) scaler — stores per-column mean/std."""

    mean_vals: dict[str, float] = field(default_factory=dict)
    std_vals: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> StandardScaler:
        """Fit scaler on training data."""
        for col in df.columns:
            self.mean_vals[col] = float(df[col].mean())
            self.std_vals[col] = float(df[col].std())
        logger.info("StandardScaler fitted on %d features", len(df.columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted mean/std values."""
        df = df.copy()
        for col in df.columns:
            std = self.std_vals[col]
            if std == 0:
                df[col] = 0.0
            else:
                df[col] = (df[col] - self.mean_vals[col]) / std
        return df

    def to_dict(self) -> dict:
        """Serialize for MLflow artifact logging."""
        return {"mean_vals": self.mean_vals, "std_vals": self.std_vals}

    @classmethod
    def from_dict(cls, data: dict) -> StandardScaler:
        """Deserialize from MLflow artifact."""
        return cls(mean_vals=data["mean_vals"], std_vals=data["std_vals"])


def create_scaler(method: str) -> MinMaxScaler | StandardScaler:
    """Factory to create scaler based on config string."""
    if method == "min_max":
        return MinMaxScaler()
    elif method == "standard":
        return StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'min_max' or 'standard'.")


# ──────────────────────────────────────────────
# Step 1: Drop low-variance sensors
# ──────────────────────────────────────────────


def drop_low_variance_sensors(
    df: pd.DataFrame,
    drop_sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Remove sensors with near-zero variance (identified in EDA).

    Args:
        df: DataFrame with all sensor columns.
        drop_sensors: List of sensor column names to drop.
            Loaded from config if None.

    Returns:
        DataFrame with low-variance sensor columns removed.
    """
    if drop_sensors is None:
        config = load_feature_config()
        drop_sensors = config["drop_sensors"]

    cols_to_drop = [c for c in drop_sensors if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped %d low-variance sensors: %s", len(cols_to_drop), cols_to_drop)
    return df


# ──────────────────────────────────────────────
# Step 2: Rolling statistics
# ──────────────────────────────────────────────


def add_rolling_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
    rolling_window_sizes: list[int] | None = None,
    rolling_statistics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling statistics per unit for each sensor.

    Creates new columns like: sensor_2_rolling_mean_5, sensor_2_rolling_std_10, etc.

    Args:
        df: DataFrame with unit_id and sensor columns.
        sensor_columns: Sensor columns to compute rolling features for.
        rolling_window_sizes: Window sizes for rolling computation.
        rolling_statistics: Statistics to compute (mean, std).

    Returns:
        DataFrame with additional rolling feature columns.
    """
    if rolling_window_sizes is None or rolling_statistics is None:
        config = load_feature_config()
        rolling_window_sizes = rolling_window_sizes or config["rolling_window_sizes"]
        rolling_statistics = rolling_statistics or config["rolling_statistics"]

    df = df.copy()
    new_cols_count = 0

    # Only compute for sensor columns that exist in df
    available_sensors = [c for c in sensor_columns if c in df.columns]

    for window in rolling_window_sizes:
        for stat in rolling_statistics:
            for col in available_sensors:
                new_col = f"{col}_rolling_{stat}_{window}"

                grouped = df.groupby("unit_id")[col]
                if stat == "mean":
                    df[new_col] = grouped.transform(
                        lambda x, w=window: x.rolling(window=w, min_periods=1).mean()
                    )
                elif stat == "std":
                    df[new_col] = grouped.transform(
                        lambda x, w=window: x.rolling(window=w, min_periods=1).std()
                    )
                else:
                    raise ValueError(f"Unsupported rolling statistic: {stat}")

                new_cols_count += 1

    # Fill any NaN from rolling std (single-value windows)
    df = df.fillna(0.0)

    logger.info(
        "Added %d rolling features (%d sensors x %d windows x %d stats)",
        new_cols_count,
        len(available_sensors),
        len(rolling_window_sizes),
        len(rolling_statistics),
    )
    return df


# ──────────────────────────────────────────────
# Step 3: Normalize
# ──────────────────────────────────────────────


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns (sensors + rolling), excluding metadata."""
    exclude = {"unit_id", "time_cycles", "rul", "is_anomaly", "timestamp"}
    return [c for c in df.columns if c not in exclude]


def fit_scaler(
    df: pd.DataFrame,
    method: str | None = None,
) -> MinMaxScaler | StandardScaler:
    """Fit a scaler on the training data.

    Args:
        df: DataFrame with feature columns for fitting the scaler.
        method: Normalization method ('min_max' or 'standard').

    Returns:
        A fitted scaler instance.
    """
    if method is None:
        config = load_feature_config()
        method = config["normalization"]

    feature_cols = get_feature_columns(df)
    scaler = create_scaler(method)
    scaler.fit(df[feature_cols])
    logger.info("Fitted scaler on %d feature columns using '%s' method", len(feature_cols), method)
    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: MinMaxScaler | StandardScaler,
) -> pd.DataFrame:
    """Normalize feature columns using a pre-fitted scaler.

    Args:
        df: DataFrame with feature columns.
        scaler: A pre-fitted scaler instance.

    Returns:
        The DataFrame with normalized feature columns.
    """
    feature_cols = get_feature_columns(df)
    df_transformed = df.copy()
    df_transformed[feature_cols] = scaler.transform(df[feature_cols])
    logger.info("Transformed %d feature columns with pre-fitted scaler", len(feature_cols))
    return df_transformed



# ──────────────────────────────────────────────
# Step 4: Sliding windows
# ──────────────────────────────────────────────


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows per unit for model input.

    Args:
        df: Normalized DataFrame with feature and label columns.
        window_size: Number of timesteps per window.
        feature_columns: Columns to include in windows. Auto-detected if None.

    Returns:
        Tuple of:
            windows: np.ndarray of shape (num_windows, window_size, num_features)
            labels:  np.ndarray of shape (num_windows,) — anomaly label for last
                     timestep in each window (0 or 1)
            metadata: np.ndarray of shape (num_windows, 2) — [unit_id, time_cycles]
                      for the last timestep in each window
    """
    if window_size is None:
        config = load_feature_config()
        window_size = config["window_size"]

    if feature_columns is None:
        feature_columns = get_feature_columns(df)

    windows_list: list[np.ndarray] = []
    labels_list: list[int] = []
    metadata_list: list[list[int]] = []

    has_labels = "is_anomaly" in df.columns

    for _, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("time_cycles")
        features = unit_df[feature_columns].values
        n_rows = len(features)

        if n_rows < window_size:
            logger.debug(
                "Unit %s has %d rows < window_size %d — skipping",
                unit_df["unit_id"].iloc[0],
                n_rows,
                window_size,
            )
            continue

        for i in range(n_rows - window_size + 1):
            window = features[i : i + window_size]
            windows_list.append(window)

            if has_labels:
                # Label = anomaly status of the last timestep in the window
                labels_list.append(int(unit_df["is_anomaly"].iloc[i + window_size - 1]))

            last_row = unit_df.iloc[i + window_size - 1]
            metadata_list.append([int(last_row["unit_id"]), int(last_row["time_cycles"])])

    windows = np.array(windows_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64) if has_labels else np.array([], dtype=np.int64)
    metadata = np.array(metadata_list, dtype=np.int64)

    logger.info(
        "Created %d sliding windows: shape=%s, positive_labels=%d/%d (%.1f%%)",
        len(windows),
        windows.shape,
        labels.sum() if len(labels) > 0 else 0,
        len(labels) if len(labels) > 0 else 0,
        100 * labels.mean() if len(labels) > 0 else 0,
    )
    return windows, labels, metadata


# ──────────────────────────────────────────────
# Single-window transform (for serving)
# ──────────────────────────────────────────────


def transform_single_window(
    raw_window: np.ndarray,
    scaler: MinMaxScaler | StandardScaler,
    feature_columns: list[str],
) -> np.ndarray:
    """Transform a single raw sensor window for inference.

    This applies the SAME normalization as training — ensuring
    train/serve consistency.

    Args:
        raw_window: np.ndarray of shape (window_size, num_raw_sensors)
        scaler: Pre-fitted scaler loaded from MLflow.
        feature_columns: Column names matching scaler's fitted columns.

    Returns:
        Normalized window as np.ndarray of shape (window_size, num_features).
    """
    df = pd.DataFrame(raw_window, columns=feature_columns)
    df = scaler.transform(df)
    return df.values.astype(np.float32)


# ──────────────────────────────────────────────
# Full pipeline (for training)
# ──────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    scaler: MinMaxScaler | StandardScaler,
    config_path: Path = MODEL_CONFIG_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Full feature engineering pipeline for a given dataset (train, val, or test).

    CRITICAL: A pre-fitted scaler must be provided. Fitting happens
    only once on the training data.

    Pipeline: drop → rolling → normalize → window.

    Args:
        df: Raw DataFrame from ingest.py.
        scaler: A pre-fitted scaler from the training set.
        config_path: Path to model_config.yaml.

    Returns:
        Tuple of:
            windows:         (num_windows, window_size, num_features)
            labels:          (num_windows,)
            metadata:        (num_windows, 2) — [unit_id, time_cycles]
            feature_columns: List of feature column names
    """
    config = load_feature_config(config_path)

    # Step 1: Drop low-variance sensors
    df = drop_low_variance_sensors(df, config["drop_sensors"])

    # Determine which sensor columns remain
    remaining_sensors = [c for c in config["sensor_columns"] if c not in config["drop_sensors"]]

    # Step 2: Rolling statistics
    df = add_rolling_features(
        df,
        sensor_columns=remaining_sensors,
        rolling_window_sizes=config["rolling_window_sizes"],
        rolling_statistics=config["rolling_statistics"],
    )

    # Step 3: Normalize using the pre-fitted scaler
    df = transform_features(df, scaler)

    # Capture feature columns after all transforms
    feature_columns = get_feature_columns(df)

    # Step 4: Sliding windows
    windows, labels, metadata = create_sliding_windows(
        df,
        window_size=config["window_size"],
        feature_columns=feature_columns,
    )

    logger.info(
        "Feature pipeline complete: %d windows, %d features per timestep",
        len(windows),
        windows.shape[2] if len(windows) > 0 else 0,
    )
    return windows, labels, metadata, feature_columns

