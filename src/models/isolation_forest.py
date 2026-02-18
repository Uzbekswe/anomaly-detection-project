"""Isolation Forest baseline for time-series anomaly detection.

Input preparation:
    Sliding windows (N, seq_len, num_features) are converted to 2D by
    computing summary statistics per window: mean + std per feature.
    Result: (N, num_features * 2)

    This preserves distributional information from the time window while
    giving sklearn a flat feature vector it can work with.

Anomaly logic:
    sklearn IsolationForest.decision_function() returns anomaly scores.
    Lower (more negative) = more anomalous.
    We negate and normalize so higher score = more anomalous (consistent
    with the LSTM-AE interface).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def load_iforest_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load Isolation Forest config from model_config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["isolation_forest"]


# ──────────────────────────────────────────────
# Window flattening
# ──────────────────────────────────────────────


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Convert 3D sliding windows to 2D feature vectors via summary stats.

    For each window, computes mean and std per feature across the time axis.

    Args:
        windows: (N, seq_len, num_features)

    Returns:
        features: (N, num_features * 2) — [means..., stds...]
    """
    # Mean across time axis
    means = np.mean(windows, axis=1)  # (N, num_features)
    # Std across time axis
    stds = np.std(windows, axis=1)  # (N, num_features)

    features = np.concatenate([means, stds], axis=1)  # (N, num_features * 2)
    logger.info(
        "Flattened %d windows from shape %s to (%d, %d)",
        len(windows),
        windows.shape,
        features.shape[0],
        features.shape[1],
    )
    return features


# ──────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────


class IsolationForestDetector:
    """Wrapper around sklearn IsolationForest with a consistent interface.

    Provides the same predict_anomaly() signature as the LSTM Autoencoder
    for unified evaluation.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_samples: str | int = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(self, windows: np.ndarray) -> IsolationForestDetector:
        """Fit the Isolation Forest on training windows.

        Args:
            windows: (N, seq_len, num_features) — 3D sliding windows.

        Returns:
            self
        """
        features = flatten_windows(windows)
        self.model.fit(features)
        self._is_fitted = True
        logger.info(
            "Fitted IsolationForest: n_estimators=%d, contamination=%.3f, samples=%d",
            self.model.n_estimators,
            self.contamination,
            len(features),
        )
        return self

    def get_anomaly_scores(self, windows: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for windows.

        Scores are negated and shifted so that:
            - Higher score = more anomalous (consistent with LSTM-AE)
            - Scores are roughly in [0, 1] range

        Args:
            windows: (N, seq_len, num_features)

        Returns:
            scores: (N,) — anomaly scores (higher = more anomalous)
        """
        features = flatten_windows(windows)
        # sklearn: decision_function returns negative for anomalies
        raw_scores = self.model.decision_function(features)

        # Negate so higher = more anomalous
        scores = -raw_scores

        # Normalize to [0, 1] range
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min
        scores = (scores - score_min) / score_range if score_range > 0 else np.zeros_like(scores)

        return scores.astype(np.float64)

    def predict_anomaly(
        self,
        windows: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomalies.

        If threshold is provided, uses it for binary decision.
        Otherwise falls back to sklearn's built-in contamination-based decision.

        Args:
            windows: (N, seq_len, num_features)
            threshold: Optional score threshold for binary decision.

        Returns:
            Tuple of:
                anomaly_scores: (N,) — normalized anomaly scores
                is_anomaly:     (N,) — binary predictions (0 or 1)
        """
        anomaly_scores = self.get_anomaly_scores(windows)

        if threshold is not None:
            is_anomaly = (anomaly_scores > threshold).astype(np.int64)
        else:
            # Use sklearn's built-in prediction (based on contamination)
            features = flatten_windows(windows)
            preds = self.model.predict(features)
            # sklearn: -1 = anomaly, 1 = normal → convert to 0/1
            is_anomaly = (preds == -1).astype(np.int64)

        return anomaly_scores, is_anomaly


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def build_isolation_forest(
    contamination: float | None = None,
    config_path: Path = MODEL_CONFIG_PATH,
) -> IsolationForestDetector:
    """Build IsolationForestDetector from config.

    Args:
        contamination: Override contamination value (for grid search).
            Uses config value if None.
        config_path: Path to model_config.yaml.

    Returns:
        IsolationForestDetector instance.
    """
    config = load_iforest_config(config_path)

    if contamination is not None:
        config["contamination"] = contamination

    detector = IsolationForestDetector(
        n_estimators=config["n_estimators"],
        contamination=config["contamination"],
        max_samples=config["max_samples"],
        max_features=config["max_features"],
        random_state=config["random_state"],
    )

    logger.info(
        "Built IsolationForestDetector: n_estimators=%d, contamination=%.3f",
        config["n_estimators"],
        config["contamination"],
    )
    return detector
